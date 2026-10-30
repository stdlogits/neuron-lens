"""Activations of every MLP neuron and attention head over the activation corpora (pipeline stage 6).

One forward pass per batch of documents, eager attention in bf16, over every document of
``05_act_corpus.py``'s slices (web, Chinese, Python, chat), up to 256 tokens each. Collected:

- **neurons,** at each layer's ``down_proj`` input (the activations ``silu(gate)·up``), from
  position 1 (position 0 is the attention sink, where activations are massive): per slice, the
  sum and sum of squares, and exact counts of each pole's activation (``±act``) past every point
  of a quarter-octave grid from 2⁻¹⁰ to 2¹⁰ (rates and quantiles interpolate between them); per
  pole, the top ``TOP_DOCS`` documents by its extreme in the document, with the position;
- **heads,** at each layer's ``o_proj`` input, from position 16 (where the lenses start; a head
  with few keys attends sharply): the norm of each head's output ``W_O^h z``, counted on the same
  grid, and its top ``TOP_DOCS`` documents, each with the position it attends to most past the sink
  and the output vector. Also each head's mean attention from those queries to position 0
  (``sink``), the position before (``prev``) and itself (``self``);
- **QK statistics** for ``08_reads.py`` (findings §11): per head, the mean query after ``q_norm``;
  the sums giving the true derivative of the input norm's scaling and ``q_norm`` or ``k_norm``
  (Gemma 4: and the weightless ``v_norm``, for the OV reads) averaged over positions; attention by offset; and the attention each key token receives (from
  queries at 16+, renormalized without the sink, times the number of keys: 1 is uniform).

Then two small passes: every kept token alone, the first position, where a causal model's
activations depend on the token alone (``first_max``, ``first_tokens``, ``first_chat`` on the chat
template's opening token); and ``REP_SEQS`` sequences of ``REP_LEN`` random kept tokens said twice,
for the heads' ``induction`` (attention from the second copy to the token after the first
occurrence) and ``duplicate`` (to the occurrence itself) scores.

Gemma 4 12B (``NEURON_LENS_MODEL=google/gemma-4-12B``): GELU-tanh activations; per-layer head
sizes and key heads (the QK statistics padded to the largest, 8 key heads of 512); each head's
output as it reaches the residual (the post-attention norm's weight and its scale per position);
every document starts with ``<bos>``, so the first-position and repeated-token passes put it
before their tokens (the token's position is 1); RoPE per layer type; the released R-lens.

The state is saved to ``data/acts/<model>.pt`` every ``--save-minutes`` and resumes from there.
Once the pass is complete, the script writes the acts tables, ``neurons/acts/L<layer>.parquet``
and ``heads/acts.parquet``, and ``docs.parquet`` (each document's slice and line in the slice's
corpus file); a rerun rebuilds them. The heads' ``out_R_z`` reads their firings' outputs through
the R-lens from the MLP input.

    python scripts/06_activations.py [--detach] [--docs N]
"""
import argparse
import functools
import os
import subprocess
import sys
import time
from collections import Counter

import pyarrow as pa
import torch
from transformers import AutoModelForCausalLM

from neuron_lens.catalog import (
    MODEL,
    N_CONTEXTS,
    TOP,
    acts_path,
    catalog_dir,
    context,
    fixed,
    head_keys,
    head_table,
    kept,
    neuron_keys,
    neuron_table,
    put,
    tokenizer,
    write,
)
from neuron_lens.corpus import ACT_SLICES, act_docs, rows_sha256
from neuron_lens import arch
from neuron_lens.paths import LOGS, ROOT
from neuron_lens.readout import rms_norm_columns
from neuron_lens.transport import SKIP_FIRST, decoder_stack
from neuron_lens.weights import tensors

SEQ = 256
NEURON_FROM, HEAD_FROM = 1, SKIP_FIRST
EDGES = torch.tensor([0.0] + [2.0 ** (k / 4) for k in range(-40, 41)])  # counts of |x| past each
NB = len(EDGES) + 1  # bins: 0 is x = 0, i is (EDGES[i-1], EDGES[i]], the last past 2¹⁰
TOP_DOCS = 24
FLUSH = 50  # batches between moving the device's running sums into the float64 totals
CHUNK = 2048  # positions per counting step
REP_SEQS, REP_LEN = 64, 64
FIRST_K = 3
R_FIT = arch.fits(MODEL)[0]
ACTS = acts_path()


def buffers(L: int, f: int, H: int, kvh: int, hd: int, V: int) -> dict:
    """Running sums: ``name: (shape, integer, per slice)``."""
    return {"n": ((), True, True), "sum": ((L, f), False, True), "sumsq": ((L, f), False, True),
            "count": ((L, 2, NB, f), True, True),
            "h_n": ((), True, True), "h_sum": ((L, H), False, True), "h_sumsq": ((L, H), False, True),
            "h_count": ((L, NB, H), True, True),
            "sink": ((L, H), False, False), "prev": ((L, H), False, False), "self": ((L, H), False, False),
            "n_q": ((), False, False), "n_k": ((), False, False), "q": ((L, H, hd), False, False),
            "q_r": ((L, H), False, False), "q_m": ((L, H, hd, hd), False, False), "k_r": ((L, kvh), False, False),
            "k_m": ((L, kvh, hd, hd), False, False), "offset": ((L, H, SEQ), False, False),
            "v_r": ((L, kvh), False, False), "v_m": ((L, kvh, hd, hd), False, False),  # Gemma 4's v_norm (else 0)
            "recv": ((L, H, V), False, False), "pairs": ((V,), False, False), "occ": ((V,), False, False)}


class Collector:
    def __init__(self, model, n_slices: int, device):
        cfg = model.config  # the decoder stack's (text) config
        L, f, H = cfg.num_hidden_layers, cfg.intermediate_size, cfg.num_attention_heads
        kvh, hd = arch.padded(MODEL)  # per-layer statistics sit in the leading block of these
        d, V = cfg.hidden_size, cfg.vocab_size
        self.geo = arch.layout(MODEL)
        self.L, self.f, self.H, self.hd, self.d, self.dev = L, f, H, hd, d, device
        self.spec = buffers(L, f, H, kvh, hd, V)
        self.total = {k: torch.zeros(((n_slices,) if per else ()) + shape, dtype=torch.int64 if integer else torch.float64)
                      for k, (shape, integer, per) in self.spec.items()}
        self.seg = {k: torch.zeros(shape, dtype=torch.int32 if integer else torch.float32, device=device)
                    for k, (shape, integer, per) in self.spec.items()}
        k = TOP_DOCS
        self.top = {"val": torch.full((L, 2, k, f), -torch.inf, device=device),
                    "doc": torch.zeros((L, 2, k, f), dtype=torch.int32, device=device),
                    "pos": torch.zeros((L, 2, k, f), dtype=torch.int16, device=device),
                    "h_val": torch.full((L, H, k), -torch.inf, device=device),
                    "h_doc": torch.zeros((L, H, k), dtype=torch.int32, device=device),
                    "h_pos": torch.zeros((L, H, k), dtype=torch.int16, device=device),
                    "h_src": torch.zeros((L, H, k), dtype=torch.int16, device=device),
                    "h_out": torch.zeros((L, H, k, d), dtype=torch.float16, device=device)}
        self.extra = {"first_val": torch.full((L, 2, FIRST_K, f), -torch.inf), "first_tok": torch.zeros((L, 2, FIRST_K, f), dtype=torch.int32),
                      "first_chat": torch.zeros(L, f), "induction": torch.zeros(L, H), "duplicate": torch.zeros(L, H)}
        self.layers = model.layers
        self.wo = [arch.head_write(layer).view(d, H, g.head_dim) for layer, g in zip(model.layers, self.geo)]
        self.gram = [torch.einsum("dhi,dhj->hij", w, w) for w in self.wo]  # |W_O^h z|² = zᵀ G_h z
        self.first_pos = 0  # the token's position in the first-position and repeated passes (1 after <bos>)
        self.edges = EDGES.to(device)
        self.ctx, self.stash = {}, {}
        self.mode = "main"

    def state(self) -> dict:
        return {"total": self.total, "top": {k: v.cpu() for k, v in self.top.items()}, "extra": self.extra}

    def load(self, state: dict) -> None:
        self.total = state["total"]
        self.top = {k: v.to(self.dev) for k, v in state["top"].items()}
        self.extra = state["extra"]

    def flush(self, slice_i: int) -> None:
        for k, (_, _, per) in self.spec.items():
            t = self.total[k][slice_i] if per else self.total[k]
            t += self.seg[k].cpu().to(t.dtype)
            self.seg[k].zero_()

    def count(self, cnt: torch.Tensor, x: torch.Tensor, poles: bool) -> None:
        """Add the grid counts of ``x`` ``[P, n]`` into ``cnt`` (``[2, NB, n]`` with ``poles``, else ``[NB, n]``)."""
        n = x.shape[1]
        cols = torch.arange(n, device=x.device)
        flat = cnt.view(-1)
        for s in range(0, len(x), CHUNK):
            xc = x[s:s + CHUNK].float()
            b = torch.bucketize(xc.abs(), self.edges)
            idx = (((xc < 0).long() * NB + b) if poles else b) * n + cols
            flat.index_add_(0, idx.flatten(), torch.ones(idx.numel(), dtype=flat.dtype, device=x.device))

    def neuron(self, l: int, a: torch.Tensor) -> None:  # a: [B, T, f]
        if self.mode == "first":
            x = a[:, self.first_pos].float()  # [B, f]
            if self.ctx["chat"]:
                self.extra["first_chat"][l] = x[0].cpu()
                return
            ids = self.ctx["ids"]
            for p, sign in enumerate((1, -1)):
                v, i = torch.cat([self.extra["first_val"][l, p].to(x.device), sign * x]).topk(FIRST_K, 0)
                cand = torch.cat([self.extra["first_tok"][l, p].to(x.device), ids[:, None].expand_as(x).int()])
                self.extra["first_val"][l, p], self.extra["first_tok"][l, p] = v.cpu(), cand.gather(0, i).cpu()
            return
        if self.mode != "main":
            return
        m = self.ctx["n_mask"]
        xb = a[m]  # [P, f]
        for s in range(0, len(xb), CHUNK):
            xc = xb[s:s + CHUNK].float()
            self.seg["sum"][l] += xc.sum(0)
            self.seg["sumsq"][l] += xc.pow(2).sum(0)
        self.count(self.seg["count"][l], xb, poles=True)
        f = a.shape[-1]
        for p, sign in enumerate((1, -1)):
            ext, pos = (a * sign).masked_fill(~m[..., None], -torch.inf).max(1)  # [B, f]
            vals = torch.cat([self.top["val"][l, p], ext.float()])
            v, i = vals.topk(TOP_DOCS, dim=0)
            self.top["val"][l, p] = v
            self.top["doc"][l, p] = torch.cat([self.top["doc"][l, p], self.ctx["doc"][:, None].expand(-1, f)]).gather(0, i)
            self.top["pos"][l, p] = torch.cat([self.top["pos"][l, p], pos.to(torch.int16)]).gather(0, i)

    def input_norm(self, l: int, x: torch.Tensor, eps: float) -> None:
        if self.mode == "main":
            self.ctx["inv_rho"] = torch.rsqrt(x.float().pow(2).mean(-1) + eps)  # [B, T]

    def qk_norm(self, l: int, side: str, x: torch.Tensor, out: torch.Tensor, eps: float) -> None:
        if self.mode != "main":
            return
        x = x.float()  # [B, T, heads, hd], before the norm
        nh, hd = x.shape[-2:]  # this layer's heads and head size, the leading block of the buffers
        r = torch.rsqrt(x.pow(2).mean(-1) + eps)
        valid = self.ctx["qv"] if side == "q" else self.ctx["kv"]  # v: the values, at the keys
        w = (valid * self.ctx["inv_rho"])[..., None]  # positions, times the input norm's scaling
        self.seg[f"{side}_r"][l, :nh] += (r * w).sum((0, 1))
        self.seg[f"{side}_m"][l, :nh, :hd, :hd] += torch.einsum("bth,bthi,bthj->hij", r.pow(3) * w, x, x)
        if side == "q":
            self.seg["q"][l, :, :hd] += (out.float() * valid[..., None, None]).sum((0, 1))

    def attention(self, l: int, a: torch.Tensor, attn_out: torch.Tensor) -> None:  # a: [B, H, T, T]
        if self.mode == "rep":
            o = self.first_pos  # <bos> first, for Gemma
            t2 = torch.arange(o + REP_LEN, o + 2 * REP_LEN - 1, device=a.device)  # the second copy, with a successor
            self.extra["induction"][l] += a[:, :, t2, t2 - REP_LEN + 1].float().mean(-1).sum(0).cpu()
            self.extra["duplicate"][l] += a[:, :, t2, t2 - REP_LEN].float().mean(-1).sum(0).cpu()
            return
        if self.mode != "main":
            return
        z = self.stash.pop(l)  # [B, T, H·hd]
        B, T = z.shape[:2]
        H, hd = self.H, self.geo[l].head_dim
        hm = self.ctx["h_mask"]
        zh = z.view(B, T, H, hd).float()
        scale = arch.attn_scale(self.layers[l], attn_out)  # Gemma 4: the post-attention norm's scale
        if scale is not None:
            zh = zh * scale[..., None, None]
        norm = torch.einsum("bthi,hij,bthj->bth", zh, self.gram[l].to(zh), zh).clamp_min(0).sqrt()
        nv = norm[hm]  # [P, H]
        self.seg["h_sum"][l] += nv.sum(0)
        self.seg["h_sumsq"][l] += nv.pow(2).sum(0)
        self.count(self.seg["h_count"][l], nv, poles=False)
        v, t = norm.masked_fill(~hm[..., None], -1).max(1)  # [B, H]
        bi, hi = torch.arange(B, device=a.device)[:, None], torch.arange(H, device=a.device)[None, :]
        rows = a[bi, hi, t].float()  # [B, H, T]: the attention rows there
        rows[..., 0] = 0
        src = rows.argmax(-1)
        o = torch.einsum("bhi,dhi->bhd", zh[bi, t, hi], self.wo[l])
        vals = torch.cat([self.top["h_val"][l], v.T], 1)  # [H, K + B]
        keep = vals.topk(TOP_DOCS, dim=1).indices
        for k, new in (("h_val", v.T), ("h_doc", self.ctx["doc"][None, :].expand(H, -1)), ("h_pos", t.T.to(torch.int16)),
                       ("h_src", src.T.to(torch.int16))):
            self.top[k][l] = torch.cat([self.top[k][l], new], 1).gather(1, keep)
        cand = torch.cat([self.top["h_out"][l], o.transpose(0, 1).half()], 1)
        self.top["h_out"][l] = cand.gather(1, keep[..., None].expand(-1, -1, self.d))
        qv, kv = self.ctx["qv"], self.ctx["kv"]  # [B, T] float
        af = a.float()
        self.seg["sink"][l] += (af[..., 0] * qv[:, None]).sum((0, 2))
        self.seg["self"][l] += (af.diagonal(dim1=2, dim2=3) * qv[:, None]).sum((0, 2))
        self.seg["prev"][l] += (af.diagonal(-1, dim1=2, dim2=3) * qv[:, None, 1:]).sum((0, 2))
        rest = (1 - af[..., :1]).clamp_min(1e-6)  # renormalize without the sink
        af = af * qv[:, None, :, None] * kv[:, None, None, :] / rest
        recv = (af * torch.arange(T, device=a.device)[None, None, :, None]).sum(2)  # [B, H, T]: per key, times t keys
        self.seg["recv"][l].index_add_(1, self.ctx["ids"].flatten(), recv.transpose(0, 1).reshape(H, -1))
        self.seg["offset"][l, :, :T].index_add_(1, self.ctx["delta"][:T, :T].flatten(), af.sum(0).reshape(H, -1))

    def hooks(self, model) -> list:
        hs = []
        for l, layer in enumerate(model.layers):
            at = layer.self_attn
            hs.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, x, l=l: self.neuron(l, x[0])))
            hs.append(layer.input_layernorm.register_forward_pre_hook(
                lambda m, x, l=l: self.input_norm(l, x[0], arch.norm_eps(m))))
            for side, norm in (("q", at.q_norm), ("k", at.k_norm)) + ((("v", at.v_norm),) if hasattr(at, "v_norm") else ()):
                hs.append(norm.register_forward_hook(
                    lambda m, x, out, l=l, side=side: self.qk_norm(l, side, x[0], out, arch.norm_eps(m))))
            hs.append(at.o_proj.register_forward_pre_hook(lambda m, x, l=l: self.stash.__setitem__(l, x[0])))
            hs.append(at.register_forward_hook(lambda m, x, out, l=l: self.attention(l, out[1], out[0])))
        return hs

    def batch(self, ids: torch.Tensor, mask: torch.Tensor, docs: torch.Tensor) -> None:
        """Per-batch context: masks, token ids, and the key-token pair counts for ``recv``."""
        dev, T = self.dev, ids.shape[1]
        ar = torch.arange(T, device=dev)
        real = mask.bool()
        self.ctx.update(ids=ids, doc=docs, n_mask=real & (ar >= NEURON_FROM), h_mask=real & (ar >= HEAD_FROM),
                        qv=(real & (ar >= HEAD_FROM)).float(), kv=(real & (ar >= 1)).float())
        if "delta" not in self.ctx:
            ad = torch.arange(SEQ, device=dev)
            self.ctx["delta"] = (ad[:, None] - ad[None, :]).clamp_min(0)
        pairs = self.ctx["qv"].flip(1).cumsum(1).flip(1) * self.ctx["kv"]  # queries at or after each key
        self.seg["pairs"].index_add_(0, ids.flatten(), pairs.flatten())
        self.seg["occ"].index_add_(0, ids.flatten(), (pairs > 0).float().flatten())
        self.seg["n_q"] += self.ctx["qv"].sum()
        self.seg["n_k"] += self.ctx["kv"].sum()
        self.seg["n"] += int(self.ctx["n_mask"].sum())
        self.seg["h_n"] += int(self.ctx["h_mask"].sum())


def plan_batches(slice_of: list[int], batch_docs: int) -> list[tuple[int, int]]:
    """Runs of consecutive documents of one slice, ``batch_docs`` at a time."""
    out, start = [], 0
    while start < len(slice_of):
        end = start
        while end < len(slice_of) and end - start < batch_docs and slice_of[end] == slice_of[start]:
            end += 1
        out.append((start, end))
        start = end
    return out


def collect(args, docs: list) -> dict:
    device = args.device
    tok = tokenizer()
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    lm.requires_grad_(False)
    model = decoder_stack(lm)  # the decoder stack: Qwen3's .model, Gemma 4's language model
    rows = [r for _, _, r in docs]
    slice_of = [s for s, _, _ in docs]
    meta = {"model": MODEL, "slices": list(ACT_SLICES), "docs": len(rows), "rows_sha256": rows_sha256(rows), "seq_len": SEQ,
            "neuron_from": NEURON_FROM, "head_from": HEAD_FROM, "edges": EDGES.tolist(), "top_docs": TOP_DOCS,
            "batch_docs": args.batch_docs, "dtype": "bfloat16", "attention": "eager"}
    col = Collector(model, len(ACT_SLICES), device)
    plan = plan_batches(slice_of, args.batch_docs)
    next_batch, stage = 0, "main"
    if ACTS.exists():
        saved = torch.load(ACTS, weights_only=False, map_location="cpu")
        if saved["meta"] != meta:
            raise SystemExit(f"{ACTS} was collected with {saved['meta']}, not {meta}")
        col.load(saved["state"])
        next_batch, stage = saved["next_batch"], saved["stage"]
        if stage == "done":
            return saved
    lengths = torch.tensor([len(r) for r in rows])
    flat = torch.cat([torch.tensor(r, dtype=torch.int32) for r in rows])
    start = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])

    def save(stage: str) -> dict:
        out = {"meta": meta, "state": col.state(), "next_batch": next_batch, "stage": stage, "tokens": flat,
               "doc_start": start, "slice_of": torch.tensor(slice_of, dtype=torch.int8),
               "line_of": torch.tensor([ln for _, ln, _ in docs], dtype=torch.int32)}
        ACTS.parent.mkdir(parents=True, exist_ok=True)
        tmp = ACTS.with_suffix(".tmp")
        torch.save(out, tmp)
        os.replace(tmp, ACTS)
        return out

    hooks = col.hooks(model)
    saved_at, t0, done = time.monotonic(), time.perf_counter(), 0
    for bi in range(next_batch, len(plan)):
        a, b = plan[bi]
        T = int(lengths[a:b].max())
        ids = torch.zeros(b - a, T, dtype=torch.long)
        mask = torch.zeros(b - a, T, dtype=torch.long)
        for j, r in enumerate(rows[a:b]):
            ids[j, :len(r)] = torch.tensor(r)
            mask[j, :len(r)] = 1
        ids, mask = ids.to(device), mask.to(device)
        col.batch(ids, mask, torch.arange(a, b, dtype=torch.int32, device=device))
        with torch.inference_mode():
            model(input_ids=ids, attention_mask=mask)
        done += int(mask.sum())
        next_batch = bi + 1
        last = next_batch == len(plan)
        if last or bi % FLUSH == FLUSH - 1 or slice_of[plan[min(next_batch, len(plan) - 1)][0]] != slice_of[a]:
            col.flush(slice_of[a])
        if last or time.monotonic() - saved_at > 60 * args.save_minutes:
            col.flush(slice_of[a])
            save("main")
            saved_at = time.monotonic()
        if bi % 20 == 0 or last:
            print(f"{time.strftime('%H:%M:%S')} batch {bi + 1}/{len(plan)} ({ACT_SLICES[slice_of[a]]}), "
                  f"{done / (time.perf_counter() - t0):.0f} tokens/s", flush=True)

    # Every kept token alone: the first position.
    col.mode = "first"
    vocab_ids = kept()
    pre = arch.bos(tok)  # Gemma's documents open with <bos>: the token comes after it
    col.first_pos = len(pre)
    with_pre = lambda x: torch.cat([torch.tensor(pre, dtype=x.dtype, device=x.device).expand(len(x), -1), x], 1)  # noqa: E731
    with torch.inference_mode():
        col.ctx.update(chat=True)
        model(input_ids=with_pre(torch.tensor([[arch.chat_open(MODEL, tok)]], device=device)))
        col.ctx["chat"] = False
        for s in range(0, len(vocab_ids), 1024):
            col.ctx["ids"] = vocab_ids[s:s + 1024].to(device)
            model(input_ids=with_pre(col.ctx["ids"][:, None]))
        # Repeated random tokens, for induction and duplicate-token scores.
        col.mode = "rep"
        special = set(tok.all_special_ids)
        pool = torch.tensor([i for i in vocab_ids.tolist() if i not in special])
        half = pool[torch.randint(len(pool), (REP_SEQS, REP_LEN), generator=torch.Generator().manual_seed(0))]
        rep = torch.cat([half, half], 1)
        for s in range(0, REP_SEQS, 16):
            model(input_ids=with_pre(rep[s:s + 16].to(device)))
        col.extra["induction"] /= REP_SEQS
        col.extra["duplicate"] /= REP_SEQS
    col.extra["inv_freq"] = arch.inv_freq(model, MODEL)  # RoPE's angle per position (per layer type), for 08_reads
    for h in hooks:
        h.remove()
    print(f"{time.strftime('%H:%M:%S')} done: {int(col.total['n'].sum())} neuron positions, {len(rows)} documents", flush=True)
    return save("done")


def interp(E: torch.Tensor, n, t: torch.Tensor) -> torch.Tensor:
    """The share past ``t`` ``[k]`` from counts past the grid ``E`` ``[len(EDGES), k]``, linear between points."""
    edges = EDGES.to(t.dtype)
    i = (torch.searchsorted(edges, t.contiguous(), right=True) - 1).clamp(0, len(edges) - 2)
    w = ((t - edges[i]) / (edges[i + 1] - edges[i])).clamp(0, 1)
    e = E.gather(0, i[None]).squeeze(0) * (1 - w) + E.gather(0, (i + 1)[None]).squeeze(0) * w
    return e / n


def quantiles(grid: torch.Tensor, S: torch.Tensor, qs=(0.5, 0.9, 0.99, 0.999)) -> list[torch.Tensor]:
    """Quantiles from the survival ``S`` ``[len(grid), k]`` (nonincreasing) on ``grid``, linear between points."""
    out = []
    for q in qs:
        j = ((S >= 1 - q).sum(0) - 1).clamp(0, len(grid) - 2)
        s0, s1 = S.gather(0, j[None]).squeeze(0), S.gather(0, (j + 1)[None]).squeeze(0)
        w = ((s0 - (1 - q)) / (s0 - s1).clamp_min(1e-12)).clamp(0, 1)
        out.append(grid[j] + w * (grid[j + 1] - grid[j]))
    return out


def past(counts: torch.Tensor) -> torch.Tensor:
    """Counts past each grid point from bin counts (bins on dim -2)."""
    return counts.flip(-2).cumsum(-2).flip(-2)[..., 1:, :]


_SAVED = {}  # the collected state, shared with forked table workers


def neuron_tables(saved: dict, workers: int = 1) -> None:
    """One ``neurons/acts`` table per layer; with ``workers`` > 1, layers in forked processes (the
    per-pole contexts are Python string work, ~45 s a layer for Gemma 4's 30,720 poles)."""
    L = saved["state"]["top"]["val"].shape[0]
    _SAVED["s"] = saved
    if workers <= 1:
        for layer in range(L):
            _neuron_layer(layer)
        return
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(workers, L)) as pool:
        for _ in pool.imap_unordered(_neuron_layer, range(L)):
            pass


def _neuron_layer(layer: int) -> None:
    torch.set_num_threads(1)
    saved = _SAVED["s"]
    tok = tokenizer()
    dec = functools.cache(lambda i: tok.decode([i]))
    st, top, extra = saved["state"]["total"], saved["state"]["top"], saved["state"]["extra"]
    flat, start = saved["tokens"], saved["doc_start"]
    n_s = st["n"].double()
    n = n_s.sum()
    grid = torch.cat([-EDGES[1:].flip(0), EDGES]).double()
    f = top["val"].shape[-1]
    t0 = time.perf_counter()
    E_s = past(st["count"][:, layer].double())  # [S, 2, len(EDGES), f]
    E = E_s.sum(0)
    mean = st["sum"][:, layer].sum(0) / n
    std = (st["sumsq"][:, layer].sum(0) / n - mean.pow(2)).clamp_min(0).sqrt()
    cols = neuron_keys(layer, f)
    acc = {k: [] for k in ("act_top", "act_max", "pos_share", "rate", *[f"rate_{s}" for s in ACT_SLICES],
                           "q50", "q90", "q99", "q999", "mean", "std", "peak_tokens", "contexts")}
    for p in range(2):
        val = top["val"][layer, p]  # [K, f]
        act_top = val.median(0).values.clamp_min(0).double()
        acc["act_top"].append(act_top)
        acc["act_max"].append(val[0].double())
        acc["pos_share"].append(E[p, 0] / n)
        acc["rate"].append(interp(E[p], n, act_top / 4))
        for si, s in enumerate(ACT_SLICES):
            acc[f"rate_{s}"].append(interp(E_s[si, p], n_s[si].clamp_min(1), act_top / 4))
        S = torch.cat([1 - E[1 - p, 1:].flip(0) / n, E[p] / n])  # P(s·act > t) on the grid
        for k, q in zip(("q50", "q90", "q99", "q999"), quantiles(grid, S)):
            acc[k].append(q)
        acc["mean"].append(mean)
        acc["std"].append(std)
        docs, poss = top["doc"][layer, p].long(), top["pos"][layer, p].long()
        peak = flat[start[docs] + poss]  # [K, f]
        for nn in range(f):
            fired = (val[:, nn] > 0).nonzero().squeeze(1).tolist()
            c = Counter(int(peak[j, nn]) for j in fired)
            acc["peak_tokens"].append(" | ".join(f"{dec(t)!r}×{k}" for t, k in c.most_common(5)))
            acc["contexts"].append(" ‖ ".join(
                context(tok, flat[start[docs[j, nn]]:start[docs[j, nn] + 1]], int(poss[j, nn])) for j in fired[:N_CONTEXTS]))
    for k, v in acc.items():
        cols[k] = pa.array(v) if isinstance(v[0], str) else pa.array(torch.cat(v).float().numpy())
    for k in ("doc", "pos", "val"):
        cols[f"top_{k}"] = fixed(torch.cat([top[k][layer, 0].T, top[k][layer, 1].T]))
    fv, ft = extra["first_val"][layer], extra["first_tok"][layer]  # [2, K, f]
    cols["first_max"] = pa.array(fv[:, 0].flatten().numpy())
    cols["first_tokens"] = pa.array([" | ".join(f"{dec(int(ft[p, j, nn]))!r} {fv[p, j, nn]:.3g}" for j in range(FIRST_K))
                                     for p in range(2) for nn in range(f)])
    cols["first_chat"] = pa.array(torch.cat([extra["first_chat"][layer], -extra["first_chat"][layer]]).numpy())
    write(cols, neuron_table("acts", layer))
    print(f"{time.strftime('%H:%M:%S')} neurons/acts L{layer} in {time.perf_counter() - t0:.0f}s", flush=True)


def head_tables(saved: dict, device) -> None:
    tok = tokenizer()
    dec = functools.cache(lambda i: tok.decode([i]))
    st, top, extra = saved["state"]["total"], saved["state"]["top"], saved["state"]["extra"]
    flat, start = saved["tokens"], saved["doc_start"]
    L, H, K = top["h_val"].shape
    n_s = st["h_n"].double()
    n = n_s.sum()
    E_s = past(st["h_count"].double().reshape(len(n_s), L, NB, H)).transpose(1, 2)  # [S, len(EDGES), L, H]
    E = E_s.sum(0)
    out_top = top["h_val"].median(-1).values.double()  # [L, H]
    cols = head_keys(L, H, arch.groups(MODEL))
    cols["out_top"] = pa.array(out_top.flatten().float().numpy())
    cols["out_max"] = pa.array(top["h_val"][..., 0].flatten().numpy())
    flatE = E.reshape(len(EDGES), L * H)
    cols["rate"] = pa.array(interp(flatE, n, out_top.flatten() / 4).float().numpy())
    for si, s in enumerate(ACT_SLICES):
        cols[f"rate_{s}"] = pa.array(interp(E_s[si].reshape(len(EDGES), L * H), n_s[si].clamp_min(1), out_top.flatten() / 4).float().numpy())
    for k, q in zip(("q50", "q90", "q99", "q999"), quantiles(EDGES.double(), flatE / n)):
        cols[k] = pa.array(q.float().numpy())
    mean = st["h_sum"].sum(0) / n
    cols["mean"] = pa.array(mean.flatten().float().numpy())
    cols["std"] = pa.array((st["h_sumsq"].sum(0) / n - mean.pow(2)).clamp_min(0).sqrt().flatten().float().numpy())
    for k in ("sink", "prev", "self"):  # means over queries
        cols[k] = pa.array((st[k] / st["n_q"]).flatten().float().numpy())
    cols["induction"] = pa.array(extra["induction"].flatten().numpy())
    cols["duplicate"] = pa.array(extra["duplicate"].flatten().numpy())
    docs, pos, src = top["h_doc"].long(), top["h_pos"].long(), top["h_src"].long()
    cur, att = flat[start[docs] + pos], flat[start[docs] + src]
    cols["top_doc"], cols["top_pos"], cols["top_val"] = (fixed(x.reshape(L * H, K)) for x in (top["h_doc"], top["h_pos"], top["h_val"]))
    cols["src_pos"] = fixed(top["h_src"].reshape(L * H, K))
    for name, t in (("peak_tokens", cur), ("src_tokens", att)):
        cols[name] = pa.array([" | ".join(f"{dec(int(x))!r}×{c}" for x, c in Counter(t[l, h].tolist()).most_common(5))
                               for l in range(L) for h in range(H)])
    cols["contexts"] = pa.array([" ‖ ".join(context(tok, flat[start[docs[l, h, j]]:start[docs[l, h, j] + 1]], int(pos[l, h, j]),
                                                    int(src[l, h, j])) for j in range(N_CONTEXTS))
                                 for l in range(L) for h in range(H)])
    # The firings' outputs through the R-lens from the MLP input, each token's logit z-scored across the
    # layer's firings, averaged over the head's.
    keep = kept()
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted({nm["unembed"], nm["norm"]}))
    unembed = head[nm["unembed"]][keep].to(device, torch.float16)
    gamma = head[nm["norm"]].to(device, torch.float32)
    J = arch.lens(MODEL, "R_mid", R_FIT)
    ids, vals = [], []
    for l in range(L):
        O = top["h_out"][l].reshape(H * K, -1).to(device, torch.float32).T  # [d, H·K]
        lg = arch.cap(MODEL, unembed @ rms_norm_columns(J[l].to(device) @ O, gamma).half()).float()  # [V, H·K]
        z = (lg - lg.mean(1, keepdim=True)) / lg.std(1, keepdim=True).clamp_min(1e-6)
        mz = z.view(-1, H, K).mean(-1).T  # [H, V]
        v, i = mz.topk(TOP, dim=1)
        ids.append(keep[i.cpu()])
        vals.append(v.cpu())
    put(cols, "out_R_z", torch.cat(ids), torch.cat(vals))
    write(cols, head_table("acts"))
    print(f"{time.strftime('%H:%M:%S')} heads/acts written", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=None,
                    help="at most this many lines per slice: a smoke test of the pass, which writes no tables")
    ap.add_argument("--batch-docs", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    ap.add_argument("--save-minutes", type=float, default=10)
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("--table-workers", type=int, default=16 if torch.cuda.is_available() else 1,
                    help="processes writing the neuron tables, one layer each (forked)")
    args = ap.parse_args()
    if args.detach:
        LOGS.mkdir(exist_ok=True)
        log = LOGS / "activations.log"
        cmd = [sys.executable, __file__, *[a for a in sys.argv[1:] if a != "--detach"]]
        if sys.platform == "darwin":
            cmd = ["caffeinate", "-i", *cmd]
        with open(log, "a") as out:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                                 env={**os.environ, "PYTHONUNBUFFERED": "1"})
        print(f"detached: pid {p.pid}, log {log}")
        return
    global ACTS
    if args.docs is not None:
        ACTS = ACTS.with_name(f"{ACTS.stem}-smoke{args.docs}.pt")
    docs = act_docs(tokenizer(), SEQ, NEURON_FROM + 8, args.docs)
    saved = collect(args, docs)
    if args.docs is not None:  # the tables are the catalog's; a partial pass must not replace them
        print(f"smoke test done: {ACTS.name} saved, tables not written", flush=True)
        return
    neuron_tables(saved, args.table_workers)
    head_tables(saved, args.device)
    write({"doc": pa.array(range(len(docs)), pa.int32()), "slice": pa.array([ACT_SLICES[s] for s, _, _ in docs]),
               "line": pa.array([ln for _, ln, _ in docs], pa.int32()),
               "tokens": pa.array([len(r) for _, _, r in docs], pa.int16())}, catalog_dir() / "docs.parquet")


if __name__ == "__main__":
    main()
