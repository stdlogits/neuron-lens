"""Activations of every MLP neuron and attention head over the activation corpora (pipeline stage 6).

One forward pass per batch of documents, eager attention in bf16, over every document of
``05_act_corpus.py``'s slices (web, Chinese, Python, chat), up to 256 tokens each. Collected:

- **neurons,** at each layer's ``down_proj`` input (the activations ``silu(gate)·up``), from
  position 1 (position 0 is the attention sink, where activations are massive): per slice, the
  sum and sum of squares, and exact counts of each pole's activation (``±act``) past every point
  of a quarter-octave grid from 2⁻¹⁰ to 2¹⁰ (rates and quantiles interpolate between them); per
  pole, the top ``TOP`` documents by its extreme in the document, with the position and the gate
  pre-activation there (``neg_lobe``: the activation-weighted share of its firings on SiLU's negative lobe,
  below zero);
- **heads,** at each layer's ``o_proj`` input, from position 16 (where the lenses start; a head
  with few keys attends sharply): the norm of each head's output ``W_O^h z``, counted on the same
  grid, and its top ``TOP`` documents, each with the position it attends to most past the sink
  and the output vector. Also each head's mean attention from those queries to position 0
  (``sink``), the position before (``prev``) and itself (``self``);
- **QK statistics** for ``08_reads.py`` (findings §10): per head, the mean query after ``q_norm``;
  the sums giving the true derivative of the input norm's scaling and ``q_norm`` or ``k_norm``
  (Gemma 4: and the weightless ``v_norm``, for the OV reads; GPT-NeoX, without q/k norms: the input norm's own,
  ``in_r`` and ``in_m``) averaged over positions; attention by offset; and the attention each key token receives (from
  queries at 16+, renormalized without the sink, times the number of keys: 1 is uniform). Rows are
  renormalized by their non-sink weights' sum: ``1 − a₀`` rounds to 0 in bf16 where the sink takes
  nearly everything. The released Qwen3-8B and Gemma 4 passes predate this and inflate sink heads' rows,
  which understates late heads' ``qk_ctx_rho`` slightly (Qwen3-8B: by up to 0.03 in a band; findings §10).

Then two small passes: every kept token alone, the first position, where a causal model's
activations depend on the token alone (``first_max``, ``first_tokens``, ``first_chat`` on the chat
template's opening token); and ``REP_SEQS`` sequences of ``REP_LEN`` random kept tokens said twice,
for the heads' ``induction`` (attention from the second copy to the token after the first
occurrence) and ``duplicate`` (to the occurrence itself) scores.

**Gated DeltaNet heads** (Qwen3.5's ``linear_attention`` layers, Olmo Hybrid's): a head is a value head (its block of
the output projection), and a layer's heads may outnumber its attention layers' (Qwen3.5-0.8B: 16 against 8), so head
statistics sit in a grid of ``arch.max_heads`` a layer, each layer's in its leading entries, and the tables list the
heads that exist. A DeltaNet head mixes its values through ``A = W L⁻¹ diag(β)`` (``o = A v`` exactly,
``layers.delta_pattern``, computed from the mixer's input). The attention statistics read ``|A|`` with each row at
unit sum, a position's share of what its output mixes from each earlier position's value: ``sink``, ``prev``,
``self``, the attended position, the offsets, ``recv``, ``induction`` and ``duplicate`` mean the same as for softmax
heads, on that pattern (the value's four-tap convolution, which reaches three positions further back, stays out of
it). For ``08_reads.py``: the sums of the L2 norms' true derivative on the queries (per value head, as each reads its
key head's query) and keys, the mean query after its norm and scale, SiLU's ratio ``σ(c)`` per convolution channel,
and per channel the output's gate (the output norm's scale and rsqrt times ``silu(z)``); and gated attention's
``σ(gate)`` per channel (Qwen3.5's full layers), the gates the OV reads average.

Gemma 4 12B (``NEURON_LENS_MODEL=google/gemma-4-12B``): GELU-tanh activations; per-layer head
sizes and key heads (the QK statistics padded to the largest, 8 key heads of 512); each head's
output as it reaches the residual (the post-attention norm's weight and its scale per position);
every document starts with ``<bos>``, so the first-position and repeated-token passes put it
before their tokens (the token's position is 1); RoPE per layer type; the released R-lens.

A sparse MLP (Qwen3.5-35B-A3B, Qwen3-30B-A3B, OLMoE): its neurons are its experts' side by side, then its shared
expert's, each activation ``ρ · silu(g) · u`` where the router chooses the neuron's expert (``ρ`` the routing weight)
and 0 where it doesn't (``neuron_lens.experts``). Only the routed pairs are computed and counted; a routed neuron's other
positions land in the grid's zero bin, and its top documents are those its expert is routed to at all (padded where
fewer than ``TOP``). Per expert, for ``experts/acts.parquet``: the positions routed to it, its routing weights, its top
documents by routing weight, and its neurons' RelP factors summed where routed (``expert_tables``).

**Layer groups** (``--layers a-b``): the neurons of those layers alone, so a model whose top lists don't fit beside its
weights runs in several passes. The group from layer 0 collects the heads too and saves to the activations file; the
others save to ``data/acts/<model>-L<a>-<b>.pt`` and stop each forward pass after layer ``b``. Each pass writes its
layers' tables. The neuron buffers on the device take ``f · (P · TOP · 12 + 2 · NB · 4)`` bytes a layer (``P`` poles,
the top lists' value, document, position and gate, the grid counts): 290 MB for Qwen3.5-35B-A3B's 131,584 neurons,
11.6 GB for its 40 layers, which with its 67 GB of weights and the heads' statistics (``recv`` alone 1.3 GB) doesn't
fit an 80 GB GPU in one pass; four groups of ten layers do. On the CPU the totals take ``f · 2 · NB · 8 · 4`` bytes a
layer for the four slices' counts (700 MB for the 35B-A3B).

The state is saved to ``data/acts/<model>.pt`` every ``--save-minutes`` and resumes from there.
Once the pass is complete, the script writes the acts tables, ``neurons/acts/L<layer>.parquet``
and ``heads/acts.parquet``, and ``docs.parquet`` (each document's slice and line in the slice's
corpus file); a rerun rebuilds them. The heads' ``out_R_z`` reads their firings' outputs through
the R-lens from the MLP input. ``cur_mode`` and ``next_mode``: the share of a pole's top firings on its
most common peak token, or before its most common next token; for a head, ``cur_mode`` and ``src_mode``,
on its most common query token, or attending most to its most common token (findings §8, §11).

    python scripts/06_activations.py [--detach] [--docs N] [--layers a-b]
"""
import argparse
import functools
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from neuron_lens import anatomy, arch, experts
from neuron_lens.layers import delta_pattern
try:
    from neuron_lens import kernels
except ImportError:
    kernels = None
from neuron_lens.catalog import (
    MODEL,
    N_CONTEXTS,
    TOP,
    TYPICAL,
    acts_path,
    catalog_dir,
    context,
    expert_keys,
    expert_table,
    firing_tokens,
    fixed,
    head_keys,
    head_table,
    kept,
    neuron_keys,
    neuron_table,
    put,
    tokenizer,
    write,
    write_heads,
)
from neuron_lens.corpus import ACT_SLICES, act_docs
from neuron_lens.paths import LOGS, ROOT
from neuron_lens.readout import rms_norm_columns
from neuron_lens.rules import attention
from neuron_lens.stats import modal_share
from neuron_lens.transport import SKIP_FIRST
from neuron_lens.weights import tensors

SEQ = 256
NEURON_FROM, HEAD_FROM = 1, SKIP_FIRST
EDGES = torch.tensor([0.0] + [2.0 ** (k / 4) for k in range(-40, 41)])  # counts of |x| past each
NB = len(EDGES) + 1  # bins: 0 is x = 0, i is (EDGES[i-1], EDGES[i]], the last past 2¹⁰
FLUSH = 50  # batches between moving the device's running sums into the float64 totals
MERGE_EVERY = 4  # batches buffered before merging neuron top-k (cuts GPU→CPU .nonzero() syncs 4×)
CHUNK = 2048  # positions per counting step
REP_SEQS, REP_LEN = 64, 64
FIRST_K = 3
R_FIT = arch.fits(MODEL)[0]
ACTS = acts_path()


def bf16_bins() -> torch.Tensor:
    """Every bf16 bit pattern's bin in ``count``: its bin among ``EDGES``, plus ``NB`` if it is negative."""
    v = torch.from_numpy((np.arange(65536, dtype=np.uint32) << 16).view(np.float32).copy())  # a bf16 is the top half of a float
    return ((v < 0).int() * NB + torch.bucketize(v.abs(), EDGES)).int()


NEURON_BUFFERS = {"n", "sum", "sumsq", "count", "e_n", "e_w", "e_cg", "e_cu"}  # the rest are the heads'


def buffers(L: int, nl: int, f: int, E: int, H: int, kvh: int, hd: int, V: int, dn: int, C: int) -> dict:
    """Running sums: ``name: (shape, integer, per slice)``, the neurons' over ``nl`` layers (``--layers``). ``E``: a
    sparse MLP's experts (0: dense), for the expert tables: the positions routed to each (``e_n``), their routing weights
    (``e_w``) and RelP's factors per neuron (``e_cg``, ``e_cu``: ``experts.factors``), summed. ``dn``: the residual's
    width where the scores read the input norm's own derivative (no q/k norms, GPT-NeoX), else 0; ``C``: a DeltaNet's
    convolution channels, else 0."""
    routed = {"e_n": ((nl, E), True, True), "e_w": ((nl, E), False, False), "e_cg": ((nl, f), False, False),
              "e_cu": ((nl, f), False, False)} if E else {}
    return {"n": ((), True, True), "sum": ((nl, f), False, True), "sumsq": ((nl, f), False, True),
            "count": ((nl, 2, NB, f), True, True), **routed,
            "h_n": ((), True, True), "h_sum": ((L, H), False, True), "h_sumsq": ((L, H), False, True),
            "h_count": ((L, NB, H), True, True),
            "sink": ((L, H), False, False), "prev": ((L, H), False, False), "self": ((L, H), False, False),
            "n_q": ((), False, False), "n_k": ((), False, False), "q": ((L, H, hd), False, False),
            "q_r": ((L, H), False, False), "q_m": ((L, H, hd, hd), False, False), "k_r": ((L, kvh), False, False),
            "k_m": ((L, kvh, hd, hd), False, False), "offset": ((L, H, SEQ), False, False),
            "v_r": ((L, kvh), False, False), "v_m": ((L, kvh, hd, hd), False, False),  # Gemma 4's v_norm (else 0)
            "in_r": ((L,), False, False), "in_m": ((L, dn, dn), False, False),  # the input norm, without q/k norms
            "og": ((L, H, hd), False, False),  # an output gate's factor per channel (gated attention, DeltaNet), else 0
            "sig": ((L, C), False, False),  # a DeltaNet's SiLU factor per convolution channel
            "recv": ((L, H, V), False, False), "pairs": ((V,), False, False), "occ": ((V,), False, False)}


class Collector:
    def __init__(self, model, n_slices: int, device, layers: list[int] | None = None, heads: bool = True):
        cfg = model.config  # the decoder stack's (text) config
        L, H = cfg.num_hidden_layers, arch.max_heads(MODEL)  # heads: a grid, each layer's first
        f = experts.stack_width(model)  # a sparse layer's: its experts' neurons side by side, then its shared expert's
        layers = list(range(L)) if layers is None else layers
        self.nl, self.heads = {l: i for i, l in enumerate(layers)}, heads  # a neuron layer's row in the neuron buffers
        sparse = [lay.mlp.experts for lay in model.layers if experts.sparse(lay)]
        self.E, self.I = (sparse[0].num_experts, sparse[0].intermediate_dim) if sparse else (0, 0)
        kvh, hd = arch.padded(MODEL)  # per-layer statistics sit in the leading block of these
        d, V = cfg.hidden_size, cfg.vocab_size
        self.geo = arch.layout(MODEL)
        self.L, self.f, self.H, self.hd, self.d, self.dev = L, f, H, hd, d, device
        self.dn = d if arch.is_neox(MODEL) else 0  # no q/k norms: the scores read the input norm's true derivative
        C = max((m.linear_attn.conv_dim for m in model.layers if hasattr(m, "linear_attn")), default=0)
        self.spec = buffers(L, len(layers), f, self.E, H, kvh, hd, V, self.dn, C)
        if not heads:
            self.spec = {k: v for k, v in self.spec.items() if k in NEURON_BUFFERS}
        self.total = {k: torch.zeros(((n_slices,) if per else ()) + shape, dtype=torch.int64 if integer else torch.float64)
                      for k, (shape, integer, per) in self.spec.items()}
        self.seg = {k: torch.zeros(shape, dtype=torch.int32 if integer else torch.float32, device=device)
                    for k, (shape, integer, per) in self.spec.items()}
        k, P = TOP, arch.n_poles(MODEL)
        self.signs = arch.signs(MODEL)
        nl = len(layers)
        self.top = {"val": torch.full((nl, P, k, f), -torch.inf, device=device),
                    "doc": torch.zeros((nl, P, k, f), dtype=torch.int32, device=device),
                    "pos": torch.zeros((nl, P, k, f), dtype=torch.int16, device=device),
                    "gate": torch.zeros((nl, P, k, f), dtype=torch.float16, device=device),
                    "h_val": torch.full((L, H, k), -torch.inf, device=device),
                    "h_doc": torch.zeros((L, H, k), dtype=torch.int32, device=device),
                    "h_pos": torch.zeros((L, H, k), dtype=torch.int16, device=device),
                    "h_src": torch.zeros((L, H, k), dtype=torch.int16, device=device),
                    "h_out": torch.zeros((L, H, k, d), dtype=torch.float16, device=device)}
        if self.E:  # each expert's top documents by its routing weight
            self.top |= {"e_val": torch.full((nl, self.E, k), -torch.inf, device=device),
                         "e_doc": torch.zeros((nl, self.E, k), dtype=torch.int32, device=device),
                         "e_pos": torch.zeros((nl, self.E, k), dtype=torch.int16, device=device)}
        self.extra = {**({"expert_width": self.I} if self.E else {}),"first_val": torch.full((nl, P, FIRST_K, f), -torch.inf), "first_tok": torch.zeros((nl, P, FIRST_K, f), dtype=torch.int32),
                      "first_chat": torch.zeros(nl, f), "induction": torch.zeros(L, H), "duplicate": torch.zeros(L, H)}
        self.layers = model.layers
        self.wo = [arch.head_write(layer).view(d, g.q_heads, g.head_dim) for layer, g in zip(model.layers, self.geo)] if heads else []
        self.gram = [torch.einsum("dhi,dhj->hij", w, w).float() for w in self.wo]  # |W_O^h z|² = zᵀ G_h z
        self.first_pos = 0  # the token's position in the first-position and repeated passes (1 after <bos>)
        self.edges, self.bins = EDGES.to(device), bf16_bins().to(device)
        self.pending_top: dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
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

    def count(self, cnt: torch.Tensor, x: torch.Tensor) -> None:
        """Add the grid counts of ``x`` ``[P, n]`` into ``cnt``: by sign and bin (``[2, NB, n]``; bf16 read off a table of
        every bit pattern), or by bin (``[NB, n]``)."""
        n = x.shape[1]
        cols = torch.arange(n, device=x.device, dtype=torch.int32)
        flat = cnt.view(-1)
        for s in range(0, len(x), CHUNK):
            xc = x[s:s + CHUNK]
            if xc.dtype == torch.bfloat16:
                b = self.bins[xc.view(torch.int16).int() & 0xFFFF]
            else:
                b = torch.bucketize(xc.abs().float(), self.edges, out_int32=True)
                if cnt.dim() == 3:  # signed (fp16 activations: Pythia)
                    b = b + NB * (xc < 0).int()
            flat.index_add_(0, (b * n + cols).flatten(), torch.ones(xc.numel(), dtype=flat.dtype, device=x.device))

    def count_at(self, cnt: torch.Tensor, x: torch.Tensor, cols: torch.Tensor) -> None:
        """``count`` of fp32 values ``x`` at the columns ``cols`` (as ``x``): by sign and bin, into ``cnt`` ``[2, NB, f]``."""
        n, flat = cnt.shape[-1], cnt.view(-1)
        for s in range(0, len(x), CHUNK):
            xc = x[s:s + CHUNK]
            b = torch.bucketize(xc.abs(), self.edges, out_int32=True) + NB * (xc < 0).int()
            flat.index_add_(0, (b * n + cols[s:s + CHUNK]).flatten(), torch.ones(xc.numel(), dtype=flat.dtype, device=x.device))

    def first(self, li: int, x: torch.Tensor) -> None:
        """The first-position pass: ``x`` ``[B, f]``, each token's activations (NaN: a routed neuron whose expert isn't
        chosen, no entry). Runs along contiguous ``dim=1`` on GPU-resident ``first_val``/``first_tok`` buffers."""
        ids = self.ctx["ids"].int()[None, :].expand(x.shape[1], -1)  # [f, B]
        xt = x.T.contiguous()  # [f, B]
        nan = xt.isnan()
        fv, ft = self._first_val, self._first_tok
        for p, sign in enumerate(self.signs):
            sx = torch.where(nan, -torch.inf, xt if sign == 1 else -xt)
            v, i = torch.cat([fv[li, p], sx], dim=1).topk(FIRST_K, dim=1)
            cand = torch.cat([ft[li, p], ids], dim=1)
            fv[li, p], ft[li, p] = v, cand.gather(1, i)

    def _merge_now(self, li: int, p: int, ext: torch.Tensor, pos: torch.Tensor, gate_at: torch.Tensor,
                   doc: torch.Tensor) -> None:
        """Merge buffered extremes ``ext`` ``[B, f]``, ``pos`` ``[B, f]``, ``gate_at`` ``[B, f]`` and ``doc`` ``[B]``
        into ``self.top`` for ``(li, p)``."""
        cols = (ext.amax(0) > self.top["val"][li, p, -1]).nonzero().squeeze(1)
        if not len(cols):
            return
        v, i = torch.cat([self.top["val"][li, p][:, cols], ext[:, cols]]).topk(TOP, dim=0)
        self.top["val"][li, p][:, cols] = v
        new = {"gate": gate_at[:, cols].half(), "doc": doc[:, None].expand(-1, len(cols)), "pos": pos[:, cols].to(torch.int16)}
        for k, x in new.items():
            self.top[k][li, p][:, cols] = torch.cat([self.top[k][li, p][:, cols], x]).gather(0, i)

    def queue_top(self, li: int, p: int, ext: torch.Tensor, pos: torch.Tensor, gate_at: torch.Tensor) -> None:
        """Buffer a batch's extremes and merge every ``MERGE_EVERY`` batches to amortize ``.nonzero()`` syncs."""
        buf = self.pending_top.setdefault((li, p), [])
        buf.append((ext, pos, gate_at, self.ctx["doc"]))
        if len(buf) >= MERGE_EVERY:
            self.flush_top((li, p))

    def flush_top(self, key: tuple[int, int] | None = None) -> None:
        keys = [key] if key is not None else list(self.pending_top)
        for k in keys:
            buf = self.pending_top.get(k)
            if not buf:
                continue
            ext = torch.cat([x[0] for x in buf], 0)
            pos = torch.cat([x[1] for x in buf], 0)
            gate_at = torch.cat([x[2] for x in buf], 0)
            doc = torch.cat([x[3] for x in buf], 0)
            buf.clear()
            self._merge_now(k[0], k[1], ext, pos, gate_at, doc)

    def neuron(self, l: int, a: torch.Tensor) -> None:  # a: [B, T, f]
        li = self.nl[l]
        if self.mode == "first":
            x = a[:, self.first_pos].float()  # [B, f]
            if self.ctx["chat"]:
                self.extra["first_chat"][li] = x[0].cpu()
                return
            self.first(li, x)
            return
        if self.mode != "main":
            return
        m = self.ctx["n_mask"]
        xb = a[m]  # [P, f]
        if kernels is not None and xb.is_cuda and xb.dtype == torch.bfloat16:
            kernels.bf16_stats_and_hist(xb, self.bins, self.seg["sum"][li], self.seg["sumsq"][li], self.seg["count"][li])
        else:
            for s in range(0, len(xb), CHUNK):
                xc = xb[s:s + CHUNK].float()
                self.seg["sum"][li] += xc.sum(0)
                self.seg["sumsq"][li] += xc.pow(2).sum(0)
            self.count(self.seg["count"][li], xb)
        gate = self.stash.pop(("gate", l))  # [B, T, f]: the gate's pre-activation (GPT-NeoX: dense_h_to_4h's output)
        nm = ~m[..., None]
        for p, sign in enumerate(self.signs):
            if sign > 0:
                ext, pos = a.masked_fill(nm, -torch.inf).max(1)
            else:
                ext, pos = a.masked_fill(nm, torch.inf).min(1)
                ext = -ext
            gate_at = gate.gather(1, pos[:, None, :]).squeeze(1)
            self.queue_top(li, p, ext.float(), pos, gate_at)

    def sparse_neuron(self, l: int, r: experts.Routed) -> None:
        """``neuron`` for a sparse MLP's pass (``experts.Routed``): its routed pairs counted, summed and ranked where they
        are, every other valid position of a routed neuron in bin 0 (its activation, 0), the shared expert's neurons as a
        dense MLP's; and the expert tables' sums (positions routed, routing weights, RelP's factors) and each expert's top
        documents by its routing weight."""
        li, f, E, I = self.nl[l], self.f, self.E, self.I
        if self.mode == "first":
            T = self.first_pos + 1
            at = torch.arange(r.n, device=r.act.device) % T == self.first_pos
            if self.ctx["chat"]:
                self.extra["first_chat"][li] = r.dense(at)[0].cpu()
                return
            self.first(li, r.dense(at, float("nan")))
            return
        if self.mode != "main":
            return
        m = self.ctx["n_mask"]
        B, T = m.shape
        valid = m.flatten()
        keep = valid[r.rows]
        rows, act, gate = r.rows[keep], r.act[keep], r.gate[keep]
        e = r.expert[keep]
        cols = r.cols()[keep]
        self.seg["sum"][li].index_add_(0, cols.flatten(), act.flatten())
        self.seg["sumsq"][li].index_add_(0, cols.flatten(), act.pow(2).flatten())
        cnt = self.seg["count"][li]
        self.count_at(cnt, act, cols)
        routed_e = torch.bincount(e, minlength=E)
        cnt[0, 0, :E * I] += (valid.sum() - routed_e).repeat_interleave(I).to(cnt.dtype)  # the unrouted positions: 0
        sh = None if r.shared is None else r.shared.view(B, T, -1)
        if sh is not None:
            sv = sh[m]  # [P, Is]
            for s in range(0, len(sv), CHUNK):
                self.seg["sum"][li, E * I:] += sv[s:s + CHUNK].sum(0)
                self.seg["sumsq"][li, E * I:] += sv[s:s + CHUNK].pow(2).sum(0)
            self.count_at(cnt, sv, (E * I + torch.arange(sv.shape[1], device=sv.device)).expand_as(sv))
        # the expert tables: positions, weights and factors where routed; each expert's top documents by routing weight
        self.seg["e_n"][li] += routed_e.to(torch.int32)
        w = r.weight[keep]
        self.seg["e_w"][li].index_add_(0, e, w)
        fc, cg, cu = experts.factors(r, valid, self.layers[l].mlp.experts.act_fn)
        self.seg["e_cg"][li].index_add_(0, fc.flatten(), cg.flatten())
        self.seg["e_cu"][li].index_add_(0, fc.flatten(), cu.flatten())
        doc_row, t = rows // T, rows % T
        ev = torch.full((B * E,), -torch.inf, device=w.device).scatter_reduce_(0, doc_row * E + e, w, "amax")
        epos = torch.zeros(B * E, dtype=torch.long, device=w.device)
        hit = w == ev[doc_row * E + e]
        epos[(doc_row * E + e)[hit]] = t[hit]
        ev, epos = ev.view(B, E).T, epos.view(B, E).T  # [E, B]
        top = torch.cat([self.top["e_val"][li], ev], 1).topk(TOP, dim=1).indices
        for k, new in (("e_val", ev), ("e_doc", self.ctx["doc"][None, :].expand(E, -1)), ("e_pos", epos.to(torch.int16))):
            self.top[k][li] = torch.cat([self.top[k][li], new.to(self.top[k].dtype)], 1).gather(1, top)
        # each pole's extreme in each document, its position and gate there
        key = (doc_row[:, None] * f + cols).flatten()
        tk = t[:, None].expand_as(cols).flatten()
        for p, sign in enumerate(self.signs):
            v = (sign * act).flatten()
            ext = torch.full((B * f,), -torch.inf, device=v.device).scatter_reduce_(0, key, v, "amax")
            pos = torch.zeros(B * f, dtype=torch.long, device=v.device)
            g_at = torch.zeros(B * f, device=v.device)
            hit = v == ext[key]
            pos[key[hit]], g_at[key[hit]] = tk[hit], gate.flatten()[hit]
            ext, pos, g_at = ext.view(B, f), pos.view(B, f), g_at.view(B, f)
            if sh is not None:
                se, sp = (sign * sh).masked_fill(~m[..., None], -torch.inf).max(1)  # [B, Is]
                ext[:, E * I:], pos[:, E * I:] = se, sp
                g_at[:, E * I:] = r.shared_gate.view(B, T, -1).gather(1, sp[:, None]).squeeze(1)
            self._merge_now(li, p, ext, pos, g_at, self.ctx["doc"])

    def no_input_norm(self, x: torch.Tensor) -> None:
        if self.mode == "main":
            self.ctx["inv_rho"] = torch.ones(x.shape[:2], device=x.device)

    def input_norm(self, l: int, x: torch.Tensor, eps: float) -> None:
        if self.mode == "main":
            x = x.float()
            if arch.centred(MODEL):
                x = x - x.mean(-1, keepdim=True)
            self.ctx["inv_rho"] = r = torch.rsqrt(x.pow(2).mean(-1) + eps)  # [B, T]
            if self.dn:  # the sums of its true derivative r I − r³ x xᵀ / d, at the queries' positions
                w = r * self.ctx["qv"]
                self.seg["in_r"][l] += w.sum()
                xf = x.flatten(0, 1)
                self.seg["in_m"][l] += (xf * (r.pow(2) * w).flatten()[:, None]).T @ xf

    def query(self, l: int, qkv: torch.Tensor) -> None:
        """The mean query without q/k norms (GPT-NeoX's, from its fused projection: per head the query rows first,
        bias included, before RoPE), as ``qk_norm`` takes it after the q norm."""
        if self.mode != "main":
            return
        B, T = qkv.shape[:2]
        H = self.geo[l].q_heads
        q = qkv.view(B, T, H, 3, -1)[..., 0, :].float()
        self.seg["q"][l, :H, :q.shape[-1]] += (q * self.ctx["qv"][..., None, None]).sum((0, 1))

    def qk_norm(self, l: int, side: str, x: torch.Tensor, out: torch.Tensor, eps: float) -> None:
        """A q/k norm's sums per head: over each head (``[B, T, heads, hd]``), or over the whole projection (OLMo's,
        ``[B, T, heads·hd]``: one rsqrt a position, the heads' blocks of its derivative; ``08_reads.py`` divides by the
        width the norm spans)."""
        if self.mode != "main":
            return
        x = x.float()  # before the norm
        if x.dim() == 3:  # over the whole projection
            g = self.geo[l]
            nh = g.q_heads if side == "q" else g.kv_heads
            r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps).expand(*x.shape[:2], nh)
            x, out = x.view(*x.shape[:2], nh, -1), out.view(*x.shape[:2], nh, -1)
        else:
            r = torch.rsqrt(x.pow(2).mean(-1) + eps)
        nh, hd = x.shape[-2:]  # this layer's heads and head size, the leading block of the buffers
        valid = self.ctx["qv"] if side == "q" else self.ctx["kv"]  # v: the values, at the keys
        w = (valid * self.ctx["inv_rho"])[..., None]  # positions, times the input norm's scaling
        self.seg[f"{side}_r"][l, :nh] += (r * w).sum((0, 1))
        self.seg[f"{side}_m"][l, :nh, :hd, :hd] += torch.einsum("bth,bthi,bthj->hij", r.pow(3) * w, x, x)
        if side == "q":
            self.seg["q"][l, :nh, :hd] += (out.float() * valid[..., None, None]).sum((0, 1))

    def out_gate(self, l: int, q: torch.Tensor) -> None:
        """Gated attention's output gate (Qwen3.5: ``q_proj`` gives per head the query, then the gate): ``σ(gate)`` per
        channel, summed over the queries, the factor the OV reads take on a head's output."""
        if self.mode != "main":
            return
        g = self.geo[l]
        gate = q.view(*q.shape[:2], g.q_heads, 2 * g.head_dim)[..., g.head_dim:].float()
        self.seg["og"][l, :g.q_heads, :g.head_dim] += (torch.sigmoid(gate) * self.ctx["qv"][..., None, None]).sum((0, 1))

    def delta(self, l: int, la, x: torch.Tensor) -> None:
        """A Gated DeltaNet layer (its input ``x`` ``[B, T, d]``): its mixing matrix ``A`` (``o = A v``, ``layers.delta_pattern``)
        as the attention statistics' pattern, ``|A|`` with each row at unit sum (a position's share of what its output
        mixes, from each earlier position's value); the sums of its L2 norms' true derivative as ``qk_norm``'s (an L2 norm
        spans width 1: ``r I − r³ x xᵀ``, ``r = 1/‖x‖``), the queries' per value head, the keys' per key head; the
        SiLU's factor ``σ(c)`` per convolution channel (the identity rule's, at the queries for the query channels, at
        the keys for the rest); and the output's gate, the output norm's scale and rsqrt times ``silu(z)``, per channel."""
        if self.mode not in ("main", "rep"):
            return
        p = delta_pattern(la, x)
        if self.mode == "main":
            g, qv, kv, inv = self.geo[l], self.ctx["qv"], self.ctx["kv"], self.ctx["inv_rho"]
            hv, hk, dk = g.q_heads, g.kv_heads, g.kd
            for side, x0, valid in (("q", p.q0, qv), ("k", p.k0, kv)):
                r = torch.rsqrt(x0.pow(2).sum(-1) + 1e-6)  # [B, T, Hk]: layers._l2's
                w = (valid * inv)[..., None]
                rs, ms = (r * w).sum((0, 1)), torch.einsum("bth,bthi,bthj->hij", r.pow(3) * w, x0, x0)
                if side == "q":  # per value head, as each reads its key head's query
                    rs, ms = rs.repeat_interleave(hv // hk), ms.repeat_interleave(hv // hk, 0)
                    mean = (x0 * (r * dk ** -0.5 * valid[..., None])[..., None]).sum((0, 1))
                    self.seg["q"][l, :hv, :dk] += mean.repeat_interleave(hv // hk, 0)
                nh = len(rs)
                self.seg[f"{side}_r"][l, :nh] += rs
                self.seg[f"{side}_m"][l, :nh, :dk, :dk] += ms
            s, kd = torch.sigmoid(p.c), la.key_dim
            self.seg["sig"][l, :kd] += (s[..., :kd] * qv[..., None]).sum((0, 1))
            self.seg["sig"][l, kd:s.shape[-1]] += (s[..., kd:] * kv[..., None]).sum((0, 1))
            self.seg["og"][l, :hv, :g.head_dim] += (p.gate * qv[..., None, None]).sum((0, 1))
        a = p.A.abs()
        self.attention(l, a / a.sum(-1, keepdim=True).clamp_min(1e-30), None)

    def attention(self, l: int, a: torch.Tensor, attn_out: torch.Tensor | None) -> None:  # a: [B, H, T, T]
        H = self.geo[l].q_heads  # this layer's heads, the leading block of the grid's
        if self.mode == "rep":
            o = self.first_pos  # <bos> first, for Gemma
            t2 = torch.arange(o + REP_LEN, o + 2 * REP_LEN - 1, device=a.device)  # the second copy, with a successor
            self.extra["induction"][l, :H] += a[:, :, t2, t2 - REP_LEN + 1].float().mean(-1).sum(0).cpu()
            self.extra["duplicate"][l, :H] += a[:, :, t2, t2 - REP_LEN].float().mean(-1).sum(0).cpu()
            return
        if self.mode != "main":
            return
        z = self.stash.pop(l)  # [B, T, H·hd]
        B, T = z.shape[:2]
        hd = self.geo[l].head_dim
        hm = self.ctx["h_mask"]
        zh = z.view(B, T, H, hd).float()
        scale = arch.attn_scale(self.layers[l], attn_out)  # Gemma 4: the post-attention norm's scale
        if scale is not None:
            zh = zh * scale[..., None, None]
        zh_bmm = zh.permute(2, 0, 1, 3).reshape(H, B * T, hd)
        norm = (torch.bmm(zh_bmm, self.gram[l]) * zh_bmm).sum(-1).clamp_min(0).sqrt().view(H, B, T).permute(1, 2, 0)
        nv = norm[hm]  # [P, H]
        self.seg["h_sum"][l, :H] += nv.sum(0)
        self.seg["h_sumsq"][l, :H] += nv.pow(2).sum(0)
        self.count(self.seg["h_count"][l], torch.nn.functional.pad(nv, (0, self.H - H)))  # (the grid's rest: zeros)
        v, t = norm.masked_fill(~hm[..., None], -1).max(1)  # [B, H]
        bi, hi = torch.arange(B, device=a.device)[:, None], torch.arange(H, device=a.device)[None, :]
        rows = a[bi, hi, t].float()  # [B, H, T]: the attention rows there
        rows[..., 0] = 0
        src = rows.argmax(-1)
        o = torch.einsum("bhi,dhi->bhd", zh[bi, t, hi], self.wo[l])
        vals = torch.cat([self.top["h_val"][l, :H], v.T], 1)  # [H, K + B]
        keep = vals.topk(TOP, dim=1).indices
        for k, new in (("h_val", v.T), ("h_doc", self.ctx["doc"][None, :].expand(H, -1)), ("h_pos", t.T.to(torch.int16)),
                       ("h_src", src.T.to(torch.int16))):
            self.top[k][l, :H] = torch.cat([self.top[k][l, :H], new], 1).gather(1, keep)
        cand = torch.cat([self.top["h_out"][l, :H], o.transpose(0, 1).half()], 1)
        self.top["h_out"][l, :H] = cand.gather(1, keep[..., None].expand(-1, -1, self.d))
        qv, kv = self.ctx["qv"], self.ctx["kv"]  # [B, T] float
        af = a.float()
        self.seg["sink"][l, :H] += (af[..., 0] * qv[:, None]).sum((0, 2))
        self.seg["self"][l, :H] += (af.diagonal(dim1=2, dim2=3) * qv[:, None]).sum((0, 2))
        self.seg["prev"][l, :H] += (af.diagonal(-1, dim1=2, dim2=3) * qv[:, None, 1:]).sum((0, 2))
        # Without the sink: divided by the rest's sum as stored, not 1 − a₀, which bf16 rounds to 0 for sink heads.
        rest = af[..., 1:].sum(-1, keepdim=True).clamp_min(1e-30)
        af = af * qv[:, None, :, None] * kv[:, None, None, :] / rest
        recv = (af * torch.arange(T, device=a.device)[None, None, :, None]).sum(2)  # [B, H, T]: per key, times t keys
        self.seg["recv"][l, :H].index_add_(1, self.ctx["ids"].flatten(), recv.transpose(0, 1).reshape(H, -1))
        self.seg["offset"][l, :H, :T].index_add_(1, self.ctx["delta"][:T, :T].flatten(), af.sum(0).reshape(H, -1))

    def hooks(self, model) -> list:
        hs = []
        for l, layer in enumerate(model.layers):
            p = anatomy.parts(layer)
            hs += (self.mlp_hooks(l, layer, p) if l in self.nl else []) + (self.mixer_hooks(l, layer, p) if self.heads else [])
        return hs

    def mlp_hooks(self, l: int, layer, p) -> list:
        """The neurons' activations (the MLP's down projection's input) and the gate's pre-activation; a sparse MLP's
        through ``experts.Tap``."""
        if p.sparse:
            return experts.Tap(layer.mlp, lambda r, l=l: self.sparse_neuron(l, r)).hooks
        gate, down = anatomy.module(layer, p.gate), anatomy.module(layer, p.down)
        return [gate.register_forward_hook(lambda m, x, out, l=l: self.stash.__setitem__(("gate", l), out)),
                down.register_forward_pre_hook(lambda m, x, l=l: self.neuron(l, x[0]))]

    def mixer_hooks(self, l: int, layer, p) -> list:
        """The heads' outputs (the output projection's input), the input norm's scaling, the q/k norms' sums, an output
        gate's factor and the attention pattern (a DeltaNet's: ``delta``)."""
        n1 = anatomy.module(layer, p.n1)
        if n1 is not None:
            hs = [n1.register_forward_pre_hook(lambda m, x, l=l: self.input_norm(l, x[0], arch.norm_eps(m)))]
        else:  # the heads read the residual itself (OLMo 2 and 3): no input norm's scaling
            hs = [layer.register_forward_pre_hook(lambda m, x, kw: self.no_input_norm(x[0] if x else kw["hidden_states"]),
                                                  with_kwargs=True)]
        hs.append(anatomy.module(layer, p.out).register_forward_pre_hook(lambda m, x, l=l: self.stash.__setitem__(l, x[0])))
        if p.deltanet:  # after the output projection has run: the hook sees the mixer's input and the heads' outputs
            hs.append(anatomy.module(layer, p.mixer).register_forward_hook(
                lambda m, x, kw, out, l=l: self.delta(l, m, x[0] if x else kw["hidden_states"]), with_kwargs=True))
            return hs
        at = attention(layer)
        if self.geo[l].gated:
            hs.append(at.q_proj.register_forward_hook(lambda m, x, out, l=l: self.out_gate(l, out)))
        if hasattr(at, "query_key_value"):  # no q/k norms
            hs.append(at.query_key_value.register_forward_hook(lambda m, x, out, l=l: self.query(l, out)))
        else:
            for side, norm in (("q", at.q_norm), ("k", at.k_norm)) + ((("v", at.v_norm),) if hasattr(at, "v_norm") else ()):
                hs.append(norm.register_forward_hook(
                    lambda m, x, out, l=l, side=side: self.qk_norm(l, side, x[0], out, arch.norm_eps(m))))
        hs.append(at.register_forward_hook(lambda m, x, out, l=l: self.attention(l, out[1], out[0])))
        return hs

    def batch(self, ids: torch.Tensor, mask: torch.Tensor, docs: torch.Tensor) -> None:
        """Per-batch context: masks, token ids, and the key-token pair counts for ``recv``."""
        dev, T = self.dev, ids.shape[1]
        ar = torch.arange(T, device=dev)
        real = mask.bool()
        self.ctx.update(ids=ids, doc=docs, n_mask=real & (ar >= NEURON_FROM), h_mask=real & (ar >= HEAD_FROM),
                        qv=(real & (ar >= HEAD_FROM)).float(), kv=(real & (ar >= 1)).float())
        self.seg["n"] += self.ctx["n_mask"].sum().to(torch.int32)
        if not self.heads:
            return
        if "delta" not in self.ctx:
            ad = torch.arange(SEQ, device=dev)
            self.ctx["delta"] = (ad[:, None] - ad[None, :]).clamp_min(0)
        pairs = self.ctx["qv"].flip(1).cumsum(1).flip(1) * self.ctx["kv"]  # queries at or after each key
        self.seg["pairs"].index_add_(0, ids.flatten(), pairs.flatten())
        self.seg["occ"].index_add_(0, ids.flatten(), (pairs > 0).float().flatten())
        self.seg["n_q"] += self.ctx["qv"].sum()
        self.seg["n_k"] += self.ctx["kv"].sum()
        self.seg["h_n"] += self.ctx["h_mask"].sum().to(torch.int32)


def plan_batches(slice_of: list[int], lengths: list[int], batch_docs: int) -> list[tuple[int, list[int]]]:
    """Batches of document indices within each slice, sorted by length to minimize padding waste.
    Returns ``(slice_id, [doc_idx, ...])``; document ids and ``docs.parquet`` order stay unchanged."""
    out = []
    start = 0
    while start < len(slice_of):
        end = start
        s = slice_of[start]
        while end < len(slice_of) and slice_of[end] == s:
            end += 1
        order = sorted(range(start, end), key=lambda idx: lengths[idx], reverse=True)
        for i in range(0, len(order), batch_docs):
            out.append((s, order[i:i + batch_docs]))
        start = end
    return out


class _Stop(Exception):
    """Raised past the last layer a group without the heads reads (``--layers``), to end the pass there."""


def neuron_layers(saved: dict) -> list[int]:
    """The layers whose neurons a saved pass holds (``--layers``; all by default)."""
    lo, hi = saved["meta"].get("layers", (0, saved["state"]["top"]["val"].shape[0] - 1))
    return list(range(lo, hi + 1))


def group_path(lo: int, hi: int, n_layers: int):
    """Where a pass over the neurons of layers ``lo..hi`` saves: the catalog's activations file (``acts_path``) for
    the group from layer 0, which holds the heads; ``<model>-L<lo>-<hi>.pt`` beside it for the others."""
    return ACTS if lo == 0 else ACTS.with_name(f"{ACTS.stem}-L{lo}-{hi}.pt")


def collect(args, docs: list) -> dict:
    device = args.device
    tok = tokenizer()
    L = arch.text_config(MODEL).num_hidden_layers
    lo, hi = (0, L - 1) if args.layers is None else map(int, args.layers.split("-"))
    heads, path = lo == 0, group_path(lo, hi, L)
    if str(device).startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    lm, model = arch.load_model(MODEL, device, eager=heads)  # eager attention only when collecting heads
    rows = [r for _, _, r in docs]
    slice_of = [s for s, _, _ in docs]
    meta = {"model": MODEL, "slices": list(ACT_SLICES), "docs": len(rows), "seq_len": SEQ,
            "neuron_from": NEURON_FROM, "head_from": HEAD_FROM, "edges": EDGES.tolist(), "top_docs": TOP,
            "batch_docs": args.batch_docs, "dtype": str(lm.dtype).removeprefix("torch."),
            "attention": lm.config._attn_implementation}
    if (lo, hi) != (0, L - 1):
        meta["layers"] = [lo, hi]
    col = Collector(model, len(ACT_SLICES), device, list(range(lo, hi + 1)), heads)
    def halt(*_):
        raise _Stop
    stop = [] if heads or hi == L - 1 else [model.layers[hi].register_forward_hook(halt)]

    def run(**kw) -> None:
        try:
            model(**kw)
        except _Stop:
            pass
    lengths_list = [len(r) for r in rows]
    plan = plan_batches(slice_of, lengths_list, args.batch_docs)
    next_batch, stage = 0, "main"
    if path.exists():
        saved = torch.load(path, weights_only=False, map_location="cpu")
        if saved["meta"] != meta:
            raise SystemExit(f"{path} was collected with {saved['meta']}, not {meta}")
        col.load(saved["state"])
        next_batch, stage = saved["next_batch"], saved["stage"]
        if stage == "done":
            return saved
    lengths = torch.tensor(lengths_list)
    flat = torch.cat([torch.tensor(r, dtype=torch.int32) for r in rows])
    start = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])

    def save(stage: str) -> dict:
        col.flush_top()
        out = {"meta": meta, "state": col.state(), "next_batch": next_batch, "stage": stage, "tokens": flat,
               "doc_start": start, "slice_of": torch.tensor(slice_of, dtype=torch.int8),
               "line_of": torch.tensor([ln for _, ln, _ in docs], dtype=torch.int32)}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(out, tmp)
        os.replace(tmp, path)
        return out

    hooks = col.hooks(model)
    saved_at, t0, done = time.monotonic(), time.perf_counter(), 0
    for bi in range(next_batch, len(plan)):
        sl, idxs = plan[bi]
        T = max(lengths_list[idx] for idx in idxs)
        ids = torch.zeros(len(idxs), T, dtype=torch.long)
        mask = torch.zeros(len(idxs), T, dtype=torch.long)
        for j, idx in enumerate(idxs):
            r = rows[idx]
            ids[j, :len(r)] = torch.tensor(r)
            mask[j, :len(r)] = 1
        ids, mask = ids.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        col.batch(ids, mask, torch.tensor(idxs, dtype=torch.int32, device=device))
        with torch.inference_mode():
            run(input_ids=ids, attention_mask=mask)
        done += sum(lengths_list[idx] for idx in idxs)
        next_batch = bi + 1
        last = next_batch == len(plan)
        if last or bi % FLUSH == FLUSH - 1 or plan[min(next_batch, len(plan) - 1)][0] != sl:
            col.flush(sl)
        if last or time.monotonic() - saved_at > 60 * args.save_minutes:
            col.flush(sl)
            save("main")
            saved_at = time.monotonic()
        if bi % 20 == 0 or last:
            print(f"{time.strftime('%H:%M:%S')} batch {bi + 1}/{len(plan)} ({ACT_SLICES[sl]}), "
                  f"{done / (time.perf_counter() - t0):.0f} tokens/s", flush=True)
    col.flush_top()

    # Every kept token alone: the first position.
    col.mode = "first"
    vocab_ids = kept()
    pre = arch.bos(tok)  # Gemma's documents open with <bos>: the token comes after it
    col.first_pos = len(pre)
    with_pre = lambda x: torch.cat([torch.tensor(pre, dtype=x.dtype, device=x.device).expand(len(x), -1), x], 1)  # noqa: E731
    # Keep first_val / first_tok on GPU as [nl, P, f, FIRST_K] so topk runs on contiguous dim=1 with zero CPU syncs.
    col._first_val = col.extra["first_val"].transpose(-2, -1).contiguous().to(device)
    col._first_tok = col.extra["first_tok"].transpose(-2, -1).contiguous().to(device)
    with torch.inference_mode():
        col.ctx.update(chat=True)
        run(input_ids=with_pre(torch.tensor([[arch.chat_open(MODEL, tok)]], device=device)))
        col.ctx["chat"] = False
        for s in range(0, len(vocab_ids), 1024):
            col.ctx["ids"] = vocab_ids[s:s + 1024].to(device)
            run(input_ids=with_pre(col.ctx["ids"][:, None]))
        col.extra["first_val"] = col._first_val.transpose(-2, -1).contiguous().cpu()
        col.extra["first_tok"] = col._first_tok.transpose(-2, -1).contiguous().cpu()
        del col._first_val, col._first_tok
        if heads:  # Repeated random tokens, for induction and duplicate-token scores.
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
    for h in hooks + stop:
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


def neuron_modes(saved: dict, layer: int) -> dict:
    """``[P·f]`` (+ poles, then −): the share of a pole's top firings (positive activation) on its most common
    peak token (``cur_mode``), and followed by its most common next token (``next_mode``)."""
    top = saved["state"]["top"]
    P = top["val"].shape[1]
    val = torch.cat([top["val"][layer, p].T for p in range(P)]).float()  # [P·f, K]
    doc, pos = (torch.cat([top[k][layer, p].T for p in range(P)]).long() for k in ("doc", "pos"))
    fired = val > 0
    flat, start = saved["tokens"], saved["doc_start"]
    cur, nxt = firing_tokens(flat, start, doc, pos), firing_tokens(flat, start, doc, pos, 1)
    return {"cur_mode": modal_share(cur, fired), "next_mode": modal_share(nxt, fired & (nxt >= 0))}


def head_modes(saved: dict) -> dict:
    """``[L·H]``: the share of a head's top firings on its most common query token (``cur_mode``), and
    attending most to its most common token (``src_mode``)."""
    top = saved["state"]["top"]
    L, H, K = top["h_val"].shape
    doc, pos, src = top["h_doc"].long(), top["h_pos"].long(), top["h_src"].long()
    cur, att = (firing_tokens(saved["tokens"], saved["doc_start"], doc, p) for p in (pos, src))
    return {"cur_mode": modal_share(cur, torch.ones_like(cur, dtype=torch.bool)).flatten(),
            "src_mode": modal_share(att, att >= 0).flatten()}


def neuron_tables(saved: dict, workers: int = 1) -> None:
    """One ``neurons/acts`` table per layer of the pass (``neuron_layers``); with ``workers`` > 1, layers in forked
    processes (the per-pole contexts are Python string work, ~45 s a layer for Gemma 4's 30,720 poles)."""
    L = len(neuron_layers(saved))
    _SAVED["s"] = saved
    if workers <= 1:
        for i in range(L):
            _neuron_layer(i)
        return
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(workers, L)) as pool:
        for _ in pool.imap_unordered(_neuron_layer, range(L)):
            pass


def _neuron_layer(li: int) -> None:
    """The ``neurons/acts`` table of the pass's ``li``-th layer. A pole's list of top documents shorter than ``TOP`` (a
    routed neuron's, whose expert few documents reach) is padded: ``top_doc``, ``top_pos`` −1, ``top_val``, ``top_gate``
    NaN; so is ``first_max``, NaN where no token alone routes to it."""
    torch.set_num_threads(1)
    saved = _SAVED["s"]
    layer = neuron_layers(saved)[li]
    tok = tokenizer()
    dec = functools.cache(lambda i: tok.decode([i]))
    st, top, extra = saved["state"]["total"], saved["state"]["top"], saved["state"]["extra"]
    flat, start = saved["tokens"], saved["doc_start"]
    n_s = st["n"].double()
    n = n_s.sum()
    grid = torch.cat([-EDGES[1:].flip(0), EDGES]).double()
    f = top["val"].shape[-1]
    t0 = time.perf_counter()
    E_s = past(st["count"][:, li].double())  # [S, 2, len(EDGES), f]
    E = E_s.sum(0)
    mean = st["sum"][:, li].sum(0) / n
    std = (st["sumsq"][:, li].sum(0) / n - mean.pow(2)).clamp_min(0).sqrt()
    cols = neuron_keys(layer, f)
    acc = {k: [] for k in ("act_top", "act_max", "pos_share", "rate", *[f"rate_{s}" for s in ACT_SLICES],
                           "q50", "q90", "q99", "q999", "mean", "std", "neg_lobe", "peak_tokens", "next_tokens", "contexts")}
    P = top["val"].shape[1]  # the poles (arch.n_poles); count's sign axis stays two-wide (a pole's quantiles need both)
    for p in range(P):
        val = top["val"][li, p]  # [K, f]
        act_top = val[:TYPICAL].median(0).values.clamp_min(0).double()
        acc["act_top"].append(act_top)
        acc["act_max"].append(val[0].double().nan_to_num(nan=float("nan"), neginf=float("nan")))
        acc["pos_share"].append(E[p, 0] / n)
        acc["rate"].append(interp(E[p], n, act_top / 4))
        for si, s in enumerate(ACT_SLICES):
            acc[f"rate_{s}"].append(interp(E_s[si, p], n_s[si].clamp_min(1), act_top / 4))
        S = torch.cat([1 - E[1 - p, 1:].flip(0) / n, E[p] / n])  # P(s·act > t) on the grid
        for k, q in zip(("q50", "q90", "q99", "q999"), quantiles(grid, S)):
            acc[k].append(q)
        acc["mean"].append(mean)
        acc["std"].append(std)
        docs, poss = top["doc"][li, p].long(), top["pos"][li, p].long()
        peak = flat[start[docs] + poss]  # [K, f]
        after = start[docs] + poss + 1  # the token after each firing; past its document's end, none
        following = torch.where(after < start[docs + 1], flat[after.clamp_max(len(flat) - 1)].long(), -1)
        size = val.clamp_min(0)  # activation-weighted, so a pole's weak tail of top firings doesn't decide its side
        acc["neg_lobe"].append((size * (top["gate"][li, p] < 0)).sum(0).double() / size.sum(0).clamp_min(1e-9))
        for nn in range(f):
            fired = (val[:, nn] > 0).nonzero().squeeze(1).tolist()
            c = Counter(int(peak[j, nn]) for j in fired)
            acc["peak_tokens"].append(" | ".join(f"{dec(t)!r}×{k}" for t, k in c.most_common(5)))
            c = Counter(int(following[j, nn]) for j in fired if following[j, nn] >= 0)
            acc["next_tokens"].append(" | ".join(f"{dec(t)!r}×{k}" for t, k in c.most_common(5)))
            acc["contexts"].append(" ‖ ".join(
                context(tok, flat[start[docs[j, nn]]:start[docs[j, nn] + 1]], int(poss[j, nn])) for j in fired[:N_CONTEXTS]))
    for k, v in acc.items():
        cols[k] = pa.array(v) if isinstance(v[0], str) else pa.array(torch.cat(v).float().numpy())
    none = torch.cat([top["val"][li, p].T for p in range(P)]) == -torch.inf  # a shorter list's padding
    for k in ("doc", "pos", "val", "gate"):
        x = torch.cat([top[k][li, p].T for p in range(P)])
        cols[f"top_{k}"] = fixed(x.masked_fill(none, -1 if k in ("doc", "pos") else float("nan")))
    fv, ft = extra["first_val"][li], extra["first_tok"][li]  # [P, K, f]
    cols["first_max"] = pa.array(fv[:, 0].flatten().masked_fill(fv[:, 0].flatten() == -torch.inf, float("nan")).numpy())
    cols["first_tokens"] = pa.array([" | ".join(f"{dec(int(ft[p, j, nn]))!r} {fv[p, j, nn]:.3g}" for j in range(FIRST_K)
                                                if fv[p, j, nn] > -torch.inf)
                                     for p in range(P) for nn in range(f)])
    cols["first_chat"] = pa.array(torch.cat([s * extra["first_chat"][li] for s in arch.signs(MODEL)]).numpy())
    cols |= {k: pa.array(v.float().numpy()) for k, v in neuron_modes(saved, li).items()}
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
    out_top = top["h_val"][..., :TYPICAL].median(-1).values.double()  # [L, H]
    cols = head_keys()  # over the grid of heads; write_heads keeps those that exist
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
    # The firings' outputs through the R-lens from where heads write (the MLP input; a parallel block's output),
    # each token's logit z-scored across the layer's firings, averaged over the head's.
    keep = kept()
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted({nm["unembed"], nm["norm"]}))
    unembed = head[nm["unembed"]][keep].to(device, torch.float16)
    gamma = arch.final_scale(MODEL, device)
    J = arch.head_write_lens(MODEL, "R", R_FIT)
    ids, vals = torch.full((L, H, TOP), -1), torch.full((L, H, TOP), float("nan"))
    for l, g in enumerate(arch.layout(MODEL)):
        O = top["h_out"][l, :g.q_heads].reshape(g.q_heads * K, -1).to(device, torch.float32).T  # [d, H·K]
        lg = arch.cap(MODEL, unembed @ rms_norm_columns(J[l].to(device) @ O, gamma, centre=arch.centred(MODEL)).half()).float()
        z = (lg - lg.mean(1, keepdim=True)) / lg.std(1, keepdim=True).clamp_min(1e-6)
        mz = z.view(-1, g.q_heads, K).mean(-1).T  # [H, V]
        v, i = mz.topk(TOP, dim=1)
        ids[l, :g.q_heads], vals[l, :g.q_heads] = keep[i.cpu()], v.cpu()
    put(cols, "out_R_z", ids.flatten(0, 1), vals.flatten(0, 1))
    cols |= {k: pa.array(v.float().numpy()) for k, v in head_modes(saved).items()}
    write_heads(cols, head_table("acts"))
    print(f"{time.strftime('%H:%M:%S')} heads/acts written", flush=True)


def expert_tables(saved: dict) -> None:
    """``experts/acts.parquet``, a row per (layer, expert) of a sparse model, this pass's layers in place of the
    file's rows for them (a model collected in groups, ``--layers``, fills it a group at a time): ``rate`` (the share of
    positions routed to it, and ``rate_<slice>``), ``weight`` (its mean routing weight where routed), its top ``TOP``
    documents by routing weight (``top_doc``, ``top_pos``, ``top_val``; padded where fewer route to it), ``peak_tokens``
    and ``contexts`` there, and ``mean_cg``, ``mean_cu`` (``[I]`` each, its neurons' RelP factors times the routing
    weight, ``experts.factors``, averaged over its routed positions: the expert's mean map, ``08_reads.py``)."""
    tok = tokenizer()
    dec = functools.cache(lambda i: tok.decode([i]))
    st, top = saved["state"]["total"], saved["state"]["top"]
    flat, start = saved["tokens"], saved["doc_start"]
    layers = neuron_layers(saved)
    nl, E, K = top["e_val"].shape
    I = saved["state"]["extra"]["expert_width"]
    n_s = st["n"].double()
    routed = st["e_n"].double()  # [S, nl, E]
    count = routed.sum(0)
    cols = expert_keys(layers, E)
    cols["rate"] = pa.array((count / n_s.sum()).flatten().float().numpy())
    for si, s in enumerate(ACT_SLICES):
        cols[f"rate_{s}"] = pa.array((routed[si] / n_s[si].clamp_min(1)).flatten().float().numpy())
    cols["weight"] = pa.array((st["e_w"] / count.clamp_min(1)).flatten().float().numpy())
    val, docs, pos = top["e_val"].reshape(nl * E, K), top["e_doc"].reshape(nl * E, K).long(), top["e_pos"].reshape(nl * E, K).long()
    none = val == -torch.inf
    cols["top_doc"], cols["top_pos"] = fixed(docs.masked_fill(none, -1).int()), fixed(pos.masked_fill(none, -1).short())
    cols["top_val"] = fixed(val.masked_fill(none, float("nan")))
    peak = flat[start[docs] + pos]
    cols["peak_tokens"] = pa.array([" | ".join(f"{dec(int(t))!r}×{c}" for t, c in Counter(peak[r][~none[r]].tolist()).most_common(5))
                                    for r in range(nl * E)])
    cols["contexts"] = pa.array([" ‖ ".join(context(tok, flat[start[docs[r, j]]:start[docs[r, j] + 1]], int(pos[r, j]))
                                            for j in range(N_CONTEXTS) if not none[r, j]) for r in range(nl * E)])
    for k, name in (("e_cg", "mean_cg"), ("e_cu", "mean_cu")):  # the experts' neurons come first, a shared expert's after
        m = st[k][:, :E * I].view(nl, E, I) / count[..., None].clamp_min(1)
        cols[name] = fixed(m.reshape(nl * E, I).float())
    path = expert_table("acts")
    table = pa.table(cols)
    if path.exists():
        old = pq.read_table(path)
        table = pa.concat_tables([old.filter(pc.invert(pc.is_in(old.column("layer"), pa.array(layers, pa.int16())))), table])
        table = table.sort_by([("layer", "ascending"), ("expert", "ascending")])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"{time.strftime('%H:%M:%S')} experts/acts L{layers[0]}-L{layers[-1]} written", flush=True)


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
    ap.add_argument("--layers", default=None,
                    help="a-b: the neurons of these layers alone (default all); the group from layer 0 also collects the heads")
    args = ap.parse_args()
    if args.detach:
        LOGS.mkdir(exist_ok=True)
        log = LOGS / f"activations_{MODEL.split('/')[-1].lower()}.log"
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
    if "e_val" in saved["state"]["top"]:
        expert_tables(saved)
    if neuron_layers(saved)[0] == 0:
        head_tables(saved, args.device)
    write({"doc": pa.array(range(len(docs)), pa.int32()), "slice": pa.array([ACT_SLICES[s] for s, _, _ in docs]),
               "line": pa.array([ln for _, ln, _ in docs], pa.int32()),
               "tokens": pa.array([len(r) for _, _, r in docs], pa.int16())}, catalog_dir() / "docs.parquet")


if __name__ == "__main__":
    main()
