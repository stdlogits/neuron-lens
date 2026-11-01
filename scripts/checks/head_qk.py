"""Which input lens reads what a head attends to (findings §11): ``08_reads.py``'s weight-only QK read
against the attention key tokens receive in context, with the S-lens footprint as the key's input
beside raw embeddings and the R-lens.

``08_reads.py``'s ``qk_S`` scores each key token ``v`` under a head's mean query: ``q̄ᵀ R̄ K_v`` with
``K_v = A_k W_K γ₁ X_v / ‖X_v‖``, ``A_k`` the true derivative of the input norm's scaling and
``k_norm`` averaged over positions, ``R̄`` RoPE's rotation averaged over the head's own attention
offsets and ``q̄`` the mean query after ``q_norm`` (``06_activations.py``'s statistics), z-scored over
the vocabulary. Here the same map with three inputs ``X_v``:

- **S:** the S-lens footprint at layer ``L − 1``'s output (the centred embedding at L0): ``qk_S``
  itself, recomputed and checked against ``data/heads/<model>-token-z.pt``;
- **raw:** the centred embedding at every layer;
- **R:** the R-lens direction at ``L − 1``'s output that promotes ``v`` (the position says ``v``
  rather than holds it), from L1.

The attention a key token receives: from queries at 16+, renormalized without the sink, times the
number of keys past the sink (1 is uniform), averaged over the token's occurrences as a key, for
tokens seen as keys 10+ times. Targets:

- **key (all):** ``06_activations.py``'s statistics over all 63,990 documents, by the key's token.
  For S this is ``08_reads.py``'s ``qk_ctx_rho``, the catalog's measure;
- over a pass of every 16th document (4,000; eager attention, bf16): **key** (the same measure,
  with fewer occurrences behind each token), **key, far** (from queries two or more positions after
  the key) and **next, far** (that attention by the token after the key, the R input's target; the
  gap keeps the query itself out of the target). The pass renormalizes each attention row by its
  non-sink weights' sum as stored, as ``06_activations.py`` now does; the released pass divided by
  ``1 − a₀``, which bf16 rounds to 0 where the sink takes nearly everything, inflating those rows (a
  mean row sum above 1.05 for 433 heads, mostly sink heads; on a 176-document slice it moved band
  means by up to 0.03, at L28-L35).

Per head, Spearman over tokens between the read and the attention received, against another head of
the layer's read (the control, one fixed draw per head, as ``08_reads.py``). The band means depend
on which tokens pass the floor: with all documents, 66,350 tokens are seen 10+ times, most of them
rare and measured on few occurrences, and own and control means are lower than on the pass's 11,369
(S at L13-L20: +0.01 against +0.10 on the same full-corpus attention). Compare inputs and targets
within one document set.

Also the query reads findings §11 quotes: the keys a query token favours, ``Q_vᵀ R̄ K_u`` over
``u`` (``Q_v = A_q W_Q γ₁ X_v / ‖X_v‖``, S input). Needs the activations file (the QK statistics
and documents); the pass is cached in ``data/heads/<model>-qk-context.pt``. Qwen3-8B only. Summary
in ``data/summary/head_qk_<model>.{log,json}``.

    python scripts/checks/head_qk.py
"""
import json
import time

import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import MODEL, acts_path, doc_tokens, head_table, kept, vocab
from neuron_lens.heads import norm_derivative, rope_mean
from neuron_lens.paths import DATA, SUMMARY
from neuron_lens.stats import centered_ranks
from neuron_lens.transport import SKIP_FIRST
from neuron_lens.weights import tensors

STEM = arch.stem(MODEL)
TOKEN_Z = DATA / "heads" / f"{STEM}-token-z.pt"
CACHE = DATA / "heads" / f"{STEM}-qk-context.pt"
EVERY = 16
BATCH = 16
MIN_OCC = 10  # key occurrences for a token's in-context attention, as 08_reads.py
TARGETS = ("key", "far", "next")
TOP = 8
EXAMPLES = ((1, 28, " Paris"), (20, 17, " she"), (20, 17, " he"))  # (layer, head, query token), findings §11


def collect(seqs: list[torch.Tensor], dev) -> dict:
    """The attention key tokens receive over ``seqs``, summed per target (``recv_<target>``
    ``[L, H, vocabulary]``), with the query-key pairs and key occurrences behind each."""
    lm, stack = arch.load_model(MODEL, dev)
    cfg = lm.config
    L, H, V = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.vocab_size
    st = {f"recv_{t}": torch.zeros(L, H, V, device=dev) for t in TARGETS}
    st |= {f"{k}_{t}": torch.zeros(V, device=dev) for k in ("pairs", "occ") for t in TARGETS}
    T_max = max(len(q) for q in seqs)
    ar = torch.arange(T_max, device=dev)
    delta = ar[:, None] - ar[None, :]  # query − key
    ctx = {}

    def attn(l: int):
        def hook(m, args, out):
            a = out[1].float()  # [B, H, T, T]
            T = a.shape[-1]
            # Without the sink: divided by the rest's sum as stored, not 1 − a₀, which bf16 rounds to 0 for sink heads.
            a = a * ctx["qv"][:, None, :, None] * ctx["kv"][:, None, None, :] / a[..., 1:].sum(-1, keepdim=True).clamp_min(1e-30)
            a = a * ar[:T][None, None, :, None]  # times the t keys past the sink: 1 is uniform
            far = (a * (delta[:T, :T] >= 2)).sum(2)  # [B, H, T]: per key position
            for t, r in (("key", a.sum(2)), ("far", far), ("next", far * ctx["has_next"][:, None])):
                st[f"recv_{t}"][l].index_add_(1, ctx[t].flatten(), r.transpose(0, 1).reshape(H, -1))
        return hook
    hooks = [layer.self_attn.register_forward_hook(attn(l)) for l, layer in enumerate(stack.layers)]
    t0 = time.perf_counter()
    for s in range(0, len(seqs), BATCH):
        part = seqs[s:s + BATCH]
        Tm = max(len(q) for q in part)
        ids = torch.zeros(len(part), Tm, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for r, q in enumerate(part):
            ids[r, :len(q)], mask[r, :len(q)] = torch.as_tensor(q).long(), 1
        ids, mask = ids.to(dev), mask.to(dev)
        real = mask.bool()
        qv, kv = (real & (ar[:Tm] >= SKIP_FIRST)).float(), (real & (ar[:Tm] >= 1)).float()
        nxt = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, :1])], 1)  # the token after each key
        has_next = torch.cat([real[:, 1:].float(), torch.zeros_like(qv[:, :1])], 1) * kv
        later = qv.flip(1).cumsum(1).flip(1)  # queries at or after each position
        far = torch.cat([later[:, 2:], torch.zeros_like(later[:, :2])], 1) * kv  # two or more after
        ctx.update(qv=qv, kv=kv, has_next=has_next, key=ids, far=ids, next=nxt)
        for t, p in (("key", later * kv), ("far", far), ("next", far * has_next)):
            st[f"pairs_{t}"].index_add_(0, ctx[t].flatten(), p.flatten())
            st[f"occ_{t}"].index_add_(0, ctx[t].flatten(), (p > 0).float().flatten())
        with torch.inference_mode():
            stack(input_ids=ids, attention_mask=mask)
        if s // BATCH % 50 == 0:
            print(f"{time.strftime('%H:%M:%S')} {s + len(part)}/{len(seqs)} documents, {time.perf_counter() - t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()
    return {k: v.cpu() for k, v in st.items()}


def main() -> None:
    if MODEL != arch.QWEN:
        raise SystemExit(f"head_qk.py reads Qwen3-8B's heads only, not {MODEL}")
    dev = "mps" if torch.backends.mps.is_available() else "cuda"
    saved = torch.load(acts_path(), weights_only=False, map_location="cpu", mmap=True)
    tot, extra = saved["state"]["total"], saved["state"]["extra"]
    docs = list(range(0, saved["meta"]["docs"], EVERY))
    cached = torch.load(CACHE, weights_only=False) if CACHE.exists() else None
    if cached is None or cached["docs"] != docs:
        seqs = doc_tokens(saved)
        cached = collect([seqs[i] for i in docs], dev) | {"docs": docs}
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cached, CACHE)

    geo = arch.layout(MODEL)
    L, H = len(geo), geo[0].q_heads
    keep = kept()
    V = len(keep)
    v_str = [vocab()[i] for i in keep.tolist()]
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted(set(nm.values())))
    E = head[nm["embed"]][keep].float()
    E = (E - E.mean(0)).to(dev)
    Wu = (head[nm["unembed"]][keep].float() * head[nm["norm"]].float()).to(dev)  # W_U diag(γ_f)
    del head
    R_FIT, S_FIT = arch.fits(MODEL)
    K_out, J_out = arch.lens(MODEL, "S_out", S_FIT), arch.lens(MODEL, "R_out", R_FIT)
    variants = ("S", "raw", "R")
    att = {v: torch.full((L, H, V), torch.nan, dtype=torch.float16) for v in variants}  # attends-to z
    queries = {}
    for layer in range(L):
        lay = geo[layer]
        kvh, hd = lay.kv_heads, lay.head_dim
        qk = arch.qk(MODEL, lay, dev)  # Wq, Wk times γ₁
        Aq = norm_derivative(qk["q_norm"], tot["q_r"][layer].float().to(dev), tot["q_m"][layer, :, :hd, :hd].float().to(dev),
                             float(tot["n_q"]))
        Ak = norm_derivative(qk["k_norm"], tot["k_r"][layer, :kvh].float().to(dev),
                             tot["k_m"][layer, :kvh, :hd, :hd].float().to(dev), float(tot["n_k"]))
        off = tot["offset"][layer].float().to(dev)
        Rbar = rope_mean(off / off.sum(-1, keepdim=True), arch.layer_inv_freq(extra["inv_freq"], MODEL, layer).to(dev))
        qbar = (tot["q"][layer, :, :hd] / tot["n_q"]).float().to(dev)
        inputs = {"S": E @ K_out[layer - 1].to(dev).T if layer else E, "raw": E}
        if layer:
            inputs["R"] = Wu @ J_out[layer - 1].to(dev)
        for v, X in inputs.items():
            Xu = X / X.norm(dim=1, keepdim=True)  # directions: a footprint's size would rank keys across tokens
            Kv = torch.einsum("gij,vgj->vgi", Ak, (Xu @ qk["Wk"].T).view(V, kvh, hd)).repeat_interleave(H // kvh, dim=1)
            Kr = torch.einsum("hij,vhj->vhi", Rbar, Kv)  # [V, H, hd]: R̄ K
            del Kv
            a = torch.einsum("vhi,hi->vh", Kr, qbar)
            att[v][layer] = ((a - a.mean(0)) / a.std(0)).T.half().cpu()
            for l, h, t in EXAMPLES:
                if v == "S" and l == layer:
                    i = v_str.index(t)
                    q = Aq[h] @ (Xu[i] @ qk["Wq"].T).view(H, hd)[h]
                    row = Kr[:, h] @ q
                    queries[f"L{l}h{h} {t!r}"] = [[v_str[j] for j in x.topk(TOP).indices.tolist()] for x in (row, -row)]
            del Kr, X, Xu
        print(f"{time.strftime('%H:%M:%S')} L{layer}", flush=True)
    diff = (att["S"].float() - torch.load(TOKEN_Z, weights_only=False, mmap=True)["qk_S"].float()).abs().max().item()

    other = (torch.arange(H)[None, :] + torch.randint(1, H, (L, H), generator=torch.Generator().manual_seed(0))) % H
    hh = torch.arange(H, device=dev)
    nrm = lambda x: x / x.norm(dim=-1, keepdim=True)
    targets = {"key (all)": (tot["recv"], tot["pairs"], tot["occ"])}  # the catalog's documents; the rest the pass's
    targets |= {name: tuple(cached[f"{k}_{t}"] for k in ("recv", "pairs", "occ"))
                for name, t in (("key", "key"), ("key, far", "far"), ("next, far", "next"))}
    rho, n_seen = {}, {}
    for name, (recv, pairs, occ) in targets.items():
        seen = occ[keep] >= MIN_OCC
        r = recv[:, :, keep][:, :, seen].float() / pairs[keep][seen].float()  # [L, H, seen]
        n_seen[name] = int(seen.sum())
        for v in variants:
            own, ctl = torch.full((L, H), torch.nan), torch.full((L, H), torch.nan)
            for l in range(1 if v == "R" else 0, L):
                c = nrm(centered_ranks(att[v][l][:, seen].to(dev).float())) @ nrm(centered_ranks(r[l].to(dev))).T  # [H, H']
                own[l], ctl[l] = c.diagonal().cpu(), c[other[l].to(dev), hh].cpu()
            rho[v, name] = (own, ctl)

    lines = [(f"attention heads' QK reads against the attention key tokens receive in context, {MODEL}: {L} × {H} heads, "
             f"{V} kept tokens. Inputs: S (S-lens footprint at L−1's output; 08_reads.py's qk_S, recomputed: max |Δz| "
             f"{diff:.3f}), raw (embeddings), R (the R-lens direction at L−1's output that promotes the token; from L1). "
             f"Targets: key (all) over all {saved['meta']['docs']:,} documents (06_activations.py; {n_seen['key (all)']:,} "
             f"tokens seen {MIN_OCC}+ times as keys); over every {EVERY}th document ({len(docs):,}): key "
             f"({n_seen['key']:,} tokens), key, far ({n_seen['key, far']:,}: from queries two or more after the key) and "
             f"next, far ({n_seen['next, far']:,}: the same attention by the token after the key)"), "",
             ("Spearman over tokens between the read and the attention received, mean over heads (own / control head), and "
             "the share of heads past +0.2 (own / control):")]
    bands = arch.bands(L) + [(0, L - 1)]
    table = {"context": {}, "layers": {}, "queries": queries, "tokens": n_seen, "qk_S_max_diff": diff}
    for name in targets:
        for v in variants:
            own, ctl = rho[v, name]
            row = {}
            for lo, hi in bands:
                o, c = own[lo:hi + 1].flatten(), ctl[lo:hi + 1].flatten()
                o, c = o[~o.isnan()], c[~c.isnan()]
                row[f"L{lo}-{hi}"] = [o.mean().item(), c.mean().item(), (o > 0.2).float().mean().item(), (c > 0.2).float().mean().item()]
            table["context"][f"{v} vs {name}"] = row
            lines.append(f"  {v:3s} vs {name:9s} " + "  ".join(f"{b} {x[0]:+.2f}/{x[1]:+.2f} ({x[2]:.2f}/{x[3]:.2f})"
                                                                      for b, x in row.items()))
    own = rho["S", "key (all)"][0]
    table["layers"] = own.nanmean(-1).tolist()
    lines += ["", "S vs key (all), by layer: " + " ".join(f"L{l} {x:+.2f}" for l, x in enumerate(table["layers"]))]
    reads = pq.read_table(head_table("reads"), columns=["qk_S_str"]).column("qk_S_str").to_pylist()
    lines += ["", "heads best read by S vs key (all), by band, with heads/reads' qk_S:"]
    for lo, hi in arch.bands(L):
        s = own[lo:hi + 1].nan_to_num(-9).flatten()
        for i in s.topk(3).indices.tolist():
            lines.append(f"  L{lo + i // H}h{i % H} {s[i]:+.2f}: {reads[(lo + i // H) * H + i % H]}")
    lines += ["", "query reads (S input): the keys a query token favours ; disfavours"]
    lines += [f"  {k} → {' | '.join(f)} ; {' | '.join(d)}" for k, (f, d) in queries.items()]
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"head_qk_{STEM}.log").write_text(text + "\n")
    (SUMMARY / f"head_qk_{STEM}.json").write_text(json.dumps(table))


if __name__ == "__main__":
    main()
