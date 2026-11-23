"""Weight-only reads of every neuron pole and attention head through the lenses (pipeline stage 8).

The lens rule: ``out`` reads MLP outputs and attention inputs, ``mid`` reads MLP inputs and
attention outputs. Every read lists the top 64 kept tokens (``catalog.kept``), with a ``_z`` variant
that z-scores each token's score across the layer's poles or heads, which removes the hub tokens that
top most raw reads (findings §7).

A lens takes vectors or covectors as it was built: the R-lens transports a vector, the S-lens pulls back a
covector. A read row (what a unit applies to the residual) is a covector, a write a vector, so a read that feeds
a read row to the R-lens goes through the residual's metric (``neuron_lens.metric``: ``v = s² ⊙ c``,
``s² = σ / |γ|``), which keeps the read free of the units each residual coordinate carries; one that sets a vector
against the S-lens's footprints goes through the lens's polar factor (``metric.polar``: ``w · U ē``, the footprints
with the lens's stretch removed), which asks which input token would put it there (findings §15).

**Neurons** (``neurons/reads/L<layer>.parquet``), no forward pass:

- ``write_R``: the write ``±W_down[:, n]`` through the R-lens, ``softmax(W_U · norm(J_L w))``
  (``_val``: probabilities; the logit lens at the last layer);
- ``write_S``: the write against the S-lens's footprints at the layer's output, tokens ``v`` by
  ``±w · U (e_v − ē)``, ``K^out_L = (K Kᵀ)^½ U`` (``metric.polar``): the input tokens whose footprint the write
  imitates (legible to about L8 in Qwen3-8B, where early neurons write back the tokens they detect; §7, §8, §15);
- ``write_orth_R``, ``write_orth_S``: the same of the write minus its parts along the neuron's gate and up
  patterns, so that neither row reads it (``neurons.orthogonal_write``), what it adds beyond anything it reads (§6);
- ``read_R``: the read covector ``c = â ± b̂`` (gate and up rows times the MLP input norm's weight, each at
  unit length in the metric; the up row alone, ``∓b̂``, for the poles on SiLU's negative lobe, ``acts``'
  ``neg_lobe`` past a half, §7) as a vector, ``s² ⊙ c`` (the raw rows times the residual's scale), through the
  R-lens from the layer's MLP input (``J^mid_L``): what the residual says when it drives the pole (§15);
- ``read_S``: the read covector through the S-lens to the MLP input, ``c · K^mid_L (e_v − ē)``:
  input tokens that drive the pole (legible at L0-L8);
- weights: ``w_norm`` (``‖W_down[:, n]‖``), ``cos_gate_up``, ``cos_write_gate``, ``cos_write_up`` (cosines in the metric
  at the MLP input; the latter two of the pole's write with the gate and up sides of what it reads: positive where it
  writes toward them; §6), ``null_share`` (the share of the write's squared length in the bottom 1% of right-singular
  directions of ``W_U diag(γ)``, Euclidean as the final norm's rms takes it, where entropy neurons write; Stolfo et al.
  2024), and ``vocab_skew``, ``vocab_kurt`` of ``W_U · norm(w)`` over
  the vocabulary (Gurnee et al. 2024's prediction and suppression neurons).

**Heads** (``heads/reads.parquet``; findings §10): a head reads token ``v`` as its S-lens
footprint at layer ``L − 1``'s output (the centred embedding at L0) and writes into layer ``L``'s
MLP input (``neuron_lens.heads``).

- ``ov_R``: the OV map read through the R-lens: each token's z in its own column, where ``v`` ranks
  in what the head makes the model say when it reads ``v``; the tokens it copies most, and
  ``ov_R_neg`` those it suppresses most. ``ov_S``: the same against the S-lens's footprints at the MLP input
  through its polar factor, as ``write_S``: whether the write makes the residual look as if ``v`` were there (the
  record through L8; §10, §15).
  ``ov_R_copy``, ``ov_S_copy``: Elhage et al.'s copying score Σλ/Σ|λ| of each map (§10);
- ``qk_S``: the QK map, ``q̃ᵀ R̄ k̃`` with the q/k norms' true derivative averaged over positions (without q/k
  norms, GPT-NeoX: the input norm's, which the scores read as the lenses do) and RoPE averaged over the head's own attention offsets (``06_activations.py``'s statistics),
  footprints at unit length: each key token's z over the vocabulary under the head's mean query,
  what it attends to (``qk_S_neg``: avoids); ``qk_self_S``: each token's z in its own row, where it
  ranks among the keys it favours as a query whose position holds it (its S-lens footprint as the query too),
  and the scalar ``qk_self_S`` the median of that z over tokens, which finds duplicate-token heads (§10);
  ``qk_self_R``: the same with the query the R-lens direction at ``L − 1``'s output that promotes the token
  (the covector ``W_U diag(γ_f) J^out_{L−1}`` as a vector in the metric, the position is about to say it), which finds the late heads that look up
  earlier copies of what is about to be said, with a negative ``ov_R_copy`` the copy suppressors (§10; from L1,
  empty at L0);
- ``qk_ctx_rho``: Spearman over tokens between ``qk_S`` and the attention each key token receives
  in context (tokens seen as keys 10+ times); ``qk_ov_rho``: between ``qk_S`` and ``ov_R``'s z,
  whether a head attends to what it copies.

- ``beta_S``, ``decay_S`` (a Gated DeltaNet's heads; empty for softmax heads): the head's write-strength row (``β =
  σ(W_b x̂)``: the tokens whose footprint makes the head write its value into its memory harder) and decay row (the
  tokens that make its memory forget faster; ``_neg``: slower), footprints at unit length as for ``qk_S``, each token's
  z over the vocabulary.

**Gated DeltaNet heads** (Qwen3.5, Olmo Hybrid): a head is a value head, mixing through ``A`` (``o = A v``). Its OV
map runs through the value rows times the four-tap convolution summed over its taps (a token held over the window),
SiLU's ratio ``σ(c)`` and the output's gate (the output norm's scale and ``silu(z)``) averaged over positions
(``06_activations.py``), then its block of the output projection; its QK map, ``q̂ · k̂``, through its key head's query
and key rows the same way, their L2 norms' true derivative averaged over positions, no RoPE. ``A`` also carries the
decay and the delta rule's corrections, which a token map leaves out. Gated attention's output gate ``σ(gate)`` enters
its OV map the same way.

The per-token z of the head reads go to ``data/heads/<model>-token-z.pt`` for
the checks (``reads_at_firings.py``, ``head_seek.py``, ``head_census.py``); a summary to ``data/summary/reads_<model>.{log,json}``.

**Experts** (``experts/reads.parquet``, a sparse model's, a row per layer and expert; ``neuron_lens.experts``): the
router's row for the expert times the MLP input norm's scale, a read covector, read as a pole's (``router_R`` through the
metric and the R-lens from the MLP input, ``router_S`` through the S-lens there): what the residual says, and which input
tokens are there, where the router chooses the expert; and the expert's mean map, its neurons' maps added at their mean
RelP factors over the positions routed to it (``06_activations.py``'s ``mean_cg``, ``mean_cu``, the routing weight
included), ``M = W_down (diag(cg) W_gate + diag(cu) W_up)``, read as a head's OV map is: each input token's footprint at
the MLP input (at unit length, as the norm takes it) through ``M``, against the R-lens's readout from the layer's
output (``ov_R``) or the S-lens's footprints there (``ov_S``, through its polar factor), each token's z in its own column,
the expert's copying score of each (``ov_R_copy``, ``ov_S_copy``). A mean over the routed positions, where a neuron's gate
varies from position to position: what the expert does on average where it is chosen, as a head's OV map is what the head
does per unit of attention.

Gemma 4 12B (``NEURON_LENS_MODEL=google/gemma-4-12B``) reads its released lenses. A neuron's write is
what reaches the residual, ``s · γ_post ⊙ W_down[:, n]``, and its read rows are times the
pre-feedforward norm (``neuron_lens/neurons.py``); probabilities and logit statistics are the
model's, after the final softcap; embeddings are × √d, the residual's input. A head's ``W_O`` is
times the post-attention norm weight, its full layers read keys as values, its values pass the
weightless ``v_norm`` (its true derivative averaged over positions, from ``06_activations.py``, as
for ``q_norm`` and ``k_norm``), and RoPE is per layer type.

    python scripts/08_reads.py [--parts neurons,heads,experts] [--layers 0-35]
"""
import argparse
import json
import time

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from neuron_lens import anatomy, arch, metric
from neuron_lens.catalog import (
    MODEL,
    TOP,
    acts_path,
    column,
    expert_keys,
    expert_table,
    head_keys,
    head_table,
    kept,
    lobe_signs,
    neuron_keys,
    neuron_table,
    put,
    write,
    write_heads,
)
from neuron_lens.heads import copy_score, norm_derivative, own_token_z, rope_mean
from neuron_lens.heads import weights as head_weights
from neuron_lens.neurons import (
    directions,
    orthogonal_write,
    poles,
    softcap,
    write_cosines,
)
from neuron_lens.paths import DATA, SUMMARY
from neuron_lens.readout import layer_reads, s_tops
from neuron_lens.stats import auroc, spearman
from neuron_lens.weights import tensors

R_FIT, S_FIT = arch.fits(MODEL)
STEM = MODEL.split("/")[-1].lower()
ACTS = acts_path()
TOKEN_Z = DATA / "heads" / f"{STEM}-token-z.pt"
NULL_SHARE = 0.01  # the bottom 1% of W_U diag(γ)'s singular directions
MIN_OCC = 10  # key occurrences for a token's in-context attention
CHUNK = 2048
BANDS = tuple(arch.bands(MODEL))  # the reading plan's bands (arch.depth_bands)
CAP = softcap(MODEL)
SIGNS, CENTRE = arch.signs(MODEL), arch.centred(MODEL)  # the poles; a LayerNorm's centring (the final norm's too)


def by_sign(x: dict) -> torch.Tensor:
    """A per-sign read (``{sign: tensor}``) as the table's rows, poles in order."""
    return torch.cat([x[s] for s in SIGNS])


def neurons(layers: list[int], dev) -> None:
    keep = kept()
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted(set(nm.values())))
    unembed = head[nm["unembed"]][keep].to(dev, torch.float16)
    gamma = arch.final_scale(MODEL, dev)
    # Euclidean, as the final norm's rms takes a write's size (an entropy neuron works through it); a LayerNorm's
    # readout is W_U diag(γ) P, the constant direction in its null space
    wu = arch.centre(head[nm["unembed"]].float() * arch.final_scale(MODEL), MODEL)
    null = torch.linalg.eigh(wu.T @ wu)[1][:, :max(1, int(NULL_SHARE * wu.shape[1]))].to(dev)  # ascending
    del wu
    emb = head[nm["embed"]][keep].float() * arch.embed_scale(MODEL)  # the residual's input (Gemma: × √d)
    emb = (emb - emb.mean(0)).to(dev, torch.float16)
    J_out, J_mid = arch.lens(MODEL, "R_out", R_FIT), arch.lens(MODEL, "R_mid", R_FIT)  # [L−1, d, d] (no Jacobian at the last layer), [L, d, d]
    K_mid, K_out = arch.lens(MODEL, "S_mid", S_FIT), arch.lens(MODEL, "S_out", S_FIT)  # [L, d_out, d_in]
    for layer in layers:
        t0 = time.perf_counter()
        lobe = lobe_signs(layer)
        p = poles(MODEL, layer, lobe, dev)  # the write as the residual sees it; read covectors and vectors in the metric
        wd, a, b, s2 = p["write"], p["a"], p["b"], p["s2"]
        f = wd.shape[1]
        cols = neuron_keys(layer, f)
        P = len(SIGNS)
        w_norm = wd.norm(dim=0).cpu()
        cos_gate_up = torch.full((f,), float("nan")) if b is None else (a * metric.vector(b, s2)).sum(1).cpu()
        cos_gate, cos_up = (x.cpu() for x in write_cosines(wd, a, b, lobe.to(dev), s2))
        cols["w_norm"] = pa.array(torch.cat([w_norm] * P).numpy())
        cols["cos_gate_up"] = pa.array(torch.cat([cos_gate_up] * P).numpy())
        cols["cos_write_gate"], cols["cos_write_up"] = pa.array(cos_gate.numpy()), pa.array(cos_up.numpy())
        cols["null_share"] = pa.array(torch.cat([((null.T @ wd).pow(2).sum(0) / wd.pow(2).sum(0)).cpu()] * P).numpy())
        logit = layer_reads(wd, unembed, gamma, None, signs=(1,), k=1, kz=1, softcap=CAP, centre=CENTRE)
        cols["vocab_skew"] = pa.array(torch.cat([s * logit["skew"] for s in SIGNS]).numpy())
        cols["vocab_kurt"] = pa.array(torch.cat([logit["kurt"]] * P).numpy())
        r = layer_reads(wd, unembed, gamma, J_out[layer] if layer < len(J_out) else None, signs=SIGNS, k=TOP, kz=TOP,
                        softcap=CAP, centre=CENTRE)
        put(cols, "write_R", keep[by_sign(r["ids"])], by_sign(r["probs"]))
        put(cols, "write_R_z", keep[by_sign(r["zids"])], by_sign(r["z"]))
        K = K_out[layer].to(dev)
        PK = metric.polar(K) @ K  # [d, d_in]: the footprints' rotation, U (§15)
        sw = wd.T @ PK  # [f, d_in]: each write as a covector on the embeddings
        s = s_tops(sw / sw.norm(dim=1, keepdim=True), emb, SIGNS, TOP)
        for kind, name in (("raw", "write_S"), ("z", "write_S_z")):
            put(cols, name, keep[torch.cat([s[kind, x][0] for x in SIGNS])], torch.cat([s[kind, x][1] for x in SIGNS]))
        orth = orthogonal_write(wd, a, b, s2)
        r = layer_reads(orth, unembed, gamma, J_out[layer] if layer < len(J_out) else None, signs=SIGNS, k=TOP, kz=TOP,
                        softcap=CAP, centre=CENTRE)
        put(cols, "write_orth_R", keep[by_sign(r["ids"])], by_sign(r["probs"]))
        put(cols, "write_orth_R_z", keep[by_sign(r["zids"])], by_sign(r["z"]))
        so = orth.T @ PK
        s = s_tops(so / so.norm(dim=1, keepdim=True), emb, SIGNS, TOP)
        for kind, name in (("raw", "write_orth_S"), ("z", "write_orth_S_z")):
            put(cols, name, keep[torch.cat([s[kind, x][0] for x in SIGNS])], torch.cat([s[kind, x][1] for x in SIGNS]))
        del orth, so, K, PK
        c, v = p["cov"], p["vec"]  # [P·f, d]: the read covectors and their vectors, + poles then −
        del p
        reads = [layer_reads(v[i * f:(i + 1) * f].T, unembed, gamma, J_mid[layer], signs=(1,), k=TOP, kz=TOP, softcap=CAP,
                             centre=CENTRE) for i in range(P)]
        put(cols, "read_R", keep[torch.cat([r["ids"][1] for r in reads])], torch.cat([r["probs"][1] for r in reads]))
        put(cols, "read_R_z", keep[torch.cat([r["zids"][1] for r in reads])], torch.cat([r["z"][1] for r in reads]))
        cs = c @ K_mid[layer].to(dev)
        s = s_tops(cs / cs.norm(dim=1, keepdim=True), emb, (1,), TOP)  # unit rows, as the R side's norm
        for kind, name in (("raw", "read_S"), ("z", "read_S_z")):
            put(cols, name, keep[s[kind, 1][0]], s[kind, 1][1])
        write(cols, neuron_table("reads", layer))
        print(f"{time.strftime('%H:%M:%S')} neurons L{layer}: {len(SIGNS) * f} poles in {time.perf_counter() - t0:.0f}s", flush=True)


def lists(cols: dict, name: str, z: torch.Tensor, keep: torch.Tensor, neg: bool = True) -> None:
    """A head read's token lists from its per-token z ``[L, H, V]`` (the grid of heads): the top tokens by z, by z
    z-scored across the layer's heads (``_z``), and with ``neg`` the bottom ones (``_neg``). A row without the read
    (NaN: L0's ``qk_self_R``, an attention head's β, a place in the grid with no head) gets empty lists."""
    has = ~z.isnan().all(-1)  # [L, H]
    zc = (z - z.mean(1, keepdim=True)) / z.std(1, keepdim=True).clamp_min(1e-6)
    part = has.any(1) & ~has.all(1)  # layers where some places lack the read: z across those that have it
    if part.any():
        w = has[part][..., None].float()
        n = w.sum(1, keepdim=True)
        zm = z[part].nan_to_num(0.0)
        mu = (zm * w).sum(1, keepdim=True) / n
        sd = ((zm - mu).pow(2) * w).sum(1, keepdim=True).div((n - 1).clamp_min(1)).sqrt()
        zc[part] = (z[part] - mu) / sd.clamp_min(1e-6)
    for suffix, x in (("", z), ("_z", zc)):
        for sign, tag in ((1, ""), (-1, "_neg")) if neg else ((1, ""),):
            v, i = (sign * x).flatten(0, 1).nan_to_num(0.0).topk(TOP, dim=-1)
            ids, v = keep[i], sign * v
            ids[~has.flatten()], v[~has.flatten()] = -1, float("nan")
            put(cols, f"{name}{tag}{suffix}", ids, v)


def heads(dev) -> None:
    geo = arch.layout(MODEL)
    L, H = len(geo), arch.max_heads(MODEL)  # a grid of heads, each layer's first (Qwen3.5's DeltaNet layers have more)
    grid = arch.head_grid(MODEL)
    keep = kept()
    V = len(keep)
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted(set(nm.values())))
    E = head[nm["embed"]][keep].float() * arch.embed_scale(MODEL)
    E = (E - E.mean(0)).to(dev)
    Wu = arch.centre(head[nm["unembed"]][keep].float() * arch.final_scale(MODEL), MODEL).to(dev)  # the readout, W_U diag(γ) (P)
    del head
    K_out, J_out = arch.lens(MODEL, "S_out", S_FIT), arch.lens(MODEL, "R_out", R_FIT)
    K_w, J_w = arch.head_write_lens(MODEL, "S", S_FIT), arch.head_write_lens(MODEL, "R", R_FIT)  # where the heads write
    saved = torch.load(ACTS, weights_only=False, map_location="cpu", mmap=True)
    st, extra = saved["state"]["total"], saved["state"]["extra"]
    n_q, n_k = float(st["n_q"]), float(st["n_k"])
    z = {k: torch.full((L, H, V), float("nan"), dtype=torch.float16)
         for k in ("ov_R", "ov_S", "qk_S", "qk_self_S", "qk_self_R", "beta_S", "decay_S")}
    score = {k: torch.full((L, H), float("nan")) for k in ("ov_R_copy", "ov_S_copy")}
    for layer, g in enumerate(geo):
        Hl, kvh, hd, kd = g.q_heads, g.kv_heads, g.head_dim, g.kd
        qk = head_weights(MODEL, g, st, dev)  # a DeltaNet's SiLU ratio and an output gate averaged over positions
        Wq, Wk, bq = qk["Wq"], qk["Wk"], qk["bq"]  # times the input norm's weight; GPT-NeoX's query bias
        Wv, Wo = qk["Wv"], qk["Wo"]  # W_V (W_K where keys are values) × γ_in; [d, H·hd], × γ_post for Gemma
        X = arch.centre(E @ K_out[layer - 1].to(dev).T if layer else E, MODEL)  # [V, d]: each token's footprint where the head reads
        # OV: M[u, v] = A_u · B_v, A the output lens's token rows through W_O^h, B each input through W_V
        # (and Gemma's v_norm's true derivative, averaged over positions).
        Bv = (X @ Wv.T).view(V, g.v_heads, hd)
        if "v_r" in st and float(st["v_r"][layer, :kvh].abs().sum()) > 0:  # Gemma's v_norm (Qwen3's files have none)
            Av = norm_derivative(torch.ones(hd, device=dev), st["v_r"][layer, :kvh].float().to(dev),
                                 st["v_m"][layer, :kvh, :hd, :hd].float().to(dev), n_k)
            Bv = torch.einsum("gij,vgj->vgi", Av, Bv)
        B = Bv.repeat_interleave(Hl // g.v_heads, dim=1)
        del Bv
        Km = K_w[layer].to(dev)
        for name, A in (("ov_R", Wu @ (J_w[layer].to(dev) @ Wo)), ("ov_S", E @ (Km.T @ (metric.polar(Km) @ Wo)))):
            A = A.view(V, Hl, hd)
            z[name][layer, :Hl] = own_token_z(A, B).T.half().cpu()
            score[f"{name}_copy"][layer, :Hl] = copy_score(A, B)
            del A
        del B, Km
        # QK: logit q̃ᵀ R̄ k̃, footprints at unit length (a footprint's size would rank keys across tokens), Euclidean
        # as the input norm's rms takes it.
        Xu = X / X.norm(dim=1, keepdim=True)
        # per-layer blocks of the statistics (padded to the largest head size and key-head count); without q/k norms
        # (GPT-NeoX), the identity, the queries and keys reading the input norm's own true derivative averaged over
        # positions (as the lenses do, layers._scores_norm), at a footprint's unit rms: x̂ (I − E[r³ x xᵀ] / (d E[r])),
        # and a footprint's query at the input norm's unit rms, √d x̂, beside the query bias (the key bias adds one
        # score to every key of a query, and cancels)
        An = None
        if qk["q_norm"] is None:
            Aq, Ak = torch.eye(kd, device=dev).expand(Hl, kd, kd), torch.eye(kd, device=dev).expand(kvh, kd, kd)
            d = Xu.shape[1]
            An = torch.eye(d, device=dev) - st["in_m"][layer].float().to(dev) / (d * float(st["in_r"][layer]))
            Xu = Xu @ An  # symmetric
        else:
            Aq = norm_derivative(qk["q_norm"], st["q_r"][layer, :Hl].float().to(dev),
                                 st["q_m"][layer, :Hl, :kd, :kd].float().to(dev), n_q, qk["q_width"])
            Ak = norm_derivative(qk["k_norm"], st["k_r"][layer, :kvh].float().to(dev),
                                 st["k_m"][layer, :kvh, :kd, :kd].float().to(dev), n_k, qk["k_width"])
        def query(x, Wq=Wq, bq=bq):
            return x @ Wq.T if bq is None else x.shape[-1] ** 0.5 * x @ Wq.T + bq
        inv = arch.layer_inv_freq(extra["inv_freq"], MODEL, layer)
        if inv is None:  # no RoPE (a DeltaNet; Olmo Hybrid's attention)
            Rbar = torch.eye(kd, device=dev).expand(Hl, kd, kd)
        else:
            off = st["offset"][layer, :Hl].float().to(dev)
            Rbar = rope_mean(off / off.sum(-1, keepdim=True), inv.to(dev), kd)
        Kv = torch.einsum("gij,vgj->vgi", Ak, (Xu @ Wk.T).view(V, kvh, kd)).repeat_interleave(Hl // kvh, dim=1)
        Kr = torch.einsum("hij,vhj->vhi", Rbar, Kv)  # [V, H, hd]: R̄ K
        del Kv
        Qv = torch.einsum("hij,vhj->vhi", Aq, query(Xu).view(V, Hl, kd))
        a = torch.einsum("vhi,hi->vh", Kr, (st["q"][layer, :Hl, :kd] / n_q).float().to(dev))
        z["qk_S"][layer, :Hl] = ((a - a.mean(0)) / a.std(0)).T.half().cpu()
        z["qk_self_S"][layer, :Hl] = own_token_z(Kr, Qv).T.half().cpu()
        del Qv
        if layer:  # the query the position's R-lens direction (what it says, a covector, as a vector), the key its footprint
            Ru = arch.centre(metric.vector(Wu @ J_out[layer - 1].to(dev), metric.s2(MODEL, "in", layer, dev)), MODEL)
            Ru = Ru / Ru.norm(dim=1, keepdim=True)
            if An is not None:
                Ru = Ru @ An
            Qr = torch.einsum("hij,vhj->vhi", Aq, query(Ru).view(V, Hl, kd))
            z["qk_self_R"][layer, :Hl] = own_token_z(Kr, Qr).T.half().cpu()
            del Ru, Qr
        if g.deltanet:  # the write strength's and the decay's rows, each token's footprint z over the vocabulary
            for name, w in arch.delta_gates(MODEL, g, dev).items():
                a = Xu @ w.T  # [V, H]
                z[f"{name}_S"][layer, :Hl] = ((a - a.mean(0)) / a.std(0)).T.half().cpu()
        del Kr, X, Xu
        print(f"{time.strftime('%H:%M:%S')} heads L{layer}", flush=True)
    TOKEN_Z.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"keep": keep, **z}, TOKEN_Z)

    # In context: attends-to against the attention each key token receives; alignment with OV.
    seen = st["occ"][keep] >= MIN_OCC
    recv = (st["recv"][:, :, keep][:, :, seen] / st["pairs"][keep][seen]).float()  # [L, H, seen]
    draw = torch.randint(1, H, (L, H), generator=torch.Generator().manual_seed(0))  # another head of the layer, a control
    ctx, ctx_ctl, align, align_ctl = (torch.full((L, H), float("nan")) for _ in range(4))
    rk = lambda x: x.argsort(-1).argsort(-1).float()
    for l, g in enumerate(geo):
        Hl = g.q_heads
        hh = torch.arange(Hl, device=dev)
        other = ((torch.arange(Hl) + 1 + (draw[l, :Hl] - 1) % (Hl - 1)) % Hl).to(dev)  # (H heads in every layer: the draw itself)
        za, zo, rr = (rk(x[:Hl].to(dev).float()) for x in (z["qk_S"][l], z["ov_R"][l], recv[l]))
        za_s = rk(z["qk_S"][l, :Hl][:, seen].to(dev).float())
        za, zo, rr, za_s = (x - x.mean(-1, keepdim=True) for x in (za, zo, rr, za_s))
        nrm = lambda x: x / x.norm(dim=-1, keepdim=True)
        c1, c2 = nrm(za_s) @ nrm(rr).T, nrm(za) @ nrm(zo).T  # [H, H']
        ctx[l, :Hl], ctx_ctl[l, :Hl] = c1.diagonal().cpu(), c1[other, hh].cpu()
        align[l, :Hl], align_ctl[l, :Hl] = c2.diagonal().cpu(), c2[hh, other].cpu()
    med = {k: z[k].float().median(-1).values for k in ("qk_self_S", "qk_self_R")}

    cols = head_keys()
    cols |= {k: pa.array(v.flatten().numpy()) for k, v in score.items()}
    for k, v in med.items():
        cols[k] = pa.array(v.flatten().numpy())
    cols["qk_ctx_rho"] = pa.array(ctx.flatten().numpy())
    cols["qk_ov_rho"] = pa.array(align.flatten().numpy())
    for name in ("ov_R", "ov_S", "qk_S", "beta_S", "decay_S"):
        lists(cols, name, z[name].float(), keep)
    lists(cols, "qk_self_S", z["qk_self_S"].float(), keep, neg=False)
    lists(cols, "qk_self_R", z["qk_self_R"].float(), keep, neg=False)  # L0 has no R-lens input
    write_heads(cols, head_table("reads"))

    dup = extra["duplicate"][grid]  # the heads that exist, layer-major
    is_dup = dup > 0.3
    late = grid.clone()
    late[0] = False  # qk_self_R from L1
    real = lambda x: x[grid]
    lines = [(f"head reads, {MODEL}: {int(grid.sum())} heads in {L} layers, {V} tokens; QK statistics over "
              f"{saved['meta']['docs']} documents, {int(seen.sum())} tokens seen {MIN_OCC}+ times as keys"), "",
             (f"qk_self_S against duplicate-token heads (duplicate > 0.3, {int(is_dup.sum())} heads): AUROC "
              f"{auroc(real(med['qk_self_S']), is_dup):.2f}, Spearman with the duplicate score "
              f"{spearman(real(med['qk_self_S']), dup):+.2f}; qk_self_R {auroc(med['qk_self_R'][late], extra['duplicate'][late] > 0.3):.2f} "
              f"(from L1)"),
             "", ("by band: qk_ctx_rho mean (own / control head), share past +0.2 (own / control); qk_ov_rho mean "
                  "(own / control), share past +0.2 and under −0.2 (own / control); ov_R_copy and ov_S_copy medians; "
                  "qk_self_S and qk_self_R medians")]
    table = {"qk_self_S_auroc": auroc(real(med["qk_self_S"]), is_dup),
             "qk_self_R_auroc": auroc(med["qk_self_R"][late], extra["duplicate"][late] > 0.3), "bands": {}}
    for lo, hi in BANDS + ((0, L - 1),):
        s = slice(lo, hi + 1)
        b = lambda x, s=s: x[s][grid[s]]  # the band's heads
        row = {"ctx": [b(ctx).mean().item(), b(ctx_ctl).mean().item(), (b(ctx) > 0.2).float().mean().item(),
                       (b(ctx_ctl) > 0.2).float().mean().item()],
               "align": [b(align).mean().item(), b(align_ctl).mean().item(), (b(align) > 0.2).float().mean().item(),
                         (b(align_ctl) > 0.2).float().mean().item(), (b(align) < -0.2).float().mean().item(),
                         (b(align_ctl) < -0.2).float().mean().item()],
               "ov_R_copy": b(score["ov_R_copy"]).median().item(), "ov_S_copy": b(score["ov_S_copy"]).median().item(),
               **{k: b(v).nanmedian().item() for k, v in med.items()}}
        table["bands"][f"L{lo}-{hi}"] = row
        c, a_ = row["ctx"], row["align"]
        lines.append(f"  L{lo}-{hi}: ctx {c[0]:+.2f}/{c[1]:+.2f} ({c[2]:.2f}/{c[3]:.2f})  align {a_[0]:+.3f}/{a_[1]:+.3f} "
                     f"(+ {a_[2]:.2f}/{a_[3]:.2f}, − {a_[4]:.2f}/{a_[5]:.2f})  copy R {row['ov_R_copy']:+.2f} S {row['ov_S_copy']:+.2f}  "
                     f"self S {row['qk_self_S']:+.2f} R {row['qk_self_R']:+.2f}")
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"reads_{STEM}.log").write_text(text + "\n")
    (SUMMARY / f"reads_{STEM}.json").write_text(json.dumps(table))


EXPERT_BLOCK = 2**31  # entries of an expert block's token maps, A and B ([V, experts, 2I] each), at a time


def experts_reads(layers: list[int], dev) -> None:
    """``experts/reads.parquet`` (module docstring), the rows of ``layers`` in place of the file's rows for them."""
    keep = kept()
    V = len(keep)
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted(set(nm.values())))
    unembed = head[nm["unembed"]][keep].to(dev, torch.float16)
    gamma = arch.final_scale(MODEL, dev)
    Wu = arch.centre(head[nm["unembed"]][keep].float() * arch.final_scale(MODEL), MODEL).to(dev)  # the readout, W_U diag(γ)
    E0 = head[nm["embed"]][keep].float() * arch.embed_scale(MODEL)
    E0 = (E0 - E0.mean(0)).to(dev)
    del head
    J_out, J_mid = arch.lens(MODEL, "R_out", R_FIT), arch.lens(MODEL, "R_mid", R_FIT)
    K_mid, K_out = arch.lens(MODEL, "S_mid", S_FIT), arch.lens(MODEL, "S_out", S_FIT)
    acts = pq.read_table(expert_table("acts"), columns=["layer", "expert", "mean_cg", "mean_cu"])
    geo = arch.layout(MODEL)
    out = []
    for layer in layers:
        t0 = time.perf_counter()
        rows = (acts.column("layer").to_numpy() == layer).nonzero()[0]
        sub = acts.take(rows)
        n_exp = len(rows)
        cg, cu = column(sub, "mean_cg").to(dev), column(sub, "mean_cu").to(dev)  # [E, I]
        I = cg.shape[1]
        p = anatomy.layer_parts(MODEL, layer)
        router = arch._weights(MODEL, geo[layer], {"r": ("mlp.gate", "weight")}, dev)["r"]  # [E, d]
        g = arch.norm_scale(MODEL, geo[layer], p.n2, dev)
        router = router if g is None else router * g
        cols = expert_keys([layer], n_exp)
        s2 = metric.s2(MODEL, "mid", layer, dev)
        v = metric.vector(router, s2)
        r = layer_reads((v / v.norm(dim=1, keepdim=True)).T, unembed, gamma, J_mid[layer], signs=(1,), k=TOP, kz=TOP,
                        softcap=CAP, centre=CENTRE)
        put(cols, "router_R", keep[r["ids"][1]], r["probs"][1])
        put(cols, "router_R_z", keep[r["zids"][1]], r["z"][1])
        cs = router @ K_mid[layer].to(dev)
        s = s_tops(cs / cs.norm(dim=1, keepdim=True), E0.half(), (1,), TOP)
        for kind, name in (("raw", "router_S"), ("z", "router_S_z")):
            put(cols, name, keep[s[kind, 1][0]], s[kind, 1][1])
        # the mean map: input footprints at the MLP input, at unit length, through the gate and up rows; the output side
        # each token's readout through W_down at the mean factors
        w = directions(MODEL, layer, dev)
        X = E0 @ K_mid[layer].to(dev).T
        X = X / X.norm(dim=1, keepdim=True)
        Jl = J_out[layer].to(dev) if layer < len(J_out) else torch.eye(X.shape[1], device=dev)
        Ko = K_out[layer].to(dev)
        reads = {"ov_R": Wu @ Jl, "ov_S": E0 @ (Ko.T @ metric.polar(Ko))}  # [V, d]: each token's row on a write
        z = {k: torch.zeros(n_exp, V, dtype=torch.float16) for k in reads}
        score = {k: torch.zeros(n_exp) for k in reads}
        step = max(1, EXPERT_BLOCK // (V * 2 * I))
        for e0 in range(0, n_exp, step):
            es = range(e0, min(e0 + step, n_exp))
            B = torch.stack([torch.cat([X @ w["gate"][e * I:(e + 1) * I].T, X @ w["up"][e * I:(e + 1) * I].T], 1)
                             for e in es], 1)  # [V, eb, 2I]
            Wd = torch.stack([torch.cat([w["write"][:, e * I:(e + 1) * I] * cg[e], w["write"][:, e * I:(e + 1) * I] * cu[e]], 1)
                              for e in es])  # [eb, d, 2I]
            for k, R in reads.items():
                A = torch.einsum("vd,edj->vej", R, Wd)
                z[k][e0:e0 + len(es)] = own_token_z(A, B).T.half().cpu()
                score[k][e0:e0 + len(es)] = copy_score(A, B)
                del A
            del B, Wd
        for k in reads:
            lists(cols, k, z[k][None].float(), keep)
            cols[f"{k}_copy"] = pa.array(score[k].numpy())
        out.append(pa.table(cols))
        print(f"{time.strftime('%H:%M:%S')} experts L{layer}: {n_exp} in {time.perf_counter() - t0:.0f}s", flush=True)
    table = pa.concat_tables(out)
    path = expert_table("reads")
    if path.exists():
        import pyarrow.compute as pc
        old = pq.read_table(path)
        table = pa.concat_tables([old.filter(pc.invert(pc.is_in(old.column("layer"), pa.array(layers, pa.int16())))),
                                  table]).sort_by([("layer", "ascending"), ("expert", "ascending")])
    pq.write_table(table, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="neurons,heads,experts", help="experts: a sparse model's alone")
    ap.add_argument("--layers", default=None, help="neuron and expert layers, a-b (default all)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    n = arch.text_config(MODEL).num_hidden_layers
    lo, hi = (0, n - 1) if args.layers is None else map(int, args.layers.split("-"))
    if "neurons" in parts:
        neurons(list(range(lo, hi + 1)), args.device)
    if "heads" in parts:
        heads(args.device)
    if "experts" in parts and expert_table("acts").exists():
        experts_reads([l for l in range(lo, hi + 1) if anatomy.layer_parts(MODEL, l).sparse], args.device)


if __name__ == "__main__":
    main()
