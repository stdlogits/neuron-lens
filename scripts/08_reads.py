"""Weight-only reads of every neuron pole and attention head through the lenses (pipeline stage 8).

The lens rule: ``out`` reads MLP outputs and attention inputs, ``mid`` reads MLP inputs and
attention outputs. Every read lists the top 64 kept tokens (``catalog.kept``), with a ``_z`` variant
that z-scores each token's score across the layer's poles or heads, which removes the hub tokens that
top most raw reads (findings §6).

A lens takes vectors or covectors as it was built: the R-lens transports a vector, the S-lens pulls back a
covector. A read row (what a unit applies to the residual) is a covector, a write a vector, so a read that feeds
a read row to the R-lens goes through the residual's metric (``neuron_lens.metric``: ``v = s² ⊙ c``,
``s² = σ / |γ|``), which keeps the read free of the units each residual coordinate carries; one that sets a vector
against the S-lens's footprints goes through the lens's polar factor (``metric.polar``: ``w · U ē``, the footprints
with the lens's stretch removed), which asks which input token would put it there (findings §13).

**Neurons** (``neurons/reads/L<layer>.parquet``), no forward pass:

- ``write_R``: the write ``±W_down[:, n]`` through the R-lens, ``softmax(W_U · norm(J_L w))``
  (``_val``: probabilities; the logit lens at the last layer);
- ``write_S``: the write against the S-lens's footprints at the layer's output, tokens ``v`` by
  ``±w · U (e_v − ē)``, ``K^out_L = (K Kᵀ)^½ U`` (``metric.polar``): the input tokens whose footprint the write
  imitates (legible to about L8 in Qwen3-8B, where early neurons write back the tokens they detect; §6, §7, §13);
- ``write_orth_R``, ``write_orth_S``: the same of the write minus its parts along the neuron's gate and up
  patterns, so that neither row reads it (``neurons.orthogonal_write``), what it adds beyond anything it reads (§18);
- ``read_R``: the read covector ``c = â ± b̂`` (gate and up rows times the MLP input norm's weight, each at
  unit length in the metric; the up row alone, ``∓b̂``, for the poles on SiLU's negative lobe, ``acts``'
  ``neg_lobe`` past a half, §17) as a vector, ``s² ⊙ c`` (the raw rows times the residual's scale), through the
  R-lens from the layer's MLP input (``J^mid_L``): what the residual says when it drives the pole (§13);
- ``read_S``: the read covector through the S-lens to the MLP input, ``c · K^mid_L (e_v − ē)``:
  input tokens that drive the pole (legible at L0-L8);
- weights: ``w_norm`` (``‖W_down[:, n]‖``), ``cos_gate_up``, ``cos_write_gate``, ``cos_write_up`` (cosines in the metric
  at the MLP input; the latter two of the pole's write with the gate and up sides of what it reads: positive where it
  writes toward them; §18), ``null_share`` (the share of the write's squared length in the bottom 1% of right-singular
  directions of ``W_U diag(γ)``, Euclidean as the final norm's rms takes it, where entropy neurons write; Stolfo et al.
  2024), and ``vocab_skew``, ``vocab_kurt`` of ``W_U · norm(w)`` over
  the vocabulary (Gurnee et al. 2024's prediction and suppression neurons).

**Heads** (``heads/reads.parquet``; findings §11): a head reads token ``v`` as its S-lens
footprint at layer ``L − 1``'s output (the centred embedding at L0) and writes into layer ``L``'s
MLP input (``neuron_lens.heads``).

- ``ov_R``: the OV map read through the R-lens: each token's z in its own column, where ``v`` ranks
  in what the head makes the model say when it reads ``v``; the tokens it copies most, and
  ``ov_R_neg`` those it suppresses most. ``ov_S``: the same against the S-lens's footprints at the MLP input
  through its polar factor, as ``write_S``: whether the write makes the residual look as if ``v`` were there (the
  record through L8; §11, §13).
  ``ov_R_copy``, ``ov_S_copy``: Elhage et al.'s copying score Σλ/Σ|λ| of each map (§11);
- ``qk_S``: the QK map, ``q̃ᵀ R̄ k̃`` with the q/k norms' true derivative averaged over positions
  and RoPE averaged over the head's own attention offsets (``06_activations.py``'s statistics),
  footprints at unit length: each key token's z over the vocabulary under the head's mean query,
  what it attends to (``qk_S_neg``: avoids); ``qk_self_S``: each token's z in its own row, where it
  ranks among the keys it favours as a query whose position holds it (its S-lens footprint as the query too),
  and the scalar ``qk_self_S`` the median of that z over tokens, which finds duplicate-token heads (§11);
  ``qk_self_R``: the same with the query the R-lens direction at ``L − 1``'s output that promotes the token
  (the covector ``W_U diag(γ_f) J^out_{L−1}`` as a vector in the metric, the position is about to say it), which finds the late heads that look up
  earlier copies of what is about to be said, with a negative ``ov_R_copy`` the copy suppressors (§11; from L1,
  empty at L0);
- ``qk_ctx_rho``: Spearman over tokens between ``qk_S`` and the attention each key token receives
  in context (tokens seen as keys 10+ times); ``qk_ov_rho``: between ``qk_S`` and ``ov_R``'s z,
  whether a head attends to what it copies.

The per-token z of the five head reads go to ``data/heads/<model>-token-z.pt`` for
the checks (``reads_at_firings.py``, ``head_seek.py``, ``head_census.py``); a summary to ``data/summary/reads_<model>.{log,json}``.

Gemma 4 12B (``NEURON_LENS_MODEL=google/gemma-4-12B``) reads its released lenses. A neuron's write is
what reaches the residual, ``s · γ_post ⊙ W_down[:, n]``, and its read rows are times the
pre-feedforward norm (``neuron_lens/neurons.py``); probabilities and logit statistics are the
model's, after the final softcap; embeddings are × √d, the residual's input. A head's ``W_O`` is
times the post-attention norm weight, its full layers read keys as values, its values pass the
weightless ``v_norm`` (its true derivative averaged over positions, from ``06_activations.py``, as
for ``q_norm`` and ``k_norm``), and RoPE is per layer type.

    python scripts/08_reads.py [--parts neurons,heads] [--layers 0-35]
"""
import argparse
import json
import time

import pyarrow as pa
import torch

from neuron_lens import arch, metric
from neuron_lens.catalog import (
    MODEL,
    TOP,
    acts_path,
    head_keys,
    head_table,
    kept,
    lobe_signs,
    neuron_keys,
    neuron_table,
    put,
    write,
)
from neuron_lens.heads import copy_score, norm_derivative, own_token_z, rope_mean
from neuron_lens.neurons import orthogonal_write, poles, softcap, write_cosines
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
BANDS = tuple(arch.bands(arch.text_config(MODEL).num_hidden_layers))  # Qwen3-8B's five, else quarters
CAP = softcap(MODEL)


def neurons(layers: list[int], dev) -> None:
    keep = kept()
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted(set(nm.values())))
    unembed = head[nm["unembed"]][keep].to(dev, torch.float16)
    gamma = head[nm["norm"]].to(dev, torch.float32)
    # Euclidean, as the final norm's rms takes a write's size (an entropy neuron works through it)
    wu = head[nm["unembed"]].float() * head[nm["norm"]].float()
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
        w_norm, cos_gate_up = wd.norm(dim=0).cpu(), (a * metric.vector(b, s2)).sum(1).cpu()
        cos_gate, cos_up = (x.cpu() for x in write_cosines(wd, a, b, lobe.to(dev), s2))
        cols["w_norm"] = pa.array(torch.cat([w_norm, w_norm]).numpy())
        cols["cos_gate_up"] = pa.array(torch.cat([cos_gate_up, cos_gate_up]).numpy())
        cols["cos_write_gate"], cols["cos_write_up"] = pa.array(cos_gate.numpy()), pa.array(cos_up.numpy())
        cols["null_share"] = pa.array(torch.cat([((null.T @ wd).pow(2).sum(0) / wd.pow(2).sum(0)).cpu()] * 2).numpy())
        logit = layer_reads(wd, unembed, gamma, None, signs=(1,), k=1, kz=1, softcap=CAP)
        cols["vocab_skew"] = pa.array(torch.cat([logit["skew"], -logit["skew"]]).numpy())
        cols["vocab_kurt"] = pa.array(torch.cat([logit["kurt"], logit["kurt"]]).numpy())
        r = layer_reads(wd, unembed, gamma, J_out[layer] if layer < len(J_out) else None, k=TOP, kz=TOP, softcap=CAP)
        put(cols, "write_R", keep[torch.cat([r["ids"][1], r["ids"][-1]])], torch.cat([r["probs"][1], r["probs"][-1]]))
        put(cols, "write_R_z", keep[torch.cat([r["zids"][1], r["zids"][-1]])], torch.cat([r["z"][1], r["z"][-1]]))
        K = K_out[layer].to(dev)
        PK = metric.polar(K) @ K  # [d, d_in]: the footprints' rotation, U (§13)
        sw = wd.T @ PK  # [f, d_in]: each write as a covector on the embeddings
        s = s_tops(sw / sw.norm(dim=1, keepdim=True), emb, (1, -1), TOP)
        for kind, name in (("raw", "write_S"), ("z", "write_S_z")):
            put(cols, name, keep[torch.cat([s[kind, 1][0], s[kind, -1][0]])], torch.cat([s[kind, 1][1], s[kind, -1][1]]))
        orth = orthogonal_write(wd, a, b, s2)
        r = layer_reads(orth, unembed, gamma, J_out[layer] if layer < len(J_out) else None, k=TOP, kz=TOP, softcap=CAP)
        put(cols, "write_orth_R", keep[torch.cat([r["ids"][1], r["ids"][-1]])], torch.cat([r["probs"][1], r["probs"][-1]]))
        put(cols, "write_orth_R_z", keep[torch.cat([r["zids"][1], r["zids"][-1]])], torch.cat([r["z"][1], r["z"][-1]]))
        so = orth.T @ PK
        s = s_tops(so / so.norm(dim=1, keepdim=True), emb, (1, -1), TOP)
        for kind, name in (("raw", "write_orth_S"), ("z", "write_orth_S_z")):
            put(cols, name, keep[torch.cat([s[kind, 1][0], s[kind, -1][0]])], torch.cat([s[kind, 1][1], s[kind, -1][1]]))
        del orth, so, K, PK
        c, v = p["cov"], p["vec"]  # [2f, d]: the read covectors and their vectors, + poles then −
        del p
        reads = [layer_reads(v[i * f:(i + 1) * f].T, unembed, gamma, J_mid[layer], signs=(1,), k=TOP, kz=TOP, softcap=CAP)
                 for i in (0, 1)]
        put(cols, "read_R", keep[torch.cat([r["ids"][1] for r in reads])], torch.cat([r["probs"][1] for r in reads]))
        put(cols, "read_R_z", keep[torch.cat([r["zids"][1] for r in reads])], torch.cat([r["z"][1] for r in reads]))
        cs = c @ K_mid[layer].to(dev)
        s = s_tops(cs / cs.norm(dim=1, keepdim=True), emb, (1,), TOP)  # unit rows, as the R side's norm
        for kind, name in (("raw", "read_S"), ("z", "read_S_z")):
            put(cols, name, keep[s[kind, 1][0]], s[kind, 1][1])
        write(cols, neuron_table("reads", layer))
        print(f"{time.strftime('%H:%M:%S')} neurons L{layer}: {2 * f} poles in {time.perf_counter() - t0:.0f}s", flush=True)


def lists(cols: dict, name: str, z: torch.Tensor, keep: torch.Tensor, neg: bool = True, start: int = 0) -> None:
    """A head read's token lists from its per-token z ``[L, H, V]``: the top tokens by z, by z
    z-scored across the layer's heads (``_z``), and with ``neg`` the bottom ones (``_neg``); layers
    below ``start`` (no read there) get empty lists."""
    zc = (z - z.mean(1, keepdim=True)) / z.std(1, keepdim=True).clamp_min(1e-6)
    for suffix, x in (("", z), ("_z", zc)):
        for sign, tag in ((1, ""), (-1, "_neg")) if neg else ((1, ""),):
            v, i = (sign * x).flatten(0, 1).nan_to_num(0.0).topk(TOP, dim=-1)
            ids, v = keep[i], sign * v
            ids[:start * z.shape[1]], v[:start * z.shape[1]] = -1, float("nan")
            put(cols, f"{name}{tag}{suffix}", ids, v)


def heads(dev) -> None:
    geo = arch.layout(MODEL)
    L, H = len(geo), geo[0].q_heads
    keep = kept()
    V = len(keep)
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted(set(nm.values())))
    E = head[nm["embed"]][keep].float() * arch.embed_scale(MODEL)
    E = (E - E.mean(0)).to(dev)
    Wu = (head[nm["unembed"]][keep].float() * head[nm["norm"]].float()).to(dev)
    del head
    K_out, K_mid = arch.lens(MODEL, "S_out", S_FIT), arch.lens(MODEL, "S_mid", S_FIT)
    J_mid, J_out = arch.lens(MODEL, "R_mid", R_FIT), arch.lens(MODEL, "R_out", R_FIT)
    saved = torch.load(ACTS, weights_only=False, map_location="cpu", mmap=True)
    st, extra = saved["state"]["total"], saved["state"]["extra"]
    z = {k: torch.zeros(L, H, V, dtype=torch.float16) for k in ("ov_R", "ov_S", "qk_S", "qk_self_S", "qk_self_R")}
    z["qk_self_R"][0] = float("nan")  # no R-lens below L0
    score = {k: torch.zeros(L, H) for k in ("ov_R_copy", "ov_S_copy")}
    for layer in range(L):
        kvh, hd = geo[layer].kv_heads, geo[layer].head_dim
        g = H // kvh
        qk = arch.qk(MODEL, geo[layer], dev)
        Wq, Wk = qk["Wq"], qk["Wk"]  # times the input norm's weight
        Wv, Wo = arch.ov(MODEL, geo[layer], dev)  # W_V (W_K where keys are values) × γ_in; [d, H·hd], × γ_post for Gemma
        X = E @ K_out[layer - 1].to(dev).T if layer else E  # [V, d]: each token's footprint where the head reads
        # OV: M[u, v] = A_u · B_v, A the output lens's token rows through W_O^h, B each input through W_V
        # (and Gemma's v_norm's true derivative, averaged over positions).
        Bv = (X @ Wv.T).view(V, kvh, hd)
        if "v_r" in st and float(st["v_r"][layer, :kvh].abs().sum()) > 0:  # Gemma's v_norm (Qwen3's files have none)
            Av = norm_derivative(torch.ones(hd, device=dev), st["v_r"][layer, :kvh].float().to(dev),
                                 st["v_m"][layer, :kvh, :hd, :hd].float().to(dev), float(st["n_k"]))
            Bv = torch.einsum("gij,vgj->vgi", Av, Bv)
        B = Bv.repeat_interleave(g, dim=1)
        del Bv
        Km = K_mid[layer].to(dev)
        for name, A in (("ov_R", Wu @ (J_mid[layer].to(dev) @ Wo)), ("ov_S", E @ (Km.T @ (metric.polar(Km) @ Wo)))):
            A = A.view(V, H, hd)
            z[name][layer] = own_token_z(A, B).T.half().cpu()
            score[f"{name}_copy"][layer] = copy_score(A, B)
            del A
        del B, Km
        # QK: logit q̃ᵀ R̄ k̃, footprints at unit length (a footprint's size would rank keys across tokens), Euclidean
        # as the input norm's rms takes it.
        Xu = X / X.norm(dim=1, keepdim=True)
        # per-layer blocks of the statistics (padded to the largest head size and key-head count)
        Aq = norm_derivative(qk["q_norm"], st["q_r"][layer].float().to(dev),
                             st["q_m"][layer, :, :hd, :hd].float().to(dev), float(st["n_q"]))
        Ak = norm_derivative(qk["k_norm"], st["k_r"][layer, :kvh].float().to(dev),
                             st["k_m"][layer, :kvh, :hd, :hd].float().to(dev), float(st["n_k"]))
        off = st["offset"][layer].float().to(dev)
        Rbar = rope_mean(off / off.sum(-1, keepdim=True), arch.layer_inv_freq(extra["inv_freq"], MODEL, layer).to(dev))
        Kv = torch.einsum("gij,vgj->vgi", Ak, (Xu @ Wk.T).view(V, kvh, hd)).repeat_interleave(g, dim=1)
        Kr = torch.einsum("hij,vhj->vhi", Rbar, Kv)  # [V, H, hd]: R̄ K
        del Kv
        Qv = torch.einsum("hij,vhj->vhi", Aq, (Xu @ Wq.T).view(V, H, hd))
        a = torch.einsum("vhi,hi->vh", Kr, (st["q"][layer, :, :hd] / st["n_q"]).float().to(dev))
        z["qk_S"][layer] = ((a - a.mean(0)) / a.std(0)).T.half().cpu()
        z["qk_self_S"][layer] = own_token_z(Kr, Qv).T.half().cpu()
        del Qv
        if layer:  # the query the position's R-lens direction (what it says, a covector, as a vector), the key its footprint
            Ru = metric.vector(Wu @ J_out[layer - 1].to(dev), metric.s2(MODEL, "in", layer, dev))
            Ru = Ru / Ru.norm(dim=1, keepdim=True)
            Qr = torch.einsum("hij,vhj->vhi", Aq, (Ru @ Wq.T).view(V, H, hd))
            z["qk_self_R"][layer] = own_token_z(Kr, Qr).T.half().cpu()
            del Ru, Qr
        del Kr, X, Xu
        print(f"{time.strftime('%H:%M:%S')} heads L{layer}", flush=True)
    TOKEN_Z.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"keep": keep, **z}, TOKEN_Z)

    # In context: attends-to against the attention each key token receives; alignment with OV.
    seen = st["occ"][keep] >= MIN_OCC
    recv = (st["recv"][:, :, keep][:, :, seen] / st["pairs"][keep][seen]).float()  # [L, H, seen]
    other = (torch.arange(H)[None, :] + torch.randint(1, H, (L, H), generator=torch.Generator().manual_seed(0))) % H
    ctx, ctx_ctl, align, align_ctl = (torch.zeros(L, H) for _ in range(4))
    rk = lambda x: x.argsort(-1).argsort(-1).float()
    for l in range(L):
        za, zo, rr = (rk(x.to(dev).float()) for x in (z["qk_S"][l], z["ov_R"][l], recv[l]))
        za_s = rk(z["qk_S"][l][:, seen].to(dev).float())
        za, zo, rr, za_s = (x - x.mean(-1, keepdim=True) for x in (za, zo, rr, za_s))
        nrm = lambda x: x / x.norm(dim=-1, keepdim=True)
        c1, c2 = nrm(za_s) @ nrm(rr).T, nrm(za) @ nrm(zo).T  # [H, H']
        hh = torch.arange(H, device=dev)
        ctx[l], ctx_ctl[l] = c1.diagonal().cpu(), c1[other[l].to(dev), hh].cpu()
        align[l], align_ctl[l] = c2.diagonal().cpu(), c2[hh, other[l].to(dev)].cpu()
    med = {k: z[k].float().median(-1).values for k in ("qk_self_S", "qk_self_R")}

    cols = head_keys(L, H, arch.groups(MODEL))
    cols |= {k: pa.array(v.flatten().numpy()) for k, v in score.items()}
    for k, v in med.items():
        cols[k] = pa.array(v.flatten().numpy())
    cols["qk_ctx_rho"] = pa.array(ctx.flatten().numpy())
    cols["qk_ov_rho"] = pa.array(align.flatten().numpy())
    for name in ("ov_R", "ov_S", "qk_S"):
        lists(cols, name, z[name].float(), keep)
    lists(cols, "qk_self_S", z["qk_self_S"].float(), keep, neg=False)
    lists(cols, "qk_self_R", z["qk_self_R"].float(), keep, neg=False, start=1)  # L0 has no R-lens input
    write(cols, head_table("reads"))

    dup = extra["duplicate"].flatten()
    is_dup = dup > 0.3
    lines = [(f"head reads, {MODEL}: {L} × {H} heads, {V} tokens; QK statistics over {saved['meta']['docs']} documents, "
              f"{int(seen.sum())} tokens seen {MIN_OCC}+ times as keys"), "",
             (f"qk_self_S against duplicate-token heads (duplicate > 0.3, {int(is_dup.sum())} heads): AUROC "
              f"{auroc(med['qk_self_S'].flatten(), is_dup):.2f}, Spearman with the duplicate score "
              f"{spearman(med['qk_self_S'].flatten(), dup):+.2f}; qk_self_R {auroc(med['qk_self_R'][1:].flatten(), is_dup[H:]):.2f} "
              f"(from L1)"),
             "", ("by band: qk_ctx_rho mean (own / control head), share past +0.2 (own / control); qk_ov_rho mean "
                  "(own / control), share past +0.2 and under −0.2 (own / control); ov_R_copy and ov_S_copy medians; "
                  "qk_self_S and qk_self_R medians")]
    table = {"qk_self_S_auroc": auroc(med["qk_self_S"].flatten(), is_dup),
             "qk_self_R_auroc": auroc(med["qk_self_R"][1:].flatten(), is_dup[H:]), "bands": {}}
    for lo, hi in BANDS + ((0, L - 1),):
        s = slice(lo, hi + 1)
        row = {"ctx": [ctx[s].mean().item(), ctx_ctl[s].mean().item(), (ctx[s] > 0.2).float().mean().item(),
                       (ctx_ctl[s] > 0.2).float().mean().item()],
               "align": [align[s].mean().item(), align_ctl[s].mean().item(), (align[s] > 0.2).float().mean().item(),
                         (align_ctl[s] > 0.2).float().mean().item(), (align[s] < -0.2).float().mean().item(),
                         (align_ctl[s] < -0.2).float().mean().item()],
               "ov_R_copy": score["ov_R_copy"][s].median().item(), "ov_S_copy": score["ov_S_copy"][s].median().item(),
               **{k: v[s].nanmedian().item() for k, v in med.items()}}
        table["bands"][f"L{lo}-{hi}"] = row
        c, a_ = row["ctx"], row["align"]
        lines.append(f"  L{lo}-{hi}: ctx {c[0]:+.2f}/{c[1]:+.2f} ({c[2]:.2f}/{c[3]:.2f})  align {a_[0]:+.3f}/{a_[1]:+.3f} "
                     f"(+ {a_[2]:.2f}/{a_[3]:.2f}, − {a_[4]:.2f}/{a_[5]:.2f})  copy R {row['ov_R_copy']:+.2f} S {row['ov_S_copy']:+.2f}  "
                     f"self S {row['qk_self_S']:+.2f} R {row['qk_self_R']:+.2f}")
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"reads_{STEM}.log").write_text(text + "\n")
    (SUMMARY / f"reads_{STEM}.json").write_text(json.dumps(table))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="neurons,heads")
    ap.add_argument("--layers", default=None, help="neuron layers, a-b (default all)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    if "neurons" in parts:
        n = arch.text_config(MODEL).num_hidden_layers
        lo, hi = (0, n - 1) if args.layers is None else map(int, args.layers.split("-"))
        neurons(list(range(lo, hi + 1)), args.device)
    if "heads" in parts:
        heads(args.device)


if __name__ == "__main__":
    main()
