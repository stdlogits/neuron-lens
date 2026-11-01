"""Attention heads' OV copying scores between their read and write points, against what the heads
measurably do (findings §11, "an earlier, coarser test").

A head ``h`` of layer ``L`` reads layer ``L − 1``'s output and writes into layer ``L``'s MLP input.
Its token map is ``M[u, v] = A_u · B_v`` (``neuron_lens.heads``): ``B_v = W_V^{kv} γ₁ X_v``, the head
reading token ``v``, and ``A_u = U_u W_O^h``, its write read as token ``u``, between those points:

- **input** ``X_v`` at ``L − 1``'s output: **S**, the S-lens footprint ``K^out_{L−1}(e_v − ē)`` (the
  centred embedding at L0); **R**, the R-lens direction ``J^{out⊤}_{L−1} γ_f u_v`` that promotes
  ``v`` (none at L0);
- **output** ``U_u`` at ``L``'s MLP input: **R**, ``J^{mid⊤}_L γ_f u_u``, what the write makes the
  model say; **S**, ``K^mid_L(e_u − ē)``, the footprint the write imitates.

The grid S→S, S→R, R→S, R→R, with **raw** (centred embeddings in, ``W_U diag(γ_f)`` out, as in
Elhage et al. 2021) as the lens-free reference. Per head and variant, Elhage's copying score
``Σλ/Σ|λ|`` of the map over the kept tokens (``heads.copy_score``: +1 copies what it attends to, −1
suppresses it). S→R and S→S are ``08_reads.py``'s ``ov_R_copy`` and ``ov_S_copy``; the largest
difference from ``heads/reads`` is logged. Under S→R also each head's median over tokens of its
own-column z (``heads.own_token_z``, ``08_reads.py``'s ``ov_R`` z): how broadly it copies.

Against the heads' patterns (``heads/acts``: induction, duplicate-token and previous-token heads,
score past ``CLASS``) and ``head_patterns.py``'s single-head ablation of in-context copying
(``data/heads/<model>-ablate-copy.pt``, the change in the second copy's loss):

- AUROC of each score for each pattern class; Spearman with the induction score; the mean score of
  induction, duplicate-token and non-induction heads;
- Spearman with the ablation effect, and its difference from raw's with a 95% bootstrap interval
  (``BOOT`` resamples of the heads, seeded 0); AUROC for the heads whose ablation costs over
  ``HELPS`` nats, and of the negated score for those whose ablation lowers the loss by over
  ``−SUPPRESSES`` (copy suppression, McDougall et al. 2023);
- by band (``arch.bands``): Spearman with ablation, AUROC for induction and duplicate-token heads;
  the most negative and most positive heads under the variant that tracks ablation best;
- the median z: its Spearman with the S→R score, by class and band, and at the strongest induction
  heads;
- the column reads §11 quotes (``EXAMPLES``): what a head writes under S→R when it reads one token,
  the top and bottom of ``M[:, v]``.

R inputs start at L1, so the R→ variants' statistics leave L0 out (raw's, in the bootstrap
difference, on the same heads). Needs the lens fits of record (``arch.fits``; else the released
lenses) and the ablation file; Qwen3-8B only. Summary in
``data/summary/head_circuits_<model>.{log,json}``.

    python scripts/checks/head_patterns.py    # the ablation, first
    python scripts/checks/head_circuits.py
"""
import json
import time

import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import MODEL, head_table, kept, vocab
from neuron_lens.heads import copy_score, own_token_z
from neuron_lens.paths import DATA, SUMMARY
from neuron_lens.stats import auroc, spearman
from neuron_lens.weights import tensors

STEM = arch.stem(MODEL)
ABLATE = DATA / "heads" / f"{STEM}-ablate-copy.pt"
VARIANTS = {"raw": ("raw", "raw"), "S→S": ("S", "S"), "R→S": ("R", "S"), "S→R": ("S", "R"), "R→R": ("R", "R")}  # input, output
CLASS = 0.3
HELPS, SUPPRESSES = 0.01, -0.005  # ablation effects on the second copy's loss, nats
BOOT = 2000
TOP = 8  # tokens per column read
EXAMPLES = ((26, 31, " population"), (20, 17, " herself"), (20, 17, " himself"), (19, 7, "Whilst"))  # §11


def scores(dev) -> tuple[dict, torch.Tensor, list[str]]:
    """Per variant the copying score ``[L, H]`` (NaN where the input lens has no layer), the median
    own-column z under S→R ``[L, H]``, and the ``EXAMPLES``' column reads."""
    geo = arch.layout(MODEL)
    L, H = len(geo), geo[0].q_heads
    keep = kept()
    V = len(keep)
    voc = vocab()
    row = {t: next(r for r, i in enumerate(keep.tolist()) if voc[i] == t) for _, _, t in EXAMPLES}
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted(set(nm.values())))
    E = head[nm["embed"]][keep].float() * arch.embed_scale(MODEL)
    E = (E - E.mean(0)).to(dev)
    Wu = (head[nm["unembed"]][keep].float() * head[nm["norm"]].float()).to(dev)
    del head
    lens = arch.lens_means(MODEL, *arch.fits(MODEL))
    K_out, K_mid, J_out, J_mid = (lens[k] for k in ("K_out", "K_mid", "J_out", "J_mid"))  # J_out: [L − 1, d, d]
    copy = {v: torch.full((L, H), torch.nan) for v in VARIANTS}
    median = torch.zeros(L, H)
    cols = []
    for layer in range(L):
        lay = geo[layer]
        g = H // lay.kv_heads
        Wv, Wo = arch.ov(MODEL, lay, dev)  # W_V γ₁ [kv·hd, d], W_O [d, H·hd]
        X = {"raw": E, "S": E @ K_out[layer - 1].to(dev).T if layer else E}  # [V, d]: each input token
        if layer:
            X["R"] = Wu @ J_out[layer - 1].to(dev)
        B = {k: (x @ Wv.T).view(V, lay.kv_heads, lay.head_dim).repeat_interleave(g, dim=1) for k, x in X.items()}
        del X
        A = {"raw": Wu @ Wo, "S": E @ (K_mid[layer].to(dev).T @ Wo), "R": Wu @ (J_mid[layer].to(dev) @ Wo)}
        A = {k: a.view(V, H, lay.head_dim) for k, a in A.items()}  # each output token through W_O^h
        for name, (i, o) in VARIANTS.items():
            if i in B:
                copy[name][layer] = copy_score(A[o], B[i])
        median[layer] = own_token_z(A["R"], B["S"]).median(0).values.cpu()
        for l, h, t in EXAMPLES:
            if l == layer:
                col = A["R"][:, h] @ B["S"][row[t], h]  # [V]: M[:, v], what the head writes reading t
                ends = [" | ".join(voc[i] for i in keep[x.topk(TOP).indices.cpu()].tolist()) for x in (col, -col)]
                cols.append(f"  L{l}h{h} reads {t!r} → writes {ends[0]} ; suppresses {ends[1]}")
        del A, B
        print(f"{time.strftime('%H:%M:%S')} L{layer}: " + "  ".join(f"{v} {copy[v][layer].mean():+.2f}" for v in VARIANTS),
              flush=True)
    return copy, median, cols


def report(copy: dict, median: torch.Tensor, cols: list[str], abl: torch.Tensor) -> tuple[list[str], dict]:
    L, H = abl.shape
    acts = pq.read_table(head_table("acts"), columns=["induction", "duplicate", "prev"]).to_pydict()
    ind, dup, prev = (torch.tensor(acts[k]) for k in ("induction", "duplicate", "prev"))
    reads = pq.read_table(head_table("reads"), columns=["ov_R_copy", "ov_S_copy"]).to_pydict()
    d_R = (copy["S→R"].flatten() - torch.tensor(reads["ov_R_copy"])).abs().max().item()
    d_S = (copy["S→S"].flatten() - torch.tensor(reads["ov_S_copy"])).abs().max().item()
    abl, median = abl.flatten(), median.flatten()
    lay = torch.arange(L).repeat_interleave(H)
    pi, pd, pp = ind > CLASS, dup > CLASS, prev > CLASS
    hl = lambda i: f"L{i // H}h{i % H}"  # noqa: E731
    lines = [(f"attention heads' OV copying scores (Σλ/Σ|λ| of the token map), input at layer L−1's output and output "
              f"at layer L's MLP input, {MODEL}, {len(kept())} kept tokens, lenses {' and '.join(arch.fits(MODEL))}; "
              f"induction heads: {int(pi.sum())}, duplicate-token: {int(pd.sum())}, previous-token: {int(pp.sum())} "
              f"(score past {CLASS}, heads/acts)"),
             (f"ablation (head_patterns.py, change in the second copy's loss): largest {abl.max():+.3f} nats; heads past "
              f"{HELPS}: {int((abl > HELPS).sum())}, below {SUPPRESSES}: {int((abl < SUPPRESSES).sum())}. R inputs start at L1."),
             f"S→R and S→S against heads/reads' ov_R_copy and ov_S_copy: max |Δ| {d_R:.1e} / {d_S:.1e}", "",
             ("variant  AUROC induction  AUROC duplicate  AUROC previous  Spearman with induction  "
              "mean copy (induction / duplicate / not induction)  Spearman with ablation (minus raw, 95% bootstrap)  "
              "AUROC helps  AUROC suppresses (−copy)")]
    table = {"catalog_max_diff": {"S→R": d_R, "S→S": d_S}}
    boot = torch.randint(len(abl), (BOOT, len(abl)), generator=torch.Generator().manual_seed(0))
    raw = copy["raw"].flatten()
    for v in VARIANTS:
        c = copy[v].flatten()
        ok = ~c.isnan()  # the R inputs have no L0
        diff = torch.tensor([spearman(c[i][ok[i]], abl[i][ok[i]]) - spearman(raw[i][ok[i]], abl[i][ok[i]]) for i in boot])
        row = {"auroc_induction": auroc(c[ok], pi[ok]), "auroc_duplicate": auroc(c[ok], pd[ok]),
               "auroc_previous": auroc(c[ok], pp[ok]), "spearman_induction": spearman(c[ok], ind[ok]),
               "mean_induction": c[ok & pi].mean().item(), "mean_duplicate": c[ok & pd].mean().item(),
               "mean_other": c[ok & ~pi].mean().item(), "spearman_ablation": spearman(c[ok], abl[ok]),
               "spearman_ablation_minus_raw": [diff.mean().item(), diff.quantile(0.025).item(), diff.quantile(0.975).item()],
               "auroc_helps": auroc(c[ok], abl[ok] > HELPS), "auroc_suppresses": auroc(-c[ok], abl[ok] < SUPPRESSES),
               "bands": {}}
        for lo, hi in arch.bands(L):
            b = ok & (lay >= lo) & (lay <= hi)
            row["bands"][f"L{lo}-L{hi}"] = {"spearman_ablation": spearman(c[b], abl[b]), "auroc_induction": auroc(c[b], pi[b]),
                                            "auroc_duplicate": auroc(c[b], pd[b])}
        table[v] = row
        d = row["spearman_ablation_minus_raw"]
        lines.append(f"{v:7s}  {row['auroc_induction']:.2f}             {row['auroc_duplicate']:.2f}             "
                     f"{row['auroc_previous']:.2f}            {row['spearman_induction']:+.2f}                    "
                     f"{row['mean_induction']:+.2f} / {row['mean_duplicate']:+.2f} / {row['mean_other']:+.2f}"
                     f"                         {row['spearman_ablation']:+.2f} ({d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}])"
                     f"                  {row['auroc_helps']:.2f}         {row['auroc_suppresses']:.2f}")
    lines += ["", "by band: Spearman with ablation / AUROC induction / AUROC duplicate"]
    for lo, hi in arch.bands(L):
        k = f"L{lo}-L{hi}"
        lines.append(f"  {k}: " + "   ".join(f"{v} {table[v]['bands'][k]['spearman_ablation']:+.2f}/"
                                             f"{table[v]['bands'][k]['auroc_induction']:.2f}/"
                                             f"{table[v]['bands'][k]['auroc_duplicate']:.2f}" for v in VARIANTS))
    best = max(VARIANTS, key=lambda v: table[v]["spearman_ablation"])
    c = copy[best].flatten()
    order = c.nan_to_num(0).argsort()
    lines += ["", f"most negative heads under {best}, the variant that tracks ablation best (copy suppression candidates): "
              + ", ".join(f"{hl(i)} {c[i]:+.2f} (ind {ind[i]:.2f})" for i in order[:10].tolist()),
              "most positive: " + ", ".join(f"{hl(i)} {c[i]:+.2f} (ind {ind[i]:.2f})" for i in order.flip(0)[:10].tolist())]

    qs = torch.tensor([0.05, 0.5, 0.95])
    strongest = ind.argsort(descending=True)[:5].tolist()
    table["median_z"] = {"spearman_copy_S_R": spearman(median, copy["S→R"].flatten()),
                         "classes": {k: median[m].median().item() for k, m in
                                     (("induction", pi), ("duplicate", pd), ("other", ~pi & ~pd))},
                         "strongest_induction": [[hl(i), median[i].item()] for i in strongest], "bands": {}}
    mz = table["median_z"]
    lines += ["", f"S→R per token (each token's z in its own column, 08_reads.py's ov_R z): the median over tokens tracks "
              f"copy S→R at Spearman {mz['spearman_copy_S_R']:+.2f}; median over heads: "
              + ", ".join(f"{k} {x:+.2f}" for k, x in mz["classes"].items()),
              "  the strongest induction heads: " + ", ".join(f"{h} {x:+.2f}" for h, x in mz["strongest_induction"]),
              "  by band (5% / 50% / 95% of heads):"]
    for lo, hi in arch.bands(L):
        q = median[(lay >= lo) & (lay <= hi)].quantile(qs).tolist()
        mz["bands"][f"L{lo}-L{hi}"] = q
        lines.append(f"    L{lo}-L{hi}: " + " / ".join(f"{x:+.2f}" for x in q))
    lines += ["", "column reads quoted in findings §11 (S→R: what the head writes when it reads the token):"] + cols
    table["columns"] = cols
    return lines, table


def main() -> None:
    if MODEL != arch.QWEN:
        raise SystemExit("head_circuits.py reads Qwen3-8B only: Gemma 4's OV maps also need its value norm's derivative "
                         "(06_activations.py's statistics, as 08_reads.py applies them), and the examples are Qwen3-8B's heads")
    if not ABLATE.exists():
        raise SystemExit(f"{ABLATE} is missing: run scripts/checks/head_patterns.py first")
    abl = torch.load(ABLATE, weights_only=False)["ablate_copy"]
    dev = "mps" if torch.backends.mps.is_available() else "cuda"
    copy, median, cols = scores(dev)
    lines, table = report(copy, median, cols, abl)
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"head_circuits_{STEM}.log").write_text(text + "\n")
    (SUMMARY / f"head_circuits_{STEM}.json").write_text(json.dumps(table))


if __name__ == "__main__":
    main()
