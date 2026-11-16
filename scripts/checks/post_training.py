"""What post-training changed in a Qwen3 model, read through the lenses (findings §19).

The post-trained model (``NEURON_LENS_MODEL``, a Qwen3) against its base (``<model>-Base``), weights only; post-training
keeps the neurons' indices. Both models' lenses of record (``arch.fits``: local fits, else the release), and for both the
post-trained model's catalog: lobe signs, near-silent poles (``act_top`` under 0.05) and drivers; with ``--base-drivers``
the base's drivers too (its catalog if local, else the Hub dataset).

- ``weights``: per layer, each matrix's relative change ``|ΔW| / |W_base|`` (Frobenius); per neuron, the cosine of its
  write and of its read rows between the models, and the overlap of its write's R-lens top 10 (each model's write
  through its own lens); ``ΔW_down``'s participation ratio (of its squared singular values) and its top directions read
  through the post-trained model's R-lens, z-scored against the layer's writes (a direction's sign is arbitrary: both
  ends).
- ``reads``: ``read_R`` (``reads_at_firings.py``: a pole's read direction read by the R-lens from the MLP input, scored as
  the mean z over its drivers) under six conditions: each model's read rows through its own R-lens; each model's rows
  through the other's; the post-trained rows through the base's Jacobian with the post-trained readout (unembedding and
  final norm); and through the identity (the logit lens). Two nulls. ``pooled``, ``reads_at_firings.py``'s: one
  mismatched partner per pole, the layer's partners pooled, BH at 5%. ``per pole``: a pole's score on its own drivers
  against its score on 16 other poles' (from its half, + or −, with enough drivers), passing when it beats all 16 (1/17
  by chance). And ``tail``: the share of the pooled null's 99%-point exceedances (each pole on its 16 partners) that come
  from the 1% of poles with the highest mean score on others' drivers.

    NEURON_LENS_MODEL=Qwen/Qwen3-8B python scripts/checks/post_training.py [--base-drivers] [--parts weights,reads]

Writes ``data/summary/post_training_<model>.{json,log}``.
"""
import argparse
import json
import time

import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import (
    MODEL,
    column,
    kept,
    kept_index,
    lobe_signs,
    neuron_table,
    tokenizer,
)
from neuron_lens.neurons import poles
from neuron_lens.paths import SUMMARY
from neuron_lens.readout import (
    layer_logit_stats,
    layer_reads,
    read_z,
    rms_norm_columns,
    transport,
)
from neuron_lens.stats import bh, null_p
from neuron_lens.weights import tensors

POST, BASE = MODEL, MODEL + "-Base"
STEM = arch.stem(POST)
CFG = arch.text_config(POST)
L, F = CFG.num_hidden_layers, CFG.intermediate_size
QUARTERS = [(i * (L // 4), (i + 1) * (L // 4) - 1 if i < 3 else L - 1) for i in range(4)]
MATS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
        "mlp.down_proj", "input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm")
SILENT, FDR, MIN_TOKENS = 0.05, 0.05, 3
PARTNERS, N_DIRS, CHUNK = 16, 4, 1024
# condition: (whose read rows, whose Jacobian (None: the identity), whose readout)
CONDITIONS = {"post": ("post", "post", "post"), "base": ("base", "base", "base"),
              "post rows, base lens": ("post", "base", "base"), "base rows, post lens": ("base", "post", "post"),
              "base J, post readout": ("post", "base", "post"), "identity": ("post", None, "post")}
KEEP, KIDX = kept(), kept_index()
TOK = tokenizer()
ID = {"post": POST, "base": BASE}


def readout(model: str, dev) -> tuple[torch.Tensor, torch.Tensor]:
    nm = arch.names(model)
    w = tensors(model, sorted({nm["unembed"], nm["norm"]}))
    return w[nm["unembed"]][KEEP].to(dev, torch.float16), w[nm["norm"]].to(dev, torch.float32)


def words(ids) -> list[str]:
    return [TOK.decode([int(KEEP[i])]) for i in ids]


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def quantiles(t: torch.Tensor) -> list[float]:
    return [round(x, 4) for x in torch.quantile(t.float().cpu(), torch.tensor([0.1, 0.5, 0.9])).tolist()]


def weights(dev) -> tuple[list[str], dict]:
    ro = {m: readout(ID[m], dev) for m in ID}
    J = {m: arch.lens(ID[m], "R_out", arch.fits(ID[m])[0]) for m in ID}
    names = arch.names(POST)
    emb = {m: tensors(ID[m], [names["embed"]])[names["embed"]] for m in ID}
    out = {"global": {"embed": rel(emb["post"], emb["base"]), "unembed": rel(ro["post"][0], ro["base"][0]),
                      "final norm": rel(ro["post"][1], ro["base"][1])}, "layers": []}
    del emb
    for layer in range(L):
        t0 = time.perf_counter()
        prefix = arch.layout(POST)[layer].prefix
        wt = {m: tensors(ID[m], [prefix + x + ".weight" for x in MATS]) for m in ID}
        row = {"layer": layer, "rel": {x.split(".")[-1]: rel(wt["post"][prefix + x + ".weight"], wt["base"][prefix + x + ".weight"])
                                       for x in MATS}}
        del wt
        p = {m: poles(ID[m], layer, lobe_signs(layer), dev) for m in ID}
        d = p["post"]["write"] - p["base"]["write"]
        cr = (p["post"]["cov"] * p["base"]["cov"]).sum(1) / (p["post"]["cov"].norm(dim=1) * p["base"]["cov"].norm(dim=1))
        row |= {"cos_write": quantiles(torch.nn.functional.cosine_similarity(p["post"]["write"], p["base"]["write"], dim=0)),
                "rel_write": quantiles(d.norm(dim=0) / p["base"]["write"].norm(dim=0)), "cos_read": quantiles(cr)}
        Jl = {m: J[m][layer].to(dev) if layer < len(J[m]) else None for m in ID}
        top = {m: layer_reads(p[m]["write"], *ro[m], Jl[m], signs=(1,), k=10, kz=1)["ids"][1] for m in ID}
        row["write_R_top10_kept"] = quantiles(torch.tensor([len(set(a.tolist()) & set(b.tolist())) / 10
                                                            for a, b in zip(top["post"], top["base"])]))
        lam, U = torch.linalg.eigh((d @ d.T).cpu().double())
        lam, U = lam.flip(0).clamp_min(0), U.flip(1).float()
        row["dW_participation"] = (lam.sum() ** 2 / lam.pow(2).sum()).item()
        mean, std = layer_logit_stats(torch.cat([p["post"]["write"], -p["post"]["write"]], 1), *ro["post"], Jl["post"])
        V = U[:, :N_DIRS].to(dev)
        z = read_z(torch.cat([V, -V], 1), *ro["post"], Jl["post"], mean, std, k=10)
        row["dW_dirs"] = [{"share": (lam[k] / lam.sum()).item(), "one_end": words(z[k]), "other_end": words(z[N_DIRS + k])}
                          for k in range(N_DIRS)]
        out["layers"].append(row)
        r = row["rel"]
        print(f"weights L{layer:2d}: rel q {r['q_proj']:.3f} o {r['o_proj']:.3f} gate {r['gate_proj']:.3f} down {r['down_proj']:.3f}; "
              f"cos write {row['cos_write'][1]:.4f} read {row['cos_read'][1]:.4f}; write_R top 10 kept {row['write_R_top10_kept'][1]:.1f}; "
              f"ΔW_down participation {row['dW_participation']:.0f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        del p, d, Jl, V
    lines = [f"weights, {POST} against {BASE}: global relative change " +
             ", ".join(f"{k} {v:.3f}" for k, v in out["global"].items()), "",
             "layers    q     k     v     o     gate  up    down  in-norm  cos write  cos read  write_R top 10 kept  ΔW_down PR  top direction"]
    for row in out["layers"]:
        r = row["rel"]
        lines.append(f"L{row['layer']:<4d} " + " ".join(f"{r[k]:.3f}" for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")) +
                     f"  {r['input_layernorm']:.3f}    {row['cos_write'][1]:.4f}     {row['cos_read'][1]:.4f}    {row['write_R_top10_kept'][1]:.1f}"
                     f"                  {row['dW_participation']:6.0f}      {row['dW_dirs'][0]['share']:.4f}")
    lines += ["", "ΔW_down's top directions (z through the post-trained model's R-lens; both ends):"]
    for row in out["layers"]:
        for k, x in enumerate(row["dW_dirs"][:2]):
            lines.append(f"L{row['layer']} #{k + 1} ({x['share']:.4f}): {' '.join(map(repr, x['one_end'][:8]))}  |  {' '.join(map(repr, x['other_end'][:8]))}")
    return lines, out


def drivers_table(model: str, layer: int):
    local = neuron_table("drivers", layer, model)
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download("stdlogits/neuron-lens", f"{arch.stem(model)}/neurons/drivers/L{layer:02d}.parquet", repo_type="dataset")


def score(rr: torch.Tensor, un: torch.Tensor, tok: torch.Tensor, ok: torch.Tensor, partners: torch.Tensor):
    """Each row's mean z over its own tokens and over each partner's (a token's score under a row z-scored across
    the rows, as ``reads_at_firings.mean_z``): ``own`` [n] and ``cross`` [n, P], NaN with fewer than MIN_TOKENS; and the
    tokens' mean and std across the rows."""
    n, dev = len(rr), rr.device
    s1, s2 = torch.zeros(len(un), device=dev), torch.zeros(len(un), device=dev)
    for s in range(0, n, CHUNK):
        sc = (rr[s:s + CHUNK] @ un.T).float()
        s1 += sc.sum(0)
        s2 += sc.pow(2).sum(0)
    mu = s1 / n
    sd = (s2 / n - mu.pow(2)).clamp_min(1e-12).sqrt()
    t, o = tok.clamp_min(0), ok.float()
    cnt = o.sum(1)
    own, cross = torch.empty(n), torch.empty(n, partners.shape[1])
    for s in range(0, n, CHUNK):
        idx = torch.arange(s, min(s + CHUNK, n))
        z = ((rr[s:s + CHUNK] @ un.T).float() - mu) / sd
        for col, rows in [(None, idx)] + [(p, partners[idx, p]) for p in range(partners.shape[1])]:
            g = (z.gather(1, t[rows].to(dev)) * o[rows].to(dev)).sum(1).cpu() / cnt[rows].clamp_min(1)
            g = torch.where(cnt[rows] >= MIN_TOKENS, g, torch.nan)
            if col is None:
                own[idx] = g
            else:
                cross[idx, col] = g
    return own, cross, mu, sd


def reads(dev, base_drivers: bool) -> tuple[list[str], dict]:
    ro = {m: readout(ID[m], dev) for m in ID}
    J = {m: arch.lens(ID[m], "R_mid", arch.fits(ID[m])[0]) for m in ID}
    targets = {"post": POST} | ({"base": BASE} if base_drivers else {})
    out = {"layers": []}
    for layer in range(L):
        t0 = time.perf_counter()
        g = torch.Generator().manual_seed(layer)
        silent = torch.tensor(pq.read_table(neuron_table("acts", layer), columns=["act_top"]).column(0).to_numpy()) < SILENT
        vec = {m: poles(ID[m], layer, lobe_signs(layer), dev)["vec"].T for m in ID}
        row = {"layer": layer}
        for tn, tm in targets.items():
            drv = column(pq.read_table(drivers_table(tm, layer), columns=["drivers_ids"]), "drivers_ids").long()
            tok = torch.where(drv >= 0, KIDX[drv.clamp_min(0)], -1)
            ok = tok >= 0
            usable = (ok.sum(1) >= MIN_TOKENS) & ~silent
            partners = torch.empty(2 * F, PARTNERS, dtype=torch.long)
            for h in (0, 1):  # partners from the pole's own half (+ or −), with enough drivers, never itself
                pool = torch.arange(h * F, (h + 1) * F)[usable[h * F:(h + 1) * F]]
                pick = pool[torch.randint(len(pool), (F, PARTNERS), generator=g)]
                self_ = pick == torch.arange(h * F, (h + 1) * F)[:, None]
                pick[self_] = pool[(torch.searchsorted(pool, pick[self_]) + 1) % len(pool)]
                partners[h * F:(h + 1) * F] = pick
            for cn, (rows, jm, rm) in CONDITIONS.items():
                rr = rms_norm_columns(transport(vec[rows], None if jm is None else J[jm][layer]), ro[rm][1]).T.half()
                own, cross, mu, sd = score(rr, ro[rm][0], tok, ok, partners)
                use = usable & ~own.isnan()
                q = torch.ones(2 * F)
                q[use] = bh(null_p(own[use], cross[:, 0][usable]))
                c = cross[use]
                q99 = c.flatten().quantile(0.99).item()
                hits = (c > q99).float().sum(1)
                top = hits[c.mean(1).argsort(descending=True)[:max(1, len(c) // 100)]].sum() / hits.sum().clamp_min(1)
                promiscuous = use.nonzero().flatten()[c.mean(1).argsort(descending=True)[:3]]
                row[f"{cn} | {tn} drivers"] = {
                    "pooled": (q <= FDR).float().mean().item(),
                    "per_pole": (use & (own > cross.nan_to_num(-torch.inf).max(1).values)).float().mean().item(),
                    "own_median": own[use].median().item(), "null_q99": q99, "tail": top.item(),
                    "promiscuous": [{"pole": int(i), "read": words((((rr[i:i + 1] @ ro[rm][0].T).float() - mu) / sd).topk(8).indices[0].cpu())}
                                    for i in promiscuous.tolist()] if cn in ("post", "base") else []}
        out["layers"].append(row)
        x = {k: v for k, v in row.items() if k.endswith("| post drivers")}
        print(f"reads L{layer:2d} (post drivers) pooled / per pole: " + " | ".join(
            f"{k.split(' |')[0]} {v['pooled']:.2f}/{v['per_pole']:.2f}" for k, v in x.items()) + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
        del vec
    lines = [(f"reads, {POST} against {BASE}: read_R's share of poles passing, by quarter of depth (mean over layers). "
             "pooled: reads_at_firings.py's test (BH at 5% over a null pooled across the layer); per pole: beats its own "
             f"score on all {PARTNERS} partners' drivers (chance 1/{PARTNERS + 1}); tail: the share of the pooled null's 99%-point "
             "exceedances from the top 1% of poles by mean score on others' drivers; null q99 that point.")]
    for tn in targets:
        lines += ["", f"on the {tn} model's drivers", f"{'condition':24s} " + "  ".join(f"{f'L{a}-{b}':>26s}" for a, b in QUARTERS)]
        for cn in CONDITIONS:
            k = f"{cn} | {tn} drivers"
            cells = []
            for a, b in QUARTERS:
                rs = [r[k] for r in out["layers"] if a <= r["layer"] <= b]
                m = {key: sum(r[key] for r in rs) / len(rs) for key in ("pooled", "per_pole", "null_q99", "tail")}
                cells.append(f"{m['pooled']:.2f} / {m['per_pole']:.2f} q99 {m['null_q99']:.2f} tail {m['tail']:.2f}".rjust(26))
            lines.append(f"{cn:24s} " + "  ".join(cells))
    lines += ["", "the most promiscuous poles' reads (highest mean score on others' drivers), post-trained drivers:"]
    for r in out["layers"]:
        if r["layer"] % 4 == 0 or r["layer"] == L - 1:
            for cn in ("post", "base"):
                lines.append(f"L{r['layer']} {cn}: " + " || ".join(" ".join(map(repr, e["read"][:6])) for e in r[f"{cn} | post drivers"]["promiscuous"]))
    return lines, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="weights,reads")
    ap.add_argument("--base-drivers", action="store_true")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
    args = ap.parse_args()
    path = SUMMARY / f"post_training_{STEM}.json"
    table = json.loads(path.read_text()) if path.exists() else {}
    for part in args.parts.split(","):
        lines, table[part] = {"weights": weights, "reads": lambda d: reads(d, args.base_drivers)}[part](args.device)
        table[f"{part}_log"] = lines
        text = "\n\n".join("\n".join(table[f"{p}_log"]) for p in ("weights", "reads") if f"{p}_log" in table)
        (SUMMARY / f"post_training_{STEM}.log").write_text(text + "\n")
        path.write_text(json.dumps(table, ensure_ascii=False))
    print(text)


if __name__ == "__main__":
    main()
