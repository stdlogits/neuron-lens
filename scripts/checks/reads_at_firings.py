"""How often each weight-only read names what is counted at a unit's firings and drivers, by layer and band (findings §7,
§12, §13, §18).

Each test scores a read on a list of tokens as a mean z, against mismatched pairs: another unit of the layer scored on this
unit's tokens. A unit's p-value on a test is the share of its layer's mismatched pairs that score as high; the summary gives
the share of each layer's units that pass at a false discovery rate of 5% (Benjamini-Hochberg over the layer's units for
neurons, over all heads for heads) and the share past the 99% point of mismatched pairs net of theirs. These are population
results about the reads, not verdicts on units: a firing is one context summarized by a token, and the targets are narrow
(the token under a firing, the token after it), so a read that misses them isn't wrong (§7), and the catalog keeps no
per-unit score. Units with fewer than 3 usable tokens, and near-silent poles (``acts``' ``act_top`` under 0.05), are not
scored.

**Neurons**, after ``07_drivers.py`` and ``08_reads.py``, z of each token across the layer's poles:

- ``read_S``: the read direction's S-lens read names the peak tokens of the top firings (``06_activations.py``: the strongest
  position in each of the pole's top documents);
- ``read_R``: its R-lens read (from the MLP input) names the pole's exact drivers (``07_drivers.py``, token swaps). An R-lens read
  reads forward, so it isn't scored on the token under a firing;
- ``write_R``: the write's R-lens read promotes the tokens that follow the firings;
- ``write_S``: the write's S-lens read names the peak tokens (early neurons write back what they detect);
- ``write_orth_R``, ``write_orth_S``: the same of the write orthogonal to its neuron's gate and up rows (§18);
- the shares of near-silent poles, and of poles with a third of their top firings on one peak token or before one token
  (``acts``' ``cur_mode``, ``next_mode``).

**Heads**, each read's per-token z from ``08_reads.py`` (``data/heads/<model>-token-z.pt``), at the token the
firing attends to most (past the sink):

- ``ov_R``: the OV column of the attended token through the R-lens (what the head writes when it
  reads it) promotes the token that follows the query;
- ``ov_copy``: the attended token's z in its own ``ov_R`` column: the head promotes what it reads;
- ``ov_S``: the same through the S-lens: the write carries the attended token's footprint;
- ``qk_S``: the attended token's ``qk_S`` z: the QK read predicts what the head attends to;
- the shares with a third of firings on one query or attended token (``acts``' ``cur_mode``, ``src_mode``), and with an
  attention pattern score past a threshold (``prev`` or ``self`` past 0.5, ``induction`` or ``duplicate`` past 0.3);
- ``<test>_p05``: the share with an uncorrected p of 0.05 or less (5% by chance). A layer's H(H − 1) pairs bound p
  below by 1/(H(H − 1) + 1): with 16 heads a layer (Gemma 4), BH over all heads passes only if dozens sit at that
  floor, though a head can clearly stand out of its layer.

Per layer and band into ``data/summary/reads_at_firings_<model>.{log,json}``. Each unit's q-values (and the heads'
uncorrected p) go to ``data/checks/reads_at_firings-<model>.pt``, unreleased, for the checks that count a subgroup of units
(``library_crosscheck.py``'s coverage).

Gemma 4 12B (``NEURON_LENS_MODEL=google/gemma-4-12B``): the released lenses; each neuron's write as
the residual sees it and its read rows times the pre-feedforward norm (``neuron_lens/neurons.py``);
embeddings × √d; each head's ``W_O`` times the post-attention norm weight, keys as values on the full
layers, and the values through ``v_norm``'s averaged true derivative, as ``08_reads.py`` reads them.

    python scripts/checks/reads_at_firings.py [--parts neurons,heads]
"""
import argparse
import json
import time

import pyarrow.parquet as pq
import torch

from neuron_lens import arch, metric
from neuron_lens.catalog import (
    MODEL,
    acts_path,
    column,
    firing_tokens,
    head_table,
    kept,
    kept_index,
    lobe_signs,
    neuron_table,
)
from neuron_lens.heads import norm_derivative
from neuron_lens.neurons import orthogonal_write, poles
from neuron_lens.paths import DATA, SUMMARY
from neuron_lens.readout import rms_norm_columns, transport
from neuron_lens.stats import bh, null_p
from neuron_lens.weights import tensors

R_FIT, S_FIT = arch.fits(MODEL)
STEM = MODEL.split("/")[-1].lower()
ACTS = acts_path()
TOKEN_Z = DATA / "heads" / f"{STEM}-token-z.pt"
UNITS = DATA / "checks" / f"reads_at_firings-{STEM}.pt"
PAST = 1 / 3
SILENT = 0.05
MIN_FIRINGS = 3
FDR = 0.05
NULL_Q = 0.99
CHUNK = 2048
BANDS = tuple(arch.bands(arch.text_config(MODEL).num_hidden_layers))  # Qwen3-8B's five, else quarters
TESTS = ("read_R", "read_S", "write_R", "write_S", "write_orth_R", "write_orth_S")
HEAD_TESTS = ("ov_R", "ov_copy", "ov_S", "qk_S")
PATTERN = {"prev": 0.5, "self": 0.5, "induction": 0.3, "duplicate": 0.3}


def mean_z(vecs: torch.Tensor, table: torch.Tensor, tok: torch.Tensor, ok: torch.Tensor,
           perm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's mean z over its usable tokens, where a token's score under a row is
    ``table[token] · vecs[row]``, z-scored per token across the rows; and the same for the rows in
    ``perm`` order, scored on each row's own tokens (the mismatched pairs)."""
    n, dev = len(vecs), vecs.device
    s1, s2 = torch.zeros(len(table), device=dev), torch.zeros(len(table), device=dev)
    for s in range(0, n, CHUNK):
        sc = (vecs[s:s + CHUNK] @ table.T).float()
        s1 += sc.sum(0)
        s2 += sc.pow(2).sum(0)
    mean = s1 / n
    std = (s2 / n - mean.pow(2)).clamp_min(1e-12).sqrt()
    t = tok.clamp_min(0)
    out = []
    for v in (vecs, vecs[perm]):
        z = torch.empty(tok.shape)
        for s in range(0, n, CHUNK):
            tt = t[s:s + CHUNK].to(dev)
            g = torch.einsum("ckd,cd->ck", table[tt], v[s:s + CHUNK]).float()
            z[s:s + CHUNK] = ((g - mean[tt]) / std[tt]).cpu()
        stat = (z * ok).sum(1) / ok.sum(1).clamp_min(1)
        out.append(torch.where(ok.sum(1) >= MIN_FIRINGS, stat, torch.nan))
    return out[0], out[1]


def band_rows(per: list[dict], L: int) -> dict:
    out = {}
    for lo, hi in BANDS + ((0, L - 1),):
        rows = [r for r in per if lo <= r["layer"] <= hi]
        keys = {k for r in rows for k in r if k != "layer"}
        out[f"L{lo}-{hi}"] = {k: sum(r[k] for r in rows if k in r) / max(1, sum(k in r for r in rows)) for k in keys}
    return out


def neurons(saved: dict, dev) -> tuple[list[str], dict]:
    cfg = arch.text_config(MODEL)
    L, f = cfg.num_hidden_layers, cfg.intermediate_size
    keep, kidx = kept(), kept_index()
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted(set(nm.values())))
    unembed = head[nm["unembed"]][keep].to(dev, torch.float16)
    gamma = head[nm["norm"]].to(dev, torch.float32)
    emb = head[nm["embed"]][keep].float() * arch.embed_scale(MODEL)
    emb = (emb - emb.mean(0)).to(dev, torch.float16)
    del head
    J_out, J_mid = arch.lens(MODEL, "R_out", R_FIT), arch.lens(MODEL, "R_mid", R_FIT)
    K_mid, K_out = arch.lens(MODEL, "S_mid", S_FIT), arch.lens(MODEL, "S_out", S_FIT)
    top = saved["state"]["top"]
    g = torch.Generator().manual_seed(0)
    per, unit_q = [], {}
    for layer in range(L):
        t0 = time.perf_counter()
        val = torch.cat([top["val"][layer, p].T for p in (0, 1)]).float()  # [2f, K]: + poles, then −
        doc = torch.cat([top["doc"][layer, p].T for p in (0, 1)]).long()
        pos = torch.cat([top["pos"][layer, p].T for p in (0, 1)]).long()
        fired = val > 0
        flat, start = saved["tokens"], saved["doc_start"]
        cur, nxt = firing_tokens(flat, start, doc, pos).long(), firing_tokens(flat, start, doc, pos, 1).long()
        acts = pq.read_table(neuron_table("acts", layer), columns=["act_top", "cur_mode", "next_mode"]).to_pydict()
        silent = torch.tensor(acts["act_top"]) < SILENT
        cur_mode, next_mode = torch.tensor(acts["cur_mode"]), torch.tensor(acts["next_mode"])
        drivers = column(pq.read_table(neuron_table("drivers", layer), columns=["drivers_ids"]), "drivers_ids").long()
        k_cur, k_nxt, k_drv = (torch.where(t >= 0, kidx[t.clamp_min(0)], -1) for t in (cur, nxt, drivers))
        ok_cur, ok_nxt = fired & (k_cur >= 0), fired & (k_nxt >= 0)
        perm = torch.cat([torch.randperm(f, generator=g), f + torch.randperm(f, generator=g)])
        p = poles(MODEL, layer, lobe_signs(layer), dev)  # as 08_reads.py, in the residual's metric
        wd = p["write"]
        r = rms_norm_columns(transport(wd, J_out[layer] if layer < len(J_out) else None), gamma).T  # [f, d]
        K = K_out[layer].to(dev)
        PK = metric.polar(K) @ K  # as 08_reads.py: writes against the footprints' rotation
        sw = wd.T @ PK
        sw = sw / sw.norm(dim=1, keepdim=True)
        orth = orthogonal_write(wd, p["a"], p["b"], p["s2"])
        ro = rms_norm_columns(transport(orth, J_out[layer] if layer < len(J_out) else None), gamma).T
        so = orth.T @ PK
        so = so / so.norm(dim=1, keepdim=True)
        c = p["cov"]
        rr = rms_norm_columns(transport(p["vec"].T, J_mid[layer]), gamma).T
        cs = c @ K_mid[layer].to(dev)
        del p, K, PK
        stats = {"write_R": mean_z(torch.cat([r, -r]).half(), unembed, k_nxt, ok_nxt, perm),
                 "write_S": mean_z(torch.cat([sw, -sw]).half(), emb, k_cur, ok_cur, perm),
                 "read_R": mean_z(rr.half(), unembed, k_drv, k_drv >= 0, perm),
                 "read_S": mean_z((cs / cs.norm(dim=1, keepdim=True)).half(), emb, k_cur, ok_cur, perm),
                 "write_orth_R": mean_z(torch.cat([ro, -ro]).half(), unembed, k_nxt, ok_nxt, perm),
                 "write_orth_S": mean_z(torch.cat([so, -so]).half(), emb, k_cur, ok_cur, perm)}
        row = {"layer": layer, "silent": silent.float().mean().item(),
               "token_detector": ((cur_mode >= PAST) & ~silent).float().mean().item(),
               "next_detector": ((next_mode >= PAST) & ~silent).float().mean().item()}
        for test, (m, n) in stats.items():
            ok = ~m.isnan() & ~silent
            q = torch.ones(2 * f)
            q[ok] = bh(null_p(m[ok], n))
            unit_q.setdefault(test, []).append(q)
            thr = n[~n.isnan()].quantile(NULL_Q).item()
            row[test] = (q <= FDR).float().mean().item()
            row[f"{test}_net"] = (((m > thr) & ~silent).float().mean() - ((n > thr) & ~silent).float().mean()).item()
        per.append(row)
        print(f"{time.strftime('%H:%M:%S')} neurons L{layer}: " + " ".join(f"{k} {row[k]:.2f}" for k in
              ("silent", *TESTS)) + f" in {time.perf_counter() - t0:.0f}s", flush=True)
    keys = ("silent", "token_detector", "next_detector", *TESTS)
    lines = [(f"neurons, {MODEL}: {2 * f * L:,} poles; share of each band's poles (mean over layers) passing each test at "
              f"a {FDR:.0%} false discovery rate; in brackets, the share past the {NULL_Q:.0%} point of mismatched pairs net "
              "of theirs. Detectors: a third of top firings share a token."), "",
             "band     " + "  ".join(f"{k:>14s}" for k in keys)]
    bands = band_rows(per, L)
    for name, r in bands.items():
        lines.append(f"{name:8s} " + "  ".join((f"{r[k]:.2f} [{r[k + '_net']:.2f}]" if k + "_net" in r else f"{r[k]:.2f}").rjust(14)
                                             for k in keys))
    return lines, {"layers": per, "bands": bands}, {"q": {t: torch.stack(v) for t, v in unit_q.items()}}


def heads(saved: dict, dev) -> tuple[list[str], dict]:
    geo = arch.layout(MODEL)
    L, H = len(geo), geo[0].q_heads
    tot = saved["state"]["total"]  # (the scoring loop below uses `st` for its own statistics)
    tz = torch.load(TOKEN_Z, weights_only=False, mmap=True)
    keep = tz["keep"]
    V = len(keep)
    kidx = kept_index()
    top = saved["state"]["top"]
    doc, pos, src = top["h_doc"].long(), top["h_pos"].long(), top["h_src"].long()  # [L, H, K]
    flat, start = saved["tokens"], saved["doc_start"]
    att, nxt = firing_tokens(flat, start, doc, src).long(), firing_tokens(flat, start, doc, pos, 1).long()
    k_att, k_nxt = (torch.where(t >= 0, kidx[t.clamp_min(0)], -1) for t in (att, nxt))
    acts = pq.read_table(head_table("acts"), columns=[*PATTERN, "cur_mode", "src_mode"]).to_pydict()
    pattern = torch.stack([torch.tensor(acts[k]).view(L, H) > v for k, v in PATTERN.items()]).any(0)
    cur_mode, src_mode = (torch.tensor(acts[k]).view(L, H) for k in ("cur_mode", "src_mode"))
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted(set(nm.values())))
    E = head[nm["embed"]][keep].float() * arch.embed_scale(MODEL)
    E = (E - E.mean(0)).to(dev)
    Wu = (head[nm["unembed"]][keep].float() * head[nm["norm"]].float()).to(dev)
    del head
    K_out, J_mid = arch.lens(MODEL, "S_out", S_FIT), arch.lens(MODEL, "R_mid", R_FIT)
    eye = torch.eye(H, dtype=torch.bool)
    stat, pval = {t: torch.full((L, H), torch.nan) for t in HEAD_TESTS}, {t: torch.ones(L, H) for t in HEAD_TESTS}

    for l in range(L):
        ok_att = k_att[l] >= 0  # [H, K]
        ka = k_att[l].clamp_min(0)
        scores = {}  # test -> [H, H', K]: firing j of head h scored under head h'
        for test, name in (("ov_copy", "ov_R"), ("ov_S", "ov_S"), ("qk_S", "qk_S")):
            zl = tz[name][l].float()  # [H', V]
            scores[test] = (zl[:, ka.flatten()].view(H, H, -1).transpose(0, 1), ok_att)
        # ov_R: z of the next token in the attended token's column, M[u, v] = A_u · B_v.
        kvh, hd = geo[l].kv_heads, geo[l].head_dim
        grp = H // kvh
        Wv, Wo = arch.ov(MODEL, geo[l], dev)  # W_V (W_K where keys are values) × γ_in; W_O, × γ_post for Gemma
        A = (Wu @ (J_mid[l].to(dev) @ Wo)).view(V, H, hd)
        abar = A.mean(0)
        cov = torch.einsum("vhi,vhj->hij", A - abar, A - abar) / V
        X = (E[ka.flatten().to(dev)] @ K_out[l - 1].to(dev).T) if l else E[ka.flatten().to(dev)]  # [H·K, d]
        B = (X @ Wv.T).view(H, -1, kvh, hd)  # [H, K, kvh, hd]: each firing through every kv head's W_V
        if "v_r" in tot and float(tot["v_r"][l, :kvh].abs().sum()) > 0:  # Gemma's v_norm, as 08_reads.py
            Av = norm_derivative(torch.ones(hd, device=dev), tot["v_r"][l, :kvh].float().to(dev),
                                 tot["v_m"][l, :kvh, :hd, :hd].float().to(dev), float(tot["n_k"]))
            B = torch.einsum("gij,hkgj->hkgi", Av, B)
        Bh = B[:, :, torch.arange(H) // grp]  # [H, K, H', hd]
        An = A[k_nxt[l].clamp_min(0).flatten().to(dev)].view(H, -1, H, hd)  # [H, K, H', hd]
        num = ((An - abar) * Bh).sum(-1)
        var = torch.einsum("hkgi,gij,hkgj->hkg", Bh, cov, Bh)
        scores["ov_R"] = ((num / var.clamp_min(1e-30).sqrt()).permute(0, 2, 1).cpu(), ok_att & (k_nxt[l] >= 0))
        del A, cov, X, B, Bh, An
        for test, (sc, ok) in scores.items():  # sc [H, H', K]
            st = (sc * ok[:, None, :]).sum(-1) / ok.sum(-1, keepdim=True).clamp_min(1)
            m = st.diagonal().clone()
            m[ok.sum(-1) < MIN_FIRINGS] = torch.nan
            stat[test][l] = m
            pval[test][l] = null_p(m, st[~eye])
        print(f"{time.strftime('%H:%M:%S')} heads L{l}", flush=True)
    q = {t: bh(pval[t].flatten()).view(L, H) for t in HEAD_TESTS}
    flags = {t: q[t] <= FDR for t in HEAD_TESTS} | {"cur_det": cur_mode >= PAST, "src_det": src_mode >= PAST,
                                                    "mode": (cur_mode >= PAST) | (src_mode >= PAST), "pattern": pattern}
    flags |= {f"{t}_p05": pval[t] <= 0.05 for t in HEAD_TESTS}  # uncorrected; 5% by chance
    lines = [(f"heads, {MODEL}: {L} × {H} heads; share of each band's heads passing each test at a {FDR:.0%} false discovery "
              "rate (BH over all heads); detectors: a third of firings share a query or attended token; pattern: "
              f"{', '.join(f'{k} > {v}' for k, v in PATTERN.items())}"), "", "band     " + "  ".join(f"{k:>9s}" for k in flags)]
    bands = {}
    for lo, hi in BANDS + ((0, L - 1),):
        bands[f"L{lo}-{hi}"] = {k: v[lo:hi + 1].float().mean().item() for k, v in flags.items()}
        lines.append(f"L{lo}-{hi}".ljust(8) + " " + "  ".join(f"{x:9.2f}" for x in bands[f"L{lo}-{hi}"].values()))
    layers = [{"layer": l, **{k: v[l].float().mean().item() for k, v in flags.items()}} for l in range(L)]
    return lines, {"layers": layers, "bands": bands}, {"q": q, "p": pval}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="neurons,heads")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    saved = torch.load(ACTS, weights_only=False, map_location="cpu", mmap=True)
    path = SUMMARY / f"reads_at_firings_{STEM}.json"
    units_path = UNITS
    units = torch.load(units_path, weights_only=False) if units_path.exists() else {}
    table = json.loads(path.read_text()) if path.exists() else {}
    for part, fn in (("neurons", neurons), ("heads", heads)):
        if part in parts:
            lines, table[part], units[part] = fn(saved, args.device)
            table[f"{part}_log"] = lines
    text = "\n\n".join("\n".join(table[f"{p}_log"]) for p in ("neurons", "heads") if f"{p}_log" in table)
    print(text)
    (SUMMARY / f"reads_at_firings_{STEM}.log").write_text(text + "\n")
    path.write_text(json.dumps(table))
    units_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(units, units_path)


if __name__ == "__main__":
    main()
