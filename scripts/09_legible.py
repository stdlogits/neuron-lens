"""Which weight-only reads say something measurable about each neuron pole and attention head, at
its top firings (pipeline stage 9; findings §7, §12).

Every test scores a read on the tokens of the unit's own top firings (``06_activations.py``: the
strongest position in each of its top 24 documents) as a mean z, against mismatched pairs: another
unit of the layer scored on this unit's tokens. A unit's p-value on a test is the share of its
layer's mismatched pairs that score as high, and a test passes at a false discovery rate of 5%
(Benjamini-Hochberg over the layer's units for neurons, over all heads for heads). Units with fewer
than 3 usable firings are not scored. Columns give the z and q-value of each test.

**Neurons** (``neurons/legible/L<layer>.parquet``), z of each token across the layer's poles:

- ``write_R``: the write's R-lens read promotes the tokens that follow the firings;
- ``write_S``: the write's S-lens read names the peak tokens (early neurons write back what they
  detect);
- ``read_R``, ``read_S``: the read direction's R-lens (from the MLP input) and S-lens reads name the
  peak tokens;
- ``silent``: no firing, or ``act_top`` under 0.05; ``cur_mode``, ``next_mode``: the share of top
  firings on one peak token, or followed by one token; ``dark``: fires, but no test passes and
  neither mode reaches a third.

**Heads** (``heads/legible.parquet``), each read's per-token z from ``08_reads.py``, at the token the
firing attends to most (past the sink):

- ``ov_R``: the OV column of the attended token through the R-lens (what the head writes when it
  reads it) promotes the token that follows the query;
- ``ov_copy``: the attended token's z in its own ``ov_R`` column: the head promotes what it reads;
- ``ov_S``: the same through the S-lens: the write carries the attended token's footprint;
- ``qk_S``: the attended token's ``qk_S`` z: the QK read predicts what the head attends to;
- ``cur_mode``, ``src_mode``: the share of firings on one query token, or attending to one token;
  ``pattern``: ``prev`` or ``self`` past 0.5, ``induction`` or ``duplicate`` past 0.3; ``dark``: no
  test passes, neither mode reaches a third, and no pattern.
- ``<test>_p``: the uncorrected p-value against the layer's mismatched pairs. A layer's H(H − 1)
  pairs bound p below by 1/(H(H − 1) + 1): with 16 heads a layer (Gemma 4), BH over all heads passes
  only if dozens sit at that floor, so ``_q`` can be 1 where a head clearly stands out of its layer.

Per band into ``data/summary/legible_<model>.{log,json}``.

Gemma 4 12B (``NEURON_LENS_MODEL=google/gemma-4-12B``): the released lenses; each neuron's write as
the residual sees it and its read rows times the pre-feedforward norm (``neuron_lens/neurons.py``);
embeddings × √d; each head's ``W_O`` times the post-attention norm weight, keys as values on the full
layers, and the values through ``v_norm``'s averaged true derivative, as ``08_reads.py`` reads them.

    python scripts/09_legible.py [--parts neurons,heads]
"""
import argparse
import json
import time

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from neuron_lens.catalog import (
    MODEL,
    acts_path,
    head_keys,
    head_table,
    kept,
    kept_index,
    neuron_keys,
    neuron_table,
    write,
)
from neuron_lens import arch
from neuron_lens.heads import norm_derivative
from neuron_lens.neurons import directions
from neuron_lens.paths import DATA, SUMMARY
from neuron_lens.readout import rms_norm_columns, transport
from neuron_lens.stats import bh, modal_share, null_p
from neuron_lens.weights import tensors

R_FIT, S_FIT = arch.fits(MODEL)
STEM = MODEL.split("/")[-1].lower()
ACTS = acts_path()
TOKEN_Z = DATA / "heads" / f"{STEM}-token-z.pt"
PAST = 1 / 3
SILENT = 0.05
MIN_FIRINGS = 3
FDR = 0.05
NULL_Q = 0.99
CHUNK = 2048
BANDS = tuple(arch.bands(arch.text_config(MODEL).num_hidden_layers))  # Qwen3-8B's five, else quarters
NEURON_TESTS = ("write_R", "write_S", "read_R", "read_S")
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


def firing_tokens(saved: dict, doc: torch.Tensor, pos: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """The token at ``pos + offset`` of each firing's document, −1 past its end."""
    flat, start = saved["tokens"], saved["doc_start"]
    p = pos + offset
    inside = p < start[doc + 1] - start[doc]
    return torch.where(inside, flat[(start[doc] + p).clamp_max(len(flat) - 1)].long(), -1)


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
    per = []
    for layer in range(L):
        t0 = time.perf_counter()
        val = torch.cat([top["val"][layer, p].T for p in (0, 1)]).float()  # [2f, K]: + poles, then −
        doc = torch.cat([top["doc"][layer, p].T for p in (0, 1)]).long()
        pos = torch.cat([top["pos"][layer, p].T for p in (0, 1)]).long()
        fired = val > 0
        cur, nxt = firing_tokens(saved, doc, pos), firing_tokens(saved, doc, pos, 1)
        silent = (fired.sum(1) == 0) | (val.median(1).values.clamp_min(0) < SILENT)
        k_cur, k_nxt = (torch.where(t >= 0, kidx[t.clamp_min(0)], -1) for t in (cur, nxt))
        ok_cur, ok_nxt = fired & (k_cur >= 0), fired & (k_nxt >= 0)
        perm = torch.cat([torch.randperm(f, generator=g), f + torch.randperm(f, generator=g)])
        w = directions(MODEL, layer)  # the write as the residual sees it; read rows times the MLP input norm
        wd = w["write"].to(dev)
        r = rms_norm_columns(transport(wd, J_out[layer] if layer < len(J_out) else None), gamma).T  # [f, d]
        sw = (K_out[layer].to(dev).T @ wd).T
        sw = sw / sw.norm(dim=1, keepdim=True)
        a, b = w["gate"], w["up"]
        a, b = a / a.norm(dim=1, keepdim=True), b / b.norm(dim=1, keepdim=True)
        del w
        c = torch.cat([a + b, a - b]).to(dev)
        rr = rms_norm_columns(transport(c.T, J_mid[layer]), gamma).T
        cs = c @ K_mid[layer].to(dev)
        stats = {"write_R": mean_z(torch.cat([r, -r]).half(), unembed, k_nxt, ok_nxt, perm),
                 "write_S": mean_z(torch.cat([sw, -sw]).half(), emb, k_cur, ok_cur, perm),
                 "read_R": mean_z(rr.half(), unembed, k_cur, ok_cur, perm),
                 "read_S": mean_z((cs / cs.norm(dim=1, keepdim=True)).half(), emb, k_cur, ok_cur, perm)}
        cur_mode, next_mode = modal_share(cur, fired), modal_share(nxt, fired & (nxt >= 0))
        cols = neuron_keys(layer, f)
        cols |= {"silent": pa.array(silent.numpy()), "cur_mode": pa.array(cur_mode.numpy()), "next_mode": pa.array(next_mode.numpy())}
        row = {"layer": layer, "silent": silent.float().mean().item(),
               "token_detector": ((cur_mode >= PAST) & ~silent).float().mean().item(),
               "next_detector": ((next_mode >= PAST) & ~silent).float().mean().item()}
        passed = []
        for test, (m, n) in stats.items():
            ok = ~m.isnan() & ~silent
            q = torch.ones(2 * f)
            q[ok] = bh(null_p(m[ok], n))
            cols[f"{test}_z"], cols[f"{test}_q"] = pa.array(m.numpy()), pa.array(q.numpy())
            passed.append(q <= FDR)
            thr = n[~n.isnan()].quantile(NULL_Q).item()
            row[test] = passed[-1].float().mean().item()
            row[f"{test}_net"] = (((m > thr) & ~silent).float().mean() - ((n > thr) & ~silent).float().mean()).item()
        any_test = torch.stack(passed).any(0)
        dark = ~silent & ~any_test & (cur_mode < PAST) & (next_mode < PAST)
        cols["dark"] = pa.array(dark.numpy())
        row["any_test"] = any_test.float().mean().item()
        row["dark"] = dark.float().mean().item()
        write(cols, neuron_table("legible", layer))
        per.append(row)
        print(f"{time.strftime('%H:%M:%S')} neurons L{layer}: " + " ".join(f"{k} {row[k]:.2f}" for k in
              ("silent", *NEURON_TESTS, "dark")) + f" in {time.perf_counter() - t0:.0f}s", flush=True)
    keys = ("silent", "token_detector", "next_detector", *NEURON_TESTS, "any_test", "dark")
    lines = [(f"neurons, {MODEL}: {2 * f * L:,} poles; share of each band's poles (mean over layers) passing each test at "
              f"a {FDR:.0%} false discovery rate; in brackets, the share past the {NULL_Q:.0%} point of mismatched pairs net "
              "of theirs. Detectors: a third of top firings share a token. Dark: fires, but no test and no detector."), "",
             "band     " + "  ".join(f"{k:>14s}" for k in keys)]
    bands = band_rows(per, L)
    for name, r in bands.items():
        lines.append(f"{name:8s} " + "  ".join((f"{r[k]:.2f} [{r[k + '_net']:.2f}]" if k + "_net" in r else f"{r[k]:.2f}").rjust(14)
                                             for k in keys))
    return lines, {"layers": per, "bands": bands}


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
    cur, att, nxt = firing_tokens(saved, doc, pos), firing_tokens(saved, doc, src), firing_tokens(saved, doc, pos, 1)
    k_att, k_nxt = (torch.where(t >= 0, kidx[t.clamp_min(0)], -1) for t in (att, nxt))
    acts = pq.read_table(head_table("acts"), columns=list(PATTERN)).to_pydict()
    pattern = torch.stack([torch.tensor(acts[k]).view(L, H) > v for k, v in PATTERN.items()]).any(0)
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
    # Uncorrected p too: a layer's H(H − 1) mismatched pairs bound p below by 1/(H(H − 1) + 1), so with
    # Gemma 4's 16 heads a layer (240 pairs) BH over 768 heads passes only if dozens sit at that floor.
    passed = torch.stack([q[t] <= FDR for t in HEAD_TESTS]).any(0)
    cur_mode, src_mode = modal_share(cur, torch.ones_like(cur, dtype=torch.bool)), modal_share(att, att >= 0)
    dark = ~passed & (cur_mode < PAST) & (src_mode < PAST) & ~pattern
    cols = head_keys(L, H, arch.groups(MODEL))
    for t in HEAD_TESTS:
        cols[f"{t}_z"], cols[f"{t}_q"] = pa.array(stat[t].flatten().numpy()), pa.array(q[t].flatten().numpy())
    cols |= {f"{t}_p": pa.array(pval[t].flatten().numpy()) for t in HEAD_TESTS}  # after Qwen's columns
    cols |= {"cur_mode": pa.array(cur_mode.flatten().numpy()), "src_mode": pa.array(src_mode.flatten().numpy()),
             "pattern": pa.array(pattern.flatten().numpy()), "dark": pa.array(dark.flatten().numpy())}
    write(cols, head_table("legible"))
    flags = {t: q[t] <= FDR for t in HEAD_TESTS} | {"cur_det": cur_mode >= PAST, "src_det": src_mode >= PAST,
                                                    "pattern": pattern, "any_test": passed, "dark": dark}
    flags |= {f"{t}_p05": pval[t] <= 0.05 for t in HEAD_TESTS}  # uncorrected; 5% by chance
    lines = [(f"heads, {MODEL}: {L} × {H} heads; share of each band's heads passing each test at a {FDR:.0%} false discovery "
              "rate (BH over all heads); detectors: a third of firings share a query or attended token; pattern: "
              f"{', '.join(f'{k} > {v}' for k, v in PATTERN.items())}"), "", "band     " + "  ".join(f"{k:>9s}" for k in flags)]
    bands = {}
    for lo, hi in BANDS + ((0, L - 1),):
        bands[f"L{lo}-{hi}"] = {k: v[lo:hi + 1].float().mean().item() for k, v in flags.items()}
        lines.append(f"L{lo}-{hi}".ljust(8) + " " + "  ".join(f"{x:9.2f}" for x in bands[f"L{lo}-{hi}"].values()))
    return lines, {"bands": bands}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="neurons,heads")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    saved = torch.load(ACTS, weights_only=False, map_location="cpu", mmap=True)
    path = SUMMARY / f"legible_{STEM}.json"
    table = json.loads(path.read_text()) if path.exists() else {}
    for part, fn in (("neurons", neurons), ("heads", heads)):
        if part in parts:
            lines, table[part] = fn(saved, args.device)
            table[f"{part}_log"] = lines
    text = "\n\n".join("\n".join(table[f"{p}_log"]) for p in ("neurons", "heads") if f"{p}_log" in table)
    print(text)
    (SUMMARY / f"legible_{STEM}.log").write_text(text + "\n")
    path.write_text(json.dumps(table))


if __name__ == "__main__":
    main()
