"""Attention heads read from their actual outputs at their top firings (findings §11, §12): does a
head's output promote the token after the firing, or the token it attends to; and do the weight-only
per-token OV reads predict where the attended token ranks in that output?

Firings are ``06_activations.py``'s: each head's 24 top documents over all 63,990, at the position
(16+) of largest output norm ``‖W_O^h z_t‖`` in each, with the output vector ``W_O^h z_t`` (fp16, from
the bf16 pass) and the position the head attends to most past the sink. At each firing, the next
token (after the query) and the attended token, among the kept tokens (``catalog.kept``).

- **next, copy** (§12): each firing's output read through the R-lens's ``mid`` target at its layer
  (attention writes into the MLP input, where ``mid`` starts; normed and unembedded as ``heads/acts``'
  ``out_R_z``) and through the logit lens; each token's logit z-scored across the layer's 768
  firings. A head's statistic is the mean z of its next (or attended) tokens at its own firings. The
  mismatched pairs score each other head of the layer at its own firings on this head's tokens (31
  per head); a head's p-value is the share of its layer's mismatched pairs that score as high, and a
  test passes at a false discovery rate of 5% (Benjamini-Hochberg over all 1,152 heads). Heads with
  fewer than 3 usable firings are not scored.
- **weight-only reads at the firings** (§11): the attended token's z in its own column of the head's
  OV map, under the head and under another head of its layer (the control, one fixed draw per
  head), against its z over the vocabulary in the head's actual output read; Spearman over firings.
  The per-token z: ``S→R`` is ``08_reads.py``'s ``ov_R`` (S-lens footprint in, R-lens ``mid`` out),
  checked against the R-lens and logit-lens reads of the output; against the logit-lens read also
  ``raw`` (embedding in, ``W_U diag(γ_f)`` out, Elhage et al. 2021) and ``S`` (S-lens footprint in,
  ``W_U diag(γ_f)`` out), computed here with ``neuron_lens.heads.own_token_z``.
- **attend to what they copy** (§11): where the attended token falls among the vocabulary in the
  head's own ``ov_R`` per-token z (its percentile, 0.5 by chance), and in the control head's; per
  head the mean over firings, and the share of heads past 0.9 (attend to what they copy) or under
  0.1 (to what they suppress).

The other shares (sink and pattern heads, token modes) are ``reads_at_firings.py``'s, from ``heads/acts``, with its
weight-only tests at the same firings. Needs the activations file
(``data/acts/<model>.pt``; the output vectors are not released) and ``08_reads.py``'s per-token z
(``data/heads/<model>-token-z.pt``, which itself needs the activations file). Qwen3-8B only. The
summary goes to ``data/summary/head_census_<model>.{log,json}``.

    python scripts/checks/head_census.py
"""
import json

import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import MODEL, acts_path, head_table, kept, kept_index
from neuron_lens.heads import own_token_z
from neuron_lens.paths import DATA, SUMMARY
from neuron_lens.readout import rms_norm_columns, transport
from neuron_lens.stats import bh, null_p, spearman
from neuron_lens.weights import tensors

STEM = arch.stem(MODEL)
TOKEN_Z = DATA / "heads" / f"{STEM}-token-z.pt"
FDR = 0.05
MIN_FIRINGS = 3
TESTS = ("next_mid", "next_logit", "copy_mid", "copy_logit")
AGREE = (("raw", "logit"), ("S", "logit"), ("S→R", "logit"), ("S→R", "mid"))


def main() -> None:
    if MODEL != arch.QWEN:
        raise SystemExit(f"head_census.py reads Qwen3-8B's heads only, not {MODEL}")
    dev = "mps" if torch.backends.mps.is_available() else "cuda"
    saved = torch.load(acts_path(), weights_only=False, map_location="cpu", mmap=True)
    top, flat, start = saved["state"]["top"], saved["tokens"], saved["doc_start"]
    doc, pos, src = top["h_doc"].long(), top["h_pos"].long(), top["h_src"].long()  # [L, H, K]
    L, H, K = doc.shape
    fired = top["h_val"] > 0
    nxt = torch.where(pos + 1 < start[doc + 1] - start[doc], flat[(start[doc] + pos + 1).clamp_max(len(flat) - 1)].long(), -1)
    kidx = kept_index()
    k_nxt = torch.where(nxt >= 0, kidx[nxt.clamp_min(0)], -1)
    k_att = kidx[flat[start[doc] + src].long()]
    tz = torch.load(TOKEN_Z, weights_only=False, mmap=True)
    keep = kept()
    assert torch.equal(tz["keep"], keep), f"{TOKEN_Z} covers other tokens than catalog.kept(): rerun 08_reads.py"
    V = len(keep)

    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted(set(nm.values())))
    unembed = head[nm["unembed"]][keep].to(dev, torch.float16)
    gamma = head[nm["norm"]].to(dev, torch.float32)
    E = head[nm["embed"]][keep].float()
    E = (E - E.mean(0)).to(dev)
    Wu = (head[nm["unembed"]][keep].float() * head[nm["norm"]].float()).to(dev)  # W_U diag(γ_f)
    del head
    R_FIT, S_FIT = arch.fits(MODEL)
    J_mid, K_out = arch.lens(MODEL, "R_mid", R_FIT), arch.lens(MODEL, "S_out", S_FIT)
    geo = arch.layout(MODEL)
    other = (torch.arange(H)[None, :] + torch.randint(1, H, (L, H), generator=torch.Generator().manual_seed(0))) % H

    eye = torch.eye(H, dtype=torch.bool)
    stat = {t: torch.full((L, H), torch.nan) for t in TESTS}
    pval = {t: torch.ones(L, H) for t in TESTS}
    ctx = {lens: torch.full((L, H, K), torch.nan) for lens in ("mid", "logit")}  # the attended token's z over the vocabulary
    wz = {(v, side): torch.full((L, H, K), torch.nan) for v in ("raw", "S", "S→R") for side in ("own", "ctl")}
    pct = {side: torch.full((L, H, K), torch.nan) for side in ("own", "ctl")}
    for l in range(L):
        a = k_att[l].clamp_min(0)  # [H, K]
        O = top["h_out"][l].reshape(H * K, -1).to(dev, torch.float32).T  # [d, H·K]
        for lens, J in (("mid", J_mid[l]), ("logit", None)):
            lg = arch.cap(MODEL, unembed @ rms_norm_columns(transport(O, J), gamma).half()).float()  # [V, H·K]
            zv = (lg - lg.mean(0)) / lg.std(0).clamp_min(1e-6)
            ctx[lens][l] = zv.gather(0, a.flatten().to(dev)[None]).view(H, K).cpu()
            z = (lg - lg.mean(1, keepdim=True)) / lg.std(1, keepdim=True).clamp_min(1e-6)  # across the layer's firings
            del lg, zv
            for side, toks in (("next", k_nxt[l]), ("copy", k_att[l])):
                ok = fired[l] & (toks >= 0)
                # zz[h, h', j]: head h's j-th token at head h''s j-th firing
                zz = z[toks.clamp_min(0).flatten().to(dev)].view(H, K, H, K).diagonal(dim1=1, dim2=3).cpu()
                st = (zz * ok[:, None, :]).sum(-1) / ok.sum(-1, keepdim=True).clamp_min(1)
                m = st.diagonal().clone()
                m[ok.sum(-1) < MIN_FIRINGS] = torch.nan
                stat[f"{side}_{lens}"][l], pval[f"{side}_{lens}"][l] = m, null_p(m, st[~eye])
            del z
        # The weight-only z at the attended tokens: raw and S (W_U diag(γ_f) out) here, S→R from 08_reads.py.
        kvh, hd = geo[l].kv_heads, geo[l].head_dim
        Wv, Wo = arch.ov(MODEL, geo[l], dev)  # W_V diag(γ₁) [kvh·hd, d], W_O [d, H·hd]
        A = (Wu @ Wo).view(V, H, hd)
        for v, X in (("raw", E), ("S", E @ K_out[l - 1].to(dev).T if l else E)):
            B = (X @ Wv.T).view(V, kvh, hd).repeat_interleave(H // kvh, dim=1)
            zt = own_token_z(A, B).T.cpu()  # [H, V]
            wz[v, "own"][l], wz[v, "ctl"][l] = zt.gather(1, a), zt[other[l]].gather(1, a)
            del B, zt
        del A
        W = tz["ov_R"][l].float()  # [H, V]
        wz["S→R", "own"][l], wz["S→R", "ctl"][l] = W.gather(1, a), W[other[l]].gather(1, a)
        srt = W.sort(-1).values
        for side, hh in (("own", torch.arange(H)), ("ctl", other[l])):
            pct[side][l] = torch.searchsorted(srt[hh], W[hh].gather(1, a).contiguous()).float() / V
        print(f"L{l}: next_mid {stat['next_mid'][l].nanmean():+.2f}  copy_mid {stat['copy_mid'][l].nanmean():+.2f}", flush=True)

    q = {t: bh(pval[t].flatten()).view(L, H) for t in TESTS}
    bands = arch.bands(L) + [(0, L - 1)]
    lay = torch.arange(L)[:, None, None].expand(L, H, K)
    ok_att = fired & (k_att >= 0)
    lines = [(f"attention heads read from their outputs at their top firings, {MODEL}: {L} × {H} heads, {K} firings each "
             f"(largest output norm, one per document, positions 16+, over {saved['meta']['docs']:,} documents; "
             f"06_activations.py), {int(ok_att.sum()):,} firings whose attended token is kept"), "",
             ("share of heads whose output promotes the next or the attended token: the mean z of those tokens under each "
             "firing's output read (R-lens mid or logit lens), z-scored across the layer's firings, at a 5% false "
             "discovery rate against the layer's mismatched pairs (BH over all heads)"), "",
             "band     " + "  ".join(f"{t:>10s}" for t in TESTS)]
    table = {"tests": {}, "agree": {}, "pct": {}}
    for lo, hi in bands:
        row = {t: (q[t][lo:hi + 1] <= FDR).float().mean().item() for t in TESTS}
        table["tests"][f"L{lo}-{hi}"] = row
        lines.append(f"L{lo}-{hi}".ljust(8) + " " + "  ".join(f"{x:10.2f}" for x in row.values()))

    lines += ["", ("weight-only per-token OV reads at the firings: Spearman over firings between the attended token's z in its "
              "own column (under the head / under another head of its layer, the control) and its z over the vocabulary in "
              "the head's output read through each lens:")]
    for v, lens in AGREE:
        row = {}
        for lo, hi in bands:
            b = ok_att & (lay >= lo) & (lay <= hi)
            row[f"L{lo}-{hi}"] = [spearman(wz[v, side][b], ctx[lens][b]) for side in ("own", "ctl")]
        table["agree"][f"{v} vs {lens}"] = row
        lines.append(f"  {v:3s} vs {lens:5s}  " + "  ".join(f"{k} {o:+.2f}/{c:+.2f}" for k, (o, c) in row.items()))

    hpct = {k: torch.where(ok_att, x, 0).sum(-1) / ok_att.sum(-1).clamp_min(1) for k, x in pct.items()}  # [L, H]
    scored = ok_att.sum(-1) >= MIN_FIRINGS
    lines += ["", ("where the attended token falls in the head's ov_R per-token read (own head / control head): mean "
              "percentile over firings, share of firings in the top 1%, share of heads whose mean percentile is past 0.9 "
              "(attend to what they copy) or under 0.1 (to what they suppress):")]
    for lo, hi in bands:
        b = ok_att & (lay >= lo) & (lay <= hi)
        hb = scored & (lay[..., 0] >= lo) & (lay[..., 0] <= hi)
        row = {k: [pct[k][b].mean().item(), (pct[k][b] > 0.99).float().mean().item(), (hpct[k][hb] > 0.9).float().mean().item(),
                   (hpct[k][hb] < 0.1).float().mean().item()] for k in pct}
        table["pct"][f"L{lo}-{hi}"] = row
        lines.append(f"  L{lo}-{hi}: " + ",  ".join(f"{n} {row['own'][i]:.3f} / {row['ctl'][i]:.3f}" for i, n in
                                                     enumerate(("mean", "top 1%", "heads > 0.9", "heads < 0.1"))))

    reads = pq.read_table(head_table("acts"), columns=["out_R_z_str"]).column("out_R_z_str").to_pylist()
    lines += ["", "strongest heads per band (next and copy through the R-lens mid read), with heads/acts' out_R_z:"]
    for lo, hi in arch.bands(L):
        for t in ("next_mid", "copy_mid"):
            s = stat[t][lo:hi + 1].nan_to_num(-1e9).flatten()
            for i in s.topk(3).indices.tolist():
                l, h = lo + i // H, i % H
                lines.append(f"  L{l}h{h} {t} z {stat[t][l, h]:.2f}: {reads[l * H + h]}")
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"head_census_{STEM}.log").write_text(text + "\n")
    (SUMMARY / f"head_census_{STEM}.json").write_text(json.dumps(table))


if __name__ == "__main__":
    main()
