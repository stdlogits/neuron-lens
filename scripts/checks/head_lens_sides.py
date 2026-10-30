"""Which lens on each side of an attention head's OV circuit (findings §11): the four per-token
reads, input lens → output lens, S→S, S→R, R→S and R→R, each checked at the census firings
against the targets it claims.

A per-token OV read says "reading token v, the head writes token u". The input lens says what
reading v means:
- **S:** the attended position holds v (v's S-lens footprint at L−1's output); the target is the
  attended token;
- **R:** the attended position says v (the R-lens direction at L−1's output that promotes v); the
  target is the token after the attended position.

The output lens says what writing u means:
- **R:** the write makes the model say u (the R-lens read from L's MLP input);
- **S:** the write makes the residual look as if u were there (the S-lens footprint read there).

At each of ``06_activations.py``'s head firings (each head's 24 largest outputs, one per document),
with v the input target and the attended position before the query:
- **copy:** v's weight-only z in its own column of the map (as ``08_reads.py``'s ``ov_*``), against
  v's z over the vocabulary in the head's actual output read through the output lens. Spearman
  over firings;
- **column:** the whole predicted column (what the head writes when it reads v) against the read
  of the actual output, Spearman over the vocabulary. Mean over firings.

The control is the same token through another head of the layer, one fixed control per head.
Every variant is scored on the same firings: from L1 (R inputs need L−1), with both targets kept.
By band, by pattern class (``heads/acts``), and the share of heads each variant reads best. Findings
§11 ran it on a 4,000-document activation pass and again on the full one.

    python scripts/checks/head_lens_sides.py
"""
import json

import pyarrow.parquet as pq
import torch
from transformers import AutoConfig

from neuron_lens.catalog import MODEL, acts_path, head_table, kept, kept_index
from neuron_lens.lenses import load_fits
from neuron_lens.paths import FITS, SUMMARY
from neuron_lens.stats import centered_ranks, spearman
from neuron_lens.weights import readout_names, tensors

STEM = MODEL.split("/")[-1].lower()
ACTS = acts_path()
CLASSES = {"prev": 0.5, "induction": 0.3, "duplicate": 0.3, "sink": 0.9}
R_FIT, S_FIT = "r-qwen3-8b-L36-dclm", "s-qwen3-8b-L36-dclm"
VARIANTS = ("S→S", "S→R", "R→S", "R→R")
BANDS = [(1, 8), (9, 12), (13, 20), (21, 27), (28, 35), (1, 35)]


def main() -> None:
    dev = "mps" if torch.backends.mps.is_available() else "cuda"
    cfg = AutoConfig.from_pretrained(MODEL)
    L, H, kvh, hd = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    g = H // kvh
    saved = torch.load(ACTS, weights_only=False, map_location="cpu", mmap=True)
    top, flat, start = saved["state"]["top"], saved["tokens"], saved["doc_start"]
    doc, pos, src = top["h_doc"].long(), top["h_pos"].long(), top["h_src"].long()
    K = doc.shape[2]
    keep, kidx = kept(), kept_index()
    V = len(keep)
    after = torch.where(src + 1 < start[doc + 1] - start[doc],  # the token after the attended position
                        flat[(start[doc] + src + 1).clamp_max(len(flat) - 1)].long(), -1)
    tgt = {"S": kidx[flat[start[doc] + src].long()], "R": torch.where(after >= 0, kidx[after.clamp_min(0)], -1)}
    ok = (src < pos) & (tgt["S"] >= 0) & (tgt["R"] >= 0)
    ok[0] = False
    other = (torch.arange(H)[None, :] + torch.randint(1, H, (L, H), generator=torch.Generator().manual_seed(0))) % H

    head = tensors(MODEL, readout_names() + ["model.embed_tokens.weight"])
    E = head["model.embed_tokens.weight"][keep].float()
    E = (E - E.mean(0)).to(dev)
    Wu = (head["lm_head.weight"][keep].float() * head["model.norm.weight"].float()).to(dev)
    _, s_acc = load_fits([FITS / f"{S_FIT}.pt"], "S", MODEL)
    K_out, K_mid = s_acc.mean("out"), s_acc.mean("mid")
    del s_acc
    _, r_acc = load_fits([FITS / f"{R_FIT}.pt"], "R", MODEL)
    J_out, J_mid = r_acc.mean("out"), r_acc.mean("mid")
    del r_acc

    res = {v: {k: torch.full((L, H, K), torch.nan) for k in ("w_own", "w_ctl", "ctx", "col_own", "col_ctl")}
           for v in VARIANTS}
    for layer in range(1, L):
        p = f"model.layers.{layer}."
        w = tensors(MODEL, [p + "self_attn.v_proj.weight", p + "self_attn.o_proj.weight", p + "input_layernorm.weight"])
        Wv = (w[p + "self_attn.v_proj.weight"].float() * w[p + "input_layernorm.weight"].float()).to(dev)  # [kvh·hd, d]
        Wo = w[p + "self_attn.o_proj.weight"].float().to(dev)  # [d, H·hd]
        O = top["h_out"][layer].reshape(H * K, -1).to(dev, torch.float32).T  # [d, H·K]: the actual outputs
        jm, km = J_mid[layer].to(dev), K_mid[layer].to(dev)
        reads = {"R": Wu @ (jm @ O), "S": E @ (km.T @ O)}  # [V, H·K]: each output read through each lens
        A = {"R": (Wu @ (jm @ Wo)).view(V, H, hd), "S": (E @ (km.T @ Wo)).view(V, H, hd)}  # U_u through W_O^h
        X = {"S": lambda t: E[t] @ K_out[layer - 1].to(dev).T, "R": lambda t: Wu[t] @ J_out[layer - 1].to(dev)}
        hh, oth = torch.arange(H, device=dev), other[layer].to(dev)
        for out in ("R", "S"):
            rd = reads[out]
            ctx_z = (rd - rd.mean(0)) / rd.std(0)
            rd_r = centered_ranks(rd, 0)
            for inp in ("S", "R"):
                v = f"{inp}→{out}"
                t = tgt[inp][layer].clamp_min(0).flatten().to(dev)  # [H·K]
                B = (X[inp](t) @ Wv.T).view(H, K, kvh, hd)  # each firing's target through every kv head's W_V
                own = B[hh, :, hh // g]  # [H, K, hd]
                ctl = B[hh, :, oth // g]
                cols = {"own": torch.einsum("vhi,hki->vhk", A[out], own).reshape(V, H * K),
                        "ctl": torch.einsum("vhi,hki->vhk", A[out][:, oth], ctl).reshape(V, H * K)}
                for side, col in cols.items():
                    z = (col - col.mean(0)) / col.std(0)
                    res[v][f"w_{side}"][layer] = z.gather(0, t[None]).view(H, K).cpu()
                    rc = centered_ranks(col, 0)
                    res[v][f"col_{side}"][layer] = ((rc * rd_r).sum(0) / (rc.norm(dim=0) * rd_r.norm(dim=0))).view(H, K).cpu()
                    del col, z, rc
                res[v]["ctx"][layer] = ctx_z.gather(0, t[None]).view(H, K).cpu()
            del rd, ctx_z, rd_r
        print(f"L{layer}: " + "  ".join(f"{v} {res[v]['col_own'][layer][ok[layer]].mean():+.2f}/"
                                        f"{res[v]['col_ctl'][layer][ok[layer]].mean():+.2f}" for v in VARIANTS), flush=True)

    pat = pq.read_table(head_table("acts"), columns=list(CLASSES)).to_pydict()
    classes = {k: torch.tensor(pat[k]).view(L, H) > v for k, v in CLASSES.items()}
    lay = torch.arange(L)[:, None, None].expand(L, H, K)
    lines = [f"which lens on each side of each head's OV circuit, {MODEL}: per-token reads checked at the census "
             f"firings (24 per head, {saved['meta']['docs']:,} documents), {int(ok.sum())} firings from L1 whose attended position is "
             "before the query and whose attended token and the token after it are kept. Input S: the target is the "
             "attended token; input R: the token after it. Output: the head's actual output read through the R-lens "
             "or the S-lens from the MLP input. Control: another head of the layer.", "",
             "copy: Spearman over firings between the target's weight-only z in its own column and its z in the "
             "actual output's read (own / control)"]
    table = {"copy": {}, "column": {}, "classes": {}, "best": {}}
    for v in VARIANTS:
        r = res[v]
        row = {}
        for lo, hi in BANDS:
            b = ok & (lay >= lo) & (lay <= hi)
            row[f"L{lo}-{hi}"] = [spearman(r["w_own"][b], r["ctx"][b]), spearman(r["w_ctl"][b], r["ctx"][b])]
        table["copy"][v] = row
        lines.append(f"  {v}  " + "  ".join(f"{k} {a:+.2f}/{c:+.2f}" for k, (a, c) in row.items()))
    lines += ["", "column: Spearman over the vocabulary between the predicted column and the actual output's read, "
              "mean over firings (own / control)"]
    for v in VARIANTS:
        r = res[v]
        row = {}
        for lo, hi in BANDS:
            b = ok & (lay >= lo) & (lay <= hi)
            row[f"L{lo}-{hi}"] = [r["col_own"][b].mean().item(), r["col_ctl"][b].mean().item()]
        table["column"][v] = row
        lines.append(f"  {v}  " + "  ".join(f"{k} {a:+.3f}/{c:+.3f}" for k, (a, c) in row.items()))
    margin = {v: torch.where(ok, res[v]["col_own"] - res[v]["col_ctl"], torch.nan).nanmean(-1) for v in VARIANTS}  # [L, H]
    lines += ["", "by pattern class, L1-35: column margin (own − control), mean over the class's heads"]
    for name, cls in classes.items():
        c = cls.clone()
        c[0] = False
        x = {v: margin[v][c].nanmean().item() for v in VARIANTS}
        table["classes"][name] = x | {"heads": int(c.sum())}
        lines.append(f"  {name:9s} ({int(c.sum()):2d} heads)  " + "  ".join(f"{v} {x[v]:+.3f}" for v in VARIANTS))
    stack = torch.stack([margin[v] for v in VARIANTS])  # [4, L, H]
    best = stack.nan_to_num(-9).argmax(0)
    lines += ["", "share of heads each variant reads best (largest column margin), by band"]
    for lo, hi in BANDS:
        b = best[lo:hi + 1][~stack[:, lo:hi + 1].isnan().all(0)]
        x = {v: (b == i).float().mean().item() for i, v in enumerate(VARIANTS)}
        table["best"][f"L{lo}-{hi}"] = x
        lines.append(f"  L{lo}-{hi}: " + "  ".join(f"{v} {s:.2f}" for v, s in x.items()))
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"head_lens_sides_{STEM}.log").write_text(text + "\n")
    (SUMMARY / f"head_lens_sides_{STEM}.json").write_text(json.dumps(table))


if __name__ == "__main__":
    main()
