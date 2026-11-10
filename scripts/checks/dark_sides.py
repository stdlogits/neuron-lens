"""What the census's ``dark`` depends on (findings §7, §13).

``sides`` (no model), each band split by the ``legible`` table's ``dark``:

- **the write side**, on §3's random poles (``causal_lens.py``'s file): the share whose R-lens write read predicts the effect
  of steering them (1× and 4× ``act_top``) and of removing them at their own top firings, by §3's test (r past the 99th
  percentile of mismatched pairs, pooled over layers), with the mismatched pairs' share and the median r;
- **what drives a firing** (``sources``, first ``N_CONTEXTS`` firings): the mean share of the activation from the embedding,
  the heads and earlier neurons;
- **the read against the residual**: the share of ``read_R_z``'s top 64 in ``before_R_z``'s, against a random other pole of the
  layer and sign. The census leaves this out of ``dark`` (see ``null``), but it is where a read direction that a lens
  can't transport shows up.

``null`` (the model, MPS or CUDA; Qwen3-8B's layout): what selecting firings on a direction guarantees. Random unit
directions in the MLP's normed input stand in for neurons: per direction, the strongest position of each of ``--docs``
documents, its top 64 documents, the mean of the residual there through the R-lens ``mid`` target (top 64 by z across the
directions), and the direction's own read, as ``read_R``'s (the direction times the norm weight, unit, through the same
lens). The mean z of the residual read's tokens under the direction's read, against mismatched pairs at 5% FDR, with the
same test on the layer's real poles beside it (their ``before_R_z`` lists), and the top-64 overlap for both.

``norm`` (no model): ``read_R`` with the read rows as they meet the input norm's weight γ. The catalog's row × γ is the
covector the activation applies to the residual, which suits the S-lens, a pullback. The R-lens transports a vector. On each
of ``--layers``: row × γ, the raw row, and row / γ, each read through the R-lens ``mid`` target and scored against
``before_R_z`` and the exact drivers, with γ's spread.

Writes ``data/summary/dark_sides_<model>.{log,json}``, a section per part.

    python scripts/checks/dark_sides.py sides
    python scripts/checks/dark_sides.py null --layers 16,24,30
    NEURON_LENS_MODEL=google/gemma-4-12B python scripts/checks/dark_sides.py norm --layers 6,18,30,42
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent)]
from causal_lens import CAUSAL, random_neurons  # noqa: E402
from neuron_lens import arch  # noqa: E402
from neuron_lens.catalog import MODEL, TOP, acts_path, column, doc_tokens, kept, kept_index, lobe_signs, neuron_table  # noqa: E402
from neuron_lens.neurons import directions, mlp_names, read_rows, softcap  # noqa: E402
from neuron_lens.paths import SUMMARY  # noqa: E402
from neuron_lens.readout import layer_reads, rms_norm_columns, transport  # noqa: E402
from neuron_lens.stats import bh, null_p, pearson  # noqa: E402
from neuron_lens.weights import tensors  # noqa: E402

STEM = arch.stem(MODEL)
CFG = arch.text_config(MODEL)
L, F = CFG.num_hidden_layers, CFG.intermediate_size
BANDS = tuple(arch.bands(L)) + ((0, L - 1),)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
PASS_Q = 0.99
CHUNK = 2048
TARGETS = ("steer 1x", "steer 4x", "ablate win")
CLASSES = ("dark", "not dark", "silent")


def band_name(lo: int, hi: int) -> str:
    return f"L{lo}-L{hi}"


def flags(layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    t = pq.read_table(neuron_table("legible", layer), columns=["silent", "dark"])
    return tuple(torch.from_numpy(t[c].to_numpy(zero_copy_only=False).copy()) for c in ("silent", "dark"))


def shared(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per row, the share of ``a``'s tokens in ``b`` (−1 is padding)."""
    out = torch.zeros(len(a))
    for s in range(0, len(a), CHUNK):
        x, y = a[s:s + CHUNK], b[s:s + CHUNK]
        out[s:s + CHUNK] = (((x[:, :, None] == y[:, None, :]).any(-1)) & (x >= 0)).float().mean(1)
    return out


def sides() -> tuple[list[str], dict]:
    from transformers import AutoTokenizer

    saved = torch.load(CAUSAL / f"{STEM}-causal.pt", weights_only=False, mmap=True)
    picks, nv = saved["picks"], len(AutoTokenizer.from_pretrained(MODEL))
    kv = (torch.load(arch.counts_path(MODEL), weights_only=False)["counts"][:nv] >= 5).to(DEV)
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted({nm["unembed"], nm["norm"]}))
    unembed, gamma = head[nm["unembed"]][:nv].to(DEV, torch.float32), head[nm["norm"]].to(DEV, torch.float32)
    J_out = arch.lens(MODEL, "R_out", arch.fits(MODEL)[0])
    rand = set(random_neurons(L, F, saved["meta"]["per_layer"]))
    targets = {"steer 1x": saved["steer"][:, :, 0], "steer 4x": saved["steer"][:, :, 1], "ablate win": saved["ablate_win"]}
    fired = (saved["fired"] > 0).any(-1)
    recs = []  # (layer, class, target, r, control r)
    g = torch.Generator().manual_seed(0)
    per = {}
    for layer in range(L):
        silent, dark = flags(layer)
        idx = [i for i, p in enumerate(picks) if p[0] == layer and p in rand]
        if idx:
            nt = torch.tensor([picks[i][1] for i in idx])
            sl, dk = torch.cat([silent[nt], silent[F + nt]]), torch.cat([dark[nt], dark[F + nt]])
            other = torch.arange(len(idx)).roll(1)
            jac = J_out[layer] if layer < len(J_out) else None
            lg = (unembed @ rms_norm_columns(transport(directions(MODEL, layer, DEV)["write"][:, nt.tolist()], jac), gamma)).T
            pred, ctrl = torch.cat([lg, -lg]), torch.cat([lg[other], -lg[other]])
            for name, t in targets.items():
                y = torch.cat([t[idx, 0, :nv], t[idx, 1, :nv]]).to(DEV, torch.float32)
                keep = torch.ones(len(y), dtype=torch.bool) if name.startswith("steer") else torch.cat([fired[idx, 0], fired[idx, 1]])
                keep &= y.abs().sum(1).cpu() > 0
                r, rc = pearson(pred[:, kv], y[:, kv]).cpu(), pearson(ctrl[:, kv], y[:, kv]).cpu()
                recs += [(layer, "silent" if sl[j] else "dark" if dk[j] else "not dark", name, r[j].item(), rc[j].item())
                         for j in range(len(r)) if keep[j]]
        perm = torch.cat([torch.randperm(F, generator=g), F + torch.randperm(F, generator=g)])
        read = column(pq.read_table(neuron_table("reads", layer), columns=["read_R_z_ids"]), "read_R_z_ids").long()
        before = column(pq.read_table(neuron_table("firings", layer), columns=["before_R_z_ids"]), "before_R_z_ids").long()
        agree, ctl = shared(read, before), shared(read, before[perm])
        row = {"dark": dark & ~silent, "not dark": ~dark & ~silent}
        out = {c: {"n": int(m.sum()), "agree": agree[m].sum().item(), "agree_ctl": ctl[m].sum().item()} for c, m in row.items()}
        if layer:
            src = pq.read_table(neuron_table("sources", layer), columns=["emb_share", "heads_share", "neurons_share"])
            for k in src.column_names:
                v = torch.nan_to_num(torch.from_numpy(src[k].to_numpy().copy()).float())
                for c, m in row.items():
                    out[c][k] = v[m].sum().item()
        per[layer] = out
        print(f"{time.strftime('%H:%M:%S')} sides L{layer}", flush=True)
    thr = {t: torch.tensor([r[4] for r in recs if r[2] == t]).quantile(PASS_Q).item() for t in TARGETS}
    table = {"thresholds": thr, "write": {}, "residual": {}}
    lines = [f"the write side, {MODEL}: §3's random poles by the census's dark; share whose write read predicts the target "
             f"(r past the mismatched pairs' 99th percentile, pooled over layers: {thr}) [mismatched pairs]; median r"]
    for t in TARGETS:
        lines += ["", t]
        table["write"][t] = {}
        for lo, hi in BANDS:
            cells, entry = [], {}
            for c in CLASSES:
                rs = [r for r in recs if lo <= r[0] <= hi and r[1] == c and r[2] == t]
                if not rs:
                    continue
                r_, rc = torch.tensor([x[3] for x in rs]), torch.tensor([x[4] for x in rs])
                entry[c] = {"n": len(rs), "pass": (r_ > thr[t]).float().mean().item(), "pass_control": (rc > thr[t]).float().mean().item(), "r_med": r_.median().item()}
                cells.append(f"{c} n={entry[c]['n']:<5d}{entry[c]['pass']:.2f} [{entry[c]['pass_control']:.2f}] r {entry[c]['r_med']:.2f}")
            table["write"][t][band_name(lo, hi)] = entry
            lines.append(f"{band_name(lo, hi):9s}" + " | ".join(cells))
    lines += ["", "what drives a firing (embedding / heads / earlier neurons, mean share over the first firings) and read_R_z's top 64 "
              "in before_R_z's [another pole]; dark | not dark"]
    for lo, hi in BANDS:
        name, cells, entry = band_name(lo, hi), [], {}
        for c in ("dark", "not dark"):
            rows = [per[l][c] for l in per if lo <= l <= hi]
            src = [per[l][c] for l in per if lo <= l <= hi and l]
            n, ns = sum(r["n"] for r in rows) or 1, sum(r["n"] for r in src) or 1
            entry[c] = {k: sum(r[k] for r in src) / ns for k in ("emb_share", "heads_share", "neurons_share")} | {
                "agree": sum(r["agree"] for r in rows) / n, "agree_ctl": sum(r["agree_ctl"] for r in rows) / n, "n": n}
            cells.append(f"{entry[c]['emb_share']:.3f} / {entry[c]['heads_share']:.3f} / {entry[c]['neurons_share']:.3f}  agree {entry[c]['agree']:.3f} [{entry[c]['agree_ctl']:.3f}]")
        table["residual"][name] = entry
        lines.append(f"{name:9s}" + " | ".join(cells))
    return lines, table


def null(layers: list[int], n_docs: int) -> tuple[list[str], dict]:
    from transformers import AutoModelForCausalLM

    leg = importlib.import_module("09_legible")
    n_dir, seq = 2048, 128
    keep, kidx = kept(), kept_index()
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted(set(nm.values())))
    unembed, gamma = head[nm["unembed"]][keep].to(DEV, torch.float16), head[nm["norm"]].to(DEV, torch.float32)
    J_mid = arch.lens(MODEL, "R_mid", arch.fits(MODEL)[0])
    saved = torch.load(acts_path(), weights_only=False, map_location="cpu", mmap=True)
    docs = [d[:seq].long() for d in doc_tokens(saved)[:n_docs] if len(d) >= 32]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    store = {l: [] for l in layers}
    hooks = [model.model.layers[l].post_attention_layernorm.register_forward_pre_hook(
        lambda _, a, l=l: store[l].append(a[0][:, 1:].float().cpu().half())) for l in layers]
    ends = []
    with torch.inference_mode():
        for i in range(0, len(docs), 32):
            b = docs[i:i + 32]
            n = max(map(len, b))
            ids = torch.stack([torch.cat([d, torch.zeros(n - len(d), dtype=torch.long)]) for d in b]).to(DEV)
            model(input_ids=ids, attention_mask=(torch.stack([torch.arange(n) < len(d) for d in b]).long().to(DEV)))
            ends += [len(d) - 1 for d in b]
    for h in hooks:
        h.remove()
    del model
    g = torch.Generator().manual_seed(0)
    lines = [f"selection null, {MODEL}: {n_dir} random directions, top {TOP} of {len(docs)} documents of up to {seq} tokens; "
             "share passing read>before at 5% FDR (mean z, against mismatched pairs), and the top-64 overlap [mismatched]; real poles beside"]
    table = {}
    for layer in layers:
        xs, owner, d = [], [], 0
        for batch in store[layer]:
            for j in range(len(batch)):
                xs.append(batch[j, :ends[d]])
                owner.append(torch.full((ends[d],), d))
                d += 1
        X, D = torch.cat(xs).float().to(DEV), torch.cat(owner).to(DEV)
        X = X / X.pow(2).mean(1, keepdim=True).sqrt()
        norm = tensors(MODEL, [mlp_names(MODEL, layer)["norm"]])[mlp_names(MODEL, layer)["norm"]].float().to(DEV)
        U = torch.randn(X.shape[1], n_dir, generator=g).to(DEV)
        U = U / U.norm(dim=0, keepdim=True)
        acts = (X * norm) @ U
        best = torch.stack([acts[D == i].max(0).values for i in range(d)])
        where = torch.stack([(D == i).nonzero().squeeze(1)[acts[D == i].argmax(0)] for i in range(d)])
        tops = best.topk(TOP, dim=0).indices
        mean_x = torch.stack([X[where[tops[:, j], j]].mean(0) for j in range(n_dir)], 1)
        bz = layer_reads(mean_x, unembed, gamma, J_mid[layer], signs=(1,), k=TOP, kz=TOP)["zids"][1].long()
        c = U * norm[:, None]
        c = c / c.norm(dim=0, keepdim=True)
        perm = torch.randperm(n_dir, generator=g)
        leg.CHUNK = 768
        rr = rms_norm_columns(transport(c, J_mid[layer]), gamma).T.half()
        m, n = leg.mean_z(rr, unembed, bz, bz >= 0, perm)
        rz = layer_reads(c, unembed, gamma, J_mid[layer], signs=(1,), k=TOP, kz=TOP)["zids"][1].long()
        rand = {"pass": (bh(null_p(m.double(), n.double())) <= 0.05).float().mean().item(), "shared": shared(rz.cpu(), bz.cpu()).mean().item(),
                "shared_ctl": shared(rz.cpu(), bz[perm].cpu()).mean().item()}
        silent, _ = flags(layer)
        w = directions(MODEL, layer)
        a, b = w["gate"], w["up"]
        a, b = a / a.norm(dim=1, keepdim=True), b / b.norm(dim=1, keepdim=True)
        rr = rms_norm_columns(transport(read_rows(a, b, lobe_signs(layer)).to(DEV).T, J_mid[layer]), gamma).T.half()
        fr = pq.read_table(neuron_table("firings", layer), columns=["before_R_z_ids"])
        rd = pq.read_table(neuron_table("reads", layer), columns=["read_R_z_ids"])
        ids = column(fr, "before_R_z_ids").long()
        k = torch.where(ids >= 0, kidx[ids.clamp_min(0)], torch.tensor(-1))
        perm = torch.cat([torch.randperm(F, generator=g), F + torch.randperm(F, generator=g)])
        m, n = leg.mean_z(rr, unembed, k, k >= 0, perm)
        ok = ~m.isnan() & ~silent
        ours = column(rd, "read_R_z_ids").long()
        real = {"pass": (bh(null_p(m[ok].double(), n[ok].double())) <= 0.05).float().mean().item(),
                "shared": shared(ours, ids)[ok].mean().item(), "shared_ctl": shared(ours, ids[perm])[ok].mean().item()}
        table[layer] = {"random": rand, "poles": real}
        lines.append(f"L{layer}: random directions pass {rand['pass']:.2f}, shared {rand['shared']:.3f} [{rand['shared_ctl']:.3f}]; "
                     f"the layer's poles pass {real['pass']:.2f}, shared {real['shared']:.3f} [{real['shared_ctl']:.3f}]")
    return lines, table


def norm_variants(layers: list[int]) -> tuple[list[str], dict]:
    keep = kept()
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted({nm["unembed"], nm["norm"]}))
    unembed, gamma = head[nm["unembed"]][keep].to(DEV, torch.float16), head[nm["norm"]].to(DEV, torch.float32)
    J_mid = arch.lens(MODEL, "R_mid", arch.fits(MODEL)[0])
    cap = softcap(MODEL)
    g = torch.Generator().manual_seed(0)
    lines = [f"read_R by how the rows meet the input norm's weight γ, {MODEL}: read_R_z's top 64 in before_R_z's [another pole]; share of "
             "the plain read's top 10 in the top 20 exact drivers"]
    table = {}
    for layer in layers:
        names = mlp_names(MODEL, layer)
        w = {k: tensors(MODEL, [names[k]])[names[k]].float().to(DEV) for k in ("gate", "up", "norm")}
        n = w["norm"]
        q = n.abs().quantile(torch.tensor([0.5, 0.99], device=DEV)).tolist()
        fr = pq.read_table(neuron_table("firings", layer), columns=["before_R_z_ids", "n_firings"])
        before, ok = column(fr, "before_R_z_ids").long(), torch.from_numpy(fr["n_firings"].to_numpy().copy()) > 0
        drv = column(pq.read_table(neuron_table("drivers", layer), columns=["drivers_ids"]), "drivers_ids").long()
        perm = torch.cat([torch.randperm(F, generator=g), F + torch.randperm(F, generator=g)])
        lobe = lobe_signs(layer)
        table[layer] = {"std_over_mean": (n.std() / n.mean()).item(), "p99_over_median": q[1] / q[0]}
        lines.append(f"L{layer}: γ std/mean {table[layer]['std_over_mean']:.2f}, 99th percentile / median {table[layer]['p99_over_median']:.0f}")
        for name, scale in (("row x norm", n), ("row", torch.ones_like(n)), ("row / norm", 1 / n.abs().clamp_min(1e-3) * n.sign())):
            a, b = w["gate"] * scale, w["up"] * scale
            a, b = a / a.norm(dim=1, keepdim=True), b / b.norm(dim=1, keepdim=True)
            c = read_rows(a.cpu(), b.cpu(), lobe).to(DEV)
            reads = [layer_reads(c[i * F:(i + 1) * F].T, unembed, gamma, J_mid[layer], signs=(1,), k=TOP, kz=TOP, softcap=cap) for i in (0, 1)]
            ids = keep[torch.cat([r["ids"][1] for r in reads])].long()
            zids = keep[torch.cat([r["zids"][1] for r in reads])].long()
            e = {"agree": shared(zids, before)[ok].mean().item(), "agree_ctl": shared(zids, before[perm])[ok].mean().item(),
                 "drivers": shared(ids[:, :10], drv[:, :20])[ok].mean().item()}
            table[layer][name] = e
            lines.append(f"  {name:11s} agree {e['agree']:.3f} [{e['agree_ctl']:.3f}]  drivers {e['drivers']:.3f}")
    return lines, table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("sides", "null", "norm"))
    ap.add_argument("--layers", default="16,24,30")
    ap.add_argument("--docs", type=int, default=3000)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    lines, table = {"sides": sides, "null": lambda: null(layers, args.docs), "norm": lambda: norm_variants(layers)}[args.part]()
    path = SUMMARY / f"dark_sides_{STEM}.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    out[args.part], out[f"{args.part}_log"] = table, lines
    path.write_text(json.dumps(out))
    text = "\n\n".join("\n".join(out[f"{p}_log"]) for p in ("sides", "null", "norm") if f"{p}_log" in out)
    print(text)
    (SUMMARY / f"dark_sides_{STEM}.log").write_text(text + "\n")


if __name__ == "__main__":
    main()
