"""Qwen3-4B and Qwen3-0.6B descend from Qwen3-8B and Qwen3-1.7B by width pruning (findings §20).

Weights and the catalogs' ``acts`` and ``reads`` tables only. The model (``NEURON_LENS_MODEL``: Qwen3-4B or 0.6B, or a
base) and its parent (``PARENT``: the 8B or the 1.7B, the same kind), which have the same layers and heads:

- ``order``: every local catalog's neurons, per layer: Spearman ρ between a neuron's index and its RMS activation over
  the activation corpus (``acts``' ``std`` and ``mean``), and the near-silent neurons (both poles' ``act_top`` under
  0.05, as §2 and §19 count them) with their mean position in the layer. Indices a model was trained with from a
  random start are exchangeable: ρ ≈ 0, near-silent neurons at a mean position of 0.5.
- ``heads``: per layer, the correlation over heads of each head's q, k, v and o weight norms between the two models,
  and of the q_norm and k_norm gains over the head dimension; the control shuffles the parent's heads.
- ``map``: the residual's correspondence. Coordinates: each model's embedding columns over the vocabulary, each of the
  model's coordinates' best |r| among the parent's. A linear map ``B`` (a parent row times ``B`` is the model's row):
  least squares on the even layers' q and k rows (times the input norm's gain, unit length), which act on the normed
  residual and keep their head-internal basis (q_norm and k_norm); tested on the odd layers' rows, against the next
  head's rows. The neurons part uses ``B`` refitted on every layer's rows.
- ``neurons``: each neuron's score against each parent neuron of its layer, the mean cosine of its gate and up rows
  with the parent's mapped through ``B`` (rows times the post-attention norm's gain). Per layer: the best match's
  score against a null (the best score against the parent's layer half the depth away), the share of distinct best
  matches, a one-to-one assignment (Hungarian); ρ between the model's index and its match's RMS activation and index,
  and between matched pairs' RMS activations; the parent's RMS activation for the neurons the assignment leaves out
  (the pruned ones); where the near-silent neurons map; and the overlap of matched pairs' write reads (``write_R`` top
  16, each model through its own lens) against random pairs.
- ``coact`` (a GPU): both models over the same documents (the first ``--docs`` full 256-token rows of the dclm, zh and
  code activation slices, a third each); each neuron's activations (``down_proj``'s input, from position 1: position 0
  holds the sink's huge values) correlated with every parent neuron's of its layer, by Pearson's r and by Spearman's ρ.
  Per neuron and statistic: its top parents, the margin of the best over the second, whether the match is mutual (each
  the other's best), each parent's best neuron, and a null (the best and the margin in a parent layer half a group of
  ``--group`` layers away). No catalog or residual map: it runs where the models are, and writes
  ``data/checks/lineage_coact_<model>.pt``.
- ``certainty``: the weight match (``neurons``'s score) against three independent signals: the two activation
  correlations (``coact``) and the overlap of the write reads (``write_R``'s 64 tokens of both poles, each model through
  its own lens; a match needs 4 shared tokens and a unique best). A parent is *identified* when a signal's best is the
  weight match (chance, against shuffled neurons, is reported beside it); *supported* when the weight match is in a
  signal's top 4.
- ``refine``: round 2. The identified pairs (where two share a parent, the one with the better weight score) are anchors for maps of each layer's own (rows: gate and up; writes: down
  columns; ridge), cross-fitted in two folds so a held-out anchor tests the map; the other neurons are rescored against
  the parents no anchor holds (a one-to-one assignment) and checked against the three signals again. Every neuron ends
  with a parent and a status: identified (round 1 or 2), supported, or weights only.

    NEURON_LENS_MODEL=Qwen/Qwen3-4B python scripts/checks/lineage.py [--parts order,heads,map,neurons,certainty]
    NEURON_LENS_MODEL=Qwen/Qwen3-4B python scripts/checks/lineage.py --parts coact --device cuda

Writes ``data/summary/lineage_<model>.{json,log}``, and ``B`` and the correspondence (per layer: the assignment, the best
match, their scores) to ``data/checks/lineage_<model>.pt``.
"""
import argparse
import json
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from neuron_lens import arch
from neuron_lens.catalog import MODEL, catalog_dir, neuron_table
from neuron_lens.paths import DATA, SUMMARY
from neuron_lens.weights import tensors

PARENTS = {"Qwen/Qwen3-4B": "Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B": "Qwen/Qwen3-1.7B"}
PARENTS |= {k + "-Base": v + "-Base" for k, v in PARENTS.items()}
PARENT = PARENTS[MODEL]
STEM = arch.stem(MODEL)
CFG, PCFG = arch.text_config(MODEL), arch.text_config(PARENT)
L, H, K, HD = CFG.num_hidden_layers, CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim
assert (L, H, K, HD) == (PCFG.num_hidden_layers, PCFG.num_attention_heads, PCFG.num_key_value_heads, PCFG.head_dim)
SILENT, TOP, SEED, SEQ, VOCAB, MIN_READS = 0.05, 16, 0, 256, 160_000, 4
OUT = DATA / "checks" / f"lineage_{STEM}.pt"
COACT = DATA / "checks" / f"lineage_coact_{STEM}.pt"


def get(repo: str, names: list[str]) -> list[torch.Tensor]:
    w = tensors(repo, names)
    return [w[n].float() for n in names]


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def rho(a, b) -> float:
    return float(spearmanr(a, b).statistic)


def acts(model: str, layer: int) -> dict:
    """Per neuron: RMS activation (both poles' rows carry the same moments), and near-silent: both poles' ``act_top`` under ``SILENT``."""
    a = pq.read_table(neuron_table("acts", layer, model), columns=["std", "mean", "act_top"]).to_pandas()
    n = len(a) // 2
    top = np.fmax(a.act_top.values[:n], a.act_top.values[n:])
    return {"rms": np.hypot(a["std"].values[:n], a["mean"].values[:n]), "dead": top < SILENT}


def order() -> tuple[list[str], dict]:
    out = {}
    for d in sorted((DATA / "catalog").iterdir()):
        if not (d / "neurons" / "acts").is_dir():
            continue
        rows = []
        for layer in range(len(list((d / "neurons" / "acts").glob("L*.parquet")))):
            a = acts(d.name, layer)  # catalog_dir takes the folder's name as well as a Hub id
            n = len(a["rms"])
            rows.append({"rho": rho(np.arange(n), a["rms"]), "dead": int(a["dead"].sum()),
                         "dead_at": float(np.flatnonzero(a["dead"]).mean() / n) if a["dead"].any() else None})
        out[d.name] = rows
    lines = ["order: ρ(neuron index, RMS activation) by layer; near-silent neurons (count@mean position) where any"]
    for m, rows in out.items():
        r = np.array([x["rho"] for x in rows])
        lines.append(f"{m:14s} ρ {r.min():+.2f} to {r.max():+.2f}, |ρ| > 0.1 on {(abs(r) > 0.1).sum()}/{len(r)} layers: "
                     + " ".join(f"{x:+.2f}" for x in r))
        dead = [f"L{i}:{x['dead']}@{x['dead_at']:.2f}" for i, x in enumerate(rows) if x["dead"] > 20]
        if dead:
            lines.append(f"{'':14s} near-silent {' '.join(dead)}")
    return lines, out


def heads() -> tuple[list[str], dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    for layer in range(L):
        p = f"model.layers.{layer}.self_attn."
        names = [p + x for x in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "q_norm.weight", "k_norm.weight")]
        a, b = get(MODEL, names), get(PARENT, names)

        def norms(w):
            return {"q": w[0].view(H, HD, -1).norm(dim=(1, 2)), "k": w[1].view(K, HD, -1).norm(dim=(1, 2)),
                    "v": w[2].view(K, HD, -1).norm(dim=(1, 2)), "o": w[3].view(-1, H, HD).norm(dim=(0, 2))}
        na, nb = norms(a), norms(b)
        row = {x: float(np.corrcoef(na[x], nb[x])[0, 1]) for x in "qkvo"}
        row |= {"q_norm": float(np.corrcoef(a[4], b[4])[0, 1]), "k_norm": float(np.corrcoef(a[5], b[5])[0, 1]),
                "o_shuffled": float(np.corrcoef(na["o"], nb["o"][rng.permutation(H)])[0, 1])}
        rows.append(row)
    med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
    lo = {k: float(np.min([r[k] for r in rows])) for k in rows[0]}
    lines = [f"heads, {MODEL} against {PARENT}: correlation over heads, median (min) over layers",
             "  " + ", ".join(f"{k} {med[k]:+.2f} ({lo[k]:+.2f})" for k in rows[0])]
    return lines, {"layers": rows, "median": med}


def qk_rows(repo: str, layer: int) -> torch.Tensor:
    p = f"model.layers.{layer}."
    q, k, g = get(repo, [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "input_layernorm.weight"])
    return unit(torch.cat([q, k]) * g)


def residual_map() -> tuple[list[str], dict, torch.Tensor]:
    emb = arch.names(MODEL)["embed"]
    Ea, Eb = get(MODEL, [emb])[0], get(PARENT, [emb])[0]
    v = min(len(Ea), len(Eb))
    ok = (Ea[:v].norm(dim=1) > 1e-3) & (Eb[:v].norm(dim=1) > 1e-3)

    def z(x):
        x = x - x.mean(0)
        return x / x.norm(dim=0)
    C = (z(Ea[:v][ok]).T @ z(Eb[:v][ok])).abs()
    del Ea, Eb
    top2 = C.topk(2, dim=1)
    coords = {"best_r": [round(float(x), 3) for x in top2.values[:, 0].quantile(torch.tensor([0.1, 0.5, 0.9]))],
              "second_r": float(top2.values[:, 1].median()), "distinct": len(set(top2.indices[:, 0].tolist())),
              "of": int(C.shape[0])}
    del C
    ra = {layer: qk_rows(MODEL, layer) for layer in range(L)}
    rb = {layer: qk_rows(PARENT, layer) for layer in range(L)}
    train, test = range(0, L, 2), range(1, L, 2)
    B = torch.linalg.pinv(torch.cat([rb[i] for i in train]), rtol=1e-4) @ torch.cat([ra[i] for i in train])
    per = {}
    for layer in range(L):
        pred = unit(rb[layer] @ B)
        per[layer] = {"same": float((pred * ra[layer]).sum(1).median()),
                      "next_head": float((pred * ra[layer].roll(HD, 0)).sum(1).median())}
    held = [per[i]["same"] for i in test]
    B = torch.linalg.pinv(torch.cat([rb[i] for i in range(L)]), rtol=1e-4) @ torch.cat([ra[i] for i in range(L)])
    lines = [f"map: embedding coordinates, each of {coords['of']} {STEM} coordinates' best |r| among {PARENT}'s: "
             f"q10/50/90 {coords['best_r']}, second best (median) {coords['second_r']:.3f}, distinct best {coords['distinct']}",
             f"map: B fitted on the even layers' q/k rows; held-out (odd) rows' cosine with their prediction, median "
             f"{np.median(held):.2f} (by layer {min(held):.2f}-{max(held):.2f}); the next head's rows "
             f"{np.median([per[i]['next_head'] for i in test]):.2f}; training layers {np.median([per[i]['same'] for i in train]):.2f}"]
    return lines, {"coords": coords, "qk": per}, B


def mlp_rows(repo: str, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    p = f"model.layers.{layer}."
    g, gate, up = get(repo, [p + "post_attention_layernorm.weight", p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"])
    return gate * g, up * g


def write_ids(model: str, layer: int) -> list[list[int]]:
    return pq.read_table(neuron_table("reads", layer, model), columns=["write_R_ids"]).column(0).to_pylist()


def neurons(B: torch.Tensor) -> tuple[list[str], dict, dict]:
    rng = np.random.default_rng(SEED)
    rows, corr = [], {}
    mapped = {}

    def parent(layer):
        if layer not in mapped:
            gate, up = mlp_rows(PARENT, layer)
            mapped.clear()
            mapped[layer] = (unit(gate @ B), unit(up @ B))
        return mapped[layer]
    for layer in range(L):
        t0 = time.perf_counter()
        gate, up = (unit(x) for x in mlp_rows(MODEL, layer))
        null_layer = (layer + L // 2) % L
        pg, pu = parent(null_layer)
        null = ((gate @ pg.T + up @ pu.T) / 2).max(dim=1).values
        pg, pu = parent(layer)
        S = (gate @ pg.T + up @ pu.T) / 2
        best, arg = S.max(dim=1)
        thr = float(null.quantile(0.99))
        ri, ci = linear_sum_assignment(-S.numpy())
        assign = np.full(len(arg), -1)
        assign[ri] = ci
        score = S.numpy()[np.arange(len(assign)), assign]
        a, b = acts(MODEL, layer), acts(PARENT, layer)
        n, idx = len(assign), np.arange(len(assign))
        kept = np.zeros(S.shape[1], bool)
        kept[assign] = True
        ws, wp = write_ids(MODEL, layer), write_ids(PARENT, layer)
        sample = rng.choice(n, min(n, 2000), replace=False)

        def overlap(i, j):
            return len(set(ws[i][:TOP]) & set(wp[j][:TOP])) / TOP
        row = {"layer": layer, "best": float(best.median()), "second": float(S.topk(2, dim=1).values[:, 1].median()),
               "null99": thr, "above_null": float((best > thr).float().mean()),
               "distinct": len(set(arg.tolist())) / n, "assign_score": float(np.median(score)),
               "assign_is_best": float((assign == arg.numpy()).mean()),
               "rho_parent_rms": rho(idx, b["rms"][assign]), "rho_own_rms": rho(idx, a["rms"]),
               "rho_parent_index": rho(idx, assign), "rho_pair_rms": rho(a["rms"], b["rms"][assign]),
               "pruned_rms": float(np.median(b["rms"][~kept])), "kept_rms": float(np.median(b["rms"][kept])),
               "dead": int(a["dead"].sum()), "parent_dead": int(b["dead"].sum()),
               "dead_to_dead": float(b["dead"][assign[a["dead"]]].mean()) if a["dead"].any() else None,
               "live_to_dead": float(b["dead"][assign[~a["dead"]]].mean()),
               "pruned_dead": float(b["dead"][~kept].mean()),
               "write_overlap": float(np.mean([(overlap(i, assign[i]) + overlap(n + i, len(wp) // 2 + assign[i])) / 2
                                               for i in sample])),
               "write_overlap_random": float(np.mean([overlap(i, rng.integers(len(wp) // 2)) for i in sample]))}
        rows.append(row)
        corr[layer] = {"assign": torch.from_numpy(assign), "assign_score": torch.from_numpy(score),
                       "best": arg, "best_score": best}
        print(f"neurons L{layer:2d}: best {row['best']:.2f} (null 99% {thr:.2f}, above {row['above_null']:.2f}), "
              f"distinct {row['distinct']:.2f}; ρ index~parent RMS {row['rho_parent_rms']:+.2f}, ~parent index "
              f"{row['rho_parent_index']:+.2f}; pair RMS {row['rho_pair_rms']:+.2f}; write overlap {row['write_overlap']:.3f} "
              f"(random {row['write_overlap_random']:.3f}) ({time.perf_counter() - t0:.0f}s)", flush=True)
    lines = [f"neurons, {MODEL} against {PARENT} (score: mean cosine of gate and up rows through B)",
             "layer  best  null99  above  distinct  assign  ρ idx~par.RMS  ρ idx~own RMS  ρ idx~par.idx  ρ pair RMS  "
             "pruned RMS  kept RMS  dead (parent)  dead→dead  live→dead  pruned dead  write overlap (random)"]
    for r in rows:
        dd = f"{r['dead_to_dead']:.2f}" if r["dead_to_dead"] is not None else "  - "
        lines.append(f"L{r['layer']:<4d} {r['best']:.2f}  {r['null99']:.2f}    {r['above_null']:.2f}   {r['distinct']:.2f}      "
                     f"{r['assign_score']:.2f}    {r['rho_parent_rms']:+.2f}          {r['rho_own_rms']:+.2f}          "
                     f"{r['rho_parent_index']:+.2f}          {r['rho_pair_rms']:+.2f}       {r['pruned_rms']:.2e}    "
                     f"{r['kept_rms']:.2e}  {r['dead']:5d} ({r['parent_dead']:5d})   {dd}       {r['live_to_dead']:.2f}       "
                     f"{r['pruned_dead']:.2f}         {r['write_overlap']:.3f} ({r['write_overlap_random']:.3f})")
    return lines, {"layers": rows}, corr


def mlp_acts(model: str, layers: list[int], ids: torch.Tensor, dev: str, batch: int = 16) -> dict[int, torch.Tensor]:
    """``{layer: [positions, neurons]}`` (fp16, on the CPU): every neuron's activation, ``down_proj``'s input."""
    lm, stack = arch.load_model(model, dev, eager=False)
    n = arch.text_config(model).intermediate_size
    buf = {layer: torch.empty(len(ids) * (ids.shape[1] - 1), n, dtype=torch.float16) for layer in layers}
    at = [0]

    def keep(layer):
        def hook(_, args):
            x = args[0][:, 1:].reshape(-1, n).clamp(-6e4, 6e4)
            buf[layer][at[0]:at[0] + len(x)] = x.to(torch.float16).cpu()
        return hook
    hooks = [stack.layers[layer].mlp.down_proj.register_forward_pre_hook(keep(layer)) for layer in layers]
    with torch.inference_mode():
        for b in range(0, len(ids), batch):
            x = ids[b:b + batch].to(dev)
            stack(input_ids=x)
            at[0] += len(x) * (x.shape[1] - 1)
    for h in hooks:
        h.remove()
    del lm, stack
    torch.cuda.empty_cache()
    return buf


def zscores(x: torch.Tensor, dev: str, ranks: bool, chunk: int = 512) -> torch.Tensor:
    """Each column standardized (``ranks``: its ranks), so their products averaged over positions are Pearson's r
    (Spearman's ρ)."""
    out = torch.empty(x.shape, device=dev)
    for c in range(0, x.shape[1], chunk):
        col = x[:, c:c + chunk].to(dev)
        out[:, c:c + chunk] = col.argsort(0).argsort(0).float() if ranks else col.float()
    out -= out.mean(0)
    return out / out.std(0).clamp_min(1e-12)


def read_overlap(layer: int) -> torch.Tensor:
    """``[neurons, parent neurons]``: how many of the 64 ``write_R`` tokens their + poles share, plus their − poles'.
    Each write is read through its own model's R-lens, so this needs no map between the models."""
    from scipy.sparse import csr_matrix

    def incidence(model):
        ids = pq.read_table(neuron_table("reads", layer, model), columns=["write_R_ids"]).column(0).to_pylist()
        n = len(ids) // 2
        r, c = zip(*[(i % n, t + (VOCAB if i >= n else 0)) for i, row in enumerate(ids) for t in row if t >= 0])
        return csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, 2 * VOCAB))
    return torch.from_numpy((incidence(MODEL) @ incidence(PARENT).T).toarray())


def signals(layer: int, act: dict) -> dict[str, dict]:
    """Each independent signal's best parent per neuron (and top 4): the two activation correlations and the reads.
    A reads match counts when it shares at least ``MIN_READS`` tokens and beats the second best."""
    out = {stat: {"best": d[layer]["top"][:, 0].long(), "top4": d[layer]["top"].long(), "mutual": d[layer]["mutual"]}
           for stat, d in act.items()}
    R = read_overlap(layer)
    top = R.topk(4, dim=1)
    valid = (top.values[:, 0] >= MIN_READS) & (top.values[:, 0] > top.values[:, 1])
    back = R.argmax(0)
    out["reads"] = {"best": torch.where(valid, top.indices[:, 0], -1), "top4": torch.where(top.values > 0, top.indices, -1),
                    "mutual": valid & (back[top.indices[:, 0]] == torch.arange(len(R)))}
    return out


def weight_scores(B: torch.Tensor, layer: int, parent_rows: dict | None = None) -> torch.Tensor:
    """``[neurons, parent neurons]``: the mean cosine of gate and up rows, the parent's through ``B``."""
    gate, up = (unit(x) for x in mlp_rows(MODEL, layer))
    pg, pu = mlp_rows(PARENT, layer) if parent_rows is None else parent_rows
    return (gate @ unit(pg @ B).T + up @ unit(pu @ B).T) / 2


def certainty(B: torch.Tensor) -> tuple[list[str], dict, dict]:
    act = torch.load(COACT)
    rng = torch.Generator().manual_seed(SEED)
    rows, out = [], {}
    for layer in range(L):
        S = weight_scores(B, layer)
        wtop = S.topk(2, dim=1)
        wbest = wtop.indices[:, 0]
        n = len(S)
        perm = torch.randperm(n, generator=rng)
        row = {"layer": layer, "n": n}
        per = {}
        signals_cache = signals(layer, act["layers"])
        for stat, a in signals_cache.items():
            abest, top4 = a["best"], a["top4"]
            same = wbest == abest
            in4 = (top4 == wbest[:, None]).any(1)
            row[stat] = {"agree": float(same.float().mean()), "in_top4": float(in4.float().mean()),
                         "chance": float((wbest == abest[perm]).float().mean()),
                         "chance_top4": float((top4[perm] == wbest[:, None]).any(1).float().mean()),
                         "agree_if_mutual": float(same[a["mutual"]].float().mean()) if a["mutual"].any() else None,
                         "mutual": float(a["mutual"].float().mean())}
            per[stat] = {"same": same, "in4": in4}
        # identified: a signal's best match is the weight match; supported: the weight match is in a signal's top 4
        ident = per["pearson"]["same"] | per["spearman"]["same"] | per["reads"]["same"]
        supp = ~ident & (per["pearson"]["in4"] | per["spearman"]["in4"] | per["reads"]["in4"])
        chance = torch.zeros(n, dtype=torch.bool)
        for a in signals_cache.values():
            chance |= wbest == a["best"][perm]
        row["chance_any"] = float(chance.float().mean())
        row["identified"], row["supported"] = float(ident.float().mean()), float(supp.float().mean())
        row["w_margin_identified"] = float((wtop.values[ident, 0] - wtop.values[ident, 1]).median()) if ident.any() else None
        row["w_margin_rest"] = float((wtop.values[~ident, 0] - wtop.values[~ident, 1]).median()) if (~ident).any() else None
        rows.append(row)
        out[layer] = {"parent": wbest.int(), "identified": ident, "supported": supp,
                      "w_score": wtop.values[:, 0].half(), "w_margin": (wtop.values[:, 0] - wtop.values[:, 1]).half()}
        print(f"certainty L{layer:2d}: identified {row['identified']:.3f} (chance {row['chance_any']:.4f}), supported "
              f"{row['supported']:.3f}; " + "; ".join(f"{k} agree {row[k]['agree']:.3f}" for k in signals_cache), flush=True)
    lines = act["lines"] + [f"certainty, {MODEL} against {PARENT}: the weight match (neurons) against three independent "
                            "signals, the activation correlations (coact: Pearson, Spearman) and the write reads' shared "
                            "tokens; identified = a signal's best is the weight match, supported = the weight match is in a "
                            "signal's top 4; chance = the same against shuffled neurons",
                            "layer  identified (chance)  supported  agree: pearson  spearman  reads  (reads chance)  "
                            "weight margin: identified, rest"]
    for r in rows:
        lines.append(f"L{r['layer']:<4d} {r['identified']:.3f} ({r['chance_any']:.4f})     {r['supported']:.3f}      "
                     f"{r['pearson']['agree']:.3f}           {r['spearman']['agree']:.3f}     {r['reads']['agree']:.3f}  "
                     f"({r['reads']['chance']:.4f})         {r['w_margin_identified'] or 0:.3f}, {r['w_margin_rest'] or 0:.3f}")
    return lines, {"layers": rows, "coact": act["rows"]}, out


def ridge(X: torch.Tensor, Y: torch.Tensor, lam: float = 0.05) -> torch.Tensor:
    """``W`` minimizing ``|XW − Y|² + lam·mean(diag XᵀX)·|W|²``."""
    G = X.T @ X
    G.diagonal().add_(lam * G.diagonal().mean())
    return torch.linalg.solve(G, X.T @ Y)


def refine() -> tuple[list[str], dict, dict]:
    """Round 2: the identified pairs (``certainty``) as anchors for a map of each layer's own (rows: gate and up; writes:
    down columns), cross-fitted in two folds; the rest rescored against the parents no anchor holds, and checked against
    the activations again."""
    act = torch.load(COACT)["layers"]
    cert = torch.load(OUT)["certainty"]
    rng = torch.Generator().manual_seed(SEED)
    rows, out = [], {}
    for layer in range(L):
        p = f"model.layers.{layer}."
        g, u, dc = (unit(x) for x in (*mlp_rows(MODEL, layer), get(MODEL, [p + "mlp.down_proj.weight"])[0].T))
        pg, pu = mlp_rows(PARENT, layer)
        pd = get(PARENT, [p + "mlp.down_proj.weight"])[0].T
        pg, pu, pd = unit(pg), unit(pu), unit(pd)
        c = cert[layer]
        parent = c["parent"].long()
        # a parent has one child: where identified neurons share one, the best weight score keeps it
        ident = c["identified"].clone()
        idx = torch.nonzero(ident).squeeze(1)
        order = idx[torch.argsort(c["w_score"][idx].float(), descending=True)]
        seen = set()
        for i in order.tolist():
            if parent[i].item() in seen:
                ident[i] = False
            seen.add(parent[i].item())
        anchor = torch.nonzero(ident).squeeze(1)
        rest = torch.nonzero(~ident).squeeze(1)
        fold = torch.randint(0, 2, (len(anchor),), generator=rng)

        def fit(idx):
            a, b = anchor[idx], parent[anchor[idx]]
            return ridge(torch.cat([pg[b], pu[b]]), torch.cat([g[a], u[a]])), ridge(pd[b], dc[a])

        def score(maps, kids):
            Bl, Wl = maps
            return (g[kids] @ unit(pg @ Bl).T + u[kids] @ unit(pu @ Bl).T + dc[kids] @ unit(pd @ Wl).T) / 3
        held = []
        for k in (0, 1):  # held-out anchors: does a map fitted without them put their parent first?
            kids = anchor[fold == k]
            held.append(score(fit(fold != k), kids).argmax(1) == parent[kids])
        held = torch.cat(held)
        S = score(fit(torch.ones(len(anchor), dtype=torch.bool)), rest)
        S[:, parent[anchor]] = -2  # parents the anchors hold
        top = S.topk(2, dim=1)
        new = top.indices[:, 0]
        ri, ci = linear_sum_assignment(-S.numpy())
        one = torch.from_numpy(ci)[torch.argsort(torch.from_numpy(ri))]
        perm = torch.randperm(len(rest), generator=rng)
        same = torch.zeros(len(rest), dtype=torch.bool)
        chance = torch.zeros(len(rest), dtype=torch.bool)
        in4 = torch.zeros(len(rest), dtype=torch.bool)
        for d in signals(layer, act).values():  # judged on the one-to-one assignment, which is what each neuron keeps
            ab = d["best"][rest]
            same |= one == ab
            chance |= one == ab[perm]
            in4 |= (d["top4"][rest] == one[:, None]).any(1)
        final = parent.clone()
        final[rest] = one  # anchors hold distinct parents, the rest are assigned among the others: one child per parent
        status = torch.zeros(len(parent), dtype=torch.int8)  # 0 weights only, 1 identified (round 1), 2 identified (round 2), 3 supported
        status[anchor] = 1
        status[rest[same]] = 2
        status[rest[~same & in4]] = 3
        n = len(parent)
        row = {"layer": layer, "round1": len(anchor) / n, "shared_parent": float((c["identified"] & ~ident).sum()) / n,
               "held_out_recovered": float(held.float().mean()),
               "round2": float(same.sum()) / n, "round2_chance": float(chance.sum()) / n, "supported": float((~same & in4).sum()) / n,
               "identified": float((status == 1).sum() + (status == 2).sum()) / n,
               "margin_rest": float((top.values[:, 0] - top.values[:, 1]).median()) if len(rest) else None,
               "one_to_one_agrees": float((one == new).float().mean()) if len(rest) else None}
        rows.append(row)
        out[layer] = {"parent": final.int(), "status": status}
        print(f"refine L{layer:2d}: identified {row['round1']:.3f} → {row['identified']:.3f} (round 2 {row['round2']:.3f}, chance "
              f"{row['round2_chance']:.4f}); supported {row['supported']:.3f}; held-out anchors recovered by weights alone "
              f"{row['held_out_recovered']:.3f}", flush=True)
    lines = [f"refine, {MODEL} against {PARENT}: round 2 with each layer's own maps fitted on the identified pairs",
             "layer  round 1  held-out anchors recovered  round 2 (chance)  identified  supported  weights only"]
    lines += [f"L{r['layer']:<4d} {r['round1']:.3f}    {r['held_out_recovered']:.3f}                       {r['round2']:.3f} "
              f"({r['round2_chance']:.4f})   {r['identified']:.3f}       {r['supported']:.3f}      "
              f"{1 - r['identified'] - r['supported']:.3f}" for r in rows]
    return lines, {"layers": rows}, out


STATS = {"pearson": False, "spearman": True}


def coact(dev: str, n_docs: int, group: int) -> tuple[list[str], dict, dict]:
    from neuron_lens.catalog import tokenizer
    from neuron_lens.corpus import _rows, act_file
    torch.backends.cuda.matmul.allow_tf32 = True
    tok = tokenizer(MODEL)
    ids = []
    for name in ("dclm", "zh", "code"):
        full = (r for r in _rows(act_file(name), tok, SEQ) if len(r) == SEQ)
        ids += [r for r, _ in zip(full, range(n_docs // 3))]
    ids = torch.tensor(ids)
    T = len(ids) * (SEQ - 1)
    rows, data = {k: [] for k in STATS}, {k: {} for k in STATS}
    for g0 in range(0, L, group):
        layers = list(range(g0, min(L, g0 + group)))
        t0 = time.perf_counter()
        a = {m: mlp_acts(m, layers, ids, dev) for m in (MODEL, PARENT)}
        print(f"coact L{layers[0]}-L{layers[-1]}: activations over {T:,} positions ({time.perf_counter() - t0:.0f}s)", flush=True)
        for i, layer in enumerate(layers):
            wrong = layers[(i + len(layers) // 2) % len(layers)]
            for stat, ranks in STATS.items():
                z = zscores(a[MODEL][layer], dev, ranks)
                C = z.T @ zscores(a[PARENT][layer], dev, ranks) / T  # [neurons, parent neurons]
                null = (z.T @ zscores(a[PARENT][wrong], dev, ranks) / T).topk(2, dim=1).values
                top = C.topk(4, dim=1)
                back = C.max(dim=0)
                mutual = back.indices[top.indices[:, 0]] == torch.arange(len(C), device=dev)
                margin = top.values[:, 0] - top.values[:, 1]
                thr = float((null[:, 0] - null[:, 1]).quantile(0.999))
                ok = mutual & (margin > thr)
                row = {"layer": layer, "best": float(top.values[:, 0].median()), "second": float(top.values[:, 1].median()),
                       "null_best": float(null[:, 0].median()), "margin": float(margin.median()), "null_margin999": thr,
                       "above_null": float((margin > thr).float().mean()), "mutual": float(mutual.float().mean()),
                       "mutual_above": float(ok.float().mean()), "distinct": len(set(top.indices[:, 0].tolist())) / len(C)}
                rows[stat].append(row)
                data[stat][layer] = {"top": top.indices.cpu().int(), "top_r": top.values.cpu().half(),
                                     "null": null.cpu().half(), "null_margin999": thr, "mutual": mutual.cpu(),
                                     "parent_best": back.indices.cpu().int(), "parent_best_r": back.values.cpu().half()}
                print(f"coact L{layer:2d} {stat:8s}: best {row['best']:.2f} (2nd {row['second']:.2f}; wrong layer "
                      f"{row['null_best']:.2f}); margin {row['margin']:.3f} (null 99.9% {thr:.3f}); mutual {row['mutual']:.3f}, "
                      f"and above {row['mutual_above']:.3f}; distinct {row['distinct']:.3f}", flush=True)
                del z, C
        del a
    lines = []
    for stat in STATS:
        lines += [f"coact, {MODEL} against {PARENT}: {stat} correlation of activations over {T:,} positions ({len(ids)} "
                  f"documents of {SEQ}, from position 1)",
                  "layer  best  2nd   wrong layer  margin  null margin 99.9%  above  mutual  mutual+above  distinct"]
        lines += [f"L{r['layer']:<4d} {r['best']:.2f}  {r['second']:.2f}  {r['null_best']:.2f}         {r['margin']:.3f}   "
                  f"{r['null_margin999']:.3f}              {r['above_null']:.3f}  {r['mutual']:.3f}   {r['mutual_above']:.3f}         "
                  f"{r['distinct']:.3f}" for r in rows[stat]]
        lines.append("")
    return lines, {"layers": rows, "positions": T}, data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="order,heads,map,neurons")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--docs", type=int, default=192)
    ap.add_argument("--group", type=int, default=12)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    if args.parts == "coact":  # on a GPU machine, without the catalogs; the summary is written by ``certainty``
        lines, rows, data = coact(args.device, args.docs, args.group)
        COACT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"lines": lines, "rows": rows, "layers": data}, COACT)
        print("\n".join(lines))
        return
    assert catalog_dir(MODEL).exists() and catalog_dir(PARENT).exists(), "both catalogs are needed locally"
    path = SUMMARY / f"lineage_{STEM}.json"
    table = json.loads(path.read_text()) if path.exists() else {}
    saved = torch.load(OUT) if OUT.exists() else {}
    for part in args.parts.split(","):
        if part == "map":
            lines, table[part], saved["B"] = residual_map()
        elif part == "neurons":
            lines, table[part], saved["neurons"] = neurons(saved["B"])
        elif part == "certainty":
            lines, table[part], saved["certainty"] = certainty(saved["B"])
        elif part == "refine":
            lines, table[part], saved["refine"] = refine()
        else:
            lines, table[part] = {"order": order, "heads": heads}[part]()
        table[f"{part}_log"] = lines
        OUT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, OUT)
        path.write_text(json.dumps(table, ensure_ascii=False))
        text = "\n\n".join("\n".join(table[f"{p}_log"]) for p in ("order", "heads", "map", "neurons", "certainty", "refine")
                           if f"{p}_log" in table)
        (SUMMARY / f"lineage_{STEM}.log").write_text(text + "\n")
        print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
