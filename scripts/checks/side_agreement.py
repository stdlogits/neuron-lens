"""Do a pole's weight-only reads match where it fires? Every pole of the catalog, no model (findings §5, §6).

Each pole's top firings are the catalog's (``acts``: the strongest position in each of its top 24
documents, ``06_activations.py``), those with a positive value; at each, the peak token and the token
after it (none past the document's end). Token lists are the catalog's top 10 (``08_reads.py``, over
``catalog.kept()``); a firing on a token outside them counts as a miss.

- **predict** (``write_R``): the share of firings whose next token is in the top 10 of the pole's
  ``write_R`` read. A pole that predicts the next word scores high.
- **suppress** (``suppress_R``): the same with the opposite pole's ``write_R`` top 10, which is this
  pole's bottom 10 (its most negative logits: Gurnee et al. 2024's suppression neurons).
- **read → peak token** (``read_R``, ``read_S``, ``read_S_z``): the share whose peak token is in the top
  10 of the read direction's R-lens read (from the MLP input), or of its S-lens read, raw and
  layer-contrastive.
- **read → exact drivers** (``drivers_R``, ``drivers_S``, ``drivers_S_z``): the share of each read's top
  10 among the pole's top 20 exact drivers at the token (``07_drivers.py``).

Control: the same with the lists of a random other pole of the same layer and sign (one permutation
per layer, seed 0). Per layer, the mean share and the share of poles past a third, matched and control.

Then:
- **Gurnee et al.'s weight classifier** over the last third of the layers (Qwen3-8B: L24-L35): poles in
  the top 10% of ``vocab_kurt`` (kurtosis of the write's logit-lens logits over the kept tokens), by
  the sign of ``vocab_skew``, against their predict and suppress shares (mean, share past a third).
  Per layer, each neuron in its layer's top 10% of kurtosis has a positive-skew pole P and a
  negative-skew pole S: the share of P past a third on predict, of S on suppress, of both, and the
  median ``act_top`` of P and of S.
- **Shared tokens:** per layer, the mean number of tokens two of a pole's lists share (control: a
  random pole of the same layer and sign, seed 1): ``read_S`` and ``read_S_z`` with ``read_R`` (input
  embeddings against output unembeddings, so a shared token is a shared concept), ``read_S_z`` with
  ``write_R``, and ``write_S``'s top 10 with the top 20 exact drivers (early poles write back the
  footprint of what drives them).

Documents come from ``catalog.flat_tokens``: the activations file, else rebuilt from the corpus, as
released. Log and JSON in
``data/summary/side_agreement_<model>.{log,json}`` (figures 10 and 11).

    python scripts/checks/side_agreement.py
    NEURON_LENS_MODEL=google/gemma-4-12B python scripts/checks/side_agreement.py   # any catalog model
"""
import json

import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import MODEL, column, firing_tokens, flat_tokens, neuron_table
from neuron_lens.paths import SUMMARY

STEM = arch.stem(MODEL)
PAST = 1 / 3
K = 10  # each list's top 10, as findings §5-§6 define the shares
DRIVERS = 20  # the top 20 exact drivers
GURNEE_TOP = 0.1  # the top 10% of kurtosis
READS = ("read_R", "read_S", "read_S_z")
SHARED = (("read_S", "read_R"), ("read_S_z", "read_R"), ("read_S_z", "write_R"), ("write_S", "drivers"))
LABELS = {"write_R": "predict", "suppress_R": "suppress", "drivers_R": "read_R→drv", "drivers_S": "read_S→drv",
          "drivers_S_z": "read_S_z→drv"}


def scalar(t, name: str) -> torch.Tensor:
    return torch.from_numpy(t.column(name).to_numpy().copy())


def hits(ids: torch.Tensor, tok: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    """Per row, the share of usable items (``ok``) whose token is among the row's ``ids``."""
    m = (tok[:, :, None] == ids[:, None, :]).any(-1) & ok
    return m.sum(1) / ok.sum(1).clamp_min(1)


def in_list(ids: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per row, how many of ``ids`` are among ``ref``."""
    return (ids[:, :, None] == ref[:, None, :]).any(-1).float().sum(1)


def gurnee(per: list[dict], layers: range) -> tuple[list[str], dict]:
    """Gurnee et al.'s weight classifier against the measured shares, and each high-kurtosis neuron's
    P and S poles per layer."""
    x = {k: torch.cat([p[k] for p in per]) for k in per[0]}
    hi = x["kurt"] > x["kurt"].quantile(1 - GURNEE_TOP)
    out = {"layers": [layers[0], layers[-1]], "top": GURNEE_TOP, "classes": {}, "per_layer": []}
    lines = [f"Gurnee's weight classifier, L{layers[0]}-L{layers[-1]}: poles in the top {GURNEE_TOP:.0%} of "
             "vocabulary kurtosis, by skew sign, against the measured shares (mean; share of poles past a third)"]
    for name, m in (("skew > 0 (prediction)", hi & (x["skew"] > 0)), ("skew < 0 (suppression)", hi & (x["skew"] < 0)),
                    ("other poles", ~hi)):
        row = {"n": int(m.sum()), "predict": x["pred"][m].mean().item(), "predict_past": (x["pred"][m] > PAST).float().mean().item(),
               "suppress": x["sup"][m].mean().item(), "suppress_past": (x["sup"][m] > PAST).float().mean().item()}
        out["classes"][name] = row
        lines.append(f"  {name:24s} n {row['n']:6d}  predict {row['predict']:.3f} ({row['predict_past']:.2f})  "
                     f"suppress {row['suppress']:.3f} ({row['suppress_past']:.2f})")
    lines.append("per layer, high-kurtosis neurons: P (positive-skew pole) predicts, S suppresses (past a third), both; "
                 "median act_top of P / S")
    for layer, p in zip(layers, per):
        f = len(p["kurt"]) // 2
        hi = p["kurt"][:f] > p["kurt"][:f].quantile(1 - GURNEE_TOP)  # a neuron's poles share their kurtosis
        n = torch.arange(f)
        p_pos = torch.where(p["skew"][:f] > 0, n, n + f)
        s_pos = torch.where(p["skew"][:f] > 0, n + f, n)
        pp, ss = p["pred"][p_pos][hi] > PAST, p["sup"][s_pos][hi] > PAST
        row = {"layer": layer, "n": int(hi.sum()), "p_predicts": pp.float().mean().item(),
               "s_suppresses": ss.float().mean().item(), "both": (pp & ss).float().mean().item(),
               "act_top_p": p["act_top"][p_pos][hi].median().item(), "act_top_s": p["act_top"][s_pos][hi].median().item()}
        out["per_layer"].append(row)
        lines.append(f"  L{layer}: n {row['n']:5d}  P predicts {row['p_predicts']:.2f}  S suppresses {row['s_suppresses']:.2f}  "
                     f"both {row['both']:.2f}  act_top {row['act_top_p']:.1f} / {row['act_top_s']:.1f}")
    return lines, out


def main() -> None:
    L = arch.text_config(MODEL).num_hidden_layers
    flat, start = flat_tokens()
    g, g_shared = torch.Generator().manual_seed(0), torch.Generator().manual_seed(1)
    late = range(2 * L // 3, L)
    rows, shared, per = [], [], []
    for layer in range(L):
        acts = pq.read_table(neuron_table("acts", layer), columns=["act_top", "top_doc", "top_pos", "top_val"])
        reads = pq.read_table(neuron_table("reads", layer), columns=[f"{c}_ids" for c in ("write_R", "write_S", *READS)]
                              + ["vocab_skew", "vocab_kurt"])
        ids = {c.removesuffix("_ids"): column(reads, c, K) for c in reads.column_names if c.endswith("_ids")}
        ids["drivers"] = column(pq.read_table(neuron_table("drivers", layer), columns=["drivers_ids"]), "drivers_ids", DRIVERS)
        val, doc, pos = column(acts, "top_val").float(), column(acts, "top_doc").long(), column(acts, "top_pos").long()
        f = len(val) // 2
        fired = val > 0
        cur = firing_tokens(flat, start, doc, pos)
        nxt = firing_tokens(flat, start, doc, pos, 1)
        perm = torch.cat([torch.randperm(f, generator=g), f + torch.randperm(f, generator=g)])
        row, pole = {"layer": layer}, {}

        def score(key: str, m: torch.Tensor, c: torch.Tensor) -> None:
            pole[key] = m
            row[key] = (m.mean().item(), c.mean().item(), (m > PAST).float().mean().item(), (c > PAST).float().mean().item())

        ok = fired & (nxt >= 0)
        w = ids["write_R"]
        score("write_R", hits(w, nxt, ok), hits(w[perm], nxt, ok))
        opp = w[torch.cat([torch.arange(f, 2 * f), torch.arange(f)])]  # the opposite pole's top 10: this pole's bottom 10
        score("suppress_R", hits(opp, nxt, ok), hits(opp[perm], nxt, ok))
        for name in READS:
            score(name, hits(ids[name], cur, fired), hits(ids[name][perm], cur, fired))
        for name in READS:
            score(name.replace("read", "drivers"), in_list(ids[name], ids["drivers"]) / K,
                  in_list(ids[name][perm], ids["drivers"]) / K)
        rows.append(row)
        if layer in late:
            per.append({"skew": scalar(reads, "vocab_skew"), "kurt": scalar(reads, "vocab_kurt"),
                        "act_top": scalar(acts, "act_top"), "pred": pole["write_R"], "sup": pole["suppress_R"]})
        sp = torch.cat([torch.randperm(f, generator=g_shared), f + torch.randperm(f, generator=g_shared)])
        shared.append({f"{a}~{b}": (in_list(ids[a], ids[b]).mean().item(), in_list(ids[a][sp], ids[b]).mean().item())
                       for a, b in SHARED})
        print(f"L{layer:02d} " + "  ".join(f"{LABELS.get(k, k)} {v[0]:.3f}/{v[1]:.3f} ({v[2]:.2f}/{v[3]:.2f})"
                                          for k, v in row.items() if k != "layer"), flush=True)

    keys = [k for k in rows[0] if k != "layer"]
    lines = [f"side agreement, {MODEL}: per layer, mean share of a pole's top firings (matched / control: a random "
             f"pole's lists), and in parentheses the share of poles past {PAST:.2f}",
             "predict: the next token is in write_R's top 10; suppress: in the opposite pole's (this pole's bottom 10); "
             "read_R, read_S, read_S_z: the peak token is in the read's top 10; *→drv: share of the read's top 10 among "
             f"the pole's {DRIVERS} exact drivers", ""]
    for r in rows:
        lines.append(f"L{r['layer']:2d}  " + "  ".join(f"{LABELS.get(k, k)} {r[k][0]:.3f}/{r[k][1]:.3f} "
                                                       f"({r[k][2]:.2f}/{r[k][3]:.2f})" for k in keys))
    g_lines, g_table = gurnee(per, late)
    lines += ["", *g_lines, "", f"shared tokens: tokens two lists share (top {K}; drivers: top {DRIVERS}), matched / control: "
              + ", ".join(f"{a}~{b}" for a, b in SHARED)]
    for layer, s in enumerate(shared):
        lines.append(f"  L{layer:2d}: " + "  ".join(f"{k} {v[0]:.3f}/{v[1]:.3f}" for k, v in s.items()))
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"side_agreement_{STEM}.log").write_text(text + "\n")
    table = {k: {"layer": [r["layer"] for r in rows],
                 **{name: [r[k][i] for r in rows] for i, name in enumerate(("mean", "control_mean", "past", "control_past"))}}
             for k in keys}
    cross = {k: {"layer": list(range(L)), "mean": [s[k][0] for s in shared], "control_mean": [s[k][1] for s in shared]}
             for k in shared[0]}
    (SUMMARY / f"side_agreement_{STEM}.json").write_text(
        json.dumps({"model": MODEL, "past": PAST, "reads": table, "gurnee": g_table, "shared": cross}))


if __name__ == "__main__":
    main()
