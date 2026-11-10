"""Detection scoring of the description pilot (findings §7, §8): do a pole's labels pick out where it fires?

A scorer who sees only a label gets up to ten contexts, each ending at one token marked ``⟦ ⟧``: only
what the model has seen when the pole fires there, so a label cannot be checked against what follows.
``N_POS`` (5) are the pole's held-out top firings, drawn from its top documents past the first
``N_CONTEXTS`` (which its card showed); ``N_NEG`` (5) are quiet positions, drawn from
``lens_blind.N_RANDOM_DOCS`` (160) random documents of 64+ tokens, from position ``lens_blind.MIN_POS``
(8) on, where the pole's signed activation stays under a tenth of its typical high activation
(``act_top``). The scorer gives each marked token a probability that the pole fires there; the score
is the AUROC (ties count half). Every set is also scored under a control label, another pole's (one
derangement over the scored poles), in tasks the scorer cannot tell apart from matched ones, so the
control AUROC says how much any plausible label earns on these negatives (detection scoring in the
style of Paulo et al. 2024). Poles with ``act_top`` under ``lens_blind.SILENT`` (0.05) have no set.

Stages, under ``data/checks/describe/<model>/`` (``describe_cards.py`` writes the cards and ``poles.json``):

- ``make`` (the model, bf16, eager attention; MPS or CUDA): the positives from the ``acts`` tables, the
  documents from the activations file or else rebuilt from the corpus (``catalog.documents``, as
  released); one pass over the positives' documents and the random ones
  (``lens_blind.pole_acts``) gives every sampled pole's activation at every position, for the quiet
  negatives (``lens_blind.quiet``) and a recheck of the positives against the catalog. Sets to
  ``sets.json``. ``random.Random(0)``. 189 poles, 1,094 documents: about 9 minutes on a shared MPS (0.46 s a
  document measured).
- explainers (``describe_cards.py``) write ``cards/labels_XX.jsonl`` (``describe_prompt.md``'s format);
- ``prompt``: the labels and the sets to scorer tasks ``tasks/score_XX.md`` (``TASKS_PER_FILE`` a file,
  matched and control shuffled together) and the key ``tasks/key.json``. A pole's label is its labels
  joined by ``; `` (weights dropped). The explainers' files, when there are any, become the record,
  ``data/summary/describe_labels_<model>.jsonl`` (in pole order); without them the record is scored
  as it stands, so its labels can be rescored on fresh sets.
- scorers (Claude Code subagents, Sonnet, briefed with ``describe_score_prompt.md``) write
  ``tasks/scored_XX.jsonl``;
- ``grade``: the scorers' answers against the key, to ``data/summary/describe_score_<model>.{log,json}``:
  matched and control AUROC, all poles, by thirds of depth and by the explainer's confidence; means
  with 95% bootstrap intervals over neurons (both poles together, 2,000 resamples), and per pole in
  the JSON.

    python scripts/checks/describe_score.py make [--device mps] [--batch 16]
    python scripts/checks/describe_score.py prompt
    python scripts/checks/describe_score.py grade
"""
import argparse
import json
import random
from collections import defaultdict
from statistics import mean, median

import lens_blind as lb
import pyarrow.parquet as pq
import torch
from describe_cards import NAME, OUT

from neuron_lens import arch
from neuron_lens.catalog import (
    MODEL,
    HELD_OUT,
    N_CONTEXTS,
    context,
    documents,
    neuron_table,
    tokenizer,
)
from neuron_lens.paths import SUMMARY

N_POS, N_NEG = 5, 5
TASKS_PER_FILE = 20
LABELS = SUMMARY / f"describe_labels_{NAME}.jsonl"
THIRDS = ((0, 11), (12, 23), (24, 35))


def parse(pid: str) -> tuple[int, int, int]:
    """``"L8 n344 +"`` (``lens_blind.pole_id``) as ``(8, 344, 1)``."""
    l, n, s = pid.split()
    return int(l[1:]), int(n[1:]), 1 if s == "+" else -1


def make(args) -> None:
    rng = random.Random(0)
    poles = [tuple(p) for p in json.loads((OUT / "poles.json").read_text())]
    by_layer = defaultdict(list)
    for p in poles:
        by_layer[p[0]].append(p)
    scored, silent = [], []  # scored: {layer, neuron, sign, id, act_top, pos: [(doc, position, catalog value)]}
    for l, ps in sorted(by_layer.items()):
        t = pq.read_table(neuron_table("acts", l), columns=["act_top", "top_doc", "top_pos", "top_val"])
        for _, n, s in ps:
            e = t.slice(n if s > 0 else t.num_rows // 2 + n, 1).to_pylist()[0]
            if e["act_top"] < lb.SILENT:
                silent.append(lb.pole_id(l, n, s))
                continue
            held = [j for j, v in enumerate(e["top_val"]) if v > 0 and j < HELD_OUT][N_CONTEXTS:]  # the card showed the first
            scored.append({"layer": l, "neuron": n, "sign": s, "id": lb.pole_id(l, n, s), "act_top": e["act_top"],
                           "pos": [(int(e["top_doc"][j]), int(e["top_pos"][j]), float(e["top_val"][j]))
                                   for j in rng.sample(held, min(N_POS, len(held)))]})
    tokens = documents()
    random_docs = sorted(rng.sample([d for d in range(len(tokens)) if len(tokens[d]) >= 64], lb.N_RANDOM_DOCS))
    docs = {d: [int(x) for x in tokens[d]] for d in sorted({d for p in scored for d, _, _ in p["pos"]} | set(random_docs))}
    tok = tokenizer()
    lm, _ = arch.load_model(MODEL, args.device)
    acts = lb.pole_acts(lm, scored, docs, tok.pad_token_id, args.batch)
    sets, ratios = {}, []
    for i, p in enumerate(scored):
        pos = [(d, t, v, float(acts[d][t, i])) for d, t, v in p["pos"]]
        neg = lb.quiet(acts, i, random_docs, p["act_top"] / 10, N_NEG, rng)
        items = [(context(tok, docs[d], t, after=0), 1, v) for d, t, v, _ in pos]
        items += [(context(tok, docs[d], t, after=0), 0, float(acts[d][t, i])) for d, t in neg]
        rng.shuffle(items)
        sets[p["id"]] = {"items": items, "recheck": [(v, r) for _, _, v, r in pos]}
        ratios += [r / v for _, _, v, r in pos]
    (OUT / "sets.json").write_text(json.dumps(sets, ensure_ascii=False))
    short = [pid for pid, s in sets.items() if sum(x[1] for x in s["items"]) < N_POS or
             sum(1 - x[1] for x in s["items"]) < N_NEG]
    print(f"{len(sets)} sets ({len(silent)} silent poles skipped: {', '.join(silent)}); positives rechecked: "
          f"{len(ratios)}, recomputed/catalog median {median(ratios):.3f}, under 0.5 {sum(r < 0.5 for r in ratios)}; "
          f"short sets {short or 'none'} -> {OUT / 'sets.json'}")


def prompt(args) -> None:
    sets = json.loads((OUT / "sets.json").read_text())
    order = [lb.pole_id(*p) for p in json.loads((OUT / "poles.json").read_text())]
    fresh = sorted((OUT / "cards").glob("labels_*.jsonl"))
    if fresh:
        got = lb.read_jsonl(fresh)
        LABELS.write_text("".join(json.dumps(got[pid], ensure_ascii=False) + "\n" for pid in order if pid in got))
        print(f"{len(got)} explainers' labels -> {LABELS}")
    labels = {pid: {"label": "; ".join(x["label"] for x in o["labels"]), "confidence": o["confidence"]}
              for pid, o in lb.read_jsonl([LABELS]).items()}
    ids = sorted(set(sets) & set(labels))
    rng = random.Random(0)
    cycle = ids[:]
    rng.shuffle(cycle)
    control = {a: cycle[(i + 1) % len(cycle)] for i, a in enumerate(cycle)}
    tasks = [{"pole": pid, "kind": kind, "label": labels[src]["label"], "confidence": labels[pid]["confidence"],
              "items": sets[pid]["items"]} for pid in ids for kind, src in (("matched", pid), ("control", control[pid]))]
    rng.shuffle(tasks)
    (OUT / "tasks").mkdir(exist_ok=True)
    for old in (OUT / "tasks").glob("score_*.md"):
        old.unlink()
    key = {}
    for b in range(0, len(tasks), TASKS_PER_FILE):
        blocks = []
        for i, t in enumerate(tasks[b:b + TASKS_PER_FILE], start=b):
            tid = f"t{i:04d}"
            key[tid] = {k: t[k] for k in ("pole", "kind", "confidence")} | {"truth": [x[1] for x in t["items"]]}
            blocks.append(f"### {tid}\nlabel: {t['label']}\n" + "\n".join(f"{j + 1}. {x[0]}" for j, x in enumerate(t["items"])))
        (OUT / "tasks" / f"score_{b // TASKS_PER_FILE:02d}.md").write_text("\n\n".join(blocks) + "\n")
    (OUT / "tasks" / "key.json").write_text(json.dumps(key, ensure_ascii=False))
    print(f"{len(tasks)} tasks ({len(ids)} poles, matched and control) in {(len(tasks) + TASKS_PER_FILE - 1) // TASKS_PER_FILE} "
          f"files; labels for {len(labels)}, sets for {len(sets)} -> {OUT / 'tasks'}")


def grade(args) -> None:
    key = json.loads((OUT / "tasks" / "key.json").read_text())
    got = lb.read_jsonl((OUT / "tasks").glob("scored_*.jsonl"))
    res = {}  # pole -> {confidence, matched, control}
    for tid, k in key.items():
        if tid in got and len(got[tid]["p"]) == len(k["truth"]):
            res.setdefault(k["pole"], {"confidence": k["confidence"]})[k["kind"]] = lb.auroc(got[tid]["p"], k["truth"])
    both = {p: r for p, r in res.items() if "matched" in r and "control" in r}
    for p in both:
        both[p] |= {"layer": parse(p)[0]}
    lines = [(f"detection scoring of the description pilot, {MODEL}: {len(both)} poles scored under both labels "
              f"({len(got)} of {len(key)} tasks answered)"),
             ("AUROC of the scorer's probabilities, matched / control label (one derangement); means with 95% bootstrap "
              "intervals over neurons (they leave out document, explainer, scorer and derangement variation)")]
    rng = random.Random(0)
    rows = {}

    def ci(ps: list[str], kind: str) -> tuple[float, float]:
        """95% bootstrap interval of the mean, resampling neurons (both poles together)."""
        groups = defaultdict(list)
        for p in ps:
            groups[p.rsplit(" ", 1)[0]].append(both[p][kind])
        keys = list(groups)
        boot = sorted(mean(x for k in rng.choices(keys, k=len(keys)) for x in groups[k]) for _ in range(2000))
        return boot[50], boot[1949]

    def row(name: str, ps: list[str]) -> None:
        if not ps:
            return
        m, c = [both[p]["matched"] for p in ps], [both[p]["control"] for p in ps]
        r = {"n": len(ps), "matched": mean(m), "matched_ci": ci(ps, "matched"), "control": mean(c),
             "control_ci": ci(ps, "control"), "matched_median": median(m), "control_median": median(c),
             "matched_gt_control": mean(a > b for a, b in zip(m, c)), "matched_ge_0.9": mean(a >= 0.9 for a in m)}
        rows[name] = r
        lines.append(f"  {name:28s} n {r['n']:3d}  mean {r['matched']:.2f} [{r['matched_ci'][0]:.2f}, "
                     f"{r['matched_ci'][1]:.2f}] / {r['control']:.2f} [{r['control_ci'][0]:.2f}, {r['control_ci'][1]:.2f}]  "
                     f"median {r['matched_median']:.2f} / {r['control_median']:.2f}  matched > control "
                     f"{r['matched_gt_control']:.2f}")
    row("all", list(both))
    for lo, hi in THIRDS:
        row(f"layers {lo}-{hi}", [p for p in both if lo <= both[p]["layer"] <= hi])
    for lo, hi in ((0, 0.4), (0.4, 0.7), (0.7, 1.01)):
        row(f"explainer confidence {lo}-{min(hi, 1):.1f}", [p for p in both if lo <= both[p]["confidence"] < hi])
    lines.append("\nlowest matched AUROC")
    for p in sorted(both, key=lambda p: both[p]["matched"])[:10]:
        lines.append(f"  {p:14s} {both[p]['matched']:.2f} (control {both[p]['control']:.2f})")
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"describe_score_{NAME}.log").write_text(text + "\n")
    (SUMMARY / f"describe_score_{NAME}.json").write_text(json.dumps({"rows": rows, "poles": both}, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("make", "prompt", "grade"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    ap.add_argument("--batch", type=int, default=16, help="make: documents a forward pass")
    args = ap.parse_args()
    if MODEL != arch.QWEN:
        raise SystemExit(f"the description pilot is Qwen3-8B's (its named poles and causal sample); not {MODEL}")
    {"make": make, "prompt": prompt, "grade": grade}[args.what](args)


if __name__ == "__main__":
    main()
