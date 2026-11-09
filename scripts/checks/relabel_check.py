"""Did relabelling the negative-lobe poles help (findings §8)?

The first labels of these poles (``acts``' ``neg_lobe`` past a half) were written from input reads pointing the wrong way;
``10_labels.py cards --flipped`` labels those neurons again from the corrected reads. Blind, on ``PER_MODEL`` random such
poles with at least ``HELD_OUT`` firings, as ``10_labels.py check`` tests the head labels: a scorer (Claude Sonnet) sees a
pole's ``fires`` label and 8 contexts, 4 of its held-out firings (ranks 6-24, off every card) and 4 of another
flipped pole of its layer, shuffled, ending at the marked token, and gives each the probability that it is the described
pole's. AUROC per pole for three labels on the same contexts: the new one, the first one (the release on the Hub's
``main``), and the new one of a third flipped pole of the layer as a control.

    python scripts/checks/relabel_check.py    # after 10_labels.py shards; into data/summary/relabel_check_<model>.json
"""
import importlib.util
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from neuron_lens.catalog import HELD_OUT, MODEL, N_CONTEXTS, context, documents, neuron_table, tokenizer
from neuron_lens.paths import SUMMARY

spec = importlib.util.spec_from_file_location("labels", Path(__file__).resolve().parent.parent / "10_labels.py")
labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels)

PER_MODEL = 150
SHOWN = 4
RELEASE = "stdlogits/neuron-lens"
SYSTEM = """You test descriptions of neurons in a language model against text. Each context shows a position where some \
neuron's activation was large: the token in ⟦ ⟧, where the context ends. Given one neuron's description, say for each context \
how likely it is that this neuron produced it: some of the contexts are this neuron's, the others another neuron's. \
Reply with one JSON line and nothing else: {"p": [p1, ..., pn]}, a probability from 0 to 1 per context, in order."""


def flipped_poles() -> dict[int, list[dict]]:
    """Per layer, the flipped poles that fire with enough held-out firings to show."""
    out = {}
    for l in range(labels.LAYERS):
        t = pq.read_table(neuron_table("acts", l), columns=["layer", "neuron", "pole", "act_top", "neg_lobe", "top_doc", "top_pos", "top_val"])
        lobe, top = t.column("neg_lobe").to_pylist(), t.column("act_top").to_pylist()
        rows = [t.slice(i, 1).to_pylist()[0] for i, (a, b) in enumerate(zip(lobe, top)) if a > 0.5 and b >= labels.SILENT]
        rows = [r for r in rows if r["top_val"][HELD_OUT - 1] > 0]
        if len(rows) >= 3:
            out[l] = rows
    return out


def label_table(layer: int, path) -> dict:
    return {(r["neuron"], r["pole"]): r for r in pq.read_table(path).to_pylist()}


def main() -> None:
    stem = labels.NAME
    by_layer = flipped_poles()
    pool = [(l, r) for l, rows in by_layer.items() for r in rows]
    rng = random.Random(1109)
    picks = rng.sample(pool, min(PER_MODEL, len(pool)))
    new, old = {}, {}
    for l in sorted({l for l, _ in picks}):
        new[l] = label_table(l, neuron_table("labels", l))
        old[l] = label_table(l, hf_hub_download(RELEASE, f"{stem}/neurons/labels/L{l:02d}.parquet", repo_type="dataset", revision="main"))
    tok, docs = tokenizer(), documents()
    reqs, meta = [], {}
    for l, r in picks:
        other, third = rng.sample([x for x in by_layer[l] if x is not r], 2)
        ctx = []
        for src, truth in ((r, 1), (other, 0)):
            for j in rng.sample(range(N_CONTEXTS, HELD_OUT), SHOWN):
                ctx.append((context(tok, docs[src["top_doc"][j]], src["top_pos"][j], after=0), truth))
        rng.shuffle(ctx)
        pid = f"L{l}_n{r['neuron']}_{'p' if r['pole'] > 0 else 'm'}"
        text = "\n".join(f"{i + 1}. {c}" for i, (c, _) in enumerate(ctx))
        key = (r["neuron"], r["pole"])
        arms = {"new": new[l][key], "old": old[l][key], "control": new[l][third["neuron"], third["pole"]]}
        for arm, desc in arms.items():
            reqs.append({"custom_id": f"{arm}_{pid}", "params": {
                "model": labels.SCORER, "max_tokens": 2000, "system": SYSTEM,
                "messages": [{"role": "user", "content": f"The neuron fires: {desc['fires']}\n\nContexts:\n{text}"}]}})
        meta[pid] = {"layer": l, "truth": [t for _, t in ctx], "conf": {"new": arms["new"]["fires_conf"], "old": arms["old"]["fires_conf"]}}
    c = labels.client()
    batch = c.messages.batches.create(requests=reqs)
    print(f"check batch {batch.id}: {len(reqs)} requests", flush=True)
    labels.wait(c, batch.id)
    got = {}
    for res in c.messages.batches.results(batch.id):
        if res.result.type == "succeeded":
            m = re.search(r"\{.*\}", labels.text_of(res.result.message), re.DOTALL)
            try:
                got[res.custom_id] = json.loads(m[0]) if m else None
            except json.JSONDecodeError:
                got[res.custom_id] = None
    per = {}
    for pid, mt in meta.items():
        out = {"layer": mt["layer"], "conf": mt["conf"]}
        for arm in ("new", "old", "control"):
            p = (got.get(f"{arm}_{pid}") or {}).get("p")
            if isinstance(p, list) and len(p) == len(mt["truth"]):
                out[arm] = labels.auroc([float(x) for x in p], mt["truth"])
        per[pid] = out
    both = [v for v in per.values() if "new" in v and "old" in v]
    stat = lambda xs: (sum(xs) / len(xs), *labels.boot(xs))
    summary = {arm: stat([v[arm] for v in per.values() if arm in v]) for arm in ("new", "old", "control")}
    summary["new_minus_old"] = stat([v["new"] - v["old"] for v in both])
    summary["fires_conf"] = {arm: sum(v["conf"][arm] for v in both) / len(both) for arm in ("new", "old")}
    (SUMMARY / f"relabel_check_{stem}.json").write_text(json.dumps({"model": MODEL, "batch": batch.id, "scorer": labels.SCORER, "summary": summary, "poles": per}, indent=1))
    print(stem, len(both), "poles:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
