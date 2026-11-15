"""Which weight read names a pole's theme at each depth, read by eye: the evidence, beside the census of
``checks/reads_at_firings.py``, for the labeller's reading plans (``10_labels.py``'s ``PLANS``: which input read and which
write read a card shows at each depth, and what its prompt says about them).

``deck``: for ``N_LAYERS`` layers spread over the model's depth, ``PER_LAYER`` random poles each that fire (``act_top`` at
least ``SILENT``) and aren't a faint shadow of their partner. Per pole, its top contexts and the eight weight reads as a
label card shows them (the first 24 tokens, raw reads without the layer's hub tokens, variants merged): the four input
reads (``read_S``, ``read_S_z``, ``read_R``, ``read_R_z``) shuffled and named I1-I4, the four write reads (``write_R``,
``write_R_z``, ``write_S``, ``write_S_z``) shuffled and named O1-O4. Poles are shuffled too and named P001..., so a rater
sees neither the model, the layer nor the lens. ``data/checks/depth_reads/<model>/<round>/deck.md`` is the rater's file;
the answers, ``key.json``, are kept apart in ``data/summary/depth_reads/<model>/<round>/``, where the ratings go too. ``--layers`` picks the layers instead (a later round, at the depths where the
first one left the choice open), ``--round`` names the round (``r1`` by default).

Raters (fresh agents, each its own deck pass, the instructions in ``RATER``; for the plans of record a Claude Opus
subagent and Codex, two rounds a model) write one JSON line per pole to the round's ``ratings_<rater>.jsonl``: each list's coherence (2 a clear, specific theme; 1 a loose or partial one; 0 none) and, for
input lists, whether that theme agrees with where the contexts show the pole firing (1 or 0).

``score``: per sampled layer and read, over every round, the mean coherence and agreement over raters and poles, the raters'
agreement with each other, and the census beside them, into ``data/summary/depth_reads_<model>.{json,log}``. These are readings of a
sample (8 poles a layer), a qualitative check on the census, not a measurement of every pole.

    NEURON_LENS_MODEL=... python scripts/checks/depth_reads.py deck [--layers 3,6,8 --round r2]
    NEURON_LENS_MODEL=... python scripts/checks/depth_reads.py score
"""
import argparse
import collections
import importlib.util
import json
import random
import statistics
from pathlib import Path

from neuron_lens import arch
from neuron_lens.catalog import MODEL, N_CONTEXTS, load_layer, tokenizer
from neuron_lens.paths import DATA, SUMMARY

spec = importlib.util.spec_from_file_location("labels", Path(__file__).resolve().parents[1] / "10_labels.py")
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)

STEM = arch.stem(MODEL)
L = arch.text_config(MODEL).num_hidden_layers
N_LAYERS, PER_LAYER, SHOW = 8, 8, 24
INS = ("read_S", "read_S_z", "read_R", "read_R_z")
OUTS = ("write_R", "write_R_z", "write_S", "write_S_z")
DECKS = DATA / "checks" / "depth_reads" / STEM  # what raters read
OUT = SUMMARY / "depth_reads" / STEM  # the keys and the ratings
RATER = """You are rating word lists read from the weights of neurons in a language model, one neuron "pole" at a time.
Each pole shows its top contexts (the token where it fired most is in ⟦ ⟧, with the text around it), then
four input-side lists (I1-I4: what drives the pole) and four output-side lists (O1-O4: what the pole's output
pushes the model toward). Each list is up to 24 tokens, ranked, in tiers (1-8 ‖ 9-24); ×n merges variants.
The lists come from different methods, shuffled per pole; nothing tells you which is which, and none is the
right answer by default.

For every list, rate coherence:
- 2: a clear, specific theme (one word family, concept, topic, grammatical form, script or format), visible
  across the list, not just one token;
- 1: a loose or partial theme (a few related tokens among noise, or a broad theme);
- 0: no theme (noise, fragments, unrelated tokens).
English and Chinese words for one concept count as one theme.

For input lists (I1-I4) only, also rate agreement with the contexts: 1 if the list's theme matches where the
pole fires: the marked token, the word it belongs to, the words just after it, or what the context is about;
0 otherwise (or if the list has no theme).

Judge each list on its own. Rate every pole. Output exactly one JSON object per pole, one per line, nothing else:
{"pole": "P001", "I1": [coherence, agreement], "I2": [c, a], "I3": [c, a], "I4": [c, a], "O1": c, "O2": c, "O3": c, "O4": c}
"""


def deck(args) -> None:
    rng = random.Random(args.seed)
    tok = tokenizer()
    cache = {}

    def dec(i: int) -> str:
        if i not in cache:
            t = tok.decode([i], skip_special_tokens=True).replace("\n", "⏎").replace("\r", "␍")
            cache[i] = t if t.strip() else repr(t)
        return cache[i]

    layers = (sorted(int(x) for x in args.layers.split(",")) if args.layers
              else sorted({round(i * (L - 1) / (N_LAYERS - 1)) for i in range(N_LAYERS)}))
    poles = []
    for layer in layers:
        t = load_layer(layer)
        f = t.num_rows // 2
        hub = {}
        for b in INS + OUTS:
            c = collections.Counter()
            for ids in t.column(b + "_ids").to_pylist():
                c.update(set(ids))
            hub[b] = set() if b.endswith("_z") else {k for k, v in c.items() if v / t.num_rows >= lb.HUB}
        rows = t.select(["act_top", "neg_lobe", "contexts"] + [b + "_ids" for b in INS + OUTS]).to_pylist()
        ok = []
        for i, r in enumerate(rows):
            other = rows[i + f if i < f else i - f]
            shadow = r["neg_lobe"] > 0.5 and other["act_top"] >= lb.SILENT and r["act_top"] < lb.SHADOW * other["act_top"]
            if r["act_top"] >= lb.SILENT and not shadow and r["contexts"]:
                ok.append(i)
        for i in rng.sample(ok, min(PER_LAYER, len(ok))):
            r = rows[i]
            lists = {b: lb.grouped([dec(x) for x in r[b + "_ids"] if x not in hub[b]][:SHOW])[0] for b in INS + OUTS}
            poles.append({"layer": layer, "neuron": i % f, "pole": 1 if i < f else -1,
                          "contexts": str(r["contexts"]).split(" ‖ ")[:N_CONTEXTS], "lists": lists})
    rng.shuffle(poles)
    out, deck_dir = OUT / args.round, DECKS / args.round
    out.mkdir(parents=True, exist_ok=True)
    deck_dir.mkdir(parents=True, exist_ok=True)
    lines, key = ["# Rating deck", "", RATER], {}
    for k, p in enumerate(poles, 1):
        pid = f"P{k:03d}"
        ins, outs = list(INS), list(OUTS)
        rng.shuffle(ins)
        rng.shuffle(outs)
        key[pid] = {"layer": p["layer"], "neuron": p["neuron"], "pole": p["pole"],
                    **{f"I{j + 1}": b for j, b in enumerate(ins)}, **{f"O{j + 1}": b for j, b in enumerate(outs)}}
        lines += ["", f"## {pid}", *[f"  - context: {c}" for c in p["contexts"] if c]]
        lines += [f"I{j + 1}: {p['lists'][b] or '(empty)'}" for j, b in enumerate(ins)]
        lines += [f"O{j + 1}: {p['lists'][b] or '(empty)'}" for j, b in enumerate(outs)]
    (deck_dir / "deck.md").write_text("\n".join(lines) + "\n")
    (out / "key.json").write_text(json.dumps({"model": MODEL, "layers": layers, "poles": key}, indent=1))
    print(f"{len(poles)} poles from layers {layers} -> {deck_dir / 'deck.md'}; key in {out}")


def score(args) -> None:
    key, raters = {"layers": set(), "poles": {}}, collections.defaultdict(list)  # every round's poles, as round/pole
    for rd in sorted(p.parent for p in OUT.glob("*/key.json")):
        k = json.loads((rd / "key.json").read_text())
        key["layers"] |= set(k["layers"])
        key["poles"] |= {f"{rd.name}/{pid}": v for pid, v in k["poles"].items()}
        for p in sorted(rd.glob("ratings_*.jsonl")):
            raters[p.stem.removeprefix("ratings_")] += [r | {"pole": f"{rd.name}/{r.get('pole')}"}
                                                        for r in map(json.loads, filter(str.strip, p.read_text().splitlines()))]
    key["layers"] = sorted(key["layers"])
    census = {r["layer"]: r for r in json.loads((SUMMARY / f"reads_at_firings_{STEM}.json").read_text())["neurons"]["layers"]}
    coh, agr = collections.defaultdict(list), collections.defaultdict(list)  # (layer, read) -> values
    per_rater = collections.defaultdict(dict)  # rater -> (pole, slot) -> coherence
    for name, rows in raters.items():
        for r in rows:
            k = key["poles"].get(r.get("pole"))
            if not k:
                continue
            for slot in [f"I{j}" for j in range(1, 5)] + [f"O{j}" for j in range(1, 5)]:
                v = r.get(slot)
                if v is None:
                    continue
                c, a = (v[0], v[1]) if isinstance(v, list) else (v, None)
                coh[k["layer"], k[slot]].append(c)
                per_rater[name][r["pole"], slot] = c
                if a is not None:
                    agr[k["layer"], k[slot]].append(a)
    names = sorted(per_rater)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = sorted(set(per_rater[a]) & set(per_rater[b]))
            if len(both) > 2:
                x, y = [per_rater[a][s] for s in both], [per_rater[b][s] for s in both]
                pairs.append({"raters": [a, b], "lists": len(both), "pearson": statistics.correlation(x, y),
                              "exact": sum(u == v for u, v in zip(x, y)) / len(both)})
    out = {"model": MODEL, "raters": names, "agreement": pairs, "layers": []}
    log = [f"depth_reads, {MODEL}: raters {', '.join(names)}; per layer, each read's mean coherence (0-2) and, for input reads, "
           "agreement with the contexts (0-1), over raters and poles; the census (reads_at_firings net shares of firing poles) beside", ""]
    for p in pairs:
        log.append(f"raters {p['raters'][0]} / {p['raters'][1]}: {p['lists']} lists, Pearson {p['pearson']:.2f}, same score {p['exact']:.0%}")
    log.append("")
    log.append("layer  " + "  ".join(f"{b:>15}" for b in INS) + " | " + "  ".join(f"{b:>10}" for b in OUTS)
               + " | census in·S in·R out·R out·S")
    for layer in key["layers"]:
        row = {"layer": layer}
        cells_in = []
        for b in INS:
            c, a = coh[layer, b], agr[layer, b]
            row[b] = {"coherence": statistics.mean(c) if c else None, "agreement": statistics.mean(a) if a else None, "n": len(c)}
            cells_in.append(f"{row[b]['coherence']:.2f}/{row[b]['agreement']:.2f}" if c else "-")
        cells_out = []
        for b in OUTS:
            c = coh[layer, b]
            row[b] = {"coherence": statistics.mean(c) if c else None, "n": len(c)}
            cells_out.append(f"{row[b]['coherence']:.2f}" if c else "-")
        cr = census.get(layer, {})
        fire = max(1e-9, 1 - cr.get("silent", 0))
        row["census"] = {k: cr.get(f"{k}_net", 0) / fire for k in ("read_S", "read_R", "write_R", "write_S")}
        out["layers"].append(row)
        log.append(f"L{layer:<5}" + "  ".join(f"{x:>15}" for x in cells_in) + " | " + "  ".join(f"{x:>10}" for x in cells_out)
                   + " | " + " ".join(f"{row['census'][k]:.2f}" for k in ("read_S", "read_R", "write_R", "write_S")))
    (SUMMARY / f"depth_reads_{STEM}.json").write_text(json.dumps(out, indent=1))
    (SUMMARY / f"depth_reads_{STEM}.log").write_text("\n".join(log) + "\n")
    print("\n".join(log))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("deck", "score"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default=None, help="deck: these layers (comma-separated), not N_LAYERS spread over the depth")
    ap.add_argument("--round", default="r1", help="deck: the round's name, its folder under the model's")
    args = ap.parse_args()
    {"deck": deck, "score": score}[args.what](args)


if __name__ == "__main__":
    main()
