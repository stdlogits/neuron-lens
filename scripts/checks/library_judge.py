"""The Neuron Persona Library's personas against a pole's reads, judged generously and blind, from
each side of the pole (findings §10).

``library_crosscheck.py`` counts a persona as read when its name's own first tokens rank high in
the pole's read: "horse" misses 马, caballo, saddle and cavalry. Here a judge reads the tokens
generously. For a sample of persona rows per layer group (``GROUPS``), the pole's reads from the
catalog, in three conditions (``SIDES``):

- **write:** what the pole writes, under the lens of record: the top 10 tokens and the top 8
  layer-contrastive ones, as ``library_crosscheck.py`` scores them;
- **input:** what drives it: the R-lens read of its read direction from the layer's MLP input
  (top 10), and the S-lens read of the input tokens behind it, raw (top 10) through L8 and
  layer-contrastive (top 8) after, as the catalog's bands advise (findings §6);
- **both:** the four lists together.

Each read is shown twice, once with its own persona's name and once with another sampled row's
name from the same group (the control), all shuffled together, so the judge cannot tell which
pairing is real. Scores:

- 3: a token is the concept itself, in any language or form (horse: 马, caballo, Horses);
- 2: a token is a strongly associated thing, part, attribute or act (saddle, cavalry, mane);
- 1: a loose or thematic link (farm, animal, ride);
- 0: none.

``make`` writes each side's batches to ``data/judge/library/<side>/`` and its key outside them;
Claude instances write ``answers_XX.jsonl`` beside the batches; ``grade`` reports matched against
control by group for each side, for all names and for specific ones (generic speaker frames such
as "Human Person" or "Student" left out), with the strict hit (a name word's first token among
the write side's tokens) on the same rows, and how the sides combine on the matched rows.

    python scripts/checks/library_judge.py make [--per-group 60] [--sides write,input,both]
    python scripts/checks/library_judge.py grade
"""
import argparse
import json
import random
import re

import pyarrow.parquet as pq
from library_crosscheck import GROUPS, LIBRARY, MODEL, name_tokens
from transformers import AutoTokenizer

from neuron_lens.catalog import neuron_table
from neuron_lens.paths import DATA, SUMMARY

JUDGE = DATA / "judge"
SIDES = ("write", "input", "both")
BATCH = 120
S_RAW_THROUGH = 8  # the S-lens read is legible raw through L8, layer-contrastive after (findings §6)
GENERIC = re.compile(r"\b(human|person|people|individual|student|teacher|man|woman|guy|someone|user|assistant)\b", re.I)
LISTS = {
    "top": "the tokens the neuron's output most promotes",
    "distinct": "the tokens its output promotes most unusually compared with its neighbours",
    "in_lens": "what the model is disposed to say when this neuron is driven (a read of the neuron's input direction)",
    "in_tokens": "input tokens whose presence drives the neuron",
}
SHOWN = {"write": ("top", "distinct"), "input": ("in_lens", "in_tokens"), "both": ("top", "distinct", "in_lens", "in_tokens")}
PROMPT = """You will rate {n} items. Each item pairs a name with lists of tokens from a language
model's vocabulary that describe one neuron:
{lists}
Tokens can be in any language, and can be word fragments (" Cav" may begin "cavalry").

Read the tokens generously and rate how clearly any of them points to the name's concept:
- 3: a token is the concept itself, in any language, spelling or form (for "Horse": 马, caballo,
  Horses, equine);
- 2: a token is a strongly associated thing, part, attribute or act (saddle, cavalry, mane,
  gallop);
- 1: a loose or thematic link (farm, animal, ride);
- 0: nothing related.

Some names are paired with random lists. Judge each item on its own.

Read {batch}. Write one JSON object per line to {answers}, in the items' order:
{{"item": "<id>", "score": <0-3>, "tokens": ["<the tokens that justify the score, if any>"]}}
Then reply with only the number of lines written.
"""


def paths(side: str):
    return JUDGE / "library" / side, JUDGE / f"library_key_{side}.json"  # the key outside the judges' directory


def persona_rows() -> list[dict]:
    lib = json.loads(LIBRARY.read_text())
    persona = lib["categories"].index("persona")
    return [r for r in (dict(zip(lib["fields"], r)) for r in lib["rows"]) if r["cat"] == persona]


def rows_sampled(per_group: int) -> list[dict]:
    rows = persona_rows()
    rng = random.Random(0)
    out = []
    for lo, hi in GROUPS:
        mine = [r for r in rows if lo <= r["L"] <= hi]
        out += rng.sample(mine, min(per_group, len(mine)))
    return out


def show(ids: list[int], tok) -> list[str]:
    return [tok.decode([i]).replace("\n", "⏎") for i in ids]


def make(args) -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = rows_sampled(args.per_group)
    cols = ["write_R_ids", "write_R_z_ids", "read_R_ids", "read_S_ids", "read_S_z_ids"]
    reads = {}
    for layer in sorted({r["L"] for r in rows}):
        t = pq.read_table(neuron_table("reads", layer), columns=cols)
        reads[layer] = {c: t.column(c).to_pylist() for c in cols}
    f = len(reads[rows[0]["L"]][cols[0]]) // 2

    def lists(r: dict) -> dict:
        d, row = reads[r["L"]], r["n"] if r["s"] > 0 else f + r["n"]
        s = d["read_S_ids"][row][:10] if r["L"] <= S_RAW_THROUGH else d["read_S_z_ids"][row][:8]
        return {"top": d["write_R_ids"][row][:10], "distinct": d["write_R_z_ids"][row][:8],
                "in_lens": d["read_R_ids"][row][:10], "in_tokens": s}
    for side in args.sides.split(","):
        out, key_path = paths(side)
        items, key = [], {}
        for g, (lo, hi) in enumerate(GROUPS):
            mine = [i for i, r in enumerate(rows) if lo <= r["L"] <= hi]
            paired = {i: mine[(k + 1) % len(mine)] for k, i in enumerate(mine)}
            for i in mine:
                r = rows[i]
                ls = lists(r)
                write_shown = set(ls["top"]) | set(ls["distinct"])
                for kind, name in (("matched", r["persona"]), ("control", rows[paired[i]]["persona"])):
                    iid = f"{len(items):04d}"
                    items.append({"item": iid, "name": name} | {k: show(ls[k], tok) for k in SHOWN[side]})
                    key[iid] = {"row": i, "kind": kind, "group": g, "layer": r["L"], "neuron": r["n"], "sign": r["s"],
                                "name": name, "persona": r["persona"], "generic": bool(GENERIC.search(name)),
                                "strict": bool(set(name_tokens(tok, name)) & write_shown)}
        order = list(range(len(items)))
        random.Random(1).shuffle(order)
        items = [items[i] for i in order]
        out.mkdir(parents=True, exist_ok=True)
        described = "\n".join(f"- `{k}`: {LISTS[k]};" for k in SHOWN[side])
        for b in range(0, len(items), BATCH):
            k = b // BATCH
            (out / f"batch_{k:02d}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items[b:b + BATCH]) + "\n")
            (out / f"prompt_{k:02d}.md").write_text(PROMPT.format(
                n=len(items[b:b + BATCH]), lists=described, batch=out / f"batch_{k:02d}.jsonl",
                answers=out / f"answers_{k:02d}.jsonl"))
        key_path.write_text(json.dumps(key, ensure_ascii=False))
        print(f"{side}: {len(rows)} rows, {len(items)} items in {(len(items) + BATCH - 1) // BATCH} batches under {out}")


def scored(side: str) -> dict:
    out, key_path = paths(side)
    if not key_path.exists():
        return {}
    key = json.loads(key_path.read_text())
    got = {}
    for path in sorted(out.glob("answers_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                got[o["item"]] = o
    return {k: v | {"score": got[k]["score"], "why": got[k].get("tokens", [])} for k, v in key.items() if k in got}


def grade(args) -> None:
    sides = {s: scored(s) for s in SIDES}
    sides = {s: v for s, v in sides.items() if v}
    lines = ["Neuron Persona Library personas against a pole's reads, judged generously and blind by Claude, from each "
             "side: write (the lens of record's top 10 and top 8 layer-contrastive tokens), input (the R-lens read from "
             "the MLP input, top 10, and the S-lens read, raw top 10 through L8, layer-contrastive top 8 after), both. "
             + "; ".join(f"{s}: {len(v)} items scored" for s, v in sides.items()) + ". Share of rows scoring 2+ (clear: "
             "the concept, a translation, or a strongly associated thing), 3 (the concept itself) and 1+ (loose), "
             "matched | control (another sampled row's name); strict: a name word's first token among the write side's "
             "tokens", ""]
    table = {}
    library = persona_rows()
    for side, sc in sides.items():
        for subset, keep in (("all names", lambda v: True), ("specific names", lambda v: not v["generic"])):
            # The sample takes the same number of rows per group; the library is mostly L6-L17.
            weight = [sum(lo <= r["L"] <= hi and keep({"generic": bool(GENERIC.search(r["persona"]))}) for r in library)
                      for lo, hi in GROUPS]
            lines.append(f"{side}, {subset}")
            lines.append("layers    rows   strict       2+ clear      3 itself      1+ loose")
            t = table.setdefault(side, {}).setdefault(subset, {})
            for g, (lo, hi) in list(enumerate(GROUPS)) + [(None, (0, 35))]:
                sel = [v for v in sc.values() if keep(v) and (g is None or v["group"] == g)]
                m = [v for v in sel if v["kind"] == "matched"]
                c = [v for v in sel if v["kind"] == "control"]
                if not m:
                    continue

                def share(vs: list, cond) -> float:
                    return sum(map(cond, vs)) / max(1, len(vs))
                row = {name: (share(m, cond), share(c, cond)) for name, cond in
                       (("strict", lambda v: v["strict"]), ("clear", lambda v: v["score"] >= 2),
                        ("itself", lambda v: v["score"] == 3), ("loose", lambda v: v["score"] >= 1))}
                t[f"L{lo}-{hi}"] = row | {"rows": len(m)}
                lines.append(f"L{lo:>2d}-{hi:<2d}  {len(m):5d}   " + "   ".join(f"{a:.2f} | {b:.2f}" for a, b in row.values()))
            groups = [t[f"L{lo}-{hi}"] for lo, hi in GROUPS]
            weighted = {k: tuple(sum(w * gr[k][i] for w, gr in zip(weight, groups)) / sum(weight) for i in (0, 1))
                        for k in ("strict", "clear", "itself", "loose")}
            t["library-weighted"] = weighted
            lines.append(f"weighted {sum(weight):6d}   " + "   ".join(f"{a:.2f} | {b:.2f}" for a, b in weighted.values())
                         + "   (groups weighted by the library's rows)")
            lines.append("")
    if all(s in sides for s in SIDES):
        # The same matched rows under each condition: does "both" find what either side alone finds?
        by = {s: {(v["row"], v["kind"]): v["score"] >= 2 for v in sides[s].values()} for s in SIDES}
        common = set.intersection(*(set(d) for d in by.values()))
        group = {(v["row"], v["kind"]): v["group"] for v in sides["write"].values()}
        lines.append("how the sides combine (clear, 2+), by group, matched | control: write only, input only, both judged "
                     "together, and either side judged alone")
        comb = {}
        for g, (lo, hi) in list(enumerate(GROUPS)) + [(None, (0, 35))]:
            cells = []
            for kind in ("matched", "control"):
                ks = [k for k in common if k[1] == kind and (g is None or group[k] == g)]
                n = max(1, len(ks))
                cells.append({"write": sum(by["write"][k] for k in ks) / n, "input": sum(by["input"][k] for k in ks) / n,
                              "both": sum(by["both"][k] for k in ks) / n,
                              "either": sum(by["write"][k] or by["input"][k] for k in ks) / n, "rows": len(ks)})
            comb[f"L{lo}-{hi}"] = {"matched": cells[0], "control": cells[1]}
            lines.append(f"  L{lo:>2d}-{hi:<2d} " + "   ".join(
                f"{name} {cells[0][name]:.2f} | {cells[1][name]:.2f}" for name in ("write", "input", "both", "either")))
        table["combined"] = comb
        lines.append("")
    for side, sc in sides.items():
        hits = sorted((v for v in sc.values() if v["kind"] == "matched" and v["score"] >= 2), key=lambda v: (v["layer"], v["neuron"]))
        lines.append(f"examples of clear matches, {side}: " + "; ".join(
            f"L{v['layer']} n{v['neuron']} {'+' if v['sign'] > 0 else '−'} {v['name']}: {', '.join(v['why'][:4])}"
            for v in hits[::max(1, len(hits) // 12)]))
    text = "\n".join(lines)
    print(text)
    (SUMMARY / "library_judge_qwen3-8b.log").write_text(text + "\n")
    (SUMMARY / "library_judge_qwen3-8b.json").write_text(json.dumps(table))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("make", "grade"))
    ap.add_argument("--per-group", type=int, default=60)
    ap.add_argument("--sides", default=",".join(SIDES))
    args = ap.parse_args()
    make(args) if args.what == "make" else grade(args)


if __name__ == "__main__":
    main()
