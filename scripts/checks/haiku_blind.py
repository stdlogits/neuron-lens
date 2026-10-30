"""``lens_blind.py``'s blind test on 100 more poles, with Haiku explainers and the depth-aware cards
of ``10_labels.py`` as extra arms (findings §8).

Only the ``cards`` stage is new: these poles (``data/summary/haiku_blind_poles.json``: both poles of
50 random neurons over 24 layers) instead of a fresh sample, and two more card kinds:
- ``v8``: ``10_labels.py``'s card without its activation and context lines;
- ``v8both``: the card as labelled.

``measure``, ``prompt`` and ``grade`` are ``lens_blind.py``'s, unchanged, with ``OUT`` and the
summary names moved (``lens_blind_qwen3-8b-haiku.*``) so ``lens_blind.py``'s files are untouched.
Arms (card, explainer): ``h_lens`` and ``h_both``: ``lens_blind.py``'s cards and explainer brief
(``lens_blind_explain.md``), Haiku; ``h_v8`` and ``h_v8both``: the new cards with
``haiku_explain_lens.md`` and ``scripts/prompts/qwen3-8b/neurons.md``, Haiku; ``s_both``: ``lens_blind.py``'s
both card, Sonnet. Explainers, scorers and judges ran as Claude Code subagents with
``lens_blind.py``'s briefs.

    python scripts/checks/haiku_blind.py cards    # after 10_labels.py cards
    python scripts/checks/haiku_blind.py measure --device cuda
    python scripts/checks/haiku_blind.py prompt | grade
"""
import argparse
import importlib.util
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import lens_blind as lb

from neuron_lens.paths import DATA, SUMMARY

HERE = Path(__file__).resolve().parent
lb.OUT = DATA / "blind" / f"{lb.NAME}-haiku"
lb.NAME = f"{lb.NAME}-haiku"  # summary files lens_blind_qwen3-8b-haiku.*
lb.PER_BAND = "about 20"  # only printed in grade's header
lb.ARMS = {"h_lens": ("lens", "haiku"), "h_both": ("both", "haiku"), "h_v8": ("v8", "haiku"),
           "h_v8both": ("v8both", "haiku"), "s_both": ("both", "sonnet")}
POLES = SUMMARY / "haiku_blind_poles.json"
CONTEXT_LINE = re.compile(r"^(- activation: |  - context: |- near-silent: )")


def cards(args) -> None:
    import torch
    from transformers import AutoTokenizer

    from neuron_lens.catalog import (
        N_CONTEXTS,
        acts_path,
        context,
        doc_tokens,
        load_layer,
    )

    spec = importlib.util.spec_from_file_location("labels", HERE.parent / "10_labels.py")
    labels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(labels)
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(lb.MODEL)
    show = lambda ids: " | ".join(tok.decode([i]).replace("\n", "⏎") for i in ids[:lb.N_SHOW])
    tokens = doc_tokens(torch.load(acts_path(), weights_only=False, map_location="cpu", mmap=True))
    by_layer = defaultdict(list)
    for p in json.loads(POLES.read_text()):
        by_layer[p["layer"]].append(p)
    poles, lens_c, ctx_c, v8, checked, matched = {}, {}, {}, {}, 0, 0
    for layer, ps in sorted(by_layer.items()):
        t = load_layer(layer)
        f = t.num_rows // 2
        built = {}
        with open(labels.OUT["neurons"] / "cards" / f"L{layer:02d}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                o = json.loads(line)
                if any(o["unit"] == p["neuron"] for p in ps):
                    built[o["unit"]] = o["card"]
        for p in ps:
            n, s = p["neuron"], p["pole"]
            e = t.slice(n if s > 0 else f + n, 1).to_pylist()[0]
            pid = lb.pole_id(layer, n, s)
            lens_c[pid] = [f"### {pid}", f"- layer {layer} of 36"] + [f"- {title}: {show(e[key])}" for key, title in lb.LENS_READS]
            silent = e["act_top"] < lb.SILENT
            lines = [f"- activation: typical high {e['act_top']:.2f}, max {e['act_max']:.2f}; fires (past a "
                     f"quarter of typical) at {e['rate']:.1e} of positions; by slice: "
                     + ", ".join(f"{k} {e[f'rate_{k}']:.1e}" for k in ("dclm", "zh", "code", "chat"))]
            shown = [c for c in str(e["contexts"]).split(" ‖ ")[:N_CONTEXTS] if c]
            lines += ["- near-silent: no top context recorded"] if silent else [f"  - context: {c}" for c in shown]
            ctx_c[pid] = lines
            fired = [j for j, v in enumerate(e["top_val"]) if v > 0]
            if not silent:  # the activation file's tokens must reproduce the catalog's decoded contexts
                for j, c in zip(fired[:N_CONTEXTS], shown):
                    checked += 1
                    matched += context(tok, tokens[e["top_doc"][j]], int(e["top_pos"][j])) == c
            held = fired[N_CONTEXTS:]
            pos = [] if silent else [(int(e["top_doc"][j]), int(e["top_pos"][j]), float(e["top_val"][j]))
                                     for j in sorted(rng.sample(held, min(lb.N_POS, len(held))))]
            poles[pid] = {"id": pid, "layer": layer, "neuron": n, "sign": s, "band": lb.band_of(layer),
                          "act_top": e["act_top"], "silent": silent, "pos": pos, "write_R_top": e["write_R_ids"]}
            v8[(layer, n)] = built[n]
    print(f"context check: {matched}/{checked} rebuilt contexts equal the catalog's", flush=True)
    assert matched >= 0.98 * checked, "the activation file's tokens don't line up with the catalog"
    order = list(poles)
    rng.shuffle(order)
    out = lb.OUT / "cards"
    out.mkdir(parents=True, exist_ok=True)
    for kind, body in (("lens", lambda pid: lens_c[pid]), ("both", lambda pid: lens_c[pid] + ctx_c[pid])):
        for b in range(0, len(order), lb.CARDS_PER_FILE):
            (out / f"{kind}_{b // lb.CARDS_PER_FILE:02d}.md").write_text(
                "\n\n".join("\n".join(body(pid)) for pid in order[b:b + lb.CARDS_PER_FILE]) + "\n")
    neurons = list(dict.fromkeys((poles[pid]["layer"], poles[pid]["neuron"]) for pid in order))
    for kind in ("v8", "v8both"):
        for b in range(0, len(neurons), 20):
            blocks = [v8[k] if kind == "v8both" else "\n".join(x for x in v8[k].split("\n") if not CONTEXT_LINE.match(x))
                      for k in neurons[b:b + 20]]
            (out / f"{kind}_{b // 20:02d}.md").write_text("\n\n".join(blocks) + "\n")
    (lb.OUT / "poles.json").write_text(json.dumps(list(poles.values()), ensure_ascii=False))
    docs = {d for p in poles.values() for d, _, _ in p["pos"]}
    random_docs = sorted(rng.sample([d for d in range(len(tokens)) if len(tokens[d]) >= 64], lb.N_RANDOM_DOCS))
    torch.save({"poles": list(poles.values()), "random_docs": random_docs,
                "docs": {d: tokens[d].tolist() for d in sorted(docs | set(random_docs))}}, lb.OUT / "gpu_in.pt")
    print(f"{len(poles)} poles, {len(docs)} positive documents, {lb.N_RANDOM_DOCS} random -> {lb.OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("cards", "measure", "prompt", "grade"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261031)
    args = ap.parse_args()
    lb.OUT.mkdir(parents=True, exist_ok=True)
    {"cards": cards, "measure": lb.measure, "prompt": lb.prompt, "grade": lb.grade}[args.what](args)


if __name__ == "__main__":
    main()
