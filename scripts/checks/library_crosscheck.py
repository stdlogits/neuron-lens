"""Write-side lens reads against an outside steering census: the Neuron Persona Library.

The library (jfonsecarivera on X, inspired by Sauers' single-neuron steering; neuronpersonas.com,
also qwen-neurons-site.vercel.app) pushed every Qwen3-8B MLP neuron up and down
on "Who are you?" in the chat template, thinking off, greedy, and had Claude Haiku label the
reply. Its labels say what a strong push makes the model claim to be, a method of its own (one
prompt, a speaker frame, strong doses), not what the neuron means. A local copy of its data is
at ``data/external/qwen_neuron_library/library.json`` (fetched 2026-10-27).

For each persona row (layer, neuron, sign, name), and each lens, the best rank in the vocabulary
of the name's words (first token of each word, with and without a leading space and capital)
under the lens's logits for the pole ``s·W_down[:, n]``, plainly and z-scored across the
library's rows of that layer (layer-contrastive). Control: the same with each row's name swapped
for the next row's of the same layer. Hit rates at ranks 10, 100 and 1,000 by layer group.

    python scripts/checks/library_crosscheck.py --lenses logit,J,R:dclm
"""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from neuron_lens.lenses import load_lens
from neuron_lens.paths import LIBRARY, SUMMARY
from neuron_lens.readout import rms_norm_columns, transport
from neuron_lens.weights import down_proj, readout_names, tensors

MODEL = "Qwen/Qwen3-8B"
STOP = {"the", "of", "and", "old", "year", "years", "with", "from", "for", "who", "one", "your", "you"}
GROUPS = [(0, 5), (6, 11), (12, 17), (18, 23), (24, 29), (30, 35)]


def name_tokens(tok, name: str) -> list[int]:
    ids = set()
    for w in re.findall(r"[A-Za-z]+", name):
        if len(w) < 3 or w.lower() in STOP:
            continue
        for form in (w, w.lower(), w.capitalize()):
            for text in (form, " " + form):
                ids.add(tok.encode(text, add_special_tokens=False)[0])
    return sorted(ids)


def best_rank(logits: torch.Tensor, cands: list[list[int]]) -> torch.Tensor:
    """For each column of ``logits`` ``[V, n]``, the best (0-based) vocabulary rank of its candidates."""
    width = max(1, max(len(c) for c in cands))
    idx = torch.tensor([c + [c[0] if c else 0] * (width - len(c)) for c in cands]).T  # [width, n]
    best = logits.gather(0, idx.to(logits.device)).max(0).values  # the best candidate's logit
    rank = (logits > best[None]).sum(0).cpu()
    rank[torch.tensor([not c for c in cands])] = 10**9
    return rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, default=LIBRARY)
    ap.add_argument("--lenses", default="logit,J,R:dclm")
    args = ap.parse_args()
    lib = json.loads(args.library.read_text())
    fields, persona = lib["fields"], lib["categories"].index("persona")
    rows = [dict(zip(fields, r)) for r in lib["rows"]]
    rows = [r for r in rows if r["cat"] == persona]
    tok = AutoTokenizer.from_pretrained(MODEL)
    nv = len(tok)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    head = tensors(MODEL, readout_names())
    unembed = head["lm_head.weight"][:nv].to(dev, torch.float16)
    gamma = head["model.norm.weight"].to(dev, torch.float32)
    specs = args.lenses.split(",")
    lenses = {s: load_lens(s, MODEL) for s in specs}
    ranks = {(s, kind, ctrl): [] for s in specs for kind in ("plain", "z") for ctrl in (0, 1)}
    layer_of = []
    for layer in range(36):
        mine = [r for r in rows if r["L"] == layer]
        if len(mine) < 2:
            continue
        wd = tensors(MODEL, [down_proj(layer)])[down_proj(layer)].to(dev, torch.float32)
        cols = torch.stack([r["s"] * wd[:, r["n"]] for r in mine], 1)
        cands = [name_tokens(tok, r["persona"]) for r in mine]
        swapped = cands[1:] + cands[:1]
        for s in specs:
            jac = None if lenses[s] is None else lenses[s].get(layer)
            lg = (unembed @ rms_norm_columns(transport(cols, jac), gamma).half()).float()
            z = (lg - lg.mean(1, keepdim=True)) / lg.std(1, keepdim=True).clamp_min(1e-6)
            for kind, m in (("plain", lg), ("z", z)):
                ranks[(s, kind, 0)].append(best_rank(m, cands))
                ranks[(s, kind, 1)].append(best_rank(m, swapped))
        layer_of += [layer] * len(mine)
        print(f"L{layer}: {len(mine)} personas", flush=True)
    layer_of = torch.tensor(layer_of)
    groups = GROUPS + [(0, 35)]
    table = {"lenses": specs, "groups": groups, "ks": [10, 100, 1000],
             "rows": [int(((layer_of >= lo) & (layer_of <= hi)).sum()) for lo, hi in groups], "share": {}}
    lines = [f"Neuron Persona Library personas ({len(layer_of)} rows) against write-side lens reads, "
             f"{MODEL}: share of rows whose name ranks in the top 10 / 100 / 1000 of the pole's read "
             "(matched | control: next row's name, same layer)"]
    for kind in ("plain", "z"):
        lines.append(f"\n{kind} logits")
        lines.append("layers  rows " + " ".join(f"{s:>34s}" for s in specs))
        for lo, hi in groups:
            sel = (layer_of >= lo) & (layer_of <= hi)
            cells = []
            for s in specs:
                m, c = torch.cat(ranks[(s, kind, 0)])[sel], torch.cat(ranks[(s, kind, 1)])[sel]
                cells.append(" ".join(f"{(m < k).float().mean():.2f}|{(c < k).float().mean():.2f}"
                                      for k in (10, 100, 1000)))
                for which, v in (("matched", m), ("control", c)):
                    share = table["share"].setdefault(kind, {}).setdefault(s, {}).setdefault(which, {})
                    for k in table["ks"]:
                        share.setdefault(str(k), []).append(round((v < k).float().mean().item(), 4))
            lines.append(f"L{lo:>2d}-{hi:<2d} {int(sel.sum()):5d} " + " ".join(f"{c:>34s}" for c in cells))
    text = "\n".join(lines)
    print(text)
    (SUMMARY / "library_crosscheck_qwen3-8b.log").write_text(text + "\n")
    (SUMMARY / "library_crosscheck_qwen3-8b.json").write_text(json.dumps(table))


if __name__ == "__main__":
    main()
