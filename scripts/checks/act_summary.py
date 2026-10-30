"""Summarize the catalog's neuron activation tables (``06_activations.py``) by layer.

Per layer, over its poles: the median typical high activation (``act_top``) and the largest;
the share of poles that never fired in the corpus (no top document), that fire rarely
(``rate`` under 1e-4 of positions), that stay near zero (``act_top`` under 0.05), and that fire
harder at the first position than anywhere else in the corpus (``first_max`` over ``act_max``);
and the share selective for one slice (its exact
rate at least 10 times every other slice's and above 1e-3). Then the model's largest activations
with their peak tokens, and the most selective poles of each slice.

    python scripts/checks/act_summary.py
"""
import argparse

import pyarrow.parquet as pq
import torch

from neuron_lens.catalog import MODEL, catalog_dir
from neuron_lens.paths import SUMMARY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    files = sorted((catalog_dir() / "neurons" / "acts").glob("L*.parquet"))
    lines, big, sel = [], [], {}
    slices = None
    lines.append("layer  act_top med  max      never  rare   quiet  first  selective by slice")
    for path in files:
        layer = int(path.stem[1:])
        t = pq.read_table(path).to_pydict()
        f = len(t["act_top"]) // 2
        slices = slices or [k.removeprefix("rate_") for k in t if k.startswith("rate_")]
        first = (torch.tensor(t["first_max"]) > torch.tensor(t["act_max"])).float().mean().item()
        top = torch.tensor(t["act_top"])
        mx = torch.tensor(t["act_max"])
        rate = torch.tensor(t["rate"])
        never = (mx <= 0).float().mean()
        rare = (rate < 1e-4).float().mean()
        quiet = (top < 0.05).float().mean()
        rates = torch.stack([torch.tensor(t[f"rate_{s}"]) for s in slices], 1)  # [poles, slices]
        cells = []
        for i, s in enumerate(slices):
            others = torch.cat([rates[:, :i], rates[:, i + 1:]], 1).max(1).values
            m = (rates[:, i] > 1e-3) & (rates[:, i] >= 10 * others)
            cells.append(f"{s} {m.float().mean():.3f}")
            score = rates[:, i] / (others + 1e-4)
            for row in torch.where(m)[0][score[m].argsort(descending=True)[:3]].tolist():
                sel.setdefault(s, []).append((score[row].item(), layer, row % f, 1 if row < f else -1,
                                              rates[row, i].item(), t["peak_tokens"][row]))
        lines.append(f"L{layer:<4d}  {top.median():9.3f}  {mx.max():7.1f}  {never:.3f}  {rare:.3f}  "
                     f"{quiet:.3f}  {first:.3f}  " + "  ".join(cells))
        for row in mx.topk(3).indices.tolist():
            big.append((mx[row].item(), layer, row % f, 1 if row < f else -1, t["peak_tokens"][row]))
    lines.append("\nlargest activations (act_max, pole, peak tokens)")
    for v, layer, n, p, peaks in sorted(big, reverse=True)[:args.top]:
        lines.append(f"  {v:8.1f}  L{layer} n{n} {'+' if p > 0 else '−'}  {peaks}")
    for s in slices or []:
        lines.append(f"\nmost {s}-selective poles (rate there / max elsewhere, rate there, peak tokens)")
        for score, layer, n, p, r, peaks in sorted(sel.get(s, []), reverse=True)[:args.top]:
            lines.append(f"  {score:8.0f}  L{layer} n{n} {'+' if p > 0 else '−'}  {r:.3f}  {peaks}")
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"act_summary_{MODEL.split('/')[-1].lower()}.log").write_text(text + "\n")


if __name__ == "__main__":
    main()
