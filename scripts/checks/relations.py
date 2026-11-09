"""How the three relations between units compare (findings §17): wired to (``edges``, ``16_edges.py``), driven by
and fires with (``sources``, ``18_sources.py``), per pole, by band.

For each pole's top ``K`` of each list: the share of its driven-by sources that are in its wired-to list, in its
fires-with list, in either; and the same against the next pole's lists, the control. Writes
``data/summary/relations_<model>.{log,json}``.

    python scripts/checks/relations.py [--k 16]
"""
import argparse
import json

import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import MODEL, column, neuron_table
from neuron_lens.paths import SUMMARY

STEM = arch.stem(MODEL)


def ids(t, side: str, k: int, f: int) -> torch.Tensor:
    l, n, p = (column(t, f"{side}_{c}", k).long() for c in ("layer", "neuron", "pole"))
    return torch.where(l >= 0, l * 2 * f + n + f * (p < 0), -1)


def share(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per row, the share of ``a``'s valid entries found in ``b``'s row."""
    hit = ((a[:, :, None] == b[:, None, :]) & (a[:, :, None] >= 0)).any(-1).float()
    return hit.sum(1) / (a >= 0).sum(1).clamp_min(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()
    L = arch.text_config(MODEL).num_hidden_layers
    rows, table = [f"relations, {MODEL}: of each pole's top {args.k} driven-by sources, the share also in its top {args.k} wired-to / fires-with / either (control: the next pole's lists); medians over poles with a firing", ""], {}
    for lo, hi in arch.bands(L) + [(0, L - 1)]:
        lo = max(lo, 1)
        acc = {k: [] for k in ("w", "f", "either", "w_c", "f_c")}
        for l in range(lo, hi + 1):
            s, e = pq.read_table(neuron_table("sources", l)), pq.read_table(neuron_table("edges", l))
            f = s.num_rows // 2
            d = ids(s, "driven", args.k, f)
            w, fw = ids(e, "in", args.k, f), ids(s, "with", args.k, f)
            used = torch.from_numpy(s.column("n_firings").to_numpy().copy()) > 0
            acc["w"].append(share(d, w)[used]); acc["f"].append(share(d, fw)[used])
            acc["either"].append(share(d, torch.cat([w, fw], 1))[used])
            acc["w_c"].append(share(d, w.roll(1, 0))[used]); acc["f_c"].append(share(d, fw.roll(1, 0))[used])
        row = {k: torch.cat(v).median().item() for k, v in acc.items()}
        table[f"L{lo}-{hi}"] = row
        rows.append(f"L{lo}-{hi}".ljust(8) + "  ".join(f"{k} {v:.3f}" for k, v in row.items()))
    text = "\n".join(rows)
    print(text)
    (SUMMARY / f"relations_{STEM}.log").write_text(text + "\n")
    (SUMMARY / f"relations_{STEM}.json").write_text(json.dumps(table))


if __name__ == "__main__":
    main()
