"""How close a pole's strongest single-token swap comes to its firings (findings §6, §7). No model.

Per pole that isn't silent, ``|drivers_val[0]| / act_top``: the activation of its top exact driver (``07_drivers.py``: one token at one
position, averaged over 8 documents) as a share of its typical top firing. Medians over each band's poles. Writes
``data/summary/driver_reach_<model>.{log,json}``.

    python scripts/checks/driver_reach.py
"""
import json

import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import MODEL, column, neuron_table
from neuron_lens.paths import SUMMARY

STEM = arch.stem(MODEL)
L = arch.text_config(MODEL).num_hidden_layers


def main() -> None:
    per = []
    for layer in range(L):
        top = column(pq.read_table(neuron_table("drivers", layer), columns=["drivers_val"]), "drivers_val")[:, 0].float().abs()
        act = torch.from_numpy(pq.read_table(neuron_table("acts", layer), columns=["act_top"])["act_top"].to_numpy().copy()).float()
        quiet = torch.from_numpy(pq.read_table(neuron_table("legible", layer), columns=["silent"])["silent"].to_numpy(zero_copy_only=False).copy())
        per.append((top / act.clamp_min(1e-3))[~quiet])
    table, lines = {}, [f"the top exact driver's activation over act_top, {MODEL}: medians over each band's poles that aren't silent", ""]
    for lo, hi in tuple(arch.bands(L)) + ((0, L - 1),):
        table[f"L{lo}-{hi}"] = torch.cat(per[lo:hi + 1]).median().item()
        lines.append(f"L{lo}-{hi}".ljust(9) + f"{table[f'L{lo}-{hi}']:.2f}")
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"driver_reach_{STEM}.log").write_text(text + "\n")
    (SUMMARY / f"driver_reach_{STEM}.json").write_text(json.dumps(table))


if __name__ == "__main__":
    main()
