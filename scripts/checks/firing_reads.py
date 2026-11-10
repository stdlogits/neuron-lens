"""Which firing reads say what the weight reads do not? (``17_firings.py``, findings §18). Run after the firing pass.

For each pole with firings, by band, against the next pole of the layer (the control):

- how much each read's top ``K`` tokens overlaps the pole's ``write_R``, ``read_R`` and the other firing reads;
- how often the pole's steering effect (§3's sample; 1× ``act_top``) promotes the read's tokens: of the effect's top ``K``
  tokens, the share in each read's top ``K``;
- whether ``pred`` names what follows: of its top ``K``, the share among the tokens after the pole's firings.

Writes ``data/summary/firing_reads_<model>.{log,json}``.

    python scripts/checks/firing_reads.py [--k 16]
"""
import argparse
import json

import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import MODEL, column, firing_tokens, flat_tokens, kept, neuron_table
from neuron_lens.paths import DATA, SUMMARY

STEM = arch.stem(MODEL)
SITES = ("before_R", "mlp_R", "pred", "before_R_z", "mlp_R_z", "pred_z")
REFS = ("write_R", "write_orth_R", "read_R")


def overlap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a[:, :, None] == b[:, None, :]).any(-1).float().mean(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()
    k = args.k
    L = arch.text_config(MODEL).num_hidden_layers
    flat, start = flat_tokens()
    causal = torch.load(DATA / "causal" / f"{STEM}-causal.pt", weights_only=False)
    picks = {(int(l), int(n)): i for i, (l, n) in enumerate(causal["picks"])}
    keep = kept()
    table, lines = {}, [f"firing reads, {MODEL}: top {k} tokens; matched (control: the next pole), medians over poles with firings", ""]
    for lo, hi in arch.bands(L):
        acc = {}
        for l in range(lo, hi + 1):
            fr = pq.read_table(neuron_table("firings", l), columns=[f"{s}_ids" for s in SITES] + ["n_firings"])
            rd = pq.read_table(neuron_table("reads", l), columns=[f"{r}_ids" for r in REFS])
            ac = pq.read_table(neuron_table("acts", l), columns=["top_doc", "top_pos", "top_val"])
            ok = torch.from_numpy(fr.column("n_firings").to_numpy().copy()) > 0
            for s in SITES:
                x = column(fr, f"{s}_ids", k).long()
                for r in REFS:
                    y = column(rd, f"{r}_ids", k).long()
                    acc.setdefault(f"{s}~{r}", []).append(overlap(x, y)[ok])
                    acc.setdefault(f"{s}~{r}|c", []).append(overlap(x, y.roll(1, 0))[ok])
                if s.startswith("pred"):
                    doc, pos = column(ac, "top_doc").long(), column(ac, "top_pos").long()
                    nxt = firing_tokens(flat, start, doc, pos, 1).where(column(ac, "top_val") > 0, torch.tensor(-1))
                    acc.setdefault(f"{s}~next", []).append(overlap(x, nxt)[ok])
                    acc.setdefault(f"{s}~next|c", []).append(overlap(x.roll(1, 0), nxt)[ok])
            f = fr.num_rows // 2
            rows, effs = [], []
            for n in range(f):
                i = picks.get((l, n))
                if i is None:
                    continue
                for pi in (0, 1):
                    rows.append(pi * f + n)
                    effs.append(causal["steer"][i, pi, 0].float()[keep].topk(k).indices)
            if rows:
                idx = torch.tensor(rows)
                eff = keep[torch.stack(effs)]
                for s in SITES:
                    x = column(fr, f"{s}_ids", k).long()[idx]
                    acc.setdefault(f"{s}~steer", []).append(overlap(x, eff)); acc.setdefault(f"{s}~steer|c", []).append(overlap(x.roll(1, 0), eff))
                for r in REFS:
                    x = column(rd, f"{r}_ids", k).long()[idx]
                    acc.setdefault(f"{r}~steer", []).append(overlap(x, eff)); acc.setdefault(f"{r}~steer|c", []).append(overlap(x.roll(1, 0), eff))
        row = {name: torch.cat(v).mean().item() for name, v in acc.items()}
        table[f"L{lo}-{hi}"] = row
        lines.append(f"L{lo}-{hi}")
        for name, v in sorted(row.items()):
            if not name.endswith("|c"):
                lines.append(f"  {name:30s} {v:.3f}  (control {row.get(name + '|c', float('nan')):.3f})")
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"firing_reads_{STEM}.log").write_text(text + "\n")
    (SUMMARY / f"firing_reads_{STEM}.json").write_text(json.dumps(table))


if __name__ == "__main__":
    main()
