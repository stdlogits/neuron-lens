"""How deep a read's signal goes (findings §7): per rank of each catalog read (to its 64 tokens), how
often that token shows up in the pole's behaviour, against a mismatched pole. It set the catalog's
``TOP``. No model.

Each pole's top firings are the catalog's (``acts``: the strongest position in each of its top 24
documents), those with a positive value. A pole's read hits at rank r when its rank-r token is
- for writes (``write_R``, ``write_R_z``): the token after one of its firings (none past the
  document's end);
- for reads (``read_S``, through the first third of the layers, L12 for Qwen3-8B; ``read_R``,
  ``read_R_z``): the peak token of one of its firings.

Per layer, read and rank band (1-8, 9-16, 17-24, 25-32, 33-48, 49-64): the share of all the layer's
poles that hit, averaged over the band's ranks, matched and mismatched (each pole's tokens against
the read of a random other pole of the same layer and sign, one permutation for every layer, seed 0).
Nine layers: L2, L4, then every fourth from L8 to four before the last (Qwen3-8B: L8-L32). The token
lists are ``08_reads.py``'s, over ``catalog.kept()``: a rank-64 token is the 64th kept token.

Documents come from ``catalog.flat_tokens``: the activations file, else rebuilt from the corpus, as
released. Log and JSON in
``data/summary/rank_depth_<model>.{log,json}`` (figure 13).

    python scripts/checks/rank_depth.py
    NEURON_LENS_MODEL=google/gemma-4-12B python scripts/checks/rank_depth.py   # any catalog model
"""
import json

import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import MODEL, TOP, column, firing_tokens, flat_tokens, neuron_table
from neuron_lens.paths import SUMMARY

STEM = arch.stem(MODEL)
BINS = [(1, 8), (9, 16), (17, 24), (25, 32), (33, 48), (49, 64)]
READS = {"write_R": "next", "write_R_z": "next", "read_S": "peak", "read_R": "peak", "read_R_z": "peak"}


def main() -> None:
    L = arch.text_config(MODEL).num_hidden_layers
    assert BINS[-1][1] == TOP, f"the bands end at rank {BINS[-1][1]}, the catalog keeps {TOP}"
    layers = (2, 4, *range(8, L - 3, 4))
    flat, start = flat_tokens()
    f = pq.read_metadata(neuron_table("acts", 0)).num_rows // 2
    g = torch.Generator().manual_seed(0)
    perm = torch.cat([torch.randperm(f, generator=g), f + torch.randperm(f, generator=g)])
    out = {k: {"target": v, "layer": [], "matched": [], "mismatched": []} for k, v in READS.items()}
    lines = [f"read rank depth, {MODEL}: share of poles whose rank-r token shows up in its behaviour (writes: the "
             "token after a top firing; reads: the peak token), matched / mismatched pole, by rank band"]
    for layer in layers:
        acts = pq.read_table(neuron_table("acts", layer), columns=["top_doc", "top_pos", "top_val"])
        doc, pos = column(acts, "top_doc").long(), column(acts, "top_pos").long()
        fired = column(acts, "top_val") > 0
        tokens = {"peak": torch.where(fired, firing_tokens(flat, start, doc, pos), -2),
                  "next": torch.where(fired, firing_tokens(flat, start, doc, pos, 1), -2)}  # −1 past the end: no hit either
        names = [k for k in READS if k != "read_S" or layer <= L // 3]
        reads = pq.read_table(neuron_table("reads", layer), columns=[f"{k}_ids" for k in names])
        for k in names:
            ids, tg = column(reads, f"{k}_ids"), tokens[READS[k]]
            m, c = ((x[:, :, None] == tg[:, None, :]).any(-1).float().mean(0) for x in (ids, ids[perm]))
            o = out[k]
            o["layer"].append(layer)
            o["matched"].append([m[a - 1:b].mean().item() for a, b in BINS])
            o["mismatched"].append([c[a - 1:b].mean().item() for a, b in BINS])
            lines.append(f"L{layer:<2d} {k:9s} -> {READS[k]}  " + "  ".join(
                f"r{a}-{b} {x:.4f}/{y:.4f}" for (a, b), x, y in zip(BINS, o["matched"][-1], o["mismatched"][-1])))
            print(lines[-1], flush=True)
    (SUMMARY / f"rank_depth_{STEM}.log").write_text("\n".join(lines) + "\n")
    (SUMMARY / f"rank_depth_{STEM}.json").write_text(json.dumps({"model": MODEL, "bins": BINS, "reads": out}))


if __name__ == "__main__":
    main()
