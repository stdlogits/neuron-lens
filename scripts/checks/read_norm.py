"""How the read rows meet the input norm's weight, and what that does to ``read_R`` (findings §13).

The catalog's read rows are the gate and up rows times the input norm's weight γ, the covector the activation applies to the
residual. That suits the S-lens, a pullback; the R-lens transports a vector. Two measurements, no model:

- per band, the share of ``read_R_z``'s top 64 in ``before_R_z``'s (what the residual holds at the top firings), against another
  pole of the layer and sign;
- on each of ``--layers``, ``read_R`` with the rows as row × γ, the raw row and row / γ, each read through the R-lens ``mid`` target
  and scored the same way and against the exact drivers (the plain read's top 10 in the top 20), with γ's spread.

Writes ``data/summary/read_norm_<model>.{log,json}``.

    python scripts/checks/read_norm.py --layers 16,30
    NEURON_LENS_MODEL=google/gemma-4-12B python scripts/checks/read_norm.py --layers 6,18,30,42
"""
import argparse
import json

import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import MODEL, TOP, column, kept, lobe_signs, neuron_table
from neuron_lens.neurons import mlp_names, read_rows, softcap
from neuron_lens.paths import SUMMARY
from neuron_lens.readout import layer_reads
from neuron_lens.weights import tensors

STEM = arch.stem(MODEL)
CFG = arch.text_config(MODEL)
L, F = CFG.num_hidden_layers, CFG.intermediate_size
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
CHUNK = 2048


def shared(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per row, the share of ``a``'s tokens in ``b`` (−1 is padding)."""
    out = torch.zeros(len(a))
    for s in range(0, len(a), CHUNK):
        x, y = a[s:s + CHUNK], b[s:s + CHUNK]
        out[s:s + CHUNK] = (((x[:, :, None] == y[:, None, :]).any(-1)) & (x >= 0)).float().mean(1)
    return out


def pair(g: torch.Generator) -> torch.Tensor:
    return torch.cat([torch.randperm(F, generator=g), F + torch.randperm(F, generator=g)])


def bands() -> tuple[list[str], dict]:
    g = torch.Generator().manual_seed(0)
    per = {}
    for layer in range(L):
        read = column(pq.read_table(neuron_table("reads", layer), columns=["read_R_z_ids"]), "read_R_z_ids").long()
        fr = pq.read_table(neuron_table("firings", layer), columns=["before_R_z_ids", "n_firings"])
        before, ok = column(fr, "before_R_z_ids").long(), torch.from_numpy(fr["n_firings"].to_numpy().copy()) > 0
        per[layer] = (shared(read, before)[ok].mean().item(), shared(read, before[pair(g)])[ok].mean().item())
    table, lines = {}, [f"read_R_z's top 64 in before_R_z's [another pole], {MODEL}, mean over layers"]
    for lo, hi in tuple(arch.bands(L)) + ((0, L - 1),):
        rows = [per[l] for l in range(lo, hi + 1)]
        table[f"L{lo}-{hi}"] = {"agree": sum(r[0] for r in rows) / len(rows), "control": sum(r[1] for r in rows) / len(rows)}
        lines.append(f"L{lo}-{hi}".ljust(9) + f"{table[f'L{lo}-{hi}']['agree']:.3f} [{table[f'L{lo}-{hi}']['control']:.3f}]")
    return lines, table


def variants(layers: list[int]) -> tuple[list[str], dict]:
    keep = kept()
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted({nm["unembed"], nm["norm"]}))
    unembed, gamma = head[nm["unembed"]][keep].to(DEV, torch.float16), head[nm["norm"]].to(DEV, torch.float32)
    J_mid = arch.lens(MODEL, "R_mid", arch.fits(MODEL)[0])
    g = torch.Generator().manual_seed(0)
    lines = [f"\nread_R by how the rows meet γ, {MODEL}: read_R_z's top 64 in before_R_z's [another pole]; the plain read's top 10 in the top 20 exact drivers"]
    table = {}
    for layer in layers:
        names = mlp_names(MODEL, layer)
        w = {k: tensors(MODEL, [names[k]])[names[k]].float().to(DEV) for k in ("gate", "up", "norm")}
        n = w["norm"]
        q = n.abs().quantile(torch.tensor([0.5, 0.99], device=DEV)).tolist()
        fr = pq.read_table(neuron_table("firings", layer), columns=["before_R_z_ids", "n_firings"])
        before, ok = column(fr, "before_R_z_ids").long(), torch.from_numpy(fr["n_firings"].to_numpy().copy()) > 0
        drv = column(pq.read_table(neuron_table("drivers", layer), columns=["drivers_ids"]), "drivers_ids").long()
        perm = pair(g)
        table[layer] = {"std_over_mean": (n.std() / n.mean()).item(), "p99_over_median": q[1] / q[0]}
        lines.append(f"L{layer}: γ std/mean {table[layer]['std_over_mean']:.2f}, 99th percentile / median {table[layer]['p99_over_median']:.0f}")
        for name, scale in (("row x norm", n), ("row", torch.ones_like(n)), ("row / norm", 1 / n.abs().clamp_min(1e-3) * n.sign())):
            a, b = w["gate"] * scale, w["up"] * scale
            a, b = a / a.norm(dim=1, keepdim=True), b / b.norm(dim=1, keepdim=True)
            c = read_rows(a.cpu(), b.cpu(), lobe_signs(layer)).to(DEV)
            reads = [layer_reads(c[i * F:(i + 1) * F].T, unembed, gamma, J_mid[layer], signs=(1,), k=TOP, kz=TOP, softcap=softcap(MODEL)) for i in (0, 1)]
            ids, zids = (keep[torch.cat([r[key][1] for r in reads])].long() for key in ("ids", "zids"))
            e = {"agree": shared(zids, before)[ok].mean().item(), "control": shared(zids, before[perm])[ok].mean().item(),
                 "drivers": shared(ids[:, :10], drv[:, :20])[ok].mean().item()}
            table[layer][name] = e
            lines.append(f"  {name:11s} {e['agree']:.3f} [{e['control']:.3f}]  drivers {e['drivers']:.3f}")
    return lines, table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="16,30")
    args = ap.parse_args()
    lines, by_band = bands()
    more, by_layer = variants([int(x) for x in args.layers.split(",")])
    text = "\n".join(lines + more)
    print(text)
    (SUMMARY / f"read_norm_{STEM}.log").write_text(text + "\n")
    (SUMMARY / f"read_norm_{STEM}.json").write_text(json.dumps({"bands": by_band, "layers": by_layer}))


if __name__ == "__main__":
    main()
