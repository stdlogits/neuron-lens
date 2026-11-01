"""Do the catalog's exact drivers survive bf16? ``07_drivers.py`` runs the model in bf16; this reruns its
neuron pass in fp32 through L9 and compares (findings §6).

The pass is 07's, imported from it: the same 8 documents, every kept token (``catalog.kept``) at
position 24 followed by the documents' own 16 tokens, the same order, the same reductions (the
activation at the token, and summed over it and the 16 after, averaged over the documents; a − pole
ranks by the negated activation), eager attention, the top 64 per pole. Two changes, neither of
which alters what is computed: the model is cut after layer 9 (``LAYERS``), and the documents' first
24 tokens, the same in every pass, run once, their keys and values copied to every batch. Fp32 on
MPS takes 6.5 s per batch of 128 tokens, about 90 minutes for the 103,830, so the fp32 drivers are
kept in ``data/checks/drivers-fp32-L00-L09.pt``, and with that file present only the comparison
runs. (The kept file is from the pre-port script, with SDPA attention; on 256 tokens this pass
reproduces its drivers and their values to 1e-5.) Delete it to rerun the pass.

Per layer, against the catalog's ``drivers`` table:
- overlap of the fp32 top 20 with the bf16 top 20 (the share of one in the other), at the token and
  summed, over all poles and over the responsive half (whose strongest fp32 driver at the token is at
  least the layer's median: many poles barely move for any token at position 24); the same for the
  top 64;
- the drivers both top 64s keep at the token: Pearson r of their fp32 and bf16 activations, and the
  median relative difference.

The bf16 drivers stand if the mean top-20 overlap, at the token and summed, stays at 0.95 or above
at L7-L9, the deepest layers checked; later layers are unchecked. Log and JSON in
``data/summary/drivers_precision_<model>.{log,json}``. Qwen3-8B only.

    python scripts/checks/drivers_precision.py [--device mps]
"""
import argparse
import copy
import importlib.util
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM

from neuron_lens import arch
from neuron_lens.catalog import MODEL, TOP, neuron_table
from neuron_lens.paths import DATA, SUMMARY
from neuron_lens.transport import decoder_stack

spec = importlib.util.spec_from_file_location("drivers", Path(__file__).resolve().parents[1] / "07_drivers.py")
drivers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drivers)

LAYERS = 10  # L0-L9
HITS = 20  # the top drivers the overlap counts (findings §6)
BATCH = 128  # tokens per pass, each in every document
KINDS = ("at", "sum")
SAVED = DATA / "checks" / "drivers-fp32-L00-L09.pt"


def overlap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per row, the share of ``a``'s tokens among ``b``'s."""
    return (a[:, :, None] == b[:, None, :]).any(-1).float().mean(-1)


def sweep(body, docs: torch.Tensor, order: torch.Tensor, dev) -> dict:
    """07's neuron pass in fp32 through L9: ``{kind: (ids, values)}``, each ``[LAYERS, 2f, TOP]``,
    the + poles first, values times the pole's sign (as the catalog stores them)."""
    D, f = len(docs), body.config.intermediate_size
    best = {k: torch.full((LAYERS, 2, f, TOP), -torch.inf, device=dev) for k in KINDS}
    best_ids = {k: torch.zeros((LAYERS, 2, f, TOP), dtype=torch.long, device=dev) for k in KINDS}
    cur = {}
    t0 = time.time()
    with torch.no_grad():
        prefix = body(input_ids=docs[:, :drivers.AT], use_cache=True).past_key_values
        for l in range(LAYERS):  # hooked after the prefix pass: from here on every pass starts at the token
            body.layers[l].mlp.down_proj.register_forward_pre_hook(
                lambda m, x, l=l: cur.__setitem__(l, drivers.pole_drivers(x[0].unflatten(0, (D, -1)))))
        for i in range(0, len(order), BATCH):
            ids = order[i:i + BATCH].to(dev)
            batch = docs[:, None, drivers.AT:].repeat(1, len(ids), 1)  # [D, B, 1 + AFTER]
            batch[:, :, 0] = ids
            cache = copy.deepcopy(prefix)
            cache.batch_repeat_interleave(len(ids))  # document-major, as the batch is flattened
            body(input_ids=batch.flatten(0, 1), past_key_values=cache, use_cache=True)
            del cache
            for l in range(LAYERS):
                for k, v in cur[l].items():  # [B, f]
                    for p, sign in enumerate((1, -1)):
                        vals, j = torch.cat([best[k][l, p], sign * v.T], 1).topk(TOP, dim=1)
                        best_ids[k][l, p] = torch.cat([best_ids[k][l, p], ids.expand(f, -1)], 1).gather(1, j)
                        best[k][l, p] = vals
            cur.clear()
            if (i // BATCH) % 20 == 0:
                print(f"{time.strftime('%H:%M:%S')} {i + len(ids)}/{len(order)} tokens, {time.time() - t0:.0f}s",
                      flush=True)
    return {k: (best_ids[k].flatten(1, 2).cpu(), best[k].flatten(1, 2).cpu()) for k in KINDS}


def matrix(t, name: str) -> torch.Tensor:
    """A fixed-size list column as ``[rows, TOP]``."""
    return torch.tensor(t.column(name).combine_chunks().flatten().to_numpy()).view(t.num_rows, -1)


def compare(fp32: dict) -> tuple[list[dict], list[str]]:
    rows, lines = [], []
    for l in range(LAYERS):
        t = pq.read_table(neuron_table("drivers", l), columns=["drivers_ids", "drivers_val", "drivers_sum_ids"])
        old = {"at": matrix(t, "drivers_ids").long(), "sum": matrix(t, "drivers_sum_ids").long()}
        old_val = matrix(t, "drivers_val")
        new = {k: fp32[k][0][l] for k in KINDS}
        new_val = fp32["at"][1][l]
        strongest = new_val[:, 0]
        resp = strongest >= strongest.median()
        row = {"layer": l}
        for k in KINDS:
            o = overlap(new[k][:, :HITS], old[k][:, :HITS])
            row[f"top20_{k}"] = (o.mean().item(), o[resp].mean().item(), o.median().item())
            row[f"top64_{k}"] = overlap(new[k], old[k]).mean().item()
        # the drivers both keep at the token: each fp32 value against the bf16 one of the same token
        match = new["at"][:, :, None] == old["at"][:, None, :]  # [2f, fp32 rank, bf16 rank]
        both = match.any(2)
        x, y = new_val[both], (match.float() * old_val[:, None, :]).sum(2)[both]
        row["shared_r"] = torch.corrcoef(torch.stack([x, y]))[0, 1].item()
        row["shared_reldiff"] = ((x - y).abs() / x.abs().clamp_min(1e-6)).median().item()
        rows.append(row)
        lines.append(f"L{l:2d}  top20 {row['top20_at'][0]:.3f}/{row['top20_sum'][0]:.3f} "
                     f"(resp {row['top20_at'][1]:.3f}/{row['top20_sum'][1]:.3f})  "
                     f"top64 {row['top64_at']:.3f}/{row['top64_sum']:.3f}  "
                     f"shared r {row['shared_r']:.3f} reldiff {row['shared_reldiff']:.3f}")
        print(lines[-1], flush=True)
    return rows, lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
    args = ap.parse_args()
    if MODEL != arch.QWEN:
        raise SystemExit(f"drivers_precision.py checks Qwen3-8B's drivers only, not {MODEL}")
    order = drivers.token_order()
    t0 = time.time()
    if SAVED.exists():
        x = torch.load(SAVED, weights_only=False)
        fp32, run = {k: (x["ids"][k], x["val"][k]) for k in KINDS}, f"fp32 drivers from {SAVED.name} ({x['device']})"
    else:
        dev = torch.device(args.device)
        lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager")
        body = decoder_stack(lm)
        body.layers = body.layers[:LAYERS]
        body.to(dev).eval().requires_grad_(False)
        fp32 = sweep(body, drivers.documents(dev), order, dev)
        SAVED.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ids": {k: v[0] for k, v in fp32.items()}, "val": {k: v[1] for k, v in fp32.items()},
                    "model": MODEL, "dtype": "float32", "device": args.device}, SAVED)
        run = f"fp32 drivers computed on {args.device} in {time.time() - t0:.0f}s"
    head = [(f"exact drivers in fp32 against the catalog's bf16 ones (07_drivers.py), {MODEL}, L0-L{LAYERS - 1}: "
             f"{len(order)} kept tokens at position {drivers.AT} of 8 documents; {run}"),
            (f"per layer: mean top-{HITS} overlap at the token / summed over it and the {drivers.AFTER} after "
             "(all poles; responsive half); mean top-64 overlap; drivers both keep at the token: "
             "Pearson r and median relative difference of their activations"), ""]
    print("\n".join(head), flush=True)
    rows, lines = compare(fp32)
    lo = {k: min(r[f"top20_{k}"][0] for r in rows) for k in KINDS}
    hi = {k: max(r[f"top20_{k}"][0] for r in rows) for k in KINDS}
    deep = min(r[f"top20_{k}"][0] for r in rows if r["layer"] >= 7 for k in KINDS)
    tail = ["",
            (f"L0-L{LAYERS - 1}: mean top-{HITS} overlap {lo['at']:.3f}-{hi['at']:.3f} at the token, "
             f"{lo['sum']:.3f}-{hi['sum']:.3f} summed"),
            (f"verdict: the bf16 drivers {'stand' if deep >= 0.95 else 'do NOT stand'} "
             f"(lowest mean top-{HITS} overlap at L7-L9 {deep:.3f}; the bar is 0.95)")]
    print("\n".join(tail), flush=True)
    stem = arch.stem(MODEL)
    (SUMMARY / f"drivers_precision_{stem}.log").write_text("\n".join(head + lines + tail) + "\n")
    (SUMMARY / f"drivers_precision_{stem}.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
