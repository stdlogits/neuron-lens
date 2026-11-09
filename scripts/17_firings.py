"""What the model holds where each neuron pole fires (pipeline stage 17; findings §18).

Each read in ``reads`` comes from weights alone. These come from activations: per pole, the mean over its
top firings (``acts``' ``top_doc``, ``top_pos``, where ``top_val > 0``) of the residual at those positions,
read through the lenses. ``neurons/firings/L<layer>.parquet``, a row per pole:

- ``before_R``, ``before_S``: the residual entering the layer's MLP, through ``mid``: what the pole's
  input holds where it fires;
- ``mlp_R``, ``mlp_S``: the layer's whole MLP output there, through ``out``: what the layer writes when the
  pole fires, the pole and the neurons that fire with it;
- ``after_R``, ``after_S``: the residual after the layer, through ``out``;
- ``pred``: the model's own mean logit there, its prediction of what comes next.

Each list as in ``reads`` (``_ids``, ``_val``, ``_str``, the top ``TOP``), with a ``_z`` variant: each token's
score z-scored across the layer's poles (for ``pred``, the logit). ``n_firings`` counts a pole's firings.
``pred`` is the mean logit, with the size it has (Gemma 4: the softcap on the mean, not on each firing's logits);
the other reads are directions (scale-free, as every lens read).

**On the cards.** These average over every top firing, so they must stay off the label evidence cards,
which hold firings past ``N_CONTEXTS`` out for scoring (AGENTS.md).

Runs from the ``acts`` tables and documents, in bf16 (as ``06_activations.py``), saving every
``--save-minutes`` and resuming there.

    python scripts/17_firings.py [--detach] [--layers a-b] [--docs N] [--batch-docs 32]

``--docs N`` is a smoke test: only the first N documents, printing what it collected, writing no table.
"""
import argparse
import os
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import MODEL, TOP, column, documents, kept, neuron_keys, neuron_table, put, write
from neuron_lens.firings import Accumulator, firing_plan
from neuron_lens.neurons import softcap
from neuron_lens.paths import DATA, LOGS, ROOT
from neuron_lens.readout import layer_reads, s_tops
from neuron_lens.weights import tensors

R_FIT, S_FIT = arch.fits(MODEL)
STEM = arch.stem(MODEL)
STATE = DATA / "acts" / f"{STEM}-firings.pt"
CAP = softcap(MODEL)


def layer_plan(layer: int) -> tuple:
    t = pq.read_table(neuron_table("acts", layer), columns=["top_doc", "top_pos", "top_val"])
    return firing_plan(column(t, "top_doc").long(), column(t, "top_pos").long(), column(t, "top_val"))


def collect(args, layers: list[int], docs: list[torch.Tensor]) -> Accumulator:
    lm, stack = arch.load_model(MODEL, args.device, eager=False)
    cfg = stack.config
    plans = {l: layer_plan(l) for l in layers}
    acc = Accumulator(plans, cfg.intermediate_size, cfg.hidden_size, args.device)
    meta = {"model": MODEL, "layers": layers, "top": TOP, "docs": len(docs)}
    next_doc = 0
    if STATE.exists() and args.docs is None:
        saved = torch.load(STATE, weights_only=False, map_location="cpu")
        if saved["meta"] != meta:
            raise SystemExit(f"{STATE} was collected with {saved['meta']}, not {meta}: move it to start afresh")
        acc.load(saved["state"])
        next_doc = saved["next_doc"]
    hooks = acc.hooks(stack)
    batches = list(range(next_doc, len(docs), args.batch_docs))
    saved_at, t0, tokens = time.monotonic(), time.perf_counter(), 0
    for i, a in enumerate(batches):
        b = min(a + args.batch_docs, len(docs))
        T = max(len(d) for d in docs[a:b])
        ids, mask = torch.zeros(b - a, T, dtype=torch.long), torch.zeros(b - a, T, dtype=torch.long)
        for j, d in enumerate(docs[a:b]):
            ids[j, :len(d)], mask[j, :len(d)] = d.long(), 1
        acc.begin(a, b)
        with torch.inference_mode():
            stack(input_ids=ids.to(args.device), attention_mask=mask.to(args.device))
        tokens += int(mask.sum())
        last = i + 1 == len(batches)
        if args.docs is None and (last or time.monotonic() - saved_at > 60 * args.save_minutes):
            STATE.parent.mkdir(parents=True, exist_ok=True)
            tmp = STATE.with_suffix(".tmp")
            torch.save({"meta": meta, "state": acc.state(), "next_doc": b}, tmp)
            os.replace(tmp, STATE)
            saved_at = time.monotonic()
        if i % 20 == 0 or last:
            print(f"{time.strftime('%H:%M:%S')} batch {i + 1}/{len(batches)}, {tokens / (time.perf_counter() - t0):.0f} tokens/s", flush=True)
    for h in hooks:
        h.remove()
    del lm, stack
    return acc


def tables(acc: Accumulator, layers: list[int], dev) -> None:
    keep = kept()
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted(set(nm.values())))
    unembed = head[nm["unembed"]][keep].to(dev, torch.float16)
    gamma = head[nm["norm"]].to(dev, torch.float32)
    emb = head[nm["embed"]][keep].float() * arch.embed_scale(MODEL)
    emb = (emb - emb.mean(0)).to(dev, torch.float16)
    del head
    J_out, J_mid = arch.lens(MODEL, "R_out", R_FIT), arch.lens(MODEL, "R_mid", R_FIT)
    K_mid, K_out = arch.lens(MODEL, "S_mid", S_FIT), arch.lens(MODEL, "S_out", S_FIT)
    for l in layers:
        t0 = time.perf_counter()
        m = {k: v.to(dev) for k, v in acc.means(l).items()}
        cols = neuron_keys(l, m["before"].shape[0] // 2)
        cols["n_firings"] = pa.array(acc.count[l].numpy())
        for site, J, K in (("before", J_mid[l], K_mid[l]), ("mlp", J_out[l] if l < len(J_out) else None, K_out[l]),
                           ("after", J_out[l] if l < len(J_out) else None, K_out[l])):
            r = layer_reads(m[site].T, unembed, gamma, J, signs=(1,), k=TOP, kz=TOP, softcap=CAP)
            put(cols, f"{site}_R", keep[r["ids"][1]], r["probs"][1])
            put(cols, f"{site}_R_z", keep[r["zids"][1]], r["z"][1])
            cov = m[site] @ K.to(dev)
            s = s_tops(cov / cov.norm(dim=1, keepdim=True).clamp_min(1e-9), emb, (1,), TOP)
            for kind, name in (("raw", f"{site}_S"), ("z", f"{site}_S_z")):
                put(cols, name, keep[s[kind, 1][0]], s[kind, 1][1])
        r = layer_reads(m["pred"].T, unembed, gamma, None, signs=(1,), k=TOP, kz=TOP, softcap=CAP, renorm=False)
        put(cols, "pred", keep[r["ids"][1]], r["probs"][1])
        put(cols, "pred_z", keep[r["zids"][1]], r["z"][1])
        write(cols, neuron_table("firings", l))
        print(f"{time.strftime('%H:%M:%S')} firings L{l} in {time.perf_counter() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=None, help="a-b (default all)")
    ap.add_argument("--docs", type=int, default=None, help="a smoke test over the first N documents, writing no table")
    ap.add_argument("--batch-docs", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    ap.add_argument("--save-minutes", type=float, default=30)
    ap.add_argument("--detach", action="store_true")
    args = ap.parse_args()
    if args.detach:
        LOGS.mkdir(exist_ok=True)
        log = LOGS / f"firings_{STEM}.log"
        cmd = [sys.executable, __file__, *[a for a in sys.argv[1:] if a != "--detach"]]
        if sys.platform == "darwin":
            cmd = ["caffeinate", "-i", *cmd]
        with open(log, "a") as out:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                                 env={**os.environ, "PYTHONUNBUFFERED": "1"})
        print(f"detached: pid {p.pid}, log {log}")
        return
    n = arch.text_config(MODEL).num_hidden_layers
    lo, hi = (0, n - 1) if args.layers is None else map(int, args.layers.split("-"))
    layers = list(range(lo, hi + 1))
    docs = documents()
    if args.docs is not None:
        docs = docs[:args.docs]
    acc = collect(args, layers, docs)
    if args.docs is not None:
        for l in layers:
            print(f"L{l}: {int((acc.count[l] > 0).sum())} poles with a firing in {args.docs} documents, {int(acc.count[l].sum())} firings")
        print("smoke test done: no table written", flush=True)
        return
    tables(acc, layers, args.device)
    STATE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
