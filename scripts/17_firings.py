"""What each neuron pole's top firings hold, and what drives them (pipeline stage 17; findings §17, §18).

One forward pass over the activation documents (fp32, with TF32 matrix products on CUDA: 3.6 times the speed of
fp32 products, and top-64 lists 99.9% the same), after ``06_activations.py`` has found the firings, fills two
tables, a row per pole.

``neurons/firings/L<layer>.parquet``. Each read in ``reads`` comes from weights alone. These come from
activations: per pole, the mean over its top firings (``acts``' ``top_doc``, ``top_pos``, where ``top_val > 0``) of
the residual at those positions, read through the lenses:

- ``before_R``, ``before_S``: the residual entering the layer's MLP, through ``mid``: what the pole's
  input holds where it fires;
- ``mlp_R``, ``mlp_S``: the layer's whole MLP output there, through ``out``: what the layer writes when the
  pole fires, the pole and the neurons that fire with it;
- ``after_R``, ``after_S``: the residual after the layer, through ``out``;
- ``pred``: the model's own mean logit there, its prediction of what comes next.

Each list as in ``reads`` (``_ids``, ``_val``, ``_str``, the top ``TOP``), with a ``_z`` variant: each token's
score z-scored across the layer's poles that fired (for ``pred``, the logit). ``n_firings`` counts a pole's firings;
a pole without any has ``−1``, NaN and empty previews.

``parallel``, ``orthogonal``: the pole's own write against what it reads, mean over its firings. ``parallel`` is how much
its write amplifies (positive) or suppresses (negative) the part of the residual it reads, as a multiple of that part;
``orthogonal`` is how much it adds off that direction, on the same scale (``neuron_lens.firings``).
``pred`` is the mean logit, with the size it has (Gemma 4: the softcap on the mean, not on each firing's logits);
the other reads are directions (scale-free, as every lens read).

**On the cards.** These average over every top firing, so they must stay off the label evidence cards,
which hold firings past ``N_CONTEXTS`` out for scoring (AGENTS.md).

``neurons/sources/L<layer>.parquet``, from L1, at each pole's first ``N_CONTEXTS`` top firings (the firings the
cards show, so these can sit on them):

- **driven by** (``driven_*``): the exact split of each firing's activation over the embedding, the heads
  and earlier neurons (``neuron_lens.sources``), each upstream neuron in the pole its activation's sign gives.
  ``driven_layer``, ``driven_neuron``, ``driven_pole``, ``driven_share``: the ``TOP`` largest |mean share over
  the firings|; ``driven_head_layer``, ``driven_head``, ``driven_head_share``: the same of the heads;
  ``emb_share``, ``heads_share``, ``neurons_share``: each kind of source's mean share (they sum to 1);
- **fires with** (``with_*``): the earlier poles active at ``act_top / 4`` or more at the pole's firings,
  by ``with_lift``, the share of the firings where they are active over their overall ``rate``; ``with_co``, that share.

``n_contexts``: the firings used. Each list is built from ``RANK`` candidates a firing (``RANK_WITH`` for
fires-with), a source outside them counting 0 at that firing; ``with_*`` needs a source active at two of the pole's
firings (one, if the pole has only one), and its ``rate`` is at least one position in the corpus. Per layer in
``data/summary/sources_<model>.json``: each kind's share and the decomposition's largest error.
``checks/relations.py`` compares the three relations (driven by, wired to, fires with).

``heads/sources.parquet``, a row per head, at each head's first ``N_CONTEXTS`` top firings (``--parts heads``, the documents
holding one, each alone): what drives it through each input (``neuron_lens.sources``), per channel ``c`` of ``v`` (value, what it
copies), ``q`` and ``k`` (query and key, why it looks there): ``<c>_emb_share``, ``<c>_heads_share``, ``<c>_neurons_share``,
which sum to 1; the lists ``<c>_driven_head_layer``, ``<c>_driven_head``, ``<c>_driven_head_share`` and ``<c>_driven_layer``,
``<c>_driven_neuron``, ``<c>_driven_pole``, ``<c>_driven_share``; ``with_*``, the poles firing at the query's position, as above.
Per layer in ``data/summary/head_sources_<model>.json``.

The neurons' pass saves every ``--save-minutes`` and resumes there; the checkpoints go with the tables.

    python scripts/17_firings.py [--detach] [--layers a-b] [--docs N] [--batch-docs 32] [--parts neurons,heads]

``--docs N`` is a smoke test: only the first N documents, printing what it collected, writing no table.
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM

from neuron_lens import arch
from neuron_lens.catalog import MODEL, TOP, column, documents, head_table, kept, neuron_keys, neuron_table, put, write
from neuron_lens.firings import Accumulator, firing_plan
from neuron_lens.neurons import softcap
from neuron_lens.paths import DATA, LOGS, ROOT, SUMMARY
from neuron_lens.readout import layer_reads, s_tops
from neuron_lens.sources import HeadSources, Sources
from neuron_lens.transport import decoder_stack
from neuron_lens.weights import tensors

R_FIT, S_FIT = arch.fits(MODEL)
STEM = arch.stem(MODEL)
STATES = {name: DATA / "acts" / f"{STEM}-{name}.pt" for name in ("firings", "sources")}
CAP = softcap(MODEL)


def layer_plan(layer: int) -> tuple:
    t = pq.read_table(neuron_table("acts", layer), columns=["top_doc", "top_pos", "top_val"])
    return firing_plan(column(t, "top_doc").long(), column(t, "top_pos").long(), column(t, "top_val"))


def save(name: str, meta: dict, state: dict, next_doc: int) -> None:
    STATES[name].parent.mkdir(parents=True, exist_ok=True)
    tmp = STATES[name].with_suffix(".tmp")
    torch.save({"meta": meta, "state": state, "next_doc": next_doc}, tmp)
    os.replace(tmp, STATES[name])


def collect(args, layers: list[int], docs: list[torch.Tensor]) -> tuple[Accumulator, Sources]:
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(args.device).eval().requires_grad_(False)
    stack = decoder_stack(lm)
    cfg = stack.config
    acc = Accumulator({l: layer_plan(l) for l in layers}, cfg.intermediate_size, cfg.hidden_size, args.device)
    src = Sources(stack, [l for l in layers if l > 0], args.device, sum(len(d) for d in docs))
    meta = {"model": MODEL, "layers": layers, "top": TOP, "docs": len(docs)}
    next_doc = 0
    if args.docs is None and all(p.exists() for p in STATES.values()):  # saved one after the other: a save cut short between them can't resume
        saved = {name: torch.load(p, weights_only=False, map_location="cpu") for name, p in STATES.items()}
        if any(s["meta"] != meta for s in saved.values()) or saved["firings"]["next_doc"] != saved["sources"]["next_doc"]:
            raise SystemExit(f"{[str(p) for p in STATES.values()]} don't resume this pass: move them to start afresh")
        acc.load(saved["firings"]["state"])
        src.load(saved["sources"]["state"])
        next_doc = saved["firings"]["next_doc"]
        del saved
    hooks = acc.hooks(stack) + src.hooks()
    batches = list(range(next_doc, len(docs), args.batch_docs))
    saved_at, t0, tokens = time.monotonic(), time.perf_counter(), 0
    for i, a in enumerate(batches):
        b = min(a + args.batch_docs, len(docs))
        T = max(len(d) for d in docs[a:b])
        ids, mask = torch.zeros(b - a, T, dtype=torch.long), torch.zeros(b - a, T, dtype=torch.long)
        for j, d in enumerate(docs[a:b]):
            ids[j, :len(d)], mask[j, :len(d)] = d.long(), 1
        acc.begin(a, b)
        src.locate(a, b, T)
        with torch.inference_mode():
            stack(input_ids=ids.to(args.device), attention_mask=mask.to(args.device))
        src.split()
        tokens += int(mask.sum())
        last = i + 1 == len(batches)
        if args.docs is None and (last or time.monotonic() - saved_at > 60 * args.save_minutes):
            save("firings", meta, acc.state(), b)
            save("sources", meta, src.state(), b)
            saved_at = time.monotonic()
        if i % 20 == 0 or last:
            print(f"{time.strftime('%H:%M:%S')} batch {i + 1}/{len(batches)}, {tokens / (time.perf_counter() - t0):.0f} tokens/s", flush=True)
    for h in hooks:
        h.remove()
    src.stack = src.flow = None
    del lm, stack
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return acc, src


def widen(x: torch.Tensor, fired: torch.Tensor, pad) -> torch.Tensor:
    """``[n, k]`` rows of the poles that fired as ``[2f, k]``, the others padded."""
    out = torch.full((len(fired), x.shape[1]), pad, dtype=x.dtype)
    out[fired] = x.cpu()
    return out


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
        fired = acc.count[l] > 0
        m = {k: v[fired].to(dev) for k, v in acc.means(l).items()}
        cols = neuron_keys(l, len(fired) // 2)
        cols["n_firings"] = pa.array(acc.count[l].to(torch.int16).numpy())
        gains = acc.gains(l)
        cols["parallel"], cols["orthogonal"] = pa.array(gains[:, 0].numpy()), pa.array(gains[:, 1].numpy())
        add = lambda name, ids, val: put(cols, name, widen(keep[ids], fired, -1), widen(val, fired, float("nan")))
        for site, J, K in (("before", J_mid[l], K_mid[l]), ("mlp", J_out[l] if l < len(J_out) else None, K_out[l]),
                           ("after", J_out[l] if l < len(J_out) else None, K_out[l])):
            r = layer_reads(m[site].T, unembed, gamma, J, signs=(1,), k=TOP, kz=TOP, softcap=CAP)
            add(f"{site}_R", r["ids"][1], r["probs"][1])
            add(f"{site}_R_z", r["zids"][1], r["z"][1])
            cov = m[site] @ K.to(dev)
            s = s_tops(cov / cov.norm(dim=1, keepdim=True).clamp_min(1e-9), emb, (1,), TOP)
            for kind, name in (("raw", f"{site}_S"), ("z", f"{site}_S_z")):
                add(name, s[kind, 1][0], s[kind, 1][1])
        r = layer_reads(m["pred"].T, unembed, gamma, None, signs=(1,), k=TOP, kz=TOP, softcap=CAP, renorm=False)
        add("pred", r["ids"][1], r["probs"][1])
        add("pred_z", r["zids"][1], r["z"][1])
        write(cols, neuron_table("firings", l))
        print(f"{time.strftime('%H:%M:%S')} firings L{l} in {time.perf_counter() - t0:.0f}s", flush=True)


def source_tables(src: Sources, smoke: bool) -> None:
    summary = {}
    for M in src.layers:
        d = src.dense[M]
        n = d["n"].sum().clamp_min(1)
        summary[M] = {"firings": int(d["n"].sum()), "err": max(d["err"]) if d["err"] else float("nan"), "emb": float(d["emb"].sum() / n),
                      "neurons": float(d["neurons"].sum() / n), "heads": float(d["heads"].sum() / n)}
        print(f"L{M}: {summary[M]}", flush=True)
        if not smoke:
            write(src.table(M), neuron_table("sources", M))
    if smoke:
        src.table(src.layers[-1])  # the merge, on the partial data
    else:
        (SUMMARY / f"sources_{STEM}.json").write_text(json.dumps(summary))


def head_sources(args, docs: list[torch.Tensor]) -> None:
    """``heads/sources.parquet``: what drives each head when it fires, by channel."""
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(args.device).eval().requires_grad_(False)
    src = HeadSources(decoder_stack(lm), args.device, sum(len(d) for d in docs))
    t0 = time.perf_counter()
    src.run(docs)
    print(f"{time.strftime('%H:%M:%S')} heads: {int(src.count.sum())} firings in {time.perf_counter() - t0:.0f}s", flush=True)
    summary = src.summary()
    for M in sorted(summary)[::max(1, len(summary) // 8)]:
        print(f"L{M}: {summary[M]}", flush=True)
    worst = {c: max(s[c]["err"] for s in summary.values()) for c in ("v", "q", "k")}
    print("largest error per channel:", worst, flush=True)
    if args.docs is None:
        write(src.table(), head_table("sources"))
        (SUMMARY / f"head_sources_{STEM}.json").write_text(json.dumps(summary))
    else:
        src.table()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=None, help="a-b (default all)")
    ap.add_argument("--docs", type=int, default=None, help="a smoke test over the first N documents, writing no table")
    ap.add_argument("--batch-docs", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    ap.add_argument("--save-minutes", type=float, default=30)
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("--parts", default="neurons,heads", help="neurons (firings and sources), heads (heads/sources)")
    args = ap.parse_args()
    parts = args.parts.split(",")
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
    if "neurons" in parts:
        acc, src = collect(args, layers, docs)
        if args.docs is not None:
            for l in layers:
                print(f"L{l}: {int((acc.count[l] > 0).sum())} poles with a firing in {args.docs} documents, {int(acc.count[l].sum())} firings")
            source_tables(src, smoke=True)
        else:
            tables(acc, layers, args.device)
            source_tables(src, smoke=False)
            for p in STATES.values():
                p.unlink(missing_ok=True)
    if "heads" in parts:
        head_sources(args, docs)
    if args.docs is not None:
        print("smoke test done: no table written", flush=True)


if __name__ == "__main__":
    main()
