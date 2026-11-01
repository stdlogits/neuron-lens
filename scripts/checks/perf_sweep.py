"""Performance sweep of the lens estimators on one GPU, before a long fit: time one full-length
document per configuration, with peak memory, throughput and each configuration's estimates
against a reference run. The model loads once.

Axes (comma lists; every combination runs, per lens in ``--lenses``): ``--dim-batch``, ``--tf32``
(fp32 matrix products in TF32: DeltaNet's ``T × T`` terms and the attention scores), ``--compile``
(``transport._step``'s torch.compile on CUDA) and ``--acc-device`` (where the sums live). Each configuration runs a warm-up document, then
``--repeat`` timed ones. Its error is ``‖K − K_ref‖ / ‖K_ref‖`` per layer and target (median, max)
against the same lens with TF32 off and no compile (the most precise run). An
out-of-memory configuration is recorded and skipped. Results go to
``data/summary/perf_<model>.json`` after every configuration, so a stopped sweep keeps them.

``--profile`` also prints, per configuration, the CUDA kernels that took the most time in the
timed document (torch.profiler), to show what a code change should target.

``--parts`` times each component's map and transpose instead (softmax attention, Gated DeltaNet,
the MLP; compiled, one dim batch, on the first layer of each kind) and scales each to a document:
which component a code change should target.

``--fp32-layers N`` checks precision instead: the first N layers load twice, in fp32 (the
reference) and bf16, and each configuration's bf16 estimates are compared with fp32's on that
slice: what bf16 weights and the precision settings cost together.

    python scripts/checks/perf_sweep.py --model Qwen/Qwen3.5-35B-A3B --seq-len 512 \\
        --dim-batch 16,32,64,128 --tf32 0,1 --compile 0,1
    python scripts/checks/perf_sweep.py --model Qwen/Qwen3.5-35B-A3B --seq-len 512 --parts --dim-batch 32
    python scripts/checks/perf_sweep.py --model Qwen/Qwen3.5-35B-A3B --seq-len 512 \\
        --fp32-layers 4 --dim-batch 32 --tf32 0,1 --compile 0,1
"""
import argparse
import gc
import itertools
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from fit_bench import flops  # noqa: E402

from neuron_lens import layers, transport  # noqa: E402
from neuron_lens.corpus import lens_docs  # noqa: E402
from neuron_lens.paths import CORPUS_DIR, DATA  # noqa: E402
from neuron_lens.transport import SKIP_FIRST, TARGETS, Accumulator, decoder_stack, target_layers  # noqa: E402

_STEP = transport._step


def load(model_id: str, dtype, layers: int | None):
    config = AutoConfig.from_pretrained(model_id)
    if layers is not None:
        tc = config.get_text_config()
        tc.num_hidden_layers = layers
        tc.layer_types = tc.layer_types[:layers]
    lm = AutoModelForCausalLM.from_pretrained(model_id, config=config, dtype=dtype)
    return decoder_stack(lm).to("cuda").eval().requires_grad_(False)


def configure(tf32: bool, compiled: bool) -> None:
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    transport._step = _STEP if compiled else (lambda step, device: step)


def run(model, ids, lens: str, dim_batch: int, acc_device: str) -> Accumulator:
    n, d = len(model.layers), model.config.hidden_size
    acc = Accumulator({t: target_layers(lens, t, n) for t in TARGETS[lens]}, d, device=acc_device)
    (transport.r_lens_doc if lens == "R" else transport.s_lens_doc)(model, ids, acc, 0, dim_batch=dim_batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return acc


def errors(acc: Accumulator, ref: Accumulator) -> dict:
    out = {}
    for t in sorted({t for t, _ in ref.sums}):
        a, b = acc.mean(t).cpu(), ref.mean(t).cpu()
        e = torch.tensor([((a[l] - b[l]).norm() / b[l].norm()).item() for l in range(b.shape[0])])
        out[t] = {"median": e.median().item(), "max": e.max().item()}
    return out


def parts(model, ids, dim_batch: int, repeat: int = 5) -> None:
    """Milliseconds per call of each component's map and transpose at ``dim_batch``, compiled,
    on the first layer of each kind, and seconds per document (every layer of the kind, every
    block of directions); ``layer`` is the whole layer step (``layers.jvp``, ``layers.vjp``)."""
    maps = layers.linearize(model, ids)
    N, d = maps[0].n1.a.shape
    dt = layers._dtype(maps[0])
    blocks = -(-d // dim_batch)
    fns = {"Attention": (layers._attn_jvp, layers._attn_vjp), "DeltaNet": (layers._delta_jvp, layers._delta_vjp),
           "MoE": (layers._moe_jvp, layers._moe_vjp), "GatedMLP": (layers._mlp_jvp, layers._mlp_vjp)}
    X = torch.randn(dim_batch, N, d, device="cuda")

    def time_ms(f, *a):
        f(*a)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            f(*a)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeat

    total = {"jvp": 0.0, "vjp": 0.0}
    for kind in ("mix", "mlp", "layer"):
        seen = {}
        for m in maps:
            part = m if kind == "layer" else getattr(m, kind)
            name = "layer:" + type(m.mix).__name__ if kind == "layer" else type(part).__name__
            seen.setdefault(name, [part, 0])[1] += 1
        for name, (part, count) in seen.items():
            jvp, vjp = (layers.jvp, layers.vjp) if kind == "layer" else fns[name]
            row = []
            for direction, f in (("jvp", jvp), ("vjp", vjp)):
                ms = time_ms(torch.compile(f, dynamic=False), part, X if f in (layers.jvp, layers.vjp) or direction == "vjp" else X.to(dt))
                per_doc = ms * blocks * count / 1e3
                if kind == "layer":
                    total[direction] += per_doc
                row.append(f"{direction} {ms:7.1f} ms/call, {per_doc:6.1f} s/doc")
            print(f"{name:>15} ×{count:<3} " + "   ".join(row), flush=True)
    print(f"all layers: jvp {total['jvp']:.1f} s/doc, vjp {total['vjp']:.1f} s/doc", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lenses", default="R,S")
    ap.add_argument("--dim-batch", default="32")
    ap.add_argument("--tf32", default="0,1")
    ap.add_argument("--compile", default="1")
    ap.add_argument("--acc-device", default="cuda")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--fp32-layers", type=int, default=None)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--parts", action="store_true")
    args = ap.parse_args()
    # Every dim batch, arm and layer kind is its own compiled graph; past the limit dynamo would
    # fall back to eager and the timings would mislead.
    torch._dynamo.config.recompile_limit = torch._dynamo.config.accumulated_recompile_limit = 1024
    grid = list(itertools.product(args.lenses.split(","),
                                  [int(x) for x in args.dim_batch.split(",")],
                                  [bool(int(x)) for x in args.tf32.split(",")],
                                  [bool(int(x)) for x in args.compile.split(",")], args.acc_device.split(",")))
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = torch.tensor(lens_docs(CORPUS_DIR / "dclm.jsonl", tok, 1, args.seq_len, SKIP_FIRST + 2, full=True)[0])
    t0 = time.perf_counter()
    model = load(args.model, torch.bfloat16, args.fp32_layers)
    ref_model = None if args.fp32_layers is None else load(args.model, torch.float32, args.fp32_layers)
    if args.parts:
        for tf32 in [bool(int(x)) for x in args.tf32.split(",")]:
            configure(tf32, True)
            print(f"tf32 {int(tf32)}, dim batch {args.dim_batch.split(',')[0]}:", flush=True)
            parts(model, ids, int(args.dim_batch.split(",")[0]), args.repeat * 5)
        return
    name = args.model.split("/")[-1].lower() + ("" if args.fp32_layers is None else f"-fp32check-L{args.fp32_layers}")
    out_path = DATA / "summary" / f"perf_{name}.json"
    print(f"{torch.cuda.get_device_name()}; loaded in {time.perf_counter() - t0:.0f}s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB; one document of {len(ids)} tokens; "
          f"{len(grid)} configurations", flush=True)
    refs, rows = {}, []
    for lens, b, tf32, compiled, acc_dev in grid:
        if lens not in refs:  # the most precise run of this lens
            configure(False, False)
            refs[lens] = run(ref_model or model, ids, lens, b, "cuda")
        configure(tf32, compiled)
        row = {"lens": lens, "dim_batch": b, "tf32": tf32, "compile": compiled, "acc_device": acc_dev}
        try:
            run(model, ids, lens, b, acc_dev)  # warm-up (and compile)
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(args.repeat):
                t1 = time.perf_counter()
                acc = run(model, ids, lens, b, acc_dev)
                times.append(time.perf_counter() - t1)
            if args.profile:
                from torch.profiler import ProfilerActivity, profile
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    run(model, ids, lens, b, acc_dev)
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25), flush=True)
            s = min(times)
            row |= {"s_per_doc": s, "tflops": flops(model, len(ids), lens) / s / 1e12,
                    "peak_gib": torch.cuda.max_memory_allocated() / 2**30, "error": errors(acc, refs[lens])}
            err = " ".join(f"{t} {e['median']:.1e}/{e['max']:.1e}" for t, e in row["error"].items())
            print(f"{lens} b{b:<4} tf32 {int(tf32)} compile {int(compiled)} acc {acc_dev:4}: "
                  f"{s:6.1f} s/doc, {row['tflops']:4.0f} TFLOP/s, peak {row['peak_gib']:5.1f} GiB, "
                  f"err (median/max) {err}", flush=True)
            del acc
        except torch.OutOfMemoryError:
            row["oom"] = True
            print(f"{lens} b{b:<4} tf32 {int(tf32)} compile {int(compiled)} acc {acc_dev:4}: out of memory",
                  flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        rows.append(row)
        out_path.write_text(json.dumps({"meta": vars(args) | {"tokens": len(ids)}, "rows": rows}, indent=1) + "\n")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
