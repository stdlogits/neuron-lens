"""Benchmark and smoke-test the lens estimators on one document, before a long fit.

For each model: the time per document of the R-lens and the S-lens (both targets) at each
``--dim-batch``, after a warm-up, with peak memory and the matrix-product throughput that time
implies. With ``--compare-fp32``, also the bf16 estimates' error against fp32. Run it first on
a new machine:

    python scripts/checks/fit_bench.py --device cuda --models Qwen/Qwen3-0.6B,Qwen/Qwen3-8B
        [--dim-batch 128,256,512] [--compare-fp32]
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_lens.corpus import lens_docs
from neuron_lens.paths import CORPUS_DIR
from neuron_lens.transport import (
    SKIP_FIRST,
    TARGETS,
    Accumulator,
    decoder_stack,
    r_lens_doc,
    s_lens_doc,
    target_layers,
)


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def load(model_id: str, dtype, device: str):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    return decoder_stack(model).to(device).eval().requires_grad_(False)


def flops(model, T: int, lens: str) -> float:
    """Matrix-product FLOPs of one document: per layer passed, ``d_model`` directions times
    the valid positions times the projections, plus attention over the positions kept (per
    layer, so models whose layers differ in head shapes, as Gemma 4's, count right)."""
    d, n = model.config.hidden_size, T - 1 - SKIP_FIRST
    total = 0
    for layer in model.layers[:len(model.layers) - (lens == "R")]:
        at, mlp = layer.self_attn, layer.mlp
        proj = sum(m.weight.numel() for m in (at.q_proj, at.k_proj, at.v_proj, at.o_proj,
                                               mlp.gate_proj, mlp.up_proj, mlp.down_proj) if m is not None)
        attn = 4 * at.q_proj.weight.shape[0] * (T - 1)  # scores and values, both terms, over the keys kept
        total += 2 * d * n * (proj + attn)
    return total


def fit_one(lens: str, model, ids, device: str, dim_batch: int):
    n, d = len(model.layers), model.config.hidden_size
    acc = Accumulator({t: target_layers(lens, t, n) for t in TARGETS[lens]}, d, device=device)
    sync(device)
    t0 = time.perf_counter()
    (r_lens_doc if lens == "R" else s_lens_doc)(model, ids, acc, 0, dim_batch=dim_batch)
    sync(device)
    return acc, time.perf_counter() - t0


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dim-batch", default="256")
    ap.add_argument("--lenses", default="R,S")
    ap.add_argument("--compare-fp32", action="store_true")
    args = ap.parse_args()
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    print(f"device {args.device}, {dtype}", flush=True)
    for model_id in args.models.split(","):
        tok = AutoTokenizer.from_pretrained(model_id)
        row = lens_docs(CORPUS_DIR / "dclm.jsonl", tok, 1, 128, SKIP_FIRST + 2)[0]
        ids = torch.tensor(row)
        model = load(model_id, dtype, args.device)
        print(f"one document of {len(row)} tokens", flush=True)
        print(f"\n{model_id}: {len(model.layers)} layers, d {model.config.hidden_size}", flush=True)
        est = {}
        for lens in args.lenses.split(","):
            for b in (int(x) for x in args.dim_batch.split(",")):
                fit_one(lens, model, ids, args.device, b)  # warm-up
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                est[lens], t = fit_one(lens, model, ids, args.device, b)
                mem = f", peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" \
                    if args.device == "cuda" else ""
                print(f"  {lens}-lens, dim_batch {b}: {t:.2f} s per document, "
                      f"{flops(model, len(row), lens) / t / 1e12:.0f} TFLOP/s{mem}", flush=True)
        if args.compare_fp32 and dtype != torch.float32:
            del model
            model = load(model_id, torch.float32, args.device)
            for lens, acc in est.items():
                ref, _ = fit_one(lens, model, ids, args.device, int(args.dim_batch.split(",")[0]))
                for target in sorted({t for t, _ in ref.sums}):
                    errs = [rel(acc.mean(target)[l], ref.mean(target)[l])
                            for l in range(ref.sums[(target, 0)].shape[0])]
                    print(f"  {lens}-lens {target} bf16 vs fp32 rel err: median "
                          f"{sorted(errs)[len(errs) // 2]:.1e}, max {max(errs):.1e}", flush=True)
        del model, est
        if args.device == "cuda":
            torch.cuda.empty_cache()
        elif args.device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
