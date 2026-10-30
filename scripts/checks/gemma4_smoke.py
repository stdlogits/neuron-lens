"""Smoke checks of the lens estimators on the real Gemma 4 12B, before renting GPUs.

- ``--maps``: the hand-written layer maps against autograd through HF's model under the RelP
  rules, in fp32 on the CPU, for a few basis directions at every layer: S-lens columns (forward
  mode) and R-lens rows (reverse mode), from one corpus document.
- ``--bench``: one full document per lens in bf16 on ``--device``, timed, with the FLOP count
  and the bf16 error against fp32 on the first ``--dirs`` directions.
- ``--noise``: per-document spread of both lenses on the first ``--dirs`` directions over
  ``--docs`` documents (``‖est_i − est_j‖ / ‖mean‖``, median over pairs and layers), and the
  split-half difference it predicts at 500 and 1,000 documents: ``spread · √(2 / per half) / √2``.

    python scripts/checks/gemma4_smoke.py --maps --bench --noise [--device mps]
"""
import argparse
import itertools
import json
import time

import torch
import torch.autograd.forward_ad as fwAD
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from neuron_lens.corpus import lens_docs
from neuron_lens.paths import CORPUS_DIR, DATA
from neuron_lens.rules import relp_rules
from neuron_lens.transport import (SKIP_FIRST, TARGETS, Accumulator, decoder_stack, r_lens_doc,
                                   s_lens_doc, target_layers, valid_positions)

MODEL = "google/gemma-4-12B"


LAYERS = None  # --layers: the first N decoder layers only (the 12B in fp32 is 48 GB)


def load(dtype, device):
    config = AutoConfig.from_pretrained(MODEL)
    if LAYERS is not None:
        tc = config.text_config
        tc.num_hidden_layers, tc.layer_types = LAYERS, tc.layer_types[:LAYERS]
    lm = AutoModelForCausalLM.from_pretrained(MODEL, config=config, dtype=dtype, attn_implementation="eager")
    return decoder_stack(lm).to(device).eval().requires_grad_(False)


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def flops(model, T: int, lens: str) -> float:
    d, n = model.config.hidden_size, T - 1 - SKIP_FIRST
    total = 0
    for layer in model.layers[:len(model.layers) - (lens == "R")]:
        at, mlp = layer.self_attn, layer.mlp
        proj = sum(m.weight.numel() for m in (at.q_proj, at.k_proj, at.v_proj, at.o_proj,
                                               mlp.gate_proj, mlp.up_proj, mlp.down_proj) if m is not None)
        total += 2 * d * n * (proj + 4 * at.q_proj.weight.shape[0] * (T - 1))
    return total


def hooks_on(model, store, grads=False):
    """Record each layer's MLP input (``mid``) and output, as tensors in the graph."""
    hs = []
    for i, b in enumerate(model.layers):
        hs.append(b.pre_feedforward_layernorm.register_forward_pre_hook(
            lambda _m, a, i=i: store.__setitem__(("mid", i), a[0])))
        hs.append(b.register_forward_hook(lambda _m, _a, o, i=i: store.__setitem__(("out", i), o)))
    return hs


def few_dirs(maps, dirs, lens):
    """Our estimates for basis directions ``dirs`` only: S columns ``{(kind, L): [d, k]}``,
    R rows ``{(kind, L): [k, d]}``."""
    from neuron_lens import gemma4
    N, d = maps[0].a1.shape
    X = torch.zeros(len(dirs), N, d, device=maps[0].a1.device)
    X[torch.arange(len(dirs)), :, torch.tensor(dirs)] = 1
    out = {}
    if lens == "S":
        for L, m in enumerate(maps):
            X, mid = gemma4.jvp(m, X)
            out[("mid", L)], out[("out", L)] = mid.mean(1).T, X.mean(1).T
    else:
        G = X
        for L in range(len(maps) - 1, -1, -1):
            G, mid, est = gemma4.vjp(maps[L], G)
            out[("mid", L)] = mid
            if L > 0:
                out[("out", L - 1)] = est
    return out


def check_maps(ids, dirs, device="cpu"):
    """fp32: our few-direction estimates against autograd under the RelP rules."""
    from neuron_lens import gemma4
    t0 = time.perf_counter()
    model = load(torch.float32, device)
    print(f"fp32 model on {device} in {time.perf_counter() - t0:.0f}s", flush=True)
    ids = ids.to(device)
    valid = valid_positions(len(ids), device)
    maps = gemma4.linearize(model, ids)
    n = len(model.layers)
    ours = {lens: few_dirs(maps, dirs, lens) for lens in ("S", "R")}
    del maps
    emb = model.embed_tokens(ids[None, :-1])[0]
    errs = {"S": [], "R": []}
    with relp_rules(model):
        # S: forward mode, tangent e_j at every valid position.
        for k, j in enumerate(dirs):
            store = {}
            hs = hooks_on(model, store)
            tan = torch.zeros_like(emb)
            tan[valid, j] = 1
            with fwAD.dual_level(), torch.no_grad():
                model(inputs_embeds=fwAD.make_dual(emb, tan)[None])
                ref = {key: fwAD.unpack_dual(v).tangent[0, valid].mean(0) for key, v in store.items()}
            for h in hs:
                h.remove()
            for key, v in ref.items():
                got = ours["S"][key][:, k]
                errs["S"].append((key, j, ((got - v).norm() / v.norm()).item()))
        # R: reverse mode, cotangent e_j at every valid position of the last layer's output.
        for k, j in enumerate(dirs):
            store = {}
            hs = hooks_on(model, store)
            e = emb.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                model(inputs_embeds=e[None])
                keys = [("mid", L) for L in range(n)] + [("out", L) for L in range(n - 1)]
                for key in keys:
                    store[key].retain_grad()
                store[("out", n - 1)][0, valid, j].sum().backward()
            for h in hs:
                h.remove()
            for key in keys:
                v = store[key].grad[0, valid].mean(0)
                got = ours["R"][key][k]
                errs["R"].append((key, j, ((got - v).norm() / v.norm()).item()))
            model.zero_grad(set_to_none=True)
    for lens, es in errs.items():
        worst = max(es, key=lambda x: x[2])
        print(f"maps {lens}: {len(es)} (target, layer, dir) checks, rel err median "
              f"{sorted(x[2] for x in es)[len(es) // 2]:.1e}, max {worst[2]:.1e} at {worst[:2]}", flush=True)
    del model
    return {lens: [x[2] for x in es] for lens, es in errs.items()}


def first_dirs(lens, model, ids, dirs, device):
    """Our few-direction estimates in the model's dtype, as fp32 on the CPU."""
    from neuron_lens import gemma4
    maps = gemma4.linearize(model, ids.to(device))
    out = few_dirs(maps, dirs, lens)
    return {k: v.float().cpu() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--maps-device", default="cpu", help="where the fp32 map check runs (cuda on an H100)")
    ap.add_argument("--dirs", type=int, default=256, help="directions for the bf16 error and the noise")
    ap.add_argument("--docs", type=int, default=16)
    ap.add_argument("--dim-batch", type=int, default=256)
    ap.add_argument("--layers", type=int, default=None, help="first N decoder layers only")
    args = ap.parse_args()
    global LAYERS
    LAYERS = args.layers
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = lens_docs(CORPUS_DIR / "dclm.jsonl", tok, 1000, 128, SKIP_FIRST + 2)
    rows = [r for r in rows if len(r) == 128]
    ids = torch.tensor(rows[0])
    report = {"model": MODEL, "doc_tokens": len(rows[0]), "layers": args.layers}
    torch.manual_seed(0)
    if args.maps:
        dirs = torch.randperm(3840)[:3].tolist()
        report["maps_rel_err"] = check_maps(ids, dirs, args.maps_device)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if args.bench or args.noise:
        t0 = time.perf_counter()
        model = load(torch.bfloat16, args.device)
        print(f"bf16 model on {args.device} in {time.perf_counter() - t0:.0f}s", flush=True)
    if args.bench:
        n, d = len(model.layers), model.config.hidden_size
        for lens in ("S", "R"):
            acc = Accumulator({t: target_layers(lens, t, n) for t in TARGETS[lens]}, d, device=args.device)
            sync(args.device)
            t0 = time.perf_counter()
            (s_lens_doc if lens == "S" else r_lens_doc)(model, ids, acc, 0, dim_batch=args.dim_batch)
            sync(args.device)
            t = time.perf_counter() - t0
            f = flops(model, len(ids), lens)
            print(f"bench {lens}: {t:.0f}s per document on {args.device}, {f / 1e15:.2f} PFLOP, "
                  f"{f / t / 1e12:.1f} TFLOP/s", flush=True)
            report[f"bench_{lens}_s"], report[f"flops_{lens}"] = t, f
            del acc
        dirs = list(range(args.dirs))
        bf = {lens: first_dirs(lens, model, ids, dirs, args.device) for lens in ("S", "R")}
        del model
        torch.mps.empty_cache() if args.device == "mps" else torch.cuda.empty_cache()
        fp = load(torch.float32, args.device)
        for lens in ("S", "R"):
            ref = first_dirs(lens, fp, ids, dirs, args.device)
            errs = [((bf[lens][k] - ref[k]).norm() / ref[k].norm()).item() for k in ref]
            print(f"bf16 vs fp32 {lens} ({args.dirs} directions, every target and layer): median "
                  f"{sorted(errs)[len(errs) // 2]:.2%}, max {max(errs):.2%}", flush=True)
            report[f"bf16_err_{lens}"] = errs
        del fp
        torch.mps.empty_cache() if args.device == "mps" else torch.cuda.empty_cache()
        if args.noise:
            model = load(torch.bfloat16, args.device)
    if args.noise:
        dirs = list(range(args.dirs))
        for lens in ("S", "R"):
            per_doc = [first_dirs(lens, model, torch.tensor(r), dirs, args.device) for r in rows[:args.docs]]
            keys = sorted(per_doc[0])
            spread_by_key = {}
            for key in keys:
                mean = torch.stack([p[key] for p in per_doc]).mean(0)
                pair = [((a[key] - b[key]).norm() / mean.norm()).item() for a, b in itertools.combinations(per_doc, 2)]
                spread_by_key[key] = sorted(pair)[len(pair) // 2]
            vals = sorted(spread_by_key.values())
            spread = vals[len(vals) // 2]
            pred = {n: spread * (2 / n) ** 0.5 / 2 ** 0.5 for n in (48, 250, 500)}
            by_layer = {f"{k[0]}{k[1]}": round(v, 3) for k, v in spread_by_key.items()}
            print(f"noise {lens}: two documents' estimates differ by {spread:.0%} (median over "
                  f"{len(keys)} targets x layers; range {vals[0]:.0%}-{vals[-1]:.0%}); predicted "
                  f"split-half at 96 / 500 / 1,000 documents: "
                  + " / ".join(f"{p:.1%}" for p in pred.values()), flush=True)
            report[f"noise_{lens}"] = {"pair_spread_median": spread, "by_key": by_layer,
                                       "pred_split_half": {2 * n: p for n, p in pred.items()}}
    out = DATA / "summary" / f"gemma4_smoke{'' if args.layers is None else f'_L{args.layers}'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
