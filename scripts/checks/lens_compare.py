"""Compare lenses on every neuron of a few layers, weight-only (no forward pass).

Lens specs: ``logit``; ``J`` (Neuronpedia J-lens); ``R`` (the default Drowse home's R-lens);
``R:<fit>`` (``data/lenses/<fit>``); and ``R:<fit1>+<fit2>`` (their prompt-weighted merge). For
each layer and lens it prints the median top-1 mass and the share of neurons above 0.1/0.5; the
common mode of ``J_L`` (top singular direction's energy share and the median share of a
transported write on it); pairwise top-1 agreement and top-10 overlap between lenses, over the
catalog's kept tokens (with a fit's even and odd documents, exported with
``04_export_rlens.py --half``, the split-half reliability); and, with ``--samples``, random
neurons and the persona experiment's known neurons, both poles, plain and layer-contrastive
(z-scored) reads.

    python scripts/checks/lens_compare.py --lenses logit,J,R:dclm [--layers 2,8,14,20,26,32] [--samples 4]
    python scripts/checks/lens_compare.py --lenses R:dclm-even,R:dclm-odd,R:dclm,J
"""
import argparse
import itertools
import random

import torch
from transformers import AutoTokenizer

from neuron_lens.catalog import kept
from neuron_lens.known import KNOWN
from neuron_lens.lenses import load_lens
from neuron_lens.readout import layer_logit_stats, read, read_z
from neuron_lens.weights import down_proj, readout_names, tensors

MODEL = "Qwen/Qwen3-8B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lenses", default="logit,J,R:dclm")
    ap.add_argument("--layers", default="2,8,14,20,26,32")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    layers = [int(x) for x in args.layers.split(",")]
    known = {k: v for k, v in KNOWN.items() if args.samples}
    all_layers = sorted(set(layers) | {v[0] for v in known.values()})
    w = tensors(MODEL, readout_names() + [down_proj(l) for l in all_layers])
    keep = kept()  # the catalog's token mask
    unembed = w["lm_head.weight"][keep].to(dev, torch.float16)
    gamma = w["model.norm.weight"].to(dev, torch.float32)
    specs = args.lenses.split(",")
    lenses = {s: load_lens(s, MODEL) for s in specs}

    def fmt(ids, k=8):
        return " ".join(repr(tok.decode([i]))[1:-1] for i in keep[ids[:k].cpu()].tolist())

    for layer in all_layers:
        wd = w[down_proj(layer)].to(dev, torch.float32)
        jac = {s: None if m is None else m[layer] for s, m in lenses.items()}
        reads = {(s, sgn): read(sgn * wd, unembed, gamma, jac[s]) for s in specs for sgn in (1, -1)}
        if layer in layers:
            print(f"\n=== L{layer} ({wd.shape[1]} neurons)")
            for s in specs:
                t1 = torch.cat([reads[(s, 1)][0][:, 0], reads[(s, -1)][0][:, 0]])
                line = (f"  {s:16s} top-1 median {t1.median():.3f}  >0.1 {(t1 > 0.1).float().mean():.2f}"
                        f"  >0.5 {(t1 > 0.5).float().mean():.2f}")
                if jac[s] is not None:
                    jl = jac[s].float()
                    u, sv, _ = torch.linalg.svd(jl)
                    t = jl.to(dev) @ wd
                    share = ((u[:, 0].to(dev) @ t).pow(2) / t.pow(2).sum(0)).median()
                    line += f"  | sigma1 energy {sv[0] ** 2 / sv.pow(2).sum():.2f}, write share on u1 {share:.2f}"
                print(line)
            for a, b in itertools.combinations(specs, 2):
                ia = torch.cat([reads[(a, 1)][1], reads[(a, -1)][1]])
                ib = torch.cat([reads[(b, 1)][1], reads[(b, -1)][1]])
                top1 = (ia[:, 0] == ib[:, 0]).float().mean()
                overlap = torch.tensor([len(set(x.tolist()) & set(y.tolist())) / ia.shape[1]
                                        for x, y in zip(ia, ib)]).mean()
                print(f"  {a} vs {b}: top-1 agree {top1:.2f}, top-10 overlap {overlap:.2f}")
        picks = [(f"n{n}", n, sgn) for n in random.sample(range(wd.shape[1]), args.samples)
                 for sgn in (1, -1)] if layer in layers else []
        picks += [(name, n, sgn) for name, (l, n, sgn) in known.items() if l == layer]
        stats = {s: layer_logit_stats(wd, unembed, gamma, jac[s]) for s in specs} if picks else {}
        for label, n, sgn in picks:
            print(f"  L{layer} {label} {'+' if sgn > 0 else '-'}")
            for s in specs:
                p, ids = reads[(s, sgn)]
                print(f"    {s:16s} p1 {p[n, 0]:.2f} | {fmt(ids[n])}")
                z = read_z(sgn * wd[:, n:n + 1], unembed, gamma, jac[s], *stats[s])[0]
                print(f"    {'':16s} z    | {fmt(z)}")


if __name__ == "__main__":
    main()
