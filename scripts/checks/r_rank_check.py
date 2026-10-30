"""Could the R-lens be seeded in a subspace? Its reads are ``W_U diag(γ) J_L w``: only directions
of the final residual that the unembedding weighs heavily matter. Seeding the top ``k`` right
singular vectors ``V_k`` of ``W_U diag(γ)`` would fit ``V_kᵀ J_L`` at ``k/d`` of the cost; this
compares each write's top tokens through ``J_L`` and through ``V_k V_kᵀ J_L``, on an R-lens fit,
for ``--neurons`` random neurons per layer (CPU, no GPU).

    python scripts/checks/r_rank_check.py --r-fit r-gemma-4-12b-L48-dclm
"""
import argparse

import torch

from neuron_lens.paths import FITS
from neuron_lens.transport import Accumulator
from neuron_lens.weights import tensors

MODEL = "google/gemma-4-12B"
P = "model.language_model."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-fit", default="r-gemma-4-12b-L48-dclm")
    ap.add_argument("--neurons", type=int, default=64)
    ap.add_argument("--layers", default="0,4,8,12,16,20,24,28,32,36,40,44")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    torch.manual_seed(0)
    saved = torch.load(FITS / f"{args.r_fit}.pt", map_location="cpu", weights_only=False, mmap=True)
    acc = Accumulator.from_state(saved["state"])
    n = sum(acc.n_docs)
    w = tensors(MODEL, [P + "embed_tokens.weight", P + "norm.weight"])
    U = w[P + "embed_tokens.weight"].float() * w[P + "norm.weight"].float()  # [V, d], tied
    evals, evecs = torch.linalg.eigh(U.T @ U)  # ascending
    V = evecs.flip(1)  # [d, d], columns by decreasing singular value
    energy = evals.flip(0).cumsum(0) / evals.sum()
    print(f"R-lens fit of {n} documents; unembedding energy at k=512/1024/1536/2048: "
          + " ".join(f"{float(energy[k - 1]):.3f}" for k in (512, 1024, 1536, 2048)), flush=True)
    ks = (256, 512, 1024, 1536, 2048, 3072)
    res = {k: [] for k in ks}
    for L in (int(x) for x in args.layers.split(",")):
        lw = tensors(MODEL, [P + f"layers.{L}.mlp.down_proj.weight", P + f"layers.{L}.post_feedforward_layernorm.weight"])
        Wd = lw[P + f"layers.{L}.mlp.down_proj.weight"].float() * lw[P + f"layers.{L}.post_feedforward_layernorm.weight"].float()[:, None]
        cols = torch.randperm(Wd.shape[1])[:args.neurons]
        J = (acc.sums[("out", 0)][L].float() + acc.sums[("out", 1)][L].float()) / n  # [d_final, d_L]
        v = J @ Wd[:, cols]  # [d, k]: each write at the final residual
        full = (U @ v).topk(args.top, dim=0).indices  # [top, neurons]
        for k in ks:
            vk = V[:, :k] @ (V[:, :k].T @ v)
            top = (U @ vk).topk(args.top, dim=0).indices
            ov = [len(set(full[:, j].tolist()) & set(top[:, j].tolist())) / args.top for j in range(len(cols))]
            res[k].append(sum(ov) / len(ov))
        print(f"L{L}: top-{args.top} overlap with the full read at k=" + " ".join(f"{k}:{res[k][-1]:.2f}" for k in ks), flush=True)
    print("mean over layers: " + " ".join(f"k={k}: {sum(v) / len(v):.3f}" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
