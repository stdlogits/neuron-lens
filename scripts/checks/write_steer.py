"""Write-side check of the Gemma 4 12B R-lens: does pushing a pole's write raise the tokens its
R-lens read names, more than other poles' read tokens?

For each pole of media_reads.py's panel, the write is ``±s_L · γ_post ⊙ W_down[:, n]`` (the
post-feedforward norm under the LN rule, then the layer scalar), on layer ``L``'s output. Its
read through a lens is ``W_U diag(γ_final) J_L w`` (``J_L`` the R-lens's ``out`` target, the
identity at the last layer: the logit lens), top ``--top`` tokens over those seen 5+ times in
the lens corpus. The push adds ``--beta`` times each position's residual norm along ``ŵ`` to
layer ``L``'s output at every position from 16 on; the lift of a token set is its mean change
in next-token log-probability (over the positions read) minus the vocabulary's median change.
A pole's score: the rank of the lift of its own read's tokens among the lifts of the other
poles' read tokens under its push (1 = the most, 0.5 = chance); the null permutes ownership.

Contexts: held-out text (DCLM rows past the fit's, 64 tokens, read at positions 16-62) and
image prompts (COCO, the 64-patch photograph prompt of media_lens.py, read at the text after
the image).

    python scripts/checks/write_steer.py --device cuda --r-fit r-gemma-4-12b-L48-dclm
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from media_lens import images, processed, stack, to  # noqa: E402
from media_reads import MODEL  # noqa: E402

from neuron_lens.corpus import lens_docs  # noqa: E402
from neuron_lens.paths import CORPUS_DIR, DATA, FITS  # noqa: E402
from neuron_lens.transport import SKIP_FIRST, Accumulator, decoder_stack  # noqa: E402

OUT = DATA / "summary" / "gemma4_media"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--r-fit", default="r-gemma-4-12b-L48-dclm")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--contexts", type=int, default=16)
    args = ap.parse_args()
    t0 = time.perf_counter()
    dev = args.device
    saved = torch.load(FITS / f"{args.r_fit}.pt", map_location="cpu", weights_only=False, mmap=True)
    acc = Accumulator.from_state(saved["state"])
    n_docs = acc.n_docs
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager")
    lm = lm.to(dev).eval().requires_grad_(False)
    model = decoder_stack(lm)
    tok = AutoTokenizer.from_pretrained(MODEL)
    proc = AutoProcessor.from_pretrained(MODEL)
    poles = torch.load(OUT / "reads.pt", weights_only=False)["poles"]
    P, n_layers = len(poles), len(model.layers)
    print(f"loaded in {time.perf_counter() - t0:.0f}s; R-lens fit of {sum(n_docs)} documents", flush=True)

    # Writes and their reads.
    W = []
    for L, n, s in poles:
        layer = model.layers[L]
        W.append(s * float(layer.layer_scalar) * layer.post_feedforward_layernorm.weight.float().cpu()
                 * layer.mlp.down_proj.weight[:, n].float().cpu())
    W = torch.stack(W)  # [P, d]
    counts = torch.bincount(torch.tensor([t for r in lens_docs(CORPUS_DIR / "dclm.jsonl", tok, 1000, 4096, 1) for t in r]),
                            minlength=model.embed_tokens.weight.shape[0])
    keep = (counts >= 5).nonzero().squeeze(1)
    U = (model.embed_tokens.weight[keep].float() * model.norm.weight.float()).to(dev)  # tied W_U, final norm γ
    reads = {}
    for kind in ("R", "logit"):
        V = []
        for i, (L, _, _) in enumerate(poles):
            w = W[i]
            if kind == "R" and L < n_layers - 1:
                J = sum(acc.sums[("out", h)][L].float() for h in (0, 1)) / sum(n_docs)  # [d_out, d_in]
                w = J @ w
            V.append(w)
        logits = torch.stack(V).to(dev) @ U.T  # [P, keep]
        reads[kind] = keep[logits.topk(args.top, dim=1).indices.cpu()]  # vocabulary ids
    overlap = np.mean([len(set(reads["R"][i].tolist()) & set(reads["logit"][i].tolist())) / args.top for i in range(P)])
    print(f"reads done: R vs logit-lens top-{args.top} overlap {overlap:.2f}", flush=True)
    samples = {f"L{L}.n{n}{'+' if s > 0 else '-'}": [tok.decode([int(t)]) for t in reads["R"][i][:8]]
               for i, (L, n, s) in enumerate(poles[::16])}

    # Contexts.
    rows = lens_docs(CORPUS_DIR / "dclm.jsonl", tok, 1000, 64, 64)[-args.contexts:]
    text_in = {"input_ids": torch.tensor(rows)}
    text_read = torch.zeros(64, dtype=torch.bool)
    text_read[SKIP_FIRST:63] = True
    img_in = stack([processed(proc, "image", x) for x in images(args.contexts)])
    ids = img_in["input_ids"][0]
    img_read = torch.zeros(len(ids), dtype=torch.bool)
    img_read[int((ids == tok.convert_tokens_to_ids(proc.eoi_token)).nonzero()[0]):len(ids) - 1] = True  # the text after the image

    push = {}

    def hook(L):
        def f(_m, _a, o):
            if push.get("L") == L:
                h = o.clone()
                norm = h[:, SKIP_FIRST:].float().norm(dim=-1, keepdim=True)
                h[:, SKIP_FIRST:] += (args.beta * norm * push["w"]).to(h.dtype)
                return h
        return model.layers[L].register_forward_hook(f)
    hooks = [hook(L) for L in sorted({L for L, _, _ in poles})]
    report = {"fit_docs": sum(n_docs), "beta": args.beta, "samples": samples, "results": {}}
    rng = np.random.default_rng(0)
    for cname, inputs, read in (("text", text_in, text_read), ("image", img_in, img_read)):
        inputs = to(inputs, dev)
        with torch.no_grad():
            base = torch.log_softmax(lm(**inputs).logits[:, read.to(dev)].float(), -1).mean((0, 1))
        lifts = {k: torch.zeros(P, P) for k in reads}  # [pole pushed, pole whose tokens]
        t1 = time.perf_counter()
        for i, (L, _, _) in enumerate(poles):
            push.update(L=L, w=(W[i] / W[i].norm()).to(dev))
            with torch.no_grad():
                d = torch.log_softmax(lm(**inputs).logits[:, read.to(dev)].float(), -1).mean((0, 1)) - base
            push.clear()
            d = (d - d.median()).cpu()
            for k, R in reads.items():
                lifts[k][i] = d[R].mean(1)
        for k, M in lifts.items():
            rank = lambda own: torch.tensor([float((M[i] < M[i, own[i]]).double().mean() * P / (P - 1)) for i in range(P)])
            real = rank(list(range(P)))
            null = torch.cat([rank(list(rng.permutation(P))) for _ in range(5)])
            p95 = float(null.quantile(0.95))
            bands = {}
            for b, lo, hi in (("L0-5", 0, 5), ("L6-11", 6, 11), ("L12-23", 12, 23), ("L24-35", 24, 35), ("L36-47", 36, 47)):
                sel = torch.tensor([lo <= L <= hi for L, _, _ in poles])
                bands[b] = round(float(real[sel].median()), 3)
            report["results"][f"{cname}_{k}"] = {"median": float(real.median()), "share_above_null_p95":
                                                 float((real > p95).double().mean()), "bands": bands,
                                                 "own_lift_median": float(M.diagonal().median())}
            print(f"{cname:5s} {k:5s}: median rank {real.median():.3f}, above null p95 {(real > p95).double().mean():.0%}"
                  f" (chance 5%); own-token lift median {M.diagonal().median():+.3f} nats; by band "
                  + " ".join(f"{b} {v:.2f}" for b, v in bands.items()) + f" [{time.perf_counter() - t1:.0f}s]", flush=True)
    for h in hooks:
        h.remove()
    path = OUT / f"write_steer_{sum(n_docs)}docs.json"
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"wrote {path}; done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
