"""Findings §3's steering test on Qwen3.5's expert neurons: does an R-lens's read of a write predict
what pushing it does? Doses come from the residual's norm (Qwen3.5 has no activation statistics
yet). ``--fits`` takes the fit of record, or several fits to compare (a variant's rules).

For ``--per-layer`` random (expert, neuron) pairs per layer, plus ``--shared`` shared-expert
neurons, and both signs, the write ``w = s·W_down[:, n]`` is pushed onto layer ``L``'s output as
``β·‖h_t‖·ŵ`` at every position from 16, in ``--contexts`` full-length documents of the DCLM
activation slice (not the lens corpus); the effect is the mean change in next-token log-prob over
those positions and documents, against an unsteered row in the same batch. A lens's read of the
write is ``W_U (γ ⊙ norm(J_L w))`` (each fit's R-lens ``out`` target; the logit lens beside them;
the identity at the last layer). Score: Pearson r over tokens seen 5+ times in the lens corpus.
Control: each pick's effect against the next pick's read. Prints median r per band of layers and
lens, and each fit's split-half by layer.

    python scripts/checks/expert_steer.py --device cuda --fits r-qwen3.5-35b-a3b-L40-dclm-T512
"""
import argparse
import json
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_lens.corpus import lens_docs
from neuron_lens.paths import CORPUS_DIR, DATA, FITS
from neuron_lens.rules import norm_weight
from neuron_lens.transport import SKIP_FIRST, Accumulator, decoder_stack

MODEL = "Qwen/Qwen3.5-35B-A3B"


def picks(model, per_layer: int, shared: int, seed: int = 0) -> list[tuple]:
    """``(layer, expert, neuron)`` per pick; expert ``-1`` is the shared expert."""
    rng = random.Random(seed)
    out = []
    for L, layer in enumerate(model.layers):
        ex = layer.mlp.experts
        out += [(L, rng.randrange(ex.num_experts), rng.randrange(ex.intermediate_dim)) for _ in range(per_layer)]
        out += [(L, -1, n) for n in rng.sample(range(layer.mlp.shared_expert.down_proj.weight.shape[1]), shared)]
    return out


def write(model, L: int, e: int, n: int) -> torch.Tensor:
    mlp = model.layers[L].mlp
    w = mlp.shared_expert.down_proj.weight[:, n] if e < 0 else mlp.experts.down_proj[e][:, n]
    return w.float()


@torch.inference_mode()
def effects(model, lm_head, ctx, ws: list[torch.Tensor], L: int, beta: float, keep, batch: int) -> torch.Tensor:
    """``[len(ws), K]``: each push's mean change in next-token log-prob over the kept tokens."""
    C, T = ctx.shape
    dev = ctx.device
    pos = (torch.arange(T, device=dev) >= SKIP_FIRST)[None, :, None]
    out = []
    for s in range(0, len(ws), batch):
        part = ws[s:s + batch]
        dirs = torch.stack([torch.zeros_like(part[0])] + [w / w.norm() for w in part]).to(dev)  # row 0 unsteered
        rows = dirs.repeat_interleave(C, 0)[:, None, :]

        def push(_m, _a, o):
            return o + (beta * o.float().norm(dim=-1, keepdim=True) * rows * pos).to(o.dtype)
        hook = model.layers[L].register_forward_hook(push)
        hs = model(input_ids=ctx.repeat(len(dirs), 1), use_cache=False).last_hidden_state[:, SKIP_FIRST:-1]
        hook.remove()
        hs = hs.reshape(len(dirs), -1, hs.shape[-1])
        mean = torch.zeros(len(dirs), len(keep), device=dev)
        for c in range(0, hs.shape[1], 128):  # [pushes, 128, vocab] logits at a time
            lp = torch.log_softmax(lm_head(hs[:, c:c + 128]).float(), -1)[..., keep]
            mean += lp.sum(1)
        mean /= hs.shape[1]
        out.append((mean[1:] - mean[:1]).cpu())
    return torch.cat(out)


def pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--fits", required=True, help="comma-separated R-lens fits in data/fits/")
    ap.add_argument("--per-layer", type=int, default=12)
    ap.add_argument("--shared", type=int, default=4)
    ap.add_argument("--contexts", type=int, default=8)
    ap.add_argument("--ctx-len", type=int, default=512)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=4, help="pushes per forward pass (times --contexts rows)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    model, lm_head = decoder_stack(lm), lm.lm_head
    n_layers = len(model.layers)
    counts = torch.bincount(torch.tensor([t for r in lens_docs(CORPUS_DIR / "dclm.jsonl", tok, 1000, 4096, 1) for t in r]),
                            minlength=lm_head.weight.shape[0])
    keep = (counts >= 5).nonzero().squeeze(1).to(dev)
    ctx = torch.tensor(lens_docs(CORPUS_DIR / "act-dclm.jsonl", tok, args.contexts, args.ctx_len, args.ctx_len,
                                 full=True), device=dev)
    U, gamma = lm_head.weight[keep].float(), norm_weight(model.norm).float()
    fits = {}
    for name in args.fits.split(","):
        acc = Accumulator.from_state(torch.load(FITS / f"{name}.pt", map_location="cpu", weights_only=False)["state"])
        fits[name] = acc.mean("out")  # [L − 1, d, d]
        print(f"{name}: {sum(acc.n_docs)} documents; split-half out by layer: "
              + " ".join(f"{x:.2f}" for x in acc.split_half("out")), flush=True)
    sel = picks(model, args.per_layer, args.shared)
    print(f"loaded in {time.perf_counter() - t0:.0f}s; {len(sel)} picks, both signs, "
          f"{ctx.shape[0]} contexts of {ctx.shape[1]} tokens", flush=True)

    def read(J, w):
        t = w if J is None else J.to(dev, torch.float32) @ w
        t = t * torch.rsqrt(t.pow(2).mean() + 1e-6) * gamma
        return U @ t

    rows = []
    for L in range(n_layers):
        mine = [p for p in sel if p[0] == L]
        ws = [s * write(model, *p) for p in mine for s in (1, -1)]
        eff = effects(model, lm_head, ctx, ws, L, args.beta, keep, args.batch)  # [P, K]
        lenses = {"logit": None} | {n: (None if L == n_layers - 1 else J[L]) for n, J in fits.items()}
        for name, J in lenses.items():
            reads = torch.stack([read(J, w.to(dev)) for w in ws]).cpu()
            r = pearson(eff, reads)
            ctrl = pearson(eff, reads.roll(-2, 0))  # the next pick, same sign
            for (Lp, e, n), s, ri, ci in zip([p for p in mine for _ in (1, -1)], [1, -1] * len(mine), r[::1], ctrl):
                rows.append({"layer": L, "expert": e, "neuron": n, "sign": s, "lens": name,
                             "r": ri.item(), "control": ci.item()})
        print(f"{time.strftime('%H:%M:%S')} L{L}: " + "  ".join(
            f"{name} {torch.tensor([x['r'] for x in rows if x['layer'] == L and x['lens'] == name]).median():.2f}"
            for name in lenses), flush=True)
    out = DATA / "summary" / "qwen35_expert_steer.json"
    out.write_text(json.dumps({"meta": vars(args), "rows": rows}) + "\n")
    names = ["logit"] + list(fits)
    print("\nmedian r (routed experts | shared expert), control in brackets, by band of layers:")
    for lo in range(0, n_layers, 4):
        cells = []
        for name in names:
            sub = [x for x in rows if lo <= x["layer"] < lo + 4 and x["lens"] == name]
            med = lambda f: torch.tensor([x["r"] for x in sub if f(x)]).median().item()
            ctl = torch.tensor([x["control"] for x in sub]).median().item()
            cells.append(f"{name}: {med(lambda x: x['expert'] >= 0):.2f} | {med(lambda x: x['expert'] < 0):.2f} [{ctl:.2f}]")
        print(f"L{lo}-L{lo + 3}  " + "   ".join(cells), flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
