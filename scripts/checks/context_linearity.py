"""Per-context linearization (findings §4): does a pole push the logits the same way wherever it fires,
and how much of that push does its lens read capture?

**Poles.** §3's random neurons (``causal_lens.random_neurons``: 32 a layer, seed 0), both poles. A pole
is measured at its top firings, the strongest position in each of its ``FIRINGS`` (24) top documents (the ``acts``
table's ``top_doc``, ``top_pos``, ``top_val``; positions from 1), those where it fired (``top_val > 0``),
and is scored if it fired in at least 2 (Qwen3-8B: 2,290 of 2,304 poles).

**Two effects per firing.** The document up to ``t + WINDOW`` (``t`` the firing's position) runs as four
rows of one batch, with the neuron's activation ``a`` (``down_proj``'s input) at ``t`` multiplied by 1
(clean), 0 (removed), ``1 + EPS`` and ``1 − EPS``. From the log-probabilities over the vocabulary at
positions ``t … t + WINDOW`` (fewer where the document ends):

- **removed:** clean − removed, the firing's whole effect;
- **first-order:** ``(l(1 + EPS) − l(1 − EPS)) / 2·EPS``, a central difference for ``a · ∂logp/∂a``,
  what an exact lens of this one context would read;

each at ``t`` (``pos``) and summed over the ``WINDOW + 1`` positions (``win``, which the findings
report), as vectors over the kept tokens (``catalog.kept``). Every correlation is a Pearson r over them.

**Per firing** (a pole's value is its median over its firings):

- ``nonlin``: removed against first-order;
- ``ctx LOO``: first-order against the mean of the pole's other firings' (leave one out): how much of
  one context's effect any context-free vector could carry;
- ``lens/ctx``: the lens read of the pole's write against first-order.

**Per pole:** ``lens/mean``, the lens read against its mean first-order effect; ``abl~lin mean``, its
mean removed effect against its mean first-order effect; ``split lin``, ``split abl``, the mean over
its even firings against the mean over its odd ones (a ceiling on how stable the mean is);
``steer~lin``, ``steer~abl``, ``steer~lens``, §3's steering at 1× (``causal_lens.py``: the write times
the pole's typical activation added at positions 16+ of 16 neutral DCLM documents; the mean change in
log-probs) against its mean first-order effect, its mean removed effect and its lens read. The
controls (``… ctrl``) score each pole's steering against the next random pole of its layer's values.

The lens read is §3's: ``W_U (γ ⊙ norm(J_L w))`` for the pole's write ``±w`` (``neurons.directions``),
``J_L`` the R-lens of record's ``out`` target (``arch.fits``: the local fit, else the released lens;
the logit lens at the last layer). Layer and band values are medians over the scored poles.

Precision: in bf16 a weak pole's write partly rounds away in the residual, so removed and first-order
disagree from rounding alone (§4: r 0.05-0.77 in bf16 against 0.83-0.99 in fp32 on five test neurons);
the record runs in fp32 with TF32 matmuls (57 minutes on one H100 for Qwen3-8B), and ``--dtype bf16``
reruns it in bf16, to its own file. MPS has no TF32: full fp32, slower.

``measure`` (GPU) saves the per-firing and per-pole values and each pole's mean effects (window-summed,
over the kept tokens) to ``data/causal/<model>-context[-bf16].pt``, after every layer, and resumes
there. ``score`` (any device) needs ``causal_lens.py measure``'s ``data/causal/<model>-causal.pt`` for
steering and writes ``data/summary/context_linearity_<model>.{log,json}``, which figure 9
(``11_figures.py``) draws. The documents come from ``catalog.documents`` (the activations file, else
rebuilt from the corpus), so the released catalog and the model suffice.

    python scripts/checks/context_linearity.py measure [--layers 0,16,30] [--dtype bf16]
    python scripts/checks/context_linearity.py score
"""
import argparse
import json
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM

from causal_lens import CAUSAL, random_neurons
from neuron_lens import arch
from neuron_lens.catalog import MODEL, column, documents, kept, neuron_table
from neuron_lens.neurons import directions
from neuron_lens.paths import SUMMARY
from neuron_lens.readout import rms_norm_columns, transport
from neuron_lens.stats import pearson
from neuron_lens.weights import tensors

STEM = arch.stem(MODEL)
PER_LAYER = 32  # §3's random picks
FIRINGS = 24  # §4's window over the stored top firings
EPS = 0.25
WINDOW = 8  # positions after the firing, 9 in all
MIN_FIRINGS = 2
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
PER_FIRING = ("nonlin", "loo", "lens")  # [P, 2, K, 2]: at the position, window-summed
PER_POLE = ("lens_mean", "abl_lin_mean", "split_lin", "split_abl")  # [P, 2, 2]


def out_path(dtype: str):
    return CAUSAL / f"{STEM}-context{'' if dtype == 'fp32' else '-' + dtype}.pt"


def firings(layer: int, neurons: list[int]) -> tuple[torch.Tensor, ...]:
    """``(doc, pos, val)``, each ``[n, 2, K]``: the neurons' top firings per pole (``+``, ``−``) from ``acts``."""
    t = pq.read_table(neuron_table("acts", layer), columns=["top_doc", "top_pos", "top_val"])
    rows = torch.tensor(neurons + [t.num_rows // 2 + n for n in neurons])
    return tuple(column(t, k, FIRINGS)[rows].view(2, len(neurons), -1).transpose(0, 1) for k in ("top_doc", "top_pos", "top_val"))


class Reads:
    """§3's lens read of each pole's write, as logits over the kept tokens."""

    def __init__(self, dev):
        nm = arch.names(MODEL)
        head = tensors(MODEL, sorted({nm["unembed"], nm["norm"]}))
        self.unembed = head[nm["unembed"]][kept()].to(dev, torch.float32)
        self.gamma = head[nm["norm"]].to(dev, torch.float32)
        self.fit = arch.fits(MODEL)[0]
        self.J = arch.lens(MODEL, "R_out", self.fit)  # [L − 1, d, d]: none at the last layer
        self.dev = dev

    def __call__(self, layer: int, neurons: list[int]) -> torch.Tensor:  # [n, 2, Vk]
        w = directions(MODEL, layer, self.dev)["write"][:, neurons]
        lg = (self.unembed @ rms_norm_columns(transport(w, self.J[layer] if layer < len(self.J) else None),
                                              self.gamma)).T
        return torch.stack([lg, -lg], 1)


@torch.inference_mode()
def measure(args) -> None:
    from neuron_lens.transport import decoder_stack

    dev, dtype = args.device, DTYPES[args.dtype]
    torch.backends.cuda.matmul.allow_tf32 = True
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(dev).eval().requires_grad_(False)
    stack = decoder_stack(lm)
    L, f = len(stack.layers), stack.config.intermediate_size
    keep = kept().to(dev)
    Vk = len(keep)
    docs = documents()
    reads = Reads(dev)
    picks = random_neurons(L, f, PER_LAYER)
    P, K = len(picks), FIRINGS
    meta ={"model": MODEL, "dtype": args.dtype, "per_layer": PER_LAYER, "eps": EPS, "window": WINDOW,
            "lens": f"R_out:{reads.fit}", "contexts": K, "kept": Vk}
    path = out_path(args.dtype)
    if path.exists():
        out = torch.load(path, weights_only=False)
        if out["meta"] != meta:
            raise SystemExit(f"{path} was measured with {out['meta']}, not {meta}: move it to start afresh")
    else:
        nan = float("nan")
        out = {"meta": meta, "picks": picks, "done": [],
               "means": torch.zeros(P, 2, 2, Vk, dtype=torch.float16)}  # first-order, removed; window-summed
        out |= {k: torch.full((P, 2, K, 2), nan) for k in PER_FIRING}
        out |= {k: torch.full((P, 2, K), nan) for k in ("act", "dnext_abl", "dnext_lin")}
        out |= {k: torch.full((P, 2, 2), nan) for k in PER_POLE}
    layers = range(L) if args.layers is None else [int(x) for x in args.layers.split(",")]
    t0 = time.perf_counter()
    for layer in layers:
        if layer in out["done"]:
            continue
        idx = [i for i, (l, _) in enumerate(picks) if l == layer]
        neurons = [picks[i][1] for i in idx]
        doc, pos, val = firings(layer, neurons)
        read = reads(layer, neurons)
        down = stack.layers[layer].mlp.down_proj
        for j, i in enumerate(idx):
            n = neurons[j]
            for p in (0, 1):
                ks = (val[j, p] > 0).nonzero().squeeze(1).tolist()
                if len(ks) < MIN_FIRINGS:
                    continue
                at = [int(pos[j, p, k]) for k in ks]
                seqs = [docs[int(doc[j, p, k])][:q + 1 + WINDOW].long() for k, q in zip(ks, at)]
                m, T = len(seqs), max(len(s) for s in seqs)
                ids = torch.zeros(4 * m, T, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for r, s in enumerate(seqs):
                    ids[4 * r:4 * r + 4, :len(s)] = s
                    mask[4 * r:4 * r + 4, :len(s)] = 1
                rows = torch.arange(4 * m, device=dev)
                cols = torch.tensor(at, device=dev).repeat_interleave(4)
                scale = torch.tensor([1.0, 0.0, 1 + EPS, 1 - EPS], device=dev, dtype=dtype).repeat(m)

                def push(mod, a):
                    x = a[0].clone()
                    x[rows, cols, n] = x[rows, cols, n] * scale
                    return (x,)
                hook = down.register_forward_pre_hook(push)
                hs = stack(input_ids=ids.to(dev), attention_mask=mask.to(dev)).last_hidden_state
                hook.remove()
                lin = torch.zeros(m, 2, Vk, device=dev)  # [firing, pos | win, kept token]
                abl = torch.zeros_like(lin)
                for r, (q, s) in enumerate(zip(at, seqs)):
                    lsm = torch.log_softmax(arch.logits(lm, hs[4 * r:4 * r + 4, q:len(s)]), -1)  # [4, w, V]
                    a_, d_ = lsm[0] - lsm[1], (lsm[2] - lsm[3]) / (2 * EPS)
                    abl[r, 0], abl[r, 1] = a_[0, keep], a_[:, keep].sum(0)
                    lin[r, 0], lin[r, 1] = d_[0, keep], d_[:, keep].sum(0)
                    if q + 1 < len(s):
                        out["dnext_abl"][i, p, ks[r]] = a_[0, s[q + 1]].item()
                        out["dnext_lin"][i, p, ks[r]] = d_[0, s[q + 1]].item()
                out["act"][i, p, ks] = val[j, p, ks].float()
                rd = read[j, p][None]
                loo = (lin.sum(0, keepdim=True) - lin) / (m - 1)
                for w in (0, 1):
                    out["nonlin"][i, p, ks, w] = pearson(abl[:, w], lin[:, w]).cpu()
                    out["loo"][i, p, ks, w] = pearson(loo[:, w], lin[:, w]).cpu()
                    out["lens"][i, p, ks, w] = pearson(rd, lin[:, w]).cpu()
                    mean_lin, mean_abl = lin[:, w].mean(0, keepdim=True), abl[:, w].mean(0, keepdim=True)
                    out["lens_mean"][i, p, w] = pearson(rd, mean_lin).item()
                    out["abl_lin_mean"][i, p, w] = pearson(mean_abl, mean_lin).item()
                    if m >= 4:
                        even, odd = lin[0::2, w].mean(0, keepdim=True), lin[1::2, w].mean(0, keepdim=True)
                        out["split_lin"][i, p, w] = pearson(even, odd).item()
                        even, odd = abl[0::2, w].mean(0, keepdim=True), abl[1::2, w].mean(0, keepdim=True)
                        out["split_abl"][i, p, w] = pearson(even, odd).item()
                out["means"][i, p, 0], out["means"][i, p, 1] = lin[:, 1].mean(0).half().cpu(), abl[:, 1].mean(0).half().cpu()
        out["done"].append(layer)
        torch.save(out, path)
        print(f"{time.strftime('%H:%M:%S')} L{layer}: {len(idx)} neurons, {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"-> {path}", flush=True)


def score(args) -> None:
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    saved = torch.load(out_path(args.dtype), weights_only=False)
    causal = torch.load(CAUSAL / f"{STEM}-causal.pt", weights_only=False, mmap=True)
    picks, done = saved["picks"], set(saved["done"])
    L = arch.text_config(MODEL).num_hidden_layers
    at = {pk: i for i, pk in enumerate(causal["picks"])}  # §3 also sampled named neurons
    keep = kept()
    steer = causal["steer"][[at[pk] for pk in picks], :, 0][..., keep].float()  # [P, 2, Vk]: 1x
    lin, abl = saved["means"][:, :, 0].float(), saved["means"][:, :, 1].float()
    reads, lread = torch.zeros_like(steer), Reads(dev)
    ctrl = torch.arange(len(picks))  # each pole's control: the next random pole of its layer
    for layer in sorted(done):
        idx = [i for i, (l, _) in enumerate(picks) if l == layer]
        reads[idx] = lread(layer, [picks[i][1] for i in idx]).cpu()
        ctrl[idx] = torch.tensor(idx).roll(1)
    flat = lambda t: t.reshape(-1, t.shape[-1])
    vs = {"steer~lin": (steer, lin), "steer~abl": (steer, abl), "steer~lens": (steer, reads)}
    r_steer = {k: pearson(flat(a), flat(b)).view(-1, 2) for k, (a, b) in vs.items()}
    r_ctrl = {k + " ctrl": pearson(flat(a), flat(b[ctrl])).view(-1, 2) for k, (a, b) in vs.items()}

    def per_pole(name: str, w: int) -> torch.Tensor:  # [P, 2]; per-firing values: the pole's median
        return saved[name][..., w].nanmedian(-1).values if name in PER_FIRING else saved[name][..., w]

    cols = {"nonlin win": per_pole("nonlin", 1), "nonlin pos": per_pole("nonlin", 0),
            "ctx LOO win": per_pole("loo", 1), "lens/ctx win": per_pole("lens", 1),
            "lens/mean win": per_pole("lens_mean", 1), "abl~lin mean": per_pole("abl_lin_mean", 1),
            "split lin": per_pole("split_lin", 1), "split abl": per_pole("split_abl", 1)} | r_steer | r_ctrl
    scored = ~saved["lens_mean"][:, :, 1].isnan()  # [P, 2]: fired at MIN_FIRINGS+ top firings
    in_layers = lambda lo, hi: torch.tensor([lo <= l <= hi for l, _ in picks])[:, None] & scored
    row = lambda sel: {k: t[sel].nanmedian().item() for k, t in cols.items()} | {"poles": int(sel.sum())}
    bands = [(lo, hi) for lo, hi in arch.bands(L) + [(0, L - 1)] if any(lo <= l <= hi for l in done)]
    table = {f"L{lo}-{hi}": row(in_layers(lo, hi)) for lo, hi in bands}
    layers = {layer: row(in_layers(layer, layer)) for layer in sorted(done)}
    dn = saved["dnext_abl"] - saved["dnext_lin"]
    lines = [f"per-context linearization, {MODEL}: {saved['meta']}, layers {sorted(done)}",
             "medians over scored poles (fired at 2+ of their top firings) of Pearson r over the kept tokens; per-firing "
             "values are each pole's median over its firings. nonlin: removed against first-order; ctx LOO: one firing's "
             "first-order effect against the mean of the pole's others; lens/ctx, lens/mean: the R-lens write read "
             "against each firing's and against their mean; abl~lin mean: the means of the two effects; split: even "
             "firings' mean against odd; steer~: §3's 1x steering against the mean effects and the lens read; ctrl: "
             "against the next random pole's. win: summed over the firing's position and the next 8, pos: at it", ""]
    lines.append("band     poles  " + "  ".join(f"{k:>15s}" for k in cols))
    for name, r in table.items():
        lines.append(f"{name:8s} {r['poles']:5d}  " + "  ".join(f"{r[k]:15.2f}" for k in cols))
    lines += ["", "next-token log-prob change at the firing, removed − first-order, median |difference| (nats): " + ", ".join(
        f"L{lo}-{hi} {dn[in_layers(lo, hi)].abs().nanmedian():.3f}" for lo, hi in bands[:-1])]
    text = "\n".join(lines)
    print(text)
    stem = f"context_linearity_{STEM}" + ("" if args.dtype == "fp32" else f"-{args.dtype}")
    (SUMMARY / f"{stem}.log").write_text(text + "\n")
    (SUMMARY / f"{stem}.json").write_text(json.dumps({"meta": saved["meta"], "bands": table, "layers": layers}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("measure", "score"))
    ap.add_argument("--dtype", choices=tuple(DTYPES), default="fp32")
    ap.add_argument("--layers", default=None, help="measure only these layers (a,b,c)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    args = ap.parse_args()
    measure(args) if args.what == "measure" else score(args)


if __name__ == "__main__":
    main()
