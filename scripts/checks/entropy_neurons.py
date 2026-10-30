"""Entropy neurons: late neurons that act through the final norm's scale, not the unembedding.

Stolfo et al. 2024 ("Confidence regulation neurons in language models") find, in GPT-2 and
Llama-2, final-layer neurons with a large write that lies mostly in the unembedding's effective
null space: they barely move any logit directly, but they change the residual stream's norm, so
the final norm rescales every logit and the output's entropy moves. Their test ablates a neuron
and compares the total effect with the direct one, the final norm's scale frozen at its clean
value. This is the same test on Qwen3-8B's last layers, as a check on the weight-only screen in
``08_reads.py`` (``null_share``), and on a family no write-side lens can read.

Per layer in ``--layers``: the ``--top`` neurons by their null-space write at typical high
activation, ``act_top × ‖P w‖`` for the better pole (``P`` the projection on the bottom 1% of
singular directions of ``W_U diag(γ)``; the null share alone misses a small write that fires at
thousands), as many random ones as a control, and ``--extra``. Each
is zeroed at every position past the first of the documents (the first full-length ones of the
DCLM, code and chat activation slices), one forward each. Per neuron: the mean |change| in output
entropy at the valid positions, total and with the scale frozen; at the positions where each pole
fires (past a quarter of its typical high activation, from the catalog), what the firing adds to
the entropy, the loss and the log of the final RMS (clean − ablated); and the correlation over
the vocabulary between the mean change in log-probs and log unigram frequency (the paper's
token-frequency neurons).

    python scripts/checks/entropy_neurons.py [--layers 32-35] [--top 8] [--extra L:n,...]
"""
import argparse
import json
import random

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_lens.catalog import neuron_table
from neuron_lens.corpus import act_file, lens_docs
from neuron_lens.paths import DATA, SUMMARY
from neuron_lens.stats import pearson
from neuron_lens.weights import down_proj, readout_names, tensors

MODEL = "Qwen/Qwen3-8B"
NULL_SHARE = 0.01  # as in 08_reads.py
SLICES = {"dclm": 64, "code": 32, "chat": 32}
SEQ = 128
CHUNK = 1024


@torch.no_grad()
def outputs(h: torch.Tensor, rms: torch.Tensor, gamma, wu, targets, ref=None, keep=None) -> tuple:
    """Entropy and NLL of the target per position, chunked; with ``ref`` (the clean final residual
    and RMS), also the summed log-prob change (clean − this) over the ``keep`` vocabulary."""
    ent, nll, dvec = [], [], 0
    for i in range(0, len(h), CHUNK):
        x = (h[i:i + CHUNK] / rms[i:i + CHUNK] * gamma).to(wu.dtype)
        lp = torch.log_softmax((x @ wu.T).float(), dim=-1)
        ent.append(-(lp.exp() * lp).sum(-1))
        nll.append(-lp.gather(1, targets[i:i + CHUNK, None]).squeeze(1))
        if ref is not None:
            xc = (ref[0][i:i + CHUNK] / ref[1][i:i + CHUNK] * gamma).to(wu.dtype)
            dvec = dvec + (torch.log_softmax((xc @ wu.T).float(), dim=-1) - lp)[:, keep].sum(0)
    return torch.cat(ent), torch.cat(nll), dvec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="32-35")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--extra", default="", help="more layer:neuron, comma-separated")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    a, b = args.layers.split("-")
    layers = list(range(int(a), int(b) + 1))
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    head = tensors(MODEL, readout_names())
    wu_full = head["lm_head.weight"].float() * head["model.norm.weight"].float()
    _, evecs = torch.linalg.eigh(wu_full.T @ wu_full)  # ascending
    null = evecs[:, :max(1, int(NULL_SHARE * wu_full.shape[1]))]
    del wu_full
    rng = random.Random(0)
    extra = [tuple(map(int, x.split(":"))) for x in filter(None, args.extra.split(","))]
    cands, screen, acts = [], {}, {}
    for l in sorted(set(layers) | {l for l, _ in extra}):
        wd = tensors(MODEL, [down_proj(l)])[down_proj(l)].float()
        nn_ = (null.T @ wd).norm(dim=0)
        share = nn_.pow(2) / wd.pow(2).sum(0)
        acts[l] = pq.read_table(neuron_table("acts", l), columns=["act_top"]).column(0).to_pylist()
        score = torch.tensor(acts[l]).view(2, -1).amax(0) * nn_
        top = score.topk(args.top).indices.tolist() if l in layers else []
        rand = rng.sample([n for n in range(wd.shape[1]) if n not in top], args.top) if l in layers else []
        more = [n for l2, n in extra if l2 == l and n not in top + rand]
        screen[l] = {"null_share_q50": share.median().item(), "null_share_q99": share.quantile(0.99).item(),
                     "w_norm_q50": wd.norm(dim=0).median().item(),
                     "neurons": {n: {"null_share": share[n].item(), "null_norm": nn_[n].item(), "score": score[n].item(),
                                     "w_norm": wd[:, n].norm().item()} for n in top + rand + more}}
        cands += [(l, n, "top") for n in top] + [(l, n, "random") for n in rand] + [(l, n, "extra") for n in more]

    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = []
    for name, k in SLICES.items():
        rows += lens_docs(act_file(name), tok, 4 * k, SEQ, SEQ)[:k]
    ids = torch.tensor(rows)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    gamma = model.model.norm.weight.float()
    eps = model.config.rms_norm_eps
    keep = (torch.load(DATA / "token_counts.pt", weights_only=False)["counts"] >= 5).nonzero().squeeze(1)
    wu = model.lm_head.weight
    logf = torch.load(DATA / "token_counts.pt", weights_only=False)["counts"][keep].float().log().to(dev)
    # Valid positions: past the first, with a next token.
    valid = torch.zeros(ids.shape, dtype=torch.bool)
    valid[:, 1:-1] = True
    targets = ids[:, 1:].contiguous()
    targets = torch.cat([targets, targets[:, -1:]], 1)[valid].to(dev)

    def run(layer: int | None = None, neuron: int | None = None) -> tuple[torch.Tensor, dict]:
        cap, got = [], {}
        hooks = [model.model.norm.register_forward_pre_hook(lambda m, x: cap.append(x[0].float()))]
        for l in sorted({c[0] for c in cands}):
            ns = [c[1] for c in cands if c[0] == l]

            def grab(m, x, l=l, ns=ns):
                if layer is None:
                    got.setdefault(l, []).append(x[0][..., ns].float())
                elif l == layer:
                    x[0][:, 1:, neuron] = 0
            hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(grab))
        try:
            with torch.no_grad():
                for i in range(0, len(ids), args.batch):
                    model.model(ids[i:i + args.batch].to(dev))
        finally:
            for hk in hooks:
                hk.remove()
        return torch.cat(cap)[valid.to(dev)], {l: torch.cat(v)[valid.to(dev)] for l, v in got.items()}

    h_c, clean_acts = run()
    rms_c = (h_c.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    ent_c, nll_c, _ = outputs(h_c, rms_c, gamma, wu, targets)
    lines = [f"entropy-neuron check, {MODEL}: {len(cands)} neurons, {len(targets):,} positions "
             f"({', '.join(f'{k} {v}' for k, v in SLICES.items())} documents of {SEQ} tokens); clean entropy "
             f"{ent_c.mean():.2f} nats, NLL {nll_c.mean():.2f}",
             "null space: bottom 1% of W_U diag(γ)'s singular directions; per layer, median / 99th-percentile "
             "null share of a write: " + "; ".join(f"L{l} {s['null_share_q50']:.3f} / {s['null_share_q99']:.3f}"
                                                   for l, s in screen.items() if "null_share_q50" in s),
             "",
             "per neuron: score (act_top × null-space write norm), null share and |w|; |ΔH| mean |entropy change| from ablation, total / scale frozen "
             "(direct share = frozen/total); then per pole where it fires (n positions): what firing adds to "
             "entropy H (total / frozen), NLL and log RMS; freq r = corr(mean Δlog-prob, log frequency)"]
    out = []
    for l, n, kind in cands:
        h_a, _ = run(l, n)
        rms_a = (h_a.pow(2).mean(-1, keepdim=True) + eps).sqrt()
        ent_t, nll_t, dvec = outputs(h_a, rms_a, gamma, wu, targets, (h_c, rms_c), keep.to(dev))
        ent_f, _, _ = outputs(h_a, rms_c, gamma, wu, targets)
        dvec = dvec / len(targets)  # mean log-prob change (clean − ablated), against log frequency
        a = clean_acts[l][:, [c[1] for c in cands if c[0] == l].index(n)]
        d_tot, d_fro = ent_c - ent_t, ent_c - ent_f
        d_nll, d_rms = nll_c - nll_t, (rms_c / rms_a).log().squeeze(1)
        w = screen[l]["neurons"][n]
        r = {"layer": l, "neuron": n, "kind": kind, **w,
             "abs_dH_total": d_tot.abs().mean().item(), "abs_dH_frozen": d_fro.abs().mean().item(),
             "corr_act_dH": pearson(a, d_tot).item(), "freq_r": pearson(dvec, logf).item(), "poles": {}}
        r["direct_share"] = r["abs_dH_frozen"] / max(r["abs_dH_total"], 1e-12)
        for s in (1, -1):
            top = acts[l][n if s > 0 else len(acts[l]) // 2 + n]
            m = s * a > top / 4
            if m.sum() > 0:
                r["poles"]["+" if s > 0 else "−"] = {
                    "n": int(m.sum()), "act_top": top, "dH": d_tot[m].mean().item(), "dH_frozen": d_fro[m].mean().item(),
                    "dNLL": d_nll[m].mean().item(), "dlogRMS": d_rms[m].mean().item()}
        out.append(r)
        poles = "  ".join(f"{p} n {v['n']:5d}: H {v['dH']:+.3f} / {v['dH_frozen']:+.3f}, NLL {v['dNLL']:+.3f}, "
                          f"logRMS {v['dlogRMS']:+.3f}" for p, v in r["poles"].items())
        lines.append(f"  L{l} n{n:<5d} {kind:6s} score {r['score']:6.0f} null {r['null_share']:.2f} |w| {r['w_norm']:.2f}  |ΔH| "
                     f"{r['abs_dH_total']:.4f} / {r['abs_dH_frozen']:.4f} (direct {r['direct_share']:.2f})  "
                     f"r(act,ΔH) {r['corr_act_dH']:+.2f}  freq r {r['freq_r']:+.2f}  {poles}")
        print(lines[-1], flush=True)
    text = "\n".join(lines)
    print(text)
    (SUMMARY / "entropy_neurons_qwen3-8b.log").write_text(text + "\n")
    (SUMMARY / "entropy_neurons_qwen3-8b.json").write_text(json.dumps({"screen": screen, "neurons": out}, indent=1))


if __name__ == "__main__":
    main()
