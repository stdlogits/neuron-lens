"""Causal checks of write-side lens reads: does a lens predict what a neuron's pole does?

``measure`` (GPU) runs, for a sample of neurons per layer and both poles, with no lens:

- **steering:** ``s·α·W_down[:, n]`` added to layer ``L``'s MLP output at every valid position
  (``[16, T)``, as the lenses) of neutral contexts (the first full-length documents of the DCLM
  activation slice), at ``α`` = 1× and 4× the pole's typical high activation (``act_top`` in the
  catalog's ``acts`` table). Saved: the mean change in log-probs over
  the valid positions and contexts, ``[V]`` per pole and dose, against an unsteered row in the
  same batch (so batch-shape differences in bf16 kernels do not enter). Adding at every position and
  averaging over outputs sums over offsets, as the R-lens does.
- **ablation in context:** the neuron set to 0 at the position where the pole peaks in each of its
  first ``--top-contexts`` top documents (the ``acts`` table's ``top_doc`` and ``top_pos``). Saved:
  what the firing contributes (clean − ablated log-probs) at that position and summed over it and
  the next ``WINDOW`` positions, averaged over contexts, and the change in log-prob of the token that
  actually comes next.

The doses and ablation sites come from the catalog (the ``acts`` tables; documents from
``catalog.documents``), and every input is saved with the results: the steering contexts, ``α``, and
each ablation's document and position (``top_doc``, ``top_pos``: rows of ``docs.parquet``), so the file
reproduces on its own. ``measure`` saves its progress every ``--save-minutes`` and resumes when rerun.

``score`` (any device) compares each lens's read of ``s·W_down[:, n]`` (its logits
``W_U (γ ⊙ norm(J_L w))``) with those vectors: the Pearson correlation over the vocabulary (tokens
seen 5+ times in the activation corpora) and the top-10 overlap, per layer and lens. The control
pairs each pole's vectors with the read of the next random neuron of the same layer: what any
write of that layer shares (the common mode). A pole's read **predicts** a target when its r passes
the 99th percentile of the control's, pooled over layers (§3: from L5, 94-100% of random poles' writes
predict their 1× steering); the per-layer share of poles that do, and of control pairs (chance).
The per-layer medians and shares use the random picks only; the named poles are listed one by one.

Gemma 4 12B (``NEURON_LENS_MODEL=google/gemma-4-12B``): the MLP output passes a post-feedforward
RMSNorm before the residual, so the push is added after it: ``s·α·γ_post ⊙ W_down[:, n]`` times the
norm's scale at each position, the write as it reaches the residual (the lenses' LN rule; the layer
scalar is the model's own); log-probs are after the final softcap; no named poles, random ones only.

    python scripts/checks/causal_lens.py measure
    python scripts/checks/causal_lens.py score --lenses logit,J,R:dclm
    python scripts/checks/causal_lens.py score --lenses R:dclm-even,R:dclm-odd,R:chat --tag chat-vs-dclm  # §1
    NEURON_LENS_MODEL=google/gemma-4-12B python scripts/checks/causal_lens.py measure --per-layer 24
    NEURON_LENS_MODEL=google/gemma-4-12B python scripts/checks/causal_lens.py score --lenses logit,hub
"""
import argparse
import json
import random
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from neuron_lens import arch
from neuron_lens.catalog import MODEL, column, documents, neuron_table
from neuron_lens.corpus import lens_docs
from neuron_lens.known import EXTERNAL, KNOWN
from neuron_lens.paths import CORPUS_DIR, DATA, SUMMARY
from neuron_lens.stats import pearson

SKIP_FIRST = 16
WINDOW = 8
DOSES = (1.0, 4.0)
PASS_Q = 0.99  # a read predicts its pole's effect past this quantile of the control's r
CAUSAL = DATA / "causal"


def random_neurons(n_layers: int, f: int, per_layer: int, seed: int = 0) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(l, n) for l in range(n_layers) for n in sorted(rng.sample(range(f), per_layer))]


def sample_neurons(n_layers: int, f: int, per_layer: int, seed: int = 0) -> list[tuple[int, int]]:
    """``per_layer`` random neurons of every layer, plus the known and outside-labelled ones (Qwen3-8B's)."""
    if MODEL != arch.QWEN:
        return random_neurons(n_layers, f, per_layer, seed)
    named = {(l, n) for l, n, _ in KNOWN.values()} | {(l, n) for _, _, l, n, _ in EXTERNAL}
    return sorted(set(random_neurons(n_layers, f, per_layer, seed)) | named)


def named_poles() -> dict[tuple[int, int, int], str]:
    """(layer, neuron, sign) -> label for the known and outside-labelled poles (both, if unsigned)."""
    if MODEL != arch.QWEN:
        return {}
    named = {(l, n, s): f"persona experiment: {name}" for name, (l, n, s) in KNOWN.items()}
    for source, label, l, n, s in EXTERNAL:
        for sign in ((s,) if s else (1, -1)):
            named.setdefault((l, n, sign), f"{source}: {label}")
    return named


def lsm_chunks(lm_head, h: torch.Tensor, chunk: int = 1024):
    """Log-softmax of ``lm_head(h)`` rows in fp32 (after Gemma's final softcap), ``chunk`` rows at a time."""
    for s in range(0, h.shape[0], chunk):
        yield s, torch.log_softmax(arch.cap(MODEL, lm_head(h[s:s + chunk]).float()), dim=-1)


def catalog_top(layers, f: int, k: int) -> dict:
    """Per layer: each pole's ``act_top`` ``[2, f]`` and its first ``k`` top firings (``top_doc``,
    ``top_pos``, ``top_val``, ``[2, f, k]``), from the catalog's ``acts`` tables (``+`` poles first)."""
    out = {}
    for layer in layers:
        t = pq.read_table(neuron_table("acts", layer), columns=["act_top", "top_doc", "top_pos", "top_val"])
        out[layer] = {"act_top": torch.from_numpy(t.column("act_top").to_numpy().copy()).view(2, f),
                      **{c: column(t, c, k).view(2, f, k) for c in ("top_doc", "top_pos", "top_val")}}
    return out


def measure(args) -> None:
    device = args.device
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    lm.requires_grad_(False)
    from neuron_lens.transport import decoder_stack
    model, lm_head = decoder_stack(lm), lm.lm_head
    gemma = arch.is_gemma4(args.model)
    L, f = len(model.layers), model.config.intermediate_size
    picks = sample_neurons(L, f, args.per_layer)
    if args.layers:
        picks = [(l, n) for l, n in picks if l in {int(x) for x in args.layers.split(',')}]
    K = args.top_contexts
    top = catalog_top(sorted({l for l, _ in picks}), f, K)
    tokens = documents(args.model)
    ctx = lens_docs(CORPUS_DIR / "act-dclm.jsonl", tok, 4 * args.contexts, args.ctx_len, args.ctx_len)[:args.contexts]
    ctx = torch.tensor(ctx, device=device)
    C, T = ctx.shape
    V = lm_head.weight.shape[0]
    P = len(picks)
    meta = {"model": args.model, "contexts_from": "catalog", "docs": len(tokens), "per_layer": args.per_layer,
            "contexts": C, "ctx_len": T, "doses": list(DOSES), "window": WINDOW, "top_contexts": K}
    path = CAUSAL / f"{arch.stem(args.model)}-causal.pt"
    partial = path.with_name(f"{path.stem}-partial.pt")
    if partial.exists():
        out = torch.load(partial, weights_only=False)
        assert out["meta"] == meta and out["picks"] == picks and torch.equal(out["contexts"], ctx.cpu()), \
            f"{partial} is another run's"
        print(f"resuming {partial}: steering done at {out['done']['steer']}, ablation at {out['done']['ablate']}",
              flush=True)
    else:
        out = {"meta": meta, "picks": picks, "contexts": ctx.cpu(),
               "alpha": torch.tensor([[top[l]["act_top"][p, n].clamp_min(0).item() for p in (0, 1)] for l, n in picks]),
               "steer": torch.zeros(P, 2, len(DOSES), V, dtype=torch.float16),
               "ablate_pos": torch.zeros(P, 2, V, dtype=torch.float16),
               "ablate_win": torch.zeros(P, 2, V, dtype=torch.float16),
               "dlogp_next": torch.full((P, 2, K), float("nan")),
               "fired": torch.zeros(P, 2, K),
               "top_doc": torch.full((P, 2, K), -1, dtype=torch.int32),
               "top_pos": torch.full((P, 2, K), -1, dtype=torch.int16),
               "done": {"steer": [], "ablate": []}}
    CAUSAL.mkdir(parents=True, exist_ok=True)
    saved = time.perf_counter()

    def checkpoint() -> None:
        nonlocal saved
        if time.perf_counter() - saved > 60 * args.save_minutes:
            torch.save(out, partial)
            saved = time.perf_counter()

    # Steering: conditions (pick, pole, dose), args.cond_batch per forward pass, one layer at a time.
    alpha, steer = out["alpha"], out["steer"]
    t0 = time.perf_counter()
    for layer in range(L):
        conds = [(i, p, di) for i, (l, n) in enumerate(picks) if l == layer
                 for p in (0, 1) for di in range(len(DOSES))]
        if not conds or layer in out["done"]["steer"]:
            continue
        blk = model.layers[layer]
        wd = blk.mlp.down_proj.weight  # [d, f]
        if gemma:  # what reaches the residual per unit activation: γ_post ⊙ W_down (its scale per position below)
            wd = (blk.post_feedforward_layernorm.weight.float()[:, None] * wd.float()).to(wd.dtype)
        for s in range(0, len(conds), args.cond_batch):
            part = conds[s:s + args.cond_batch]
            # Row 0 is unsteered: the reference, at the same batch shape as the steered rows.
            delta = torch.stack([torch.zeros_like(wd[:, 0])] + [
                (1 - 2 * p) * DOSES[di] * alpha[i, p].item() * wd[:, picks[i][1]]
                for i, p, di in part])  # [1 + n_c, d]
            rows = delta.repeat_interleave(C, 0)[:, None, :]
            posmask = (torch.arange(T, device=device) >= SKIP_FIRST)[None, :, None].to(rows.dtype)
            if gemma:  # after the post-feedforward norm, at its scale: the LN rule's write
                pf = blk.post_feedforward_layernorm
                hook = pf.register_forward_hook(lambda mod, a, o: o + rows * posmask * torch.rsqrt(
                    a[0].float().pow(2).mean(-1, keepdim=True) + arch.norm_eps(pf)).to(o.dtype))
            else:
                hook = blk.mlp.register_forward_hook(lambda mod, a, o: o + rows * posmask)
            with torch.inference_mode():
                hs = model(input_ids=ctx.repeat(len(delta), 1)).last_hidden_state[:, SKIP_FIRST:]
                hs = hs.reshape(len(delta), -1, hs.shape[-1])
                mean = torch.zeros(len(delta), V, device=device)
                for c in range(len(delta)):
                    for _, lsm in lsm_chunks(lm_head, hs[c]):
                        mean[c] += lsm.sum(0)
                mean /= hs.shape[1]
                for c, (i, p, di) in enumerate(part, 1):
                    steer[i, p, di] = (mean[c] - mean[0]).half().cpu()
            hook.remove()
        out["done"]["steer"].append(layer)
        checkpoint()
        print(f"{time.strftime('%H:%M:%S')} steer L{layer}: {len(conds)} conditions, "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    # Ablation in context: per (pick, pole), its top documents; clean and ablated rows side by side.
    t0 = time.perf_counter()
    for layer in range(L):
        if layer in out["done"]["ablate"] or not any(l == layer for l, _ in picks):
            continue
        jobs = []  # (pick index, pole, context slot, doc, pos)
        for i, (l, n) in enumerate(picks):
            if l != layer:
                continue
            for p in (0, 1):
                for j in range(K):
                    v = top[layer]["top_val"][p, n, j].item()
                    if v > 0:
                        d, pos = int(top[layer]["top_doc"][p, n, j]), int(top[layer]["top_pos"][p, n, j])
                        jobs.append((i, p, j, d, pos))
                        out["fired"][i, p, j], out["top_doc"][i, p, j], out["top_pos"][i, p, j] = v, d, pos
        sums_pos, sums_win, counts = {}, {}, {}
        for s in range(0, len(jobs), args.ctx_batch):
            part = jobs[s:s + args.ctx_batch]
            seqs = [tokens[d][:pos + 1 + WINDOW].long() for _, _, _, d, pos in part]
            Tm = max(len(q) for q in seqs)
            ids = torch.zeros(2 * len(part), Tm, dtype=torch.long)
            attn = torch.zeros_like(ids)
            for r, q in enumerate(seqs):
                ids[2 * r, :len(q)] = ids[2 * r + 1, :len(q)] = q
                attn[2 * r, :len(q)] = attn[2 * r + 1, :len(q)] = 1
            rows = torch.arange(1, 2 * len(part), 2, device=device)
            pos_t = torch.tensor([pos for *_, pos in part], device=device)
            neu_t = torch.tensor([picks[i][1] for i, *_ in part], device=device)

            def ablate(mod, a):
                x = a[0].clone()
                x[rows, pos_t, neu_t] = 0
                return (x,)
            hook = model.layers[layer].mlp.down_proj.register_forward_pre_hook(ablate)
            with torch.inference_mode():
                hs = model(input_ids=ids.to(device), attention_mask=attn.to(device)).last_hidden_state
                for r, (i, p, j, d, pos) in enumerate(part):
                    end = min(pos + 1 + WINDOW, len(seqs[r]))
                    lc = torch.log_softmax(arch.cap(MODEL, lm_head(hs[2 * r, pos:end]).float()), -1)
                    la = torch.log_softmax(arch.cap(MODEL, lm_head(hs[2 * r + 1, pos:end]).float()), -1)
                    diff = lc - la
                    sums_pos[(i, p)] = sums_pos.get((i, p), 0) + diff[0]
                    sums_win[(i, p)] = sums_win.get((i, p), 0) + diff.sum(0)
                    counts[(i, p)] = counts.get((i, p), 0) + 1
                    if pos + 1 < len(seqs[r]):
                        out["dlogp_next"][i, p, j] = diff[0, seqs[r][pos + 1]].item()
            hook.remove()
        for key, c in counts.items():
            out["ablate_pos"][key] = (sums_pos[key] / c).half().cpu()
            out["ablate_win"][key] = (sums_win[key] / c).half().cpu()
        out["done"]["ablate"].append(layer)
        checkpoint()
        print(f"{time.strftime('%H:%M:%S')} ablate L{layer}: {len(jobs)} contexts, "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
    del out["done"]
    torch.save(out, path)
    partial.unlink(missing_ok=True)
    print(f"-> {path}", flush=True)


def overlap(a: torch.Tensor, b: torch.Tensor, k: int = 10) -> torch.Tensor:
    ta, tb = a.topk(k, dim=1).indices, b.topk(k, dim=1).indices
    return torch.tensor([len(set(x.tolist()) & set(y.tolist())) / k for x, y in zip(ta, tb)])


def score(args) -> None:
    from neuron_lens.lenses import load_lens
    from neuron_lens.readout import rms_norm_columns, transport
    from neuron_lens.neurons import directions
    from neuron_lens.weights import tensors

    saved = torch.load(CAUSAL / f"{arch.stem(args.model)}-causal.pt", weights_only=False)
    picks = saved["picks"]
    tok = AutoTokenizer.from_pretrained(args.model)
    nv = len(tok)  # rows past the tokenizer are padding
    # Undertrained tokens (seen under 5 times in the activation corpora) are left out: their
    # logits swing widely under any push and would dominate the correlation.
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    keep_vocab = (torch.load(arch.counts_path(args.model), weights_only=False)["counts"][:nv] >= 5).to(dev)
    nm = arch.names(args.model)
    head = tensors(args.model, sorted({nm["unembed"], nm["norm"]}))
    unembed = head[nm["unembed"]][:nv].to(dev, torch.float32)
    gamma = head[nm["norm"]].to(dev, torch.float32)
    specs = args.lenses.split(",")
    lenses = {s: load_lens(s, args.model) for s in specs}
    cfg = AutoConfig.from_pretrained(args.model).get_text_config()
    rand = set(random_neurons(cfg.num_hidden_layers, cfg.intermediate_size, saved["meta"]["per_layer"]))
    named = named_poles()
    named_r = {}  # (pole, target, lens) -> (r, top-10 overlap)
    targets = {"steer 1x": saved["steer"][:, :, 0], "steer 4x": saved["steer"][:, :, 1],
               "ablate pos": saved["ablate_pos"], "ablate win": saved["ablate_win"]}
    fired = (saved["fired"] > 0).any(-1)  # [picks, 2]: the pole has top contexts
    layers = sorted({l for l, _ in picks})
    rows = []  # (layer, target, lens, matched r, control r, top-10 overlap), each over the layer's random poles
    for layer in layers:
        idx = [i for i, (l, _) in enumerate(picks) if l == layer]
        is_rand = torch.tensor([picks[i] in rand for i in idx])
        other = torch.arange(len(idx))  # each pick's control: the next random pick of the layer (named: next named)
        for group in (is_rand, ~is_rand):
            other[group] = other[group].roll(1)
        wd = directions(args.model, layer, dev)["write"]  # the write as the residual sees it (Gemma: s · γ_post ⊙ W_down)
        cols = wd[:, [picks[i][1] for i in idx]]
        for s in specs:
            jac = None if lenses[s] is None else lenses[s].get(layer)
            lg = (unembed @ rms_norm_columns(transport(cols, jac), gamma)).T  # [n, V]
            pred = torch.cat([lg, -lg])  # poles +, −
            ctrl = torch.cat([lg[other], -lg[other]])
            for name, t in targets.items():
                y = torch.cat([t[idx, 0, :nv], t[idx, 1, :nv]]).to(dev, torch.float32)
                keep = torch.ones(len(y), dtype=torch.bool) if name.startswith("steer") \
                    else torch.cat([fired[idx, 0], fired[idx, 1]])
                keep &= y.abs().sum(1).cpu() > 0
                pv, cv, yv = pred[:, keep_vocab], ctrl[:, keep_vocab], y[:, keep_vocab]
                r, rc, ov = pearson(pv, yv).cpu(), pearson(cv, yv).cpu(), overlap(pv, yv)
                k = keep & torch.cat([is_rand, is_rand])  # the per-layer statistics: random picks only
                if k.any():
                    rows.append((layer, name, s, r[k], rc[k], ov[k]))
                for pi, sign in enumerate((1, -1)):
                    for j, i in enumerate(idx):
                        pole = (layer, picks[i][1], sign)
                        if pole in named and keep[pi * len(idx) + j]:
                            named_r[(pole, name, s)] = (r[pi * len(idx) + j].item(), ov[pi * len(idx) + j].item())
    pooled = {}
    for _, name, s, _, rc, _ in rows:
        pooled.setdefault((name, s), []).append(rc)
    thr = {key: torch.cat(v).quantile(PASS_Q).item() for key, v in pooled.items()}
    stats = {(l, name, s): {"r_q25": r.quantile(0.25).item(), "r_med": r.median().item(), "r_q75": r.quantile(0.75).item(),
                            "control_med": rc.median().item(), "overlap": ov.mean().item(),
                            "pass": (r > thr[name, s]).float().mean().item(),
                            "pass_control": (rc > thr[name, s]).float().mean().item()}
             for l, name, s, r, rc, ov in rows}
    lines = [f"causal check of lens reads, {args.model}: {saved['meta']}",
             "median Pearson r over the vocabulary (matched / control: next sampled neuron's read), "
             "mean top-10 overlap; random picks only (named poles listed after)"]
    for name in targets:
        lines.append(f"\n{name}")
        lines.append("layer " + " ".join(f"{s:>22s}" for s in specs))
        for layer in layers:
            cells = {s: stats.get((layer, name, s)) for s in specs}
            lines.append(f"L{layer:<4d} " + " ".join(
                f"{c['r_med']:6.2f} /{c['control_med']:5.2f} ov {c['overlap']:.2f}" if c else " " * 22
                for c in cells.values()))
    for name in ("steer 1x", "steer 4x"):
        lines.append(f"\n{name}: share of poles whose read predicts it (r past the control's {PASS_Q:.0%} point, pooled "
                     f"over layers: " + ", ".join(f"{s} {thr[name, s]:.2f}" for s in specs) + "; control pairs in brackets)")
        lines.append("layer " + " ".join(f"{s:>14s}" for s in specs))
        for layer in layers:
            cells = [stats.get((layer, name, s)) for s in specs]
            lines.append(f"L{layer:<4d} " + " ".join(
                f"{c['pass']:6.2f} [{c['pass_control']:.2f}]" if c else " " * 14 for c in cells))
    lines.append("\nnamed poles: Pearson r (top-10 overlap) per lens, steer 4x | ablate win")
    for pole in sorted({p for p, _, _ in named_r}):
        l, n, sign = pole
        cells = []
        for s in specs:
            a, b = named_r.get((pole, "steer 4x", s)), named_r.get((pole, "ablate win", s))
            cells.append(f"{s} {a[0]:.2f} ({a[1]:.1f})" + (f" | {b[0]:.2f} ({b[1]:.1f})" if b else " | —"))
        lines.append(f"  L{l} n{n} {'+' if sign > 0 else '−'} [{named[pole]}]  " + "   ".join(cells))
    text = "\n".join(lines)
    print(text)
    stem = f"causal_lens_{args.tag + '_' if args.tag else ''}{arch.stem(args.model)}"
    (SUMMARY / f"{stem}.log").write_text(text + "\n")
    table = {name: {s: {"layer": [], "pass_threshold": round(thr[name, s], 4)} for s in specs} for name in targets}
    for (l, name, s), st in stats.items():
        t = table[name][s]
        t["layer"].append(l)
        for k, v in st.items():
            t.setdefault(k, []).append(round(v, 4))
    (SUMMARY / f"{stem}.json").write_text(json.dumps({"meta": saved["meta"], "lenses": specs, "targets": table}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("measure", "score"))
    ap.add_argument("--per-layer", type=int, default=32)
    ap.add_argument("--contexts", type=int, default=16)
    ap.add_argument("--ctx-len", type=int, default=64)
    ap.add_argument("--top-contexts", type=int, default=8)
    ap.add_argument("--cond-batch", type=int, default=16)
    ap.add_argument("--ctx-batch", type=int, default=64)
    ap.add_argument("--save-minutes", type=float, default=10)
    ap.add_argument("--lenses", default="logit,J,R:dclm" if MODEL == arch.QWEN else "logit,hub")
    ap.add_argument("--layers", default=None, help="measure only these layers (a,b,c)")
    ap.add_argument("--tag", default="", help="score: name the summary causal_lens_<tag>_<model>")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    args = ap.parse_args()
    args.model = MODEL
    measure(args) if args.what == "measure" else score(args)


if __name__ == "__main__":
    main()
