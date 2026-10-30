"""Why the audio-matched read fails: the RelP derivative against the true one, on Gemma 4 12B
cut to its first ``--layers`` layers (exact for poles there), in fp32.

For the media_reads.py panel's poles below ``--layers``, on ``--docs`` fitting documents of a
modality (media_lens.py's: ESC-50 clips, COCO images), the matched read ``u`` (the pole's read
covector at its layer's MLP input, averaged over the slots, pulled to the embeddings and summed
over the slots) under:
- ``relp``: ``input_reads_doc`` (the hand-written maps under the RelP rules);
- ``relp_autograd``: autograd under ``relp_rules`` (must equal ``relp``: a check on real weights);
- ``true``: autograd through the model as it is (true derivatives everywhere);
- ``relp_attn_true``: RelP except that the post-attention norm keeps its true derivative.
Then each read on held-out documents as media_lens.py's B (its change of every panel pole's
read projection, ranked), and per layer the size of the attention branch at the slots against
text positions (the RMS of ``o_proj``'s output, which the post-attention norm divides by).

    python scripts/checks/rule_diag.py --modality audio --layers 12 --docs 16 --test 16
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from media_lens import EPS, SLOTS, images, processed, sounds, stack, to  # noqa: E402
from media_reads import MODEL, audio_reads, pixel_reads  # noqa: E402

from neuron_lens.gemma4 import multimodal_embeds  # noqa: E402
from neuron_lens.paths import DATA  # noqa: E402
from neuron_lens.rules import _norm, relp_rules  # noqa: E402
from neuron_lens.transport import SKIP_FIRST, decoder_stack, input_reads_doc  # noqa: E402

OUT = DATA / "summary" / "gemma4_media"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="audio")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--docs", type=int, default=16)
    ap.add_argument("--test", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eps-scale", default="1", help="comma list: ε multipliers for the held-out test")
    args = ap.parse_args()
    dev, t0 = args.device, time.perf_counter()
    config = AutoConfig.from_pretrained(MODEL)
    tc = config.text_config
    tc.num_hidden_layers, tc.layer_types = args.layers, tc.layer_types[:args.layers]
    lm = AutoModelForCausalLM.from_pretrained(MODEL, config=config, dtype=torch.float32, attn_implementation="eager")
    lm = lm.to(dev).eval().requires_grad_(False)
    model = decoder_stack(lm)
    proc = AutoProcessor.from_pretrained(MODEL)
    panel = torch.load(OUT / "reads.pt", weights_only=False)
    sel = [i for i, (L, _, _) in enumerate(panel["poles"]) if L < args.layers]
    poles = [panel["poles"][i] for i in sel]
    cov = panel["cov"][sel]
    layers = [L for L, _, _ in poles]
    P = len(poles)
    slot_id = proc.image_token_id if args.modality == "image" else proc.audio_token_id
    key = "pixel_values" if args.modality == "image" else "input_features"
    n = args.docs + args.test
    items = images(n) if args.modality == "image" else sounds(384 + args.test, "esc")
    items = items[:args.docs] + items[-args.test:]  # media_lens's fitting documents first, its held-out last
    docs = [processed(proc, args.modality, x) for x in items]
    print(f"loaded in {time.perf_counter() - t0:.0f}s: {args.layers} layers fp32, {P} poles, {len(docs)} {args.modality} docs",
          flush=True)

    def mids_hook(store):
        return [model.layers[L].pre_feedforward_layernorm.register_forward_pre_hook(
            lambda _m, a, L=L: store.__setitem__(L, a[0])) for L in sorted(set(layers))]

    def autograd_reads(d, mode):
        emb = multimodal_embeds(lm, d).detach().clone().requires_grad_(True)
        ids = d["input_ids"][0]
        slots = ids == slot_id
        block = slots if args.modality == "image" else None
        T = len(ids) - 1  # linearize drops the last position; so do we
        causal = torch.full((T, T), float("-inf"), device=dev).triu(1)
        sliding = causal if block is None else causal.masked_fill(block[:T, None] & block[None, :T], 0.0)
        masks = {"full_attention": causal[None, None], "sliding_attention": sliding[None, None]}
        store = {}
        hs = mids_hook(store)
        patched = []
        if mode == "relp_attn_true":  # the post-attention norm back to its true derivative
            for layer in model.layers:
                patched.append(layer.post_attention_layernorm)
        with torch.enable_grad():
            ctx = relp_rules(model) if mode.startswith("relp") else torch.enable_grad()
            with ctx:
                if mode == "relp_attn_true":
                    for norm in patched:
                        norm.forward = norm._rules_orig_forward
                model(inputs_embeds=emb[None, :T], attention_mask=masks)
                if mode == "relp_attn_true":
                    for norm in patched:
                        norm.forward = types.MethodType(_norm, norm)
                rows = slots[:T].clone()
                rows[:SKIP_FIRST] = False
                out = []
                for i, L in enumerate(layers):
                    v = (store[L][0, rows] @ cov[i].to(dev)).mean()
                    g, = torch.autograd.grad(v, emb, retain_graph=i < P - 1)
                    out.append(g[:T][rows].sum(0))
        for h in hs:
            h.remove()
        return torch.stack(out).detach()

    reads = {k: torch.zeros(P, cov.shape[1], device=dev) for k in ("relp", "relp_autograd", "true", "relp_attn_true")}
    fit = docs[:args.docs]
    for d in fit:
        d = to(d, dev)
        d = {k: (v.float() if v.is_floating_point() else v) for k, v in d.items()}
        ids = d["input_ids"][0]
        rows = (ids == slot_id)[SKIP_FIRST:len(ids) - 1]
        emb = multimodal_embeds(lm, d)
        block = (ids == slot_id) if args.modality == "image" else None
        reads["relp"] += input_reads_doc(model, ids, cov, layers, seed_rows=rows, source_rows=rows, embeds=emb, block=block)
        for mode in ("relp_autograd", "true", "relp_attn_true"):
            reads[mode] += autograd_reads(d, mode)
    reads = {k: v / len(fit) for k, v in reads.items()}
    rel = ((reads["relp"] - reads["relp_autograd"]).norm(dim=1) / reads["relp_autograd"].norm(dim=1)).max()
    print(f"check: hand-written maps vs autograd under RelP on real weights, max rel err {rel:.1e}", flush=True)
    cosine = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=1)
    report = {"modality": args.modality, "layers": args.layers, "check_rel_err": float(rel), "cos": {}, "B": {}, "attn_rms": {}}
    for a, b in (("relp", "true"), ("relp_attn_true", "true"), ("relp", "relp_attn_true")):
        c = cosine(reads[a], reads[b]).cpu()
        by = {L: round(float(c[torch.tensor(layers) == L].median()), 3) for L in sorted(set(layers))}
        report["cos"][f"{a}~{b}"] = by
        print(f"cos({a}, {b}) by layer: " + " ".join(f"L{L} {v:.2f}" for L, v in by.items()), flush=True)

    # Attention branch size at the slots and at the text positions (first fitting doc).
    d = to(fit[0], dev)
    d = {k: (v.float() if v.is_floating_point() else v) for k, v in d.items()}
    seen = {}
    hs = [layer.post_attention_layernorm.register_forward_pre_hook(lambda _m, a, i=i: seen.__setitem__(i, a[0][0]))
          for i, layer in enumerate(model.layers)]
    with torch.no_grad():
        lm(**d)
    for h in hs:
        h.remove()
    ids = d["input_ids"][0]
    slots, text = ids == slot_id, (ids != slot_id) & (torch.arange(len(ids), device=dev) >= SKIP_FIRST)
    for i in range(args.layers):
        rms = seen[i].pow(2).mean(-1).sqrt()
        report["attn_rms"][i] = (float(rms[slots].median()), float(rms[text].median()))
    print("attention-branch RMS (slots / text) by layer: " + " ".join(
        f"L{i} {a:.2g}/{b:.2g}" for i, (a, b) in report["attn_rms"].items()), flush=True)

    # Held-out: each read added to the slots, every pole's projection change ranked.
    caught = {}
    by_layer = {}
    for i, (L, nn, _) in enumerate(poles):
        by_layer.setdefault(L, []).append(i)
    hooks = []
    for L, ix in by_layer.items():
        idx = torch.tensor([poles[i][1] for i in ix], device=dev)
        mlp = model.layers[L].mlp
        hooks.append(mlp.gate_proj.register_forward_hook(lambda _m, _a, o, L=L, idx=idx: caught.__setitem__(("g", L), o[..., idx])))
        hooks.append(mlp.up_proj.register_forward_hook(lambda _m, _a, o, L=L, idx=idx: caught.__setitem__(("u", L), o[..., idx])))
    order = [i for L in by_layer for i in by_layer[L]]
    inv = torch.empty(P, dtype=torch.long)
    inv[torch.tensor(order)] = torch.arange(P)
    g = [model.layers[L].pre_feedforward_layernorm.weight.float() for L, _, _ in poles]
    na = torch.tensor([float((model.layers[L].mlp.gate_proj.weight[nn].float() * g[i]).norm()) for i, (L, nn, _) in enumerate(poles)], device=dev)
    nb = torch.tensor([float((model.layers[L].mlp.up_proj.weight[nn].float() * g[i]).norm()) for i, (L, nn, _) in enumerate(poles)], device=dev)
    sg = torch.tensor([s for _, _, s in poles], dtype=torch.float32, device=dev)

    def proj(mask):
        m = mask.float() / mask.float().sum(1, keepdim=True)
        get = lambda k: torch.cat([caught[(k, L)] for L in by_layer], -1)[..., inv.to(dev)]
        return ((get("g") / na + sg * get("u") / nb) * m[..., None]).sum(1)

    test = to(stack(docs[args.docs:]), dev)
    test = {k: (v.float() if v.is_floating_point() else v) for k, v in test.items()}
    mask = test["input_ids"] == slot_id
    with torch.no_grad():
        lm(**test)
        base = proj(mask)
    rng = np.random.default_rng(0)
    for scale in (float(x) for x in args.eps_scale.split(",")):
      for kind, u in reads.items():
        R = (pixel_reads(lm, u.cpu()).reshape(P, -1) if args.modality == "image" else audio_reads(lm, u.cpu())).to(dev)
        M = torch.zeros(P, P)
        for p in range(P):
            x = test[key]
            r = R[p][None, None]
            if args.modality == "image":
                y = x.clone()
                y[:, :SLOTS] = (x[:, :SLOTS] + scale * EPS["image"] * r / r.abs().max()).clamp(0, 1)
            else:
                y = x + scale * EPS["audio"] * x.pow(2).mean(-1, keepdim=True).sqrt() * r / r.pow(2).mean().sqrt()
            with torch.no_grad():
                lm(**{**test, key: y})
                M[:, p] = (proj(mask) - base).mean(0).cpu() / scale
        rank = lambda own: torch.tensor([float((M[i] < M[i, own[i]]).double().mean() * P / (P - 1)) for i in range(P)])
        real = rank(list(range(P)))
        null = torch.cat([rank(list(rng.permutation(P))) for _ in range(10)])
        by = {L: round(float(real[torch.tensor(layers) == L].median()), 2) for L in sorted(set(layers))}
        report["B"][f"{kind}@{scale}"] = {"median": float(real.median()), "above_null_p95": float((real > null.quantile(0.95)).double().mean()),
                                          "by_layer": by}
        print(f"held-out ε×{scale:<5} {kind:15s}: median rank {real.median():.3f}, above null p95 "
              f"{(real > null.quantile(0.95)).double().mean():.0%}; by layer " + " ".join(f"L{L} {v:.2f}" for L, v in by.items()),
              flush=True)
    for h in hooks:
        h.remove()
    (OUT / f"rule_diag_{args.modality}.json").write_text(json.dumps(report, indent=1))
    print(f"done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
