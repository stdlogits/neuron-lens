"""Image and audio reads of Gemma 4 12B neurons from image and audio contexts, held-out tests of
every kind of read, activation maximisation from them, and how much of an image or a sound
each layer keeps.

The panel and the text-context reads are media_reads.py's (``data/summary/gemma4_media/reads.pt``).

- **A. Matched reads.** The S-lens read pulled down (``input_reads_doc``) on natural documents
  of the modality: COCO images (an 8×8 grid of 48-pixel patches, centre-cropped to 384 pixels)
  and audio (ESC-50 sounds and LibriSpeech speech, 2.56 s, 64 frames), each in a short text
  prompt; seeded at the image (audio) positions and summed over them, with the image block's
  two-way attention in sliding layers as the model has it. ``--fit`` documents a modality.
- **B. Held-out test.** For each pole's read (text-context lens, no lens, matched lens,
  optimised), the read is added to every patch (frame) of ``--test`` held-out documents
  (images: ``ε`` = 0.08 of the pixel range at the read's peak; audio: 0.3 of each frame's RMS),
  and every pole's change in read projection ``c · x`` and in activation, averaged over the
  slots and documents, is ranked: a pole's change under its own read among its changes under
  the 191 others (1 = the most, 0.5 = chance). The null permutes which read is a pole's own.
- **C. Optimised reads.** Per pole, the perturbation (same size limit) that most raises its mean
  activation on the fitting documents, by gradient ascent from the matched read (``--steps``).
- **D. Recovery.** Ridge decoders from each layer's residual at a patch (frame) to that patch's
  pixels (frame's samples), fitted on the fitting documents and scored on the held-out ones
  (R², per patch normalized as the embedder's first norm sees it, and raw).

    python scripts/checks/media_lens.py --device cuda [--fit 128] [--test 32]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from media_reads import MODEL, audio_reads, pixel_reads  # noqa: E402

from neuron_lens.gemma4 import multimodal_embeds  # noqa: E402
from neuron_lens.media import PROMPT, SIDE, images, processed, sounds  # noqa: E402,F401
from neuron_lens.paths import DATA  # noqa: E402
from neuron_lens.transport import SKIP_FIRST, decoder_stack, input_reads_doc  # noqa: E402

OUT = DATA / "summary" / "gemma4_media"
SLOTS = 64
EPS = {"image": 0.08, "audio": 0.3}
RECON_LAYERS = [0, 1, 4, 7, 13, 25, 37, 48]  # hidden_states index: 0 is the embeddings, i the output of layer i-1


def stack(batch):
    return {k: torch.cat([b[k] for b in batch]) for k in batch[0]}


def to(inputs, device):
    return {k: v.to(device, torch.bfloat16) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fit", type=int, default=128)
    ap.add_argument("--test", type=int, default=32)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--modalities", default="image,audio")
    ap.add_argument("--skip", default="", help="parts to skip: A,B,C,D,F")
    ap.add_argument("--audio-source", default="mix", choices=("mix", "esc"))
    ap.add_argument("--skip-done", action="store_true", help="B: skip read kinds already in the report")
    ap.add_argument("--tag", default="", help="suffix for saved reads and the report's keys")
    ap.add_argument("--free-from", type=int, default=12, help="F: free optimisation for poles from this layer")
    ap.add_argument("--free-steps", type=int, default=40)
    ap.add_argument("--eps-scale", type=float, default=1.0, help="multiplies ε (the perturbation's size)")
    args = ap.parse_args()
    skip = set(args.skip.split(","))
    t0 = time.perf_counter()
    dev = args.device
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager")
    lm = lm.to(dev).eval().requires_grad_(False)
    model = decoder_stack(lm)
    proc = AutoProcessor.from_pretrained(MODEL)
    panel = torch.load(OUT / "reads.pt", weights_only=False)
    poles, cov, u_text = panel["poles"], panel["cov"], panel["u_lens"]
    P = len(poles)
    layers = [L for L, _, _ in poles]
    norms = []
    for L, n, s in poles:
        g = model.layers[L].pre_feedforward_layernorm.weight.float()
        norms.append((float((model.layers[L].mlp.gate_proj.weight[n].float() * g).norm()),
                      float((model.layers[L].mlp.up_proj.weight[n].float() * g).norm())))
    na, nb = (torch.tensor(x) for x in zip(*norms))
    sign = torch.tensor([s for _, _, s in poles], dtype=torch.float32)
    print(f"loaded in {time.perf_counter() - t0:.0f}s; panel of {P} poles", flush=True)

    # Hooks: gate, up and the activation of every panel neuron, kept with the graph (for C).
    caught = {}
    by_layer = {}
    for i, (L, n, _) in enumerate(poles):
        by_layer.setdefault(L, []).append(i)
    idx = {L: torch.tensor([poles[i][1] for i in ix], device=dev) for L, ix in by_layer.items()}
    hooks = []
    for L in by_layer:
        mlp = model.layers[L].mlp
        hooks.append(mlp.gate_proj.register_forward_hook(lambda _m, _a, o, L=L: caught.__setitem__(("gate", L), o[..., idx[L]])))
        hooks.append(mlp.up_proj.register_forward_hook(lambda _m, _a, o, L=L: caught.__setitem__(("up", L), o[..., idx[L]])))
        hooks.append(mlp.down_proj.register_forward_pre_hook(lambda _m, a, L=L: caught.__setitem__(("act", L), a[0][..., idx[L]])))
    order = torch.tensor([i for L in by_layer for i in by_layer[L]])
    inv = torch.empty_like(order)
    inv[order] = torch.arange(P)

    def measures(slot_mask):
        """{'act', 'proj'}: [B, P] each pole's signed activation and read projection, mean over slots."""
        m = slot_mask.to(dev).float()
        m = m / m.sum(1, keepdim=True)
        get = lambda k: torch.cat([caught[(k, L)].float() for L in by_layer], -1)[..., inv.to(dev)]  # [B, T, P]
        s = sign.to(dev)
        act = s * get("act")
        proj = get("gate") / na.to(dev) + s * get("up") / nb.to(dev)
        return {k: (v * m[..., None]).sum(1) for k, v in (("act", act), ("proj", proj))}

    report = json.loads((OUT / "media_lens.json").read_text()) if (OUT / "media_lens.json").exists() else {}
    for modality in args.modalities.split(","):
        slot_id = proc.image_token_id if modality == "image" else proc.audio_token_id
        key = "pixel_values" if modality == "image" else "input_features"
        t1 = time.perf_counter()
        items = images(args.fit + args.test) if modality == "image" else sounds(args.fit + args.test, args.audio_source)
        docs = [processed(proc, modality, x) for x in items]
        fit, test = docs[:args.fit], docs[args.fit:]
        ids0 = docs[0]["input_ids"][0]
        first = int((ids0 == slot_id).nonzero()[0])
        assert first >= SKIP_FIRST and int((ids0 == slot_id).sum()) == SLOTS, (first, int((ids0 == slot_id).sum()))
        print(f"{modality}: {len(docs)} documents of {len(ids0)} tokens, slots from {first} "
              f"[{time.perf_counter() - t1:.0f}s]", flush=True)
        rep = report.setdefault(modality + args.tag, {})
        # multimodal_embeds must be what the model feeds its first layer.
        seen = {}
        hk = model.layers[0].register_forward_pre_hook(lambda _m, a, kw: seen.__setitem__(0, (a[0] if a else kw["hidden_states"])[0]),
                                                        with_kwargs=True)
        d0 = to(docs[0], dev)
        with torch.no_grad():
            lm(**d0)
        hk.remove()
        mine = multimodal_embeds(lm, d0)
        assert torch.equal(mine, seen[0]), float((mine.float() - seen[0].float()).abs().max())

        # A. Matched reads.
        path = OUT / f"matched_{modality}{args.tag}.pt"
        if "A" not in skip or not path.exists():
            halves = [torch.zeros(P, cov.shape[1]), torch.zeros(P, cov.shape[1])]
            for j, d in enumerate(fit):
                d = to(d, dev)
                ids = d["input_ids"][0]
                emb = multimodal_embeds(lm, d)
                block = (ids == slot_id) if modality == "image" else None
                rows = (ids == slot_id)[SKIP_FIRST:len(ids) - 1]
                halves[j % 2] += input_reads_doc(model, ids, cov, layers, seed_rows=rows, source_rows=rows,
                                                 embeds=emb, block=block).cpu()
            even, odd = (h / (len(fit) / 2) for h in halves)
            u = (even + odd) / 2
            sc = torch.nn.functional.cosine_similarity(even, odd, dim=1)
            torch.save({"u": u, "even": even, "odd": odd}, path)
            rep["A_split_cos_median"] = float(sc.median())
            print(f"A {modality}: matched reads from {len(fit)} documents, split-half cosine median "
                  f"{sc.median():.3f} (10th pct {sc.quantile(0.1):.3f}); cosine to the text-context read median "
                  f"{torch.nn.functional.cosine_similarity(u, u_text, dim=1).median():.3f} "
                  f"[{time.perf_counter() - t1:.0f}s]", flush=True)
        u_matched = torch.load(path)["u"]
        to_input = (lambda u: pixel_reads(lm, u).reshape(len(u), -1)) if modality == "image" else \
            (lambda u: audio_reads(lm, u))
        reads = {"text_lens": to_input(u_text), "no_lens": to_input(cov), "matched": to_input(u_matched)}

        def perturb(x, r):
            """Add read(s) ``r`` to the slots of inputs ``x`` (batch-aligned), at ε: ``[B, D]``, one
            read in every slot, or ``[B, SLOTS, D]``, one per slot. Images: ε at the read's peak
            (per slot); audio: ε of each frame's RMS."""
            r = r[:, None] if r.dim() == 2 else r
            if modality == "image":
                y = x.clone()
                y[:, :SLOTS] = (x[:, :SLOTS] + args.eps_scale * EPS["image"] * r / r.abs().amax(-1, keepdim=True)).clamp(0, 1)
                return y
            rms = x.pow(2).mean(-1, keepdim=True).sqrt()
            rn = r / r.pow(2).mean(-1, keepdim=True).sqrt()
            return x + args.eps_scale * EPS["audio"] * rms * rn

        # C. Optimised reads (from the matched ones), on the fitting documents.
        path_c = OUT / f"optimised_{modality}{args.tag}.pt"
        if "C" not in skip:
            t2 = time.perf_counter()
            opt = torch.zeros_like(reads["matched"])
            G, B = 2, 2  # poles per step, documents per pole per step (beside a fit on the same GPU)
            for g0 in range(0, P, G):
                ps = list(range(g0, min(P, g0 + G)))
                r0 = reads["matched"][ps].to(dev)
                scale = r0.abs().amax(-1, keepdim=True) if modality == "image" else r0.pow(2).mean(-1, keepdim=True).sqrt()
                w = (r0 / scale / (1.0 if modality == "image" else 3.0)).clamp(-0.95, 0.95).atanh().clone().requires_grad_(True)
                adam = torch.optim.Adam([w], lr=0.1)
                for step in range(args.steps):
                    pick = np.random.default_rng(step + g0).choice(len(fit), B, replace=False)
                    batch = stack([fit[k] for k in pick for _ in ps])
                    r = torch.tanh(w)
                    r = r.repeat(B, 1) if modality == "image" else (3.0 * r).repeat(B, 1)
                    inp = to(batch, dev)
                    x = inp[key].float()
                    inp[key] = perturb(x, r).to(torch.bfloat16)
                    with torch.enable_grad():
                        lm(**inp)
                        m = measures(inp["input_ids"] == slot_id)["act"]  # [B·G, P]
                        own = torch.tensor(ps * B, device=dev)
                        loss = -m[torch.arange(len(own), device=dev), own].mean()
                        adam.zero_grad()
                        loss.backward()
                    adam.step()
                with torch.no_grad():
                    opt[ps] = (torch.tanh(w) * (1.0 if modality == "image" else 3.0)).cpu()
                caught.clear()
            torch.save({"r": opt}, path_c)
            print(f"C {modality}: optimised reads for {P} poles, {args.steps} steps each "
                  f"[{time.perf_counter() - t2:.0f}s]", flush=True)
        if path_c.exists():
            reads["optimised"] = torch.load(path_c)["r"]

        # F. Free optimisation: a different perturbation in every slot, from the optimised read
        # tiled, for the poles from --free-from (whole images or clips, not one texture).
        path_f = OUT / f"free_{modality}{args.tag}.pt"
        deep = [i for i, L in enumerate(layers) if L >= args.free_from]
        if "F" not in skip and "optimised" in reads:
            t2 = time.perf_counter()
            free = reads["optimised"][:, None].repeat(1, SLOTS, 1).clone()
            G, B = 2, 2
            for g0 in range(0, len(deep), G):
                ps = deep[g0:g0 + G]
                r0 = free[ps].to(dev)
                r0 = r0 / (r0.abs().amax(-1, keepdim=True) if modality == "image" else 3 * r0.pow(2).mean(-1, keepdim=True).sqrt())
                w = r0.clamp(-0.95, 0.95).atanh().clone().requires_grad_(True)
                adam = torch.optim.Adam([w], lr=0.05)
                for step in range(args.free_steps):
                    pick = np.random.default_rng(10_000 + step + g0).choice(len(fit), B, replace=False)
                    inp = to(stack([fit[k] for k in pick for _ in ps]), dev)
                    r = torch.tanh(w).repeat(B, 1, 1)
                    inp[key] = perturb(inp[key].float(), r).to(torch.bfloat16)
                    with torch.enable_grad():
                        lm(**inp)
                        m = measures(inp["input_ids"] == slot_id)["act"]
                        own = torch.tensor(ps * B, device=dev)
                        loss = -m[torch.arange(len(own), device=dev), own].mean()
                        adam.zero_grad()
                        loss.backward()
                    adam.step()
                with torch.no_grad():
                    free[ps] = torch.tanh(w).cpu()
                caught.clear()
            torch.save({"r": free, "poles": deep}, path_f)
            print(f"F {modality}: free optimisation for {len(deep)} poles from layer {args.free_from}, "
                  f"{args.free_steps} steps [{time.perf_counter() - t2:.0f}s]", flush=True)
        if path_f.exists():
            free = torch.load(path_f)["r"]
            if "optimised" in reads:  # poles below --free-from keep their tiled read
                keep_tiled = torch.tensor([L < args.free_from for L in layers])
                free[keep_tiled] = reads["optimised"][keep_tiled][:, None].expand(-1, SLOTS, -1)
            reads["free"] = free

        # B. Held-out test of every kind of read.
        if "B" not in skip:
            t2 = time.perf_counter()
            base_in = to(stack(test), dev)
            slot_mask = base_in["input_ids"] == slot_id
            with torch.no_grad():
                lm(**base_in)
                base = measures(slot_mask)  # [Ntest, P]
            rng = np.random.default_rng(0)
            rep.setdefault("B", {})
            for kind, R in reads.items():
                if args.skip_done and f"{kind}_act" in rep.get("B", {}):
                    continue
                D = {"act": torch.zeros(P, P), "proj": torch.zeros(P, P)}  # [pole measured, read]
                for p in range(P):
                    inp = dict(base_in)
                    rp = R[p:p + 1].to(dev)
                    inp[key] = perturb(base_in[key].float(), rp.expand(len(test), *rp.shape[1:])).to(torch.bfloat16)
                    with torch.no_grad():
                        lm(**inp)
                        m = measures(slot_mask)
                    for k in D:
                        D[k][:, p] = (m[k] - base[k]).mean(0).cpu()
                for k, M in D.items():
                    rank = lambda own: torch.tensor([float((M[i] < M[i, own[i]]).double().mean() * P / (P - 1))
                                                     for i in range(P)])
                    real = rank(list(range(P)))
                    null = torch.cat([rank(list(rng.permutation(P))) for _ in range(5)])
                    p95 = float(null.quantile(0.95))
                    bands = {}
                    for b, lo, hi in (("L0-5", 0, 5), ("L6-11", 6, 11), ("L12-23", 12, 23), ("L24-35", 24, 35),
                                      ("L36-47", 36, 47)):
                        sel = torch.tensor([lo <= L <= hi for L in layers])
                        bands[b] = round(float(real[sel].median()), 3)
                    rep["B"][f"{kind}_{k}"] = {"median": float(real.median()), "share_above_null_p95":
                                               float((real > p95).double().mean()), "null_p95": p95,
                                               "bands": bands, "rank": real.tolist()}
                    print(f"B {modality} {kind:9s} {k:4s}: median rank {real.median():.3f}, above null p95 "
                          f"{(real > p95).double().mean():.0%} (chance 5%); by band "
                          + " ".join(f"{b} {v:.2f}" for b, v in bands.items()), flush=True)
            print(f"B {modality} done [{time.perf_counter() - t2:.0f}s]", flush=True)

        # D. Recovery: ridge from each layer's residual at a slot to the slot's input.
        if "D" not in skip:
            t2 = time.perf_counter()
            feats = {L: [] for L in RECON_LAYERS}
            targets = []
            for chunk in range(0, len(docs), 8):
                inp = to(stack(docs[chunk:chunk + 8]), dev)
                with torch.no_grad():
                    hs = lm(**inp, output_hidden_states=True).hidden_states
                mask = inp["input_ids"] == slot_id
                for L in RECON_LAYERS:
                    feats[L].append(hs[L][mask].float().cpu())
                targets.append(inp[key].float()[:, :SLOTS].reshape(-1, inp[key].shape[-1]).cpu())
            Y = torch.cat(targets)
            n_fit = args.fit * SLOTS
            Yn = (Y - Y.mean(1, keepdim=True)) / Y.std(1, keepdim=True).clamp_min(1e-6) if modality == "image" else \
                Y / Y.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-6)
            rep["D"] = {}
            recon = {}
            for L in RECON_LAYERS:
                X = torch.cat(feats[L])
                X = (X - X[:n_fit].mean(0)) / X[:n_fit].std(0).clamp_min(1e-6)
                Xf, Xt = X[:n_fit].double().to(dev), X[n_fit:].double().to(dev)
                res = {}
                targets_d = [("normalized", Yn), ("raw", Y)]
                if modality == "audio":  # the magnitude spectrum: what survives without phase
                    targets_d.append(("logspec", torch.log(torch.fft.rfft(Yn).abs() + 1e-2)))
                for tname, T in targets_d:
                    Tf, Tt = T[:n_fit].double().to(dev), T[n_fit:].double().to(dev)
                    mu = Tf.mean(0)
                    best = None
                    gram = Xf.T @ Xf
                    # λ by a split of the fitting rows
                    h = n_fit * 3 // 4
                    for lam in (1e1, 1e2, 1e3, 1e4):
                        Wv = torch.linalg.solve(Xf[:h].T @ Xf[:h] + lam * torch.eye(X.shape[1], device=dev, dtype=torch.float64),
                                                Xf[:h].T @ (Tf[:h] - Tf[:h].mean(0)))
                        err = ((Xf[h:] @ Wv + Tf[:h].mean(0) - Tf[h:]) ** 2).sum()
                        if best is None or err < best[0]:
                            best = (err, lam)
                    W = torch.linalg.solve(gram + best[1] * torch.eye(X.shape[1], device=dev, dtype=torch.float64),
                                           Xf.T @ (Tf - mu))
                    pred = Xt @ W + mu
                    r2 = 1 - ((pred - Tt) ** 2).sum() / ((Tt - Tt.mean(0)) ** 2).sum()
                    res[tname] = round(float(r2), 4)
                    if tname == "raw":
                        recon[L] = pred.float().cpu()
                rep["D"][f"hidden_{L}"] = res
                print(f"D {modality} hidden_states[{L:2d}] (0 = embeddings): held-out R² "
                      + ", ".join(f"{k} {v:.3f}" for k, v in res.items()), flush=True)
            torch.save({"recon": recon, "truth": Y[n_fit:]}, OUT / f"recon_{modality}{args.tag}.pt")
            print(f"D {modality} done [{time.perf_counter() - t2:.0f}s]", flush=True)
        (OUT / "media_lens.json").write_text(json.dumps(report, indent=1))
    for h in hooks:
        h.remove()
    print(f"done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
