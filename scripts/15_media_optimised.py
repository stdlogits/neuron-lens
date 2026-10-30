"""Optimised image and audio reads of Gemma 4 12B's best-labelled neuron poles: the patch or sound
that most raises each, found by gradient ascent through the model (the S-lens read is the start).

The S-lens read alone moves a pole's read projection but raises its activation only in the first
layers (images; audio not at all), while reads optimised from it raise their pole through L23
(``checks/media_lens.py``). So, per modality and layer L0-L35, the ``--per-layer`` poles with the
strongest reliable exemplar label (``13_media_activations.py``: past the shuffled-label null and in
the top 5 of both halves; by the label's z) get one: a single slot pattern, the same in all 64 slots,
added at ``ε`` (images: 0.08 of the pixel range at its peak, clipped to [0, 1]; audio: 0.3 of each
frame's RMS), ``--steps`` steps of Adam (lr 0.1) on the pole's mean activation over the slots of
``--fit`` documents, two a step, from the S-lens read (``14_media_tables.py``'s ``read_S_basis``
through the embedder), as ``checks/media_lens.py`` part C.

Each is then scored on ``--test`` held-out documents: the pole's mean activation over the slots
with the read added, minus without (``gain``), and the same for the S-lens read it started from
(``gain_lens``). Images: COCO 2014 val past the image S-lens's 1,000 (``media.images``), the fit
documents first, then the test ones. Sounds: ESC-50 (``media.sounds(…, "esc")``), the fit clips
first, then the test ones; the audio S-lens was fitted on all of ESC-50's clips, so the test clips
are new to the optimisation but not to the lens.

One file per modality, ``neurons/<modality>_optimised.parquet``: ``layer``, ``neuron``, ``pole``,
``label``, ``label_z``, ``gain``, ``gain_lens``, ``act_base`` (the mean activation without), and
``opt_png`` (images: the 48 × 48 patch, as ``read_png``) or ``opt_wave`` (audio: 640 samples, fp16,
in units of a frame's RMS).

    NEURON_LENS_MODEL=google/gemma-4-12B python scripts/15_media_optimised.py --modality image [--per-layer 48]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from neuron_lens import arch
from neuron_lens.catalog import MODEL, catalog_dir, write
from neuron_lens.media import images, processed, slot_token, sounds
from neuron_lens.paths import SUMMARY
from neuron_lens.transport import decoder_stack

SLOTS = 64
EPS = {"image": 0.08, "audio": 0.3}
LAYERS = range(36)  # where optimised reads still raise their pole (L0-L35)
FIT_OFFSET = 1000  # the image S-lens was fitted on COCO val's first 1,000: optimise and test past them


def summarize(modality: str, path=None) -> None:
    """The release card's numbers for one modality, by band (L0-L11, L12-L23, L24-L35), into
    ``data/summary/media_optimised_<model>.json``: poles, the median held-out gain of the optimised
    read and of the S-lens read, and the shares raising the pole and beating the S-lens read."""
    import pandas as pd
    t = pd.read_parquet(path or catalog_dir() / "neurons" / f"{modality}_optimised.parquet", columns=["layer", "gain", "gain_lens"])
    out_path = SUMMARY / f"media_optimised_{arch.stem(MODEL)}.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out[modality] = {f"L{lo}-L{lo + 11}": {"poles": len(d), "gain": float(d["gain"].median()), "gain_lens": float(d["gain_lens"].median()),
                                           "raises": float((d["gain"] > 0).mean()), "beats_lens": float((d["gain"] > d["gain_lens"]).mean())}
                     for lo in (0, 12, 24) for d in [t[(t.layer >= lo) & (t.layer <= lo + 11)]]}
    out_path.write_text(json.dumps(out, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=("image", "audio"))
    ap.add_argument("--per-layer", type=int, default=48)
    ap.add_argument("--fit", type=int, default=128)
    ap.add_argument("--test", type=int, default=8)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--group", type=int, default=16, help="poles optimised together (each its own sequences)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", default=None, help="a-b (default L0-L35)")
    ap.add_argument("--out", default=None, help="output file (default neurons/<modality>_optimised.parquet)")
    args = ap.parse_args()
    layers = LAYERS if args.layers is None else range(int(args.layers.split("-")[0]), int(args.layers.split("-")[1]) + 1)
    dev, modality = args.device, args.modality
    t0 = time.perf_counter()
    import importlib.util
    spec = importlib.util.spec_from_file_location("media_tables", __file__.replace("15_media_optimised.py", "14_media_tables.py"))
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    to_media = mt.embedder(modality)

    # The poles: per layer, the strongest reliable labels, with their S-lens reads.
    poles, start = [], []
    for L in layers:
        t = pq.read_table(catalog_dir() / "neurons" / modality / f"L{L:02d}.parquet",
                          columns=["layer", "neuron", "pole", "label", "label_z", "sig", "stable", "read_S_basis"]).to_pandas()
        t = t[t["sig"] & t["stable"]].sort_values("label_z", ascending=False).head(args.per_layer)
        poles += [(int(r.layer), int(r.neuron), int(r.pole), r.label, float(r.label_z)) for r in t.itertuples()]
        start.append(torch.tensor(np.stack(t["read_S_basis"].to_numpy())).float())
    basis = mt.released(f"{modality}/s-lens", "mid.pt")["basis"].float()
    r_lens = to_media(torch.cat(start) @ basis).reshape(len(poles), -1)  # [P, D]: each pole's S-lens read
    P = len(poles)
    print(f"{modality}: {P} poles over {len(layers)} layers [{time.perf_counter() - t0:.0f}s]", flush=True)

    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    lm.requires_grad_(False)
    model = decoder_stack(lm)
    proc = AutoProcessor.from_pretrained(MODEL)
    items = images(args.fit + args.test, offset=FIT_OFFSET) if modality == "image" else sounds(args.fit + args.test, "esc")
    docs = [processed(proc, modality, x) for x in items]
    fit, test = docs[:args.fit], docs[args.fit:]
    sid, key = slot_token(proc, modality), ("pixel_values" if modality == "image" else "input_features")

    by_layer = {}
    for i, (L, n, _, _, _) in enumerate(poles):
        by_layer.setdefault(L, []).append(i)
    idx = {L: torch.tensor([poles[i][1] for i in ix], device=dev) for L, ix in by_layer.items()}
    caught = {}
    hooks = [model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda _m, a, L=L: caught.__setitem__(L, a[0][..., idx[L]]))
             for L in by_layer]
    order = torch.tensor([i for L in by_layer for i in by_layer[L]])
    inv = torch.empty_like(order)
    inv[order] = torch.arange(P)
    sign = torch.tensor([p for _, _, p, _, _ in poles], dtype=torch.float32, device=dev)

    def acts(slot_mask) -> torch.Tensor:  # [B, P]: each pole's signed activation, mean over the slots
        m = slot_mask.float()
        m = m / m.sum(1, keepdim=True)
        a = torch.cat([caught[L].float() for L in by_layer], -1)[..., inv.to(dev)]
        return (sign * a * m[..., None]).sum(1)

    def stack(batch):
        return {k: torch.cat([b[k] for b in batch]).to(dev) for k in batch[0]}

    def perturb(x, r):  # x [B, slots(+pad), D], r [B, D]: one pattern in every slot, at ε
        r = r[:, None]
        if modality == "image":
            y = x.clone()
            y[:, :SLOTS] = (x[:, :SLOTS] + EPS["image"] * r / r.abs().amax(-1, keepdim=True)).clamp(0, 1)
            return y
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return x + EPS["audio"] * rms * r / r.pow(2).mean(-1, keepdim=True).sqrt()

    def run(batch, r=None):
        inp = stack(batch)
        x = inp[key].float()
        inp[key] = (perturb(x, r) if r is not None else x).to(torch.bfloat16)
        lm.model(**inp, use_cache=False)
        return acts(inp["input_ids"] == sid)

    opt = torch.zeros_like(r_lens)
    G = args.group
    for g0 in range(0, P, G):
        ps = list(range(g0, min(P, g0 + G)))
        r0 = r_lens[ps].to(dev)
        scale = r0.abs().amax(-1, keepdim=True) if modality == "image" else r0.pow(2).mean(-1, keepdim=True).sqrt()
        k = 1.0 if modality == "image" else 3.0
        w = (r0 / scale / k).clamp(-0.95, 0.95).atanh().clone().requires_grad_(True)
        adam = torch.optim.Adam([w], lr=0.1)
        for step in range(args.steps):
            pick = np.random.default_rng(step + g0).choice(len(fit), 2, replace=False)
            batch = [fit[j] for j in pick for _ in ps]
            with torch.enable_grad():
                m = run(batch, (k * torch.tanh(w)).repeat(2, 1))  # [2·G, P]
                own = torch.tensor(ps * 2, device=dev)
                loss = -m[torch.arange(len(own), device=dev), own].mean()
                adam.zero_grad()
                loss.backward()
            adam.step()
        opt[ps] = (k * torch.tanh(w)).detach().cpu()
        caught.clear()
        if g0 // G % 10 == 0:
            print(f"{time.strftime('%H:%M:%S')} {modality}: optimised {g0 + len(ps)}/{P} [{time.perf_counter() - t0:.0f}s]", flush=True)

    # Held out: each pole's gain from its optimised read and from its S-lens read.
    gain, gain_lens, base = (torch.zeros(P) for _ in range(3))
    with torch.no_grad():
        for d in test:
            base += run([d])[0].cpu() / len(test)
        for g0 in range(0, P, 32):
            ps = list(range(g0, min(P, g0 + 32)))
            for d in test:
                for r_all, out in ((opt, gain), (r_lens, gain_lens)):
                    m = run([d] * len(ps), r_all[ps].to(dev))
                    out[ps] += m[torch.arange(len(ps)), torch.tensor(ps)].cpu() / len(test)
    gain, gain_lens = gain - base, gain_lens - base
    for h in hooks:
        h.remove()
    cols = {"layer": pa.array([p[0] for p in poles], pa.int16()), "neuron": pa.array([p[1] for p in poles], pa.int16()),
            "pole": pa.array([p[2] for p in poles], pa.int8()), "label": pa.array([p[3] for p in poles]),
            "label_z": pa.array([p[4] for p in poles], pa.float32()), "gain": pa.array(gain.numpy()),
            "gain_lens": pa.array(gain_lens.numpy()), "act_base": pa.array(base.numpy())}
    if modality == "image":
        cols["opt_png"] = pa.array(mt.pngs(opt.reshape(P, mt.SIDE, mt.SIDE, 3)), pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
    else:
        o = opt / opt.pow(2).mean(-1, keepdim=True).sqrt()
        cols["opt_wave"] = pa.FixedSizeListArray.from_arrays(pa.array(o.half().flatten().numpy()), o.shape[1])
    write(cols, Path(args.out) if args.out else catalog_dir() / "neurons" / f"{modality}_optimised.parquet")
    if args.out is None and args.layers is None:
        summarize(modality)
    print(f"{modality}: {P} optimised reads; held-out gain median {gain.median():.3g} (S-lens read {gain_lens.median():.3g}), "
          f"past the S-lens read for {(gain > gain_lens).float().mean():.0%}, positive for {(gain > 0).float().mean():.0%} "
          f"[{time.perf_counter() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
