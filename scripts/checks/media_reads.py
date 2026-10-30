"""Image, audio and token reads of a panel of Gemma 4 12B poles, and a check that each read
drives its pole more than the other poles' reads do.

1. **Panel:** ``--neurons`` neurons, ``--per-layer`` per layer from the bottom up (seeded), both
   poles. A pole's read covector ``c = â ± b̂`` is its gate and up rows times the
   pre-feedforward norm weight, normalized, on layer ``L``'s MLP input (as 08_reads.py
   reads Qwen's, whose MLP norm is the post-attention one).
2. **Pull:** ``u = c · K_L`` for every pole, ``K_L`` the S-lens's ``mid`` target averaged over
   ``--docs`` corpus documents, computed without fitting the S-lens (``input_reads_doc``); even
   and odd documents apart, for a split-half cosine. The no-lens control reads ``u = c`` (``K = I``).
3. **Reads** of ``u``, all under the RelP LN rule (each norm's scale held constant, so a
   direction is exact up to scale):
   - pixels: back through the encoder-free vision embedder (``LN₁ → Dense → LN₂ → +pos → LN₃
     → RMSNorm → Linear``): ``C(γ₁ ⊙ W_denseᵀ C(γ₂ ⊙ C(γ₃ ⊙ W_projᵀ u)))``, ``C`` the
     LayerNorms' centring: a 48×48×3 patch (HWC, as the image processor lays patches out);
   - audio: ``W_audioᵀ u`` (``RMSNorm → Linear`` on raw 640-sample frames): 40 ms at 16 kHz;
   - tokens: the top of ``u · (W_E[v] − ē)`` over tokens seen 5+ times in the lens corpus.
4. **Verify** (forward passes of the whole model, bf16), on inputs of 64 slots (image: an 8×8
   grid of 48-pixel patches, a 384-pixel image; audio: 64 frames; text: 64 tokens):
   - *fill*: every slot holds one pole's read (text: its top 8 tokens, cycled), the input the
     S-lens read describes (it sums over source positions). A pole's mean signed activation
     over the slots on its own read's input, ranked among the other poles' inputs: an AUROC
     with one positive;
   - *slot*: one read per slot, every pole's read in ``--rounds`` inputs at shuffled slots. A
     pole's signed activation at its own read's slots against at the other poles' slots.
   Each null permutes which read is the pole's own.

    python scripts/checks/media_reads.py [--neurons 96] [--docs 64] [--device mps]
"""
import argparse
import json
import random
import time
import wave

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from neuron_lens.corpus import lens_docs
from neuron_lens.paths import CORPUS_DIR, DATA
from neuron_lens.transport import SKIP_FIRST, decoder_stack, input_reads_doc

MODEL = "google/gemma-4-12B"
OUT = DATA / "summary" / "gemma4_media"
GRID, FRAMES, SLOTS = 8, 64, 64


def auroc(pos, neg):
    pos, neg = torch.as_tensor(pos, dtype=torch.float64), torch.as_tensor(neg, dtype=torch.float64)
    return float((pos[:, None] > neg[None]).double().mean() + 0.5 * (pos[:, None] == neg[None]).double().mean())


def centre(v):
    return v - v.mean(-1, keepdim=True)


def pixel_reads(lm, u):
    """``[k, 3840]`` embedding covectors → ``[k, 48, 48, 3]`` pixel covectors."""
    ev = lm.model.embed_vision
    w = lambda m: m.weight.float().cpu()
    v = u @ w(ev.multimodal_embedder.embedding_projection)
    v = centre(v * w(ev.pos_norm))
    v = centre(v * w(ev.patch_ln2))
    v = v @ w(ev.patch_dense)
    v = centre(v * w(ev.patch_ln1))
    return v.reshape(-1, 48, 48, 3)


def audio_reads(lm, u):
    return u @ lm.model.embed_audio.embedding_projection.weight.float().cpu()


def as_patch(r):
    return (0.5 + 0.45 * r / r.abs().amax(dim=(-3, -2, -1), keepdim=True)).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurons", type=int, default=96)
    ap.add_argument("--docs", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=4, help="inputs each pole's read appears in, per modality")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--fill-only", action="store_true", help="skip the slot test")
    ap.add_argument("--layers", type=int, default=None,
                    help="load the first N decoder layers only; the panel comes from them")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    config = AutoConfig.from_pretrained(MODEL)
    if args.layers is not None:  # a pole at layer L needs only layers 0..L: exact for the layers kept
        tc = config.text_config
        tc.num_hidden_layers, tc.layer_types = args.layers, tc.layer_types[:args.layers]
    lm = AutoModelForCausalLM.from_pretrained(MODEL, config=config, dtype=torch.bfloat16,
                                              attn_implementation="eager")
    lm = lm.to(args.device).eval().requires_grad_(False)
    model = decoder_stack(lm)
    tok = AutoTokenizer.from_pretrained(MODEL)
    proc = AutoProcessor.from_pretrained(MODEL)
    n_layers, f = len(model.layers), model.layers[0].mlp.gate_proj.weight.shape[0]
    print(f"loaded in {time.perf_counter() - t0:.0f}s: {n_layers} layers, {f} neurons each", flush=True)

    # 1. Panel.
    rng = random.Random(args.seed)
    per = max(1, args.neurons // n_layers)
    neurons = [(L, n) for L in range(n_layers) for n in rng.sample(range(f), per)][:args.neurons]
    poles = [(L, n, s) for L, n in neurons for s in (1, -1)]
    cov, norms = [], []
    for L, n, s in poles:
        mlp, g = model.layers[L].mlp, model.layers[L].pre_feedforward_layernorm.weight.float()
        a, b = mlp.gate_proj.weight[n].float() * g, mlp.up_proj.weight[n].float() * g
        cov.append(a / a.norm() + s * b / b.norm())
        norms.append((float(a.norm()), float(b.norm())))
    cov = torch.stack(cov).cpu()
    layers = [L for L, _, _ in poles]

    # 2. Pull through the S-lens, even and odd documents apart.
    rows = lens_docs(CORPUS_DIR / "dclm.jsonl", tok, 1000, 128, SKIP_FIRST + 2)
    rows = [r for r in rows if len(r) == 128][:args.docs]
    halves = [torch.zeros(len(poles), cov.shape[1]), torch.zeros(len(poles), cov.shape[1])]
    for i, r in enumerate(rows):
        t1 = time.perf_counter()
        halves[i % 2] += input_reads_doc(model, torch.tensor(r), cov, layers).cpu()
        if i == 0 or (i + 1) % 16 == 0:
            print(f"pull doc {i + 1}/{len(rows)}: {time.perf_counter() - t1:.1f}s", flush=True)
    even, odd = (h / (len(rows) / 2) for h in halves)
    u_lens = (even + odd) / 2
    split_cos = torch.nn.functional.cosine_similarity(even, odd, dim=1)
    print(f"split-half cosine of the pulled covectors ({len(rows) // 2} docs a half): median "
          f"{split_cos.median():.3f}, 10th pct {split_cos.quantile(0.1):.3f}", flush=True)
    reads = {"lens": u_lens, "nolens": cov}

    # 3. Reads.
    E = model.embed_tokens.weight.float().cpu()
    counts = torch.bincount(torch.tensor([t for r in lens_docs(CORPUS_DIR / "dclm.jsonl", tok, 1000, 4096, 1)
                                          for t in r]), minlength=E.shape[0])
    keep = (counts >= 5).nonzero().squeeze(1)
    Ek = centre(E[keep].T).T
    vocab = lambda ids: [tok.decode([int(keep[i])]) for i in ids]
    out = {"model": MODEL, "docs": len(rows), "poles": [], "split_cos": split_cos.tolist(), "verify": {}}
    px, au, tk = {}, {}, {}
    for kind, u in reads.items():
        px[kind] = pixel_reads(lm, u)
        au[kind] = audio_reads(lm, u)
        tk[kind] = (u @ Ek.T).topk(10, dim=1).indices
    for i, (L, n, s) in enumerate(poles):
        spec = np.abs(np.fft.rfft(au["lens"][i].numpy()))
        out["poles"].append({"layer": L, "neuron": n, "pole": s, "split_cos": round(float(split_cos[i]), 3),
                             "tokens_lens": vocab(tk["lens"][i]), "tokens_nolens": vocab(tk["nolens"][i]),
                             "audio_peak_hz": float(np.fft.rfftfreq(640, 1 / 16000)[spec[1:-1].argmax() + 1])})
    torch.save({"poles": poles, "cov": cov, "u_lens": u_lens, "even": even, "odd": odd,
                "pixels": px, "audio": au}, OUT / "reads.pt")
    print(f"reads done at {time.perf_counter() - t0:.0f}s", flush=True)

    # 4. Verify.
    acts = {}

    def hook(L, module, key):
        idx = torch.tensor([n for L2, n, _ in poles if L2 == L], device=args.device)
        if key == "act":
            return module.register_forward_pre_hook(lambda _m, a: acts.__setitem__((key, L), a[0][0, :, idx].float().cpu()))
        return module.register_forward_hook(lambda _m, _a, o: acts.__setitem__((key, L), o[0, :, idx].float().cpu()))
    hooks = [hook(L, getattr(model.layers[L].mlp, name), key) for L in sorted(set(layers))
             for name, key in (("down_proj", "act"), ("gate_proj", "gate"), ("up_proj", "up"))]
    METRIC = {"act": "act"}  # which measure pole_acts returns: the activation, or the read projection
    col = {}
    for L in sorted(set(layers)):
        ns = [n for L2, n, _ in poles if L2 == L]
        for j, n in enumerate(ns):
            col[(L, n)] = j

    def pole_acts(slot_positions):
        """[poles, slots] at the slot positions: each pole's signed activation (``act``), or its
        read projection ``c · x = gate/‖a‖ ± up/‖b‖`` (``proj``), what the lens predicts linearly."""
        if METRIC["act"] == "act":
            return torch.stack([s * acts[("act", L)][slot_positions, col[(L, n)]] for L, n, s in poles])
        return torch.stack([acts[("gate", L)][slot_positions, col[(L, n)]] / na
                            + s * acts[("up", L)][slot_positions, col[(L, n)]] / nb
                            for (L, n, s), (na, nb) in zip(poles, norms)])

    from PIL import Image
    tpl = {"image": proc(text=f"Look: {proc.image_token}", images=[Image.new("RGB", (GRID * 48, GRID * 48))],
                         images_kwargs={"max_soft_tokens": 70}, return_tensors="pt"),
           "audio": proc(text=f"Hear: {proc.audio_token}", audio=[np.zeros(FRAMES * 640, np.float32)],
                         return_tensors="pt")}
    assert (tpl["image"]["image_position_ids"][0, :SLOTS] ==
            torch.tensor([[c % GRID, c // GRID] for c in range(SLOTS)])).all()  # row c is cell (c % 8, c // 8)
    assert tpl["audio"]["input_features"].shape[1] == FRAMES

    last_slots = [None]

    def run(kind, modality, assignment):
        if modality == "text":  # a pole filling every slot cycles through its top 8 tokens
            rank = [sum(q == p for q in assignment[:c]) % 8 for c, p in enumerate(assignment)]
            words = torch.tensor([int(keep[tk[kind][p][r]]) for p, r in zip(assignment, rank)])
            inputs = {"input_ids": torch.cat([tok("Words:", return_tensors="pt")["input_ids"][0], words])[None]}
            slots = torch.arange(inputs["input_ids"].shape[1] - SLOTS, inputs["input_ids"].shape[1])
        else:
            inputs = {k: v.clone() for k, v in tpl[modality].items()}
            if modality == "image":
                inputs["pixel_values"][0, :SLOTS] = torch.stack([as_patch(px[kind][p]).reshape(-1) for p in assignment])
                slot_token = proc.image_token_id
            else:
                w = au[kind][list(assignment)]
                inputs["input_features"][0] = 0.05 * w / w.pow(2).mean(1, keepdim=True).sqrt()
                slot_token = proc.audio_token_id
            slots = (inputs["input_ids"][0] == slot_token).nonzero().squeeze(1)
            assert len(slots) == SLOTS
        inputs = {k: v.to(args.device, torch.bfloat16) if v.is_floating_point() else v.to(args.device)
                  for k, v in inputs.items()}
        with torch.no_grad():
            lm(**inputs)
        last_slots[0] = slots
        return pole_acts(slots)

    P = len(poles)
    bands_of = (("L0-5", 0, 5), ("L6-11", 6, 11), ("L12-15", 12, 15), ("L16-31", 16, 31), ("L32-47", 32, 47))

    def summarize(name, real, null, t1):
        real_t, null = torch.tensor(real), torch.tensor([x for x in null if x == x])
        p95 = float(null.quantile(0.95))
        bands = {}
        for b, lo, hi in bands_of:
            sel = torch.tensor([lo <= L <= hi for L, _, _ in poles])
            if sel.any():
                bands[b] = {"median_auroc": round(float(real_t[sel].median()), 3),
                            "share_above_null_p95": round(float((real_t[sel] > p95).double().mean()), 3)}
        out["verify"][name] = {"auroc": real, "null_median": float(null.median()), "null_p95": p95, "bands": bands}
        print(f"verify {name:20s}: median AUROC {real_t.median():.3f} (null median {null.median():.3f}, "
              f"p95 {p95:.3f}); above null p95 {(real_t > p95).double().mean():.0%}; by band "
              + ", ".join(f"{b} {v['median_auroc']:.2f}/{v['share_above_null_p95']:.0%}" for b, v in bands.items())
              + f" [{time.perf_counter() - t1:.0f}s]", flush=True)

    for modality in ("image", "audio", "text"):
        for kind in ("lens", "nolens"):
            # Fill: every slot holds pole k's read (the S-lens read sums over source positions, so
            # this is the input it describes); M[i, k] is pole i's mean activation over the slots.
            t1 = time.perf_counter()
            Ms = {"act": [], "proj": []}
            for k in range(P):
                for metric in Ms:
                    METRIC["act"] = metric
                    if metric == "act":
                        a = run(kind, modality, [k] * SLOTS)
                    else:
                        a = pole_acts(last_slots[0])
                    Ms[metric].append(a.mean(1))
            METRIC["act"] = "act"
            for metric, cols in Ms.items():
                M = torch.stack(cols, 1)  # [P, P]
                rank = lambda own: [float((M[i] < M[i, own[i]]).double().mean() * P / (P - 1)) for i in range(P)]
                null = []
                for _ in range(5):
                    perm = list(range(P))
                    rng.shuffle(perm)
                    null += [x for i, x in enumerate(rank(perm)) if perm[i] != i]
                summarize(f"{modality}_{kind}_fill" + ("" if metric == "act" else "_proj"),
                          rank(list(range(P))), null, t1)
            if args.fill_only:
                continue
            # Slot: one read per slot, every pole's read in --rounds inputs at shuffled slots.
            t1 = time.perf_counter()
            order = []
            for _ in range(args.rounds):
                perm = list(range(P))
                rng.shuffle(perm)
                order += perm
            while len(order) % SLOTS:
                order.append(rng.randrange(P))
            records = [(order[k:k + SLOTS], run(kind, modality, order[k:k + SLOTS]))
                       for k in range(0, len(order), SLOTS)]

            def score(owner):
                res = []
                for i in range(P):
                    o, x = [], []
                    for assign, a in records:
                        for c, p in enumerate(assign):
                            (o if owner[p] == i else x).append(float(a[i, c]))
                    res.append(auroc(o, x) if o else float("nan"))
                return res
            null = []
            for _ in range(3):
                perm = list(range(P))
                rng.shuffle(perm)
                null += score(perm)
            summarize(f"{modality}_{kind}_slot", score(list(range(P))), null, t1)
    for h in hooks:
        h.remove()
    (OUT / "media_reads.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    figure(out, px["lens"], au["lens"], poles)
    print(f"done in {time.perf_counter() - t0:.0f}s; wrote {OUT}", flush=True)


def figure(out, px, au, poles, k=12):
    """A row per pole for the first ``k`` panel neurons' + poles spread over depth: patch,
    spectrum, top tokens, and the pole's image AUROC; plus WAVs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plus = [i for i, (_, _, s) in enumerate(poles) if s == 1]
    pick = [plus[j] for j in np.linspace(0, len(plus) - 1, k).round().astype(int)]
    v = out["verify"]
    fig, ax = plt.subplots(k, 3, figsize=(12, 1.9 * k), gridspec_kw={"width_ratios": [1, 2.2, 2.8]})
    freqs = np.fft.rfftfreq(640, 1 / 16000)
    for row, i in enumerate(pick):
        L, n, s = poles[i]
        ax[row, 0].imshow(as_patch(px[i]).numpy())
        ax[row, 0].set_axis_off()
        ax[row, 0].set_title(f"L{L} n{n} {'+' if s > 0 else '−'}  img AUROC {v['image_lens_fill']['auroc'][i]:.2f}", fontsize=8)
        spec = np.abs(np.fft.rfft(au[i].numpy()))
        ax[row, 1].plot(freqs, spec / spec.max(), lw=0.7)
        ax[row, 1].set_xlim(0, 8000)
        ax[row, 1].set_yticks([])
        ax[row, 1].set_title(f"40 ms read, audio AUROC {v['audio_lens_fill']['auroc'][i]:.2f}", fontsize=8)
        ax[row, 2].set_axis_off()
        toks = " ".join(repr(t)[1:-1] for t in out["poles"][i]["tokens_lens"][:8])
        ax[row, 2].text(0, 0.5, f"{toks}\ntext AUROC {v['text_lens_fill']['auroc'][i]:.2f}", fontsize=8, va="center", wrap=True)
        w = au[i].numpy()
        w = w / np.abs(w).max()
        y = np.zeros(16000 + 640)
        for o in range(0, 16000, 320):
            y[o:o + 640] += w * np.hanning(640)
        with wave.open(str(OUT / f"L{L:02d}_n{n}_{'plus' if s > 0 else 'minus'}.wav"), "wb") as fw:
            fw.setnchannels(1)
            fw.setsampwidth(2)
            fw.setframerate(16000)
            fw.writeframes((0.8 * y[:16000] / np.abs(y).max() * 32767).astype(np.int16).tobytes())
    ax[-1, 1].set_xlabel("Hz")
    plt.tight_layout()
    plt.savefig(OUT / "media_reads.png", dpi=100)


if __name__ == "__main__":
    main()
