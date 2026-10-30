"""Whole sounds for deep Gemma 4 12B neurons, synthesised from scratch, against real ones.

For one + pole per layer from ``--from-layer`` (media_reads.py's panel): a ``--seconds`` clip
optimised from quiet noise to raise the pole's mean activation over the clip's audio tokens,
in the photograph-style audio prompt. The clip is a spectrogram (log-magnitudes per 40 ms
frame, fixed random phases), so it stays sound-like rather than sample noise. Then:
- held-out context: every pole's clip in another prompt; a pole's rank of its own clip among
  the others' (1 = it fires most for its own, 0.5 = chance);
- real sounds: every clean ESC-50 clip in the training prompt; each pole's top category (mean
  activation) and top clip, written beside its synthesised clip as WAVs.

    python scripts/checks/synth_audio.py --device cuda
"""
import argparse
import json
import time
import wave

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor

from neuron_lens.media import PROMPT, sounds
from neuron_lens.paths import DATA
from neuron_lens.transport import decoder_stack

MODEL = "google/gemma-4-12B"
OUT = DATA / "summary" / "gemma4_media" / "synth_audio"
OTHER = "Audio clip: {}. Description:"


def write(path, y, sr=16000):
    y = y / (np.abs(y).max() + 1e-9) * 0.7
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((y * 32767).astype(np.int16).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--from-layer", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--esc", type=int, default=2000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev, t0 = args.device, time.perf_counter()
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager")
    lm = lm.to(dev).eval().requires_grad_(False)
    model = decoder_stack(lm)
    proc = AutoProcessor.from_pretrained(MODEL)
    panel = torch.load(DATA / "summary" / "gemma4_media" / "reads.pt", weights_only=False)["poles"]
    poles, seen = [], set()
    for L, n, s in panel:
        if L >= args.from_layer and s > 0 and L not in seen:
            poles.append((L, n, s))
            seen.add(L)
    P = len(poles)
    frames = int(args.seconds * 16000) // 640
    print(f"loaded in {time.perf_counter() - t0:.0f}s; {P} poles, {frames} frames ({frames * 0.04:.1f} s)", flush=True)

    acts = {}
    hooks = [model.layers[L].mlp.down_proj.register_forward_pre_hook(
        lambda _m, a, i=i, n=n: acts.__setitem__(i, a[0][..., n])) for i, (L, n, _) in enumerate(poles)]

    def template(prompt, n_frames):
        return proc(text=prompt.format(proc.audio_token), audio=[np.zeros(n_frames * 640, np.float32)], return_tensors="pt")

    tpl = {k: template(p, frames) for k, p in (("train", PROMPT["audio"]), ("other", OTHER))}

    def run(wave_batch, which, grad=False):
        """[B, P]: each pole's mean activation over the audio tokens of each clip."""
        t = tpl[which]
        B = wave_batch.shape[0]
        inp = {k: v.repeat(B, *[1] * (v.dim() - 1)).to(dev) for k, v in t.items()}
        inp["input_features"] = wave_batch.reshape(B, -1, 640).to(dev, torch.bfloat16)
        mask = (inp["input_ids"] == proc.audio_token_id).float()
        with torch.set_grad_enabled(grad):
            lm.model(**inp, use_cache=False)  # no output head: only the hooks' activations are needed
            a = torch.stack([acts[i].float() for i in range(P)], -1)  # [B, T, P]
            return (a * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)

    # Synthesis: log-magnitude spectrogram, fixed random phase, frames of exactly the model's 640 samples.
    g = torch.Generator().manual_seed(0)
    phase = torch.rand(P, frames, 321, generator=g) * 2 * np.pi
    phase[..., 0] = 0
    logmag = (torch.randn(P, frames, 321, generator=g) * 0.1 - 2).to(dev).requires_grad_(True)
    phase = phase.to(dev)
    adam = torch.optim.Adam([logmag], lr=0.05)
    G = 2
    for step in range(args.steps):
        for g0 in range(0, P, G):
            ps = list(range(g0, min(P, g0 + G)))
            spec = torch.exp(logmag[ps]) * torch.exp(1j * phase[ps])
            wav = torch.fft.irfft(spec, n=640, dim=-1).reshape(len(ps), -1)
            m = run(wav, "train", grad=True)
            loss = -m[torch.arange(len(ps)), torch.tensor(ps)].mean()
            # Smoothness over time keeps it a sound rather than flicker between frames.
            loss = loss + 0.02 * (logmag[ps][:, 1:] - logmag[ps][:, :-1]).pow(2).mean()
            adam.zero_grad()
            loss.backward()
            adam.step()
        if step % 30 == 0 or step == args.steps - 1:
            print(f"step {step}: loss {loss.item():.3f} [{time.perf_counter() - t0:.0f}s]", flush=True)
    with torch.no_grad():
        clips = torch.fft.irfft(torch.exp(logmag) * torch.exp(1j * phase), n=640, dim=-1).reshape(P, -1).float().cpu()
    torch.save({"poles": poles, "clips": clips, "logmag": logmag.detach().cpu()}, OUT / "synth_audio.pt")

    # Held out: every pole's clip in the other prompt.
    with torch.no_grad():
        M = torch.cat([run(clips[k:k + 2], "other") for k in range(0, P, 2)]).cpu()  # [clip, pole]
        M_train = torch.cat([run(clips[k:k + 2], "train") for k in range(0, P, 2)]).cpu()
    rank = lambda M: [float((M[:, i] < M[i, i]).double().mean() * P / (P - 1)) for i in range(P)]
    r_other, r_train = rank(M), rank(M_train)
    print(f"own clip's rank among the {P} clips: training prompt median {np.median(r_train):.2f}; "
          f"held-out prompt median {np.median(r_other):.2f} (chance 0.5); "
          f"{sum(x >= 0.95 for x in r_other)}/{P} at ≥0.95", flush=True)

    # Real sounds: ESC-50 clips, each pole's activation over the clip.
    esc = sounds(args.esc, "esc")
    # sounds() skipped silent clips; recover each kept clip's category by re-walking the same filter
    import io
    import soundfile as sf
    from scipy.signal import resample_poly
    kept_cat = []
    for row in load_dataset("ashraq/esc50", split="train", streaming=True).decode(False):
        w, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        w = w.mean(1) if w.ndim > 1 else w
        w = resample_poly(w, 16000, sr).astype(np.float32) if sr != 16000 else w
        need = 64 * 640
        w = np.pad(w, (0, max(0, need - len(w))))
        mid = (len(w) - need) // 2
        w = w[mid:mid + need]
        if np.sqrt((w ** 2).mean()) < 1e-3 or np.sqrt((w.reshape(64, 640) ** 2).mean(1)).min() < 1e-4:
            continue
        kept_cat.append(row["category"])
        if len(kept_cat) == len(esc):
            break
    tpl["esc"] = template(PROMPT["audio"], 64)
    A = []
    with torch.no_grad():
        for k in range(0, len(esc), 8):
            A.append(run(torch.tensor(np.stack(esc[k:k + 8])), "esc").cpu())
    A = torch.cat(A)  # [clips, poles]
    report = {"poles": [], "seconds": args.seconds, "rank_other_median": float(np.median(r_other)),
              "rank_train_median": float(np.median(r_train))}
    classes = sorted(set(kept_cat))
    for i, (L, n, s) in enumerate(poles):
        by_cat = {c: float(A[[j for j, k in enumerate(kept_cat) if k == c], i].mean()) for c in classes}
        top_cat = sorted(by_cat, key=by_cat.get, reverse=True)[:3]
        top_clip = int(A[:, i].argmax())
        synth_vs_real = float(M_train[i, i] / A[top_clip, i]) if A[top_clip, i] > 0 else float("nan")
        stem = f"L{L:02d}_n{n}_plus"
        write(OUT / f"{stem}_synthesised_{args.seconds:.0f}s.wav", clips[i].numpy())
        write(OUT / f"{stem}_top_real_{kept_cat[top_clip].replace(' ', '-')}.wav", esc[top_clip])
        report["poles"].append({"layer": L, "neuron": n, "rank_other": r_other[i], "rank_train": r_train[i],
                                "top_categories": top_cat, "top_clip_category": kept_cat[top_clip],
                                "synth_activation_over_top_real": synth_vs_real})
        print(f"L{L} n{n}: own-clip rank {r_other[i]:.2f} (other prompt); top real categories {', '.join(top_cat)}; "
              f"synthesised fires {synth_vs_real:.1f}x its top real clip", flush=True)
    for h in hooks:
        h.remove()
    (OUT / "synth_audio.json").write_text(json.dumps(report, indent=1))
    print(f"done in {time.perf_counter() - t0:.0f}s; wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
