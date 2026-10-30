"""Image and audio documents for Gemma 4 12B (``gemma4_unified``), and its embedders read
backwards, for the media lenses.

A document is a short text prompt around one input of ``SLOTS`` (64) soft tokens: a COCO image
centre-cropped to 384 pixels (an 8×8 grid of 48-pixel patches; the processor's row ``c`` is
cell ``(c % 8, c // 8)``) or 2.56 s of audio at 16 kHz (64 frames of 640 samples: ESC-50
sounds, or those interleaved with LibriSpeech speech). The slots start past ``SKIP_FIRST``.

Both embedders are linear up to their norms, so under the RelP LN rule (each norm's scale held
constant) an embedding-space covector ``u`` reads back exactly, up to scale:
- pixels: ``LN₁ → Dense → LN₂ → +pos → LN₃ → RMSNorm → Linear`` gives
  ``C(γ₁ ⊙ W_denseᵀ C(γ₂ ⊙ C(γ₃ ⊙ W_projᵀ u)))``, ``C`` the LayerNorms' centring: a 48×48×3
  patch (HWC). The map has rank 2,041, so the image lens needs only the top 2,048 left
  singular directions of ``embedding ← pixels`` as sources (``basis``);
- audio: ``RMSNorm → Linear`` on raw frames gives ``W_audioᵀ u``: 40 ms of samples. Frames
  reach the 640-dimensional column space of ``W_audio``, the audio lens's sources.
"""
import io

import numpy as np
import torch

SLOTS, SIDE = 64, 384
PROMPT = {"image": "Here is a photograph that I took last weekend while travelling with some friends of mine. "
                   "Please look at it carefully: {}. It shows",
          "audio": "Here is a short recording that I made last weekend while travelling with some friends of mine. "
                   "Please listen to it carefully: {}. It sounds like"}


def images(n: int, offset: int = 0) -> list:
    """COCO 2014 validation images ``offset .. offset + n``, centre-cropped and resized to 384."""
    from datasets import load_dataset
    from PIL import Image
    ds = load_dataset("lmms-lab/COCO-Caption", split="val", streaming=True).decode(False)
    out = []
    for i, row in enumerate(ds):
        if i < offset:
            continue
        im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        s = min(im.size)
        l, t = (im.size[0] - s) // 2, (im.size[1] - s) // 2
        out.append(im.crop((l, t, l + s, t + s)).resize((SIDE, SIDE), Image.BICUBIC))
        if len(out) == n:
            return out
    return out


def labelled_images(n: int, split: str = "validation") -> tuple[list, list[str]]:
    """``n`` mini-ImageNet images (100 ImageNet classes; full-size originals), centre-cropped to
    384, spread over the classes, and each one's class (its first ImageNet name)."""
    from datasets import load_dataset
    from PIL import Image
    from timm.data import ImageNetInfo
    info = ImageNetInfo()
    ds = load_dataset("timm/mini-imagenet", split=split, streaming=True)
    names = ds.features["label"].names
    total = {"validation": 10_000, "test": 5_000, "train": 50_000}[split]
    stride = max(1, total // n)
    out, labels = [], []
    for i, row in enumerate(ds.decode(False)):
        if i % stride:
            continue
        im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        s = min(im.size)
        l, t = (im.size[0] - s) // 2, (im.size[1] - s) // 2
        out.append(im.crop((l, t, l + s, t + s)).resize((SIDE, SIDE), Image.BICUBIC))
        labels.append(info.label_name_to_description(names[row["label"]]).split(",")[0])
        if len(out) == n:
            break
    return out, labels


def sounds(n: int, source: str = "mix", categories: bool = False):
    """``n`` 2.56 s clips at 16 kHz from their middles, skipping silent ones and those with a
    digitally silent 40 ms frame (with ``categories``, also each clip's ESC-50 category, or
    ``speech``): ``mix`` interleaves
    ESC-50 sounds and LibriSpeech speech (73 clips at most), ``esc`` is ESC-50 only (2,000)."""
    import soundfile as sf
    from datasets import load_dataset
    from scipy.signal import resample_poly
    clips, cats = [], []
    sources = (("ashraq/esc50", None, "train", n - n // 2),
               ("hf-internal-testing/librispeech_asr_dummy", "clean", "validation", n // 2))
    if source == "esc":
        sources = (("ashraq/esc50", None, "train", n),)
    for name, cfg, split, k in sources:
        got = 0
        for row in load_dataset(name, cfg, split=split, streaming=True).decode(False):
            w, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            w = w.mean(1) if w.ndim > 1 else w
            if sr != 16000:
                w = resample_poly(w, 16000, sr).astype(np.float32)
            need = SLOTS * 640
            if len(w) < need:
                w = np.pad(w, (0, need - len(w)))
            mid = (len(w) - need) // 2
            w = w[mid:mid + need]
            if np.sqrt((w ** 2).mean()) < 1e-3:  # silence
                continue
            # A digitally silent frame embeds to exactly zero (the audio embedder has no bias),
            # and every norm's derivative at a zero residual is 1/√eps: one such frame swamps a lens.
            if np.sqrt((w.reshape(SLOTS, 640) ** 2).mean(1)).min() < 1e-4:
                continue
            clips.append(w)
            cats.append(row.get("category", "speech"))
            got += 1
            if got == k:
                break
    if source == "esc":
        return (clips, cats) if categories else clips
    k = n - n // 2  # interleave, so any split holds some of each
    order = [j for pair in zip(range(k), range(k, len(clips))) for j in pair] + list(range(len(clips) - k, k))
    order = order if len(clips) - k < k else [j for pair in zip(range(k), range(k, len(clips))) for j in pair]
    out = [clips[j] for j in order]
    return (out, [cats[j] for j in order]) if categories else out


def processed(proc, modality: str, item) -> dict:
    """The processor's inputs (batch 1) for one document."""
    if modality == "image":
        return proc(text=PROMPT["image"].format(proc.image_token), images=[item],
                    images_kwargs={"max_soft_tokens": 70}, return_tensors="pt")
    return proc(text=PROMPT["audio"].format(proc.audio_token), audio=[item], return_tensors="pt")


def slot_token(proc, modality: str) -> int:
    return proc.image_token_id if modality == "image" else proc.audio_token_id


def _centre(v):
    return v - v.mean(-1, keepdim=True)


def pixel_reads(lm, u: torch.Tensor) -> torch.Tensor:
    """``[k, 3840]`` embedding covectors → ``[k, 48, 48, 3]`` pixel covectors."""
    ev = lm.model.embed_vision
    w = lambda m: m.weight.float().cpu()
    v = u.float().cpu() @ w(ev.multimodal_embedder.embedding_projection)
    v = _centre(v * w(ev.pos_norm))
    v = _centre(v * w(ev.patch_ln2))
    v = v @ w(ev.patch_dense)
    v = _centre(v * w(ev.patch_ln1))
    return v.reshape(-1, 48, 48, 3)


def audio_reads(lm, u: torch.Tensor) -> torch.Tensor:
    """``[k, 3840]`` embedding covectors → ``[k, 640]`` waveform covectors (40 ms at 16 kHz)."""
    return u.float().cpu() @ lm.model.embed_audio.embedding_projection.weight.float().cpu()


def basis(lm, modality: str) -> torch.Tensor:
    """``[k, d]`` orthonormal source directions that span what the modality's embedder reaches
    (images: the top 2,048 left singular vectors of ``embedding ← pixels``, rank 2,041; audio:
    the 640-dimensional column space of ``W_audio``)."""
    if modality == "audio":
        return torch.linalg.qr(lm.model.embed_audio.embedding_projection.weight.double().cpu())[0].T.float()
    ev = lm.model.embed_vision
    w = lambda m: m.weight.double().cpu()
    C = lambda n: torch.eye(n, dtype=torch.float64) - 1.0 / n
    d = w(ev.multimodal_embedder.embedding_projection)
    M = d @ (C(d.shape[1]) * w(ev.pos_norm)) @ (C(d.shape[1]) * w(ev.patch_ln2)) @ w(ev.patch_dense) \
        @ (C(ev.patch_dense.weight.shape[1]) * w(ev.patch_ln1))
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    k = 2048
    if int((S > 1e-6 * S[0]).sum()) > k:
        raise ValueError(f"the image map has rank {int((S > 1e-6 * S[0]).sum())} > {k}")
    return U[:, :k].T.float()
