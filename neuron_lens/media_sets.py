"""Diverse labelled image and audio collections for the media exemplars (13_media_activations.py).

Each item comes with its labels, named ``source:label`` (``audioset:Dog``, ``fleurs:Swahili``,
``coco:person``; multi-label sources give several), its source, and a caption where the source
has one (kept for open-vocabulary labelling). Sources are read as streams, shard order and a
buffer shuffled (seed 0), so a sample reads a few shards rather than the whole split; the
sources load in parallel threads. Synthetic probes (``synth:``) give exact labels for
primitives real data blurs: tones, timbres, chords, noises, click trains, chirps, beats;
gratings, colours, shapes, textures, letters, scripts, emoji, maths and code.

Audio: 2.56 s at 16 kHz from the middle of each clip; clips with a digitally silent 40 ms frame
are skipped (they embed to zero, and every norm's derivative there is 1/√eps). Images: 384×384;
photographs centre-cropped, documents, charts, screens and drawings letterboxed (white).
"""
import hashlib
import io
import math
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from neuron_lens.media import SIDE, SLOTS

SR, NEED = 16000, SLOTS * 640
LETTERBOX = {"docvqa", "chartqa", "rico", "quickdraw", "mnist", "logo", "emoji", "sprite", "newyorker", "cartoon",
             "iam", "letters", "im2latex", "ai2d", "websight", "pubtabnet", "fashionmnist", "products", "slides"}
CELEBA = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair",
          "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
          "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face",
          "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
          "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]
GALAXY10 = ["disturbed", "merging", "round smooth", "in-between round smooth", "cigar-shaped smooth", "barred spiral",
            "unbarred tight spiral", "unbarred loose spiral", "edge-on without bulge", "edge-on with bulge"]


def _stream(name, cfg, split, n, seed=0, buffer=None, shard=None):
    from datasets import load_dataset
    ds = load_dataset(name, cfg, split=split, streaming=True)
    feats = ds.features
    if shard is not None:  # (of, index): one of several parallel readers of the split's files
        ds = ds.shard(num_shards=shard[0], index=shard[1])
    ds = ds.shuffle(seed=seed, buffer_size=buffer or max(1000, 2 * n)).decode(False)
    return ds, feats


def _folder_rows(name, split, n, seed=0):
    """An image-folder dataset's rows without streaming it: its files under ``split/<class>/``, the
    classes taken in turn (each class's files shuffled), fetched in parallel. A shuffle buffer
    over a class-sorted folder set fetches every file one by one before the first comes out."""
    from concurrent.futures import ThreadPoolExecutor

    from datasets import load_dataset_builder
    from huggingface_hub import HfApi, hf_hub_download
    feats = load_dataset_builder(name).info.features
    names = feats["label"].names
    by = {}
    for f in HfApi().list_repo_files(name, repo_type="dataset"):
        parts = f.split("/")
        if len(parts) == 3 and parts[0] == split and parts[1] in names and f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            by.setdefault(parts[1], []).append(f)
    rng = np.random.default_rng(seed)
    for v in by.values():
        rng.shuffle(v)
    order = [(c, v[i]) for i in range(max(map(len, by.values()))) for c, v in sorted(by.items()) if i < len(v)]
    order = order[:n + n // 10 + 8]  # a margin for unreadable files

    def fetch(item):
        c, f = item
        with open(hf_hub_download(name, f, repo_type="dataset"), "rb") as fh:
            return {"image": {"bytes": fh.read()}, "label": names.index(c)}
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(fetch, order))
    return rows, feats


def _bytes(data: dict) -> bytes:
    """A streamed file's bytes; some datasets give only an ``hf://`` path to a file in the repo."""
    if data.get("bytes"):
        return data["bytes"]
    path = data.get("path") or ""
    if path.startswith("hf://"):
        from huggingface_hub import HfFileSystem
        with HfFileSystem().open(path, "rb") as fh:
            return fh.read()
    raise ValueError(f"no bytes for {path!r}")


def _clip(audio: dict):
    """2.56 s from the middle at 16 kHz, or None (silence, a digitally silent frame, a bad file)."""
    import soundfile as sf
    from scipy.signal import resample_poly
    try:
        w, sr = sf.read(io.BytesIO(_bytes(audio)), dtype="float32")
    except Exception:
        return None
    w = w.mean(1) if w.ndim > 1 else w
    if sr != SR:
        g = math.gcd(SR, sr)
        w = resample_poly(w, SR // g, sr // g).astype(np.float32)
    w = np.pad(w, (0, max(0, NEED - len(w))))
    mid = (len(w) - NEED) // 2
    w = w[mid:mid + NEED]
    return w if _ok(w) else None


def _ok(w):
    return np.sqrt((w ** 2).mean()) >= 1e-3 and np.sqrt((w.reshape(SLOTS, 640) ** 2).mean(1)).min() >= 1e-4


def _image(data, letterbox=False):
    from PIL import Image
    try:
        im = data if isinstance(data, Image.Image) else Image.open(io.BytesIO(_bytes(data)))
        im = im.convert("RGB")
    except Exception:
        return None
    if letterbox:
        s = max(im.size)
        bg = Image.new("RGB", (s, s), (255, 255, 255))
        bg.paste(im, ((s - im.size[0]) // 2, (s - im.size[1]) // 2))
        return bg.resize((SIDE, SIDE), Image.BICUBIC)
    s = min(im.size)
    l, t = (im.size[0] - s) // 2, (im.size[1] - s) // 2
    return im.crop((l, t, l + s, t + s)).resize((SIDE, SIDE), Image.BICUBIC)


ID_KEY = re.compile(r"(^|_)(id|ids|file|filename|fname|path|name|key|url|ytid|start|end|start_time|end_time|index|idx)$|^__(key|url)__$",
                    re.IGNORECASE)
MAX_SECONDS = 900  # a source still filling after this keeps what it has


def _ref(name, cfg, split, revision, row, data) -> dict:
    """Where a document came from, to link it rather than host it: the dataset at its revision, the
    row's own identifiers (ids, file names, keys, clip times), the media file's path in the repo,
    and the media bytes' SHA-1 (a key that survives any reordering)."""
    ref = {"repo": name, "config": cfg, "split": split, "revision": revision}
    ref |= {k: v for k, v in row.items() if ID_KEY.search(k) and isinstance(v, (str, int, float)) and len(str(v)) < 200
            and not str(v).startswith("/")}  # not a local cache path
    if isinstance(data, dict):
        if data.get("path") and not data["path"].startswith("/"):
            ref["media_path"] = data["path"]
        try:
            ref["sha1"] = hashlib.sha1(_bytes(data)).hexdigest()
        except Exception:
            pass
    return ref


def _take(source, name, cfg, split, n, field, labeller, caption=None, kind="audio", letterbox=None, buffer=None,
          shard=None):
    """Up to ``n`` items of one source: ``[(item, labels, caption, source, ref)]``."""
    out, t0 = [], time.time()
    try:
        try:
            from huggingface_hub import HfApi
            revision = HfApi().dataset_info(name).sha
        except Exception:
            revision = None
        ds, feats = _folder_rows(name, split, n) if buffer == "folder" else _stream(name, cfg, split, n, buffer=buffer, shard=shard)
        lb = (source in LETTERBOX) if letterbox is None else letterbox
        for row in ds:
            x = _clip(row[field]) if kind == "audio" else _image(row[field], lb)
            if x is None:
                continue
            labs = [lab for lab in labeller(row, feats) if lab]
            out.append((x, [f"{source}:{lab}" for lab in labs], caption(row) if caption else None, source,
                        _ref(name, cfg, split, revision, row, row[field])))
            if len(out) == n:
                break
            if time.time() - t0 > MAX_SECONDS:
                print(f"  {source} ({name}): stopped at {len(out)}/{n} after {MAX_SECONDS}s", flush=True)
                break
    except Exception as e:  # one failing source must not sink the collection
        print(f"  {source}: FAILED {e!r}"[:200], flush=True)
    print(f"  {source} ({name}{f' {shard[1]}/{shard[0]}' if shard else ''}): {len(out)} in {time.time() - t0:.0f}s", flush=True)
    return out


def _names(feats, key):
    return feats[key].names


# ------------------------------------------------------------------ audio

def _fsd_labels(text):
    t = (text or "").removeprefix("The sounds of ").rstrip(".")
    return [x.strip() for x in t.replace(" and ", ", ").split(",") if x.strip()]


def audio_sources(s):
    n = lambda k: max(1, int(k * s))
    # Long audio rows overflow a big shuffle buffer (Arrow offsets), so the speech and AudioSet
    # streams shuffle within 64 rows (and over shard order).
    langs = {"dutch": "Dutch", "french": "French", "german": "German", "italian": "Italian", "polish": "Polish",
             "portuguese": "Portuguese", "spanish": "Spanish"}
    # AudioSet's balanced subsets whole (17,141 + 18,683 clips), each split read by 4 parallel shards.
    jobs = [("audioset", "agkphysics/AudioSet", None, split, n(total // 4 + 1), "audio", lambda r, f: r["human_labels"],
             None, 64, (4, i)) for split, total in (("test", 17141), ("train", 18683)) for i in range(4)]
    jobs += [
        ("fsd50k", "CLAPv2/FSD50K", None, "test", n(3000), "audio", lambda r, f: _fsd_labels(r["text"]), lambda r: r["text"], 64),
        ("clotho", "CLAPv2/Clotho", None, "test", n(1000), "audio", lambda r, f: ["captioned sound"], lambda r: r["text"], 64),
        ("fleurs", "google/fleurs", "all", "validation", n(4000), "audio",
         lambda r, f: [r["language"], "speech", _names(f, "gender")[r["gender"]] + " voice"], lambda r: r["transcription"], 64),
        ("edacc", "edinburghcstr/edacc", None, "test", n(1500), "audio",
         lambda r, f: [f"accent {r['accent']}", "English speech", f"{r['gender']} voice"], lambda r: r["text"], 64),
        ("voxpopuli", "facebook/voxpopuli", "en_accented", "test", n(1000), "audio",
         lambda r, f: [f"English by a native {_names(f, 'language')[r['language']]} speaker", "parliament speech"],
         lambda r: r["raw_text"], 64),
        ("expresso", "ylacombe/expresso", None, "train", n(1000), "audio", lambda r, f: [r["style"], "expressive speech"],
         lambda r: r["text"], 64),
        ("genre", "lewtun/music_genres_small", None, "train", n(1000), "audio", lambda r, f: [r["genre"], "music"], None),
        ("esc50", "ashraq/esc50", None, "train", n(2000), "audio", lambda r, f: [r["category"]], None),
        ("instrument", "ccmusic-database/instrument_timbre", None, "Western", n(400), "audio",
         lambda r, f: [_names(f, "instrument")[r["instrument"]], "Western instrument"], None),
        ("instrument", "ccmusic-database/instrument_timbre", None, "Chinese", n(400), "audio",
         lambda r, f: [_names(f, "instrument")[r["instrument"]], "Chinese instrument"], None),
    ]
    jobs += [("mls", "facebook/multilingual_librispeech", cfg, "test", n(150), "audio",
              lambda r, f, lang=lang: [lang, "read speech"], lambda r: r["transcript"], 64) for cfg, lang in langs.items()]
    return jobs


NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def synth_audio(scale: float = 1.0, seed: int = 0):
    """Exactly labelled synthetic sounds, 2.56 s each, every frame audible."""
    rng = np.random.default_rng(seed)
    t = np.arange(NEED) / SR
    out = []

    def add(w, labels):
        w = 0.3 * w / (np.abs(w).max() + 1e-9) + 1e-3 * rng.standard_normal(NEED)  # a floor: no silent frame
        if _ok(w):
            out.append((w.astype(np.float32), [f"synth:{x}" for x in labels], None, "synth", {"synth": labels}))

    def note(f):
        m = 69 + 12 * math.log2(f / 440)
        return NOTES[int(round(m)) % 12] + str(int(round(m)) // 12 - 1)

    def band(f):
        return "low pitch" if f < 250 else "mid pitch" if f < 2000 else "high pitch"
    reps = max(1, int(round(3 * scale)))
    for f in np.geomspace(55, 7040, 36):
        for k in range(reps):
            env = 1 if k == 0 else (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 4 * t))) if k == 1 else np.exp(-3 * (t % 0.5))
            add(np.sin(2 * np.pi * f * t) * env, ["sine tone", band(f), f"note {note(f)[:-1]}", "pure tone"])
    for wave, fn in (("sawtooth", lambda p: 2 * (p % 1) - 1), ("square", lambda p: np.sign(np.sin(2 * np.pi * p))),
                     ("triangle", lambda p: 2 * np.abs(2 * (p % 1) - 1) - 1)):
        for f in np.geomspace(65, 2093, 12 * reps):
            add(fn(f * t), [f"{wave} wave", band(f), "harmonic tone"])
    for name, iv in (("major chord", (0, 4, 7)), ("minor chord", (0, 3, 7)), ("diminished chord", (0, 3, 6)),
                     ("octave", (0, 12)), ("fifth", (0, 7)), ("tritone", (0, 6))):
        for root in np.geomspace(110, 880, 6 * reps):
            add(sum(np.sin(2 * np.pi * root * 2 ** (i / 12) * t) for i in iv), [name, "chord"])
    for colour, k in (("white noise", 0), ("pink noise", 1), ("brown noise", 2)):
        for _ in range(10 * reps):
            spec = np.fft.rfft(rng.standard_normal(NEED))
            spec /= np.maximum(np.arange(len(spec)), 1) ** (k / 2)
            add(np.fft.irfft(spec, NEED), [colour, "noise"])
    for rate in (2, 4, 8, 16, 32, 64):
        for _ in range(3 * reps):
            w = np.zeros(NEED)
            w[(np.arange(0, 2.56, 1 / rate) * SR).astype(int)] = 1
            w = np.convolve(w, np.hanning(32), "same")
            add(w, [f"click train {rate} Hz", "clicks"])
    for direction, (a, b) in (("rising", (100, 6000)), ("falling", (6000, 100))):
        for _ in range(6 * reps):
            k = np.log(b / a) / 2.56
            add(np.sin(2 * np.pi * a * (np.exp(k * t) - 1) / k), [f"{direction} chirp", "sweep"])
    for name, mod in (("tremolo", lambda f, r: np.sin(2 * np.pi * f * t) * (1 + np.sin(2 * np.pi * r * t))),
                      ("vibrato", lambda f, r: np.sin(2 * np.pi * f * t + 8 * np.sin(2 * np.pi * r * t)))):
        for f in (220, 440, 880):
            for r in (3, 6, 12)[:reps + 1]:
                add(mod(f, r), [name, "modulated tone"])
    dtmf = {"1": (697, 1209), "5": (770, 1336), "9": (852, 1477), "0": (941, 1336), "#": (941, 1477)}
    for key, (lo, hi) in dtmf.items():
        for _ in range(reps):
            add(np.sin(2 * np.pi * lo * t) + np.sin(2 * np.pi * hi * t), ["DTMF tone", "telephone tone"])
    for bpm in (60, 90, 120, 150, 180):
        for _ in range(3 * reps):
            w = np.zeros(NEED)
            for b in np.arange(0, 2.56, 60 / bpm):
                i = int(b * SR)
                hit = rng.standard_normal(1600) * np.exp(-np.arange(1600) / 300)
                w[i:i + 1600] += hit[:max(0, NEED - i)]
            add(w, [f"beat {bpm} bpm", "percussive beat"])
    return out


def audio_collection(scale: float = 1.0, log=print, workers: int | None = None):
    jobs = audio_sources(scale)
    with ThreadPoolExecutor(workers or len(jobs)) as ex:  # every source at once: the slowest bound the load
        parts = list(ex.map(lambda j: _take(*j[:8], "audio", buffer=j[8] if len(j) > 8 else None,
                                            shard=j[9] if len(j) > 9 else None), jobs))
    parts.append(synth_audio(scale))
    for j, p in zip(jobs + [("synth",)], parts):
        log(f"  {j[0]}: {len(p)}")
    return [x for p in parts for x in p]


# ------------------------------------------------------------------ images

def _places_names():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("haideraltahan/wds_places365", "classnames.txt", repo_type="dataset")
    return [x.strip() for x in open(path) if x.strip()]


def image_sources(s):
    from timm.data import ImageNetInfo
    n = lambda k: max(1, int(k * s))
    info = ImageNetInfo()
    places = _places_names()
    inet = lambda r, f: [info.label_name_to_description(info.index_to_label_name(r["label"])).split(",")[0]]
    return [
        ("imagenet", "ILSVRC/imagenet-1k", None, "validation", n(8000), "image", inet, None, 256),
        ("coco", "detection-datasets/coco", None, "val", n(3000), "image",
         lambda r, f: sorted({f["objects"]["category"].feature.names[c] for c in r["objects"]["category"]}) + ["photo scene"], None),
        ("places365", "haideraltahan/wds_places365", None, "test", n(3000), "0.webp",
         lambda r, f: [places[int(r["cls"])]], None),
        ("sun397", "tanganke/sun397", None, "train", n(2000), "image", lambda r, f: [_names(f, "label")[r["label"]]], None),
        ("cityscapes", "Chris1/cityscapes", None, "validation", n(300), "image", lambda r, f: ["street scene", "dashcam"], None),
        ("food101", "ethz/food101", None, "validation", n(1500), "image", lambda r, f: [_names(f, "label")[r["label"]], "food"], None),
        ("pets", "visual-layer/oxford-iiit-pet-vl-enriched", None, "train", n(800), "image",
         lambda r, f: [r["label_breed"], r["label_cat_dog"]], lambda r: r["caption_enriched"]),
        ("flowers", "nelorth/oxford-flowers", None, "test", n(600), "image",
         lambda r, f: ["flower", f"flower class {_names(f, 'label')[r['label']]}"], None),
        ("cars", "tanganke/stanford_cars", None, "test", n(1000), "image", lambda r, f: [_names(f, "label")[r["label"]], "car"], None),
        ("aircraft", "Multimodal-Fatima/FGVC_Aircraft_test", None, "test", n(800), "image",
         lambda r, f: [_names(f, "family")[r["family"]], _names(f, "manufacturer")[r["manufacturer"]], "aircraft"], None),
        ("dtd", "tanganke/dtd", None, "test", n(1000), "image", lambda r, f: [_names(f, "label")[r["label"]], "texture"], None),
        ("quickdraw", "Xenova/quickdraw-small", None, "test", n(2000), "image",
         lambda r, f: [_names(f, "label")[r["label"]], "doodle"], None),
        ("eurosat", "blanchon/EuroSAT_RGB", None, "test", n(500), "image",
         lambda r, f: [_names(f, "label")[r["label"]], "satellite image"], None),
        ("xray", "hf-vision/chest-xray-pneumonia", None, "test", n(300), "image",
         lambda r, f: [_names(f, "label")[r["label"]], "chest X-ray"], None),
        ("mnist", "ylecun/mnist", None, "test", n(500), "image", lambda r, f: [f"digit {r['label']}", "handwritten digit"], None),
        ("rico", "rootsautomation/RICO-Screen2Words", None, "test", n(600), "image",
         lambda r, f: ["app screen", r.get("category") or ""], lambda r: (r.get("captions") or [None])[0]),
        ("flickr8k", "jxie/flickr8k", None, "test", n(500), "image", lambda r, f: ["captioned photo"], lambda r: r["caption_0"]),
        ("wikiart", "huggan/wikiart", None, "train", n(1500), "image",
         lambda r, f: [_names(f, "style")[r["style"]], _names(f, "genre")[r["genre"]], "painting"], None),
        ("docvqa", "pixparse/docvqa-single-page-questions", None, "validation", n(500), "image", lambda r, f: ["document"], None),
        ("chartqa", "HuggingFaceM4/ChartQA", None, "val", n(500), "image", lambda r, f: ["chart"], None),
        ("caltech101", "dpdl-benchmark/caltech101", None, "train", n(500), "image", lambda r, f: [_names(f, "label")[r["label"]]], None),
        ("anime", "huggan/anime-faces", None, "train", n(600), "image", lambda r, f: ["anime face", "anime"], None),
        ("cartoon", "Norod78/cartoon-blip-captions", None, "train", n(600), "image", lambda r, f: ["cartoon"], lambda r: r["text"]),
        ("sprite", "tkarr/sprite_caption_dataset", None, "train", n(600), "image", lambda r, f: ["pixel-art sprite"], lambda r: r["text"]),
        ("logo", "logo-wizard/modern-logo-dataset", None, "train", n(500), "image", lambda r, f: ["logo"], lambda r: r["text"]),
        ("emoji", "valhalla/emoji-dataset", None, "train", n(500), "image", lambda r, f: ["emoji"], lambda r: r["text"]),
        ("newyorker", "jmhessel/newyorker_caption_contest", "explanation", "test", n(300), "image",
         lambda r, f: ["single-panel cartoon"], lambda r: r["image_description"]),
        # gaps: people, scene text, handwriting, equations, diagrams, web pages, tables, AI-generated images,
        # fine-grained birds, signs, fashion, medical, astronomy, aerial, underwater
        ("actions", "Bingsu/Human_Action_Recognition", None, "test", n(1000), "image",
         lambda r, f: [_names(f, "labels")[r["labels"]].replace("_", " "), "people"], None),
        ("textvqa", "lmms-lab/textvqa", None, "validation", n(800), "image", lambda r, f: ["scene text"], None, 64),
        ("iam", "Teklia/IAM-line", None, "test", n(500), "image", lambda r, f: ["handwritten text line", "handwriting"],
         lambda r: r["text"]),
        ("letters", "pittawat/letter_recognition", None, "train", n(500), "image",
         lambda r, f: [f"handwritten letter {_names(f, 'label')[r['label']]}", "handwriting"], None),
        ("im2latex", "yuntian-deng/im2latex-100k", None, "test", n(500), "image", lambda r, f: ["typeset equation"],
         lambda r: r["formula"]),
        ("ai2d", "lmms-lab/ai2d", None, "test", n(600), "image", lambda r, f: ["science diagram"], None, 64),
        ("websight", "HuggingFaceM4/WebSight", "v0.2", "train", n(600), "image", lambda r, f: ["web page"], None, 64),
        ("pubtabnet", "apoidea/pubtabnet-html", None, "validation", n(400), "image", lambda r, f: ["table"], None, 64),
        ("aigen", "ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions", None, "train", n(800), "jpg",
         lambda r, f: ["AI-generated image"], None, 64),
        ("birds", "bentrevett/caltech-ucsd-birds-200-2011", None, "test", n(1200), "image",
         lambda r, f: [_names(f, "label")[r["label"]].split(".", 1)[-1].replace("_", " "), "bird"], None),
        ("signs", "tanganke/gtsrb", None, "test", n(800), "image", lambda r, f: [_names(f, "label")[r["label"]], "traffic sign"], None),
        ("products", "ashraq/fashion-product-images-small", None, "train", n(800), "image",
         lambda r, f: [r["articleType"], r["masterCategory"], "product photo"], None),
        ("fashionmnist", "zalando-datasets/fashion_mnist", None, "test", n(300), "image",
         lambda r, f: [_names(f, "label")[r["label"]], "clothing"], None),
        ("skin", "marmal88/skin_cancer", None, "test", n(500), "image", lambda r, f: [f"skin lesion {r['dx']}", "dermatology"], None),
        ("histology", "1aurent/PatchCamelyon", None, "test", n(400), "image",
         lambda r, f: ["histopathology", "tumour tissue" if r["label"] else "normal tissue"], None, 64),
        ("galaxy", "matthieulel/galaxy10_decals", None, "test", n(600), "image",
         lambda r, f: [f"galaxy {GALAXY10[int(r['label'])]}", "galaxy"], None),
        ("aerial", "blanchon/UC_Merced", None, "train", n(800), "image",
         lambda r, f: [_names(f, "label")[r["label"]], "aerial image"], None),
        ("underwater", "Francesco/aquarium-qlnqy", None, "test", n(300), "image", lambda r, f: ["underwater scene"], None),
        ("slides", "NTT-hil-insight/SlideVQA", None, "val", n(500), "page_3", lambda r, f: ["presentation slide"], None, 16),
        # faces: expressions, and CelebA's visible attributes (no identities; its subjective "Attractive" left out)
        ("expressions", "LaurenGurgiolo/9_Facial_Expressions", None, "train", n(900), "image",
         lambda r, f: [f"{_names(f, 'label')[r['label']].split(' ', 1)[-1].lower()} face", "facial expression", "face"], None,
         "folder"),  # image folders sorted by class: sampled from the file list, the classes in turn
        ("expressions", "gabriellidenor/facial_expression_seven_classes", None, "train", n(700), "image",
         lambda r, f: [f"{_names(f, 'label')[r['label']]} face", "facial expression", "face"], None, "folder"),
        ("celeba", "huggan/CelebA-faces-with-attributes", None, "train", n(1500), "image",
         lambda r, f: [k.replace("_", " ").lower() for k in CELEBA if r.get(k) in (1, True)] + ["face"], None, 64),
    ]


SCRIPTS = {  # script: (font file glob, sample characters)
    "Latin": ("NotoSans-Regular", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "Greek": ("NotoSans-Regular", "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΘΛΞΠΣΦΨΩ"),
    "Cyrillic": ("NotoSans-Regular", "абвгдежзийклмнопрстуфхцчшщъыьэюяБГДЖЗИЛПФЦЧШЩЭЮЯ"),
    "Arabic": ("NotoSansArabic-Regular", "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"),
    "Hebrew": ("NotoSansHebrew-Regular", "אבגדהוזחטיכלמנסעפצקרשת"),
    "Devanagari": ("NotoSansDevanagari-Regular", "अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह"),
    "Bengali": ("NotoSansBengali-Regular", "অআইঈউঊএঐওঔকখগঘচছজঝটঠডঢণতথদধনপফবভমযরলশষসহ"),
    "Tamil": ("NotoSansTamil-Regular", "அஆஇஈஉஊஎஏஐஒஓஔகஙசஞடணதநபமயரலவழளறன"),
    "Thai": ("NotoSansThai-Regular", "กขคงจฉชซญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"),
    "Georgian": ("NotoSansGeorgian-Regular", "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ"),
    "Armenian": ("NotoSansArmenian-Regular", "աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆ"),
    "Ethiopic": ("NotoSansEthiopic-Regular", "ሀለሐመሠረሰሸቀበተቸኀነኘአከኸወዐዘዠየደጀገጠጨጰጸፀፈፐ"),
    "Chinese": ("NotoSansCJK-Regular", "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年"),
    "Japanese kana": ("NotoSansCJK-Regular", "あいうえおかきくけこさしすせそたちつてとアイウエオカキクケコサシスセソ"),
    "Korean Hangul": ("NotoSansCJK-Regular", "가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허"),
}


def _font(glob, size):
    import glob as g
    from PIL import ImageFont
    paths = sorted(g.glob(f"/usr/share/fonts/**/{glob}*", recursive=True))
    return ImageFont.truetype(paths[0], size) if paths else ImageFont.load_default()


def synth_images(scale: float = 1.0, seed: int = 0):
    """Exactly labelled synthetic images, 384×384."""
    import colorsys
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    out = []
    reps = max(1, int(round(2 * scale)))
    y, x = np.mgrid[0:SIDE, 0:SIDE] / SIDE

    def add(arr_or_im, labels):
        im = arr_or_im if isinstance(arr_or_im, Image.Image) else Image.fromarray(np.clip(arr_or_im * 255, 0, 255).astype(np.uint8))
        out.append((im.convert("RGB"), [f"synth:{lab}" for lab in labels], None, "synth", {"synth": labels}))

    for ang in range(0, 180, 22):
        for freq in (4, 12, 32, 64):
            for ph in range(reps):
                th = np.deg2rad(ang)
                g = 0.5 + 0.5 * np.sin(2 * np.pi * freq * (x * np.cos(th) + y * np.sin(th)) + ph)
                add(np.stack([g] * 3, -1), [f"grating {ang} degrees", f"spatial frequency {freq}", "grating"])
    hues = ["red", "orange", "yellow", "chartreuse", "green", "spring green", "cyan", "azure", "blue", "violet", "magenta", "rose"]
    for i, h in enumerate(hues):
        for light in (0.25, 0.5, 0.75):
            rgb = colorsys.hls_to_rgb(i / 12, light, 1.0)
            add(np.ones((SIDE, SIDE, 3)) * rgb, [f"solid {h}", "solid colour", f"lightness {light}"])
    for name, g in (("horizontal gradient", x), ("vertical gradient", y), ("radial gradient", np.hypot(x - 0.5, y - 0.5) * 1.4)):
        for _ in range(reps):
            c1, c2 = rng.random(3), rng.random(3)
            add(c1 * (1 - g[..., None]) + c2 * g[..., None], [name, "gradient"])
    for k in (2, 4, 8, 16):
        for _ in range(reps):
            add(np.stack([((x * k).astype(int) + (y * k).astype(int)) % 2] * 3, -1).astype(float), [f"checkerboard {k}", "checkerboard"])
    for _ in range(4 * reps):
        add(rng.random((SIDE, SIDE, 3)), ["random noise", "noise image"])
        add(np.stack([rng.random((SIDE, SIDE))] * 3, -1), ["grey noise", "noise image"])
    shapes = ("circle", "square", "triangle", "star", "ring", "line")
    for sh in shapes:
        for _ in range(4 * reps):
            fg, bg = tuple(int(c) for c in rng.integers(0, 256, 3)), tuple(int(c) for c in rng.integers(0, 256, 3))
            im = Image.new("RGB", (SIDE, SIDE), bg)
            d = ImageDraw.Draw(im)
            cx, cy, r = rng.integers(120, 264), rng.integers(120, 264), rng.integers(50, 110)
            if sh == "circle":
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
            elif sh == "square":
                d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fg)
            elif sh == "triangle":
                d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fg)
            elif sh == "star":
                pts = [(cx + (r if k % 2 == 0 else r / 2.5) * math.cos(math.pi * k / 5 - math.pi / 2),
                        cy + (r if k % 2 == 0 else r / 2.5) * math.sin(math.pi * k / 5 - math.pi / 2)) for k in range(10)]
                d.polygon(pts, fill=fg)
            elif sh == "ring":
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fg, width=14)
            else:
                d.line([(cx - r, cy - r), (cx + r, cy + r)], fill=fg, width=12)
            add(im, [f"shape {sh}", "geometric shape"])
    big = _font("NotoSans-Regular", 260)
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789":
        im = Image.new("RGB", (SIDE, SIDE), "white")
        ImageDraw.Draw(im).text((SIDE // 2, SIDE // 2), ch, fill="black", font=big, anchor="mm")
        add(im, [f"glyph {ch}", "digit" if ch.isdigit() else "Latin letter", "single character"])
    for script, (glob, chars) in SCRIPTS.items():
        font = _font(glob, 44)
        for _ in range(8 * reps):
            im = Image.new("RGB", (SIDE, SIDE), "white")
            d = ImageDraw.Draw(im)
            for row in range(6):
                word = " ".join("".join(pyrng.choice(chars) for _ in range(pyrng.randint(2, 6))) for _ in range(3))
                d.text((16, 20 + row * 60), word, fill="black", font=font)
            add(im, [f"{script} script", "text"])
    mono = _font("NotoSansMono-Regular", 22)
    code = ["def f(x):", "    return x * 2", "for i in range(10):", "    print(i)", "if (a > b) {", "  return a;", "}", "import numpy as np",
            "x = np.zeros((3, 3))", "class Foo:", "    pass", "SELECT * FROM t;", "let y = [1, 2, 3];"]
    maths = ["∫ x² dx = x³/3 + C", "e^{iπ} + 1 = 0", "∑ 1/n² = π²/6", "∇·E = ρ/ε₀", "a² + b² = c²", "lim x→0 sin x / x = 1",
             "det(A − λI) = 0", "∂f/∂x = 2xy"]
    sym = _font("NotoSansMath-Regular", 34)
    for _ in range(12 * reps):
        im = Image.new("RGB", (SIDE, SIDE), "white")
        d = ImageDraw.Draw(im)
        for row in range(12):
            d.text((14, 12 + row * 30), pyrng.choice(code), fill="black", font=mono)
        add(im, ["source code", "text"])
        im = Image.new("RGB", (SIDE, SIDE), "white")
        d = ImageDraw.Draw(im)
        for row in range(6):
            d.text((14, 20 + row * 58), pyrng.choice(maths), fill="black", font=sym)
        add(im, ["maths notation", "text"])
    return out


def manga_pages(n, seed=0):
    """``n`` Manga109-s pages, spread over its titles, read from the release zip (the repo has no table to stream)."""
    import zipfile

    from huggingface_hub import hf_hub_download
    from PIL import Image
    path = hf_hub_download("hal-utokyo/Manga109-s", "Manga109s_released_2026_05_21.zip", repo_type="dataset")
    out = []
    with zipfile.ZipFile(path) as z:
        pages = sorted(x for x in z.namelist() if x.lower().endswith((".jpg", ".jpeg", ".png")) and "/images/" in x)
        if not pages:
            pages = sorted(x for x in z.namelist() if x.lower().endswith((".jpg", ".jpeg", ".png")))
        rng = random.Random(seed)
        pick = rng.sample(pages, min(n, len(pages)))
        for p in pick:
            im = _image(Image.open(io.BytesIO(z.read(p))), letterbox=True)
            if im is not None:
                out.append((im, ["manga:manga page", "manga:comic"], None, "manga", {"repo": "hal-utokyo/Manga109-s", "file": p}))
    return out


def degraded(photos, seed=0):
    """Low-light and damaged versions of photographs (``(image, ref)``), exactly labelled (``synth:``)."""
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(seed)
    out = []
    for k, (im, ref) in enumerate(photos):
        ref = {**(ref or {}), "degraded": ("low light", "blurred", "JPEG artefacts", "motion blur")[k % 4]}
        a = np.asarray(im).astype(np.float32) / 255
        kind = k % 4
        if kind == 0:  # night: dark, blue-shifted, sensor noise
            b = np.clip(a ** 2.2 * 0.25 * np.array([0.8, 0.9, 1.1]) + rng.normal(0, 0.03, a.shape), 0, 1)
            out.append((Image.fromarray((b * 255).astype(np.uint8)), ["synth:low light", "synth:degraded photo"], None, "synth", ref))
        elif kind == 1:
            out.append((im.filter(ImageFilter.GaussianBlur(6)), ["synth:blurred", "synth:degraded photo"], None, "synth", ref))
        elif kind == 2:
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=4)
            out.append((Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), ["synth:JPEG artefacts", "synth:degraded photo"],
                        None, "synth", ref))
        else:  # horizontal motion blur
            ker = np.ones(25) / 25
            b = np.stack([np.apply_along_axis(lambda r: np.convolve(r, ker, "same"), 1, a[..., c]) for c in range(3)], -1)
            out.append((Image.fromarray((np.clip(b, 0, 1) * 255).astype(np.uint8)), ["synth:motion blur", "synth:degraded photo"],
                        None, "synth", ref))
    return out


def image_collection(scale: float = 1.0, log=print, workers: int | None = None):
    jobs = image_sources(scale)
    with ThreadPoolExecutor(workers or len(jobs)) as ex:  # every source at once: the slowest bound the load
        parts = list(ex.map(lambda j: _take(*j[:8], "image", buffer=j[8] if len(j) > 8 else None), jobs))
    coco = next((p for j, p in zip(jobs, parts) if j[0] == "coco"), [])
    try:
        manga = manga_pages(max(2, int(800 * scale)))
    except Exception as e:  # a gated or changed release must not sink the collection
        print(f"  manga: FAILED {e!r}"[:200], flush=True)
        manga = []
    log(f"  manga: {len(manga)}")
    parts.append(synth_images(scale) + degraded([(x[0], x[4]) for x in coco[:max(4, int(1200 * scale))]]) + manga)
    for j, p in zip(jobs + [("synth",)], parts):
        log(f"  {j[0]}: {len(p)}")
    return [x for p in parts for x in p]
