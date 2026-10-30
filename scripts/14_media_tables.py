"""Gemma 4 12B's image and audio tables: every neuron pole read through the media lenses, beside the
real images and sounds that drive it (the media counterpart of ``neurons/reads`` and ``neurons/acts``).

Per modality (``image``, ``audio``), one table per layer, ``neurons/<modality>/L<layer>.parquet``, a
row per pole with ``layer``, ``neuron``, ``pole`` in front, and no forward pass:

- ``read_S_basis``: the pole's read direction ``â ± b̂`` (gate and up rows times the pre-feedforward
  norm, normalized, as ``08_reads.py``) through the modality's released S-lens to the layer's MLP
  input, ``c · K^mid_L``: its coordinates (fp16) on the S-lens ``basis`` (``[k, d]``, orthonormal rows:
  images 2,048, the span of the patch embedder; audio 640, exact). ``u = coords @ basis`` is the
  embedding covector the pole reads; through the embedder it is
  - ``read_png`` (images): the 48 × 48 RGB patch the pole responds to (a pixel covector, centred
    per patch as the embedder's norms do), shown at ±3 standard deviations around grey;
  - ``read_wave`` (audio): the 40 ms waveform the pole responds to, 640 samples at 16 kHz (fp16).
  The S-lens sums over a document's slots (64 patches or frames), so the read is one pattern
  wherever it sits, not a map of where;
- ``write_R``, ``write_R_z``: the pole's write (``s · γ_post ⊙ W_down[:, n]``) through the
  modality's released R-lens (fitted on image or audio documents, from the first slot), what it
  makes the model say there: the top 64 kept tokens with the model's probabilities (after the final
  softcap; the logit lens at the last layer), and by layer-contrastive z;
- the exemplars (``13_media_activations.py``'s file for the modality, the mean statistic):
  ``label``, ``label_z``, ``label_mean`` (the top label by z, its z and the label's mean activation),
  ``labels_str`` (the top 8, `` | ``-joined), ``sig`` (above the shuffled-label null's 99th
  percentile for the layer), ``stable`` (in the top 5 of both halves), ``auroc_heldout`` (chosen on
  even documents, scored on odd), and ``top_docs``, ``top_vals``: the 16 documents with the largest
  mean activation, rows of ``<modality>_docs.parquet``;
- across modalities (``checks/crossmodal.py``, for poles with a reliable concept label): the label's
  rank among the modality's labels on the text side, 1 first and 0.5 chance: ``text_fire_rank``
  (the pole's activation on the label's words), ``text_read_rank`` (its read through the text
  S-lens) and ``text_write_rank`` (its write through the text R-lens); NaN where not scored.

``<modality>_docs.parquet`` (``data/catalog/gemma-4-12b/media/``): each document's source, labels,
caption and ``ref``, where it came from (dataset, config, split, revision, the row's own
identifiers, the media file's path and its bytes' SHA-1; JSON), with ``link``: a page for it where
one follows from the ref (AudioSet's YouTube clip), else the dataset's. The media themselves are
not redistributed.

    NEURON_LENS_MODEL=google/gemma-4-12B python scripts/14_media_tables.py [--modality image,audio] [--layers 0-47]
"""
import argparse
import io
import json
import time

import pyarrow as pa
import torch
from huggingface_hub import hf_hub_download

from neuron_lens import arch
from neuron_lens.catalog import MODEL, TOP, catalog_dir, kept, neuron_keys, put, vocab, write
from neuron_lens.neurons import directions, softcap
from neuron_lens.paths import DATA, SUMMARY
from neuron_lens.readout import layer_reads
from neuron_lens.weights import tensors

MEDIA = DATA / "catalog" / arch.stem(MODEL) / "media"
SIDE = 48  # an image patch is 48 × 48 RGB
SHOW_SD = 3.0  # a read patch's grey scale: ± this many standard deviations


def released(sub: str, file: str) -> dict:
    return torch.load(hf_hub_download("stdlogits/rs-lenses", f"{arch.stem(MODEL)}/{sub}/{file}"),
                      weights_only=True, map_location="cpu", mmap=True)


def embedder(modality: str):
    """The map from an embedding covector ``[n, d]`` to its pixel (``[n, 48, 48, 3]``) or waveform
    (``[n, 640]``) covector, from the checkpoint's weights (as ``neuron_lens.media``'s reads)."""
    if modality == "audio":
        w = tensors(MODEL, ["model.embed_audio.embedding_projection.weight"])["model.embed_audio.embedding_projection.weight"].float()
        return lambda u: u.float() @ w.to(u.device)
    names = ["model.embed_vision.embedding_projection.weight"] + [f"model.vision_embedder.{n}.weight" for n in
                                                                  ("pos_norm", "patch_ln2", "patch_dense", "patch_ln1")]
    w = {k.split(".")[-2]: v.float() for k, v in tensors(MODEL, names).items()}
    centre = lambda v: v - v.mean(-1, keepdim=True)  # noqa: E731

    def to_pixels(u):
        v = u.float() @ w["embedding_projection"].to(u.device)
        v = centre(v * w["pos_norm"].to(u.device))
        v = centre(v * w["patch_ln2"].to(u.device))
        v = v @ w["patch_dense"].to(u.device)
        v = centre(v * w["patch_ln1"].to(u.device))
        return v.reshape(-1, SIDE, SIDE, 3)
    return to_pixels


def pngs(px: torch.Tensor) -> list[dict]:
    """Each pixel covector as a PNG, ± ``SHOW_SD`` standard deviations around grey (HF's image struct)."""
    from PIL import Image
    sd = px.flatten(1).std(1).clamp_min(1e-12)[:, None, None, None]
    img = ((px / (SHOW_SD * sd)).clamp(-1, 1) * 127.5 + 127.5).round().byte().cpu().numpy()

    def one(a):
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="PNG", optimize=True)
        return {"bytes": buf.getvalue(), "path": None}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as ex:  # zlib releases the GIL: 30,720 patches a layer
        return list(ex.map(one, img))


def link(ref: dict | None) -> str | None:
    if not ref or "repo" not in ref:
        return None
    if ref["repo"] == "agkphysics/AudioSet" and ref.get("video_id"):
        return f"https://www.youtube.com/watch?v={ref['video_id']}"
    return f"https://huggingface.co/datasets/{ref['repo']}"


def docs_table(ex: dict, modality: str) -> None:
    refs = ex.get("doc_ref") or [None] * len(ex["doc_source"])
    write({"doc": pa.array(range(len(refs)), pa.int32()), "source": pa.array(ex["doc_source"]),
           "labels": pa.array([" | ".join(x) for x in ex["doc_labels"]]),
           "caption": pa.array([c if isinstance(c, str) else None for c in ex["doc_caption"]]),
           "ref": pa.array([json.dumps(r, default=str, ensure_ascii=False) if r else None for r in refs]),
           "link": pa.array([link(r) for r in refs])}, MEDIA / f"{modality}_docs.parquet")


def summarize(base=None) -> None:
    """The release card's numbers per modality and band, to ``data/summary/media_tables_<model>.json``:
    documents and labels (those on 8+ documents are scored), the share of poles whose top label is
    reliable (``sig`` and ``stable``), the median held-out AUROC, and the share of cross-modally scored
    poles whose label holds on all three text tests. ``base``: the catalog (default local; an
    ``hf://`` path reads the release)."""
    import pandas as pd
    base = base or catalog_dir()
    ranks = ["text_fire_rank", "text_read_rank", "text_write_rank"]
    out = {}
    for m in ("image", "audio"):
        labels = pd.read_parquet(f"{base}/media/{m}_docs.parquet", columns=["labels"])["labels"]
        per = labels.str.split(" | ", regex=False).explode().value_counts()
        out[m] = {"documents": len(labels), "labels": len(per), "labels_scored": int((per >= 8).sum()), "bands": {}}
        for lo, hi in arch.bands(arch.text_config(MODEL).num_hidden_layers):
            t = pd.concat([pd.read_parquet(f"{base}/neurons/{m}/L{layer:02d}.parquet", columns=["sig", "stable", "auroc_heldout", *ranks])
                           for layer in range(lo, hi + 1)])
            scored = t[ranks[0]].notna()
            out[m]["bands"][f"L{lo}-L{hi}"] = {"reliable": float((t["sig"] & t["stable"]).mean()),
                                              "auroc_heldout": float(t["auroc_heldout"].median()),
                                              "text_all_three": float((t[ranks] >= 0.95).all(axis=1)[scored].mean()),
                                              "scored": int(scored.sum())}
    (SUMMARY / f"media_tables_{arch.stem(MODEL)}.json").write_text(json.dumps(out, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="image,audio")
    ap.add_argument("--layers", default=None, help="a-b (default all)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    keep, v = kept(), vocab()
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted({nm["unembed"], nm["norm"]}))
    unembed = head[nm["unembed"]][keep].to(dev, torch.float16)
    gamma = head[nm["norm"]].to(dev, torch.float32)
    del head
    cap = softcap(MODEL)
    L = arch.text_config(MODEL).num_hidden_layers
    lo, hi = (0, L - 1) if args.layers is None else map(int, args.layers.split("-"))
    for modality in args.modality.split(","):
        t_mod = time.perf_counter()
        S = released(f"{modality}/s-lens", "mid.pt")  # K[L] [d, k], basis [k, d]
        basis = S["basis"].float().to(dev)
        J = released(f"{modality}/r-lens", "lens.pt")["J"]  # the identity at the last layer
        to_media = embedder(modality)
        ex = torch.load(MEDIA / f"{modality}_exemplars.pt", weights_only=False, map_location="cpu", mmap=True)
        docs_table(ex, modality)
        labels = [x.split(":", 1)[-1] for x in ex["labels"]]
        xm_path = MEDIA / "crossmodal_poles.pt"
        xm = torch.load(xm_path, weights_only=False).get(modality, {}) if xm_path.exists() else {}
        for layer in range(lo, hi + 1):
            t0 = time.perf_counter()
            w = directions(MODEL, layer, dev)
            wd = w["write"]
            f = wd.shape[1]
            a, b = w["gate"], w["up"]
            a, b = a / a.norm(dim=1, keepdim=True), b / b.norm(dim=1, keepdim=True)
            del w
            c = torch.cat([a + b, a - b])  # [2f, d]: the read directions, + poles then −
            coords = c @ S["K"][layer].float().to(dev)  # [2f, k]
            cols = neuron_keys(layer, f)
            cols["read_S_basis"] = pa.FixedSizeListArray.from_arrays(pa.array(coords.half().cpu().flatten().numpy()), coords.shape[1])
            u = coords @ basis  # [2f, d]: the embedding covector each pole reads
            if modality == "image":
                cols["read_png"] = pa.array([x for s in range(0, 2 * f, 4096) for x in pngs(to_media(u[s:s + 4096]))],
                                            pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
            else:
                wave = to_media(u)
                cols["read_wave"] = pa.FixedSizeListArray.from_arrays(pa.array(wave.half().cpu().flatten().numpy()), wave.shape[1])
            r = layer_reads(wd, unembed, gamma, J[layer].float() if layer < L - 1 else None, k=TOP, kz=TOP, softcap=cap)
            put(cols, "write_R", keep[torch.cat([r["ids"][1], r["ids"][-1]])], torch.cat([r["probs"][1], r["probs"][-1]]))
            put(cols, "write_R_z", keep[torch.cat([r["zids"][1], r["zids"][-1]])], torch.cat([r["z"][1], r["z"][-1]]))
            e = {p: ex["mean"][p] for p in (1, -1)}
            lab = torch.cat([e[p]["labels"][layer] for p in (1, -1)]).long()  # [2f, 32]
            cols["label"] = pa.array([labels[i] for i in lab[:, 0].tolist()])
            cols["label_z"] = pa.array(torch.cat([e[p]["z"][layer][:, 0] for p in (1, -1)]).float().numpy())
            cols["label_mean"] = pa.array(torch.cat([e[p]["mean"][layer][:, 0] for p in (1, -1)]).float().numpy())
            cols["labels_str"] = pa.array([" | ".join(labels[i] for i in row[:8]) for row in lab.tolist()])
            for k in ("sig", "stable"):
                cols[k] = pa.array(torch.cat([e[p][k][layer] for p in (1, -1)]).numpy())
            cols["auroc_heldout"] = pa.array(torch.cat([e[p]["auroc_heldout"][layer] for p in (1, -1)]).float().numpy())
            tv = {p: ex["top_docs"][p] for p in (1, -1)}
            cols["top_docs"] = pa.FixedSizeListArray.from_arrays(
                pa.array(torch.cat([tv[p][1][layer] for p in (1, -1)]).int().flatten().numpy()), tv[1][1].shape[-1])
            cols["top_vals"] = pa.FixedSizeListArray.from_arrays(
                pa.array(torch.cat([tv[p][0][layer] for p in (1, -1)]).float().flatten().numpy()), tv[1][0].shape[-1])
            for key, col in (("fire", "text_fire_rank"), ("s_read", "text_read_rank"), ("r_write", "text_write_rank")):
                cols[col] = pa.array([xm.get((layer, n, p), {}).get(key, float("nan")) for p in (1, -1) for n in range(f)],
                                     pa.float32())
            write(cols, catalog_dir() / "neurons" / modality / f"L{layer:02d}.parquet")
            print(f"{time.strftime('%H:%M:%S')} {modality} L{layer}: {2 * f} poles in {time.perf_counter() - t0:.0f}s", flush=True)
        print(f"{modality}: done in {time.perf_counter() - t_mod:.0f}s; write_R e.g. {v[int(keep[r['ids'][1][0, 0]])]!r}", flush=True)
    if args.layers is None and args.modality == "image,audio":
        summarize()


if __name__ == "__main__":
    main()
