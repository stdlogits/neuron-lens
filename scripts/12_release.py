"""Release the lenses and the catalog on the Hugging Face Hub (pipeline stage 12), for the model
``NEURON_LENS_MODEL`` selects: the lenses to a model repo, the catalog to a dataset repo, both from
this checkout's fits and tables. Each repo has one card, covering every released model (``MODELS``),
rebuilt on every run.

**Model, ``stdlogits/rs-lenses``.** Per model, ``<model>/r-lens/lens.pt`` is the stacked-Jacobian
dictionary of ``camilablank/workspace-lenses`` (``J``, ``n_prompts``, ``source_layers``,
``d_model``, ``provenance``, with the target layer's row the identity), which Drowse fetches as
``drowse lens fetch <model> workspace-r --repo <repo>``. ``<model>/r-lens/mid.pt`` is the R-lens
from each layer's MLP input in that layout, and ``<model>/s-lens/{mid,out}.pt`` hold the S-lens to
each layer's MLP input and output (``K``, ``layers``). Gemma 4 12B adds the same under
``image/`` and ``audio/``, fitted on image and audio documents; their S-lenses are seeded in the
subspace the embedder reaches, so ``K[L]`` is ``[d, k]`` with ``basis`` ``[k, d]`` (``K_full = K @
basis``). All are fp32 means over both halves of the documents; ``provenance`` (where the corpus
has a record) holds the split-half difference by layer.

**Dataset, ``stdlogits/neuron-lens``.** Per model, the catalog as it is laid out locally
(``neuron_lens.catalog``), one viewer config per table: ``neurons/<table>/L<layer>.parquet`` and
``heads/<table>.parquet``, and ``docs.parquet``, each activation document's source. Beside them, as
plain files, ``token_counts.pt`` and the causal sample, ``causal/causal.pt``. Gemma 4 12B adds the
image and audio tables (``14_media_tables.py``, ``15_media_optimised.py``, ``checks/head_media.py``)
and ``media/<modality>_docs.parquet``, each image's or sound's source row (the media are not
re-hosted). A full upload deletes the model's files that are no longer built.

The cards embed figures from ``docs/figures/`` (``11_figures.py``), uploaded to ``figures/`` in each
repo, and numbers from ``data/summary/``. What a card needs to know about a model's release (row
counts, documents, the lenses' fits) is kept in ``data/summary/release_<model>.json``, updated by
each run that builds that model's files, so a run for one model rebuilds the cards for all.
New repos are created private. Run where the fits and the catalog are, logged in to the Hub.
``--parts`` picks what to build and upload; ``--parts ""`` updates just the cards and figures:

    NEURON_LENS_MODEL=... python scripts/12_release.py [--parts lenses,neurons,heads,files] [--dry-run]
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi

from neuron_lens import arch
from neuron_lens.catalog import MODEL, catalog_dir, kept
from neuron_lens.paths import CORPUS_DIR, DATA, FITS, ROOT, SUMMARY
from neuron_lens.rules import HALF_RULE_BETA
from neuron_lens.transport import SKIP_FIRST, Accumulator

REPO = "stdlogits/rs-lenses"
DATASET = "stdlogits/neuron-lens"
MODELS = ("Qwen/Qwen3-8B", "google/gemma-4-12B")  # the released models, in the cards' order
PARTS = ("lenses", "neurons", "heads", "files")  # files: docs.parquet, token_counts.pt, the causal sample, media docs
MEDIA = {"image": "coco", "audio": "esc50"}  # Gemma 4's media lenses and the corpus each is fitted on
MEDIA_WHAT = {"image": ("COCO 2014 validation images", "image"), "audio": ("ESC-50 clips", "sound")}
FIGURES = ROOT / "docs" / "figures"
MODEL_FIGURES = ("02_lenses.png", "04_lens_reliability.png", "05_chat_lens.png", "07_steering.png", "11_drivers.png",
                 "16_head_lens_sides.png", "19_gemma_steering.png", "21_media_reads.png")
DATASET_FIGURES = ("01_anatomy.png", "03_depth_map.png", "18_two_models.png", "13_rank_depth.png", "06_activations.png",
                   "11_drivers.png", "12_census.png", "20_gemma_census.png", "14_lens_blind.png", "22_media_gallery.png",
                   "23_crossmodal.png", "17_head_census.png", "24_heads_media.png")
TABLES = {"neurons": {"acts": "every pole's activation statistics and top firings",
                      "drivers": "every pole's 64 token drivers in real contexts",
                      "reads": "every pole's weight-only reads through both lenses",
                      "legible": "which reads hold at every pole's top firings",
                      "labels": "every pole's labels, by Claude Haiku 5.5",
                      "image": "every pole's image read and write, the images that drive it, and its cross-modal ranks",
                      "audio": "every pole's audio read and write, the sounds that drive it, and its cross-modal ranks"},
          "heads": {"acts": "every head's output statistics, top firings and attention patterns",
                    "drivers": "every head's 64 token drivers, as query and as key",
                    "reads": "every head's OV and QK circuits read per token",
                    "legible": "which reads hold at every head's top firings",
                    "labels": "every head's labels, by Claude Haiku 5.5",
                    "media": "every head in image and audio contexts"}}
OPTIMISED = {"image": "the patch that most raises each of the best-labelled poles",
             "audio": "the 40 ms of sound that most raises each of the best-labelled poles"}
RULES = {"ln_rule": "decoder-layer RMSNorm rsqrt held constant", "identity_rule": "act(g) as (act(g)/g)·dg",
         "half_rule": HALF_RULE_BETA, "qk_norms": "true derivative", "attention": "true derivative"}


def provenance(meta: dict, acc: Accumulator, target: str, **extra) -> dict | None:
    """A lens file's record of its fit, or ``None`` for a corpus with no provenance record (the media
    lenses: their documents are COCO's and ESC-50's first rows, ``neuron_lens.media``)."""
    path = CORPUS_DIR / meta["corpus"].replace(".jsonl", ".provenance.json")
    if not path.exists():
        return None
    corpus = json.loads(path.read_text())
    config = {"estimator": "relp", "rules": RULES, "lens": meta["lens"], "target": target,
              "positions": f"summed over offsets, seeded at positions [{SKIP_FIRST}, T-1)",
              "documents": "equal weight (per-document means)", "fit_dtype": meta["dtype"],
              "code": "neuron-lens (neuron_lens/transport.py)"}
    return {"model_id": meta["model"], "dataset_id": f"{corpus['repo']}@{corpus['revision']}",
            "rows": corpus["rows"], "rows_sha256": meta["rows_sha256"], "n_prompts": sum(acc.n_docs),
            "t_max": meta["seq_len"], "skip_first": SKIP_FIRST, "config_json": json.dumps(config),
            "split_half": [round(x, 4) for x in acc.split_half(target)], "date": time.strftime("%Y-%m-%d"),
            **extra}


def spread(xs: list[float]) -> str:
    return f"{min(xs):.2f}-{max(xs):.2f}"


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def link(src: Path, dst: Path) -> None:
    """Hard links: the release as built, without a second copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    os.link(src, dst)


def summary(name: str):
    return json.loads((SUMMARY / name).read_text())


# ---------------------------------------------------------------- building a model's files

def write_lenses(model: str, out: Path) -> dict:
    """Write the model's lens files under ``out`` (text, and for Gemma 4 image and audio) from its fits
    in ``data/fits/``; return what the card says about them."""
    stem, n_layers = arch.stem(model), arch.text_config(model).num_hidden_layers
    last = n_layers - 1
    facts = {}
    for sub, corpus in [("", "dclm")] + ([(f"{m}/", c) for m, c in MEDIA.items()] if arch.is_gemma4(model) else []):
        r = torch.load(FITS / f"r-{stem}-L{n_layers}-{corpus}.pt", weights_only=False, map_location="cpu", mmap=True)
        s = torch.load(FITS / f"s-{stem}-L{n_layers}-{corpus}.pt", weights_only=False, map_location="cpu", mmap=True)
        r_acc, s_acc = Accumulator.from_state(r["state"]), Accumulator.from_state(s["state"])
        if sub == "" and r["meta"]["rows_sha256"] != s["meta"]["rows_sha256"]:
            raise ValueError(f"{model}: the R and S fits read different documents")
        (out / sub / "r-lens").mkdir(parents=True, exist_ok=True)
        (out / sub / "s-lens").mkdir(parents=True, exist_ok=True)
        J = r_acc.mean("out")  # [layers - 1, d, d]: sources 0 .. n-2
        d = J.shape[-1]
        jac = {layer: J[layer].clone() for layer in range(J.shape[0])} | {last: torch.eye(d)}
        prov = provenance(r["meta"], r_acc, "out", target_layer=last)
        torch.save({"J": jac, "n_prompts": sum(r_acc.n_docs), "source_layers": sorted(jac), "d_model": d}
                   | ({"provenance": prov} if prov else {}), out / sub / "r-lens" / "lens.pt")
        del J, jac
        J = r_acc.mean("mid")  # [layers, d, d]: each layer's MLP input
        prov = provenance(r["meta"], r_acc, "mid", target_layer=last)
        torch.save({"J": {layer: J[layer].clone() for layer in range(J.shape[0])}, "n_prompts": sum(r_acc.n_docs),
                    "source_layers": list(range(J.shape[0])), "d_model": d} | ({"provenance": prov} if prov else {}),
                   out / sub / "r-lens" / "mid.pt")
        del J
        basis = s["state"].get("basis")
        for target in ("mid", "out"):
            K = s_acc.mean(target)
            prov = provenance(s["meta"], s_acc, target)
            torch.save({"K": {layer: K[layer].clone() for layer in range(K.shape[0])}, "n_prompts": sum(s_acc.n_docs),
                        "layers": list(range(K.shape[0])), "d_model": d}
                       | ({"basis": basis.clone()} if basis is not None else {}) | ({"provenance": prov} if prov else {}),
                       out / sub / "s-lens" / f"{target}.pt")
            del K
        key = sub.strip("/") or "text"
        facts[key] = {"R_docs": sum(r_acc.n_docs), "S_docs": sum(s_acc.n_docs),
                      "R": spread(r_acc.split_half("out")), "R_mid": spread(r_acc.split_half("mid")),
                      "S_mid": spread(s_acc.split_half("mid")), "S_out": spread(s_acc.split_half("out"))}
        if basis is not None:
            facts[key]["basis"] = int(basis.shape[0])
        if sub == "":
            corpus_rec = json.loads((CORPUS_DIR / r["meta"]["corpus"].replace(".jsonl", ".provenance.json")).read_text())
            facts[key] |= {"corpus": corpus_rec["repo"], "revision": corpus_rec["revision"],
                           "rows_sha": r["meta"]["rows_sha256"][:16], "t_max": r["meta"]["seq_len"]}
        print(f"{time.strftime('%H:%M:%S')} {stem} {key} lenses written", flush=True)
    return facts


def catalog_files(model: str) -> dict:
    """The model's release files by part: ``(local source, path in the model's folder)``."""
    base, stem = catalog_dir(model), arch.stem(model)
    files = {"neurons": [(p, f"neurons/{t}/{p.name}") for t in TABLES["neurons"]
                         for p in sorted((base / "neurons" / t).glob("L*.parquet"))]
                        + [(p, f"neurons/{p.name}") for p in sorted((base / "neurons").glob("*_optimised.parquet"))],
             "heads": [(base / "heads" / f"{t}.parquet", f"heads/{t}.parquet") for t in TABLES["heads"]
                       if (base / "heads" / f"{t}.parquet").exists()],
             "files": [(base / "docs.parquet", "docs.parquet"), (arch.counts_path(model), "token_counts.pt"),
                       (DATA / "causal" / f"{stem}-causal.pt", "causal/causal.pt")]
                      + [(p, f"media/{p.name}") for p in sorted((base / "media").glob("*_docs.parquet"))]}
    return files


def catalog_facts(model: str, root: Path) -> dict:
    """What the card says about the model's staged catalog in ``root``: its viewer configs with row
    counts, its documents and positions, its kept tokens and its causal sample."""
    stem = arch.stem(model)
    configs = []
    for side, tables in TABLES.items():
        for t, what in tables.items():
            paths = sorted((root / side / t).glob("L*.parquet")) if side == "neurons" else [root / side / f"{t}.parquet"]
            paths = [p for p in paths if p.exists()]
            if paths:
                glob = f"{stem}/{side}/{t}/*.parquet" if side == "neurons" else f"{stem}/{side}/{t}.parquet"
                configs.append([f"{stem}-{side}-{t}", glob, what, sum(pq.ParquetFile(p).metadata.num_rows for p in paths)])
        if side == "neurons":
            for m, what in OPTIMISED.items():
                p = root / "neurons" / f"{m}_optimised.parquet"
                if p.exists():
                    configs.append([f"{stem}-neurons-{m}-optimised", f"{stem}/neurons/{m}_optimised.parquet", what,
                                    pq.ParquetFile(p).metadata.num_rows])
    tokens = pq.read_table(root / "docs.parquet", columns=["tokens"]).column(0).to_numpy()
    causal = torch.load(root / "causal" / "causal.pt", weights_only=False, map_location="cpu", mmap=True)
    return {"configs": configs, "docs": len(tokens), "positions": int((tokens - 1).sum()), "kept": len(kept(model)),
            "causal_picks": len(causal["picks"])}


# ---------------------------------------------------------------- the cards

def lens_files_table(f: dict) -> str:
    rows = []
    for x in f.values():
        d, last = x["d_model"], x["layers"] - 1
        rows += [f"| `{x['dir']}/r-lens/lens.pt` | R, `J[L]` | layer `L`'s output → the last layer's output, `[{d}, {d}]`, `J[{last}]` the identity |",
                 f"| `{x['dir']}/r-lens/mid.pt` | R, `J[L]` | layer `L`'s MLP input → the last layer's output |",
                 f"| `{x['dir']}/s-lens/mid.pt` | S, `K[L]` | token embeddings → layer `L`'s MLP input |",
                 f"| `{x['dir']}/s-lens/out.pt` | S, `K[L]` | token embeddings → layer `L`'s output |"]
        for media, (what, unit) in MEDIA_WHAT.items():
            if media in x["lenses"]:
                rows.append(f"| `{x['dir']}/{media}/r-lens/`, `{x['dir']}/{media}/s-lens/` | the same, fitted on {what} | "
                            f"read and seeded at the {unit}s' 64 slots; `K[L]` is `[{d}, {x['lenses'][media]['basis']}]` "
                            f"with `basis` |")
    return "\n".join(rows)


def split_text(t: dict) -> str:
    """A model's split-half differences by target, as its fits recorded them."""
    if "R_mid" in t:
        return (f"R-lens {t['R']} (from each layer's output), {t['R_mid']} (from each MLP input); S-lens {t['S_mid']} "
                f"(to each MLP input), {t['S_out']} (to each output)")
    return f"R-lens {t['R']}, S-lens {t['S']}, by layer over both targets"


def model_card(f: dict, date: str) -> str:
    q, g = f["Qwen/Qwen3-8B"], f["google/gemma-4-12B"]
    qt, gt = q["lenses"]["text"], g["lenses"]["text"]
    gi, ga = g["lenses"]["image"], g["lenses"]["audio"]
    return f"""---
license: cc-by-4.0
base_model:
- Qwen/Qwen3-8B
- google/gemma-4-12B
tags:
- interpretability
- jacobian-lens
- logit-lens
- relp
- r-lens
- s-lens
---

# R-lens and S-lens for Qwen3-8B and Gemma 4 12B

Linear lenses that turn a vector inside a model into a ranking over tokens, for
[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) and
[`google/gemma-4-12B`](https://huggingface.co/google/gemma-4-12B), fitted with one estimator on the
same {qt['R_docs']:,} web documents:

- the **R-lens** reads what a residual-stream vector at layer `L` makes the model say, after the later
  layers have acted on it (read with the unembedding `W_U`);
- the **S-lens** reads which inputs put a vector at layer `L` (read with the embeddings `W_E`).

Gemma 4 takes images and sound through linear embedders, so its lenses are also fitted on images and on
sounds, where the S-lens reads back to pixels or audio samples. Every MLP neuron and attention head of
both models read through these lenses is the dataset
[`{DATASET}`](https://huggingface.co/datasets/{DATASET}). Code, methods and findings (cited here as
§N) are on [GitHub](https://github.com/stdlogits/neuron-lens).

![The logit lens, the J-lens and R-lens, and the K-lens and S-lens](figures/02_lenses.png)

## Files

| file | lens | maps |
|---|---|---|
{lens_files_table(f)}

Each file is a dict of fp32 matrices by layer: `[4096, 4096]` for Qwen3-8B's 36 layers (2.4 GB a file),
`[3840, 3840]` for Gemma 4's 48. The "MLP input" is the residual after the layer's attention, where the
MLP reads. `lens.pt` follows [`camilablank/workspace-lenses`](https://huggingface.co/camilablank/workspace-lenses)'
layout (`J`, `n_prompts`, `source_layers`, `d_model`, and in Qwen3-8B's files `provenance`); the S-lens
files hold `K` and `layers` in its place. `provenance` records the corpus, the rules and the split-half
difference by layer. Gemma's image and audio S-lenses are seeded in the subspace the embedder reaches,
so they hold `basis` (`[k, 3840]`, orthonormal rows) beside `K`: `K_full[L] = K[L] @ basis`.

**Which file reads what** (the lens rule): `out` reads MLP outputs and attention inputs, `mid` reads MLP
inputs and attention outputs.

| to read | at | R-lens | S-lens |
|---|---|---|---|
| an MLP neuron's write | layer `L`'s output | `r-lens/lens.pt` | `s-lens/out.pt` |
| an MLP neuron's read direction (gate and up rows) | layer `L`'s MLP input | `r-lens/mid.pt` | `s-lens/mid.pt` |
| what an attention head reads | layer `L − 1`'s output | | `s-lens/out.pt` |
| what an attention head writes | layer `L`'s MLP input | `r-lens/mid.pt` | `s-lens/mid.pt` |

## Definition

Both average a Jacobian of the residual stream over the corpus, summed over positions as in the
[J-lens](https://transformer-circuits.pub/2026/workspace/), but under
[RelP](https://arxiv.org/abs/2508.21258)'s propagation rules instead of the true gradient:

```
J_L = mean_t Σ_{{t' ≥ t}} ∂h_final(t') / ∂h_L(t)     R-lens, reverse mode
K_L = mean_t Σ_{{s ≤ t}}  ∂h_L(t) / ∂e(s)            S-lens, forward mode
```

- **R-lens read** of a vector `h` at layer `L`: `softmax(W_U · norm(J_L h))`, with `norm` the final
  RMSNorm and its weight (for Gemma 4, the logits then pass its softcap, `30 · tanh(x / 30)`). This is
  the [R-lens](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers),
  with the last layer's output as its target.
- **S-lens read** of a direction `c` at layer `L`: token `v` scores `c · K_L (W_E[v] − mean(W_E))` (for
  Gemma 4, `W_E` times √3840, as embeddings enter its residual). Taking the Jacobian back to the input
  embeddings is Theia Vogel's K-lens ([@voooooogel](https://x.com/voooooogel/status/2084858861551526002),
  unpublished); the S-lens is it under RelP's rules. For images and sound, `c · K_L` is a covector on
  the embedder's output, which the embedder maps back to pixels (48 × 48 × 3 per patch) or samples (640
  per 40 ms frame).
- **Layer-contrastive reads** z-score each token's score across many vectors of one layer (all its
  neurons, say), which removes the hub tokens that top most raw reads.
- **The rules:** decoder-layer RMSNorms hold their `rsqrt` constant; the MLP's activation `act(g)` is
  linearized as `(act(g)/g)·dg` (SiLU for Qwen3, GELU-tanh for Gemma 4); and `act(g)·up` passes half of
  each factor's change (β = {HALF_RULE_BETA}). Attention, the q/k (and Gemma's v) norms and RoPE keep
  their true derivative. Under the rules each layer is a fixed linear map per document, and the
  R-lens's backward map is exactly the transpose of the S-lens's forward one.

## Fitting

- **Text:** for both models, the first {qt['R_docs']:,} rows of
  [`{qt['corpus']}`](https://huggingface.co/datasets/{qt['corpus']}) (revision `{qt['revision']}`), the
  first {qt['t_max']} tokens of each (row-set SHA-256 `{qt['rows_sha']}`… for Qwen3).
- **Gemma 4's images:** COCO 2014 validation images (the first {gi['S_docs']:,} for the S-lens,
  {gi['R_docs']:,} for the R-lens), 384 × 384, 64 slots of 48 × 48 patches, in a short text prompt.
  **Sounds:** ESC-50 clips, the middle 2.56 s at 16 kHz, 64 slots of 40 ms ({ga['S_docs']:,} clips
  for the S-lens, {ga['R_docs']:,} for the R-lens; clips with a digitally silent frame are skipped).
- Seeds at every position from {SKIP_FIRST} to `T − 2` (for media, at the slots): the attention-sink
  positions and the last are left out. Documents weigh equally.
- bf16 on one H100 with fp32 sums, about 11 s (Qwen3-8B) and 16 s (Gemma 4) per text document per
  lens, by hand-written layer maps tested against autograd Jacobians under the same rules.

## Reliability

The documents alternate between two halves. `‖mean_even − mean_odd‖ / ‖mean‖` by layer:

- Qwen3-8B: {split_text(qt)}.
- Gemma 4 12B: {split_text(gt)}.

![Split-half difference by layer](figures/04_lens_reliability.png)

This is stability within the corpus, not independence from it. An R-lens for Qwen3-8B fitted on 500
chat conversations shares the top token of a neuron's write read with this one for only 31-51% of
neurons at L2-L16, but reads the same concepts and predicts steering within 0.04 of it at every layer
(§1).

![A chat-fitted lens against the halves of this one](figures/05_chat_lens.png)

## Which lens to use where

Measured on the models' neurons and heads (§3, §6, §11, §13, §14):

- **The R-lens predicts what steering a neuron does.** Correlated with the measured change in
  log-probabilities when a neuron's write is added at about its typical activation, the R-lens read of
  the write reaches, for Qwen3-8B, a median r of 0.46-0.51 at L4-L8 and 0.9 or more from L22 (another
  neuron's read: ±0.04), ahead of Neuronpedia's J-lens by 0.03-0.12 from L4 to L23. Before about L21 it
  reads a neuron's average push, which is what steering shows, not what one firing does in context (§4).

![Lens reads against steering](figures/07_steering.png)

- For Gemma 4 12B, from L11: 0.53 at L12-L23, 0.82 at L24-L35 and 0.97 at L36-L47, ahead of the logit
  lens. Before L11, natural-size pushes are too small for either lens to predict (§13).

![Gemma 4: lens reads against steering](figures/19_gemma_steering.png)

- **The S-lens reads early layers' input side.** Through L8 of Qwen3-8B (L13 of Gemma 4) it names the
  tokens that make a neuron fire (Qwen3-8B, L0: the firing token is in its top 10 for 40% of top
  firings; another neuron's read: 0.1%), and what early neurons write back. Past them its raw reads are
  swamped by hub tokens; z-scored across the layer they name concepts.
- **From L9 of Qwen3-8B, read input directions with `r-lens/mid.pt`.** It names the exact tokens that
  drive a neuron more often than the S-lens (5-8% of its top 10 at L9-L12 and L22-L30). L13-L20 is hard
  for every weight-only input read.

![Which reads name the exact drivers](figures/11_drivers.png)

- **Attention heads:** read a head's input with the S-lens (at real firings it beats the R-lens input in
  every band, for both circuits, with one exception: copying at L28-L35 read with an S-lens output),
  and its output with the S-lens through L8 and the R-lens from L9.

![Which lens on each side of a head](figures/16_head_lens_sides.png)

- **Images and sound** (Gemma 4): a neuron's read direction through the image S-lens is what the
  direction looks like in pixels, a projection: on held-out images it moves the neuron's read
  projection most through L23, but raises its activation only at L0-L5. The audio S-lens read is at
  chance. Optimised from the read by gradient ascent through the model, a patch or a frame raises its
  neuron through L23 (§14).

![What the image and sound reads are good for](figures/21_media_reads.png)

## Loading

```python
import torch
from huggingface_hub import hf_hub_download

r = torch.load(hf_hub_download("{REPO}", "qwen3-8b/r-lens/lens.pt"), weights_only=True)
J = r["J"][20]           # [4096, 4096]: layer 20's output -> the final residual
s = torch.load(hf_hub_download("{REPO}", "qwen3-8b/s-lens/mid.pt"), weights_only=True)
K = s["K"][20]           # token embeddings -> layer 20's MLP input
img = torch.load(hf_hub_download("{REPO}", "gemma-4-12b/image/s-lens/mid.pt"), weights_only=True)
K_img = img["K"][8] @ img["basis"]   # [3840, 3840]: image embeddings -> Gemma 4's layer 8 MLP input
```

With [Drowse](https://github.com/a9lim/drowse), the R-lens is a workspace source:
`drowse lens fetch Qwen/Qwen3-8B workspace-r --repo {REPO}`.

## Notes

- A read describes a direction, not an effect. The R-lens read is scale-free (the final RMSNorm);
  S-lens scores scale with the direction, so normalize it first.
- Positions before {SKIP_FIRST} are never seeded, so reads of the first positions, where the attention
  sink forms, extrapolate.
- Undertrained tokens can dominate early reads. The catalog leaves out tokens seen fewer than 5 times
  in its corpora, and byte-fallback pieces.
- A media read is one slot pattern summed over the 64 slots, not a map of where in an image it sits.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The lenses are derived from
[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) and
[`google/gemma-4-12B`](https://huggingface.co/google/gemma-4-12B) (both Apache 2.0) and fitted on
[`{qt['corpus']}`](https://huggingface.co/datasets/{qt['corpus']}) (CC BY 4.0), and Gemma 4's media
lenses on [COCO 2014](https://cocodataset.org) images (CC BY 4.0, Flickr terms for the images) and
[ESC-50](https://github.com/karolpiczak/ESC-50) (CC BY-NC 3.0). By [@_a9lim](https://x.com/_a9lim) and
[@treetowntree](https://x.com/treetowntree) with Claude (Anthropic), {date}.
"""


def band_table(bands: dict, cols: dict, all_name: str) -> str:
    """A markdown table of band shares from a summary: ``cols`` maps keys to headers."""
    rows = ["| band | " + " | ".join(cols.values()) + " |", "|---|" + "---|" * len(cols)]
    for band, r in bands.items():
        rows.append(f"| {'all' if band == all_name else band} | " + " | ".join(pct(r[k]) for k in cols) + " |")
    return "\n".join(rows)


def head_reads_text(j: dict) -> str:
    """Qwen3-8B's head reads' checks, from ``08_reads.py``'s summary."""
    b = j["bands"]
    return (f"For Qwen3-8B, `qk_self` finds duplicate-token heads at AUROC {j['qk_self_auroc']:.2f}. `qk_S` agrees with "
            f"the attention tokens receive in context (`qk_ctx_rho` past +0.2) for {pct(b['L0-8']['ctx'][2])} of "
            f"L0-L8's heads (another head's read: {pct(b['L0-8']['ctx'][3])}) and {pct(b['L28-35']['ctx'][2])} "
            f"of L28-L35's ({pct(b['L28-35']['ctx'][3])}); {pct(b['L28-35']['align'][2])} of L28-L35's heads "
            f"attend to what they copy (`qk_ov_rho` past +0.2; another head's OV read: {pct(b['L28-35']['align'][3])}).")


def media_table(j: dict) -> str:
    rows = ["| modality | band | reliable top label | held-out AUROC (median) | its label holds on all three text tests |",
            "|---|---|---|---|---|"]
    for m in ("image", "audio"):
        for band, r in j[m]["bands"].items():
            rows.append(f"| {m} | {band} | {pct(r['reliable'])} | {r['auroc_heldout']:.2f} | "
                        f"{pct(r['text_all_three'])} of {r['scored']:,} scored |")
    return "\n".join(rows)


def optimised_table(j: dict) -> str:
    rows = ["| modality | band | poles | held-out gain, median (its S-lens read) | raises the pole | beats its S-lens read |",
            "|---|---|---|---|---|---|"]
    for m in ("image", "audio"):
        for band, r in j[m].items():
            rows.append(f"| {m} | {band} | {r['poles']:,} | {r['gain']:.3g} ({r['gain_lens']:.2g}) | {pct(r['raises'])} | "
                        f"{pct(r['beats_lens'])} |")
    return "\n".join(rows)


def head_media_text(j: dict) -> str:
    out = []
    for m in ("image", "audio"):
        r = j[m]
        read = ", ".join(x[0] for x in r["top_read_ratio"][:4])
        top = ", ".join(f"{x[0]} ({x[1]:.2f})" for x in r["top_content"][:4])
        writers = ", ".join(x[0] for x in r["top_write_mid"][:4])
        out.append(f"- **{m}** ({r['n']} documents, {r['classes']} classes): the right label's loss is below a wrong one's "
                   f"for {pct(r['accuracy'])} ({r['nll_right']:.2f} against {r['nll_wrong']:.2f} nats). Most read from the "
                   f"text: {read}; largest extra loss when zeroed (nats): {top}; top label writers: {writers}.")
    b = j["image"]["bands"]
    return "\n".join(out) + ("\n\nShare of each band's heads that cost the right label more than a wrong one at z > 3, and "
                             "that write the right class at q ≤ 0.05, images: " + "; ".join(
                                 f"{k} {pct(r['content_heads'])}, {pct(r['write_mid_heads'])}" for k, r in b.items())
                             + ". For sound, no head passes either.")


def dataset_card(f: dict, lens_rev: str, date: str) -> str:
    q, g = f["Qwen/Qwen3-8B"], f["google/gemma-4-12B"]
    configs = [c for x in f.values() for c in x["configs"]]
    yaml = "\n".join(f"- config_name: {c}\n  data_files: \"{glob}\"" + ("\n  default: true" if i == 0 else "")
                     for i, (c, glob, _, _) in enumerate(configs))
    config_rows = "\n".join(f"| `{c}` | {rows:,} | {what} |" for c, _, what, rows in configs)
    ql, gl = summary("legible_qwen3-8b.json"), summary("legible_gemma-4-12b.json")
    neuron_cols = {"silent": "silent", "write_R": "`write_R`", "write_S": "`write_S`", "read_R": "`read_R`",
                   "read_S": "`read_S`", "any_test": "any test", "dark": "dark"}
    head_labels = {m: (SUMMARY / f"labels_heads_{m}.md").read_text().strip() for m in ("qwen3-8b", "gemma-4-12b")}
    media = summary("media_tables_gemma-4-12b.json")
    xm = summary("gemma4_media/crossmodal.json")["text_image_audio"]["all_three"]
    triples = "; ".join(f"`L{a} n{b} {'+' if c > 0 else '−'}` {d} / {e}" for a, b, c, d, e in xm["poles"])
    return f"""---
license: cc-by-4.0
pretty_name: neuron-lens
tags:
- interpretability
- neurons
- attention-heads
- r-lens
- s-lens
- multimodal
configs:
{yaml}
---

# neuron-lens: every neuron and attention head of Qwen3-8B and Gemma 4 12B

A catalog of two models' MLP neurons (two poles each) and attention heads:
[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) ({q['layers']} × {q['neurons']:,} neurons,
{q['layers']} × {q['heads']} heads) and [`google/gemma-4-12B`](https://huggingface.co/google/gemma-4-12B)
({g['layers']} × {g['neurons']:,} neurons, {g['layers']} × {g['heads']} heads). Each unit is read from three
sides:

- **from the weights alone,** through two lenses: what a unit writes and what it reads; for a head, what
  it copies and what it attends to, token by token;
- **from its activations** over about 15.4M positions of web, Chinese, Python and chat text;
- **from the tokens that drive it** when swept through real contexts.

Gemma 4 takes images and sound too, so each of its neurons is also read through lenses fitted on images
and sounds, and against {media['image']['documents']:,} labelled images and {media['audio']['documents']:,}
labelled sounds.

![What neurons and heads do](figures/01_anatomy.png)

Neurons and heads get the same tables, one row per pole or per head:

| table | neurons | heads |
|---|---|---|
| `reads` | lens reads of the write and the read direction | lens reads of the OV and QK circuits, per token |
| `acts` | activation statistics and top firings | output-norm statistics, top firings, attention patterns |
| `drivers` | the tokens that drive each pole hardest | the tokens that drive each head, as query and as key |
| `legible` | which reads hold at the pole's top firings | which reads hold at the head's top firings |
| `labels` | Claude Haiku's labels: where it fires, what it writes | where it fires, what it attends to, what it writes |
| `image`, `audio` (Gemma 4) | the patch or sound it reads, its write in media contexts, the images or sounds that drive it, and whether its media label is its text concept too | |
| `media` (Gemma 4) | | how each head attends in image and audio contexts, and which carry what an image or sound shows |

| config | rows | what |
|---|---|---|
{config_rows}

The lenses are [`{REPO}`](https://huggingface.co/{REPO}) at revision `{lens_rev}`. Code, methods and
findings (cited here as §N) are on [GitHub](https://github.com/stdlogits/neuron-lens).

## The lenses

- The **R-lens** reads what a vector makes the model say: `softmax(W_U · norm(J_L x))`, with `J_L` the
  averaged Jacobian from layer `L` to the last layer (for Gemma 4, after its final softcap,
  `30 · tanh(x / 30)`).
- The **S-lens** reads which input tokens put a vector there: token `v` scores
  `c · K_L (W_E[v] − mean(W_E))`, with `K_L` the averaged Jacobian from the embeddings to layer `L`.

Each is fitted at two points of every layer: `out`, the layer's output, and `mid`, its MLP input (after
attention). `out` reads MLP outputs and attention inputs; `mid` reads MLP inputs and attention outputs.
Gemma 4's image and audio lenses are fitted the same way on images and sounds.

| what | where it is | R-lens read | S-lens read | Gemma 4: image and audio lenses |
|---|---|---|---|---|
| a neuron's write | layer `L`'s output | `write_R` | `write_S` | `image.write_R`, `audio.write_R` |
| a neuron's read direction `â ± b̂` | layer `L`'s MLP input | `read_R` | `read_S` | `image.read_png`, `audio.read_wave` |
| what a head reads | layer `L − 1`'s output | | the token's footprint (input of `ov_*`, `qk_*`) | |
| what a head writes | layer `L`'s MLP input | `ov_R` | `ov_S` | |

## Which reads to trust where

No read works at every depth (§7, §11). For Qwen3-8B (these bands are priors; `legible` says which
reads hold for each unit):

| layers | what drives a neuron | what a neuron writes | what a head writes |
|---|---|---|---|
| L0-L7 | `read_S` (literal tokens, raw at L0-L4) | `write_S` (early neurons write back what they detect); `write_R` from about L4 | `ov_S` |
| L8 | both | both | `ov_S`, `ov_R` |
| L9-L12, L21-L35 | `read_R`, with `read_S_z` as a concept-level view | `write_R` | `ov_R` |
| L13-L20 | only `drivers` and `acts` | `write_R`, for steering | `ov_R` (weak) |

![Which reads work at which depth](figures/03_depth_map.png)

Gemma 4's map has the same shape at its own depths: `read_S` through L13 (falling to under 1% at L9 and rising again at L10-L13),
`write_R` from about L33 (for steering from L11), and a dark middle (§13).

![Both models by relative depth](figures/18_two_models.png)

## Conventions

- **Keys.** Neuron tables: `layer`, `neuron` (coordinate `n` of `down_proj`'s input, 0-indexed), `pole`
  (+1 or −1). Both models' gated activations take both signs, so each neuron has two poles with separate
  meanings, {2 * q['neurons']:,} (Qwen3-8B) or {2 * g['neurons']:,} (Gemma 4) rows a layer, the + poles
  first. Head tables: `layer`, `head` (query head) and `kv_head` (the key-value head it reads). `L8 n344 +`
  is `layer` 8, `neuron` 344, `pole` 1; `L20h17` is `layer` 20, `head` 17.
- **Token lists.** `<read>_ids` are 64 token ids of the model's tokenizer, `<read>_val` their scores,
  `<read>_str` the top 8 decoded and joined by ` | ` (newlines as `⏎`). Signal holds to rank 64 (§7).
- **Kept tokens.** Reads cover the tokens seen 5 or more times in the activation corpora, without
  byte-fallback pieces (`token_counts.pt`): {q['kept']:,} for Qwen3-8B, {g['kept']:,} for Gemma 4.
  Undertrained tokens otherwise dominate early reads.
- **`_z`** reads are layer-contrastive: each token's score z-scored across the layer's poles or heads,
  which removes the hub tokens that top most raw reads. **`_neg`**: a head read's bottom.
- **Firings.** A unit's firing in a document is its strongest position there; its top firings are the
  24 documents where that is largest. `contexts` show the first 5, the firing token in `⟦ ⟧` and, for
  heads, the token it attends to most (past position 0) in `⟪ ⟫`.
- **Tests** (`legible`) score a read on the tokens of the unit's own top firings, as a mean z, against
  mismatched pairs: another unit of the layer scored on the same tokens. `_q` is the Benjamini-Hochberg
  q-value; a test passes at `q ≤ 0.05`. **Dark:** fires, but no test passes and no single token
  dominates its firings.
- **Scale.** A lens read describes a direction, not an effect size; `act_top × w_norm` is a pole's
  typical write within this corpus. A write read is the pole's average push, which steering shows;
  before about L21 of Qwen3-8B, what one firing does depends on its context (§4).
- **Gemma 4.** A neuron's write is what reaches the residual, `s · γ_post ⊙ W_down[:, n]` (its MLP output
  passes a post-feedforward RMSNorm, whose scale the lenses hold constant, and the layer's output is
  scaled by `layer_scalar` `s`); `w_norm` is its norm. The read direction is the gate and up rows times
  the pre-feedforward norm's weight. Sliding-window layers have 8 key-value heads of size 256; every sixth
  layer is global, with one of size 512 whose keys are also its values. A head's write is `W_O` times the
  post-attention norm's weight. Every document starts with `<bos>`, the attention sink, so activation
  statistics start at the first real token and `first_*` is at position 1. The base model has no chat
  template: its chat slice is plain `User:` / `Assistant:` turns.

## Neurons

### `<model>-neurons-reads`: weight-only reads

| columns | read | of |
|---|---|---|
| `write_R`, `write_R_z` | the R-lens, `softmax(W_U · norm(J^out_L w))`; `_val` are probabilities (the logit lens at the last layer) | the write `w` (Qwen3-8B: `±W_down[:, n]`) |
| `write_S`, `write_S_z` | the S-lens to the layer's output: tokens whose footprint the write imitates | the write |
| `read_R`, `read_R_z` | the R-lens from the MLP input: what the residual says when it drives the pole | the read direction `â ± b̂` |
| `read_S`, `read_S_z` | the S-lens to the MLP input: input tokens that drive the pole | the read direction |

`â` and `b̂` are the neuron's gate and up rows times the MLP input norm's weight, normalized: `â ± b̂`
is a heuristic direction for each pole. Also per row: `w_norm`, `cos_ab`, `null_share` (the share of the
write in the bottom 1% of singular directions of `W_U diag(γ)`, where entropy neurons write; Stolfo et
al. 2024) and `vocab_skew`, `vocab_kurt` of `W_U · norm(w)` (Gurnee et al. 2024's prediction and
suppression neurons).

![Read signal by rank](figures/13_rank_depth.png)

### `<model>-neurons-acts`: activation statistics

Over every document of four slices, up to 256 tokens each, in bf16: DCLM held out from the lens
corpus, FineWeb-2 Chinese, codeparrot Python, and UltraChat dialogues ({q['docs']:,} documents and
{q['positions']:,} positions for Qwen3-8B, {g['docs']:,} and {g['positions']:,} for Gemma 4; `docs.parquet`).

- `act_top`: the median of the 24 top firings, the pole's typical high activation; `act_max`.
- `rate`: the share of positions past `act_top / 4`, from exact counts on a quarter-octave grid (linear
  between its points); `rate_<slice>` per slice; `pos_share`: where the pole's sign holds; `q50`-`q999`:
  quantiles of the activation times the pole's sign, from the same counts; `mean`, `std`.
- `top_doc`, `top_pos`, `top_val`: the 24 top firings; `peak_tokens` and the first 5 `contexts`.
- `first_max`, `first_tokens`, `first_chat`: the first position (left out above, where the attention
  sink forms), where activations depend on the token alone: the largest over every kept token, the 3
  tokens that reach it, and the value on the chat transcript's opening token. Many of Qwen3-8B's L2-L3
  poles do their work there (§2).

![Activation scale and quiet poles by layer](figures/06_activations.png)

### `<model>-neurons-drivers`: exact token drivers

Every kept token at position 24 of 8 real documents (4 web, 2 code, 2 chat), averaged over them, bf16:
`drivers` are the 64 tokens with the largest activation at the token itself, `drivers_sum` summed over
it and the 16 positions after. Values are times the pole's sign. For Qwen3-8B, an fp32 check through L9
shared 98-99% of a pole's top 20 on average (95-99% for `drivers_sum`; §6).

![Which reads name the exact drivers](figures/11_drivers.png)

### `<model>-neurons-legible`: which reads hold

`write_R` (the write's read promotes the tokens that follow the top firings), `write_S` (it names the
peak tokens), `read_R`, `read_S` (the read direction's reads name the peak tokens), each with `_z` and
`_q`; `silent` (no firing, or `act_top` under 0.05), `cur_mode`, `next_mode` (the share of top firings on
one token, or followed by one), `dark`. Share of each band's poles, Qwen3-8B:

{band_table(ql["neurons"]["bands"], neuron_cols, "L0-35")}

![Which reads hold for each pole](figures/12_census.png)

Gemma 4 12B:

{band_table(gl["neurons"]["bands"], neuron_cols, "L0-47")}

![Gemma 4: which reads hold for each pole](figures/20_gemma_census.png)

### `<model>-neurons-labels`: a label for every pole

Every pole described by Claude Haiku 5.5 from its reads and top contexts (batch API, effort low, adaptive
thinking; §8): `fires` (where it fires, 12 words or fewer), `writes` (what its write pushes the model
toward when amplified), each with a confidence 0-1, and `note`. Each model has its own prompt
(`scripts/prompts/<model>/neurons.md` on GitHub), which says which reads hold at each depth, and its
cards show only those. Qwen3-8B's labels were made from an earlier build of its tables, whose reads
kept byte-fallback pieces; Gemma 4's from this release's, from text alone (its cards show no images or
sounds).

Qwen3-8B, checked blind on 100 poles: firing labels separate held-out top firings from quiet positions at AUROC
0.82 [0.78, 0.85] (another pole's label: 0.56); a judge picks a pole's write label out of four from
steered text 61% of the time [0.52, 0.69] (chance 25%). Firing labels at confidence 0.7 or more score
0.94. They hold best early (83% of L0-L8's firing labels reach AUROC 0.8, 31% of L21-L27's), and write
labels late (80% picked at L28-L35, 41% at L0-L8). Gemma 4's neuron labels are not checked blind yet.

![Random poles read blind](figures/14_lens_blind.png)

### `gemma-4-12b-neurons-image`, `gemma-4-12b-neurons-audio`: images and sound

Gemma 4 has no vision or audio encoder network: an image is 64 slots of 48 × 48-pixel patches (384
pixels a side) and a sound 64 slots of 40 ms at 16 kHz, each entering the residual through a few norms
and one linear map. One row per pole, beside the text tables:

| columns | what |
|---|---|
| `read_S_basis` | the read direction through the modality's S-lens to the MLP input, `c · K^mid_L`: coordinates (fp16) on the S-lens basis |
| `read_png` (image) | the 48 × 48 patch the direction reads: `read_S_basis` through the patch embedder back to pixels, at ±3 standard deviations around grey |
| `read_wave` (audio) | the 40 ms waveform it reads, 640 samples at 16 kHz (fp16) |
| `write_R`, `write_R_z` | the write through the modality's R-lens: what it makes the model say in image or audio contexts |
| `label`, `label_z`, `label_mean`, `labels_str` | its top labels by the z of its mean activation over each label's documents (labels on 8 or more documents: {media['image']['labels_scored']:,} for images, {media['audio']['labels_scored']:,} for sound) |
| `sig`, `stable`, `auroc_heldout` | the top label past the shuffled-label null's 99th percentile for its layer; in the top 5 of both halves of the documents; chosen on even documents, its AUROC on odd ones |
| `top_docs`, `top_vals` | the 16 documents with its largest mean activation (rows of `media/<modality>_docs.parquet`) |
| `text_fire_rank`, `text_read_rank`, `text_write_rank` | for a reliable label, its rank among the modality's labels on the text side (1 first, 0.5 chance): firing on the label's words, the text S-lens read, the text R-lens write |

A media read sums over a document's 64 slots, so it is one pattern wherever it sits, not a map of
where. Read `read_png` as what the pole's input direction looks like in pixels, a projection: added to
held-out images it raises the pole only at L0-L5 (§14). The exemplars say what fires a pole, and the
optimised reads below are patterns that drive it.

{media_table(media)}

![The patch each neuron reads, and the patch that drives it](figures/22_media_gallery.png)

**One concept in three modalities** (§15). {xm['passing_both']} poles pass all three text tests for both their
image and their audio label, of {xm['scored_in_both']:,} scored in both (independent lists would share
{xm['expected_if_independent']:.1f}): {triples}.

![One concept in text, images and sound](figures/23_crossmodal.png)

### `gemma-4-12b-neurons-image-optimised`, `gemma-4-12b-neurons-audio-optimised`

For the 48 poles a layer (L0-L35) with the strongest reliable label: one slot pattern, the same in all
64 slots, started from the S-lens read and optimised by gradient ascent through the model (24 Adam steps
over 128 images or clips; at most 0.08 of the pixel range, or 0.3 of each frame's RMS). `opt_png` or
`opt_wave` (in units of a frame's RMS), `gain` (the pole's mean activation over the slots of 8 held-out
documents with the pattern added, minus without), `gain_lens` (the same for its S-lens read),
`act_base`. Gains are in the pole's activation units:

{optimised_table(summary("media_optimised_gemma-4-12b.json"))}

The audio test clips are among the audio S-lens's fitting clips (all of ESC-50), though not the
optimisation's.

## Heads

A head reads token `v` as its S-lens footprint at layer `L − 1`'s output (the embedding at L0) and writes
into layer `L`'s MLP input. Its OV circuit maps what it reads to what it writes, and its QK circuit
decides where it reads, both over the whole vocabulary (§11).

### `<model>-heads-reads`: weight-only reads, per token

| columns | read |
|---|---|
| `ov_R`, `ov_R_neg` | each token's z in its own column of the OV map read through the R-lens: where `v` ranks in what the head makes the model say when it reads `v`. The tokens it copies most, and suppresses most |
| `ov_S`, `ov_S_neg` | the same through the S-lens: whether the write makes the residual look as if `v` were there (the record through L8) |
| `qk_S`, `qk_S_neg` | each key token's z over the vocabulary under the head's mean query: what it attends to and avoids, whatever the query |
| `qk_self_S` | each token's z in its own row of the QK map: where its own token ranks among the keys it favours as a query |
| `ov_R_copy`, `ov_S_copy` | Elhage et al.'s copying score Σλ/Σ\\|λ\\| of each OV map: +1 copies what it reads, −1 suppresses it |
| `qk_self` | the median `qk_self_S` z over tokens: duplicate-token heads score high |
| `qk_ctx_rho` | Spearman over tokens between `qk_S` and the attention each token receives in context |
| `qk_ov_rho` | Spearman over tokens between `qk_S` and `ov_R`: does the head attend to what it copies? |

The QK map is `q̃ᵀ R̄ k̃`, with the q/k norms' true derivative averaged over real positions and RoPE
averaged over the head's own attention offsets, footprints at unit length: it reads which tokens a head
favours, not position-only behaviour such as previous-token or induction attention.

{head_reads_text(summary("reads_qwen3-8b.json"))} For Gemma 4, at AUROC {summary("reads_gemma-4-12b.json")["qk_self_auroc"]:.2f}.

### `<model>-heads-acts`: firings and attention patterns

Over the same documents, from position 16 (a head with few keys attends sharply):

- `out_top`, `out_max`, `rate`, `rate_<slice>`, `q50`-`q999`, `mean`, `std`: as for neurons, of the norm
  of the head's output as it reaches the residual (`‖W_O^h z‖`; for Gemma 4, after its post-attention norm);
- `top_doc`, `top_pos`, `top_val`, `src_pos` (the position each firing attends to most, past the first),
  `peak_tokens` (query tokens), `src_tokens` (attended tokens), `contexts`;
- `sink`, `prev`, `self`: the mean attention from queries at 16+ to the first position, to the token
  before and to itself; `induction`, `duplicate`: over 64 sequences of 64 random tokens said twice,
  attention from the second copy to the token after the first occurrence, and to the occurrence;
- `out_R_z`: the firings' outputs read through the R-lens from the MLP input, each token's logit
  z-scored across the layer's firings, averaged over the head's.

### `<model>-heads-drivers`: exact token drivers

In the same pass as the neurons': `drivers`, the 64 tokens with the largest output norm at their own
position (the token as the query), and `keys`, the 64 that receive the most attention from the 16
positions after them (the token as the key).

### `<model>-heads-legible`: which reads hold

At each top firing, with the attended token: `ov_R` (its OV column promotes the token after the query),
`ov_copy` (its own `ov_R` z: the head promotes what it reads), `ov_S` (its own `ov_S` z), `qk_S` (its
`qk_S` z: the QK read predicts what the head attends to), each with `_z` and `_q` (Benjamini-Hochberg over
all heads); `cur_mode`, `src_mode`; `pattern` (`prev` or `self` past 0.5, `induction` or `duplicate` past
0.3); `dark`: no test, no mode past a third, no pattern. Share of each band's heads, Qwen3-8B:

{band_table(ql["heads"]["bands"], {"ov_R": "`ov_R`", "ov_copy": "`ov_copy`", "ov_S": "`ov_S`", "qk_S": "`qk_S`",
                                     "pattern": "pattern", "any_test": "any test", "dark": "dark"}, "L0-35")}

The S-side tests pass for almost no head. S→S copying and the mean query's favourite keys are shared by a
layer's heads, so at a head's own firings they don't tell it from its neighbours; they remain reads of
what each head does (§12).

![Which reads hold for each head](figures/17_head_census.png)

Gemma 4 has 16 heads a layer, so a head's mismatched control has 240 pairs, no p-value falls below
1/241, and BH over its 768 heads passes none. Its table adds `<test>_p`, the uncorrected p-value. Share
of each band's heads at p ≤ 0.05 (5% by chance):

{band_table(gl["heads"]["bands"], {"ov_R_p05": "`ov_R`", "ov_copy_p05": "`ov_copy`", "ov_S_p05": "`ov_S`",
                                     "qk_S_p05": "`qk_S`", "pattern": "pattern", "dark": "dark"}, "L0-47")}

### `<model>-heads-labels`: a label for every head

Qwen3-8B:

{head_labels["qwen3-8b"]}

Gemma 4 12B: its write labels are at chance, so read them as guesses.

{head_labels["gemma-4-12b"]}

### `gemma-4-12b-heads-media`: heads in image and audio contexts

64 labelled images (mini-ImageNet) and 64 sounds (ESC-50, the classes in turn), each in the media prompt
and followed by ` a <label>`, its own and another class's (§16). Per head and modality: `read`,
`read_ratio` (attention from the text after the media to its slots, and that over an even spread),
`sink`, `self`, `within`, `nbr` (attention from a slot to itself, the other slots, and its grid
neighbours or the frame before); `abl_right`, `abl_wrong`, `content`, `content_z` (each head zeroed: the
change in the label's loss, right and wrong, and their difference); `write_mid`, `write_logit` and their
`_q` (the head's output at the article read through the R-lens and the logit lens: the right class's
mean percentile among the classes, against 200 label shuffles); `top1_mid`, `top1_logit`.

{head_media_text(summary("head_media_gemma-4-12b.json"))}

![Heads in image and audio contexts](figures/24_heads_media.png)

## Plain files

Per model folder (`qwen3-8b/`, `gemma-4-12b/`):

- `docs.parquet`: each activation document's `slice`, its `line` in the slice's file and its length.
  The slices' sources, revisions and row ids are in `data/corpus/act-<slice>.provenance.json` on GitHub:
  the corpus is not re-hosted here.
- `token_counts.pt`: `counts[v]`, how often token `v` occurs in the activation corpora (`by_slice` per
  slice).
- `causal/causal.pt`: the causal sample (§3, §13). `picks`: {q['causal_picks']:,} Qwen3-8B neurons
  `(layer, neuron)`, 32 random per layer plus named ones, or {g['causal_picks']:,} of Gemma 4's, 24 random
  per layer. Per neuron and pole: `steer` `[neuron, pole, dose, vocab]`, the mean change in log-probs
  over positions 16-63 of 16 64-token DCLM contexts (`contexts`) when the write is added there at 1× and
  4× `alpha` (the pole's typical high activation); `ablate_pos` and `ablate_win`, clean − ablated
  log-probs with the neuron zeroed at its peak in 8 top documents, at that position and summed over the
  next 8; `dlogp_next`, the next token's change per document; `fired`, the activation there. bf16.
- `gemma-4-12b/media/{{image,audio}}_docs.parquet`: each image or sound document's `source`, `labels`,
  `caption`, `ref` (JSON: the dataset, config, split and revision, the row's own identifiers, the media
  file's path and its bytes' SHA-1) and `link`. The images and sounds are not re-hosted;
  `data/corpus/media-<modality>.provenance.json` on GitHub lists every source with its license.

## Loading

```python
import pandas as pd

base = "hf://datasets/{DATASET}/qwen3-8b"
reads = pd.read_parquet(f"{{base}}/neurons/reads/L08.parquet")
reads.query("neuron == 344 and pole == 1")[["write_R_str", "read_R_str", "read_S_str"]]
heads = pd.read_parquet(f"{{base}}/heads/reads.parquet").merge(pd.read_parquet(f"{{base}}/heads/acts.parquet"))
heads.query("layer == 20 and head == 17")[["ov_R_str", "qk_S_str", "duplicate", "contexts"]]

base = "hf://datasets/{DATASET}/gemma-4-12b"
image = pd.read_parquet(f"{{base}}/neurons/image/L33.parquet").query("neuron == 6099 and pole == 1").iloc[0]
docs = pd.read_parquet(f"{{base}}/media/image_docs.parquet")
image[["label", "labels_str", "text_fire_rank"]], docs.loc[image["top_docs"][:4], ["labels", "link"]]
```

## Limits

- Two models; the text lenses are fitted on English-heavy web text, Gemma 4's media lenses on COCO
  photos and ESC-50 sounds.
- Activation statistics come from bf16 passes over documents of up to 256 tokens; top firings are one
  per document.
- The tests are token-level, over at most 24 firings per unit. A dark unit may still be describable from
  its contexts (§7).
- Heads are read through their circuits one token at a time: a head whose role depends on the query's
  context, or on position alone, is read only through `acts`.
- Media labels name what a whole image or sound is about, not each patch or frame; media reads lose
  fidelity with depth (§14).

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The catalog is derived from
[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) and
[`google/gemma-4-12B`](https://huggingface.co/google/gemma-4-12B) (both Apache 2.0). Its `contexts` quote
short excerpts of the activation corpora, which keep their own terms:
[DCLM](https://huggingface.co/datasets/HuggingFaceFW/dclm_100BT-shuffled) (CC BY 4.0),
[FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (ODC-By 1.0),
[codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) (code from public GitHub
repositories, under their licenses) and
[UltraChat-200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (MIT). The media
documents' labels and captions come from their source datasets, under those datasets' terms (some
non-commercial); the images and sounds themselves are only linked. By [@_a9lim](https://x.com/_a9lim)
and [@treetowntree](https://x.com/treetowntree) with Claude (Anthropic), {date}.
"""


# ---------------------------------------------------------------- the release

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=",".join(PARTS), help=f"what to build and upload: {', '.join(PARTS)}; empty for the cards alone")
    ap.add_argument("--dry-run", action="store_true", help="build the files and cards, upload nothing")
    args = ap.parse_args()
    parts = set(filter(None, args.parts.split(",")))
    if parts - set(PARTS):
        raise ValueError(f"unknown parts: {parts - set(PARTS)}")
    stem = arch.stem(MODEL)
    facts_path = SUMMARY / f"release_{stem}.json"
    facts = json.loads(facts_path.read_text()) if facts_path.exists() else {}
    cfg = arch.text_config(MODEL)
    facts |= {"model": MODEL, "dir": stem, "layers": cfg.num_hidden_layers, "d_model": cfg.hidden_size,
              "neurons": cfg.intermediate_size, "heads": cfg.num_attention_heads}
    api = HfApi()

    # The model's lens files, then the catalog's, staged as the repos lay them out.
    lens_root, data_root = DATA / "release", DATA / "release-data"
    if "lenses" in parts:
        facts["lenses"] = write_lenses(MODEL, lens_root / stem)
    files = catalog_files(MODEL)
    for part in parts & set(files):
        shutil.rmtree(data_root / stem / part, ignore_errors=True) if part != "files" else None
        for src, rel in files[part]:
            link(src, data_root / stem / rel)
    if parts & set(files):
        facts |= catalog_facts(MODEL, data_root / stem)
    facts_path.write_text(json.dumps(facts, indent=1) + "\n")
    every = {m: json.loads((SUMMARY / f"release_{arch.stem(m)}.json").read_text()) for m in MODELS}

    # The cards, and the figures they show, for both repos.
    date = time.strftime("%Y-%m-%d")
    for root, names in ((lens_root, MODEL_FIGURES), (data_root, DATASET_FIGURES)):
        shutil.rmtree(root / "figures", ignore_errors=True)
        for name in names:
            link(FIGURES / name, root / "figures" / name)
    (lens_root / "README.md").write_text(model_card(every, date))
    if args.dry_run:
        (data_root / "README.md").write_text(dataset_card(every, "(dry run)", date))
        print(f"dry run: cards in {lens_root} and {data_root}")
        return
    api.create_repo(REPO, private=True, exist_ok=True)
    api.upload_folder(repo_id=REPO, folder_path=lens_root,
                      allow_patterns=["README.md", "figures/*"] + ([f"{stem}/**"] if "lenses" in parts else []),
                      delete_patterns=["figures/*", "*/README.md", "*/figures/*"] + ([f"{stem}/**"] if "lenses" in parts else []),
                      commit_message=f"{stem}: lenses, cards and figures" if "lenses" in parts else "Cards and figures")
    lens_rev = api.repo_info(REPO).sha[:10]
    print(f"{time.strftime('%H:%M:%S')} model repo at {lens_rev}: https://huggingface.co/{REPO}", flush=True)
    (data_root / "README.md").write_text(dataset_card(every, lens_rev, date))
    api.create_repo(DATASET, repo_type="dataset", private=True, exist_ok=True)
    built = [f"{stem}/{rel}" for part in parts & set(files) for _, rel in files[part]]
    stale = ["figures/*", "*/README.md", "*/figures/*"] + (
        [f"{stem}/*", f"{stem}/**"] if parts >= {"neurons", "heads", "files"} else [f"{stem}/{p}/**" for p in parts & {"neurons", "heads"}])
    api.upload_folder(repo_id=DATASET, repo_type="dataset", folder_path=data_root,
                      allow_patterns=["README.md", "figures/*", *built], delete_patterns=stale,
                      commit_message=(f"{stem}: {', '.join(sorted(parts - {'lenses'}))}, cards" if parts - {"lenses"} else "Cards")
                                     + f" and figures (reads through {REPO}@{lens_rev})")
    print(f"{time.strftime('%H:%M:%S')} uploaded to https://huggingface.co/datasets/{DATASET}", flush=True)


if __name__ == "__main__":
    main()
