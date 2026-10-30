# AGENTS.md

Working notes for coding agents and contributors. [README.md](README.md) is the overview,
[docs/design.md](docs/design.md) the methods, [docs/findings.md](docs/findings.md) the results
and their limits. Read the findings before making a claim about a neuron or a lens.

## What this is

A catalog of what every MLP neuron (two poles each) and attention head of two models means:
Qwen3-8B (36 layers × 12,288 neurons, 32 heads) and Gemma 4 12B (48 × 15,360, 16 heads). Each unit is
read from three sides: the weights through two lenses (a neuron's write and read direction; a head's
OV and QK circuits, per token), activations over 15M positions, and exact token drivers. Neurons and
heads get the same tables: `acts`, `drivers`, `reads`, `legible`, and `labels`.
Gemma 4 also takes images and sound, so it adds lenses fitted on images and sounds, per-neuron
`image` and `audio` tables and a `media` table of heads. `NEURON_LENS_MODEL` selects the model for
every stage (Qwen3-8B by default). The lenses and the catalog are released on the Hugging Face Hub
by `scripts/12_release.py`, one card per repo covering both models.

## Layout

- `neuron_lens/`: the package. `transport.py` is the lens estimator (hand-written RelP layer maps
  and their transposes; Gemma 4's are in `gemma4.py`); `rules.py` states the same rules as autograd
  patches and is the test reference; `arch.py` holds each model's geometry (head layouts, weight
  names, token counts, which fits are of record) and `neurons.py` its neurons' read and write
  directions; `readout.py` reads vectors through a lens; `heads.py` reads head circuits as token
  maps; `lenses.py` loads fits, released lenses and Drowse lenses; `media.py` builds Gemma's image and
  audio documents and reads its embedders backwards, `media_sets.py` streams the labelled image and
  audio collections; `corpus.py` the corpora (`act_file` names each slice's file); `catalog.py` holds
  the catalog's layout, the kept tokens and the table helpers (and browses it:
  `python -m neuron_lens.catalog`); `stats.py` the shared statistics; `known.py` named neurons.
- `scripts/NN_*.py`: the pipeline, numbered in run order (`06` activations → `07` drivers → `08`
  reads → `09` legible → `10` labels; for Gemma 4, `13` media exemplars → `checks/crossmodal.py` →
  `14` media tables → `15` optimised media reads; then `11` figures → `12` release).
  `scripts/prompts/`: the labellers' prompts. `scripts/checks/`: analyses, each behind a findings or
  design section (`causal_lens.py`, `crossmodal.py` and `head_media.py` also make released files);
  analyses the pipeline absorbed are retired, and the findings name their old scripts.
- `tests/`: `test_transport.py` (the estimator), `test_readout.py`, and `test_arch.py`,
  `test_gemma4.py` (Gemma 4's geometry and layer maps on small random models).
- `data/` is ignored except `data/summary/` (compact results and logs, tracked) and
  `data/corpus/*.provenance.json`. Locally it holds corpora, fits (`data/fits/`), Drowse homes
  (`data/lenses/<name>/`), catalog shards (`data/catalog/<model>/`), activation stores, the
  release staging folders, and outside data (`data/external/`). `neuron_lens/paths.py` names
  them. The catalog lives in `data/catalog/<model>/`, laid out as the release:
  `neurons/<table>/L<layer>.parquet` (a row per pole, the + poles first) and
  `heads/<table>.parquet` (a row per head), key columns in front. Per model: `token_counts.pt`
  (Qwen3-8B) or `token_counts-<model>.pt`; the chat slice `act-chat.jsonl` (Qwen3-8B) or
  `act-chat-<model>.jsonl`, the other slices shared; Gemma's media sources in
  `data/corpus/media-<modality>.provenance.json`. `data/summary/release_<model>.json` holds what the
  Hub cards say about each model's release, so either model's run rebuilds both cards.

## Invariants

- **The lens rule.** A lens's `out` target reads MLP outputs and attention inputs; its `mid` target
  reads MLP inputs and attention outputs. Don't fit or ship a lens for a point it isn't built for,
  and don't ship borrowed lenses (the J-lens, the logit lens) in the catalog; comparisons belong in
  checks. A head's input is read with the S-lens on both circuits (findings §11).
- **One token mask.** Every token read covers `catalog.kept()`: tokens seen 5+ times in the
  activation corpora, without byte-fallback pieces.
- **Description cards stay leak-free.** `10_labels.py check` and `checks/lens_blind.py` hold out a
  unit's top documents past the first `N_CONTEXTS`. Nothing on an evidence card may summarize them: no
  peak-token counts or agreement shares over all top documents, no ablation run in top contexts.
  Scoring contexts end at the marked token. Findings §8 records why.
- **Estimator changes** need `tests/test_transport.py` and `tests/test_gemma4.py` to pass: both
  lenses against brute-force Jacobians of the model under `relp_rules`, and the R-lens against
  Drowse's RelP fit (loosen that test only for a deliberate change). Supporting another architecture
  means writing its layer map and transpose (as `gemma4.py` does for Gemma 4) and its geometry in
  `arch.py`; `_check` refuses what they don't cover.
- **Both lenses sum over positions** (as the J-lens does) and use RelP. These are fixed, not
  options. More generally, keep one current code path: delete superseded options and formats
  rather than keeping compatibility.
- **Fit files** in `data/fits/` are the authoritative lens format:
  `<lens>-<model>-L<layers>-<corpus>[-s<i>of<n>].pt`, holding even- and odd-document sums per
  target. Both lenses have an `out` target (layer outputs) and a `mid` target (each layer's MLP
  input): R `out` reads MLP writes, R `mid` reads MLP read directions and attention writes, S
  `mid` reads read directions, S `out` reads writes. `lenses.load_fits` adds shards and halves;
  `Accumulator.mean` averages the halves.
- **Section numbers in findings are stable.** Code docstrings, figure footers and the Hub cards cite
  them. Fold a new result into the section it belongs to, or add a section at the end; don't
  renumber.

## The lenses of record

- **R-lens:** fitted on the first 1,000 rows of DCLM-shuffled (`01_lens_corpus.py`, one
  extendable file at a pinned revision) with `02_fit.py --lens R`; exported by
  `04_export_rlens.py` to `data/lenses/dclm`, with its halves in `dclm-even` and `dclm-odd`.
  Split-half 0.06 or less at every layer (findings §1).
- **S-lens:** the same documents, `02_fit.py --lens S` (`data/fits/s-qwen3-8b-L36-dclm.pt`),
  split-half 0.00-0.05. Its reads are legible at L0-L8; past L8 use the exact drivers for the
  input side (findings §6).
- Kept for comparison: an R-lens fitted on 500 chat conversations (`data/lenses/chat`; findings
  §1).
- **Gemma 4 12B:** both lenses on the same 1,000 DCLM documents (`<lens>-gemma-4-12b-L48-dclm`), and
  image and audio lenses (`02_fit.py --modality image|audio`: COCO 2014 val, 1,000 for S and 512 for
  R; ESC-50, 1,615 kept clips for S and 512 for R; fits `...-coco`, `...-esc50`). The media S-lenses
  are seeded in the embedder's subspace and carry a `basis`. The pipeline reads local fits when they
  are in `data/fits/`, else the released lenses (`arch.fits`, the `hub` lens spec).
- **Replacing the lens of record:** fit, export to a new home under `data/lenses/`, rerun
  `08_reads.py` and the stages after it, record the comparison in findings, and rerun
  `12_release.py`.
- **Drowse** ([github.com/stdlogits/drowse](https://github.com/stdlogits/drowse)) loads exported R-lenses
  (`R:<name>` specs in `lens_compare.py`). It always writes an R-lens to `local/r` under
  `$DROWSE_HOME` (lens format v7, Drowse 5.4), so each export gets its own home under
  `data/lenses/<name>/` and never overwrites another, or the user's default home. Fitting lives
  here; exact agreement with Drowse's fit is not required. A home written before 5.4 (v6, lens
  at `jlens/local/relp`) is read only after `python scripts/migrate_lens_v7.py --home <home>`
  in the Drowse checkout.

## Method notes

- **Indexing.** Neuron `n` of layer `L` is coordinate `n` of `down_proj`'s input, 0-indexed.
- **Gemma 4.** A neuron's write is what reaches the residual, `s · γ_post ⊙ W_down[:, n]` (the
  post-feedforward norm's scale held constant, `layer_scalar` `s`); its read rows are scaled by the
  pre-feedforward norm. Logits pass a softcap (`30 · tanh(x / 30)`), which R-lens probabilities
  include. Every document starts with `<bos>` (the sink): statistics start at position 1. Its bands
  are quarters (L0-L11, …, L36-L47). With 16 heads a layer, head tests can't pass BH; `heads/legible`
  adds uncorrected `_p` columns (findings §13). Design.md's Gemma section has the rest.
- **Poles.** SwiGLU activations take both signs (`up` is linear), so each neuron has a `+` and a
  `−` pole, read and described separately. From weights, a pole's read direction is `â ± b̂`: the
  gate and up rows times the post-attention norm weight, normalized (a heuristic; activations give
  the true poles). Catalog shards list the `+` poles of neurons `0..n-1`, then the `−` poles.
- **Scale.** A lens read is scale-free (RMSNorm), so its probabilities describe a direction, not an
  effect. Effect sizes need the activation scale (`act_top`); don't rank neurons by lens
  probability across layers.
- **Which side, which depth** (findings §6, §7, for Qwen3-8B; Gemma 4 has the same shape at its own
  depths, §13). Input side: the S-lens read at L0-L7 (raw at L0-L4, naming literal tokens), both
  lenses at L8, the R-lens `mid` read from L9 (`read_S_z` as a concept-level second view), and only
  exact drivers and activations at L13-L20. Write side: the
  S-lens (`write_S`) through L8, where early neurons write back the tokens they detect; the R-lens,
  legible from about L4. Heads: `ov_S` through L8, `ov_R` from L9, `qk_S` throughout (dark at
  L13-L20; findings §11). These bands are a prior: weigh each unit's sides by its own q-values
  in `legible`. Don't force a word reading where a side is uninformative, and say which side a
  description rests on.
- **In context.** Before about L21, a firing's effect varies by context, and a write read describes
  the pole's average push (what steering shows), not what one firing does (findings §4).
- **Undertrained tokens.** Token reads cover `catalog.kept()`: tokens seen 5+ times in
  `data/token_counts.pt` (`03_token_counts.py`, over the activation corpora), without byte-fallback
  pieces. Undertrained tokens push some early neurons to huge activations.
- **Descriptions are soft:** weighted labels with a confidence and scores, not one hard label.
  In the docs, let the wording say what is measured and what is interpretation ("suggests",
  "likely"), without labels.
- **Outside labels** (`known.EXTERNAL`) are cross-references, not ground truth; our reads,
  activations and causal checks arbitrate. `known.KNOWN` holds steering-verified persona neurons
  from an earlier experiment on the same model.
- **Judging a lens:** by whether its reads are coherent and make sense in context, not by matching
  a benchmark. Report the split-half difference `02_fit.py` prints. Many neurons reading alike can
  be the shared workspace subspace (the J-space), not a flaw.
- **Lens specs** for `scripts/checks/lens_compare.py`: `logit`, `J` (Neuronpedia's J-lens), `R`
  (the default Drowse home), `R:<name>` (`data/lenses/<name>`), `R:<a>+<b>` (merged).

## Running jobs

- A Mac with Apple silicon (MPS) runs everything except the lens fits and the GPU passes; the fits
  take about 3 hours (Qwen3-8B) or 4.5 hours (Gemma 4) per text lens per 1,000 documents on one H100
  (`02_fit.py --device cuda`, bf16 with fp32 sums), 45-90 minutes per media lens.
- `13_media_activations.py` keeps every document's activations for all of Gemma's neurons in RAM
  (about 220 GB for the images): run the two modalities one at a time.
- One GPU process per GPU. Before GPU work, check for running jobs (`ps`, `nvidia-smi`); on a
  shared machine, never run beside or stop a job that isn't yours. Light CPU work (weight-only
  reads, tests) can run beside a GPU job; heavy CPU work slows it on unified memory.
- Long jobs run detached (`02_fit.py --detach` and `06_activations.py --detach` start a new
  session, under `caffeinate` on macOS) and log to `logs/`. Both save every `--save-minutes`
  (default 10) and resume when rerun; `06_activations.py` rerun after a complete pass rebuilds the
  `acts` tables. The activation pass needs about 21 GB of GPU memory with eager attention (a
  24 GB card fits it at the default 32 documents a batch).
- New CUDA boxes: `scripts/gpu_box.sh` installs the package, downloads the models and corpus,
  runs the estimator tests and benchmarks. Export `OMP_NUM_THREADS=8` on many-core machines:
  torch's default of one thread per core makes CPU ops on small tensors spin for minutes. Run one
  process per GPU (`CUDA_VISIBLE_DEVICES=i ... --shard i/n`) and copy `data/fits/` off rented
  boxes before destroying them.

## Recording results

Whoever runs an analysis adds it to the section of `docs/findings.md` it belongs to (or a new section
at the end): setup (script, hardware, precision, lens), results with controls, and limits, as
concisely as the result allows. A result that changes the overview goes into the README's matching
findings paragraph too. Compact outputs go in `data/summary/`, figures through
`scripts/11_figures.py`. The code and docs are MIT; the lenses and catalog on the Hub are CC BY 4.0
(set in `12_release.py`'s cards). Contributions come as branches and pull requests.
