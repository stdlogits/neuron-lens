# AGENTS.md

Working notes for coding agents and contributors. [README.md](README.md) is the overview,
[docs/design.md](docs/design.md) the methods, [docs/findings.md](docs/findings.md) the results
and their limits. Read the findings before making a claim about a neuron or a lens.

## What this is

A catalog of what every MLP neuron (two poles each) and attention head of two models means:
Qwen3-8B (36 layers × 12,288 neurons, 32 heads) and Gemma 4 12B (48 × 15,360, 16 heads). Each unit is
read from three sides: the weights through two lenses (a neuron's write and read direction; a head's
OV and QK circuits, per token), activations over 15M positions, and exact token drivers. Neurons and
heads get the same tables: `acts`, `drivers`, `reads`, `labels` and `sources` (what drives them when
they fire; for a head, through its value, query and key); neurons also get `firings` (what the residual holds where
they fire) and `edges` (direct weight edges between poles).
Gemma 4 also takes images and sound, so it adds lenses fitted on images and sounds, per-neuron
`image` and `audio` tables per neuron and per head. `NEURON_LENS_MODEL` selects the model for
every stage (Qwen3-8B by default). The lenses and the catalog are released on the Hugging Face Hub
by `scripts/12_release.py`, one card per repo covering both models.

## Layout

- `neuron_lens/`: the package. `transport.py` is the lens estimator; `layers.py` writes every
  model's decoder layers as RelP linear maps and their transposes, one implementation per component
  (norms, softmax attention with sliding windows and gates, Gated DeltaNet, gated and sparse MLPs),
  composed from what each layer holds; `kernels.py` runs its hot spots on CUDA as Triton kernels
  (DeltaNet's derivative, the sparse MLP's experts), with `layers.py`'s torch passes as their
  reference; `rules.py` states the same rules as autograd patches, by operation, and is the test
  reference; `arch.py` holds each model's geometry (head layouts, weight
  names, token counts, which fits are of record) and `neurons.py` its neurons' read and write
  directions (`poles`: in the residual's metric); `metric.py` the residual's metric, which turns a covector into a
  vector and back (`02_fit.py --lens scale`); `readout.py` reads vectors through a lens; `heads.py` reads head circuits as token
  maps; `lenses.py` loads fits, released lenses and Drowse lenses; `media.py` builds Gemma's image and
  audio documents and reads its embedders backwards, `media_sets.py` streams the labelled image and
  audio collections; `corpus.py` the corpora (`act_file` names each slice's file); `catalog.py` holds
  the catalog's layout, the kept tokens and the table helpers (and browses it:
  `python -m neuron_lens.catalog`); `firings.py` and `sources.py` the activation-side passes (the residual at a pole's firings; the exact split of
  an activation over its sources); `stats.py` the shared statistics; `known.py` named neurons.
- `scripts/NN_*.py`: the pipeline, numbered in run order (`06` activations → `07` drivers → `08`
  reads → `10` labels; for Gemma 4, `13` media exemplars → `checks/crossmodal.py` →
  `14` media tables → `15` optimised media reads; `16` edges, and `17` firings and sources, after `06`; then `11` figures
  → `12` release). `09` is retired: `checks/reads_at_firings.py`, after `07` and `08`, gives the per-band shares the
  figures and the depth guidance rest on.
  `scripts/prompts/`: the labellers' prompts. `scripts/checks/`: analyses, each behind a findings or
  design section (`causal_lens.py`, `crossmodal.py` and `head_media.py` also make released files).
  Every script the findings name is here. Checks read the catalog's tables and get documents from
  `catalog.documents` (the activations file, else rebuilt from the corpus), so most run from the
  release and the model.
- `tests/`: only what a run can't show. The estimators against brute-force Jacobians on tiny random models
  (`test_transport.py` Qwen3, `test_gemma4.py` Gemma 4 and its media fits' mode, `test_qwen35.py` Qwen3.5), and
  `test_kernels.py` (the CUDA kernels against their torch passes; skipped without CUDA). Anything that fails loudly or
  reads wrong in the output gets fixed when it comes up; don't add tests for it.
- **Qwen3.5 (in progress):** `layers.py` covers Qwen3.5-35B-A3B's layers (Gated DeltaNet, gated
  attention, MoE with a shared expert; Qwen3.5-9B's dense MLP too). RelP's rules apply by operation (elementwise gates take
  the half rule; scalar weights, DeltaNet's mixing matrix, the router and the shared expert's gate,
  keep their true derivative, as softmax attention's do); design.md's Qwen3.5 section lists how this
  differs from the published R-lens. Fits use 512-token
  documents (`--seq-len 512 --full-length`) so the DeltaNet memories fill. Not yet in the catalog
  pipeline (`arch.py`, `neurons.py`, the tables).
- `data/` is ignored except `data/summary/` (compact results and logs, tracked) and
  `data/corpus/*.provenance.json`. Locally it holds corpora, fits (`data/fits/`), Drowse homes
  (`data/lenses/<name>/`), catalog shards (`data/catalog/<model>/`), activation stores, the
  release staging folders, and outside data (`data/external/`). `neuron_lens/paths.py` names
  them. The catalog lives in `data/catalog/<model>/`, laid out as the release:
  `neurons/<table>/L<layer>.parquet` (a row per pole, the + poles first) and
  `heads/<table>.parquet` (a row per head), key columns in front. Per model: `token_counts-<model>.pt` and
  the chat slice `act-chat-<model>.jsonl`, the other slices shared; Gemma's media sources in
  `data/corpus/media-<modality>.provenance.json`. `data/summary/release_<model>.json` holds what the
  Hub cards say about each model's release, so either model's run rebuilds both cards.

## Invariants

- **The lens rule.** A lens's `out` target reads MLP outputs and attention inputs; its `mid` target
  reads MLP inputs and attention outputs. Don't fit or ship a lens for a point it isn't built for,
  and don't ship borrowed lenses (the J-lens, the logit lens) in the catalog; comparisons belong in
  checks. A head's keys and values are read with the S-lens (what a position holds); its query with the S-lens, or for
  `qk_self_R` with the R-lens at `L − 1`'s output (what the position is about to say; findings §11).
- **Vectors and covectors.** The R-lens transports vectors (writes), the S-lens pulls back covectors (read rows). A
  read that feeds a covector to the R-lens, or takes a cosine or a projection between a covector and a vector, goes
  through the residual's metric (`neuron_lens.metric`: `v = s² ⊙ c`, `s² = σ / |γ|`, σ the
  scale fit's): the raw basis carries a unit per coordinate (Gemma 4's span 80-1,600×), and a Euclidean read there
  weighs coordinates by their units (findings §13). A read that sets a vector against the S-lens's footprints (a write
  in `write_S`, `write_orth_S`; a head's output in `ov_S`; the residual at the firings in `before_S`, `mlp_S`) goes
  through the lens's polar factor (`metric.polar`: `w · U ē`, `K = (K Kᵀ)^½ U`), the footprints with the lens's
  stretch removed: it asks which input token would put the vector there, which the transpose answers only where `K`
  doesn't stretch (§13). Pairing a covector with a vector needs no metric, and a length the model takes itself (an
  RMSNorm's rms: a QK footprint's size, a write's energy at the final norm) stays Euclidean.
- **One top-k.** `catalog.TOP` (64) is the length of every ranked list the catalog stores: token reads, top
  firings, edges, sources, media labels and documents. `_str` previews (8) and `N_CONTEXTS` (5, the firings a
  card shows) are views, and analysis windows in `scripts/checks/` stay what their findings section recorded (`HELD_OUT`
  for the label checks' held-out firings).
- **One format.** A token list is `<name>_ids`, `_val`, `_str`; a list of poles is `<name>_layer`, `_neuron`, `_pole`; shorter lists
  are padded with −1, NaN and an empty preview. Parallel sources get parallel names: R-lens and S-lens reads differ only by
  `_R` / `_S`, weight reads and firing reads share their `_z` meaning, modalities get a table each, and every model has the same
  columns and the same file names, so nothing is special-cased for the default model.
- **Another model in the catalog** comes through `arch.py` and `neurons.py`: what a layer's heads and neurons write as the
  residual sees them, and what rescales that on the way (`attn_norm`, `mlp_norm`, `layer_scale`). The stages read those instead of
  branching on the model. Analyses in `scripts/checks/` tied to one model's named units or findings section may stay that model's.
- **One token mask.** Every token read covers `catalog.kept()`: tokens seen 5+ times in the
  activation corpora, without byte-fallback pieces.
- **Description cards stay leak-free.** `10_labels.py check` and `checks/lens_blind.py` hold out a
  unit's top documents past the first `N_CONTEXTS`. Nothing on an evidence card may summarize them: no
  peak-token counts or agreement shares over all top documents, no ablation run in top contexts.
  Scoring contexts end at the marked token. Findings §8 records why.
- **Estimator changes** need `tests/test_transport.py`, `tests/test_gemma4.py` and `tests/test_qwen35.py` to pass
  (and `tests/test_kernels.py` on a CUDA machine when the kernels change; CI runs the rest on CPU
  for pushes to main, pull requests and weekly, `.github/workflows/tests.yml`, as a signal, not a
  gate): both
  lenses against brute-force Jacobians of the model under `relp_rules`. Supporting another architecture
  means composing its layers from `layers.py`'s components, adding a component (its map, transpose
  and autograd rule) where one is missing, and its geometry in `arch.py`, with a test file of tiny
  random models; `layers.check` refuses what the components don't cover.
- **Both lenses sum over positions** (as the J-lens does) and use RelP. These are fixed, not
  options. More generally, keep one current code path: delete superseded options and formats
  rather than keeping compatibility.
- **Fit files** in `data/fits/` are the authoritative lens format:
  `<lens>-<model>-L<layers>-<corpus>[-T<len>][-s<i>of<n>].pt` (`-T<len>`: documents of exactly
  that many tokens, `--full-length`), holding even- and odd-document sums per
  target. `scale-<model>-L<layers>-dclm.pt` (`02_fit.py --lens scale`, forward passes over the text lens's corpus,
  minutes) holds the residual's per-coordinate sums at every norm's input, the metric's σ; the release ships it as
  `<model>/scale.pt`. Both lenses have an `out` target (layer outputs) and a `mid` target (each layer's MLP
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
- **The scale:** for each model, `02_fit.py --lens scale --docs 1000 --seq-len 128` over the same documents
  (split-half 0.01-0.04). A new lens of record gets its scale fit beside it.
- **Gemma 4 12B:** both lenses on the same 1,000 DCLM documents (`<lens>-gemma-4-12b-L48-dclm`), and
  image and audio lenses (`02_fit.py --modality image|audio`: COCO 2014 val, 1,000 for S and 512 for
  R; ESC-50, 1,615 kept clips for S and 512 for R; fits `...-coco`, `...-esc50`). The media S-lenses
  are seeded in the embedder's subspace and carry a `basis`. The pipeline reads local fits when they
  are in `data/fits/`, else the released lenses (`arch.fits`, the `hub` lens spec).
- **Replacing the lens of record:** fit, export to a new home under `data/lenses/`, rerun
  `08_reads.py` and the stages after it, record the comparison in findings, and rerun
  `12_release.py`.
- **Drowse** ([github.com/stdlogits/drowse](https://github.com/stdlogits/drowse)) loads exported R-lenses
  (`R:<name>` specs in `lens_compare.py`), both targets with their split-half differences, and
  fetches the released ones (`drowse lens fetch <model> rs-r`, `rs-s`). It always writes an R-lens
  to `local/r` under `$DROWSE_HOME` (lens format v7, Drowse 6.0), so each export gets its own home
  under `data/lenses/<name>/` and never overwrites another, or the user's default home. Fitting
  lives here; exact agreement with Drowse's fit is not required.

## Method notes

- **Indexing.** Neuron `n` of layer `L` is coordinate `n` of `down_proj`'s input, 0-indexed.
- **Gemma 4.** A neuron's write is what reaches the residual, `s · γ_post ⊙ W_down[:, n]` (the
  post-feedforward norm's scale held constant, `layer_scalar` `s`); its read rows are scaled by the
  pre-feedforward norm. Logits pass a softcap (`30 · tanh(x / 30)`), which R-lens probabilities
  include. Every document starts with `<bos>` (the sink): statistics start at position 1. Its bands
  are quarters (L0-L11, …, L36-L47). With 16 heads a layer, head tests can't pass BH, so
  `checks/reads_at_firings.py` reports uncorrected p too (findings §13). Design.md's Gemma section has the rest.
- **Poles.** SwiGLU activations take both signs (`up` is linear), so each neuron has a `+` and a
  `−` pole, read and described separately. From weights, a pole's read direction is `â ± b̂`: the
  gate and up rows times the post-attention norm weight, normalized (a heuristic; activations give
  the true poles); a pole whose top firings ride SiLU's negative lobe (`acts`' `neg_lobe` past a half) reads the up row
  alone, `∓b̂` (`neurons.read_rows`, §17). Catalog shards list the `+` poles of neurons `0..n-1`, then the `−` poles.
- **Scale.** A lens read is scale-free (RMSNorm), so its probabilities describe a direction, not an
  effect. Effect sizes need the activation scale (`act_top`); don't rank neurons by lens
  probability across layers.
- **Which side, which depth** (findings §6, §7, for Qwen3-8B; Gemma 4 has the same shape at its own
  depths, its `read_R` from L11, §13). Input side: the S-lens read at L0-L7 (raw at L0-L4, naming literal tokens), both
  lenses at L8, the R-lens `mid` read from L9 (`read_S_z` as a concept-level second view), and at
  L13-L20 the R-lens read with exact drivers and activations beside it. Write side: the
  S-lens (`write_S`) through L8, where early neurons write back the tokens they detect; the R-lens,
  legible from about L4. Heads: OV `ov_S` through L8, `ov_R` from L9; QK `qk_S` throughout (uninformative
  at L13-L20), and its diagonal as `qk_self_S` through L20 (duplicate-token heads) and `qk_self_R` late (L28-L35, mixed
  at L21-L27; Gemma 4 from L24; findings §11, §13). These bands are a prior: weigh a unit's sides by its reads,
  contexts and drivers together. Don't force a word reading where a side is uninformative, and say which side a
  description rests on.
- **Firings are a sample, not a ground truth.** A top firing is one context, summarized by a token. Where a read and a
  firing's token differ, that alone doesn't make the read wrong: the causal test of a write is steering (§3), a pole's firings
  need not predict each other (§4), and an R-lens read reads forward. `checks/reads_at_firings.py` counts how often each read
  names what is counted at the firings, a population result about the reads; the catalog keeps no per-unit test. Don't turn a
  mismatch into a verdict on a unit, or write it up as a limit of the lens.
- **Three relations between units** (§17): wired to (`edges`), driven by and fires with (`sources`). Keep them
  apart; none is the others. Into a head only driven by and fires with exist (`heads/sources`): two heads meet through a
  matrix, not a dot product, so there is no signed weight edge to list.
- **Edges** (`edges`, findings §17) are wiring, not traffic: what a pole's firing would push into each
  later pole, directly at one position. What drives a pole when it fires is mostly units firing with it,
  which the edges rarely list; `checks/edges_in_context.py explain` gives a pole's sources in context.
  Rank edges within a pole, not across the model (early hub neurons top many lists).
- **In context.** Before about L21, a firing's effect varies by context, and a write read describes
  the pole's average push (what steering shows), not what one firing does (findings §4).
- **Undertrained tokens.** Token reads cover `catalog.kept()`: tokens seen 5+ times in
  `data/token_counts-<model>.pt` (`03_token_counts.py`, over the activation corpora), without byte-fallback
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
  take about 2.5-3 hours (Qwen3-8B), 5 hours (Qwen3-14B), 12 hours (Qwen3-32B), 4.5 hours (Gemma 4)
  or 5.5 hours (Qwen3.5-35B-A3B, 512-token documents, `--dim-batch 32`) per text lens per 1,000
  documents on one H100 (`02_fit.py --device cuda`, bf16 with fp32 sums), 45-90 minutes per media
  lens. A stack whose layers don't fit on the GPU beside a fit's working set keeps its first layers
  there and streams the rest from pinned host memory (`02_fit.py --offload N`; the sums come out the
  same to the bit): Qwen3-32B's 62 GB of layers meet about 50 GB of sums, tangents and multiplied-out
  MLPs, so it keeps 20 of its 64 (67 GiB at the peak, 43 s a document).
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
- The catalog's GPU stages: `scripts/catalog_box.sh` installs on a GPU machine and starts
  `scripts/catalog_queue.sh`, detached and resumable: `06`, `17`, `07` for each model, then
  `08`, `09`, `16` and the `measure` half of `checks/edges_in_context.py` (`MEDIA=1` adds `13`, `14`). Each table goes to the
  `UPLOAD` branch of the Hub dataset as soon as its stage has run (`12_release.py --tables`). The release is a full `12_release.py` to that branch, then the same without `--branch` for `main`; it deletes what this machine's catalog lacks, so bring the catalog up to date and run it with `--dry-run` first, which lists the deletions. `catalog_box.sh`'s header has the
  commands to copy the results back. The `score` half runs where the comparison lenses are.
- New CUDA boxes: `scripts/gpu_box.sh` installs the package, downloads the models and corpus,
  runs the estimator tests and benchmarks. Export `OMP_NUM_THREADS=8` on many-core machines:
  torch's default of one thread per core makes CPU ops on small tensors spin for minutes. Run one
  process per GPU (`CUDA_VISIBLE_DEVICES=i ... --shard i/n`) and copy `data/fits/` off a
  temporary machine before it goes. `scripts/fit_queue.sh` fits a list of models on a two-GPU box
  (the scale and the R-lens on GPU 0, the S-lens on GPU 1), downloading each model at its turn and
  dropping it from the cache after.

## Recording results

Whoever runs an analysis adds it to the section of `docs/findings.md` it belongs to (or a new section
at the end): setup (script, hardware, precision, lens), results with controls, and limits, as
concisely as the result allows. A result that changes the overview goes into the README's matching
findings paragraph too. Compact outputs go in `data/summary/`, figures through
`scripts/11_figures.py`. The code and docs are MIT; the lenses and catalog on the Hub are CC BY 4.0
(set in `12_release.py`'s cards). Contributions come as branches and pull requests.
