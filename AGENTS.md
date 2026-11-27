# AGENTS.md

Working notes for coding agents and contributors: what the repo holds, the rules a change must keep, and how to run
and record work. [README.md](README.md) is the overview, [docs/design.md](docs/design.md) the methods (each model's
differences included), [docs/findings.md](docs/findings.md) the results and their limits. Read the findings before
making a claim about a neuron or a lens.

## What this is

A catalog of what every MLP neuron (two poles each) and attention head means. Two models are read in depth, Qwen3-8B
(36 layers × 12,288 neurons, 32 heads) and Gemma 4 12B (48 × 15,360, 16 heads, and images and sound), and the rest of
the Qwen3 family to 14B, post-trained and base, the same way (labels for the post-trained ones). Each unit is read
from three sides: its weights through two lenses (the R-lens forward to the output, the S-lens back to the input
embeddings), its activations over 15M positions, and its exact token drivers. `NEURON_LENS_MODEL` selects the model
for every stage (Qwen3-8B by default). `scripts/12_release.py` releases the lenses (`stdlogits/rs-lenses`) and the
catalog (`stdlogits/neuron-lens`) on the Hugging Face Hub, one card per repo covering every model: its `OTHERS` (the
Pythia suite to 6.9B, OLMo 3 7B) have catalogs without labels, and its `LENS_ONLY` (Qwen3.5, Qwen3-30B-A3B) get
lenses alone.

Neurons and heads share the tables `acts`, `drivers`, `reads`, `labels` and `sources`; neurons add `firings` and
`edges`, the pruned Qwen3 pairs `lineage`, Gemma 4 `image` and `audio`, and sparse models `experts/acts` and
`experts/reads`. The dataset card documents every column.

## Layout

- `neuron_lens/`, the package:
  - the estimator: `transport.py` (the lenses), `layers.py` (every model's decoder layers as RelP linear maps and
    their transposes, one implementation per component, composed from what each layer holds), `rules.py` (the same
    rules as autograd patches, by operation: the tests' reference), `kernels.py` (Triton kernels for the CUDA hot
    spots, with `layers.py`'s torch passes as their reference);
  - each model: `anatomy.py` (a layer's norms, mixer and MLP by attribute path), `arch.py` (geometry, weight names,
    fits of record, reading plans, depth bands), `neurons.py` (poles' read and write directions), `metric.py` (the
    residual's metric), `experts.py` (a sparse MLP's neurons on the activation side), `weights.py` (named tensors
    without loading the model);
  - reading: `readout.py` (vectors through a lens), `heads.py` (head circuits as token maps), `lenses.py` (fits,
    released lenses, Drowse lenses);
  - data: `corpus.py`, `catalog.py` (the catalog's layout, kept tokens and table helpers; browse it with
    `python -m neuron_lens.catalog`), `firings.py` and `sources.py` (the passes at a pole's firings), `stats.py`,
    `known.py` (named neurons), `media.py` and `media_sets.py` (Gemma's images and sounds), `paths.py`.
- `scripts/NN_*.py`, the pipeline: `01`-`05` build the corpora, fit and export the lenses and count tokens; the
  catalog runs `06` activations, `17` firings and sources, `07` drivers, `08` reads, `checks/reads_at_firings.py`,
  `16` edges, then `10` labels; Gemma 4's media run `13`, `checks/crossmodal.py`, `14`, `15`; `11` draws the figures
  and `12` releases (there is no `09`). The README's Reproducing table has costs. The `*.sh` scripts set up GPU boxes
  and queue runs (Running jobs). `scripts/prompts/` holds the labellers' prompts.
- `scripts/checks/`: analyses, each behind a findings or design section; every script the findings name is here
  (`causal_lens.py`, `crossmodal.py` and `head_media.py` also make released files). Checks read the catalog's tables
  and get documents from `catalog.documents`, so most run from the release and the model.
- `tests/`: only what a run can't show. The estimators against brute-force Jacobians on tiny random models, a file per
  family (`test_transport.py` Qwen3 dense and MoE, `test_gemma4.py`, `test_qwen35.py`, `test_pythia.py`,
  `test_olmo.py`), and `test_kernels.py` (skipped without CUDA). What fails loudly or reads wrong in the output gets
  fixed when it comes up, without a test.
- `data/` is ignored except `data/summary/` (compact results and logs) and `data/corpus/*.provenance.json`;
  `paths.py` names the rest. The catalog, `data/catalog/<model>/`, is laid out as the release:
  `neurons/<table>/L<layer>.parquet` (a row per pole, the + poles first), `heads/<table>.parquet` (a row per head),
  key columns in front. `data/summary/release_<model>.json` holds what the cards say about each model, so a catalogued
  model's release rebuilds every card.

## Architectures

The stages read what differs between models from `anatomy.py`, `arch.py` and `neurons.py` and never branch on the
model (Invariants). design.md has a section for each family beyond Qwen3. Labels (`10`) need a reading plan in
`arch.PLANS`, which only the Qwen3 models and Gemma 4 have.

- **Qwen3** (0.6B-14B, post-trained and base): catalogued. Qwen3-4B and 0.6B are pruned from the 8B and the 1.7B, so
  the sizes are three lineages, not five points (findings §20).
- **Gemma 4 12B:** catalogued, with media lenses and tables. Every document starts with `<bos>`, the sink, so
  statistics start at position 1, and its logits pass a softcap that R-lens probabilities include.
- **Pythia:** estimator and stages; lenses released for 14M-6.9B. Training checkpoints are revisions
  (`02_fit.py --revision step<N>`, `fit_queue.sh model@step<N>`), fitted locally, since the release has no place for
  them. `arch` carries its differences as `n_poles` (one pole), `parallel` (blocks), `centred` (LayerNorms) and
  `half_dtype` (fp16).
- **OLMo 2 and 3, OLMoE, Olmo Hybrid:** estimator and stages; lenses released for OLMo 3 7B, whose checkpoints back
  findings §25.
- **Qwen3.5** (Gated DeltaNet layers beside gated attention; sparse MLPs in the 35B-A3B) **and Qwen3-30B-A3B**
  (Qwen3's attention with sparse MLPs): estimator and stages, with Triton kernels for DeltaNet; no catalog has run on a
  full Qwen3.5 model yet. DeltaNet models' lenses fit 512-token documents (`--seq-len 512 --full-length`: `-T512`
  fits, `arch.seq_tag`), and a DeltaNet layer's heads are its value heads. A sparse layer's neurons are its experts'
  side by side, then its shared expert's.

## Invariants

- **The lens rule.** A lens's `out` target reads MLP outputs and attention inputs, its `mid` target MLP inputs and
  attention outputs (in a parallel block, where both sublayers write `out`, a head's write takes `out`:
  `arch.head_write_lens`). Don't fit or ship a lens for a point it isn't built for, and don't ship borrowed lenses (the
  J-lens, the logit lens) in the catalog; comparisons belong in checks. A head's keys and values are read with the
  S-lens (what a position holds); its query with the S-lens, or for `qk_self_R` with the R-lens at `L − 1`'s output
  (what the position is about to say; findings §10).
- **Vectors and covectors.** The R-lens transports vectors (writes); the S-lens pulls back covectors (read rows).
  Turning one kind into the other, or comparing two of a kind, goes through the residual's metric (`metric`:
  `v = s² ⊙ c`, `s² = σ / |γ|`), since the raw basis gives each coordinate its own unit (Gemma 4's span 80-1,600×).
  A vector set against the S-lens's footprints (`write_S`, `write_orth_S`, `ov_S`, `before_S`, `mlp_S`) goes through
  the lens's polar factor (`metric.polar`). Pairing a covector with a vector needs no metric, and a length the model
  takes itself (an RMSNorm's rms) stays Euclidean (design.md's Reading neurons; findings §15).
- **Both lenses sum over positions** (as the J-lens does) **and use RelP.** These are fixed, not options.
- **One code path.** Delete superseded options and formats rather than keeping compatibility.
- **One top-k.** `catalog.TOP` (64) is the length of every ranked list the catalog stores. `_str` previews (8) and
  `N_CONTEXTS` (5, the firings a card shows) are views; analysis windows in `scripts/checks/` stay what their findings
  section recorded.
- **One format.** A token list is `<name>_ids`, `_val`, `_str`; a list of poles `<name>_layer`, `_neuron`, `_pole`;
  short lists are padded with −1, NaN and an empty preview. Parallel sources get parallel names (R and S reads differ
  only by `_R` / `_S`; weight and firing reads share their `_z` meaning), each modality gets a table, and every model
  has the same columns and file names.
- **Another model** comes in through `anatomy.py`, `arch.py` and `neurons.py`, never a branch in a stage. Only
  `anatomy.parts` reads a norm by name (`post_attention_layernorm` is the MLP's input norm in Qwen3 and the attention
  output's norm in OLMo). Heads sit in a grid of `arch.max_heads` a layer; head tables list the heads that exist
  (`catalog.write_heads`, `catalog.gridded`). Supporting an architecture means composing its layers from `layers.py`'s
  components, adding a component (map, transpose and autograd rule) where one is missing, and a test file of tiny
  random models; `layers.check` refuses what the components don't cover. Checks tied to one model's named units or
  findings section may stay that model's.
- **One token mask.** Every token read covers `catalog.kept()`: tokens seen 5+ times in the activation corpora
  (`03_token_counts.py`), without byte-fallback pieces. Undertrained tokens push some early neurons to huge
  activations.
- **Description cards stay leak-free.** `10_labels.py check` and `checks/lens_blind.py` hold out a unit's top
  documents past the first `N_CONTEXTS`, so nothing on an evidence card may summarize them: no peak-token counts or agreement shares over all top documents, no ablation in top
  contexts. Scoring contexts end at the marked token (findings §13).
- **Estimator changes** pass every `tests/` file (and `test_kernels.py` on a CUDA machine when the kernels change).
  CI runs the rest on CPU for pushes to main, pull requests and weekly, as a signal, not a gate.
- **Fit files** in `data/fits/` are the lens format of record:
  `<lens>-<model>[-<revision>]-L<layers>-<corpus>[-T<len>][-s<i>of<n>].pt`, even- and odd-document sums per target
  (`lenses.load_fits` sums shards, each half apart; `Accumulator.mean` divides by the documents). Each lens has an `out`
  target (layer outputs) and a `mid` target (each layer's MLP input). `scale-<model>-L<layers>-dclm.pt` holds the
  residual's sums at every norm's input, from which the release computes the metric's σ (`<model>/scale.pt`).
- **Findings section numbers are stable.** Docstrings, figure footers and the Hub cards cite them. Fold a new result
  into its section or add one at the end; don't renumber.

## The lenses of record

- For each model, the R-lens, the S-lens and the scale are fitted on the first 1,000 rows of DCLM-shuffled
  (`01_lens_corpus.py`, one extendable file at a pinned revision) with `02_fit.py --model <id> --lens R|S|scale --docs
  1000` (its defaults, Qwen3-0.6B and 12 documents, are a smoke test; `--full-length` takes the first 1,000 documents
  long enough). Qwen3-8B's split-half: R 0.06 or less at every layer, S 0.00-0.05, scale 0.01-0.04 (findings §1).
  The S-lens's reads are legible at L0-L8; past L8 use the exact drivers for the input side (§7).
- Qwen3-8B's R-lens is exported by `04_export_rlens.py` to `data/lenses/dclm`, its halves to `dclm-even` and
  `dclm-odd`. An R-lens fitted on 500 chat conversations (`data/lenses/chat`) is kept for comparison (§1).
- Gemma 4 adds image and audio lenses (`02_fit.py --modality image|audio`, on COCO and ESC-50; design.md's Images
  and sound).
- The pipeline reads local fits when they are in `data/fits/`, else the released lenses (`arch.fits`).
- **Replacing a lens of record:** fit (with its scale), export to a new home under `data/lenses/`, rerun the stages
  that read the lenses (`17_firings.py --parts neurons`, `08_reads.py` and the stages after it), record the comparison
  in findings, and rerun `12_release.py`.
- **Lens specs** (`lenses.load_lens`, for `checks/lens_compare.py` and the checks): `logit`, `J` (Neuronpedia's
  J-lens), `R` (the default Drowse home), `R:<name>[+<name>]` (homes in `data/lenses/`, merged), `fit:<stem>[@mid]`,
  `hub[@mid]` (the release) and `record[@mid]` (`arch.fits`).
- **Drowse** ([github.com/stdlogits/drowse](https://github.com/stdlogits/drowse), 6.0 or later) loads exported R-lenses
  and fetches the released ones (`drowse lens fetch <model> rs-r`, `rs-s`). It always writes to `local/r` under
  `$DROWSE_HOME`, so each export gets its own home under `data/lenses/<name>/`, never another's or the user's default.
  Fitting lives here; exact agreement with Drowse's fit is not required.

## Reading units

- **Indexing.** Neuron `n` of layer `L` is coordinate `n` of `down_proj`'s input, 0-indexed; in a sparse layer,
  neuron `e·I + i` is neuron `i` of expert `e`, and its activation includes the routing weight. In Qwen3-4B and 0.6B
  neurons sit in order of the parent's activation (findings §20), so an index range is a slice by activation rank:
  draw neurons at random. `checks/lineage.py` maps each to its parent and writes the `lineage` table.
- **Poles.** SwiGLU activations take both signs (`up` is linear), so each neuron has a `+` and a `−` pole, read and
  described separately; catalog shards list the `+` poles of neurons `0..n-1`, then the `−` poles. From the weights a
  pole's read direction is `â ± b̂` (a heuristic; activations give the true poles), or the up row alone, `∓b̂`, for a
  pole whose firings ride SiLU's negative lobe (`neurons.read_rows`, §7). Pythia's GELU gives one pole.
- **Scale.** A lens read is scale-free, so it describes a direction, not an effect. Effect sizes need the activation
  scale (`act_top`); don't rank neurons by lens probability across layers.
- **Which read at which depth** (§8). `arch.PLANS` holds each model's reading plan, the input, write and head read a
  label card shows at each depth, and its edges cut the depth bands every summary, label check and the site use
  (`arch.depth_bands`; after changing `PLANS`, `python -m neuron_lens.arch` rewrites `data/summary/depth.json`).
  Qwen3-8B's bands: sensory L0-L3, association L4-L8, workspace L9-L20, motor L21-L35. A base model takes its
  post-trained model's plan, and a model without one quarters. QK sits outside the plans: `qk_S` throughout
  (uninformative at L13-L20), `qk_self_S` through L20, `qk_self_R` late (L28-L35; Gemma 4 from L24; §10, §15). The
  plan is a prior: weigh a unit's sides by its reads, contexts and drivers together, don't force a word reading where
  a side is uninformative, and say which side a description rests on.
- **Firings are a sample, not a ground truth.** A top firing is one context. Where a read and a firing's token differ,
  that alone doesn't make the read wrong: steering is the causal test of a write (§3), a pole's firings need not
  predict each other (§4), and an R-lens read reads forward. Before about L21 a write read is the pole's average push,
  not what one firing does. `checks/reads_at_firings.py` gives population results about the reads; the catalog keeps
  no per-unit test, so don't turn a mismatch into a verdict on a unit or a limit of the lens.
- **Three relations between units** (§12), kept apart: wired to (`edges`), driven by and fires with (`sources`).
  Edges are wiring, not traffic: what drives a firing pole is mostly units firing with it, which the edges rarely
  list (`checks/edges_in_context.py explain`). Rank edges within a pole, not across the model. Into a head only
  driven-by and fires-with exist: two heads meet through a matrix, so there is no signed edge.
- **Descriptions are soft:** weighted labels with a confidence and scores. In the docs, let the wording say what is
  measured and what is interpretation ("suggests", "likely").
- **Outside labels** (`known.EXTERNAL`) are cross-references, not ground truth; our reads, activations and causal
  checks arbitrate. `known.KNOWN` holds steering-verified persona neurons from an earlier experiment.
- **Judging a lens:** by whether its reads are coherent and make sense in context, not by a benchmark. Report the
  split-half difference `02_fit.py` prints. Many neurons reading alike can be the shared workspace subspace (the
  J-space), not a flaw.

## Running jobs

- An Apple-silicon Mac runs everything but the lens fits and the GPU passes. A text lens takes about 2.5-3 hours per
  1,000 documents on one H100 for Qwen3-8B (Qwen3-14B 5 h, Qwen3-32B 12 h, Gemma 4 4.5 h, Qwen3.5-35B-A3B 5.5 h with
  `--dim-batch 32`), a media lens 45-90 minutes. A model whose layers don't fit beside a fit's working set keeps its
  first `--offload N` layers on the GPU and streams the rest from pinned host memory, with the same sums (Qwen3-32B:
  `--offload 20`).
- **One GPU process per GPU.** Check for running jobs first (`ps`, `nvidia-smi`); on a shared machine, never run
  beside or stop a job that isn't yours. Light CPU work (weight-only reads, tests) can run beside a GPU job; heavy CPU
  work slows it on unified memory.
- **Long jobs run detached** (`02_fit.py --detach`, `06_activations.py --detach`; under `caffeinate` on macOS), log to
  `logs/`, save every `--save-minutes` (10) and resume when rerun. `06` needs about 21 GB of GPU memory with eager
  attention at its default 32 documents a batch, and rerun after a complete pass it rebuilds the `acts` tables. `13_media_activations.py` holds about 220 GB of RAM for the images: run
  the modalities one at a time. Sparse models run `06` and `07` in layer groups (`--layers a-b`; the group from layer 0
  collects the heads) and `17` a few layers at a time, in fp32.
- **New boxes:** `scripts/gpu_box.sh` installs the package, downloads the models and corpus, and runs the tests and
  benchmarks. Export `OMP_NUM_THREADS=8` on many-core machines, and `HF_HUB_DISABLE_XET=1` if Hub downloads stall.
  Several GPUs take one process each (`CUDA_VISIBLE_DEVICES=i ... --shard i/n`). `scripts/fit_queue.sh` fits a list of
  models (the scale and the R-lens on GPU 0, the S-lens on GPU 1). Copy `data/fits/` off a temporary machine before it
  goes.
- **The catalog's GPU stages:** `scripts/catalog_box.sh` installs on a GPU machine and starts `scripts/catalog_queue.sh`
  for `MODELS` (`MEDIA=1` adds Gemma 4's media stages), detached and resumable, one queue per GPU. With `UPLOAD` set,
  each table goes to that branch of the Hub dataset when its stage finishes (`12_release.py --tables`). The queue's header lists the
  stages; the box script's has the commands to copy the results back.
- **Releasing:** a full `12_release.py --branch <b>`, then the same without `--branch` for `main`. It deletes what this
  machine's catalog lacks, so bring the catalog up to date and run `--dry-run` first, which lists the deletions. A
  `LENS_ONLY` model takes `--parts lenses` (its scale included); its catalog goes to a branch with `--tables`. A
  catalog a GPU box sent to a branch (`--tables`) reaches `main` with `--parts neurons,heads,files --from-branch <b>`
  (`--dry-run` first), a server-side copy of the model's folder; the card's numbers need the model's token counts here
  (`data/token_counts-<model>.pt`, the folder's `token_counts.pt`).

## Recording results

Whoever runs an analysis adds it to its section of `docs/findings.md` (or a new section at the end): setup (script,
hardware, precision, lens), results with controls, and limits, as concisely as the result allows. Write what was
measured, not how the run went: no dates, run costs, earlier versions or fixed bugs, unless a comparison with a
variant is itself the result. A result that changes the overview goes into the README's matching finding too.
Compact outputs go in `data/summary/`, figures through `scripts/11_figures.py`. The code and docs are MIT; the lenses
and catalog on the Hub are CC BY 4.0 (set in `12_release.py`'s cards). Contributions come as branches and pull
requests.
