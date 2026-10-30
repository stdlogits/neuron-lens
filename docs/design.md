# Design and methods

How the lenses are fitted and the catalog is built, for understanding or reproducing them. Results are
in [findings.md](findings.md) ("§N" refers to its sections); the pipeline's scripts and costs are in
the [README](../README.md#reproducing).

## The units

The first model is Qwen3-8B (chat): 36 decoder layers, a 4,096-dimensional residual stream, 12,288
SwiGLU MLP neurons and 32 attention heads (8 key-value heads, `head_dim` 128) per layer, 151,936
tokens. The second, Gemma 4 12B, is read the same way where its architecture allows; [its
section](#gemma-4-12b) gives where it differs and how its images and sounds are read.
`NEURON_LENS_MODEL` selects the model for every stage (Qwen3-8B by default).

**Neurons and poles.** Neuron `n` of layer `L` is coordinate `n` of `down_proj`'s input, 0-indexed. It
reads the normalized MLP input `x̂` (after the post-attention RMSNorm, weight `γ`) through two rows,
`a = γ ⊙ W_gate[n]` and `b = γ ⊙ W_up[n]`, and adds `act · W_down[:, n]` to the residual, with `act =
silu(a·x̂)(b·x̂)`. Since `up` is linear, `act` takes both signs: positive where `x̂` aligns with `a` and
`b`, negative where it aligns with `a` and against `b`. Each sign is a *pole*, catalogued separately:
the `+` pole writes `+W_down[:, n]` and reads along `â + b̂`, the `−` pole writes `−W_down[:, n]` and
reads along `â − b̂` (unit rows). The read direction is a heuristic (gate and up rows are nearly
orthogonal, median cosine 0.00 at every layer but L2); activations give the true poles.

**Heads.** Head `h` of layer `L` reads layer `L − 1`'s output (the embedding at L0) through its query,
key and value projections and writes `W_O^h z` into the residual before layer `L`'s MLP. Its QK circuit
decides which earlier position it reads, its OV circuit what it writes from there.

**Scale.** An R-lens read is scale-free (the final RMSNorm removes the norm), and the catalog's S-lens
reads use unit directions, so a read describes a direction, not an effect. Effect sizes need `act_top`
(a pole's typical high activation) and `w_norm = ‖W_down[:, n]‖`; don't rank units by lens probability
across layers.

## The lenses: one estimator in two directions

Both lenses average a Jacobian of the residual stream over a corpus and read it with a token matrix
(`neuron_lens/transport.py`). `h_L(t)` is layer `L`'s residual at position `t`, `h_final` the last
layer's output before the final norm, `e(s)` the token embedding at `s`.

| | R-lens | S-lens |
|---|---|---|
| Jacobian | `J_L = mean_t Σ_{t'≥t} ∂h_final(t')/∂h_L(t)` | `K_L = mean_t Σ_{s≤t} ∂h_L(t)/∂e(s)` |
| reads with | `W_U`: what a vector makes the model say, here and later | `W_E`: which tokens, here or earlier, put it there |
| read of `h` | `softmax(W_U · norm(J_L h))` | token `v` scores `h · K_L (W_E[v] − mean(W_E))` |
| mode | reverse: one cotangent per output direction | forward: one tangent per input direction |
| origin | the J-lens (Gurnee et al. 2026) under RelP: the R-lens (camilablank et al. 2026) | Theia Vogel's K-lens (the J-lens taken back to the embeddings) under RelP |

One pass per basis direction of the fixed end covers every layer at the other, so either lens costs
`d_model` passes per document, 256 at a time. Each is fitted at two points of every layer:

| | `out`: the layer's output | `mid`: the layer's MLP input (after attention) |
|---|---|---|
| R-lens | reads MLP writes (`write_R`); L0-L34, since L35's output is the target | reads MLP read directions (`read_R`) and attention writes (`ov_R`) |
| S-lens | reads MLP writes (`write_S`) and attention inputs (`ov_*`, `qk_*` at `L − 1`) | reads MLP read directions (`read_S`) |

**The lens rule:** `out` reads MLP outputs and attention inputs; `mid` reads MLP inputs and attention
outputs. A read anywhere else extrapolates, and the catalog ships none.

- **Positions.** Seeds go at positions `[16, T − 1)`: the first 16 are left out for the attention sink
  (Drowse's cut, which its loader requires) and the last has no later position. Attention is causal,
  so nothing seeded there reaches an earlier position. Documents weigh equally and alternate between
  an even and an odd half.
- **Summed over offsets,** as in the J-lens: a read mixes "this token" with "an earlier token". The
  far offsets are noisy but may hold interesting behaviour, and the sum keeps them at no extra cost.
- **RelP's rules** (Rezaei Jafari et al. 2025) instead of the true derivative: decoder RMSNorms hold
  their `rsqrt` constant (LN rule), `silu(g)` is linearized as `sigmoid(g)·dg` (identity rule), and
  `silu(g)·up` passes half of each factor's change (half rule, β = 0.5). Attention, the q/k norms and
  RoPE keep their true derivative. RelP matched or beat the true derivative wherever they were
  compared: the R-lens leads the J-lens at predicting steering from L4 to L23 (the J-lens leads at L0,
  they tie at L1-L3; §3), and on the input side RelP halves the S-lens's noise and fits footprints
  better from L1 (§1, §6).
- **Hand-written layer maps.** Under the rules each decoder layer is, for a given document, a fixed
  linear map: the norms scale each position, the MLP scales its gate and up projections per position
  and neuron, and attention keeps its true derivative. `_linearize` runs the model once per document
  and keeps those constants; `_jvp` applies a layer's map to a block of tangents and `_vjp` its
  transpose to cotangents, as a few matrix products. The R-lens's map is exactly the transpose of the
  S-lens's. `neuron_lens/rules.py` applies the same rules to the model as autograd patches, the tests'
  reference; `_check` refuses what the maps don't cover (projection biases, other norms or
  activations). Supporting another architecture means writing its maps, as `neuron_lens/gemma4.py`
  does for Gemma 4.
- **Tests** (`tests/test_transport.py`, on tiny Qwen3 models, one with grouped-query attention): both
  lenses and both targets match brute-force Jacobians of the patched model (1e-4), and the R-lens
  matches Drowse's own RelP fit (1e-5) and loads in Drowse. `tests/test_arch.py` and
  `tests/test_gemma4.py` do the same for small random Gemma 4 models, images and sound included.
  Estimator changes must pass them.
- **Fit files** (`data/fits/<lens>-<model>-L<layers>-<corpus>[-s<i>of<n>].pt`) hold even- and odd-half
  sums per target. Fits save every 10 minutes, resume when rerun (also onto a longer prefix of the
  corpus), and shard across GPUs (`--shard i/n`); `lenses.load_fits` adds shards and halves.
  `04_export_rlens.py` writes an R-lens to its own Drowse home under `data/lenses/<name>/`.

## Cost and precision

One 128-token document, every layer, bf16 (`scripts/checks/fit_bench.py`;
`data/summary/fit_bench_{h100,mps}_2026-10-28.log`):

| | R-lens (`out`) | S-lens (`mid` and `out`) |
|---|---|---|
| Qwen3-0.6B, Apple silicon (MPS) | 6.8 s | 7.6 s |
| Qwen3-0.6B, one H100 | 0.36 s | 0.36 s |
| Qwen3-8B, MPS | 231 s | 291 s |
| Qwen3-8B, one H100 | 11.2 s (autograd: 27.2 s) | 11.5 s (autograd: 63.6 s) |

That is about 6.4 PFLOP of matrix products per 8B document. On the H100 the hand-written maps under
`torch.compile` (fixed shapes; a shape-generic recompile made full-length documents 10% slower) spend
82% of the time in matrix products at about 670 TFLOP/s, against a floor of about 9 s a document.
Fused projections and CUDA graphs gave nothing more.

- **bf16 is accurate enough.** It is 0.72% (R) and 0.65% (S) from fp32 on 8B (median over layers),
  while two documents' estimates differ by 65%, so sampling error dominates: about 1.5% at 1,000
  documents.
- **FP8 is not.** Row-scaled FP8 products are 1.5× faster but put the estimate 3.8% (R) and 4.5% (S)
  from fp32, mostly as a bias from the rounded weights that does not average over documents (the
  mean of four keeps 3.6% of the 3.8%). At equal time it ends worse than bf16.
- **Random-probe sketching doesn't pay** (an argument from Monte Carlo variance, not a measurement):
  each document's sketch noise scales with its whole Jacobian, not its deviation from the mean, so at
  equal compute it loses to exact probing.

## The lenses of record

- **R-lens:** both targets fitted with `02_fit.py --lens R` on the first 1,000 rows of DCLM-shuffled,
  128 tokens each, on one H100 in bf16, exported to `data/lenses/dclm` with its 500-document halves in
  `dclm-even` and `dclm-odd`. Split-half 0.06 or less at every layer (§1).
- **S-lens:** the same documents, `02_fit.py --lens S`; split-half 0.00-0.05.
- **For comparison:** an R-lens fitted on 500 UltraChat conversations (`data/lenses/chat`, §1), and,
  in `checks/lens_compare.py`, the logit lens and Neuronpedia's J-lens. Borrowed lenses stay in checks;
  the catalog ships only the lenses of record.

**How lenses are judged:** on their merits, by whether their reads are coherent and make sense in
context, not by a fixed benchmark, which is fallible in its own ways. Every fit reports its split-half
difference per layer, which also says whether more documents would help. Causal checks (§3, §10)
inform; they are not gates. Many neurons reading alike is not a flaw in itself: J-lens reads
concentrate in a shared subspace, the J-space, loosely a global workspace (Gurnee et al. 2026).
Hub tokens, which top many reads at once from the lens's geometry, are another cause; layer-contrastive
reads remove them.

**Replacing a lens of record:** fit, export to a new home, rerun `08_reads.py` and the stages after it,
record the comparison in the findings, and rerun `12_release.py`.

## Reading neurons

`08_reads.py` reads every pole from the weights, with `w = ±W_down[:, n]`, `c = â ± b̂`, and `norm` the
final RMSNorm with its weight (`neuron_lens/readout.py`):

| columns | read |
|---|---|
| `write_R` | `softmax(W_U · norm(J_L^out w))`, the R-lens read of the write |
| `write_S` | `w · K_L^out (W_E[v] − ē)`: the tokens whose footprint the write imitates |
| `read_R` | `softmax(W_U · norm(J_L^mid c))`: what the residual says when it drives the pole |
| `read_S` | `c · K_L^mid (W_E[v] − ē)`: the input tokens that drive the pole |
| `w_norm`, `cos_ab` | `‖W_down[:, n]‖`, `cos(a, b)` |
| `vocab_skew`, `vocab_kurt` | of the pole's logit-lens logits: prediction and suppression neurons (Gurnee et al. 2024; §5) |
| `null_share` | the write's share in the bottom 1% of singular directions of `W_U diag(γ)`: entropy neurons (Stolfo et al. 2024; §9) |

- **Layer-contrastive reads** (`_z`) z-score each token's score across the layer's poles (S-lens rows
  as unit vectors), which removes hub tokens. Raw `read_S` keeps literal tokens early; `read_S_z`
  gives concepts later but loses the literal peak token.
- **Depth:** each read keeps its top 64 tokens with scores (`_ids`, `_val`; `_str` the top 8 decoded);
  signal holds to rank 64 (§7). At L35, `write_R` is the logit lens.
- **Which read to trust where** depends on depth (§7): the S-lens on the input side through L8, the
  R-lens from L9; `write_S` through L8, `write_R` from about L4; only exact drivers at L13-L20 on the
  input side. These are priors; `legible` says which reads hold for each pole.

## Reading heads

A head's token maps are read between its own read and write points (§11): a token `v` at its input is
`X_v`, its S-lens footprint at `L − 1`'s output (`K^out_{L−1}(W_E[v] − ē)`; the embedding at L0).

- **OV** (`ov_R`, `ov_S`): the head's write for `X_v`, `W_O^h W_V^h (γ₁ ⊙ X_v)` with `γ₁` the input
  norm's weight, read through the R-lens `mid` (`ov_R`) or the S-lens footprint at `mid` (`ov_S`). Per
  token, the map's diagonal z-scored within its column: where `v` ranks in what the head writes when
  it reads `v`. Top 64 copied, and `_neg` the 64 most suppressed. Columns' means and variances come
  from the mean and covariance of the head's 128-dimensional projections, so the 103,830² map is never
  built. `ov_R_copy` and `ov_S_copy` are Elhage et al.'s copying score `Σλ/Σ|λ|` of each map.
- **QK** (`qk_S`, `qk_self_S`): the attention logit between a query footprint and a key footprint,
  through `q_norm` and `k_norm` (their true derivative, averaged over real positions) and RoPE's
  relative rotation averaged over the head's own attention offsets. Footprints are taken at unit
  length, since attends-to ranks keys across tokens. `qk_S`: each key token's z over the vocabulary
  under the head's mean query (what it attends to, whatever the query); `qk_self_S`: each token's z in
  its own row (it seeks itself); `qk_self` their median; `qk_ctx_rho` and `qk_ov_rho`: Spearman of
  `qk_S` with the attention tokens receive in context, and with `ov_R`.

## Activations, drivers and tests

- **Activations** (`06_activations.py`): one bf16 pass with eager attention over the activation
  corpora (below), hooking each `down_proj` input and each head's output. Per pole: exact counts of
  `|act|` on a quarter-octave grid from 2⁻¹⁰ to 2¹⁰, which give `rate` (past `act_top / 4`) and the
  quantiles; the top firing in each document, keeping the 24 best; position 0 from every kept token run
  alone. Per head, the same for the output norm `‖W_O^h z‖`, plus attention patterns (sink, previous,
  self) and the in-context read of its firings' outputs (`out_R_z`); induction and duplicate-token
  scores from 64 random sequences said twice.
- **Drivers** (`07_drivers.py`): every kept token at position 24 of 8 documents (4 web, 2 code, 2 chat),
  before the documents' own next 16 tokens. Per pole, the 64 tokens with the largest mean activation at
  the token and summed over the 16 after; per head, the largest output norm at the token (`drivers`) and
  the most attention received from the 16 after (`keys`).
- **Legibility tests** (`09_legible.py`): each unit's reads scored on its own top firings against
  mismatched pairs, at a 5% false discovery rate (§7, §12).

## Labels

`10_labels.py` writes an evidence card per neuron (both poles) or per head and has Claude Haiku 5.5
label each through the Message Batches API: where the unit fires, what it writes when amplified (and
for heads, what it attends to), each with a confidence. Labels are soft: weighted, with a confidence,
naming the evidence they rest on.

**Cards stay leak-free.** Scoring uses a unit's top documents past the first 5 (`N_CONTEXTS`), so
nothing on a card may summarize them: no peak-token counts or agreement shares over all top documents,
no ablation in top contexts. Scoring contexts end at the firing token. `checks/lens_blind.py` and
`checks/haiku_blind.py` score labels by detection and by steering (§8); `10_labels.py check` scores the
head labels (§12).

## Token mask

Undertrained tokens drive some early neurons to huge activations (L2 n6311 + reaches 132.7 on `�`).
Every token read covers only `catalog.kept()`: tokens seen 5+ times in the four activation corpora
(87.9M tokens; `03_token_counts.py`), without byte-fallback pieces, 103,830 of 151,669. A DCLM-only
count would keep 36,008 and drop most Chinese tokens.

## Corpora

**Lens corpus** (`01_lens_corpus.py`): rows `[0, 1000)` of
[`HuggingFaceFW/dclm_100BT-shuffled`](https://huggingface.co/datasets/HuggingFaceFW/dclm_100BT-shuffled)
at revision `2fa015e4`, row ids in `data/corpus/dclm.provenance.json`. The set is globally shuffled, so
its first rows are a uniform sample, and a larger `--docs` extends the same list. Fits use each
document's first 128 tokens.

**Activation corpora** (`05_act_corpus.py`, provenance in `data/corpus/act-<slice>.provenance.json`),
up to 256 tokens a document:

| slice | source | documents | positions |
|---|---|---|---|
| dclm | DCLM-shuffled from row 1,000 (none in the lens corpus) | 40,000 | 9.46M |
| zh | [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) `cmn_Hani` | 8,000 | 1.89M |
| code | [codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) Python, leading comment and blank lines stripped | 7,990 | 1.98M |
| chat | [UltraChat-200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) `test_sft`, in the chat template | 8,000 | 2.04M |

DCLM alone would make Chinese- and code-specialized neurons look dead (they are 2-6% of some layers,
§2). The chat slice gives the speaker frame the chat model was trained for; frames matter for steering
(§3 steers in document contexts, §10 in "Who are you?", §8 both). Only the DCLM slice is a uniform
sample; the zh slice is largely traditional script, code is Python only, chat is one synthetic
dataset written by ChatGPT on both sides.

## Gemma 4 12B

The second model is the base model `google/gemma-4-12B` (`gemma4_unified`): 48 decoder layers, a
3,840-dimensional residual stream, 15,360 GELU-tanh MLP neurons and 16 query heads a layer, 262,144
tokens, and image and audio inputs. `NEURON_LENS_MODEL=google/gemma-4-12B` selects it for every
stage. `neuron_lens/arch.py` holds its geometry, `neuron_lens/neurons.py` its neurons' directions,
`neuron_lens/gemma4.py` its layer maps and `neuron_lens/media.py` its image and audio documents. The
tables, tests and conventions are Qwen3-8B's, with these differences:

- **The write and read.** The MLP output passes a post-feedforward RMSNorm before the residual, and
  each layer's output is scaled by a learned `layer_scalar` `s`. A neuron's write is what reaches the
  residual, `s · γ_post ⊙ W_down[:, n]`, with the norm's scale held constant as the LN rule holds it
  (a direction: its size at a position also depends on that position's norm). The read rows are
  `W_gate[n]` and `W_up[n]` times the pre-feedforward norm's weight. Activations are
  `gelu_tanh(a·x̂)(b·x̂)` and take both signs, so each neuron has two poles.
- **Heads.** Layers 5, 11, …, 47 are global; the others attend within a 1,024-token window. Sliding
  layers have 8 key-value heads of size 256; global layers one of size 512, whose keys are also its
  values (no `v_proj`), with RoPE on a quarter of each head. Values pass a weightless RMSNorm, read
  with its true derivative averaged over positions, as the q/k norms are. The attention output
  passes a post-attention RMSNorm, so a head's write is `W_O^h` times that norm's weight. `kv_head`
  in the head tables follows each layer's grouping.
- **RelP rules.** As for Qwen3-8B, with the four residual norms (input, post-attention, pre- and
  post-feedforward) holding their scale, and the identity rule for GELU-tanh: `gelu(g)` is linearized
  as `φ(g)·dg` with `φ(g) = gelu(g)/g`. The value norm sits inside attention and keeps its true
  derivative.
- **Tokens and output.** Embeddings enter the residual times √3840 and are tied to the unembedding;
  S-lens reads use them so scaled. Final logits pass a softcap, `30 · tanh(x / 30)`, which the
  catalog's R-lens probabilities and the causal checks include; `09_legible.py`'s z-scores use the
  logits before it (the cap keeps each token's order). Every document starts with `<bos>`, the
  attention sink, so activation statistics leave out position 0 and the `first_*` columns are at
  position 1. The kept tokens are 123,774 of 262,144 (the same corpora, 92.6M Gemma tokens).
- **Lenses.** Fitted as Qwen3-8B's, on the same 1,000 DCLM documents, at about 16 s a document per lens
  on one H100. Split-half differences are 0.01-0.12 by layer for the R-lens (median about 0.05,
  largest early) and 0.01-0.07 for the S-lens. The released files carry no provenance records.
- **Corpora.** The activation slices are Qwen3-8B's rows at the same revisions, except that the chat
  slice is rendered as plain `User:` / `Assistant:` turns (the base model has no chat template), in
  its own file, `act-chat-gemma-4-12b.jsonl`; `first_chat` is on the first token of `User:`.
- **The causal sample** (`checks/causal_lens.py`) is 24 random neurons a layer, both poles, with the
  write added after the post-feedforward norm at its scale.
- **The run.** `google/gemma-4-12B` at revision `023679ed`, transformers 5.22.0 and PyTorch 2.13.0
  (CUDA 12.9), on two H100s, 2026-10-30.

### Images and sound

Gemma 4 has no vision or audio encoder network. Each embedder is linear up to its norms: an image's
48 × 48-pixel patches pass `LN → Dense → LN → +position → LN → RMSNorm → Linear`, a sound's 40 ms
frames of 640 samples `RMSNorm → Linear`. Under the LN rule both are linear maps, so a covector on
the residual maps back exactly to pixels or samples. The patch embedder has rank 2,041; frames reach
a 640-dimensional subspace.

- **Documents** (`neuron_lens/media.py`). An image is 384 × 384 pixels, 64 slots of 48 × 48 patches
  (an 8 × 8 grid); a sound is the middle 2.56 s of a clip at 16 kHz, 64 slots of 40 ms. Each sits in
  a short text prompt. In sliding layers an image's slots attend to each other both ways, as the model
  does for one image. Clips with a digitally silent frame are skipped: it embeds to zero, where every
  norm's derivative is `1/√ε`.
- **Media lenses** (`02_fit.py --modality image|audio`). The R-lens seeds every position from the
  first slot on and averages its sources over the slots; the S-lens seeds only the subspace the
  embedder reaches (`basis`: 2,048 directions for images, 640 for audio), which is exact for media
  inputs and costs 0.1-0.3 of a text document, and averages its targets over the slots. Images: COCO
  2014 validation (1,000 for the S-lens, 512 for the R-lens); sounds: ESC-50 (1,615 kept clips, 512).
  RelP's rules beat the true derivative on held-out images and sounds too (`checks/rule_diag.py`). The
  R-lens can't be seeded in a subspace the same way: writes are read along the unembedding's
  low-energy directions too (`checks/r_rank_check.py`). Before the fits, `checks/gemma4_smoke.py`
  checked the layer maps against autograd on the real model (relative error 1e-6) and
  `checks/media_fit_check.py` a fitted media S-lens against reads taken without fitting it.
- **Exemplars** (`13_media_activations.py`, collections in `neuron_lens/media_sets.py`). 49,694
  images and 46,932 sounds streamed from 49 and 11 labelled Hub datasets with seed-0 shuffles, plus
  synthetic probes with exact labels, each labelled `source:label`. Images are centre-cropped,
  documents and drawings letterboxed. Per pole: its mean activation over each document's slots; each
  label's z over the documents; the top label is *reliable* if it passes the 99th percentile of a
  shuffled-label null for its layer and sits in the top 5 of both halves of the documents. A held-out
  AUROC scores the label chosen on even documents on the odd ones. Each pole keeps its 16 top
  documents. `data/corpus/media-{image,audio}.provenance.json` lists every source with its split,
  revision, sample size and license; `media/<modality>_docs.parquet` has each document's row and the
  SHA-1 of its bytes.
- **Reads** (`14_media_tables.py`). A pole's read direction through the media S-lens `mid`, mapped back
  through the embedder: `read_png` (a 48 × 48 patch, ±3 standard deviations around grey) or
  `read_wave` (640 samples). Its write through the media R-lens: `write_R`, what it makes the model
  say in image or audio contexts. A read is one slot pattern summed over the slots, not a map of where.
- **Optimised reads** (`15_media_optimised.py`). For the 48 poles a layer (L0-L35) with the strongest
  reliable label: one pattern in all 64 slots, started from the read and optimised by gradient ascent
  through the model (24 Adam steps over 128 images or clips; at most 8% of the pixel range, or 0.3 of
  each frame's RMS), scored by its gain on 8 held-out documents.
- **Cross-modal ranks** (`checks/crossmodal.py`) and **heads in media contexts**
  (`checks/head_media.py`): findings §15 and §16 give their setups.

## Related work

Lenses:
- Logit lens (nostalgebraist 2020,
  [LessWrong](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)) and
  tuned lens (Belrose et al. 2023, [arXiv:2303.08112](https://arxiv.org/abs/2303.08112)).
- J-lens (Gurnee et al. 2026,
  [transformer-circuits.pub](https://transformer-circuits.pub/2026/workspace/)): the averaged Jacobian
  to the output, read with `W_U`; defines the J-space. Neuronpedia publishes one for Qwen3-8B
  (`neuronpedia/jacobian-lens`, 461 wikitext prompts).
- R-lens (camilablank, agam_bhatia and Neel Nanda 2026,
  [LessWrong](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers)):
  the J-lens under LRP-style rules, for more faithful early layers.
- K-lens (Theia Vogel 2026, [shared on X](https://x.com/voooooogel/status/2084858861551526002),
  unpublished): the J-lens's Jacobian taken to the input embeddings, the idea the S-lens builds on; the
  S-lens adds RelP's rules.
- RelP (Rezaei Jafari et al. 2025, [arXiv:2508.21258](https://arxiv.org/abs/2508.21258)): the rules used
  here. J++ (Ayonrinde 2026,
  [LessWrong](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace))
  filters noisy gradients; ω-J-lenses (Ostrow et al. 2026, [arXiv:2610.31955](https://arxiv.org/abs/2610.31955))
  give a theory of what the J-lens finds.
- [Drowse](https://github.com/a9lim/drowse): the R-lens implementation this estimator is tested
  against and exports to.

Neuron descriptions:
- Bills et al. 2023
  ([OpenAI](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)): every
  GPT-2 XL neuron, from activations.
- Choi et al. 2024 ([Transluce](https://transluce.org/neuron-descriptions)): every
  Llama-3.1-8B-Instruct neuron, both polarities.
- Gur-Arieh et al. 2025 ([arXiv:2501.08319](https://arxiv.org/abs/2501.08319)): output-centric
  descriptions, combined with input-centric ones.
- Paulo et al. 2024 ([arXiv:2410.13928](https://arxiv.org/abs/2410.13928)): detection scoring.

Unit classes measured here: prediction and suppression neurons (Gurnee et al. 2024,
[arXiv:2401.12181](https://arxiv.org/abs/2401.12181)), entropy neurons (Stolfo et al. 2024,
[arXiv:2406.16254](https://arxiv.org/abs/2406.16254)), the OV copying score (Elhage et al. 2021,
[transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html)) and copy
suppression (McDougall et al. 2023, [arXiv:2310.04625](https://arxiv.org/abs/2310.04625)).
