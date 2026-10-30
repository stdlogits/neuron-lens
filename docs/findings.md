# Findings

What the catalog shows about the neurons and attention heads of Qwen3-8B (§1-§12) and Gemma 4 12B
(§13-§16, in text, images and sound), and how far each way of reading them goes. Each section gives
its setup, its results against controls, and its limits; the methods are in [design.md](design.md)
and the overview in the [README](../README.md).

Numbers are medians or means over a layer's poles unless stated. Each setup names its script: the
pipeline stages (`scripts/NN_*.py`), a check (`scripts/checks/`), or, for analyses the pipeline later
absorbed, a script since retired (marked so). Compact outputs are in `data/summary/`, and figures are drawn
by `scripts/11_figures.py`.

**Terms.** A *pole* is one sign of a neuron's activation (SwiGLU activations take both), written
`L8 n344 +`. Its *write* is `±W_down[:, n]`, its *read direction* `â ± b̂` (the gate and up rows).
Its *top firings* are its strongest position in each of its 24 top documents. The *bands* are L0-L8,
L9-L12, L13-L20, L21-L27 and L28-L35 for Qwen3-8B, and Gemma 4 12B's quarters L0-L11, L12-L23,
L24-L35 and L36-L47. A *mismatched control* scores another unit of the same layer on this unit's
tokens. *Kept tokens* are the tokens seen 5+ times in the activation corpora, without byte-fallback
pieces (103,830 for Qwen3-8B, 123,774 for Gemma 4); every token read covers them.

| § | in one line |
|---|---|
| [§1](#1-the-lenses-stability-and-corpus-dependence) | both lenses' halves agree within 0.06; a chat-fitted lens changes tokens, not concepts or causal scores |
| [§2](#2-activations-over-15m-positions) | scale, near-silent L1, first-position L2-L3, language and code selectivity at the ends of the network |
| [§3](#3-a-writes-lens-read-predicts-what-steering-it-does) | the R-lens read of a write predicts steering (r 0.46-0.51 at L4-L8, 0.9+ from L22), ahead of the J-lens |
| [§4](#4-steering-is-not-what-a-firing-does) | before L21 a firing's effect depends on its context; the lens and steering read its average |
| [§5](#5-prediction-and-suppression) | late poles fire before the word they promote; suppression from L32; two signs of one neuron |
| [§6](#6-the-input-side) | the S-lens names early neurons' tokens through L8, the R-lens read from L9; exact drivers in between |
| [§7](#7-which-reads-hold-the-legibility-census) | 38% of poles are read by some test, 57% are dark (82% at L13-L20) |
| [§8](#8-describing-neurons) | labels from the lenses alone: firing AUROC 0.80, writes picked 65%; Haiku labels every pole |
| [§9](#9-attention-sinks-the-outputs-temperature-and-entropy-neurons) | two null-space dimensions carry the attention sinks and set the output's temperature |
| [§10](#10-outside-labels) | steering-labelled personas against the catalog's reads, from both sides |
| [§11](#11-attention-heads-circuits-read-token-by-token) | a head's input reads with the S-lens; what each head copies and attends to, from the weights |
| [§12](#12-attention-heads-census-and-labels) | late heads write the next token, the middle is dark; a label for every head |
| [§13](#13-gemma-4-12b-the-text-catalog-in-a-second-model) | Gemma 4 12B reads like Qwen3-8B: inputs early, writes late (steering r 0.97 at L36-L47), a dark middle |
| [§14](#14-images-and-sound-what-drives-a-neuron) | labelled images and sounds name what Gemma's neurons respond to; lens reads are projections, optimised ones drive the neuron |
| [§15](#15-one-concept-in-text-images-and-sound) | 4,378 poles keep their image label on every text test; 18 hold one concept across all three modalities |
| [§16](#16-attention-heads-in-image-and-audio-contexts) | single heads carry what an image shows into the text; sound is too weak to trace |

## §1. The lenses: stability and corpus dependence

**Setup.** Both lenses are fitted by `02_fit.py` on the first 1,000 rows of DCLM-shuffled, 128 tokens
each, in bf16 with fp32 sums on one H100 (about 11.5 s a document per lens). The documents alternate
between an even and an odd half; the split-half difference is `‖even − odd‖ / ‖mean‖` (Frobenius) per
layer. Neuron-by-neuron agreement is from `checks/lens_compare.py` over 12 layers; the J-lens is
Neuronpedia's (461 wikitext prompts).

**The halves agree.** The R-lens's split-half difference is 0.04-0.06 at L0-L18, 0.03 at L19-L22, 0.02
at L23-L27 and 0.01 or less from L28, as expected from two documents' estimates differing by 65%
(`0.65/√500` ≈ 0.03). Its `mid` target is within 0.01 of that; the S-lens is 0.00-0.05 (`mid`) and
0.01-0.05 (`out`), highest at L16-L20. Neuron by neuron, the halves' write reads share the top token
for 89-99% of neurons:

| layer | halves: top-1 agree | halves: top-10 overlap | R-lens vs J-lens: top-1 agree |
|---|---|---|---|
| L2 | 0.89 | 0.92 | 0.02 |
| L8 | 0.89 | 0.91 | 0.05 |
| L14 | 0.91 | 0.93 | 0.06 |
| L20 | 0.94 | 0.95 | 0.20 |
| L26 | 0.97 | 0.98 | 0.47 |
| L32 | 0.99 | 0.99 | 0.73 |

![Split-half difference of each lens by layer](figures/04_lens_reliability.png)

- **The R-lens and J-lens disagree most where the R-lens predicts steering better** (L4-L23, §3).
- **RelP's rules lower the noise.** On a Qwen3-0.6B prototype of the S-lens, fitted per offset on 12
  documents, RelP halved the split-half noise of the true-gradient version (L7: 0.15 against 0.26)
  and fit exact token footprints slightly better (R² +0.02-0.03 from L1). On 8B, RelP beats the true
  derivative on both sides (§3, §6).
- **Early writes share one direction.** At L2 the R-lens's top singular direction holds 0.78 of
  `J_L`'s energy and 0.95 of a median write (0.27 at L5, 0.02 or less from L23). It reads `the, a, in,
  and` one way and code tokens (`startIndex`) the other. The halves agree there (0.89), so this is the
  early Jacobian, not noise; early neurons are read from their inputs instead (§6).
- **Raw reads have hub tokens**, the same few tokens at the top of many poles' reads: at L1, ` the`,
  ` in` and ` and` sit in 30-31% of write reads' top 10s. Layer-contrastive reads (`_z`), which
  z-score each token across the layer's poles, remove them.

**The corpus changes tokens, not concepts.** An R-lens fitted the same way on 500 UltraChat
conversations in the chat template (split-half 0.05-0.07 early, 0.01 late) was compared with the
DCLM lens's 500-document halves, so both sides carry the same noise:

| layer | top-1 agree: chat vs DCLM (DCLM halves) | top-10 overlap |
|---|---|---|
| L0 | 0.40 (0.90) | 0.55 (0.93) |
| L2 | 0.31 (0.89) | 0.54 (0.92) |
| L8 | 0.42 (0.89) | 0.54 (0.91) |
| L16 | 0.51 (0.92) | 0.62 (0.94) |
| L24 | 0.78 (0.96) | 0.84 (0.97) |
| L34 | 0.96 (1.00) | 0.96 (1.00) |

- **Every causal score is the same.** Steering prediction (§3) is within 0.04 at every layer, ablation
  identical, persona names (§10) equal from L6 (the DCLM lens is better at L0-L5: 15% of names in the
  top 10 against 9%), fused personas within 0.02. The chat lens is no better on chat-format tests.
- **Clear neurons read the same concept in another order** (L8 n1931 +: 圣经, Bible, 基督徒,
  Scripture under both). **Weak, diffuse writes drift toward each corpus's register:** markdown bold
  and `...⏎` under chat, `...⏎⏎` and curly quotes under DCLM. Where a read's top probability is low,
  register tokens are the lens's, not the neuron's.
- The DCLM lens is the lens of record.

![A chat-fitted lens against the DCLM halves](figures/05_chat_lens.png)

**Limits.** One fitting corpus of record and 128-token documents. Split-half agreement measures
stability within a corpus, not freedom from it; the chat lens is one alternative.

## §2. Activations over 15M positions

**Setup.** `06_activations.py`, one bf16 pass with eager attention on an RTX 4090 (68 minutes):
every document of four slices, up to 256 tokens, from position 1, 15.4M positions in all.

| slice | source | documents | positions |
|---|---|---|---|
| dclm | DCLM-shuffled from row 1,000 (none in the lens corpus) | 40,000 | 9.46M |
| zh | FineWeb-2 `cmn_Hani` | 8,000 | 1.89M |
| code | codeparrot-clean Python | 7,990 | 1.98M |
| chat | UltraChat-200k `test_sft`, in the chat template | 8,000 | 2.04M |

Position 0 attends only to itself, so every kept token run alone gives each pole's exact value
there. `act_top` is the median of a pole's 24 top firings; `rate` is the share of positions past
`act_top / 4`, from exact counts. Checked on Qwen3-0.6B against an unpadded fp32 brute force: counts
exact, sums within 0.2%, top values within 0.9%.

| layer | `act_top` | quiet (`act_top` < 0.05) | fires harder at position 0 than anywhere (> 2×) | zh-selective | code-selective |
|---|---|---|---|---|---|
| L0 | 0.86 | 0.00 | 0.29 (0.02) | 0.031 | 0.020 |
| L1 | 0.006 | 0.56 | 0.22 (0.17) | 0.028 | 0.007 |
| L2 | 0.011 | 0.61 | 0.70 (0.62) | 0.017 | 0.009 |
| L3 | 0.70 | 0.42 | 0.57 (0.51) | 0.024 | 0.021 |
| L4 | 2.94 | 0.09 | 0.12 (0.06) | 0.046 | 0.040 |
| L8 | 4.09 | 0 | 0.001 | 0.006 | 0.025 |
| L16 | 3.22 | 0 | 0.000 | 0.002 | 0.030 |
| L24 | 11.4 | 0 | 0.000 | 0.021 | 0.047 |
| L28 | 22.5 | 0 | 0.000 | 0.018 | 0.057 |
| L35 | 50.8 | 0 | 0.020 | 0.050 | 0.020 |

Selective: the pole's rate in that slice is at least 10× every other slice's, and above 1e-3.

![Activation scale and quiet poles by layer](figures/06_activations.png)

- **Scale is flat at 3-5 from L5 to L21, then grows tenfold to L35**, with the residual stream's norm.
- **L1 is mostly near-silent:** 56% of its poles stay under 0.05 everywhere, position 0 included,
  like the dead early neurons Voita et al. (2023) found in OPT (there zero under ReLU; here near zero).
- **L2 and L3 work at position 0.** 62% and 51% of their poles fire there more than twice as hard as
  at any later position (L2: median 3.7 there against `act_top` 0.011). They do not start the
  attention sink (zeroing their output there leaves it, §9), and matter little downstream (the loss
  rises 0.5 nats on the token predicted at position 0, 0.01 from the third). What they compute is open.
- **Language and code features sit at the ends of the network.** Chinese-selective poles are 1.6-5% of
  L0-L4 and L24-L35, and 0.2-0.6% of L6-L20, the U shape reported for language-specific neurons
  (Tang et al. 2024). Code-selective poles peak at 3.6-5.7% of L20-L29. L29 n8134 + fires at
  76% of Chinese positions and under 0.05% elsewhere: a language identity, not a word.
- **Chat-selective poles read as ChatGPT's register**, since UltraChat was generated by ChatGPT on
  both sides: L3 n6404 − fires on " delve" and " adhere", in user turns too; L28 n11134 + peaks on the
  year in "As of 20⟦2⟧1", the knowledge-cutoff hedge.
- **Rare poles are counted exactly.** An earlier 8,192-position sample put `rate` at 0 for 18.6% of
  poles; exact counts leave 0.65%, all silent. L8 n344 + (owls) fires at 1.8 per 100,000 positions.
- **`act_top` is a tail statistic** and grows with the corpus (1.4-1.8× from a 7,000-document pilot):
  compare poles within one corpus.

**Limits.** 256 tokens per document under-samples long-range features, and top firings are one per
document. Only the DCLM slice is a uniform sample; the zh slice is largely traditional script, code is
Python only, and chat is one synthetic dataset.

## §3. A write's lens read predicts what steering it does

**Setup** (`checks/causal_lens.py`, MPS, bf16). 32 random neurons per layer plus the named ones
(`neuron_lens/known.py`), 1,194 neurons, both poles. **Steering** adds `±α·W_down[:, n]` to layer
`L`'s MLP output at positions 16-63 of 16 DCLM documents of 64 tokens, at 1× and 4× the pole's typical
top activation from a pilot (about 0.6× the full pass's `act_top`), and records the mean change in
log-probabilities. **Score:** Pearson r over the tokens seen 5+ times between that change and a lens's
logits for the write. **Control:** another sampled neuron's read from the same layer. Medians over the
random picks (64 poles a layer).

| layer | logit lens | J-lens | R-lens | control |
|---|---|---|---|---|
| L0 | 0.09 | 0.38 | 0.35 | 0.01 |
| L4 | 0.09 | 0.38 | 0.46 | −0.01 |
| L8 | 0.12 | 0.42 | 0.51 | 0.01 |
| L12 | 0.25 | 0.58 | 0.64 | −0.02 |
| L16 | 0.36 | 0.65 | 0.74 | 0.00 |
| L20 | 0.59 | 0.81 | 0.87 | 0.01 |
| L24 | 0.83 | 0.92 | 0.94 | 0.00 |
| L28 | 0.90 | 0.96 | 0.98 | −0.01 |
| L32 | 0.94 | 0.99 | 0.99 | −0.01 |

![Lens reads against steering and ablation](figures/07_steering.png)

- **A read of one write vector, with no forward pass, predicts what adding it does downstream**. The
  R-lens reaches 0.46-0.51 at L4-L8, 0.64-0.74 at L12-L16 and 0.9 or more from L22; controls stay
  within ±0.04.
- **The R-lens leads the J-lens from L4 to L23** by 0.03-0.12 (L7: 0.51 against 0.39), and in top-10
  overlap. The J-lens leads at L0, they tie at L1-L3 and converge from L24. The logit lens, which
  ignores the later layers, trails until about L20.
- **Nearly every pole has a specific effect when amplified.** From L5, 92-100% of random poles' writes
  predict their own steering effect (r past 0.26, the 99th percentile of mismatched pairs), against
  0-5% of mismatched pairs. L1-L3 are lower (28-55% pass; median r about 0.16 at L1-L2) because their
  poles barely fire, so their doses are tiny.
- **Named concept poles show the R-lens's largest leads:** L8 n5315 − ("nature") at r 0.64 against
  0.10 for the J-lens; steering it raises ` trees`, ` vegetation`, ` gardens` and 植物, while the
  J-lens reads its write as brackets. Named poles are otherwise no more readable than random ones.
- **Ablation where a pole fires** (zeroing the neuron at its peak in its top 8 documents) agrees far
  less before L21 (r under 0.15 to L16, 0.87 at L32), and the lenses tie there. §4 explains why.

**Limits.** Doses come from a pilot; 16 held-out DCLM documents of 64 tokens, the lens corpus's source
(the chat lens ties, §1). Pearson r over 100k tokens is dominated by small changes; the top-10 overlap
complements it. bf16 rounds weak poles' writes partly away, so the ablation numbers are lower bounds
(§4).

## §4. Steering is not what a firing does

Steering adds a pole's write at every position of neutral text. What does the pole do when it fires
in real text?

**In context, writes rarely predict what comes next before L22.** For every pole, the census (§7)
tests whether its write read promotes the tokens that follow its top firings. Before L22, 1-4% of
poles do (net of chance); from L23 it rises to 31-32% at L28-L29, and is 17-25% at L32-L35, where
suppression takes over (§5). Amplified, 92-100% do from L5 (§3).

![Amplified against in context](figures/08_amplified_vs_in_context.png)

**Per-context linearization.** For §3's random poles that fire (2,290), at each of the 24 top
firings: the effect of removing the firing (the neuron set to 0 there), and its exact first-order
effect `a·∂logp/∂a` (central difference), each as a change in log-probabilities at the firing
position and summed over the next 8. Correlations are Pearson r over the vocabulary; one H100, fp32
with TF32 matmuls (`context_linearity.py`, retired).

| band | removed vs first-order | one context vs the pole's other 23 | lens vs one context | lens vs the mean of 24 | steering vs the mean firing effect | steering vs lens |
|---|---|---|---|---|---|---|
| L0-L8 | 0.92 | 0.05 | 0.03 | 0.11 | 0.08 | 0.40 |
| L9-L12 | 0.97 | 0.08 | 0.05 | 0.16 | 0.16 | 0.59 |
| L13-L20 | 0.98 | 0.11 | 0.09 | 0.28 | 0.28 | 0.76 |
| L21-L27 | 0.99 | 0.36 | 0.27 | 0.57 | 0.57 | 0.94 |
| L28-L35 | 1.00 | 0.78 | 0.66 | 0.86 | 0.87 | 0.99 |

Every control (another pole's values) is within 0.01 of zero.

![Per-context linearization](figures/09_context_linearity.png)

- **A firing's effect is close to linear:** removing it does what its first-order effect says
  (0.92-1.00; L1-L2 are noise, since their poles barely fire).
- **What changes is the context.** Before L21, the same pole firing equally hard pushes the logits
  different ways in different contexts: one context's effect correlates with the mean of its other
  23 at 0.05-0.11, and 12 contexts against 12 others reach only 0.16-0.27. From L28 the effect is
  nearly the same everywhere (0.78; 0.94 split-half).
- **The lens reads the shared part.** Against one context, the weight-only read gets 60-85% of the
  correlation the pole's own other firings get, in every band.
- **Lens and steering agree with each other, not with firings.** Before L21 the lens predicts steering
  at 0.40-0.76, while steering predicts the pole's mean firing effect at 0.08-0.28. From L28 all agree.
- **Precision matters.** In bf16 a weak pole's write partly rounds away in the residual: on five test
  neurons, removed against first-order was r 0.05-0.77 in bf16 and 0.83-0.99 in fp32.

**What this means.** The lens and steering agree because both average over contexts. Before L21, what a
pole does when it fires is mostly specific to the context, which no single vector, from a lens or from
steering, can carry; the pole's own other firings don't predict it much better, so this is a fact
about the model, not the lens. A write read is the pole's average push, which is what steering shows;
early and mid-network poles are better described by what makes them fire (§6).

**Limits.** 32 random neurons a layer; a 9-position window misses later effects; Pearson r weighs
all tokens alike; TF32 makes weak early poles' values lower bounds.

## §5. Prediction and suppression

**Setup** (`side_agreement.py`, retired; no model). For each pole's top firings, the token after
the peak. **Predict:** the share of firings whose next token is in the write read's top 10.
**Suppress:** the same for its bottom 10 (the opposite pole's top 10). Control: a random pole of the
same layer and sign.

| layer | predict (control) | poles past ⅓ | suppress | poles past ⅓ |
|---|---|---|---|---|
| L8 | 0.007 (0.004) | 0.00 | 0.003 | 0.00 |
| L16 | 0.007 (0.005) | 0.00 | 0.005 | 0.00 |
| L22 | 0.014 (0.001) | 0.01 | 0.003 | 0.00 |
| L26 | 0.035 (0.000) | 0.03 | 0.003 | 0.00 |
| L28 | 0.068 (0.000) | 0.07 | 0.003 | 0.00 |
| L30 | 0.073 (0.000) | 0.07 | 0.005 | 0.00 |
| L32 | 0.058 (0.000) | 0.06 | 0.009 | 0.01 |
| L34 | 0.045 (0.000) | 0.04 | 0.034 | 0.04 |
| L35 | 0.055 (0.001) | 0.05 | 0.057 | 0.07 |

![Writes predict the next token late; reads name the firing token early](figures/10_side_agreement.png)

- **Late poles fire on the token before a word their write promotes:** "is⟦ once⟧" → *again* (L30
  n1034 +), "⟦arist⟧" → *-ocracy* (L28 n8868 −). Prediction rises from L20 and peaks at L28-L31, where
  7-8% of poles have more than a third of their top firings followed by a token they promote (control
  under 0.1%).
- **Suppression starts at L32 and overtakes prediction at L35.** This is the pattern Gurnee et al.
  (2024) found in GPT-2 from weight statistics, here measured from activations.
- **Their weight classifier picks the same poles.** Among L24-L35 poles in the top 10% of kurtosis of
  their logit-lens logits, positive skew ("prediction") gives 24% of firings before a promoted token
  (other poles 4%), and negative skew 12% before a bottom-10 token (0.6%).
- **Two signs of one neuron.** Skew flips sign between a neuron's poles, so each high-kurtosis neuron
  has a promoting pole P and a suppressing pole S:

  | layer | P predicts (> ⅓) | S suppresses (> ⅓) | `act_top` P / S |
  |---|---|---|---|
  | L28 | 0.38 | 0.01 | 40 / 16 |
  | L30 | 0.47 | 0.03 | 61 / 23 |
  | L32 | 0.36 | 0.05 | 62 / 31 |
  | L34 | 0.10 | 0.37 | 49 / 54 |
  | L35 | 0.23 | 0.51 | 45 / 68 |

  Through L32 the promoting pole fires harder; from L34 the suppressing one. In SwiGLU, prediction and
  suppression neurons are not two kinds of weights but two signs of one neuron, and a weight-only
  classifier needs the activation statistics to say which a neuron is.

**Limits.** An exact token in a top 10 is strict (" par" against "airs" after "P" is a miss), so these
shares are lower bounds. A bottom-10 token is a directional association, not a measured suppression
after later layers.

## §6. The input side

Early neurons are best read from what drives them. This section asks which weight-only read names
those tokens, at which depth.

**Setup.**
- **The reads** (`08_reads.py`) of a pole's read direction `c = â ± b̂`: `read_S`, the S-lens to the
  MLP input, scores token `v` by `c · K_L (W_E[v] − mean(W_E))` (which input tokens drive the pole);
  `read_R`, the R-lens from the MLP input (what the residual says when it drives the pole); each also
  layer-contrastive (`_z`). Gate and up rows are nearly orthogonal (median cosine 0.00 ± 0.005 at
  every layer, except L2), so each pole has a distinct direction.
- **Exact drivers** (`07_drivers.py`): every kept token at position 24 of 8 real documents (4 web, 2
  code, 2 chat), followed by the documents' own next 16 tokens; per pole, the 64 tokens with the
  largest activation at the token, and summed over the 16 after. An fp32 rerun through L9 shares
  98-99% of a pole's top 20 on average at the token (95-99% summed; `drivers_precision.py`,
  retired); later layers are unchecked.

**Early neurons are token-family detectors.** Exact drivers of random early poles:

| pole | exact drivers |
|---|---|
| L0 n1208 − | 阔, 开阔, 开拓, 广阔, 宽阔 ("wide", "open up") |
| L0 n11206 − | 出生, 出行, 出入, 出门, 出土 (words starting with 出) |
| L1 n7735 + | 少年, 少女, 少量, 少许, 少数 (words with 少) |
| L1 n11672 − | 德国, german, Germany, German, Germans |
| L2 n6890 + | Whitney, Courtney, Chelsea, Kendall, Shelby, Lindsay |

**The S-lens names them from the weights, through L8.** Share of each read's top 10 among the pole's
20 exact drivers (control 0.003 or less):

| layer | `read_S` | `read_S_z` | `read_R` |
|---|---|---|---|
| L0 | 0.31 | 0.34 | 0.011 |
| L1 | 0.10 | 0.10 | 0.000 |
| L4 | 0.10 | 0.14 | 0.003 |
| L7 | 0.03 | 0.02 | 0.017 |
| L8 | 0.06 | 0.07 | 0.069 |
| L9 | 0.02 | 0.03 | 0.057 |
| L12 | 0.004 | 0.004 | 0.053 |
| L16 | 0.002 | 0.002 | 0.030 |
| L20 | 0.010 | 0.013 | 0.037 |
| L24 | 0.027 | 0.049 | 0.076 |
| L29 | 0.007 | 0.026 | 0.080 |
| L35 | 0.001 | 0.008 | 0.027 |

![Which reads name the exact drivers](figures/11_drivers.png)

- **L0-L4:** `read_S` names exact drivers (31% of its top 10 at L0, 4-11% at L1-L4), and at L0 the
  peak token of 40% of top firings (another pole's read: 0.1%); the R-lens read of the same direction
  almost none. At L1-L8 the S-lens read names 2-9% of peak tokens, against under 1.5% for the R-lens
  read (3.5% at L8).
- **L8 is the crossover:** both reads name drivers (6-7%), and a pole's S-lens, R-lens and write reads
  share the most tokens (0.26-0.32 tokens of their top 10s on average; control 0.002 or less).
- **From L9 the R-lens read names more:** 5-6% at L9-L12, 6-8% at L22-L30. On the input side too, the
  R-lens takes over where the S-lens stops.
- **L13-L20 is the weakest stretch** for every weight-only input read (2-4% for the R-lens). There the
  exact drivers and activations carry the input side.
- **Past L8, raw S-lens reads are swamped by hub tokens** (rare CJK, spam-cluster and code pieces
  shared by most poles). `read_S_z` recovers concepts (L10 n9952 +: glaciers, terrain, 公里 "km",
  Mountains; L30 n2310 +, a "green" neuron: 繁殖 "breeding", Lawn, rooting) but gives up the literal
  peak token, so both columns are kept.
- **Read and write often name one concept** from the middle on: L8 n1931 + reads 圣经 (Bible), 基督教,
  信仰 (faith) and writes 圣经, 基督徒, Bible; L20 n2281 − reads else, 否则 (otherwise) and writes else,
  otherwise; L32 n2407 + reads 斗争 (struggle), fight and writes fight, fighting, 战斗.

**Early neurons write back what they detect.** `write_S` reads a write through the S-lens to the
layer's output: the tokens whose footprint the write imitates. It shares 1.66 tokens of 10 with the
pole's exact drivers at L0 (control 0.003), 0.22-0.49 at L1-L4, 0.10-0.25 at L5-L8 and about 0.01
from L11. L0 n1649 −, which fires on `ain`, writes `ain | AIN | ains | rain | strain`, where its R-lens
write read is `... | ..." | the`. MLP0-MLP8 in effect act as an extended embedding.

**Why the S-lens read fades: a token's effect is mostly nonlinear past L0** (`checks/s_linearity.py`;
96 tokens swapped in at position 16 of 4 documents, MPS, fp32). R² of each predictor against the
token's exact footprint on the MLP input, at its own position and summed over later ones:

| layer | own position: exact linearization in context (true / RelP) | S-lens | embedding | summed: RelP linearization | S-lens |
|---|---|---|---|---|---|
| L0 | 0.86 / 0.86 | 0.63 | 0.51 | 0.56 | 0.50 |
| L1 | 0.30 / 0.33 | 0.20 | 0.08 | 0.30 | 0.28 |
| L5 | 0.12 / 0.15 | 0.09 | 0.03 | 0.07 | 0.06 |
| L15 | 0.07 / 0.17 | 0.10 | 0.01 | 0.02 | 0.02 |

- Even the exact linearization in the token's own context explains 86% of its footprint at L0 but 30%
  at L1 and 7% at L15: swapping a whole token is far outside the regime of local linearizations past
  L0's MLP (this bounds local linearizations, not every possible linear map).
- On the summed footprint, the quantity it estimates, the S-lens does about as well as that
  per-context linearization: averaging over contexts and summing over positions cost little.
- RelP beats the true derivative from L1 (0.17 against 0.07 at L15).
- **But nonlinearity does not explain the fall after L8.** The linear share is flat from L4 to L17 (R²
  about 0.10), while the S-lens read's driver hits peak at L8 and fall by L11. The cause is open
  (candidates: read-write alignment, the norm weight's selectivity, how much context L8 neurons use).

**Limits.** Drivers are measured at one position of 8 contexts. The S-lens footprint averages over
contexts, so it reads a direction on average: on the Qwen3-0.6B prototype a pole's exact top 20
tokens of 6,000 were 70% right at L0, 25% at L1 and about 0% from L5.

## §7. Which reads hold: the legibility census

Do the catalog's reads say something true about each pole? `09_legible.py` tests every pole's
weight-only reads at its own top firings.

**Setup.** For each of the 884,736 poles and its 24 top firings:
- `write_R`: the mean layer-contrastive z, under the write read, of the tokens after the firings;
  `write_S`: of the peak tokens, under the write's S-lens read; `read_R`, `read_S`: of the peak tokens,
  under the read direction's reads;
- p-values against mismatched pairs (another pole of the layer and sign, on this pole's tokens);
  Benjamini-Hochberg over the layer's poles at a 5% false discovery rate; *net of chance* is the share
  past the mismatched pairs' 99th percentile minus their own share;
- token and next-token detectors: a third of the firings share one peak token, or one next token;
- silent: no firing, or `act_top` under 0.05; dark: fires, but no test passes and no detector.

| band | silent | token detector | `write_R` | `write_S` | `read_R` | `read_S` | any weight read | dark |
|---|---|---|---|---|---|---|---|---|
| L0-L8 | 0.19 | 0.38 | 0.00 [0.02] | 0.11 [0.11] | 0.02 [0.05] | 0.35 [0.30] | 0.36 | 0.29 |
| L9-L12 | 0.00 | 0.19 | 0.01 [0.03] | 0.01 [0.03] | 0.08 [0.10] | 0.03 [0.08] | 0.11 | 0.72 |
| L13-L20 | 0.00 | 0.12 | 0.00 [0.01] | 0.00 [0.01] | 0.04 [0.07] | 0.01 [0.01] | 0.05 | 0.82 |
| L21-L27 | 0.00 | 0.28 | 0.12 [0.13] | 0.02 [0.03] | 0.06 [0.09] | 0.05 [0.06] | 0.19 | 0.59 |
| L28-L35 | 0.00 | 0.25 | 0.29 [0.25] | 0.00 [0.01] | 0.03 [0.06] | 0.01 [0.02] | 0.31 | 0.52 |
| all | 0.05 | 0.25 | 0.09 [0.09] | 0.03 [0.04] | 0.04 [0.07] | 0.10 [0.10] | 0.22 | 0.57 |

Shares of poles passing at 5% FDR, band means over layers; net of chance in brackets.

![Which reads hold for each pole](figures/12_census.png)

- **38% of poles are read by some test** (a weight-only read or a detector), 5% are silent (the
  census skips position 0, where many L1-L3 poles work, §2), and **57% are dark**.
- **The input side reads early:** `read_S` passes for 78% of poles at L0 and 35-58% at L1, L3 and L4
  (L2: 18%, since 61% of its poles are silent), and almost none at L10-L20. `read_R` passes most at L8
  and L12 (11-13%).
- **The write side reads late:** `write_R` from L23, most at L28-L29 (31-32% net). `write_S` passes
  for 11% of L0-L8's poles, nearly always together with `read_S`: early neurons write back what they
  detect (§6).
- **Dark poles peak mid-network:** 82% at L13-L20 (86% at L18), against 29% at L0-L8 and 45-71%
  from L21.
- **Dark is not indescribable.** In the description pilot (§8), explainers with full evidence cards
  described dark poles almost as well as legible ones (detection AUROC 0.90 over 111 dark poles,
  against 0.96 over 55 with a legible weight read): they are concept detectors whose firings vary in
  token, which token-level tests miss.

**Which reads to trust where,** from §3-§7. The bands are priors; each pole's q-values in
`legible` say which of its reads hold.

| layers | what drives a pole | what a pole writes |
|---|---|---|
| L0-L7 | `read_S`: literal tokens (raw at L0-L4) | `write_S`: what it detects, written back; `write_R` legible from about L4 |
| L8 | both lenses, agreeing most | both |
| L9-L12 | `read_R`, with `read_S_z` as a concept view | `write_R`: predicts steering, rarely the next token |
| L13-L20 | only exact drivers and activations | `write_R`, for steering |
| L21-L35 | `read_R`, `read_S_z` | `write_R`: the next token from about L23; suppression from L32 |

![Which reads work at which depth](figures/03_depth_map.png)

**Signal holds to rank 64** (`rank_depth.py`, retired; nine layers). Per rank, the share of
poles whose token at that rank shows up after a top firing (writes) or at the peak (reads), against a
mismatched pole. Ranks 49-64 still run several times above chance for most reads (L28's write read
30×, L12's `read_R` 6×), at 8-19% of a top-8 token's value. The exception is the plain write read at
L2-L4, hub tokens at every rank; its `_z` read has signal there. The catalog keeps the top 64.

![Read signal by rank](figures/13_rank_depth.png)

**Limits.** The tests are token-level, over at most 24 firings, and each pole must clear the FDR on
its own evidence; net shares count weak cases in aggregate. `write_R` tests only the next token: a
write that acts later or through other units counts as dark. A third of firings is a chosen threshold.

## §8. Describing neurons

Can the evidence be turned into words that predict what a pole does? Three studies: a pilot with full
evidence cards, a blind test of how far the lenses alone go, and a cheap labeller run on every pole.

**Scoring.** An explainer writes two labels with a confidence: where the pole **fires**, and what its
write **promotes** when amplified.
- **Firing** is scored by detection (Paulo et al. 2024): fresh scorers see the label and a shuffled
  set of held-out top firings (top documents the card did not show, each cut at the firing token) and
  quiet positions, and give each a probability of firing; the score is the AUROC. The control is
  another pole's label on the same set.
- **Writes** are scored by steering: the pole's write is added at doses where it changes the
  next-token distribution by 0.5, 1 and 2 nats of KL (7-15× its typical activation), on four prompts;
  a fresh judge sees the steered text and four labels from the same band and picks the pole's own
  (chance 25%).
- **Cards are leak-free:** nothing on a card summarizes the held-out documents. (The pilot's cards
  still carried a steering-sample ablation summary that overlapped 90 of its 945 scored positives;
  dropping those moved its mean AUROC from 0.926 to 0.927, and later cards omit it.)

**The pilot** (`describe_cards.py` and `describe_score.py`, retired): 197 poles (2 random neurons
a layer, both poles, and 53 named poles), Claude Sonnet explainers, full cards (lens reads, top 5
contexts, activation statistics, steering, outside labels). Over the 189 that fire, 5 held-out firings
against 5 quiet positions score a mean AUROC of **0.93** [0.91, 0.94] (control 0.54); 0.96 at
L0-L11, 0.89 at L12-L23 and 0.91 at L24-L35. Most poles are token- or phrase-level detectors, often of
what comes next: L7 n271 + poxviruses (monkey⟦p⟧ox), L9 n8548 − an AI assistant's self-reference ("my
training", "been programmed"), L16 n6934 + " upon" in "Once upon a time", L26 n4061 − the word before
"suicide". The explainers judged that the descriptions rest on activations everywhere, with the weight
reads agreeing on about 40% of cards from L12 and a quarter or fewer before.

**The blind test** (`checks/lens_blind.py`): 160 random poles, 32 per band, explainers that never see
steering, in four arms: **lens** (only the eight weight-only reads, plain and `_z`, 12 tokens each),
**contexts** (only the top 5 contexts, activation scale and rates), **both** on one card, with Sonnet
and with Opus. Six held-out firings against six quiet positions, three from random documents and three
from the firings' own documents.

| band | fires AUROC: lens / contexts / both / Opus | writes picked: lens / contexts / both / Opus |
|---|---|---|
| L0-L8 | 0.79 / 0.93 / 0.92 / 0.94 | 0.34 / 0.45 / 0.42 / 0.49 |
| L9-L12 | 0.79 / 0.91 / 0.91 / 0.90 | 0.73 / 0.55 / 0.66 / 0.64 |
| L13-L20 | 0.76 / 0.79 / 0.84 / 0.82 | 0.58 / 0.42 / 0.53 / 0.49 |
| L21-L27 | 0.81 / 0.82 / 0.88 / 0.89 | 0.66 / 0.54 / 0.60 / 0.63 |
| L28-L35 | 0.83 / 0.93 / 0.90 / 0.91 | 0.95 / 0.76 / 0.94 / 0.91 |
| all | 0.80 / 0.87 / 0.89 / 0.89 | 0.65 / 0.54 / 0.63 / 0.63 |

Controls: another pole's firing label scores 0.58; writes chance is 0.25.

![Random poles read blind](figures/14_lens_blind.png)

- **From the lenses alone, a random pole gets a usable description most of the time:** firing labels
  at AUROC 0.80 (58% of labels at 0.8 or more, against 16% of controls), write labels picked 65% of
  the time.
- **For where a pole fires, contexts beat the lenses** (0.87 against 0.80), except at L13-L27, the
  darkest bands, where they tie. Same-document negatives cost both arms about 0.05, so neither's labels
  are mostly topic labels.
- **For what it promotes, the lenses beat contexts** (65% against 54%): 95% at L28-L35, where 42% of
  strongly steered tokens are in the pole's own R-lens top 64 (none in other poles'); 58-73% at
  L9-L27, where the judge matches concepts (L11 n4351 + reads *trauma, genocide, survivors*, and
  steered, the model writes of "a powerful narrative of trauma"); near chance at L0-L8, where steering
  mostly degenerates the text (§6: early meaning is input-side).
- **Combining helps a little:** both sources on one card reach AUROC 0.89 and 63%; the lenses add to
  contexts only for firing at L13-L27 (+0.03-0.07). Picking each side's best source does as well.
- **Opus explains no better than Sonnet** (0.89 and 63% each), which suggests the bottleneck is the
  evidence, not the explainer.
- **Where the lenses mislead:** format and fragment detectors whose reads name something else. L34
  n7339 − fires on a word-initial W (W|inters), its read direction names GPUs (Nvidia, Radeon), and
  its write is 再 (again): three legible sides, unrelated to each other.
- **Confidence ranks:** lens firing labels at confidence 0.7 or more score 0.97, below 0.4 0.67.
- **17% of firing poles defeat every arm**, 19 of 25 at L9-L27: code-identifier fragments and Chinese
  punctuation whose held-out firings vary in surface form.

**A label for every pole** (`10_labels.py`, `checks/haiku_blind.py`). The same tests on 100 more
random poles compared Claude Haiku 5.5 with Sonnet, on the blind test's cards and on a depth-aware
card that shows only the reads that carry signal at the pole's layer (§7), hub tokens removed:

| arm | fires AUROC (control) | fires holds | writes picked |
|---|---|---|---|
| Haiku, lens card | 0.66 (0.55) | 0.30 | 0.57 |
| Haiku, depth-aware lens card | 0.71 (0.55) | 0.41 | 0.63 |
| Haiku, depth-aware card with contexts | 0.82 (0.56) | 0.61 | 0.61 |
| Sonnet, lens and contexts | 0.83 (0.56) | 0.62 | 0.62 |

With contexts, Haiku matches Sonnet; from the lenses alone it trails. The depth-aware card with
contexts then labelled all 884,736 poles through the Batch API (44,244 requests of 10 neurons each,
$105). Confidence ranks (firing AUROC 0.94 at 0.7 or more, 0.79 below 0.4), but only 6-15% of each
band's labels reach 0.7. The full run was checked only through this 100-pole sample.

**Limits.** Positives come from the high-activation regime (top documents only), and quiet positions
are an easier contrast than near misses. 32 poles a band is about ±15 points (100 poles for the Haiku
test, ±20). Steering is amplified past natural firing (5-18× in the blind test, 7-29× in the Haiku
test), in bf16. Explainers, scorers and judges are all Claude models, in separate instances, and the
explainers' prompt carried the depth priors.

## §9. Attention sinks, the output's temperature and entropy neurons

The largest activations in the model all act through two residual dimensions that the output ignores.

**Setup.** `checks/entropy_neurons.py` (ablations on 16,128 positions of 128 web, code and chat
documents; MPS), `checks/sinks.py` (128 documents with 65 distinct first tokens) and
`checks/external_crosscheck.py --sink`.

**Two dimensions set the output's temperature.** The final norm's weight is about 1e-4 on residual
dimensions 1838 and 2276 (median 2.0 elsewhere), so writes there move no logit directly. But they hold
64% of the final residual's squared norm at the median position, so they set the norm the final RMSNorm
divides by: the final RMS is 24.2 at the median and would be 14.6 without them. Writing to them
rescales every logit (RMSNorm algebra; the measurement is new here).

**The attention sink at position 0.** Three L6 neurons (n5723 +, n8518 −, n422 +) fire at up to 10,752
at the first position and write 95-96% of their squared norm onto dimension 2276, raising the
residual's norm from 44 to 12,091 (these are Muyu He's three dimensions, here with their poles). Later
layers' heads attend to that position by default. The trigger is layer 0's MLP output at position 0:

| at position 0 | L6 n5723 | documents with a sink | attention on position 0 at L9 / L21 / L33 | loss on token 2 / tokens 3+ |
|---|---|---|---|---|
| clean | 5,664 | 100% | 0.63 / 0.49 / 0.77 | 2.95 / 2.40 |
| L0's MLP zeroed | 39 | 20% | 0.03 / 0.04 / 0.03 | 8.02 / 2.74 |
| L0's MLP given the same token's output at position 1 | 448 | 9% | 0.54 / 0.42 / 0.66 | 6.05 / 2.56 |
| L0-L5's MLPs zeroed | 0 | 0% | 0.01 / 0.00 / 0.00 | 8.80 / 3.06 |
| L0's top 300 neurons (by attribution) zeroed | 4,096 | 97% | 0.61 / 0.48 / 0.75 | 3.23 / 2.43 |
| control: L0's MLP zeroed at position 5 | 5,664 | 100% | 0.62 / 0.49 / 0.76 | 2.95 / 2.43 |

- **Without L0's MLP at position 0 the sink mostly fails** and attention to position 0 collapses.
- **It is the MLP's response to being first,** not a few neurons: L0's top 300 neurons removed, the sink
  still forms in 97% of documents.
- **A smaller sink suffices:** halving it (zeroing L5) leaves the attention unchanged.

**A second sink on the year after a date.** Three L16 neurons (n2271 +, n1875 +, n1756 +; up to 5,984)
write 96-99% onto the same dimension at the first digit of a year ("Tuesday, May 7, ⟦2⟧013"), which
carries about 15,000 there by L20. The same massive activation appears on `__` in `from __future__`
and on position 1 of code files. Pooled over these 22 positions, later tokens attend to them from L9
as much as to position 0 (0.34 against 0.35 at L9; 0.48 against 0.38 at L30). Four L34 poles (n6620 +,
n3471 +, n8468 +, n3508 +) clear the activation before the output: at 11 year-digit and `__`
positions, n6620 + cuts the final RMS by 68%, lowers the output's entropy by 9.7 nats and the loss by
19.0; without them the output there is close to uniform. Massive activations fading in the last layers
matches Sun et al. (2024). Position 0's sink is halved by L34-L35, not cleared; the final norm divides
it away.

**Entropy neurons.** Stolfo et al. (2024) found last-layer neurons that act through the final norm's
scale. Freezing the scale at its clean value separates that path from the direct one.
- **L35 n9510 + raises entropy where the next token is hard to guess.** It is always on (median 268)
  and fires hard at 3.5% of positions: hashes, IDs and version numbers ("jsessionid=⟦0⟧000"). There
  it raises the final RMS by 9% and the entropy by 0.19 nats, though its direct effect alone would
  lower it by 0.21; it lowers the loss by 0.07 (the hedge pays).
- **L35 n9346 + does the opposite** on near-determined continuations ("Esch⟦ich⟧ia",
  `codecs⟦.char⟧map`): the largest L35 write (98% in the null space) cuts the final RMS by 31% and
  entropy by 1.58 nats, all through the scale (frozen: −0.02).
- **At L32-L33 no candidate acts mainly through the scale** (8 per layer by null-space write at typical
  activation, against 8 random).

**For the catalog.** These poles' lens reads are uninformative: the lenses keep only a direction, and
the readout ignores the two dimensions where these poles act. Their descriptions rest on activations
and on this measurement; `null_share` flags them only together with the activation scale.

**Limits.** 16,128 positions, and the second sink's clearing rests on 11 positions. Poles are ablated
one at a time, though the four L34 poles fire together.

## §10. Outside labels

How do labels from other methods compare with the catalog? The main outside source is the [Neuron
Persona Library](https://neuronpersonas.com) (by jfonsecarivera), 34,220 rows, each a pole whose push
on "Who are you?" made the model claim a persona, named by Claude Haiku. `neuron_lens/known.py` holds
smaller sets: six steering-verified persona neurons from an earlier experiment on this model, and
published neurons from Sudarsh, Sauers, Muyu He and Kazemi et al.

**Strict name matches** (`checks/library_crosscheck.py`): the first token of a persona name's words in
the pole's write read. In the top 10 for 52% of L30-L35 rows and 30% of L24-L29 rows under every lens,
2-3% at L12-L17 (control 0.2-0.3%). At L0-L5, the R-lens finds 15% where the J-lens finds 1% and the
logit lens 3%.

**Generous matches** (`checks/library_judge.py`): 360 rows, 60 per layer group; Claude Sonnet judges
score each read against its persona's name and, shuffled in, another row's (0-3; 2 or more is a clear
match: the concept in any language, or a strongly associated thing). Reads from the write side, the
input side (`read_R` and `read_S`) or both together:

| layers | write | input | both |
|---|---|---|---|
| L0-L5 | 0.35 (0.02) | 0.35 (0.02) | 0.38 (0.03) |
| L6-L11 | 0.27 (0.02) | 0.22 (0.02) | 0.33 (0.07) |
| L12-L17 | 0.20 (0.07) | 0.17 (0.07) | 0.28 (0.10) |
| L18-L23 | 0.22 (0.02) | 0.10 (0.03) | 0.27 (0.05) |
| L24-L29 | 0.45 (0.00) | 0.37 (0.02) | 0.50 (0.03) |
| L30-L35 | 0.53 (0.02) | 0.42 (0.02) | 0.62 (0.02) |

![Persona labels judged against each side](figures/15_library_judge.png)

- **A generous reading doubles the strict rate:** weighted by the library's depth mix (mostly
  L6-L17), 27% of rows clearly match the write read, 22% the input read and 34% both together
  (swapped names 3-6%). Translations and associated words count: L9 n3370 − "Dragon" writes dragon,
  龙, dragons.
- **Mid-network is hardest** (20-22% of rows at L12-L23 on the write side), the census's darkest
  stretch (§7).
- **Fused personas.** The library also pushed 2-3 neurons at once; 3,097 of 13,861 pairs gave a persona
  no parent gives alone. The R-lens read of the summed pushes ranks words in neither parent's name in
  its top 100 for 19% of pairs (mismatched names 3%; J-lens 10%, logit lens 7%;
  `checks/combo_crosscheck.py`). The best parent's read alone does about as well (24%, control 5%):
  the new concept is mostly latent in one parent's write.
- **Steering one prompt finds personas, not everything.** Of the 79,949 poles whose write predicts
  what follows their firings (§7), 4.0% have a library row, mostly because 98% sit at L21-L35, where
  the library is thin; against each layer's base rate the library is enriched for them (1.5×). What it
  misses has no self-description: idiom completers (L30 n1034 + writes *again* on "once"), morphology
  (L8 n1541 + writes *-ism*/主义 on the stem before it), everyday concepts (L28 n8868 − writes
  *democracy* on "arist|", "olig|").

**Against whole evidence cards** (§8's pilot), the explainers' verdicts on outside labels:

| source | method | agrees | partial | disagrees |
|---|---|---|---|---|
| earlier persona-steering experiment | steering-verified personas | 6 | 0 | 0 |
| Sudarsh | activation contrast × `W_down`, steering | 10 | 3 | 0 |
| Neuron Persona Library | steering on "Who are you?" | 13 | 9 | 11 |
| Sauers | steering and causal search | 7 | 3 | 4 |
| Muyu He | MLP energy at token 0 | 3 | 1 | 2 |
| Kazemi et al. | refusal gradient × activation difference | 1 | 1 | 1 |

- **Activations sharpen the steering-verified labels:** L8 n1931 + ("Jesus") fires on the Bible as
  scripture ("God's⟦ Word⟧", "non-B⟦ib⟧lical"); L21 n1180 − ("mother") on leaving work to care for
  family ("stay⟦-at⟧-home moms"); L27 n1343 + ("candidate") on deciding to run for office.
- **Labels from steering one prompt fail most often:** the library's "creator god" pole, L8 n8079 +,
  fires on time words (times, 时间, `timing_stats`); Kazemi et al.'s refusal neuron, L14 n7924 +, fires
  on light, playful request framing.
- Unsigned labels count both poles, so one pole of each often "disagrees".

So the methods are complementary: steering on one identity prompt reaches the personas, activations
and lenses reach the rest, and each checks the other where they overlap.

**Limits.** Strict matching scores only the first tokens of names. 60 rows a group is about ±12
points. The judges are the library labeller's model family. About a third of the library's strongly
pushed replies change with the batch layout.

## §11. Attention heads: circuits read token by token

A head moves information between positions: its QK circuit decides which earlier position it reads,
its OV circuit what it writes from there. Both can be read from the weights over the whole vocabulary,
one token at a time.

**Patterns** (`06_activations.py`, eager attention). Sink, previous-token and self attention from
queries at 16+; induction and duplicate-token scores from 64 sequences of 64 random kept tokens said
twice. Qwen3-8B copies such a sequence almost perfectly (13.55 nats a token on the first copy, 0.09 on
the second). It has 33 induction heads (score past 0.3; strongest L13h13 0.99, L22h23 0.97), 33
duplicate-token heads, mostly early (L1h31, L1h28, seven in L3), 4 previous-token heads (L0h1, L1h4,
L19h20, L12h0) and 60 sink heads (none before L7). Removing one head at a time (`head_patterns.py`, retired) costs
the second copy at most 0.07 nats (L1h28): copying is spread over many heads, as is usual in large
models. Duplicate-token heads cost as much as induction heads but copy almost nothing; they likely
write a marker that the token was seen.

**Where to read a head.** A head reads layer `L − 1`'s output and writes into layer `L`'s MLP input,
so its token map is read between those points. A token at its input can be its S-lens footprint (the
position holds token `v`) or its R-lens direction (the position says `v`); its output can be read with
the R-lens (the write makes the model say `u`) or the S-lens (the residual looks as if `u` were there).
`checks/head_lens_sides.py` tests the four brackets at each head's 24 top firings (23,112 firings from
L1): how well a target token's weight-only copy score predicts its rank in the head's actual output,
own head minus another head of the layer. The target is the attended token for an S-lens input, and
the token after the attended position for an R-lens input:

| band | S→R | S→S | R→R | R→S |
|---|---|---|---|---|
| L1-L8 | +0.39 | +0.44 | 0.00 | +0.08 |
| L9-L12 | +0.23 | +0.16 | +0.07 | +0.11 |
| L13-L20 | +0.21 | +0.24 | +0.02 | +0.02 |
| L21-L27 | +0.65 | +0.16 | +0.15 | +0.08 |
| L28-L35 | +0.80 | +0.08 | +0.42 | +0.11 |
| all | +0.51 | +0.23 | +0.13 | +0.08 |

![Which lens on each side of a head](figures/16_head_lens_sides.png)

- **The input side is the S-lens, for both circuits.** An S input beats an R input with the same
  output lens in every band, on this test and on whole predicted columns (the one exception is the
  copy test at L28-L35 with an S-lens output, which the catalog doesn't use there); on the QK side
  too, the S-lens key read agrees with the attention keys receive (+0.18, control +0.08) better than
  the R-lens's (+0.09, control +0.05). A head reads what a position holds, not what it is about to
  say.
- **The output side is the S-lens early and the R-lens late,** as for neuron writes: S→S leads at
  L1-L8 and edges S→R at L13-L20 (where it also predicts whole columns better), S→R leads at L9-L12
  and by far from L21. Induction heads favour S→R, duplicate-token heads S→S.
- An earlier, coarser test agrees: of the four brackets' copying scores, only S→R tracked single-head
  ablation better than the raw score (Spearman +0.22, +0.08 over raw, 95% interval +0.03 to +0.12).
- The catalog reads QK with the S-lens on both sides, and OV as `ov_R` (S→R, trusted from L9) and
  `ov_S` (S→S, trusted through L8 and a second view through L20).

**What each head copies** (`ov_R`, `ov_S` in `heads/reads`): each token's z in its own column of the
OV map, that is, where `v` ranks in what the head writes when it reads `v`. At each firing this
weight-only read predicts where the attended token ranks in the head's actual output (Spearman +0.54
over firings; the same token under another head, 0.00), from +0.23 at L13-L20 to +0.77 at L28-L35. The
S-lens input is what helps: against the same logit-lens read of the output, an S-lens input scores
+0.51 and the raw embedding +0.42. Examples:
- **Classes, across languages:** L26h31 copies quantities (` weight | 体重 | strength | temperature`);
  reading ` population` it writes `人口 | population | statistics`.
- **Pronouns by gender:** L20h17, reading ` herself`, writes ` herself | 她 | she` and suppresses
  ` himself | 他自己 | his`, and the reverse for ` himself`. L6h21 copies *you*.
- **Copy and copy suppression:** L34h1 copies surnames (` Heller | Crosby | Gallagher`); L32h9, a
  copy-suppression head (McDougall et al. 2023), suppresses the same ones.
- **Class suppressors, late:** L32h1 suppresses ` fluid | traffic | bridge | fuel`; L30h10 ` region |
  period | moment`. L19h7 and L10h21 suppress British spellings; reading `Whilst`, L19h7 writes
  American ones.
- **A class the single score misses:** L30h25 scores −0.10 as a copier, yet copies one class at z up to
  31 (`原始 | raw | formal | traditional | native`).
- Induction heads copy nearly everything (L22h23 and L24h22: median z +3.0 and +3.6); duplicate-token
  heads almost nothing.

**What each head attends to** (`qk_S`, `qk_self_S`): each key token's z under the head's mean query,
with the q/k norms' derivative and RoPE averaged over the head's own offsets.
- **Self-match finds duplicate-token heads from the weights** (`qk_self`, AUROC 0.96): as queries they
  favour a token's own surface form (L1h28 reading ` Paris` favours ` Paris | Paris | Parse`).
- **Attends-to agrees with the attention tokens receive in context early and late:** Spearman over
  tokens +0.21 at L0-L8 (control +0.09), +0.08 at L13-L20 (+0.05), +0.19 at L28-L35 (+0.03); best at
  L0 (+0.43: L0h27 attends to the starts of code and documents, L0h0 to Chinese and Japanese function
  words). The S-lens beats raw embeddings at every depth.
- **Late heads attend to what they copy:** at L28-L35, 19% of heads attend mostly to tokens high in
  their own copy read (control 2%), and class suppressors to the class they suppress (L31h23 attends to
  ` Combination | Equality` and suppresses ` alternative | balance | combination`). Mid-network, where a
  head looks says little about what it writes.
- **Query reads past L1 are noise:** L20h17 favours nearly the same keys whether its query is ` she` or
  ` he`.

**Limits.** The OV read assumes the head reads one token's footprint; in context a head mixes several
positions. Reads are scale-free. The S-lens footprint is a context-averaged stand-in (§6). The QK read
is first-order: attention that depends on the query's context or on position alone (previous-token,
induction) is not read.

## §12. Attention heads: census and labels

**Which reads hold** (`09_legible.py`). Each head's weight-only reads are tested at its 24 top firings
(largest output norm, one per document) against the layer's other heads on the same firings; BH over
all 1,152 heads at 5%. `ov_R`: the attended token's column promotes the next token; `ov_copy`: the head
promotes what it attends to; `ov_S`, `qk_S`: the attended token scores high in the head's own read.
Dark: no test, no token mode (a third of firings sharing a query or attended token), no pattern.

| band | `ov_R` | `ov_copy` | `ov_S` | `qk_S` | token mode: query / attended | pattern | dark |
|---|---|---|---|---|---|---|---|
| L0-L8 | 0.01 | 0.01 | 0.00 | 0.00 | 0.31 / 0.49 | 0.10 | 0.41 |
| L9-L12 | 0.00 | 0.01 | 0.00 | 0.00 | 0.08 / 0.13 | 0.08 | 0.78 |
| L13-L20 | 0.01 | 0.00 | 0.00 | 0.00 | 0.10 / 0.15 | 0.08 | 0.74 |
| L21-L27 | 0.05 | 0.01 | 0.00 | 0.00 | 0.15 / 0.25 | 0.06 | 0.61 |
| L28-L35 | 0.27 | 0.10 | 0.00 | 0.00 | 0.32 / 0.27 | 0.04 | 0.38 |
| all | 0.07 | 0.03 | 0.00 | 0.00 | 0.21 / 0.28 | 0.07 | 0.56 |

![Which reads hold for each head](figures/17_head_census.png)

- **Late heads write the next token:** `ov_R` holds for 27% of L28-L35's heads (50% at L33). Read
  from their actual outputs in context, 45-68% of heads from L21 promote the token that follows their
  strongest firings, and 42-46% the token they attend to, against 5-8% and 16-24% mid-network
  (`heads/acts`' `out_R_z`; `head_census.py`, retired).
- **Heads are dark mid-network, as neurons are:** 74-78% at L9-L20, 38% at L28-L35 (the same shape as
  §7).
- **The S-side reads never separate a head from its layer.** S→S copying is common to nearly every
  head (`ov_S_copy` medians +0.65 to +0.84 in every band), and the mean query's favourite keys are
  shared too. They stay useful as reads of what heads do (§11), not as evidence that a firing is this
  head's.

**A label for every head** (`10_labels.py --side heads`). Claude Haiku 5.5 labelled all 1,152 heads
from leak-free cards (patterns, output scale, OV and QK reads, drivers, the first 5 contexts): where it
fires, what it attends to, what it writes ($0.28 on the Batch API). Checked blind on 100 random heads,
20 a band, by Claude Sonnet:
- **Firing:** the labels separate 4 held-out top firings from 4 of another head of the band at AUROC
  0.71 [0.65, 0.77]; a third head's labels score 0.51. Half the heads (53%) score 0.8 or more. The
  contrast is another head's strongest firings, harder than §8's quiet text.
- **Writes:** a judge shown the firings' in-context output read picks the head's own label out of four
  49% of the time [39, 60] (chance 25%): 79% at L28-L35, 55-56% at L9-L12 and L21-L27, 33% at L0-L8,
  chance at L13-L20.
- Most labels are unconfident (median 0.25-0.30), and confidence tracks the score only weakly
  (Spearman +0.20).

**Limits.** The tests compare a head with its own layer, so a property shared by a layer's heads never
passes: the right null for "this head", the wrong one for "these heads". Firings are the top 24 by
output norm, one per document, which favours heads with large outputs; sink heads fire where they leave
the sink. 20 heads a band is noisy.

## §13. Gemma 4 12B: the text catalog in a second model

**Setup.** The pipeline (`06`-`09`, `checks/causal_lens.py`) run on the base model
`google/gemma-4-12B` with `NEURON_LENS_MODEL=google/gemma-4-12B`, on H100s in bf16: 48 layers ×
15,360 GELU-tanh neurons and 16 heads a layer. Its architecture differs in ways the readers follow
(a norm after the MLP and a per-layer scalar on the write, shared keys and values on global layers,
a softcap on the logits; design.md, [Gemma 4 12B](design.md#gemma-4-12b)). Both lenses were fitted
as Qwen3-8B's, on the same 1,000 DCLM documents (about 16 s a document per lens on one H100);
split-half differences are 0.01-0.12 by layer for the R-lens (median about 0.05) and 0.01-0.07 for
the S-lens. The activation corpora have Qwen3-8B's sizes (15.4M positions), with the chat slice as
plain `User:` / `Assistant:` turns, since the base model has no chat template. The causal sample is
24 random neurons a layer, both poles, with no J-lens to compare.

**Steering** (as §3: the write added at 1× and 4× the pole's `act_top` at positions 16-63 of 16
DCLM documents; Pearson r with the lens's read, medians over the random poles; controls within
±0.05 at every layer):

| layer | logit lens, 1× | R-lens, 1× | R-lens, 4× | R-lens, ablation where it fires |
|---|---|---|---|---|
| L0 | 0.01 | 0.07 | 0.26 | 0.00 |
| L4 | 0.00 | 0.00 | 0.04 | 0.00 |
| L8 | 0.00 | 0.09 | 0.15 | 0.01 |
| L11 | 0.04 | 0.40 | 0.42 | 0.01 |
| L16 | 0.06 | 0.47 | 0.42 | 0.06 |
| L24 | 0.21 | 0.62 | 0.47 | 0.07 |
| L32 | 0.52 | 0.82 | 0.80 | 0.24 |
| L40 | 0.85 | 0.96 | 0.95 | 0.64 |
| L46 | 0.94 | 0.99 | 0.98 | 0.91 |

![Gemma 4: lens reads against steering and ablation](figures/19_gemma_steering.png)

- **From L11 the R-lens read of a write predicts steering, as in Qwen3-8B:** a median r (over each
  band's layers) of 0.53 at L12-L23, 0.82 at L24-L35 and 0.97 at L36-L47. The logit lens trails until
  the last layers (0.08, 0.44 and 0.86 by the same bands).
- **Before L11 neither lens predicts a 1× push** (r 0.00-0.09). The correlation rises with the dose
  (L0: 0.07 at 1×, 0.26 at 4×), which suggests the natural-scale pushes are too small to measure
  over the documents' own variation rather than misread. Ranked instead of correlated, an amplified
  write raises the tokens its R-lens read names above other poles' read tokens at a median rank of
  0.98 at L0-L11 (`checks/write_steer.py`, a 192-pole panel, with an earlier R-lens fit on 174
  documents).
- **Ablation agrees only late,** as in Qwen3-8B (§4): 0.05 at L12-L23, 0.17 at L24-L35, 0.71 at
  L36-L47.

**Which reads hold** (`09_legible.py`, as §7, BH over all poles at 5%):

| band | silent | `write_R` | `write_S` | `read_R` | `read_S` | any test | token detector | dark |
|---|---|---|---|---|---|---|---|---|
| L0-L11 | 0.00 | 0.00 | 0.01 | 0.00 | 0.35 | 0.35 | 0.67 | 0.23 |
| L12-L23 | 0.00 | 0.00 | 0.04 | 0.00 | 0.06 | 0.07 | 0.36 | 0.61 |
| L24-L35 | 0.00 | 0.08 | 0.00 | 0.01 | 0.04 | 0.11 | 0.25 | 0.66 |
| L36-L47 | 0.00 | 0.20 | 0.00 | 0.00 | 0.03 | 0.22 | 0.23 | 0.60 |
| all | 0.00 | 0.07 | 0.01 | 0.00 | 0.12 | 0.19 | 0.38 | 0.52 |

![Gemma 4: which reads hold, by layer](figures/20_gemma_census.png)

- **The same shape as Qwen3-8B's:** 19% of poles pass some weight-only test (Qwen3-8B: 22%) and 52%
  are dark (57%). The input side reads early (the S-lens names the firing token for 84-88% of L0-L1's
  poles), the write side late (`write_R` 22-37% at L33-L40), and 54-85% of L16-L31 is dark.
- **The input-side stage reaches L13, at about Qwen3-8B's relative depth.** `read_S` falls to 3% at L8
  and under 1% at L9, then rises again to 57% at L11 and 33% at L12-L13, where `write_S` also holds for 10-22% of poles:
  early neurons that write back the tokens they detect (§6), ending at 27% of the depth (Qwen3-8B: L8,
  23%).
- **Late writes fall off at the end:** `write_R` drops from 29% at L40 to 8-14% at L44-L47, where
  Qwen3-8B's keeps rising to its last layer.
- **No poles are silent.** Gemma's activations take both signs at every layer, and none of its early
  layers is near-silent as Qwen3-8B's L1 is (§2).

![Both models: steering and the census by relative depth](figures/18_two_models.png)

**Heads** (`09_legible.py`, as §12). With 16 heads a layer, a head's mismatched control has 240 pairs,
so no p-value falls below 1/241, and BH over 768 heads passes none. The `legible` table adds the
uncorrected `<test>_p`. At p ≤ 0.05 (5% by chance), `ov_R` holds for 21% of L36-L47's heads and
`ov_copy` for 17%; 70% of L12-L35's heads are dark, 19% of L0-L11's. `qk_self` finds the 26
duplicate-token heads at AUROC 0.95 (Qwen3-8B: 0.96).

**Labels** (`10_labels.py`, as §8 and §12, with Gemma 4's own reading plan and prompts: four bands
from the census above, L0-L4, L5-L13, L14-L32 and L33-L47, and six worked examples of its own poles).
Claude Haiku 5.5 labelled every head ($0.19) and all 1,474,560 neuron poles (73,728 requests of 10
neurons each, $171.66).

- **Haiku is less sure of Gemma 4's neurons.** Median confidence is 0.30 for firing labels and 0.20 for
  write labels (Qwen3-8B: 0.30 and 0.25), and 3% and 6.5% of them reach 0.7 (Qwen3-8B: 9% and 11%). It
  follows the census: firing labels are least sure at L14-L32 (median 0.20; 0.30 in every other
  band), where most poles pass no test of a read, and write labels most sure there (0.30; 0.10 at
  L0-L4, 0.20 at L5-L13, 0.15 at L33-L47), where `out·R` predicts steering. Gemma 4's neuron labels
  are not checked blind.

- **Head labels: the firing labels hold, the write labels don't.** Checked blind as §12's, on 80 random
  heads (20 a band) by Claude Sonnet 5.5: the `fires` and `attends` labels tell 4 of a head's held-out
  firings (ranks 6-24) from 4 of another head's at AUROC 0.71 [0.64, 0.79] (a third head's labels on the
  same contexts: 0.47), as Qwen3-8B's do (0.71). A judge picks a head's `writes` label out of four from
  what its firings promote in context 22% of the time [13, 30], chance (25%; Qwen3-8B: 49%). Its cards'
  OV reads rarely pass a test (above), and what a head's OV map promotes is not what it writes in
  context; the labels follow the cards (`data/summary/labels_heads_gemma-4-12b.md`).

**Limits.** The base model, not a chat model; 24 random neurons a layer for the causal sample; no
J-lens for Gemma 4, and its neuron labels are not checked blind yet (`checks/haiku_blind.py` steers on a
GPU). The head tests have too little power for a false discovery rate.

## §14. Images and sound: what drives a neuron

Gemma 4 12B takes an image or a sound with no encoder network: each 48 × 48-pixel patch (an image is
an 8 × 8 grid of them, 384 pixels a side) or 40 ms frame of 16 kHz audio (64 of them, 2.56 s) enters
the residual stream through a few norms and one linear map. So the lenses can be fitted on images and
sounds, and a neuron's read direction can be carried back to pixels or samples.

**Setup.** *Media lenses* (`02_fit.py --modality image|audio`): both lenses fitted on media documents
(a short prompt around one image or sound), seeded and read at its 64 slots: COCO 2014 validation
images (1,000 for the S-lens, 512 for the R-lens) and ESC-50 clips (1,615 and 512). The S-lenses are
seeded only in the subspace the embedder reaches (2,048 directions for images, the patch embedder's
rank being 2,041; 640 for audio), which is exact and cuts their cost to 0.1-0.3 of a text document's.
Their maps match autograd to 1e-6. *Exemplars* (`13_media_activations.py`): every pole's mean
activation over the slots of 49,694 labelled images (2,805 labels, from 49 datasets: ImageNet,
Places365, food, cars, sketches, documents, charts, art, medical scans, …, plus manga pages and
synthetic gratings, shapes and scripts) and 46,932 labelled sounds (1,167 labels, from 11 datasets:
AudioSet, FSD50K, ESC-50, speech in many languages and accents, music, plus synthetic tones, chirps,
beats and clicks; `data/corpus/media-*.provenance.json`). A pole's top label (by the z of its mean
activation over that label's documents) is *reliable* if it passes the 99th percentile of a
shuffled-label null for its layer and is in the top 5 of both halves of the documents; a held-out
AUROC scores it on the odd documents after choosing it on the even ones.

| modality | band | reliable label | held-out AUROC (median) |
|---|---|---|---|
| image | L0-L11 | 34% | 0.96 |
| image | L12-L23 | 26% | 0.97 |
| image | L24-L35 | 39% | 0.97 |
| image | L36-L47 | 32% | 0.97 |
| audio | L0-L11 | 51% | 0.91 |
| audio | L12-L23 | 19% | 0.86 |
| audio | L24-L35 | 28% | 0.83 |
| audio | L36-L47 | 18% | 0.78 |

- **Labelled media name what a third of poles respond to.** Early reliable labels are often
  primitives (checkerboards, gratings, click trains at a given rate, scripts by writing system); later
  ones are objects, scenes and languages. Audio labels hold less well past L11.

**Reads in pixels and samples** (`14_media_tables.py`, tested by `checks/media_lens.py`). A pole's
read direction through the media S-lens, mapped back through the embedder, is a 48 × 48 patch
(`read_png`) or 640 samples (`read_wave`): what the direction looks like as input. To test it, a read
is added to 32 held-out images or sounds, and the pole's change is ranked among its changes under 191
other poles' reads (1 the most, 0.5 chance), over a panel of 192 poles (`checks/media_reads.py`):

| band | image read: its read projection | image read: its activation | optimised patch: its activation | optimised sound: its activation |
|---|---|---|---|---|
| L0-L5 | 0.99 | 0.92 | 1.00 | 1.00 |
| L6-L11 | 0.94 | 0.09 | 0.98 | 1.00 |
| L12-L23 | 0.89 | 0.29 | 0.86 | 1.00 |
| L24-L35 | 0.53 | 0.41 | 0.62 | 0.56 |
| L36-L47 | 0.48 | 0.49 | 0.43 | 0.47 |

![The media reads' held-out tests](figures/21_media_reads.png)

- **The image read is a projection, not a stimulus.** It moves the pole's own read projection most
  through L23, but raises the pole's activation only at L0-L5, and lowers it at L6-L23 (0.09-0.29):
  past the first layers, what aligns with the read direction at the input is not what makes the
  neuron fire. The audio read is at chance on both counts.
- **Optimised reads drive the neuron.** Starting from the read, gradient ascent through the model
  (`15_media_optimised.py`: one pattern tiled over all 64 slots, at 8% of the pixel range or 0.3 of a
  frame's RMS, 24 Adam steps over 128 images or clips) finds a patch or a 40 ms frame that raises its
  pole on held-out inputs through L23. For the catalog's 48 best-labelled poles a layer (L0-L35), it
  raises the pole on 8 held-out documents for 95%, 70% and 56% of image poles by band (L0-L11,
  L12-L23, L24-L35) and 99%, 86% and 51% of audio poles. Both are near chance past L35.
- **Deep neurons respond to whole sounds.** Past L35, where one repeated frame does nothing, a 10 s
  clip synthesised for each of 24 poles (one `+` pole a layer from L24, optimised as a spectrogram so
  it stays sound-like) drives its own pole more than the other 23 clips do in a different prompt, at a
  median rank of 0.87 (`checks/synth_audio.py`); one pole a layer is a small sample.
- **The patch reads share one pattern; the optimised patches don't.** Raw S-lens reads carry the same
  column streaks from neuron to neuron, a hub pattern like the hub tokens of text reads (§1); the
  optimised patches differ by neuron.

![The patch each neuron reads, and the patch that drives it](figures/22_media_gallery.png)

**Limits.** A read is one slot pattern summed over the 64 slots, not a map of where in an image it
sits. Media lenses are fitted on photos (COCO) and environmental sounds (ESC-50) only. Labels name
what a document is about, not each patch or frame. The audio tests' held-out clips are among the audio
S-lens's fitting clips (all of ESC-50), though not the optimisation's.

## §15. One concept in text, images and sound

**Setup** (`checks/crossmodal.py`). For every pole with a reliable image or audio label (§14; labels
that name a whole collection or a synthetic probe are left out), the label's rank among the
modality's labels on the text side, three ways: how hard the pole fires on `Here is an example:
{label}.`, its text S-lens read (`read_S`) and its text R-lens write (`write_R`), each scoring the
label's tokens. A rank of 0.95 or more passes; a shuffled pairing passes 5%.

| | image labels: share passing | audio labels: share passing |
|---|---|---|
| its words fire it | 31% (L12-L23: 42%) | 10% |
| the text S-lens reads it | 8% | 9% |
| the text R-lens writes it | 14% (L12-L23: 19%) | 8% |
| all three | 4,378 poles | 708 poles |
| poles scored | 253,749 | 232,251 |

![One concept in text, images and sound](figures/23_crossmodal.png)

- **Image concepts are text concepts mid-network.** At L12-L23, 42% of the poles that respond to an
  image label also fire on its name in text, against 5% by chance; 8% at L0-L11, where image labels
  are mostly visual primitives.
- **Sound is weaker:** 10% fire on their label's name, twice chance.
- **18 poles hold one concept across all three modalities.** Of 45,138 poles with a reliable label in
  both images and sound, 18 pass all three text tests for both, where independent lists would share
  1.8. They are places and peoples, with their language:

| pole | image label | audio label | exact text drivers | R-lens read of its write |
|---|---|---|---|---|
| `L14 n8313 +` | mosque indoor | Arabic | Amal, أ, Mohamed, Noor | Arabic, Quran, islamic, Saudi |
| `L23 n2831 +` | Sarees | Gujarati | Tendulkar, Karnataka, Bengaluru, lakhs | Mumbai, India, Hyderabad, Delhi |
| `L33 n5249 +` | tacos | Spanish | Peruvian, Ecuadorian, Mexican, Aztecs | Mexican, 墨西哥 (Mexico), Mexico |
| `L33 n6099 +` | pagoda | Mandarin Chinese | WeChat, Xiaomi, Weibo, Shanghai | Chinese, China, Beijing, 中国 |
| `L36 n14109 +` | totem pole | Maori | Cherokee, Navajo, Apache, Mohawk | Native, Indian, Indigenous, Cherokee |
| `L37 n7165 +` | giant panda | Mandarin Chinese | 熊猫 (panda), 四川 (Sichuan), 成都 (Chengdu) | panda, 熊猫, Chengdu |

  The others pair mosques or Arabian camels with Arabic (8 poles), a coral reef with Māori, tacos with
  Spanish, and Pembroke corgis with Welsh. Firing on both names alone is near chance (1,602 poles
  against 1,267 expected): the lenses' reads are what make the triple match specific.

**Limits.** Label words are English and scored in one template; a rank among the modality's labels
says the pole prefers this label to the others, not that the concept is all it does. Exemplar labels
come from the collections, whose coverage of a concept varies.

## §16. Attention heads in image and audio contexts

**Setup** (`checks/head_media.py --n 64`). 64 mini-ImageNet validation images (64 classes) and 64
ESC-50 clips (49 classes), each in the media prompt and followed by ` a <label>`, its own and another
class's. Per head: attention from the text after the image or sound to its slots, against an even
spread over the keys; within the media, attention to the same slot, the other slots and grid
neighbours; each head zeroed in turn, the rise in the right label's loss minus a wrong one's, as a z
over documents (*content carriers*); and the head's output at the article read through the R-lens
`mid`, as the right class's mean percentile among the classes, against 200 label shuffles with BH
over heads (*label writers*). Per-head values are in `heads/media`.

- **The model knows what it sees, but barely what it hears.** The right label's loss is below a wrong
  one's for 97% of images (3.64 against 10.29 nats), 55% of sounds (6.48 against 6.56).
- **Images: early heads look within the image, later ones carry it out.** L0-L3 heads attend from a
  slot to its grid neighbours (L0h13: 0.85). Reading from the text into the image peaks at L23-L31
  and L44 (L24h13 and L44h7 at 0.96 of an even spread). Zeroing single heads costs the right label
  more than a wrong one, most for L28h6 (z 9.9), L29h1 (9.7), L29h9 (8.4) and L25h2 (7.9): 29% of
  L24-L35's heads pass z > 3. By the size of the loss, early heads L8h15 and L7h13 cost the most
  (0.75 and 0.65 nats, z 5). From L12, 36-43% of heads per band write the right class (q ≤ 0.05),
  most at L35-L46 (L37h10 and L44h7: the right class's mean percentile 0.955).
- **Sound: heads read it early and nothing carries it.** The text attends to the audio most at L0-L10
  (L8h5: 1.40 times an even spread), but no head passes z > 3 and no head writes the right class.
- Which heads read the media is shared across modalities (Spearman +0.51 between image and audio
  reading), and which carry content is not (+0.09).

![Heads in image and audio contexts](figures/24_heads_media.png)

**Limits.** 64 documents a modality in one prompt; zeroing a head measures what the model loses
without it, not all it does. With the model near chance on sound, the audio results say little.

## Open questions

- Why the S-lens read of input directions falls after L8, when the linear share of a token's effect is
  flat from L4 to L17 (§6).
- What L2-L3 compute at position 0 (§2), and what duplicate-token heads write (§11).
- Whether J++ filtering (Ayonrinde 2026) or a longer context improves early and mid-network reads.
- The same catalog for Qwen3-8B-Base, as a neuron-by-neuron diff of what post-training changed.
- Why Gemma 4's early writes barely move the output at their natural scale (§13).
- Labels in words for Gemma 4, with its image and sound exemplars on the cards (§14), and media reads
  per slot, which would say where in an image a neuron looks.
