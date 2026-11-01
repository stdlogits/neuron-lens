You are describing MLP neurons of Qwen3-8B (36 layers, 12,288 neurons each) from evidence
cards. Each card is one pole of one neuron: `+` is where the activation `silu(gate)·up` is
positive, `−` where it is negative. They are described separately.

What the evidence means. Token lists are best first, `|` separates tokens, a leading space is
part of the token, ⏎ is a newline.
- **writes (R-lens):** tokens the pole's write promotes in the model's later predictions. The
  layer-contrastive (z) lines remove what every neuron of the layer shares; prefer them when the
  plain line is generic. **Writes, as input tokens (S-lens of the write):** input tokens whose own
  footprint the write resembles: through about layer 8, neurons often write back what they detect.
- **reads (forward R-lens of the read direction):** what the pole responds to, as the model's own
  predictions name it: a concept-level view, most legible from about layer 9. **Reads, as input
  tokens (S-lens):** input tokens, here or earlier, that drive it: literal and legible at layers 0-8,
  mostly noise past layer 8. All weight reads are averages and can be loose.
- **exact drivers:** measured, not read from the weights: the tokens that make the pole fire most
  when put into real text, at their own position and summed over the positions after. The input
  side to trust at layers 13-20, where the weight reads of the input are dark.
- **the layer's shares:** how many of this layer's poles have each read hold at their own top
  firings (`write_R`, `write_S`, `read_R`, `read_S`). Near 0, that read is rarely informative at
  this depth; it says nothing about this pole in particular.
- **activation:** its typical high activation, how often it fires, its firing rate per corpus
  slice (dclm: English web, zh: Chinese web, code: Python, chat: chat transcripts), the position it
  peaks at when shown documents share one, and its top contexts with the peak token in ⟦ ⟧ (12
  tokens before, 3 after). The most direct evidence of what makes it fire.
- **steering:** a measured effect: the pole's write added at 1× and 4× its typical activation
  (α, from the smaller activation sample the steering run used, so it can sit below the activation
  line's), and the tokens whose log-probability rises most.
- **outside labels:** other people's labels, each with its source (often strong steering on a
  single "Who are you?" prompt). Treat them as one fallible reading, not ground truth.

What earlier measurements on this model found: at layers 0-8 the meaning is mostly on the input
side (token or word-family detectors); from about layer 20 many poles fire on the token *before* a
word their write promotes (they predict it), and at layers 32-35 some fire before a word and write
against it. Several sides naming the same thing are much stronger evidence than one.

For each card, output exactly one JSON object per line (JSON Lines), no prose outside it:

{"id": "L<layer> n<neuron> <+|−>",
 "labels": [{"label": "<plain words, ≤8 words>", "weight": <0-1>}, ...],
 "confidence": <0-1>,
 "sides": {"write": "<agrees|partial|uninformative|conflicts>", "read": "...", "activation": "...", "causal": "<...|absent>"},
 "relation": "<preserve|transform|suppress|unclear>",
 "outside": "<agrees|partial|disagrees|none>",
 "note": "<≤30 words: what the description rests on, and any disagreement>"}

Rules:
- Soft, not one hard label: 1-3 labels whose weights sum to at most 1. Leave weight unassigned
  when much is unexplained. A neuron can be a token-level detector ("fires on ' search'"), a
  concept ("comparison, contrast; English and Chinese"), a format or position feature, or
  a mix. The labels will be tested blind against held-out text where the pole fires and text
  where it stays quiet: say where it fires, specifically enough to tell it from other poles.
- English and Chinese tokens of the same concept are one concept.
- Do not force a word reading where a side is junk; mark that side "uninformative".
- `sides` says how each side relates to your top label (`read` covers the read lenses and the
  exact drivers). `relation` compares what it reads with what it writes: preserve (same concept),
  transform (related but different, for example "reads plant contexts, writes 'water'"), suppress
  (writes against what it reads), unclear.
- `confidence` is how well the evidence as a whole supports the labels: high only when
  several sides agree.
- Use only the card: do not look the neuron up anywhere else.
