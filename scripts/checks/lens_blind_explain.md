You are describing MLP neurons of Qwen3-8B (36 layers, 12,288 neurons each) from evidence cards.
Each card is one pole of one neuron: `+` is where the activation `silu(gate)·up` is positive, `−`
where it is negative. They are described separately. For each pole, say two things:

- **fires:** where in text the pole fires (activates strongly): a token, a word family, a concept,
  a format, or the context just before some word.
- **writes:** what the pole's write pushes the model toward when it is added to the residual
  stream, amplified: which words, topics, languages or styles become more likely in the text that
  follows.

All the cards in your file show the same kind of evidence: lens reads, contexts, or both.

## Lens cards (weight-only reads, no activations)

Each line is the top 12 tokens of one read, best first. `|` separates tokens, a leading space is
part of the token, ⏎ is a newline, and � is a byte fragment of a multi-byte character.
- **writes (R-lens):** tokens the write promotes in the model's later predictions. From about
  layer 5, amplifying the write raises these tokens for nearly every pole.
- **writes, as input tokens (S-lens of the write):** input tokens whose own footprint at this
  layer the write resembles. Through about layer 8, neurons often write back the tokens they
  detect.
- **reads (forward R-lens of the read direction):** what the pole responds to, as the model's own
  predictions name it: a concept-level view, most legible from about layer 9.
- **reads, as input tokens (S-lens):** input tokens, at this position or earlier, that drive the
  pole. Literal and legible at layers 0-8; past layer 8 mostly noise.
- **Layer-contrastive lines (z)** remove what every neuron of the layer shares. Prefer them when the
  plain line is generic (punctuation, common words, byte fragments).

## Context cards (activations only, no lens reads)

- The pole's typical high activation, how often it fires, and its firing rate in each corpus slice
  (dclm: English web, zh: Chinese web, code: Python, chat: chat transcripts).
- Its top 5 contexts: the token where it peaked in ⟦ ⟧, with 12 tokens before and 3 after.
- For "writes", infer from the contexts. At late layers a pole often promotes the word that tends to
  follow its peak; earlier, it often promotes the concept of its contexts.
- A near-silent pole has no contexts: still guess, with low confidence.

## Cards with both

A card may show the lens reads and then the contexts. On this model, earlier blind tests found:
- the top contexts the better guide to where a pole fires;
- the R-lens write read the better guide to what it promotes when amplified, except at layers
  0-8, where neither side predicted it well.

Use each side for what it is good at. Where the sides agree, be confident. Where they disagree,
say in the note which side each label rests on.

## Known patterns (from earlier measurements on this model)

- Layers 0-8: meaning is mostly on the input side (token or word-family detectors).
- Layers 13-20: input-side weight reads are weakest.
- From about layer 20, many poles fire on the token *before* a word their write promotes: they
  predict it. "Fires" should then name the context where that word is likely next, for example
  "where 'boil' is likely next ('bring the water to a')". At layers 32-35 some poles fire before a
  word and write *against* it.
- Several reads or contexts naming the same thing are much stronger evidence than one.

## Output

Exactly one JSON object per card, one per line (JSON Lines), nothing else:

{"id": "L<layer> n<neuron> <+|−>", "fires": {"label": "<≤12 words>", "confidence": <0-1>}, "writes": {"label": "<≤12 words>", "confidence": <0-1>}, "note": "<≤25 words: what each label rests on>"}

Rules:
- Both labels will be tested blind. "Fires" is tested against real text where the pole fires and
  text where it stays quiet. "Writes" is tested against text generated with the write amplified,
  next to other poles' labels. Be specific enough to tell this pole from others: name concrete
  tokens, words, topics or languages where the evidence supports them.
- English and Chinese tokens of the same concept are one concept; name both if both show.
- Always give your best guess, even on weak evidence, with a confidence that says how weak it is
  (0.1: a guess; 0.9: several lines agree).
- Use only the card: do not look the neuron up anywhere else.
