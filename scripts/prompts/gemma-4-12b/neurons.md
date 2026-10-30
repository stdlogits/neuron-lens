You are describing MLP neurons of Gemma 4 12B, a base model (48 layers, 15,360 neurons each), from evidence cards. Each card is one neuron with both its poles: `+` is where the activation `gelu_tanh(gate)·up` is positive, `−` where it is negative. Describe each pole separately, and say two things:

- **fires:** where in text the pole fires (activates strongly): a token, a word family, a concept, a format, or the context just before some word.
- **writes:** what the pole's write pushes the model toward when it is added to the residual stream, amplified: which words, topics, languages or styles become more likely in the text that follows.

## Lens reads (weight-only, no activations)

Every read is a ranked list of up to 64 tokens computed from the weights through a fitted lens:
- **"adds"** lines: the z read of a lens shows only what its raw read lacks (or the reverse). Variants of one word are merged as `word×count`.
- **in·S** (S-lens): input tokens, at this position or a few before, whose presence drives the pole. It is literal: the tokens themselves.
- **in·S z**: the same scores, each compared with every other neuron of the layer, so what is specific to this pole. It names concepts more than literal tokens.
- **in·R** (R-lens on the input direction): what the model is getting ready to say when the pole is driven. It is the concept the neuron responds to; in late layers, often the word coming next.
- **out·R** (R-lens on the output): what the pole's push makes the model say.
- **out·S** (S-lens on the output): the push looks as if these input tokens were present.
- **z** versions in general: compared across the layer, so hub tokens that top every neuron's list are gone.


## How to read the lens reads
- **The depth lines tell you which reads work at this layer.** The card only shows those, primary first. Early layers read literal tokens (in·S): strongly at layers 0-4, fading by layer 9 and returning at layers 10-13. On this model in·R rarely holds as an input read. Layers 14-32 are weak for every read. From layer 33, out·R often names the next word.
- **Strength** is how strong this read is against the layer's other poles. Below about the 20th percentile, treat a read as noise unless another read agrees with it.
- **Tokens come in rank tiers:** 1-8 ‖ 9-24 ‖ 25-64. Signal fades with rank but lasts to 64. A theme spread across many tokens and across tiers counts for more than one token at the top.
- **"Shared by" lines are the strongest evidence.** Two different lenses naming the same token means the concept is real. Tokens shared by the input and output sides mean the neuron reads and writes the same thing.
- **Look for the theme, not the exact tokens.** It can be a word family, a grammatical form (gerunds, adverbs, past tense), a topic, a script or language, or a format (code, numbers, punctuation). English and Chinese words for one concept are one concept.
- **The two poles of a neuron write opposite directions,** and they often read different things. Compare them: what one pole has and the other lacks is usually its meaning.


For a pole whose reads support no theme, still give your best guess, with low confidence (rules below).

Each pole's lens reads are followed by its activation line and top contexts.

## Context cards (activations only, no lens reads)

- The pole's typical high activation, how often it fires, and its firing rate in each corpus slice
  (dclm: English web, zh: Chinese web, code: Python, chat: chat transcripts as plain `User:` / `Assistant:` turns).
- Its top 5 contexts: the token where it peaked in ⟦ ⟧, with 12 tokens before and 3 after.
- For "writes", infer from the contexts. At late layers a pole often promotes the word that tends to
  follow its peak; earlier, it often promotes the concept of its contexts.
- A near-silent pole has no contexts: still guess, with low confidence.

## Cards with both

A card may show the lens reads and then the contexts. On a sibling model, Qwen3-8B, blind tests found:
- the top contexts the better guide to where a pole fires;
- the R-lens write read the better guide to what it promotes when amplified, except at its early
  layers, where neither side predicted it well. On this model out·R predicts an amplified push
  from layer 11, not before.

Use each side for what it is good at. Where the sides agree, be confident. Where they disagree,
say in the note which side each label rests on.


## Worked examples. These are real neurons, not in your sample, whose meaning has been checked against their activations. Excerpts show the top tiers only.

1. **Layer 1, − pole.**
   - in·S (99th percentile): ` ikke |  også | 运用 (apply) | 可以说是 (one could say) |  sammen |  på | Lille×2 |  Aarhus ‖  teknik |  slik |  fordi |  eller |  hvordan |  forskellige`.
   - in·S z adds (99th): ` Faroe ‖  Stav |  Scandinavia |  sø | Norway | gaard`.
   - out·S z (93rd): ` Mest×2 |  bru×4 | kv×2 |  Krit |  Gj ‖ kva | benhav | ighed |  Kan`.
   - out·R z (95th): ` Optimization |  Histogram×2 |  offenses×2 | 限于 (limited to) |  mesmerizing`, no theme.
   - Contexts: "Politik⟦ og⟧ Administration", "Ingen⟦ kom⟧mentarer".

   → fires: "Danish and Norwegian words (og, ikke, også, sammen)", 0.9; writes: "Danish and Norwegian word pieces, as if present", 0.5.
   Two Chinese tokens in the top tier are noise beside a clear theme. This early, out·S passes the input on, and out·R is junk even at a high percentile.

2. **Layer 1, + pole** (a weak read, clear contexts).
   - in·S (24th): ` schizophren |  Lemma | 这样做 (do this) |  knn×2 |  conceptual×2 ‖  TIMESTAMP |  custodial | … |  volcanoes`.
   - out·R z (27th) and out·S z (46th): no theme.
   - All five contexts: "enormous⟦ volcanoes⟧", "many⟦ volcanoes⟧ are mountainous".

   → fires: "the word volcanoes", 0.85; writes: "unclear", 0.1.
   The read is weak and names volcanoes only in its second tier, but five contexts agree: trust them for "fires".

3. **Layer 12, + pole.**
   - in·S z (98th): `扫描 (scan) |  Scanning×5 | Scanner×5 | SCAN×6 ‖ sear |  browse×4 |  searches×7 |  PDFs`.
   - out·R z adds (82nd): ` Scan×5 |  scanned×3 ‖ FOV | 扫描 | … |  OCR |  printers`.
   - Shared by the input and output sides: scan, scanner, 扫描.
   - Contexts: Java's "⟦Scanner⟧ input = new", "import java.util.⟦Scanner⟧;".

   → fires: "Scanner and scan tokens, often Java's Scanner class", 0.9; writes: "scan, scanned and scanners (扫描)", 0.75.
   At layers 11-13 many neurons write back the tokens they detect; the shared line shows it.

4. **Layer 26, + pole** (the weak band, but these reads are strong).
   - in·R z (100th): `shaft×2 | rivers | stream |  unobstructed | point×2 | heading ‖ ostream | reach |  canal×3 | west×2 |  direc |  meridian`.
   - out·R (92nd): ` directing×11 |  direction×9 |  redirection |  redirected×6 ‖  diverted×2 |  directional×2 |  diversion | 方向 (direction)×2 |  path×3`.
   - Contexts: "and⟦ direct⟧ existing supplies", "to⟦ direct⟧ your search", "would⟦ steer⟧ more money to".

   → fires: "the verb direct and its kin (steer, Direction)", 0.8; writes: "direction, redirection and diversion (方向)", 0.85.
   in·R z reads flow and heading, the concept around the word; with out·R and the contexts it points one way. A read past the 90th percentile is real even in layers 14-32.

5. **Layer 33, − pole.**
   - in·S z (31st) and in·R (3rd): no theme.
   - out·R (81st): ` process×6 | 过程 (process) | 過程 |  proceso | 过程中 (in the process) ‖  processus |  processo |  doing×5 |  processes×4 |  proses`.
   - All five contexts: "amassing fortunes in⟦ the⟧ process", "build muscle in⟦ the⟧ process".

   → fires: "'the' in 'in the process'", 0.9; writes: "the word process, in any language", 0.9.
   Late, a pole often fires on the cue for the word it writes: the contexts show the cue, out·R names the word, and the input reads say nothing.

6. **Layer 22, + pole** (a trap).
   - in·R z (99th): ` singled |  newcomers×2 |  parades×2 | comers | events×3 |  processions ‖  poured×2 |  flocked |  greeted | saint×2 |  feast×2 |  crowned`.
   - in·S z (12th) and out·R (39th): ` 记忆 (memory) | IntVar×2 | MEMORY` and ` gubernatorial | 程序员 (programmer) |  strateg×2`, no theme.
   - Contexts: "Trends in Pharmacological Sciences.⟦ ⟧40,", "Journal of Alloys and Compounds,⟦ ⟧867", "Volume:⟦ ⟧7;".

   → fires: "the space before a volume number in a reference", 0.7; writes: "unclear", 0.1.
   in·R z sits at the 99th percentile and tells a vivid story of parades and festivals, but every context is a citation. In layers 14-32, trust the contexts over any read.

## Output

Exactly one JSON object per pole (two per card), one per line (JSON Lines), nothing else:

{"id": "<the pole's id from its ### line, e.g. L12 n345 −>", "fires": {"label": "<≤12 words>", "confidence": <0-1>}, "writes": {"label": "<≤12 words>", "confidence": <0-1>}, "note": "<≤25 words: what each label rests on>"}

Rules:
- Both labels will be tested blind. "Fires" is tested against real text where the pole fires and
  text where it stays quiet. "Writes" is tested against text generated with the write amplified,
  next to other poles' labels. Be specific enough to tell this pole from others: name concrete
  tokens, words, topics or languages where the evidence supports them.
- English and Chinese tokens of the same concept are one concept; name both if both show.
- Always give your best guess, even on weak evidence, with a confidence that says how weak it is
  (0.1: a guess; 0.9: several lines agree).
- Use only the card: do not look the neuron up anywhere else.
