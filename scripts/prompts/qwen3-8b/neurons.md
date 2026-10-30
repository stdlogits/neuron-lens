You are describing MLP neurons of Qwen3-8B (36 layers, 12,288 neurons each) from evidence cards. Each card is one neuron with both its poles: `+` is where the activation `silu(gate)·up` is positive, `−` where it is negative. Describe each pole separately, and say two things:

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
- **The depth lines tell you which reads work at this layer.** The card only shows those, primary first. Early layers read literal tokens (in·S); from layer 9 the input read is in·R. Layers 13-20 are weak for every read.
- **Strength** is how strong this read is against the layer's other poles. Below about the 20th percentile, treat a read as noise unless another read agrees with it.
- **Tokens come in rank tiers:** 1-8 ‖ 9-24 ‖ 25-64. Signal fades with rank but lasts to 64. A theme spread across many tokens and across tiers counts for more than one token at the top.
- **"Shared by" lines are the strongest evidence.** Two different lenses naming the same token means the concept is real. Tokens shared by the input and output sides mean the neuron reads and writes the same thing.
- **Look for the theme, not the exact tokens.** It can be a word family, a grammatical form (gerunds, adverbs, past tense), a topic, a script or language, or a format (code, numbers, punctuation). English and Chinese words for one concept are one concept.
- **The two poles of a neuron write opposite directions,** and they often read different things. Compare them: what one pole has and the other lacks is usually its meaning.


For a pole whose reads support no theme, still give your best guess, with low confidence (rules below).

Each pole's lens reads are followed by its activation line and top contexts.

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


## Worked examples. These are real neurons, not in your sample, whose meaning has been checked against their activations. Excerpts show the top tiers only.

1. **Layer 0, − pole.**
   - in·S (80th percentile): `ain×3 | ains×2 | lain×2 | refrain | aining | rain×6 ‖ Brain×4 | attain×2 | strain×3 | stained×4 | drain×4 | contain×2`.
   - out·R z (40th): `NING×2 | etwork | credible | vest×2 | ufact | ners`, no theme.
   - out·S z (95th): `AIN×4 | ains×2 | aining | lain×2 | strain×4 | retain×3`.

   → fires: "words containing 'ain' (rain, strain, contain)", 0.9; writes: "'ain' words, as if present", 0.6.
   The early write is junk in out·R, but out·S shows the layer passing its input on.

2. **Layer 8, + pole.**
   - in·S (78th): `owl×2 | Where | fright ‖ Peek×2`.
   - in·R (83rd): `(= | owl×2 | 星空 (starry sky) | ow×3 ‖ witch | Birds | nest | winter×2`.
   - out·R (93rd): `owl×3 | noct | cosy | bedtime | autumn | night×4 ‖ 夜间 (night) | nighttime | sleepy`.

   → fires: "owls", 0.8; writes: "owls, night and bedtime", 0.8.
   in·S and in·R agree on owl, so trust it despite the noise around it.

3. **Layer 16, + pole** (the weak band, but this read is strong).
   - in·R z (99th): `故事 (story) | tale×4 | .story×9 ‖ stories×4 | 情节 (plot) | 结局 (ending) | 讲故事 (storytelling) | storyline | saga`.
   - out·R (100th): `story×9 | stories×4 | tale×4 | narrative×3`.
   - Shared by the input and output sides: story, tale, narrative, saga, protagonist.

   → fires: "storytelling and fairy-tale narration", 0.75; writes: "story, tale and narrative words", 0.85.
   A read at the 99th percentile is real even in layers 13-20.

4. **Layer 27, + pole.**
   - in·R (91st): `候选人 (candidate) | 竞选 (run for office) | candidates×11 | candidacy | 选举 (election) ‖ ballot×2 | vote×9 | elections×6`.
   - in·S z: `Elections | referendum | Republican`.
   - out·R (91st): `候选人 | candidates×11 ‖ nominees | nominee | election×6 | incumbent | contenders×2`.

   → fires: "just before election-candidate words (candidate, 候选人)", 0.9; writes: "candidates and nominees", 0.9.
   At late layers the input read often names the word coming next: the pole fires on its cue.

5. **Layer 30, − pole.**
   - in·R (87th): `绿色 (green) | green×16 | 绿 | 綠 ‖ verde×4 | :green | 绿化 (greening)`.
   - out·R (86th): `green×16 | 绿 | 绿色 ‖ verde×4 | leaf×4 | chlor×3 | foliage`.

   → fires: "just before the word green (绿色)", 0.9; writes: "green in any language", 0.9.
   The + pole of this neuron reads green too, but its out reads sit at the 6th-9th percentile with no theme, so its writes label gets a low confidence (about 0.1).

6. **Layer 11, + pole** (a trap).
   - in·R (47th): `EVERY×2 | everyone | again | aside | 批发市场 (wholesale market) | children | everything ‖ individual×3 | families`.
   - out·R (78th): `个体 (individual) | 成分 (component) | 粒子 (particle) | entities×4 | constituents×2 | inhabitants ‖ residents×3 | particles×2`.

   → fires: "each and every individual", 0.2; writes: "members, components, constituents", 0.6.
   It really fires on dental caries ("car⟦ies⟧", "cav⟦ities⟧"), which no read shows. A middling read with a loose theme deserves low confidence.

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
