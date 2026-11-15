{about} Each card is one neuron with both its poles: `+` is where the activation is positive, `−` where it is negative. Describe each pole separately, and say two things:

- **fires:** where in text the pole fires (activates strongly): a token, a word family, a concept, a format, or the context just before some word.
- **writes:** what the pole's write pushes the model toward when it is added to the residual stream, amplified: which words, topics, languages or styles become more likely in the text that follows.

## What a card shows

For each pole:
- **an input read** (`in·…`): tokens the weights say drive the pole, through the lens that reads inputs best at the neuron's layer (the guide below);
- **a write read** (`out·…`): what the pole's push does, through the lens that reads writes best there;
- when there are any: the tokens the other lens's read **agrees on** (`in·S agrees on`, `out·S agrees on`, and so on), and the tokens **shared by both sides**;
- **how often it fires** (past a quarter of its typical high activation), with the corpus slices where it fires at least twice as often or under half as often (dclm: English web, zh: Chinese web, code: Python, chat: chat transcripts);
- its **top 5 contexts:** the token where it peaked in ⟦ ⟧, with 12 tokens before and 3 after.

The reads, all computed from the weights alone:
- **in·S** (S-lens): input tokens, at this position or a few before, whose presence drives the pole. It is literal: the tokens themselves.
- **in·R** (R-lens on the input direction): what the model is getting ready to say when the pole is driven: the concept the neuron responds to, and in late layers often the word coming next.
- **out·R** (R-lens on the output): what the pole's push makes the model say.
- **out·S** (S-lens on the output): the push looks as if these input tokens were present; early neurons often write back what they detect.
- **z** versions compare each token's score with the layer's other neurons: what is specific to this pole, without the hub tokens that top every neuron's list. They name concepts more than literal tokens.

A read is a ranked list of up to 24 tokens in two tiers, 1-8 ‖ 9-24. Variants of one word are merged as `word×count`, and `⏎×count` merges runs of line breaks.

## Which read the card shows, and how far it goes

{depths}

## How to read a card

- **Look for the theme, not the exact tokens.** It can be a word family, a grammatical form (gerunds, adverbs, past tense), a topic, a script or language, or a format (code, numbers, punctuation). English and Chinese words for one concept are one concept. A theme that runs through many tokens and both tiers counts for more than one token at the top.
- **Agreement is the strongest evidence.** The other lens naming the same tokens means the concept is real; tokens shared by both sides mean the neuron reads and writes the same thing.
- **Use each side for what it is good at.** The contexts are the best guide to where a pole fires; the write read to what it promotes when amplified. Where they agree, be confident; where they disagree, say in the note which side each label rests on.
- **The firing line** says how often the pole fires and whether one corpus dominates: `zh 6×` points to Chinese text, `code 4×` to Python, even when the five contexts look mixed.
- **A faint shadow** is a pole much weaker than its neuron's other pole, sharing its up row, so its card has no input read. Take where it fires from its contexts, and say in the note that it is a shadow.
- **The two poles of a neuron write opposite directions,** and they often read different things. Compare them: what one pole has and the other lacks is usually its meaning.
- **A near-silent pole** has no contexts: still give your best guess.

{examples}

## Output

Exactly one JSON object per pole (two per card), one per line (JSON Lines), nothing else:

{"id": "<L, n and the pole: the ## line's layer and neuron, then + or −, e.g. L12 n345 −>", "fires": {"label": "<≤12 words>", "confidence": <0-1>}, "writes": {"label": "<≤12 words>", "confidence": <0-1>}, "note": "<≤25 words: what each label rests on>"}

Rules:
- Be specific enough to tell this pole from others: name concrete tokens, words, topics or languages where the evidence supports them.
- English and Chinese tokens of the same concept are one concept; name both if both show.
- Always give your best guess, even on weak evidence. Confidence is how clearly the card shows one theme for that label: 0.1 for a guess, 0.9 when the reads and the contexts point the same way.
- Use only the card: do not look the neuron up anywhere else.
