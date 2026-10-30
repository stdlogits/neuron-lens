You are describing attention heads of Qwen3-8B (36 layers, 32 query heads each, which share 8 key-value heads) from evidence cards. An attention head looks back from each position (the query) to earlier positions (the keys), takes a weighted mix of what it finds there, and writes it into the residual stream. For each head, say three things:

- **fires:** where in text the head's output is large: the query tokens or contexts where it acts.
- **attends:** what it looks back at when it fires: a token, a word family, the previous token, an earlier copy of the current token, the start of the text, a matching bracket or quote...
- **writes:** what its output pushes the model toward: which words or tokens become more likely, or which tokens it copies forward.

## The card

- **Attention pattern:** the head's mean attention to position 0 (the sink, where heads park when they have nothing to do), to the previous token and to itself; and, on random text said twice, from the second copy to the token after the earlier occurrence (induction: high means it copies the continuation of a repeat) and to the earlier occurrence itself (duplicate: high means it looks for earlier copies of the current token). Values near 1 are strong; most heads are low on all of them.
- **Output:** its typical high output size, against the layer's other heads, and how often it is past a quarter of that.
- **OV reads** (weight-only): for each token the head could attend to, where that token ranks in what the head writes when it reads it.
  - **OV·R:** in what its output makes the model say. "Copies most" are tokens it passes on (reading `cat` makes the model say `cat`); "suppresses most" are tokens it pushes down when it reads them (copy suppression).
  - **OV·S** (shown through layer 8, where it is the better read): whether the output makes the residual look as if the token were present.
  - The copying score is +1 for a head that copies everything it reads, −1 for one that suppresses everything.
  - "Against the layer's heads, also" lists what is specific to this head, beyond the tokens already shown.
- **QK reads** (weight-only): which key tokens the head favours under its average query. They read the content a head looks for, not position: a previous-token or induction head shows little here.
  - **Self-seeking:** tokens that, as the query, favour the same token as the key. A high median means a duplicate-token head.
  - **Attends to what it copies:** high when the tokens it attends to are the ones it copies.
- **Drivers in real text:** tokens swept through real documents. As the query: the tokens at which its output is largest. As the key: the tokens it attends to most from the 16 positions after them.
- **Contexts:** its 5 strongest firings, the query token in ⟦ ⟧ and the token it attended to most in ⟪ ⟫ (when it attended to the query itself, only ⟦ ⟧ shows). "…" marks a gap when the attended token is far back.

Tokens come in rank tiers, 1-8 ‖ 9-24 ‖ 25-64; variants of one word are merged as `word×count`. A theme spread across many tokens counts for more than one token at the top. English and Chinese tokens of one concept are one concept.

## How to read it

- **The contexts are the best guide to where a head fires and what it attends to.** Look at what the ⟦ ⟧ tokens share, and at how the ⟪ ⟫ token relates to them: the same token earlier (duplicate), the token after an earlier copy (induction), the previous token, a name, an opening bracket or quote, the subject of the sentence, a keyword in code.
- **The pattern scores name the classic heads.** Previous-token, induction and duplicate-token heads show it in their scores; sink-heavy heads attend to position 0 most of the time and act only at some positions.
- **The OV reads are the best guide to what a head writes.** If it copies most of what it reads, "writes" is "copies the attended token" (say what kind of token, from the contexts and drivers). If its copy list has a theme, name it.
- **The depth line says which reads to trust.** Layers 13-20 are the weakest band for every read: say unclear freely there.
- Heads in the same layer that share a key-value head attend from the same keys but can do different things.

## Output

Exactly one JSON object per head, one per line (JSON Lines), nothing else:
{"id": "<the head's id from its ## line, e.g. L20h17>", "fires": {"label": "<≤12 words>", "confidence": <0-1>}, "attends": {"label": "<≤12 words>", "confidence": <0-1>}, "writes": {"label": "<≤12 words>", "confidence": <0-1>}, "note": "<≤25 words: what each label rests on>"}

Rules:
- The labels will be tested blind. "Fires" and "attends" are tested on other firings of the head, next to another head's firings. "Writes" is tested against what the head's output promotes in real text, next to other heads' labels. Be specific enough to tell this head from others: name concrete tokens, words, topics or structures where the evidence supports them.
- Always give your best guess, even on weak evidence, with a confidence that says how weak it is (0.1: a guess; 0.9: several lines agree).
- Use only the card: do not look the head up anywhere else.
