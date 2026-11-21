Every head described by Claude Haiku 5.5 from its card (batch API, effort low, no thinking):
`fires` (where its output is large), `attends` (what it attends to there) and `writes` (what its
output promotes), each with a confidence 0-1, and `note`. The cards show the head's pattern scores,
how often it fires, one OV read for its depth, its QK reads and its drivers as query and key (24 tokens
each), and its first 5 top contexts; nothing about its other top firings.
The prompt is `scripts/prompts/heads.md` with `scripts/prompts/gemma-4-12b/heads.md` on GitHub.

**Checked blind** on 80 random heads, 20 a band, by Claude Sonnet 5.5:

- **Firing:** given a head's `fires` and `attends` labels and 8 contexts, 4 of its held-out top firings (ranks 6-24) and 4 of another head of its band, the scorer separates them at AUROC 0.78 [0.72, 0.84]; a third head's labels on the same contexts score 0.49.
- **Writes:** given what the head's firings actually promote in context (`out_R_z`, not on the card), a judge picks the head's `writes` label among four from its band: AUROC 0.55 [0.48, 0.62] for its own label against the other three (chance 0.5).

| band | firing AUROC | control | writes AUROC |
|---|---|---|---|
| association L0-L19 | 0.82 | 0.46 | 0.58 |
| workspace L20-L32 | 0.71 | 0.50 | 0.47 |
| motor L33-L47 | 0.75 | 0.55 | 0.50 |
