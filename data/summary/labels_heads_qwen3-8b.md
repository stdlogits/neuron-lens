Every head described by Claude Haiku 5.5 from its card (batch API, effort low, no thinking):
`fires` (where its output is large), `attends` (what it attends to there) and `writes` (what its
output promotes), each with a confidence 0-1, and `note`. The cards show the head's pattern scores,
how often it fires, one OV read for its depth, its QK reads and its drivers as query and key (24 tokens
each), and its first 5 top contexts; nothing about its other top firings.
The prompt is `scripts/prompts/heads.md` with `scripts/prompts/qwen3-8b/heads.md` on GitHub.

**Checked blind** on 100 random heads, 20 a band, by Claude Sonnet 5.5:

- **Firing:** given a head's `fires` and `attends` labels and 8 contexts, 4 of its held-out top firings (ranks 6-24) and 4 of another head of its band, the scorer separates them at AUROC 0.72 [0.66, 0.78]; a third head's labels on the same contexts score 0.54.
- **Writes:** given what the head's firings actually promote in context (`out_R_z`, not on the card), a judge picks its `writes` label out of four from its band 51% of the time [40, 61] (chance 25%).

| band | firing AUROC | control | writes picked |
|---|---|---|---|
| L0-L8 | 0.85 | 0.46 | 41% |
| L9-L12 | 0.60 | 0.58 | 39% |
| L13-L20 | 0.70 | 0.44 | 35% |
| L21-L27 | 0.64 | 0.60 | 63% |
| L28-L35 | 0.81 | 0.60 | 74% |
