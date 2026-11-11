Every head described by Claude Haiku 5.5 from its card (batch API, effort low, adaptive thinking):
`fires` (where its output is large), `attends` (what it attends to there) and `writes` (what its
output promotes), each with a confidence 0-1, and `note`. The cards show the head's pattern scores,
output scale, OV and QK reads, drivers and first 5 top contexts; nothing about its other top firings.
The prompt is `scripts/prompts/gemma-4-12b/heads.md` on GitHub.

**Checked blind** on 80 random heads, 20 a band, by Claude Sonnet 5.5:

- **Firing:** given a head's `fires` and `attends` labels and 8 contexts, 4 of its held-out top firings (ranks 6-24) and 4 of another head of its band, the scorer separates them at AUROC 0.71 [0.64, 0.78]; a third head's labels on the same contexts score 0.49.
- **Writes:** given what the head's firings actually promote in context (`out_R_z`, not on the card), a judge picks its `writes` label out of four from its band 32% of the time [22, 42] (chance 25%).

| band | firing AUROC | control | writes picked |
|---|---|---|---|
| L0-L4 | 0.74 | 0.48 | 25% |
| L5-L13 | 0.74 | 0.54 | 35% |
| L14-L32 | 0.70 | 0.47 | 39% |
| L33-L47 | 0.67 | 0.47 | 30% |
