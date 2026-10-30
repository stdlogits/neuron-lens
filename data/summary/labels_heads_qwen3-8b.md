Every head described by Claude Haiku 5.5 from its card (batch API, effort low, adaptive thinking):
`fires` (where its output is large), `attends` (what it attends to there) and `writes` (what its
output promotes), each with a confidence 0-1, and `note`. The cards show the head's pattern scores,
output scale, OV and QK reads, drivers and first 5 top contexts; nothing about its other top firings.
The prompt is `scripts/prompts/qwen3-8b/heads.md` on GitHub.

**Checked blind** on 100 random heads, 20 a band, by Claude Sonnet 5.5:

- **Firing:** given a head's `fires` and `attends` labels and 8 contexts, 4 of its held-out top firings (ranks 6-24) and 4 of another head of its band, the scorer separates them at AUROC 0.71 [0.65, 0.77]; a third head's labels on the same contexts score 0.51.
- **Writes:** given what the head's firings actually promote in context (`out_R_z`, not on the card), a judge picks its `writes` label out of four from its band 49% of the time [39, 60] (chance 25%).

| band | firing AUROC | control | writes picked |
|---|---|---|---|
| L0-L8 | 0.74 | 0.50 | 33% |
| L9-L12 | 0.70 | 0.61 | 56% |
| L13-L20 | 0.79 | 0.36 | 25% |
| L21-L27 | 0.60 | 0.56 | 55% |
| L28-L35 | 0.73 | 0.51 | 79% |
