# Post-training's most changed neurons (parked)

Exploratory, Qwen3-8B against Qwen3-8B-Base, weights through the post-trained lenses; not in the findings. It extends
findings §19 (what post-training changed) from whole matrices to single neurons, and asks whether the change looks like
the "recursive generalization" of Cammarata's essay *Neural Networks Are Already Clean* (Goodfire, October 2026,
[already-clean.goodfire.com](https://already-clean.goodfire.com/)): a neuron grows by offshoots into territory adjacent
to its job, picked up because it was already closest to the new demand.

## Scripts

Run from the repository root with `PYTHONPATH=$PWD` on a Mac (MPS); both models' weights and catalogs (the base's
`acts`, `reads`, `drivers`) are needed. Outputs go anywhere; `data/movers/` is ignored.

- `stats.py OUT.pt`: per neuron, cosine, norms and distance between the two models' writes (metric at the layer's
  output) and gate and up rows (metric at the MLP input), and both models' `act_top` and `rate`.
- `reads.py STATS.pt OUT.json`: picks the most rotated live neurons (whole model, per band against the layer's median,
  most amplified and shrunk writes); reads the change through the R-lens (z against the layer's writes and against its
  changes) and gathers both catalogs' rows. `report-qwen3-8b.txt` is its report, `movers-qwen3-8b.json` the rows.
- `controls.py STATS.pt`: weight change against activity change, and the change reads of random neurons at their
  layer's median rotation (`controls-qwen3-8b.txt`).
- `related.py STATS.pt OUT.pt` and `summary.py STATS.pt RELATED.pt`: the population test below
  (`related-qwen3-8b.txt`).

## Results

- **Size.** Ranked naively, the top movers are dead neurons in L1-L3 (tiny weights). Live neurons (`act_top` ≥ 0.5 in
  both models) rotate their writes by a median 4.7° and at most 13° (L35/9346); write norms change by at most ×1.05; gate
  rows move least; write and up-row changes correlate (0.78). The change is close to perpendicular to the write and
  mostly outside the layer's top ΔW directions. The top 500 movers sit at L35 (194), L19-L22 and around them.
- **Before and after.** The write reads mostly keep their tokens. The late movers' changes read as the chat or
  reasoning form of the neuron's own job: L35/5000− however → But / Alternatively; L35/10474+ which → that / dass /
  Because / Wait; L34/3478− possible → therefore / 所以我们; L35/12276+ `=` → ` =`; L32/3613+'s change reads
  `</think>` / `<|im_end|>`.
- **The top mover**, L35/9346+, fires in the base where a post ends and site text begins ("share|improve this
  question", "Follow TV Tropes"; its read names terms-of-service boilerplate), and after post-training at
  `<|im_end|>⏎<|im_start|>` as well (chat rate 0.14‰ → 2.29‰, `act_top` 384 → 1,224): a boundary unit extended to turn
  boundaries. The `im` both peak lists share is the one in "improve"; `<|im_start|>`'s embedding is far from `im`'s.
  This reading rests on five contexts and no causal test.
- **Pruned offshoots.** L4/2468− and L6/9336− fired on `<|im_start|>` in the base (likely its undertrained embedding)
  and not after post-training.
- **Controls.** Random neurons at their layer's median rotation have change reads with no evident relation to their
  writes. Weight change does not predict activity change (correlation −0.04 over all live neurons, near zero in every
  band).
- **Population test** (`related.py`). The change perpendicular to the base write in the metric (Δ⊥; the parallel part is
  a slight shrink, which would make one end of the change read the neuron's own tokens by construction), read through
  the R-lens, against the neuron's own write read: how often the centroid of Δ⊥'s top 16 tokens (its nearer end) lies
  closer to the centroid of the write's top 16, in the centred unit unembedding, than to a permuted same-layer
  neuron's. Chance is 0.5:

  | | all live | top 1% by rotation |
  |---|---|---|
  | sensory (L0-L3) | 0.515 | 0.519 |
  | association (L4-L8) | 0.511 | 0.484 |
  | workspace (L9-L20) | 0.521 | 0.511 |
  | motor (L21-L35) | 0.557 | 0.630 |
  | L32 / L33 / L34 / L35 | 0.55 / 0.56 / 0.57 / 0.61 | 0.68 / 0.71 / 0.81 / 0.63 |

  Over the whole vocabulary the change's logits are uncorrelated with the write's (median 0.00). Both ends of Δ⊥ land
  near the neuron's own tokens (L35: 0.060 and 0.057 against 0.038 for the partner): the change tilts within a
  neighbourhood, promoting some of its tokens over others, rather than moving the write to another topic. Early writes
  read less legibly, so the test has little power there.

## Reading (synthesis)

Base to post-trained is one growth step under a new error signal, seen at two checkpoints. The head-start mechanism
predicts what the late movers show: the new demand (contrastive and hedged reasoning style, turn structure) lands on the
neuron already closest to it, and the main job stays. But on the write side this relatedness is close to forced: a
neuron's `W_down` gradient is the output error summed over the positions where it fires, so a neuron that keeps its
contexts tilts its write toward whatever the model now says there. The essay's claim is mostly about what neurons
respond to, and weight change does not predict activity change, so the informative test is on the firing side.

## Next, if resumed

1. **Firing side, 8B.** Rank poles by the change in their firings between the two catalogs (rates on the chat slice,
   peak-token and context overlap), then rate blind whether each pole's new contexts are adjacent to its old ones, against
   a shuffled control; rank-scored, no abstain option, at maximum effort.
2. **Order.** Two checkpoints show no order of growth. Pythia's checkpoints can (the lenses, the catalog port and a
   weight-diff reader already run on it): follow poles across checkpoints and ask whether offshoots trace back to one
   origin and appear in sequence, as the essay predicts.
3. **Replication** on Qwen3-4B against its base, which shares the 8B's lineage (findings §20), so it is a partial
   replication only.
