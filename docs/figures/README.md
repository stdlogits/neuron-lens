# Figures

Drawn by `scripts/11_figures.py`. Figures 1-17 are Qwen/Qwen3-8B's: 1 and 2 are schematics, the others draw `data/summary/`, the released lenses' provenance and the catalog's `heads/legible` table. Figure 18 sets it beside google/gemma-4-12B, and 19-24 are Gemma's, from `data/summary/` and its catalog's media and head tables; no image or sound from a dataset is drawn, only lens reads and optimised patterns. Each figure's footer names its findings section and source.

## 1. What attention heads and MLP neurons do

![What attention heads and MLP neurons do](01_anatomy.png)

One layer acting on five positions of "she heard the owl and". At "and", an attention head's QK circuit weighs the earlier positions (arc widths) and its OV circuit copies what it reads from "she" into the current position. In the MLP, the same neuron sits at every position; it reads the stream along one direction, its activation `silu(gate)·up` takes either sign (the `+` and `−` poles), and it adds its write `W_down[:, n]`, scaled by that activation, back to the stream. Here it fires only at "owl" (L8 n344 +, which writes owl, nocturnal and bedtime words). Below each unit: what the catalog records for every head and every neuron pole.

## 2. The lenses: reading a neuron forward to the output and back to the input

![The lenses: reading a neuron forward to the output and back to the input](02_lenses.png)

The residual stream runs from the input tokens through the embedding `W_E`, layers L0-L35, the final norm and the unembedding `W_U` to next-token logits. A neuron at layer L reads the MLP input along `â ± b̂` (one direction per pole) and writes `W_down[:, n]` back. The logit lens reads a direction `h` at layer L as `W_U · norm(h)`; the J-lens and R-lens first carry it through `J_L`, the corpus-mean Jacobian of the final residual with respect to layer L (true gradients and RelP rules respectively); the K-lens and S-lens take the Jacobian from the input embeddings up to layer L instead (`K_L`, true gradients and RelP rules) and read with `W_E`.

## 3. Each side of a neuron reads at its own depth: inputs early, outputs late, a dark middle

![Each side of a neuron reads at its own depth: inputs early, outputs late, a dark middle](03_depth_map.png)

The input side reads early: the S-lens names the firing token for 78% of L0's poles and 14-58% at L1-L8, and almost none at L10-L20. The write side reads late: the R-lens write predicts the next token after a firing for at most 2% of poles before L23, rising to 35-37% at L28-L30. In between, 77-86% of L13-L20's poles are dark. Amplified, the write side reads nearly everywhere: from L5, 92-100% of random poles' writes predict the effect of steering them.

## 4. The lenses are stable: fits on two halves of the corpus differ by under 6%

![The lenses are stable: fits on two halves of the corpus differ by under 6%](04_lens_reliability.png)

Split-half difference of each lens target by layer: the R-lens from each layer's output (reads MLP writes) and from each MLP input (reads read directions), and the S-lens to each MLP input and to each layer's output. All stay under 6%; the R-lens is least stable around L7, the S-lens from L16-L17.

## 5. An R-lens fitted on chat predicts steering as well as the web-text lens

![An R-lens fitted on chat predicts steering as well as the web-text lens](05_chat_lens.png)

Median r with 1× steering for the chat-fitted lens and the DCLM lens's two halves: within 0.04 at every layer (the chat lens is lowest against them at L12-L18), though its top tokens differ (findings §1).

## 6. Early poles are quiet or work at the first token; from L5 every pole fires

![Early poles are quiet or work at the first token; from L5 every pole fires](06_activations.png)

Left: median act_top by layer (log scale), flat at 3-5 from L5 to L20 and rising tenfold to 51 at L35. Right: 56-61% of L1-L2's poles are quiet (42% at L3), and 57-70% of L2-L3's fire harder at the first position than anywhere in the corpus; from L5 nearly every pole fires.

## 7. A neuron's write, read through the R-lens, predicts what steering it does

![A neuron's write, read through the R-lens, predicts what steering it does](07_steering.png)

Median r between each lens's write read and the measured effect, by layer. Left, steering at 1× the pole's typical activation: the R-lens reaches 0.46-0.51 at L4-L8 and 0.9 or more from L22, leads the J-lens by up to 0.12 from L4 to L23, and the logit lens trails until about L20 (L8: 0.51 against 0.12). Right, ablating the pole where it fires: far lower before L21, partly because this ran in bf16 and partly because a firing's effect depends on its context (figure 9).

## 8. Amplified, nearly every write does what the lens says; in context, few do

![Amplified, nearly every write does what the lens says; in context, few do](08_amplified_vs_in_context.png)

From L5, 92-100% of random poles' writes predict their steering effect; before L22, at most 4% of poles predict what follows their own firings, rising to 31-32% at L28-L29. The gap is the difference between a write's average push and a firing's effect in its context (figure 9).

## 9. A firing's effect is linear but depends on context; the lens reads its average

![A firing's effect is linear but depends on context; the lens reads its average](09_context_linearity.png)

Left: ablation matches the linearized effect (r 0.78-1.00 by layer) except at L1-L2, whose poles barely fire and sit at the precision floor; before L20 one firing's effect barely predicts the others' (r 0.02-0.14). Right: the lens predicts steering (both average over contexts) far better than either predicts a single firing.

## 10. Writes predict the next token late; reads name the firing token early

![Writes predict the next token late; reads name the firing token early](10_side_agreement.png)

Left: prediction rises from L20 and peaks at 7.3% at L29-L30; suppression overtakes it at L35 (early layers' writes match the control: hub tokens). Right: the S-lens names the firing token at L0-L8 (40% at L0), with a spike at L8; the R-lens read from the MLP input takes over from L9.

## 11. The S-lens names a neuron's real input tokens early, the R-lens from L8

![The S-lens names a neuron's real input tokens early, the R-lens from L8](11_drivers.png)

At L0, 31% of the S-lens read's top 10 are exact drivers, and 4-11% at L1-L4; the R-lens read of the input direction leads from L8 (5-8% at L8-L12 and L22-L29). L13-L20 is the weakest stretch for every weight-only input read (R-lens 2-4%, S-lens under 1%).

## 12. Most mid-network poles are dark: they fire, but no token-level test reads them

![Most mid-network poles are dark: they fire, but no token-level test reads them](12_census.png)

Shares of each layer's poles: some weight-only read passes, only a token detector, dark (fires but nothing passes), or silent. Over all layers 22% of poles have a passing read and 57% are dark; dark averages 82% of L13-L20's poles.

## 13. A read's signal holds down to rank 64, not just its top tokens

![A read's signal holds down to rank 64, not just its top tokens](13_rank_depth.png)

Signal fades with rank but ranks 49-64 still run several times above a mismatched pole (L28's write read 30×, L12's read 6×), which is why the catalog keeps the top 64 of every read. L4's S-lens ratio is noisy: its mismatched rate is near zero.

## 14. Labels written blind: lens reads predict what a pole promotes, contexts where it fires

![Labels written blind: lens reads predict what a pole promotes, contexts where it fires](14_lens_blind.png)

Left: mean detection AUROC of each arm's "fires" label on 6 held-out top contexts against 6 quiet positions (half from the same documents): lens reads alone 0.80, contexts 0.87, both 0.89; dashed, another pole's label (0.58). Right: how often a judge picks the pole's own "promotes" label out of four from its band, from greedy text steered at 0.5-2 nats: lens reads 65% (95% at L28-L35), contexts 54%; dashed, chance. Bars: 95% bootstrap intervals over poles.

## 15. One persona label in three matches the pole's lens reads; swapped names rarely do

![One persona label in three matches the pole's lens reads; swapped names rarely do](15_library_judge.png)

Share of persona rows whose reads clearly match the name (the concept, a translation, or a strongly associated thing): 34% from the write side, 27% from the input side and 40% with both shown together, against 2-5% for swapped names. Matches are most common at L24-L35; the input side is weakest at L18-L23 (10%).

## 16. Read a head's input with the S-lens; its output with the S-lens early, the R-lens late

![Read a head's input with the S-lens; its output with the S-lens early, the R-lens late](16_head_lens_sides.png)

Margins over a control head, by band. Solid bars (S-lens input) beat pale ones (R-lens input) with the same output lens in every band, on both tests, except the copy test at L28-L35 with an S-lens output, which the catalog doesn't use there. On the output side the S-lens leads at L1-L8 (copy +0.44 against +0.39) and roughly ties at L13-L20; the R-lens leads at L9-L12 and by far from L21 (+0.65 and +0.80).

## 17. Late heads' OV reads predict the next token; mid-network heads are mostly dark

![Late heads' OV reads predict the next token; mid-network heads are mostly dark](17_head_census.png)

Shares of each layer's heads: an OV read passes, only a token mode or attention pattern, or dark. `ov_R` (the attended token's column predicts the next token) passes for 27% of L28-L35's heads, half at L33; the S-side reads (`ov_S`, `qk_S`) never separate a head from its layer. Dark is 56% of heads, 78% at L9-L12 and 74% at L13-L20.

## 18. Qwen3-8B and Gemma 4 12B read alike: inputs early, writes late, a dark middle

![Qwen3-8B and Gemma 4 12B read alike: inputs early, writes late, a dark middle](18_two_models.png)

Both models against relative depth (layer / last layer). Left: median r between the R-lens read of a pole's write and its 1× steering effect. In Qwen3-8B it predicts steering from L4 (0.46-0.51 at L4-L8); in Gemma 4 12B it stays at 0.09 or less through L10 (relative depth 0.21), jumps to 0.40 at L11 and reaches 0.9 at L34 (Qwen3-8B: L22). The logit lens (dashed) trails the R-lens in both until late. Right: a weight-only read passes for 80% of Qwen3-8B's L0 poles and 84-88% of Gemma's L0-L1, for at most 7% of Qwen3-8B's L13-L20 and 4% of Gemma's L15-L30, and again late, mostly through the R-lens write (39% at Qwen3-8B's L29, 41% at Gemma's L33). Dark peaks mid-network: 86% at Qwen3-8B's L18 and 85% at Gemma's L22 (relative depth 0.51 and 0.47).

## 19. Gemma 4 12B: a neuron's write, read through the R-lens, predicts steering from L11

![Gemma 4 12B: a neuron's write, read through the R-lens, predicts steering from L11](19_gemma_steering.png)

Median r between each lens's write read and the measured effect, by layer, as figure 7 for Qwen3-8B (Gemma 4 12B has no J-lens). Left, steering at 1× the pole's typical activation: the R-lens stays at 0.09 or less through L10, jumps to 0.40 at L11, and reaches 0.82 at L29 and 0.96 at L40; the logit lens trails it by 0.33-0.49 from L11 to L30 and closes the gap only in the last layers. Right, ablating the pole where it fires: under 0.1 before L27, then rising to 0.67 at L41 and 0.91 at L46. Controls stay within ±0.04.

## 20. Gemma 4 12B: inputs read early, writes late, and most of the middle is dark

![Gemma 4 12B: inputs read early, writes late, and most of the middle is dark](20_gemma_census.png)

The input side reads early: the S-lens read names the firing token for 84-88% of L0-L1's poles, falls to 0.5% at L9, and rises again at L10-L13 (23-57%), where the S-lens write also holds for 10-22% at L11-L13. The write side reads late: the R-lens write predicts the next token after a firing from L32, for 22-37% of poles at L33-L40, falling to 8-14% at L44-L47. In between, 54-85% of L16-L31's poles are dark. Over all layers 19% of poles pass some test and 52% are dark (Qwen3-8B: 22% and 57%).

## 21. The S-lens's image read is a projection; an optimised read makes the neuron fire

![The S-lens's image read is a projection; an optimised read makes the neuron fire](21_media_reads.png)

Median rank of a pole's own change by band. Images: the S-lens read moves the pole's read projection (0.99, 0.94 and 0.89 at L0-L5, L6-L11 and L12-L23) but raises its activation only at L0-L5 (0.92; 0.09 at L6-L11). Sounds: the S-lens read is near chance on both (0.34-0.60 by band). The read optimised from it (one pattern tiled over the 64 slots, at 8% of the pixel range or 0.3 of a frame's RMS) raises its pole through L23 (images 0.86-1.00, sounds 1.00) and is near chance from L24.

## 22. The patch each neuron reads, and the patch that drives it

![The patch each neuron reads, and the patch that drives it](22_media_gallery.png)

Three poles from each six-layer band from L0 to L23, those with the strongest reliable exemplar labels (one pole per label) among the poles whose optimised patch raises them on held-out images. The S-lens reads share one pattern of column streaks from pole to pole, as text reads share hub tokens; the optimised patches, one pattern tiled over all 64 slots and added at 8% of the pixel range, differ by pole. Early labels are primitives (a checkerboard, a grating, scripts by writing system), later ones mostly objects.

## 23. Image labels hold on the text side from L12; sound labels only a little

![Image labels hold on the text side from L12; sound labels only a little](23_crossmodal.png)

Of 253,749 poles with an image label, the label's words fire the pole for 8% at L0-L11 and 42% at L12-L23 (5% by chance), the text R-lens writes it for up to 19% (L12-L23), the text S-lens reads it for 7-10%. Audio labels pass 6-13% by band and test. 18 of 45,138 poles with both labels pass all three text tests for both (independent lists would share 1.8): places and peoples with their language, such as mosques and Arabic, pagodas and Mandarin, tacos and Spanish.

## 24. Single heads carry what an image shows into the text; for sound, none does

![Single heads carry what an image shows into the text; for sound, none does](24_heads_media.png)

Left: attention from the text after the image or sound to its 64 slots, as a multiple of an even spread over the keys, for the layer's strongest head and its median head. The text reads images most at L23-L31 and L44 (up to 0.96 of an even spread) and sounds most at L0-L10 (L8h5: 1.40 times). Right: zeroing single heads raises the right image label's loss more than a wrong one's, most for L28h6 (z 9.9), L29h1 (z 9.7), L29h9 (z 8.4), L25h2 (z 7.9); 29% of L24-L35's heads exceed z = 3. For sound none does (the largest z is 2.8), and the model ranks a sound's own label above another's only 55% of the time, too weakly to find its carriers.
