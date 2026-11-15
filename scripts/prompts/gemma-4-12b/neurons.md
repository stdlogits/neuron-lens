You are describing MLP neurons of Gemma 4 12B, a base model (48 layers, 15,360 neurons each), from evidence cards. The activation is `gelu_tanh(gate)·up`, and the chat slice holds chat transcripts as plain `User:` / `Assistant:` turns.

## Worked examples

These are real poles of this model, not in your sample, each with part of its card and the labels it should get.

1. **Layer 3, − pole.**
   - in·R: ` Huawei×2 |  Modelling×2 |  Tensorflow |  Maths |  cosy |  utilisation |  civilisation ‖ storey |  visualise |  customised |  artefacts |  initialise`
   - out·R: `{\' | ####×2 |  [*×2 |  {\ |  \$×2 | ^{\ | *\* ‖ :\ |  __________ | {\" |  =\`
   - fires at 7.6e-03 of positions
   - contexts: `sleep-apnea-treatment⟦Eight⟧een million`, `Welfare Programs: 606th⟦Largest⟧ Populations`, `trudge thru Carson⟦'⟧s drivel`, `the name of Mr. Spock⟦'⟧s immediate`

   → fires: "a word or 's attached to what comes before without a space (Spock's, 606thLargest)", 0.6; writes: "escape characters and markup (\$, {\, ####)", 0.6.
   The input read names British spellings, which the contexts don't show: take where it fires from them, and say in the note what the read adds.

2. **Layer 16, + pole.**
   - in·R: ` transcendental | 具有 |  physicochemical |  metaphys |  epistem |  Universität |  archetype ‖  transcendent |  promulgated |  perpetuity |  edifice |  constitutive`
   - out·R: ` geologic |  viscosity |  cosmological |  physicists×2 |  nanoscale |  Feynman |  topological ‖  fractal |  theological |  Newtonian |  spacetime`
   - shared by both sides: `metaphy | theological | transcendent`
   - contexts: `a book called Hua-⟦yen⟧ Buddhism`, `佛教宗派如真言宗、三⟦论⟧宗等`, `in common with Dele⟦uze⟧, or Network`, `Gilles Dele⟦uze⟧ and`

   → fires: "names in Buddhist and continental philosophy (Hua-yen, Deleuze, 三论宗)", 0.75; writes: "scholarly science and philosophy terms (cosmological, physicists, theological)", 0.6.
   Here the input read and the contexts share a topic, metaphysics and its schools, though not the marked tokens.

3. **Layer 20, − pole.**
   - in·R: ` basketball×4 |  soccer×4 | 球员 | 篮球 |  football×4 ‖  SPORTS×5 |  hockey×2 |  PLAYER |  ESPN |  athlete`
   - out·R: ` basketball×2 |  sports×5 |  football×4 |  soccer×3 |  athletes×2 |  athletic×2 ‖  baseball |  ESPN | 球员 | 篮球`
   - shared by both sides: `athlet | baseball | basketball | football | hockey | player | soccer | sport | tournament | 体育`
   - contexts: `the puck feel and soft flex profile of⟦ a⟧ wood stick`, `just simple things like kicking⟦ a⟧ ball around`, `kicking⟦ a⟧ soccer ball`, `Running,⟦ playing⟧ soccer, and`

   → fires: "just before a ball or sport word (kicking a ball, playing soccer)", 0.85; writes: "sports: basketball, soccer, football (篮球, 球员)", 0.95.

4. **Layer 34, + pole.**
   - in·R: `最后一个 (the last one) | 最後 | 最后一 | 第一个 |  último | 最后 |  first |  fourth×2 ‖ 第 |  last | 第四 |  dernier |  sixth×2`
   - out·R: ` last×6 | 最后×2 | 最後×2 |  dernières×2 ‖  finally |  seven×2 |  five×2 |  six |  último`
   - shared by both sides: `dernier | final | last | último | 最后 | 最後`
   - contexts: `first train leaves at 00:34, while⟦ the⟧ last ride is`, `the first point is a repeat of⟦ the⟧ last point`

   → fires: "'the' before 'last', after a mention of the first", 0.85; writes: "'last' and ordinals in several languages (最后, dernier)", 0.9.
   Late, the input read names the word coming next and the write predicts it.

5. **Layer 16, − pole.**
   - in·R: ` Bakersfield | ★★★ | BLACKLIST | 周年 |  GOP | REW | ★★ |  rehab ‖ —“ |  POLIT | )—`
   - out·R: ` \\ |  IQ×2 | \| | 观看 | Dubai |  Utd |  Diffraction ‖  IELTS | 思维 |  Creativity`
   - contexts: `repeat the process all over⟦ again⟧`, `replayed her previous encounter over and⟦ again⟧`, `completely reinstall Magento 2⟦ again⟧`, `trying them all over⟦ again⟧`

   → fires: "'again' in over again and over and over again", 0.9; writes: "unclear: no theme", 0.1.
   The reads have no theme, but the contexts are unambiguous: a confident firing label and a low-confidence write label.
