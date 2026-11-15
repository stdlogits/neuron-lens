You are describing MLP neurons of Qwen3-4B, a post-trained model (36 layers, 9,728 neurons each), from evidence cards. The activation is `silu(gate)·up`.

## Worked examples

These are real poles of this model, not in your sample, each with part of its card and the labels it should get.

1. **Layer 3, + pole.**
   - in·S z: `overwrite×2 |  overwritten | rewrite×5 | WRITE×2 | .writer×2 ‖ Rew×3 |  writeFile |  rewritten | Writable`
   - out·R: ` apologise | colour×4 |  offence |  favour×2 |  favourable | avour ‖  neighbour×2 |  recognise |  analyse |  labelled`
   - fires at 1.6e-05 of positions; code 2.7×
   - contexts: `help=_("Over⟦write⟧ file if exists`, `- "⟦overwrite⟧"`, `fields.boolean('Over⟦write⟧ Existing Terms'`, `flags.DEFINE_boolean("⟦overwrite⟧"`

   → fires: "'write' in overwrite and rewrite, mostly in code options", 0.9; writes: "British spellings (colour, favour, apologise)", 0.6.
   The input read and the contexts agree, and the firing line says code. The write is another theme: label it on its own evidence.

2. **Layer 7, − pole.**
   - in·R: `⏎×16 ‖  #␍⏎ |  \␍⏎ | ialized | esco | ？`
   - out·R: `?”⏎⏎ | :⏎⏎⏎⏎⏎⏎ | .”⏎⏎×3 |  cerca | :⏎⏎⏎ | ”.⏎⏎ | !”⏎⏎ ‖ ?” | ：⏎⏎ | .”⏎⏎⏎⏎ | ’.⏎⏎`
   - contexts: `a culture that⟦ bases⟧ its decisions`, `it's not the medal that⟦ la⟧uds his mark`, `can live healthy, normal lives and⟦ expect⟧ to have`, `hate crime statistics⟦ note⟧ a troubling trend`, `celebrations⟦ accepted⟧ by Donald Trump`

   → fires: "a clause's main verb right after its subject (bases, notes, expect)", 0.7; writes: "closing a quote or sentence before a paragraph break (.”, ?”)", 0.6.
   The input read names line breaks, not the words it fires on: take where it fires from the contexts, and say so in the note.

3. **Layer 20, + pole.**
   - in·R: ` outright |  entirely×3 |  altogether |  downright |  completely×3 | 完全 ‖ 根本没有 | 干脆 |  fully | 彻底 |  wholly`
   - out·R: ` entirely×3 |  outright |  completely×3 |  altogether | 完全 |  fully | 完全没有 ‖  totally | 彻底 |  utterly`
   - shared by both sides: `abolish | altogether | complet | entir | extinct | full | outright | total | ultimat`
   - contexts: `severely reduced if⟦ not⟧ eliminated totally`, `quick improvement, if⟦ not⟧ cure`, `will reduce, if⟦ not⟧ eliminate the odor`, `help improve your seizures or⟦ even⟧ possibly stop them`

   → fires: "'if not' or 'even' before escalating to the whole: reduce, if not eliminate", 0.85; writes: "completeness: entirely, completely, outright (完全, 彻底)", 0.9.
   The input read names the concept that follows the firing word (eliminated totally), and the write pushes toward it.

4. **Layer 35, − pole.**
   - in·R: `地中 | rike | chalk | 架 | dirs | eterminate | .matcher | icism ‖ atement | 艳 | roi`
   - out·R: `hips | -Men | omens | 肌 | 熟 | 实验室 | elu | 龙门 ‖ 实验 |  privateKey |  Prof`
   - fires at 3.3e-05 of positions; code 5.5×
   - contexts: `self⟦.s⟧ilent = silent`, `views⟦.s⟧erve)`, `self⟦.s⟧map={}`, `install⟦.s⟧anitize_args(self`

   → fires: "'.s' starting an attribute or method name in Python (self.silent)", 0.9; writes: "unclear: no theme", 0.1.
   The reads have no theme, but the contexts are unambiguous: a confident firing label and a low-confidence write label.
