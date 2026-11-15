You are describing MLP neurons of Qwen3-14B, a post-trained model (40 layers, 17,408 neurons each), from evidence cards. The activation is `silu(gate)·up`.

## Worked examples

These are real poles of this model, not in your sample, each with part of its card and the labels it should get.

1. **Layer 3, + pole.**
   - in·S: `党委 (party committee) |  party×4 | 党组织 | 党支部 | 党员 | 党和国家 | 党内 ‖ 党 | 党委书记 | 黨`
   - out·R: `党内 | 党支部 | 黨 | 支部 | 党 | 党风 | 副书记 | .vote ‖ 党的建设 | 党组织 | 投票 | 执政`
   - shared by both sides: `党 | 党内 | 党员 | 党支部 | 党组织 | 黨`
   - fires at 1.4e-05 of positions; zh 4.2×
   - contexts: `1940年4月，党⟦卫⟧军`, `instructed to serve Party⟦ Wall⟧ Notice(s)`, `密切了党⟦群⟧关系`, `凝结的是x局⟦党委⟧的拳拳`

   → fires: "the token after 党 or Party: party bodies (党委, 党群, Party Wall)", 0.8; writes: "Party organisation words (党支部, 党员, 党内)", 0.85.

2. **Layer 6, + pole.**
   - in·S: `SEND×2 |  safeguard | SEN×2 |  assessment |  timetable | eguard ‖ 进修 |  nursery |  wellbeing |  vulnerable |  mainstream |  staff`
   - out·R: ` £×3 |  UK×3 |  postcode |  NHS |  organised×2 ‖  Welsh |  cosy |  bespoke |  recognised×2 |  Cornwall |  Wales |  Yorkshire`
   - shared by both sides: `specialist | uk`
   - contexts: `covered in fixed⟦ penalty⟧ notices`, `qualified CR⟦B⟧ checked`, `Effective partnership⟦ working⟧ is a must`, `references and DBS (CR⟦B⟧) checks`

   → fires: "British public-service and schools jargon (CRB checks, fixed penalty)", 0.75; writes: "British English and UK things (£, NHS, organised, postcode)", 0.85.
   Both reads point to British usage; the contexts show where it surfaces.

3. **Layer 9, − pole.**
   - in·R: ` '×3 |  humour |  '[ | Centre×3 |  traveller |  '⏎ ‖  Colour×2 |  peculiar | Whilst |  fibre |  recognise | organisation×2`
   - out·R: `哪儿 | 都没 | 还没 |  ça |  kids | ’t | ’ll |  vibe ‖ 那儿 | ’re |  veggies | 没人 | 怎么 | 啥`
   - contexts: `I shall proceed to explain those opinions⟦ that⟧ have been derived`, `distinguishing them from those portions⟦ that⟧ have`, `his account of an attempt⟦ that⟧ he once made`, `The same principle⟦ that⟧ determines the flow`

   → fires: "relative 'that' after a noun in formal prose (opinions that, portions that)", 0.85; writes: "casual speech: contractions and colloquial Chinese (’ll, 哪儿, 啥)", 0.5.
   The input read names British spellings, which the contexts don't show: take where it fires from them, and say in the note what the read adds.

4. **Layer 33, + pole.**
   - in·R: `演讲 (speech) |  speeches×4 |  lectures×3 | 讲课 |  speaker×6 | 讲座 | 讲话 ‖ 讲师 |  keynote |  lecturer`
   - out·R: ` speaker×6 | 演讲 | 讲话 |  speaking×3 | spe×6 ‖ 讲 | 发言 |  speeches×2 | 说话`
   - write shape: peaked, a next-word predictor
   - contexts: `the main⟦ or⟧ation was given`, `the Phi Beta Kappa lecture at Harvard. His⟦ or⟧ation`, `an accomplished⟦ public⟧ speaker`, `Key⟦note⟧ Speaker`

   → fires: "the word before speech or speaker: or|ation, public speaker, keynote", 0.85; writes: "speaker, speech and lecture words (演讲, 讲话)", 0.9.
   Late, the input read names the concept coming next, and the write predicts it.

5. **Layer 39, − pole.**
   - in·R: `口头 | ustin |  sez | breadcrumbs | 顶端 | .Release×2 | .addTarget | 专科`
   - out·R: `揽 | 撇 | 东部 | Cfg | 愍 | 燃料 | 擦拭 | 海滩`
   - contexts: `細聽著河水汨汨而流。⟦�⟧�多拉`, `鋼弾模型店 世⟦�⟧�機車`, `都有一條運河，和⟦�⟧�多拉`, `壓力症候群⏎⟦�⟧�、精神`

   → fires: "the first byte of a Chinese character split across tokens", 0.6; writes: "unclear: no theme", 0.1.
