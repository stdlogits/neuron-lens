You are describing MLP neurons of Qwen3-0.6B, a post-trained model (28 layers, 3,072 neurons each), from evidence cards. The activation is `silu(gate)·up`.

## Worked examples

These are real poles of this model, not in your sample, each with part of its card and the labels it should get.

1. **Layer 4, + pole.**
   - in·S: ` reliance | 靠 | 依靠 | 侥幸 | 咎 |  blame×4 |  crippling | ucky ‖  Florida×2 |  heavily | 指望 | 仰 | 靠着 | 依赖`
   - out·R: `估算 | 平均 | 金额 | 平均水平 |  averages | 偏低 | 总额 ‖ 统计数据 | 估计 | 报告显示 |  reported | _avg`
   - contexts: `you could count⟦ on⟧ Starbucks`, `You cannot count⟦ on⟧ any bit`, `they can count⟦ on⟧ a great lifestyle`, `Can we count⟦ on⟧ you to help`

   → fires: "'on' in count on: relying on someone (依靠, 指望)", 0.9; writes: "averages, estimates and statistics (平均, 估算)", 0.6.
   The literal input read and the contexts agree. The write is another theme: label it on its own evidence.

2. **Layer 6, + pole.**
   - in·R: `*␍⏎ | ⏎×10 | CTIONS | __␍⏎ | "␍⏎ ‖ --⏎⏎ | 医务 | 医疗机构 | :␍⏎␍⏎`
   - out·R: ` anywhere | —even | —including | —not | ”— | —if | —it | —but ‖  even |  inadvert | –and | —which`
   - contexts: `A shudder rippled⟦ through⟧ her`, `the coolness that seeps⟦ into⟧ the soles`, `doubts keep creeping⟦ into⟧ your mind`, `a virus had swept⟦ through⟧ her hometown`

   → fires: "'through' and 'into' after verbs of spreading (rippled, seeps, swept)", 0.8; writes: "dash-led asides and continuations (—even, —including, —but)", 0.6.
   The input read names line breaks, not the words it fires on: take where it fires from the contexts, and say so in the note.

3. **Layer 21, − pole.**
   - in·R: ` consistent×2 | 可靠的 |  trustworthy |  remarkably | 清晰 |  accurate×2 | 清楚 ‖  believable | 相近 |  reasonably |  reliable`
   - out·R: ` unusually | 更强 | 更加 | 较好的 |  easier |  greater | 更具 | 不够 ‖ 较为 | 更为 |  clearer |  tighter |  stronger`
   - shared by both sides: `clearer | consistent | easier | stronger | 更具 | 更容易 | 较为 | 较好`
   - fires at 3.6e-03 of positions; zh 3.7×
   - contexts: `Ascend P⟦2⟧ 差不多`, `已經和傳統大富⟦翁⟧截然不同`, `和其他款⟦遊戲⟧一樣`, `外型與一般掛⟦勾⟧沒兩樣`

   → fires: "the end of a compared thing before 差不多, 一樣, 截然不同", 0.8; writes: "comparatives and degree words (更加, 较好, easier, stronger)", 0.85.
   Both sides share comparison words, and the contexts are Chinese comparisons, as the firing line's zh 3.7× confirms.

4. **Layer 27, + pole.**
   - in·R: `ledge | agers | 覆 | riors×2 | ouch | resses | asics |  nulla ‖ ores |  nevertheless | ours`
   - out·R: `olin×2 | iliate | 市区 |  durations |  snaps |  colspan | ("~ | olina`
   - contexts: `◆⟦ �⟧��業`, `GYP_TO_ANDROID⟦.P⟧Y`, `state is I⟦(p⟧resent)`, `3)⟦ �⟧��將`, `-⟦ �⟧��師`

   → fires: "split pieces: a broken CJK character's first byte, odd splits in code", 0.5; writes: "unclear: word fragments", 0.1.
   Neither read has a theme. The contexts carry the firing label, and the write label gets a low confidence.
