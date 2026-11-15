You are describing MLP neurons of Qwen3-1.7B, a post-trained model (28 layers, 6,144 neurons each), from evidence cards. The activation is `silu(gate)·up`.

## Worked examples

These are real poles of this model, not in your sample, each with part of its card and the labels it should get.

1. **Layer 4, + pole.**
   - in·S: ` reasoning×4 | 推理 | 逻辑 |  logic×4 | 论证 |  argument×5 |  logical×4 ‖ 谬 | 议论 | 反驳 |  deduct`
   - out·R: `推理 | 逻辑 |  reasoning×2 | 论证 |  logically×4 | 反驳 |  logic×5 ‖ argument×2 |  deduct | 思维方式 | 命题`
   - shared by both sides: `argument | deduction | inferenc | logic | logical | rational | reason | 反驳 | 推理 | 理性 | 论证 | 辩论 | 逻辑`
   - contexts: `that which⟦ reasoning⟧ cannot exhaust`, `Reason⟦ing⟧ by analogy`, `(Argument⟦um⟧ ad Miseric`, `Fallacious Reason⟦ing⟧`

   → fires: "reasoning, logic and argument (推理, 逻辑, 论证)", 0.95; writes: "reasoning and logic words (推理, 逻辑, logically)", 0.9.
   Every line agrees, in English and Chinese: a neuron that reads and writes one concept.

2. **Layer 12, + pole.**
   - in·R: `)",⏎ | "),⏎ | ")),⏎ |  ".⏎×2 |  )" |  ";⏎×2 | ";⏎⏎⏎ | )"⏎ ‖ "]],⏎ | 强有力 | ))"⏎ | "]),⏎`
   - out·R: `...␍⏎×3 | ,... | ...'×2 | ,...⏎ | ⏎×2 | ...] | ...', ‖ ...⏎⏎⏎ | ...... | …⏎ |  [... | ..., |  [...]`
   - out·S agrees on: `... | … | …⏎ | ⏎`
   - contexts: `The application⟦ (⟧the prover`, `This⟦ (⟧mostly) means`, `Expat⟦ (⟧IIRC)`, `though the measure⟦ —⟧ called Amendment One`

   → fires: "an opening bracket or dash before an aside: (mostly), (IIRC), —called", 0.7; writes: "ellipses and trailing dots (..., …)", 0.7.
   The input read names closing quotes and brackets in code, not the tokens it fires on: the firing label rests on the contexts, and the note says what the read shows.

3. **Layer 25, + pole.**
   - in·R: `辛勤 | 勤奋 | 勤劳 | 努力 | 忙碌 | 辛苦 | 的努力 | 刻苦 ‖  work×2 | 不懈 | 拼搏 | 汗水 |  diligent×2 | 奋斗 |  tirelessly`
   - out·R: ` hard×8 | 硬 ‖ 软 |  hardness |  harder |  soft×6 | 硬度 |  hardest`
   - write shape: peaked, a next-word predictor
   - contexts: `reward their students for all⟦ their⟧ hard work`, `Thanks for all⟦ your⟧ help`, `thanks goes out to the following people for all⟦ their⟧ hard work`, `would not be true without⟦ the⟧ hard`

   → fires: "the word before 'hard work' in thanks (for all their)", 0.85; writes: "'hard', as in hard work, and hard and soft", 0.9.
   Late, the input read names the concept coming up (hard work) and the write the next word itself.

4. **Layer 27, − pole.**
   - in·R: `缩短 | 向外 | 不必要的 | rement |  vos | ’y | vro | zia ‖ iri | rror | ones | 顺`
   - out·R: `case | chia | ov×2 |  cue | arm×2 | 上游 | trust×4 | odor`
   - fires at 4.0e-04 of positions; zh 4.2×
   - contexts: `門⟦ �⟧��||`, `申 請 表 格 可⟦ �⟧�� 下`, `直接⟦�⟧�正學生`, `什麼是齒列⟦�⟧�正`

   → fires: "the first byte of a Chinese character split across tokens", 0.6; writes: "unclear: scattered fragments", 0.1.
   Neither read has a theme. The contexts and the firing line carry where it fires; the write label gets a low confidence.
