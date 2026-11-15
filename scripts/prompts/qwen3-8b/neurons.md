You are describing MLP neurons of Qwen3-8B, a post-trained model (36 layers, 12,288 neurons each), from evidence cards. The activation is `silu(gate)·up`.

## Worked examples

These are real poles of this model, not in your sample, each with part of its card and the labels it should get.

1. **Layer 0, + pole.**
   - in·S: `'        ' | '    ' | '            ' | '                ' | ⏎×9 | '\t\t    '` (runs of spaces)
   - out·R: `'  ' | ⏎×11 | '    ' | '   ' ‖ '      ' |  _ | 1 | (`
   - shared by both sides: `'        ' | '    ' | ⏎`
   - fires at 2.4e-04 of positions; code 7×
   - contexts: `_defaults = {⏎⟦        ⟧ 'start_date`, `def __init__(self):⏎⟦        ⟧ super(`, `'depends': ['marketing',⏎⟦                ⟧ 'document'`

   → fires: "the indentation that starts a line of code", 0.95; writes: "more indentation and line breaks", 0.7.

2. **Layer 8, + pole.**
   - in·R: `婿 | 星空 (starry sky) | -girl |  Owl×2 | -school |  Twig ‖ 毕业生 | 老太太 | 神奇 | 一年级`
   - out·R: ` owl×3 |  noct |  cosy |  bedtime |  autumn |  night×2 |  pupils×2 ‖ 夜间 (night) |  nighttime |  evening×2 |  sleepy`
   - in·S agrees on: `ow | owl`
   - contexts: `hear the sound of the⟦ ow⟧ls`, `behaviors of different⟦ owl⟧ species`, `nocturnal animals, such as⟦ ow⟧ls and bats`, `in forest⟦ ow⟧ls`

   → fires: "owls (ow|ls, owl)", 0.9; writes: "owls, night and bedtime", 0.85.
   The input read is mixed at this depth, but the S-lens agrees on owl and the contexts settle it.

3. **Layer 17, + pole.**
   - in·R: `学术 (academic) |  undergrad |  undergraduate | 论文 (paper) | 本科生 | academia |  Teaching×2 ‖ 研究生 | 科研 |  lecture×3`
   - out·R: `教材 (textbook) | 课本 | 教学 | 学生 | 课堂 | 课程 | Students×5 ‖  Classroom×2 |  Teaching×2 |  textbook`
   - shared by both sides: `student | teach | textbook | 学生 | 教学 | 教材`
   - contexts: `From Marcus's Number⟦ Fields⟧, I'm`, `2014, p. 29⟦8⟧)`, `provides some early lessons⟦ in⟧ meeting`, `ISBN: 9⟦7⟧808`

   → fires: "academic and textbook text: citations, ISBNs, lessons", 0.6; writes: "teaching, students and textbooks (教学, 学生)", 0.85.
   The marked tokens are ordinary; what the contexts share is the topic the reads name.

4. **Layer 30, + pole.**
   - in·R: ` again×7 | 再次 | 再一次 | 又一次 ‖  wieder |  lagi |  weer | 又是 |  likewise`
   - out·R: ` again×7 | 再次 | 再一次 | 再度 ‖ 又一次 |  wieder | 又 |  anew | また`
   - write shape: peaked, a next-word predictor
   - contexts: `property values are⟦ once⟧ again`, `(FOEM) is⟦ once⟧ again asking`, `to have it look decent⟦ once⟧ more`, `⟦Once⟧ again, thank`

   → fires: "'once' in once again, once more", 0.95; writes: "'again' in any language (再次, wieder, lagi)", 0.95.
   Late, the input read names the word coming next and the write predicts it.

5. **Layer 2, − pole.**
   - in·S: ` topics |  sky |  sk |  beauty |  prov |  prey |  Fal |  cottage ‖  Bal |  satisfaction |  suitable`
   - out·R: `" |  un |  post |  after |  w |  b | ? |  th`
   - contexts: `You, he i⟦ morning⟧ investments`, `The B⟦administrator⟧ Administrator`, `Q⟦Dear⟧ Eva`, `in our daily a⟦activities⟧ shall`

   → fires: "a word right after a stray letter or broken word (a activities, Q Dear)", 0.6; writes: "unclear: word starts and punctuation", 0.1.
   No read has a theme, but the contexts share one: label it from them.
