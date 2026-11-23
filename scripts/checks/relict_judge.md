You are judging what drives units (neuron poles) inside a language model.

Each card describes one target unit through five of its own top contexts, its strongest firings. In every context the
token where a unit fired is marked ⟦like this⟧ (⏎ is a newline). The target's main job is what those contexts share:
a topic, a word or word piece, a grammatical construction, a format, or a place in a pattern. Look at the marked token
and the words just before it, not only at what the passage is about.

Then come several candidate input groups, A, B, C, .... Each group is two contexts taken from somewhere else in the
model, contexts in which the target unit also fires strongly. For each group, decide how what those contexts share
relates to the target's main job:

- "same": the contexts are ordinary instances of the target's main job, as its own contexts show it.
- "related": not instances of the main job, but a sensible neighbour of it, such as a broader or narrower category, a
  shared feature or theme, the same thing in another language or format, a word or construction with a shared role, or
  a natural association. For example, other high-mountain regions for a unit whose job is Colorado.
- "unrelated": no sensible shared feature with the main job. For example, Python web-framework code for a unit whose
  job is Colorado.

Give a probability for each of the three, summing to 1. Only how your probabilities rank the groups is scored, so
commit to your best estimate even on a weak hunch. Equal probabilities carry no information. Also name, in a few words,
what each group's contexts share.
