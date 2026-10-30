You are judging descriptions of what MLP neurons of Qwen3-8B do when their output is amplified.

Each task shows four prompts. For each prompt you get the model's unsteered greedy continuation,
then continuations with one neuron's write added to the model's residual stream at every position,
at a light, a moderate and a strong dose. The strong one is often degenerate, but repeated tokens
can still show what the write pushes. ⏎ is a newline. Small wording changes at the light dose can
be chance divergence of greedy decoding.

Then come four descriptions, A-D, each of what some neuron promotes. Exactly one was written for
this neuron, from evidence about it. The other three were written for other neurons at a similar
depth.

Compare the steered text with the unsteered: what words, topics, languages, styles or behaviours
appear or grow with the dose? Give each description a probability that it is the one written for
this neuron (they sum to 1). If none fits, or the change is generic (degeneration with no
content), spread the probability evenly rather than guess.

Output exactly one JSON object per task, one per line (JSON Lines), and nothing else:

{"task": "s0000", "p": [0.7, 0.1, 0.1, 0.1]}

with the probabilities in the order A, B, C, D.
