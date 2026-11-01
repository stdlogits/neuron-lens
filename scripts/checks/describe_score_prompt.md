You are scoring descriptions of MLP neurons of Qwen3-8B. Each task gives a label, someone's
description of when one pole of a neuron fires, and up to ten numbered text contexts. Each context
ends at one token marked ⟦like this⟧ (⏎ is a newline): you see only the text up to it, which is all
the model has seen at that point. Some of the marked tokens are places where the pole fires
strongly and the rest are places where it stays quiet; the split varies by task.

For each context, give the probability that the pole fires on the marked token, judging only from
the label. Use the context around the token, not just the token itself, if the label describes a
concept or situation. If the label gives no basis to separate them, say so with probabilities near
0.5 rather than guessing confidently.

Output exactly one JSON object per task, one per line (JSON Lines), and nothing else:

{"task": "t0000", "p": [0.9, 0.1, 0.2, 0.8, 0.5, 0.1, 0.7, 0.2, 0.9, 0.3]}

with one probability per context, in the contexts' order.
