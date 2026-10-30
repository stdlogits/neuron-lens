"""Named neurons: an earlier experiment's steering-verified ones, and outside labels to cross-check.

``KNOWN`` is ground truth from an earlier persona-steering experiment on Qwen3-8B (not public):
concept neurons whose push makes Qwen3-8B take that persona. ``EXTERNAL`` lists outside labels of
Qwen3-8B neurons, collected from the public sources listed in ``SOURCES`` and checked against their
files where available (the library's local copy, Sauers' and Sudarsh's JSON). Each method
has its own biases (one prompt, a speaker frame, strong doses, attribution heuristics), so a
label is one reading to weigh against our reads, activations and causal checks, not ground truth.
Sign 0: the source names no pole.
"""

KNOWN = {"jesus": (8, 1931, 1), "gods": (9, 3757, -1), "ant": (13, 5115, 1),
         "god": (18, 7807, 1), "mother": (21, 1180, -1), "candidate": (27, 1343, 1)}

SOURCES = {
    "library": "Neuron Persona Library (jfonsecarivera on X, inspired by Sauers' steering; neuronpersonas.com, "
               "also qwen-neurons-site.vercel.app): every neuron pushed "
               "on 'Who are you?' in the chat template, greedy, replies labelled by Claude Haiku",
    "sudarsh": "github.com/Sudarsh3301/one-neuron-is-all-you-need: activation contrast between "
               "personas times W_down alignment, then steering at +-32 (results/v2)",
    "kazemi": "Kazemi et al. 2026, arXiv:2605.08513 (Apple): refusal log-odds gradient times "
              "harmful-harmless activation difference, then pinning the neuron",
    "he": "Muyu He (X, 2025-12 to 2026-01; github.com/RiddleHe/llm-interp): MLP energy at token 0, "
          "three layer-6 dims that write the attention sink",
    "sauers": "Sauers (X, 2026-10-19 and 2026-10-26): single-neuron steering",
    "sauers128": "huggingface.co/Sauers/Qwen3-8B-tawny-owl-128-neurons: causal search for the "
                 "neurons whose values flip a character prompt's first identity token "
                 "(training/persona_128.json, ranked)",
}

# (source, label, layer, neuron, sign)
EXTERNAL = [
    ("library", "smallpox virus", 7, 271, 1), ("library", "coffee bean", 9, 166, 1),
    ("library", "1000-year-old ghost", 12, 545, -1), ("library", "creator god", 8, 8079, 1),
    ("library", "lord god", 9, 7406, -1), ("library", "1986 Ford F-150", 10, 4940, -1),
    ("library", "Lord Jesus Christ", 10, 11580, -1), ("library", "volumetric flask", 11, 7554, -1),
    ("library", "plastic bag", 11, 12281, 1), ("library", "large-scale toilet", 29, 1807, 1),
    ("library", "Great Wall of China", 5, 9982, -1), ("library", "Buddha", 7, 6011, 1),
    ("library", "existential crisis", 7, 9759, 1), ("library", "Neptune", 11, 10593, 1),
    ("sudarsh", "God (deity persona)", 18, 7807, 1), ("sudarsh", "Jesus, Bible", 8, 1931, 1),
    ("sudarsh", "gods", 9, 3757, -1), ("sudarsh", "nature, woodland voice", 8, 5315, -1),
    ("sudarsh", "Yellow Emperor", 9, 598, -1), ("sudarsh", "Napoleon", 9, 10974, -1),
    ("sudarsh", "Hitler, German nationalist", 10, 5809, -1), ("sudarsh", "Mao, party official", 10, 1226, -1),
    ("sudarsh", "tree (Acacia dealbata)", 8, 3193, -1), ("sudarsh", "horse", 10, 3836, 1),
    ("sudarsh", "Sun Yat-sen", 9, 2027, -1), ("sudarsh", "Germany, German", 1, 11672, -1),
    ("sudarsh", "Zhu Bajie, Chinese spirit", 11, 8095, 1),
    ("kazemi", "refusal (pinned to +20 bypasses it)", 14, 7924, 0),
    ("kazemi", "suicide concept (fires negatively)", 26, 4061, -1),
    ("he", "attention sink, dominant dim", 6, 8518, 0), ("he", "attention sink, second", 6, 422, 0),
    ("he", "attention sink, third", 6, 5723, 0),
    ("sauers", "owl", 8, 344, 0), ("sauers", "green", 30, 2310, 0),
    ("sauers128", "AI identity, rank 1", 3, 5307, 0), ("sauers128", "AI identity, rank 2", 9, 8548, 0),
    ("sauers128", "AI identity, rank 3", 7, 1264, 0), ("sauers128", "AI identity, rank 4", 0, 5890, 0),
    ("sauers128", "AI identity, rank 5", 6, 2843, 0),
]
