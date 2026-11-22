"""Per-neuron change between a Qwen3 and its base: writes (vectors, metric at the layer's output) and gate/up rows
(covectors, metric at the MLP input), one ruler (the post-trained model's metric) for both. Saves a dict of [L, f] tensors."""
import sys, time
import torch, pyarrow.parquet as pq
from neuron_lens import arch, metric
from neuron_lens.catalog import MODEL, neuron_table
from neuron_lens.neurons import directions

POST, BASE = MODEL, MODEL + "-Base"
CFG = arch.text_config(POST)
L, F = CFG.num_hidden_layers, CFG.intermediate_size
out = {k: torch.empty(L, F) for k in ("cos_w", "nw_base", "nw_post", "dw", "cos_g", "ng_base", "ng_post", "dg",
                                       "cos_u", "nu_base", "nu_post", "du")}
acts = {k: torch.empty(L, 2 * F) for k in ("top_post", "top_base", "rate_post", "rate_base")}

def vstats(a, b, w):  # a, b [f, d] rows; w [d] the inner product's weight
    ab, aa, bb = (a * b * w).sum(1), (a * a * w).sum(1), (b * b * w).sum(1)
    d = ((a - b).pow(2) * w).sum(1)
    return ab / (aa * bb).sqrt(), bb.sqrt(), aa.sqrt(), d.sqrt()

for l in range(L):
    t0 = time.perf_counter()
    p, b = directions(POST, l), directions(BASE, l)
    so, sm = metric.s2(POST, "out", l), metric.s2(POST, "mid", l)
    for key, x, y, w in (("w", p["write"].T, b["write"].T, 1 / so), ("g", p["gate"], b["gate"], sm), ("u", p["up"], b["up"], sm)):
        c, nb, np_, d = vstats(x, y, w)
        out["cos_" + key][l], out[f"n{key}_base"][l], out[f"n{key}_post"][l], out["d" + key][l] = c, nb, np_, d
    for m, tag in ((POST, "post"), (BASE, "base")):
        t = pq.read_table(neuron_table("acts", l, m), columns=["act_top", "rate"])
        acts["top_" + tag][l] = torch.from_numpy(t.column(0).to_numpy().copy())
        acts["rate_" + tag][l] = torch.from_numpy(t.column(1).to_numpy().copy())
    print(f"L{l} {time.perf_counter() - t0:.0f}s  median 1-cos w {1 - out['cos_w'][l].median():.4f}", flush=True)
torch.save(out | acts, sys.argv[1])
