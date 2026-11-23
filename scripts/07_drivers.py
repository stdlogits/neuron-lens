"""Exact token drivers of every neuron pole and attention head, in real contexts (pipeline stage 7).

Every kept token (``catalog.kept``) goes into 8 real documents (4 DCLM, 2 code, 2 chat: the first
full-length ones of the activation slices) at position 24, in place of the token there, with the
documents' own 16 tokens after it; results are averaged over the documents (findings §7). Eager
attention, bf16; an fp32 rerun through L9 kept 98-99% of a pole's top 20 (§7,
``checks/drivers_precision.py``, which imports this pass's documents, order and reductions). Attention is
causal, so the first 24 positions don't depend on the token: they run once per document and every pass
continues from their cache.

- **Neurons** (``neurons/drivers/L<layer>.parquet``): per pole, the 64 tokens with the largest
  activation at their own position (``drivers``), and summed over it and the 16 after
  (``drivers_sum``). A − pole's drivers are the most negative activations.
- **Heads** (``heads/drivers.parquet``): the 64 tokens with the largest output norm
  ``|W_O^h z|`` at their own position, as the query (``drivers``), and the 64 that receive the most
  attention from the 16 positions after them, as the key (``keys``), the mean over those queries.

Gemma 4 12B (``NEURON_LENS_MODEL=google/gemma-4-12B``): per-layer head sizes, and each head's
output as it reaches the residual (the post-attention norm's weight and its scale at the token).

A sparse MLP's neurons (``neuron_lens.experts``): a routed neuron's activation is 0 in a document where the token, or a
position after it, isn't routed to its expert, and the means over the documents count those zeros: a driver is a token
that both routes to the expert and drives the neuron there. ``--layers a-b`` takes the neurons of those layers alone,
the heads with the group from layer 0 (the top lists take ``f · P · TOP · 8 · 2`` bytes a layer on the device: 270 MB
for Qwen3.5-35B-A3B's 131,584 neurons, 10.8 GB for its 40 layers beside 67 GB of weights).

    python scripts/07_drivers.py [--layers a-b]
"""
import argparse
import copy
import time

import torch

from neuron_lens import anatomy, arch, experts
from neuron_lens.catalog import (
    MODEL,
    TOP,
    head_keys,
    head_table,
    kept,
    neuron_keys,
    neuron_table,
    put,
    tokenizer,
    write,
)
from neuron_lens.corpus import act_file, lens_docs
from neuron_lens.rules import attention

SLICES = {"dclm": 4, "code": 2, "chat": 2}
SEQ = 128  # the documents' length: full-length ones are typical text
AT, AFTER = 24, 16


def documents(dev) -> torch.Tensor:
    """``[8, AT + 1 + AFTER]``: the documents the tokens go into, up to the last position read."""
    tok = tokenizer()
    rows = []
    for name, k in SLICES.items():
        rows += lens_docs(act_file(name), tok, 4 * k, SEQ, SEQ)[:k]
    return torch.tensor(rows, device=dev)[:, :AT + 1 + AFTER]


def token_order() -> torch.Tensor:
    """Every kept token, in the order of the sweep (which breaks ties among equal activations)."""
    keep = kept()
    return keep[torch.randperm(len(keep), generator=torch.Generator().manual_seed(0))]


def pole_drivers(a: torch.Tensor) -> dict[str, torch.Tensor]:
    """A pass's neuron activations from the token on, ``[D, B, 1 + AFTER, f]``, reduced as the drivers
    rank them and averaged over the documents: at the token, and summed over it and the 16 after."""
    return {"at": a[:, :, 0].mean(0), "sum": a.sum(2).mean(0)}  # [B, f]


def sparse_drivers(r: experts.Routed, D: int, B: int) -> dict[str, torch.Tensor]:
    """``pole_drivers`` from a sparse MLP's pass (``experts.Routed``, rows document-major as the batch is flattened):
    a routed neuron's activation is 0 in the documents whose token doesn't route to its expert, which the means count."""
    f, Tp = r.width, r.n // (D * B)
    db, t = r.rows // Tp, r.rows % Tp
    key = ((db % B)[:, None] * f + r.cols()).flatten()
    out = {}
    for name, sel in (("at", (t == 0)[:, None].expand_as(r.act).flatten()), ("sum", None)):
        v = r.act.flatten() if sel is None else r.act.flatten() * sel
        out[name] = torch.zeros(B * f, device=v.device).index_add_(0, key, v).view(B, f) / D
    if r.shared is not None:
        sh = r.shared.view(D, B, Tp, -1)
        for name, v in pole_drivers(sh).items():
            out[name][:, r.E * r.I:] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64, help="tokens per pass, each in every document")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    ap.add_argument("--layers", default=None,
                    help="a-b: the neurons of these layers alone (default all), the heads with the group from layer 0")
    args = ap.parse_args()
    dev = args.device
    docs = documents(dev)
    D = len(docs)
    _, model = arch.load_model(MODEL, dev)  # in its half dtype, eager attention
    cfg = model.config
    P, signs = arch.n_poles(MODEL), arch.signs(MODEL)
    L, f, H, d = cfg.num_hidden_layers, experts.stack_width(model), cfg.num_attention_heads, cfg.hidden_size
    lo, hi = (0, L - 1) if args.layers is None else map(int, args.layers.split("-"))
    layers, heads = list(range(lo, hi + 1)), lo == 0
    nl = {l: i for i, l in enumerate(layers)}
    geo = arch.layout(MODEL)
    wo = [arch.head_write(m).view(d, H, g.head_dim) for m, g in zip(model.layers, geo)]  # as the residual sees it
    gram = [torch.einsum("dhi,dhj->hij", w, w) for w in wo]  # |W_O^h z|² = zᵀ G_h z
    del wo
    best = {k: torch.full(shape, -torch.inf, device=dev) for k, shape in
            (("at", (len(layers), P, f, TOP)), ("sum", (len(layers), P, f, TOP)), ("h_out", (L, H, TOP)), ("h_key", (L, H, TOP)))}
    best_ids = {k: torch.zeros(v.shape, dtype=torch.int32, device=dev) for k, v in best.items()}
    cur, stash = {}, {}

    def merge(key: str, l: int, val: torch.Tensor, ids: torch.Tensor) -> None:  # val [rows, B]
        v, j = torch.cat([best[key][l], val], -1).topk(TOP, dim=-1)
        best_ids[key][l] = torch.cat([best_ids[key][l], ids.to(torch.int32).expand_as(val)], -1).gather(-1, j)
        best[key][l] = v

    def drivers(l: int, reduced: dict) -> None:
        for key, v in reduced.items():
            for p, sign in enumerate(signs):
                merge(key, (nl[l], p), sign * v.T, cur["ids"])

    def neuron(l: int):
        def hook(m, x):
            drivers(l, pole_drivers(x[0].float().unflatten(0, (D, -1))))  # [D, B, 1 + AFTER, f]
        return hook

    def attention_hook(l: int):
        def hook(m, x, out):
            z = stash.pop(l)[:, 0].float().view(-1, H, geo[l].head_dim)  # [D·B, H, hd]: the token as the query
            scale = arch.attn_scale(model.layers[l], out[0])  # Gemma 4: the post-attention norm's scale
            if scale is not None:
                z = z * scale[:, 0, None, None]
            norm = torch.einsum("nhi,hij,nhj->nh", z, gram[l], z).clamp_min(0).sqrt()
            merge("h_out", l, norm.unflatten(0, (D, -1)).mean(0).T, cur["ids"])
            key = out[1][:, :, 1:, AT].float().mean(-1)  # [D·B, H]: attention from the positions after it
            merge("h_key", l, key.unflatten(0, (D, -1)).mean(0).T, cur["ids"])
        return hook
    order = token_order()
    t0 = time.time()
    with torch.inference_mode():
        prefix = model(input_ids=docs[:, :AT], use_cache=True).past_key_values
        for l, m in enumerate(model.layers):  # from here on every pass starts at the token
            p = anatomy.parts(m)
            if l in nl and p.sparse:
                experts.Tap(m.mlp, lambda r, l=l: drivers(l, sparse_drivers(r, D, len(cur["ids"]))))
            elif l in nl:
                anatomy.module(m, p.down).register_forward_pre_hook(neuron(l))
            if heads:
                anatomy.module(m, p.out).register_forward_pre_hook(lambda mod, x, l=l: stash.__setitem__(l, x[0]))
                attention(m).register_forward_hook(attention_hook(l))
        for i in range(0, len(order), args.batch):
            ids = order[i:i + args.batch].to(dev)
            cur["ids"] = ids
            batch = docs[:, None, AT:].repeat(1, len(ids), 1)  # [D, B, 1 + AFTER]
            batch[:, :, 0] = ids
            cache = copy.deepcopy(prefix)
            cache.batch_repeat_interleave(len(ids))  # document-major, as the batch is flattened
            model(input_ids=batch.flatten(0, 1), past_key_values=cache, use_cache=True)
            del cache
            if (i // args.batch) % 100 == 0:
                print(f"{time.strftime('%H:%M:%S')} {i + len(ids)}/{len(order)} tokens, {time.time() - t0:.0f}s", flush=True)

    for l in layers:
        cols = neuron_keys(l, f)
        for key, name in (("at", "drivers"), ("sum", "drivers_sum")):
            put(cols, name, torch.cat([best_ids[key][nl[l], p] for p in range(P)]).cpu(),
                torch.cat([best[key][nl[l], p] for p in range(P)]).cpu())  # times the pole's sign, as top_val
        write(cols, neuron_table("drivers", l))
    if not heads:
        print(f"{time.strftime('%H:%M:%S')} drivers of L{lo}-L{hi} written, {time.time() - t0:.0f}s", flush=True)
        return
    cols = head_keys(L, H, arch.groups(MODEL))
    for key, name in (("h_out", "drivers"), ("h_key", "keys")):
        put(cols, name, best_ids[key].flatten(0, 1).cpu(), best[key].flatten(0, 1).cpu())
    write(cols, head_table("drivers"))
    print(f"{time.strftime('%H:%M:%S')} drivers written, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
