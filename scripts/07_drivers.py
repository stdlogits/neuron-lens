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
  attention from the 16 positions after them, as the key (``keys``), the mean over those queries. A Gated DeltaNet's
  heads attend through their mixing matrix ``A`` (``o = A v``), read as ``06_activations.py`` reads it: ``|A|``, each
  row at unit sum, from the mixer's input over the whole document (the prefix's, kept from its pass, and the pass's).

Gemma 4 12B (``NEURON_LENS_MODEL=google/gemma-4-12B``): per-layer head sizes, and each head's
output as it reaches the residual (the post-attention norm's weight and its scale at the token).

    python scripts/07_drivers.py
"""
import argparse
import copy
import time

import torch

from neuron_lens import anatomy, arch, layers
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
    write_heads,
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


def repeat(cache, n: int) -> None:
    """The prefix's cache for each of ``n`` tokens, document-major: HF's ``batch_repeat_interleave``, which a linear
    attention layer's states (a DeltaNet's: its convolution's and its memory's) lack."""
    for layer in cache.layers:
        if hasattr(layer, "recurrent_states"):
            for i in range(layer.number_of_states):
                for states in (layer.conv_states, layer.recurrent_states):
                    if states[i] is not None:
                        states[i] = states[i].repeat_interleave(n, 0)
        else:
            layer.batch_repeat_interleave(n)


def pole_drivers(a: torch.Tensor) -> dict[str, torch.Tensor]:
    """A pass's neuron activations from the token on, ``[D, B, 1 + AFTER, f]``, reduced as the drivers
    rank them and averaged over the documents: at the token, and summed over it and the 16 after."""
    return {"at": a[:, :, 0].mean(0), "sum": a.sum(2).mean(0)}  # [B, f]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64, help="tokens per pass, each in every document")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    args = ap.parse_args()
    dev = args.device
    docs = documents(dev)
    D = len(docs)
    _, model = arch.load_model(MODEL, dev)  # in its half dtype, eager attention
    cfg = model.config
    P, signs = arch.n_poles(MODEL), arch.signs(MODEL)
    L, f, H, d = cfg.num_hidden_layers, cfg.intermediate_size, arch.max_heads(MODEL), cfg.hidden_size
    geo = arch.layout(MODEL)  # heads: a grid of H a layer, each layer's first
    wo = [arch.head_write(m).view(d, g.q_heads, g.head_dim) for m, g in zip(model.layers, geo)]  # as the residual sees it
    gram = [torch.einsum("dhi,dhj->hij", w, w) for w in wo]  # |W_O^h z|² = zᵀ G_h z
    del wo
    best = {k: torch.full(shape, -torch.inf, device=dev) for k, shape in
            (("at", (L, P, f, TOP)), ("sum", (L, P, f, TOP)), ("h_out", (L, H, TOP)), ("h_key", (L, H, TOP)))}
    best_ids = {k: torch.zeros(v.shape, dtype=torch.long, device=dev) for k, v in best.items()}
    cur, stash = {}, {}

    def merge(key: str, l, val: torch.Tensor, ids: torch.Tensor) -> None:  # val [rows, B]; l indexes best[key]
        v, j = torch.cat([best[key][l], val], -1).topk(TOP, dim=-1)
        best_ids[key][l] = torch.cat([best_ids[key][l], ids.expand_as(val)], -1).gather(-1, j)
        best[key][l] = v

    def neuron(l: int):
        def hook(m, x):
            a = x[0].float().unflatten(0, (D, -1))  # [D, B, 1 + AFTER, f]
            for key, v in pole_drivers(a).items():
                for p, sign in enumerate(signs):
                    merge(key, (l, p), sign * v.T, cur["ids"])
        return hook

    def heads(l: int, out, key: torch.Tensor) -> None:
        """The token's output norm as the query, and ``key`` ``[D·B, H]``, the attention it receives."""
        Hl = geo[l].q_heads
        z = stash.pop(l)[:, 0].float().view(-1, Hl, geo[l].head_dim)  # [D·B, H, hd]: the token as the query
        scale = arch.attn_scale(model.layers[l], out)  # Gemma 4: the post-attention norm's scale
        if scale is not None:
            z = z * scale[:, 0, None, None]
        norm = torch.einsum("nhi,hij,nhj->nh", z, gram[l], z).clamp_min(0).sqrt()
        merge("h_out", (l, slice(0, Hl)), norm.unflatten(0, (D, -1)).mean(0).T, cur["ids"])
        merge("h_key", (l, slice(0, Hl)), key.unflatten(0, (D, -1)).mean(0).T, cur["ids"])

    def attention_hook(l: int):
        def hook(m, x, out):
            heads(l, out[0], out[1][:, :, 1:, AT].float().mean(-1))  # attention from the positions after it
        return hook

    prefix_x = {}  # a DeltaNet layer's input over the prefix, [D, AT, d]

    def delta_hook(l: int):
        def hook(m, x, kw, out):
            x = x[0] if x else kw["hidden_states"]
            if cur.get("prefix"):
                prefix_x[l] = x
                return
            full = torch.cat([prefix_x[l].repeat_interleave(len(cur["ids"]), 0), x], 1)  # document-major, as the batch
            A = layers.delta_pattern(m, full).A[:, :, AT + 1:].abs()  # [D·B, H, AFTER, T]: the rows after the token
            heads(l, None, (A[..., AT] / A.sum(-1).clamp_min(1e-30)).mean(-1))
        return hook
    order = token_order()
    t0 = time.time()
    with torch.inference_mode():
        for l, m in enumerate(model.layers):  # a DeltaNet's mixer keeps its input over the prefix
            if anatomy.parts(m).deltanet:
                anatomy.module(m, anatomy.parts(m).mixer).register_forward_hook(delta_hook(l), with_kwargs=True)
        cur["prefix"] = True
        prefix = model(input_ids=docs[:, :AT], use_cache=True).past_key_values
        cur["prefix"] = False
        for l, m in enumerate(model.layers):  # from here on every pass starts at the token
            p = anatomy.parts(m)
            anatomy.module(m, p.down).register_forward_pre_hook(neuron(l))
            anatomy.module(m, p.out).register_forward_pre_hook(lambda mod, x, l=l: stash.__setitem__(l, x[0]))
            if not p.deltanet:
                attention(m).register_forward_hook(attention_hook(l))
        for i in range(0, len(order), args.batch):
            ids = order[i:i + args.batch].to(dev)
            cur["ids"] = ids
            batch = docs[:, None, AT:].repeat(1, len(ids), 1)  # [D, B, 1 + AFTER]
            batch[:, :, 0] = ids
            cache = copy.deepcopy(prefix)
            repeat(cache, len(ids))  # document-major, as the batch is flattened
            model(input_ids=batch.flatten(0, 1), past_key_values=cache, use_cache=True)
            del cache
            if (i // args.batch) % 100 == 0:
                print(f"{time.strftime('%H:%M:%S')} {i + len(ids)}/{len(order)} tokens, {time.time() - t0:.0f}s", flush=True)

    for l in range(L):
        cols = neuron_keys(l, f)
        for key, name in (("at", "drivers"), ("sum", "drivers_sum")):
            put(cols, name, torch.cat([best_ids[key][l, p] for p in range(P)]).cpu(),
                torch.cat([best[key][l, p] for p in range(P)]).cpu())  # times the pole's sign, as top_val
        write(cols, neuron_table("drivers", l))
    cols = head_keys()
    for key, name in (("h_out", "drivers"), ("h_key", "keys")):
        put(cols, name, best_ids[key].flatten(0, 1).cpu(), best[key].flatten(0, 1).cpu())
    write_heads(cols, head_table("drivers"))
    print(f"{time.strftime('%H:%M:%S')} drivers written, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
