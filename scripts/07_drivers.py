"""Exact token drivers of every neuron pole and attention head, in real contexts (pipeline stage 7).

Every kept token (``catalog.kept``) goes into 8 real documents (4 DCLM, 2 code, 2 chat: the first
full-length ones of the activation slices) at position 24, in place of the token there, with the
documents' own 16 tokens after it; results are averaged over the documents (findings §6). Eager
attention, bf16; an fp32 rerun through L9 kept 98-99% of a pole's top 20 (§6).

- **Neurons** (``neurons/drivers/L<layer>.parquet``): per pole, the 64 tokens with the largest
  activation at their own position (``drivers``), and summed over it and the 16 after
  (``drivers_sum``). A − pole's drivers are the most negative activations.
- **Heads** (``heads/drivers.parquet``): the 64 tokens with the largest output norm
  ``|W_O^h z|`` at their own position, as the query (``drivers``), and the 64 that receive the most
  attention from the 16 positions after them, as the key (``keys``), the mean over those queries.

Gemma 4 12B (``NEURON_LENS_MODEL=google/gemma-4-12B``): per-layer head sizes, and each head's
output as it reaches the residual (the post-attention norm's weight and its scale at the token).

    python scripts/07_drivers.py
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM

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
from neuron_lens import arch
from neuron_lens.corpus import act_file, lens_docs
from neuron_lens.transport import decoder_stack

SLICES = {"dclm": 4, "code": 2, "chat": 2}
SEQ = 128  # the documents' length: full-length ones are typical text
AT, AFTER = 24, 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64, help="tokens per pass, each in every document")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    args = ap.parse_args()
    dev = args.device
    tok = tokenizer()
    keep = kept()
    rows = []
    for name, k in SLICES.items():
        rows += lens_docs(act_file(name), tok, 4 * k, SEQ, SEQ)[:k]
    docs = torch.tensor(rows, device=dev)[:, :AT + 1 + AFTER]
    D = len(docs)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    lm.requires_grad_(False)
    model = decoder_stack(lm)
    cfg = model.config
    L, f, H, d = cfg.num_hidden_layers, cfg.intermediate_size, cfg.num_attention_heads, cfg.hidden_size
    geo = arch.layout(MODEL)
    wo = [arch.head_write(m).view(d, H, g.head_dim) for m, g in zip(model.layers, geo)]  # as the residual sees it
    gram = [torch.einsum("dhi,dhj->hij", w, w) for w in wo]  # |W_O^h z|² = zᵀ G_h z
    del wo
    best = {k: torch.full(shape, -torch.inf, device=dev) for k, shape in
            (("at", (L, 2, f, TOP)), ("sum", (L, 2, f, TOP)), ("h_out", (L, H, TOP)), ("h_key", (L, H, TOP)))}
    best_ids = {k: torch.zeros(v.shape, dtype=torch.long, device=dev) for k, v in best.items()}
    cur, stash = {}, {}

    def merge(key: str, l: int, val: torch.Tensor, ids: torch.Tensor) -> None:  # val [rows, B]
        v, j = torch.cat([best[key][l], val], -1).topk(TOP, dim=-1)
        best_ids[key][l] = torch.cat([best_ids[key][l], ids.expand_as(val)], -1).gather(-1, j)
        best[key][l] = v

    def neuron(l: int):
        def hook(m, x):
            a = x[0][:, AT:].float().unflatten(0, (D, -1))  # [D, B, 1 + AFTER, f]
            for key, v in (("at", a[:, :, 0].mean(0)), ("sum", a.sum(2).mean(0))):  # [B, f]
                for p, sign in enumerate((1, -1)):
                    merge(key, (l, p), sign * v.T, cur["ids"])
        return hook

    def attention(l: int):
        def hook(m, x, out):
            z = stash.pop(l)[:, AT].float().view(-1, H, geo[l].head_dim)  # [D·B, H, hd]: the token as the query
            scale = arch.attn_scale(model.layers[l], out[0])  # Gemma 4: the post-attention norm's scale
            if scale is not None:
                z = z * scale[:, AT, None, None]
            norm = torch.einsum("nhi,hij,nhj->nh", z, gram[l], z).clamp_min(0).sqrt()
            merge("h_out", l, norm.unflatten(0, (D, -1)).mean(0).T, cur["ids"])
            key = out[1][:, :, AT + 1:, AT].float().mean(-1)  # [D·B, H]: attention from the positions after it
            merge("h_key", l, key.unflatten(0, (D, -1)).mean(0).T, cur["ids"])
        return hook
    for l, m in enumerate(model.layers):
        m.mlp.down_proj.register_forward_pre_hook(neuron(l))
        m.self_attn.o_proj.register_forward_pre_hook(lambda mod, x, l=l: stash.__setitem__(l, x[0]))
        m.self_attn.register_forward_hook(attention(l))

    order = keep[torch.randperm(len(keep), generator=torch.Generator().manual_seed(0))]
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(order), args.batch):
            ids = order[i:i + args.batch].to(dev)
            cur["ids"] = ids
            batch = docs[:, None].repeat(1, len(ids), 1)  # [D, B, T]
            batch[:, :, AT] = ids
            model(input_ids=batch.flatten(0, 1))
            if (i // args.batch) % 100 == 0:
                print(f"{time.strftime('%H:%M:%S')} {i + len(ids)}/{len(order)} tokens, {time.time() - t0:.0f}s", flush=True)

    for l in range(L):
        cols = neuron_keys(l, f)
        for key, name in (("at", "drivers"), ("sum", "drivers_sum")):
            put(cols, name, torch.cat([best_ids[key][l, 0], best_ids[key][l, 1]]).cpu(),
                torch.cat([best[key][l, 0], best[key][l, 1]]).cpu())  # times the pole's sign, as top_val
        write(cols, neuron_table("drivers", l))
    cols = head_keys(L, H, arch.groups(MODEL))
    for key, name in (("h_out", "drivers"), ("h_key", "keys")):
        put(cols, name, best_ids[key].flatten(0, 1).cpu(), best[key].flatten(0, 1).cpu())
    write(cols, head_table("drivers"))
    print(f"{time.strftime('%H:%M:%S')} drivers written, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
