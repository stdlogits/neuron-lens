"""The exact split of an activation over its sources, on a tiny random Qwen3, and the merge over firings."""
import torch
from torch.nn.functional import silu
from transformers import Qwen3Config, Qwen3Model

from neuron_lens.sources import merged, split, top_by_row


def tiny():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=64, attn_implementation="eager")
    m = Qwen3Model(cfg).float().eval().requires_grad_(False)
    for name, p in m.named_parameters():  # away from the initial unit norm weights
        if "norm" in name:
            p.data.uniform_(0.5, 1.5)
    return m


def test_shares_sum_to_one_and_the_activation_is_the_models():
    model = tiny()
    H, M, n, b, t = model.config.num_attention_heads, 2, 11, 0, 7
    ids = torch.randint(0, 50, (1, 14), generator=torch.Generator().manual_seed(2))
    got = {"z": [], "a": [], "gate": None}
    hooks = [model.embed_tokens.register_forward_hook(lambda m, a, o: got.__setitem__("emb", o[b, t]))]
    for l in range(M + 1):
        hooks.append(model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m, a: got["z"].append(a[0][b, t])))
    for l in range(M):
        hooks.append(model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, a: got["a"].append(a[0][b, t])))
    lay = model.layers[M]
    hooks.append(lay.post_attention_layernorm.register_forward_pre_hook(lambda m, a: got.__setitem__("x", a[0][b, t])))
    hooks.append(lay.mlp.down_proj.register_forward_pre_hook(lambda m, a: got.__setitem__("true_a", a[0][b, t, n])))
    with torch.no_grad():
        model(input_ids=ids)
    for h in hooks:
        h.remove()
    norm = lay.post_attention_layernorm
    x = got["x"][None]
    inv = torch.rsqrt(x.pow(2).mean(-1) + norm.variance_epsilon)[:, None]
    rg, ru = (lay.mlp.gate_proj.weight[n] * norm.weight)[None], (lay.mlp.up_proj.weight[n] * norm.weight)[None]
    s = split(rg, ru, x, inv, got["emb"][None], [z[None] for z in got["z"]], [a[None] for a in got["a"]],
              [model.layers[l].mlp.down_proj.weight for l in range(M)], [model.layers[l].self_attn.o_proj.weight for l in range(M + 1)], H)
    assert torch.allclose(s["a"][0], got["true_a"], atol=1e-5)
    assert torch.allclose(s["a"], silu(s["g"]) * ((ru * x).sum(-1) * inv[:, 0]), atol=1e-5)
    assert torch.allclose(s["emb"] + s["heads"].sum(1) + s["neurons"].sum(1), torch.ones(1), atol=1e-4)


def test_merge_and_top_by_row():
    rows, ids, total = merged(torch.tensor([0, 0, 1, 0]), torch.tensor([3, 4, 3, 3]), torch.tensor([1.0, -2.0, 5.0, 0.5]), 10)
    got = {(int(r), int(i)): float(v) for r, i, v in zip(rows, ids, total)}
    assert got == {(0, 3): 1.5, (0, 4): -2.0, (1, 3): 5.0}
    i, v = top_by_row(rows, ids, total, 3, 2)
    assert i[0].tolist() == [4, 3] and i[1].tolist() == [3, -1] and i[2].tolist() == [-1, -1]
    assert v[0, 0] == -2.0 and v[1, 1] != v[1, 1]  # NaN pad
