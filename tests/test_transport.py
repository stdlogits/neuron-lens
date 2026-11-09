"""The estimators against brute-force Jacobians of the patched model, on a tiny Qwen3."""
import pytest
import torch
from transformers import Qwen3Config, Qwen3Model

from neuron_lens.rules import relp_rules
from neuron_lens.transport import (
    SKIP_FIRST,
    Accumulator,
    r_lens_doc,
    s_lens_doc,
    valid_positions,
)

D, N_LAYERS = 16, 3


def tiny_model(**overrides) -> Qwen3Model:
    torch.manual_seed(0)
    cfg = dict(vocab_size=50, hidden_size=D, intermediate_size=32, num_hidden_layers=N_LAYERS,
               num_attention_heads=2, num_key_value_heads=1, head_dim=8, attn_implementation="eager")
    return Qwen3Model(Qwen3Config(**cfg | overrides)).float().eval().requires_grad_(False)


def realistic_model() -> Qwen3Model:
    """Two query heads per KV head (8B has four), weights large enough that the MLP's RelP
    coefficients are not near zero, and random norm weights, which the default init leaves at 1:
    the tiny model cannot tell a swapped GQA layout or a dropped norm weight from the right ones."""
    model = tiny_model(num_attention_heads=4, num_key_value_heads=2, initializer_range=0.2)
    torch.manual_seed(1)
    for name, p in model.named_parameters():
        if name.endswith("norm.weight"):  # input, post-attention, q, k and final norms
            p.uniform_(0.3, 2.0)
    return model


def run_with(model, ids, layer: int | None, h_new: torch.Tensor):
    """Mid residuals and outputs per layer, with layer ``layer``'s output replaced by ``h_new``
    (``layer=None``: ``h_new`` replaces the embeddings)."""
    mids, outs = {}, {}
    hooks = []
    for i, block in enumerate(model.layers):
        hooks.append(block.post_attention_layernorm.register_forward_pre_hook(
            lambda _m, a, i=i: mids.__setitem__(i, a[0])))

        def out_hook(_m, _a, o, i=i):
            if i == layer:
                o = h_new[None]
            outs[i] = o
            return o
        hooks.append(block.register_forward_hook(out_hook))
    if layer is None:
        model(inputs_embeds=h_new[None])
    else:
        model(input_ids=ids[None])
    for h in hooks:
        h.remove()
    return mids, outs


def exact(model, ids, f):
    """Jacobian of ``f(mids, outs)`` w.r.t. the embeddings (``[*out, T, d]``)."""
    e = model.embed_tokens(ids[None])[0].detach()
    return torch.autograd.functional.jacobian(lambda x: f(*run_with(model, ids, None, x)), e)


def exact_from(model, ids, layer, f):
    """Jacobian of ``f(outs)`` w.r.t. layer ``layer``'s output (``[*out, T, d]``)."""
    _, outs = run_with(model, ids, None, model.embed_tokens(ids[None])[0].detach())
    h = outs[layer][0].detach()
    return torch.autograd.functional.jacobian(lambda x: f(run_with(model, ids, layer, x)[1]), h)


@pytest.mark.parametrize("n_valid", [7, 8])  # keys padded to a multiple of 8 (Tp), and not
def test_estimates_match_brute_force(n_valid):
    """The hand-written maps against autograd through the model under the RelP rules."""
    model = realistic_model()
    ids = torch.randint(0, 50, (SKIP_FIRST + n_valid,))
    valid = valid_positions(len(ids))
    r = Accumulator({"out": N_LAYERS - 1, "mid": N_LAYERS}, D)
    r_lens_doc(model, ids, r, 0, dim_batch=7)
    s = Accumulator({"mid": N_LAYERS, "out": N_LAYERS}, D)
    s_lens_doc(model, ids, s, 0, dim_batch=7)
    with relp_rules(model):
        for layer in range(N_LAYERS - 1):  # J_L = mean_t Σ_t' ∂final(t')/∂h_L(t)
            J = exact_from(model, ids, layer, lambda outs: outs[N_LAYERS - 1][0, valid].sum(0))
            assert torch.allclose(r.mean("out")[layer], J[:, valid].mean(1), atol=1e-4), layer
        emb = model.embed_tokens(ids[None])[0].detach()
        for layer in range(N_LAYERS):  # the same from layer L's MLP input
            block = model.layers[layer]
            mid = run_with(model, ids, None, emb)[0][layer][0].detach()

            def final_from_mid(x, layer=layer, block=block):
                out = x + block.mlp(block.post_attention_layernorm(x[None]))[0]
                return run_with(model, ids, layer, out)[1][N_LAYERS - 1][0, valid].sum(0)
            J = torch.autograd.functional.jacobian(final_from_mid, mid)
            assert torch.allclose(r.mean("mid")[layer], J[:, valid].mean(1), atol=1e-4), ("mid", layer)
        for layer in range(N_LAYERS):  # K_L = mean_t Σ_s ∂h_L(t)/∂e(s)
            for kind in ("mid", "out"):
                K = exact(model, ids, lambda mids, outs, layer=layer, kind=kind:
                          (mids if kind == "mid" else outs)[layer][0, valid].mean(0))
                assert torch.allclose(s.mean(kind)[layer], K[:, valid].sum(1), atol=1e-4), (kind, layer)
