"""The estimators against brute-force Jacobians of the patched model, on a tiny GPT-NeoX (Pythia's
architecture): LayerNorms with biases, one fused QKV projection with biases, RoPE on a quarter of each
head, an ungated GELU MLP with biases, and parallel blocks (attention and the MLP both read the layer's
input, so ``mid`` is the layer's input), or sequential ones."""
import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXModel

from neuron_lens.rules import relp_rules
from neuron_lens.transport import SKIP_FIRST, Accumulator, r_lens_doc, s_lens_doc, valid_positions

D, N_LAYERS = 32, 3


def tiny_model(parallel: bool) -> GPTNeoXModel:
    """Two heads of 16 (RoPE on 4 dims of each), weights large enough that GELU's factors are not
    near ½, and random norm scales and biases everywhere, which the default init leaves at 1 and 0."""
    torch.manual_seed(0)
    cfg = GPTNeoXConfig(vocab_size=50, hidden_size=D, intermediate_size=64, num_hidden_layers=N_LAYERS,
                        num_attention_heads=2, hidden_act="gelu", use_parallel_residual=parallel,
                        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
                        initializer_range=0.2, attn_implementation="eager")
    model = GPTNeoXModel(cfg).float().eval().requires_grad_(False)
    torch.manual_seed(1)
    for name, p in model.named_parameters():
        if "layernorm" in name and name.endswith("weight"):
            p.uniform_(0.3, 2.0)
        elif name.endswith("bias"):
            p.uniform_(-0.5, 0.5)
    return model


def run_with(model, ids, layer: int | None, h_new: torch.Tensor):
    """The residual each layer's MLP reads, and each layer's output, with layer ``layer``'s output
    replaced by ``h_new`` (``layer=None``: ``h_new`` replaces the embeddings)."""
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


def embeddings(model, ids):
    return model.get_input_embeddings()(ids[None])[0].detach()


def exact(model, ids, f):
    """Jacobian of ``f(mids, outs)`` w.r.t. the embeddings (``[*out, T, d]``)."""
    return torch.autograd.functional.jacobian(lambda x: f(*run_with(model, ids, None, x)), embeddings(model, ids))


def exact_from(model, ids, layer, f):
    """Jacobian of ``f(outs)`` w.r.t. layer ``layer``'s output (``[*out, T, d]``)."""
    _, outs = run_with(model, ids, None, embeddings(model, ids))
    h = outs[layer][0].detach()
    return torch.autograd.functional.jacobian(lambda x: f(run_with(model, ids, layer, x)[1]), h)


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("n_valid", [7, 8])  # keys padded to a multiple of 8 (Tp), and not
def test_estimates_match_brute_force(n_valid, parallel):
    """The hand-written maps against autograd through the model under the RelP rules."""
    model = tiny_model(parallel)
    ids = torch.randint(0, 50, (SKIP_FIRST + n_valid,))
    valid = valid_positions(len(ids))
    r = Accumulator({"out": N_LAYERS - 1, "mid": N_LAYERS}, D)
    r_lens_doc(model, ids, r, 0, dim_batch=7)
    s = Accumulator({"mid": N_LAYERS, "out": N_LAYERS}, D)
    s_lens_doc(model, ids, s, 0, dim_batch=7)
    final = lambda outs: outs[N_LAYERS - 1][0, valid].sum(0)
    with relp_rules(model):
        for layer in range(N_LAYERS - 1):  # J_L = mean_t Σ_t' ∂final(t')/∂h_L(t)
            J = exact_from(model, ids, layer, final)
            assert torch.allclose(r.mean("out")[layer], J[:, valid].mean(1), atol=1e-4), layer
        for layer in range(N_LAYERS):  # the same from layer L's MLP input
            if parallel:  # the layer's input: the layer below's output, or the embeddings
                J = exact(model, ids, lambda _mids, outs: final(outs)) if layer == 0 else \
                    exact_from(model, ids, layer - 1, final)
            else:
                block = model.layers[layer]
                mid = run_with(model, ids, None, embeddings(model, ids))[0][layer][0].detach()

                def final_from_mid(x, layer=layer, block=block):
                    out = x + block.mlp(block.post_attention_layernorm(x[None]))[0]
                    return final(run_with(model, ids, layer, out)[1])
                J = torch.autograd.functional.jacobian(final_from_mid, mid)
            assert torch.allclose(r.mean("mid")[layer], J[:, valid].mean(1), atol=1e-4), ("mid", layer)
        for layer in range(N_LAYERS):  # K_L = mean_t Σ_s ∂h_L(t)/∂e(s)
            for kind in ("mid", "out"):
                K = exact(model, ids, lambda mids, outs, layer=layer, kind=kind:
                          (mids if kind == "mid" else outs)[layer][0, valid].mean(0))
                assert torch.allclose(s.mean(kind)[layer], K[:, valid].sum(1), atol=1e-4), (kind, layer)
