"""The layer maps (``neuron_lens.layers``) on the OLMo family against autograd through HF's own models under the rules
(``rules.relp_rules``), on tiny random models: OLMo 2 (norms on the sublayers' outputs only, q/k norms over the whole
projection), OLMo 3 (sliding layers, a RoPE table per layer type, YaRN on the full ones), OLMoE (pre-norm, a router whose
chosen experts keep their softmax weights, not renormalised over the top k) and Olmo Hybrid (Gated DeltaNet layers,
pre-norm, with separate projections, values wider than keys and β up to 2, beside attention layers without RoPE, normed
as OLMo 2's)."""
import pytest
import torch
from transformers import (
    Olmo2Config,
    Olmo2Model,
    Olmo3Config,
    Olmo3Model,
    OlmoeConfig,
    OlmoeModel,
)

from neuron_lens import layers
from neuron_lens.rules import relp_rules
from neuron_lens.transport import (
    SKIP_FIRST,
    Accumulator,
    r_lens_doc,
    s_lens_doc,
    valid_positions,
)

hybrid = pytest.importorskip("transformers.models.olmo_hybrid.modeling_olmo_hybrid")
from transformers.models.olmo_hybrid.configuration_olmo_hybrid import OlmoHybridConfig

D, V = 16, 50
TYPES = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"]
HYBRID = ["linear_attention", "linear_attention", "full_attention", "linear_attention"]
N_LAYERS = len(TYPES)
RTOL = 1e-4  # β near 2 makes DeltaNet's triangular solve amplify fp32 roundoff to ~2e-5 by the last layer; a wrong map
# misses by order one (dropping β's factor of 2 misses by 130%)


def tiny_model(kind: str):
    torch.manual_seed(0)
    common = dict(vocab_size=V, hidden_size=D, num_hidden_layers=N_LAYERS, num_attention_heads=4, num_key_value_heads=2,
                  initializer_range=0.2, attn_implementation="eager")
    if kind == "olmo2":
        model = Olmo2Model(Olmo2Config(**common, intermediate_size=24))
    elif kind == "olmo3":  # a window short enough to mask keys within the document
        model = Olmo3Model(Olmo3Config(
            **common, intermediate_size=24, layer_types=TYPES, sliding_window=6, max_position_embeddings=64,
            rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 5e5},
                             "full_attention": {"rope_type": "yarn", "rope_theta": 5e5, "factor": 8.0,
                                                "original_max_position_embeddings": 8, "attention_factor": 1.2,
                                                "beta_fast": 32.0, "beta_slow": 1.0}}))
    elif kind == "olmoe":
        model = OlmoeModel(OlmoeConfig(**common, intermediate_size=12, num_experts=8, num_experts_per_tok=3,
                                       norm_topk_prob=False))
    else:
        model = hybrid.OlmoHybridModel(OlmoHybridConfig(
            **common, intermediate_size=24, layer_types=HYBRID, linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=12, linear_conv_kernel_dim=4, linear_allow_neg_eigval=True,
            rope_parameters={"rope_theta": None}, pad_token_id=1))
    model = model.float().eval().requires_grad_(False)
    with torch.no_grad():  # norms and decays away from their initialisations
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.uniform_(0.3, 2.0)
            elif name.endswith("A_log"):
                p.copy_(torch.empty_like(p).uniform_(0.02, 0.5).log())
            elif name.endswith("dt_bias"):
                p.uniform_(-1, 1)
            elif "conv1d" in name:
                p.mul_(3)
            elif "b_proj" in name:  # β across (0, 2)
                p.mul_(4)
    return model


def close(a, b, *where) -> None:
    worst = ((a - b).abs() - RTOL * b.abs()).argmax()
    assert torch.allclose(a, b, atol=1e-4, rtol=RTOL), (*where, f"miss {(a - b).abs().max():.2e} at worst "
                                                       f"{(a - b).flatten()[worst]:.2e} of {b.flatten()[worst]:.2e}")


def pre_norm(block) -> bool:
    return hasattr(block, "input_layernorm")


def run_with(model, ids, layer, h_new):
    """MLP inputs and outputs per layer, with layer ``layer``'s output replaced by ``h_new`` (``layer=None``: ``h_new``
    replaces the embeddings)."""
    mids, outs = {}, {}
    hooks = []
    for i, block in enumerate(model.layers):
        reads = block.post_attention_layernorm if pre_norm(block) else block.mlp  # the MLP's input: the residual
        hooks.append(reads.register_forward_pre_hook(lambda _m, a, i=i: mids.__setitem__(i, a[0])))

        def out_hook(_m, _a, o, i=i):
            if i == layer:
                o = h_new[None]
            outs[i] = o
            return o
        hooks.append(block.register_forward_hook(out_hook))
    model(inputs_embeds=h_new[None]) if layer is None else model(input_ids=ids[None])
    for h in hooks:
        h.remove()
    return mids, outs


def mlp_out(block, x):
    if pre_norm(block):
        return x + block.mlp(block.post_attention_layernorm(x))
    return x + block.post_feedforward_layernorm(block.mlp(x))


@pytest.mark.parametrize("kind,chunk", [("olmo2", 32), ("olmo3", 32), ("olmoe", 32), ("hybrid", 32), ("hybrid", 5)])
def test_estimates_match_brute_force(kind, chunk, monkeypatch):
    """Chunks of 5 split Olmo Hybrid's positions into several, with padding: DeltaNet's products cross chunks."""
    monkeypatch.setattr(layers, "CHUNK", chunk)
    model = tiny_model(kind)
    ids = torch.randint(0, V, (SKIP_FIRST + 9,))
    valid = valid_positions(len(ids))
    r = Accumulator({"out": N_LAYERS - 1, "mid": N_LAYERS}, D)
    r_lens_doc(model, ids, r, 0, dim_batch=7)
    s = Accumulator({"mid": N_LAYERS, "out": N_LAYERS}, D)
    s_lens_doc(model, ids, s, 0, dim_batch=7)
    emb = model.embed_tokens(ids[None])[0].detach()
    with relp_rules(model):
        mids, outs = run_with(model, ids, None, emb)
        for layer in range(N_LAYERS - 1):  # J_L = mean_t Σ_t' ∂final(t')/∂h_L(t)
            h = outs[layer][0].detach()
            J = torch.autograd.functional.jacobian(
                lambda x: run_with(model, ids, layer, x)[1][N_LAYERS - 1][0, valid].sum(0), h)
            close(r.mean("out")[layer], J[:, valid].mean(1), "out", layer)
        for layer in range(N_LAYERS):  # from layer L's MLP input
            block = model.layers[layer]

            def final_from_mid(x, layer=layer, block=block):
                return run_with(model, ids, layer, mlp_out(block, x[None])[0])[1][N_LAYERS - 1][0, valid].sum(0)
            J = torch.autograd.functional.jacobian(final_from_mid, mids[layer][0].detach())
            close(r.mean("mid")[layer], J[:, valid].mean(1), "mid", layer)
        for layer in range(N_LAYERS):  # K_L = mean_t Σ_s ∂h_L(t)/∂e(s)
            for kind_ in ("mid", "out"):
                K = torch.autograd.functional.jacobian(
                    lambda x, layer=layer, kind_=kind_: run_with(model, ids, None, x)[0 if kind_ == "mid" else 1]
                    [layer][0, valid].mean(0), emb)
                close(s.mean(kind_)[layer], K[:, valid].sum(1), kind_, layer)



def test_delta_pattern():
    """The catalog's DeltaNet pattern (``layers.delta_pattern``) is exact for Olmo Hybrid's separate projections, values
    wider than keys and β up to 2: ``W_out (gate ⊙ A v)`` is the mixer's output."""
    model = tiny_model("hybrid")
    x = torch.randn(2, SKIP_FIRST + 9, D)
    for block in model.layers:
        if hasattr(block, "linear_attn"):
            la = block.linear_attn
            p = layers.delta_pattern(la, x)
            v = torch.nn.functional.silu(p.c)[..., 2 * la.key_dim:].unflatten(-1, (la.num_v_heads, la.head_v_dim))
            o = (p.A @ v.transpose(1, 2)).transpose(1, 2)
            out = (p.gate * o).flatten(2) @ layers.delta_parts(la).out.T
            assert torch.allclose(out, la(hidden_states=x), atol=1e-5)
