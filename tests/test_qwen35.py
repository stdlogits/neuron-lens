"""The layer maps (``neuron_lens.layers``) on Qwen3.5 against autograd through HF's own
``Qwen3_5MoeTextModel`` (and the dense ``Qwen3_5TextModel``) under the rules
(``rules.relp_rules``), on tiny models shaped like Qwen3.5-35B-A3B: Gated DeltaNet and gated
attention layers, two value heads per key head, partial RoPE, zero-centred norms with non-zero
weights, routed experts with a shared one. The reference runs HF's chunked delta rule, so it
checks the explicit-matrix derivatives independently."""
import pytest
import torch

q35 = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from neuron_lens import layers
from neuron_lens.rules import relp_rules
from neuron_lens.transport import SKIP_FIRST, Accumulator, input_reads_doc, r_lens_doc, s_lens_doc, valid_positions

D, V = 16, 50
TYPES = ["linear_attention", "linear_attention", "full_attention", "linear_attention"]
N_LAYERS = len(TYPES)


def tiny_model(seed=0, moe=True):
    torch.manual_seed(seed)
    common = dict(
        vocab_size=V, hidden_size=D, num_hidden_layers=N_LAYERS, layer_types=TYPES,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, initializer_range=0.2, attn_implementation="eager",
        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True})
    if moe:
        cfg = Qwen3_5MoeTextConfig(**common, num_experts=8, num_experts_per_tok=3, moe_intermediate_size=12,
                                   shared_expert_intermediate_size=12)
        model = q35.Qwen3_5MoeTextModel(cfg)
    else:
        model = Qwen3_5TextModel(Qwen3_5TextConfig(**common, intermediate_size=24))
    model = model.float().eval().requires_grad_(False)
    with torch.no_grad():  # norms and decays away from their initialisations
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.uniform_(-0.5, 0.5) if "linear_attn.norm" not in name else p.uniform_(0.5, 1.5)
            elif name.endswith("A_log"):
                p.copy_(torch.empty_like(p).uniform_(0.02, 0.5).log())
            elif name.endswith("dt_bias"):
                p.uniform_(-1, 1)
            elif "conv1d" in name:
                p.mul_(3)
    return model


def run_with(model, ids, layer, h_new):
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
    model(inputs_embeds=h_new[None]) if layer is None else model(input_ids=ids[None])
    for h in hooks:
        h.remove()
    return mids, outs


def mlp_out(block, x):
    return x + block.mlp(block.post_attention_layernorm(x))


@pytest.mark.parametrize("moe,chunk", [(True, 32), (False, 32), (True, 5)])
def test_estimates_match_brute_force(moe, chunk, monkeypatch):
    """Chunks of 5 split the 24 positions into five, with padding: DeltaNet's products cross
    chunks through their states."""
    monkeypatch.setattr(layers, "CHUNK", chunk)
    model = tiny_model(moe=moe)
    ids = torch.randint(0, V, (SKIP_FIRST + 9,))
    valid = valid_positions(len(ids))
    r = Accumulator({"out": N_LAYERS - 1, "mid": N_LAYERS}, D)
    r_lens_doc(model, ids, r, 0, dim_batch=7)
    s = Accumulator({"mid": N_LAYERS, "out": N_LAYERS}, D)
    s_lens_doc(model, ids, s, 0, dim_batch=7)
    emb = model.embed_tokens(ids[None])[0].detach()
    with relp_rules(model):
        _, outs = run_with(model, ids, None, emb)
        for layer in range(N_LAYERS - 1):  # J_L = mean_t Σ_t' ∂final(t')/∂h_L(t)
            h = outs[layer][0].detach()
            J = torch.autograd.functional.jacobian(
                lambda x: run_with(model, ids, layer, x)[1][N_LAYERS - 1][0, valid].sum(0), h)
            assert torch.allclose(r.mean("out")[layer], J[:, valid].mean(1), atol=1e-4), ("out", layer)
        for layer in range(N_LAYERS):  # from layer L's MLP input
            block = model.layers[layer]
            mid = run_with(model, ids, None, emb)[0][layer][0].detach()

            def final_from_mid(x, layer=layer, block=block):
                return run_with(model, ids, layer, mlp_out(block, x[None])[0])[1][N_LAYERS - 1][0, valid].sum(0)
            J = torch.autograd.functional.jacobian(final_from_mid, mid)
            assert torch.allclose(r.mean("mid")[layer], J[:, valid].mean(1), atol=1e-4), ("mid", layer)
        for layer in range(N_LAYERS):  # K_L = mean_t Σ_s ∂h_L(t)/∂e(s)
            for kind in ("mid", "out"):
                K = torch.autograd.functional.jacobian(
                    lambda x, layer=layer, kind=kind: run_with(model, ids, None, x)[0 if kind == "mid" else 1]
                    [layer][0, valid].mean(0), emb)
                assert torch.allclose(s.mean(kind)[layer], K[:, valid].sum(1), atol=1e-4), (kind, layer)


def test_primal_matches_the_model():
    """``linearize``'s forward reproduces HF's: the last layer's output (the hidden state before the
    final norm), and routing weights that sum to 1."""
    model = tiny_model(seed=2)
    ids = torch.randint(0, V, (SKIP_FIRST + 40,))
    outs = {}
    hook = model.layers[-1].register_forward_hook(lambda _m, _a, o: outs.__setitem__(0, o))
    model(input_ids=ids[None, :-1])
    hook.remove()
    with torch.no_grad():
        maps, h = layers._run(model, ids)
    assert len(maps) == N_LAYERS
    for m in maps:
        assert torch.allclose(m.mlp.slot_w.sum(-1), torch.ones(()), atol=1e-5)
    assert torch.allclose(h, outs[0][0], atol=1e-5)


def test_input_reads_match_the_s_lens():
    """Covectors pulled down from their layers' MLP inputs equal ``c · K_L`` from the S-lens."""
    model = tiny_model(seed=3)
    ids = torch.randint(0, V, (SKIP_FIRST + 9,))
    s = Accumulator({"mid": N_LAYERS}, D)
    s_lens_doc(model, ids, s, 0, dim_batch=5)
    layers = [0, 3, 1, 3, 2, 0]
    c = torch.randn(len(layers), D)
    got = input_reads_doc(model, ids, c, layers)
    want = torch.stack([c[i] @ s.mean("mid")[L] for i, L in enumerate(layers)])
    assert ((got - want).norm() / want.norm()).item() < 1e-5


def test_rules_keep_values_and_unpatch():
    model = tiny_model()
    e = torch.randn(1, SKIP_FIRST + 4, D)
    with torch.no_grad():
        plain = model(inputs_embeds=e).last_hidden_state
        with relp_rules(model):
            ruled = model(inputs_embeds=e).last_hidden_state
        assert torch.allclose(plain, ruled, atol=1e-5)
    for module in (model.layers[0].linear_attn, model.layers[2].self_attn, model.layers[1].mlp,
                   model.layers[1].input_layernorm):
        assert "forward" not in module.__dict__ and not hasattr(module, "_rules_orig_forward")
