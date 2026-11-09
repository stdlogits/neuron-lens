"""The layer maps (``neuron_lens.layers``) on Gemma 4 against autograd through HF's own
``Gemma4UnifiedTextModel`` under the RelP rules, on a tiny model shaped like the 12B: sliding
and full layers with different head sizes and key-head counts, keys as values on the full
layers, partial RoPE there, four norms per layer with non-unit weights, layer scalars."""
import pytest
import torch

gemma = pytest.importorskip("transformers.models.gemma4_unified.modeling_gemma4_unified")
from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig

from neuron_lens.rules import relp_rules
from neuron_lens.transport import SKIP_FIRST, Accumulator, r_lens_doc, s_lens_doc, valid_positions

D, N_LAYERS, V = 16, 4, 50
TYPES = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]


def tiny_model(seed=0, k_eq_v=True):
    torch.manual_seed(seed)
    cfg = Gemma4UnifiedTextConfig(
        vocab_size=V, hidden_size=D, intermediate_size=32, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=16,
        num_global_key_value_heads=1, attention_k_eq_v=k_eq_v, layer_types=TYPES,
        rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                         "full_attention": {"rope_type": "proportional", "rope_theta": 1e6,
                                            "partial_rotary_factor": 0.25}},
        initializer_range=0.2, attn_implementation="eager")
    model = gemma.Gemma4UnifiedTextModel(cfg).float().eval().requires_grad_(False)
    with torch.no_grad():  # norms, scalars and RoPE away from their identity initialisations
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.3, 2.0)
        for layer in model.layers:
            layer.layer_scalar.uniform_(0.5, 1.5)
    return model


def mid_norm(block):
    return block.pre_feedforward_layernorm


def mlp_out(block, x):
    """The layer's output from its MLP input ``x`` (the residual after attention)."""
    return (x + block.post_feedforward_layernorm(block.mlp(block.pre_feedforward_layernorm(x)))) * block.layer_scalar


def run_with(model, ids, layer, h_new):
    mids, outs = {}, {}
    hooks = []
    for i, block in enumerate(model.layers):
        hooks.append(mid_norm(block).register_forward_pre_hook(lambda _m, a, i=i: mids.__setitem__(i, a[0])))

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


@pytest.mark.parametrize("n_valid", [7, 8])
@pytest.mark.parametrize("k_eq_v", [True, False])
def test_estimates_match_brute_force(n_valid, k_eq_v):
    model = tiny_model(k_eq_v=k_eq_v)
    assert (model.layers[1].self_attn.v_proj is None) == k_eq_v and model.layers[0].self_attn.v_proj is not None
    ids = torch.randint(0, V, (SKIP_FIRST + n_valid + 1,))
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
            assert torch.allclose(r.mean("out")[layer], J[:, valid].mean(1), atol=1e-4), layer
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


def _media_case(seed=7):
    """A tiny model and a document with a bidirectional block (an image) inside its valid
    positions, embeddings given, and the masks HF's model takes for it."""
    model = tiny_model(seed=seed)
    T = SKIP_FIRST + 14
    ids = torch.randint(0, V, (T,))
    torch.manual_seed(seed + 1)
    emb = torch.randn(T, D) * 3
    block = torch.zeros(T, dtype=torch.bool)
    block[SKIP_FIRST + 3:SKIP_FIRST + 9] = True
    n = T - 1
    causal = torch.full((n, n), float("-inf")).triu(1)
    b = block[:-1]
    masks = {"full_attention": causal[None, None],
             "sliding_attention": causal.masked_fill(b[:, None] & b[None, :], 0.0)[None, None]}
    return model, ids, emb, block, masks


def _run(model, emb, masks, layer=None, h_new=None):
    mids, outs = {}, {}
    hooks = []
    for i, block in enumerate(model.layers):
        hooks.append(block.pre_feedforward_layernorm.register_forward_pre_hook(lambda _m, a, i=i: mids.__setitem__(i, a[0])))

        def out_hook(_m, _a, o, i=i):
            if i == layer:
                o = h_new[None]
            outs[i] = o
            return o
        hooks.append(block.register_forward_hook(out_hook))
    model(inputs_embeds=emb[None], attention_mask=masks)
    for h in hooks:
        h.remove()
    return mids, outs


def test_true_rules_match_plain_autograd():
    """The media fits' mode (a bidirectional block, chosen rows, ``rules="true"``): both lenses against
    autograd through the model as it is (no RelP)."""
    model, ids, emb, block, masks = _media_case(seed=13)
    first = int(block.nonzero()[0])
    T = len(ids)
    pos = torch.arange(first, T - 1)
    rows = block[pos]
    s = Accumulator({"mid": N_LAYERS, "out": N_LAYERS}, D)
    s_lens_doc(model, ids, s, 0, dim_batch=7, source_rows=rows, target_rows=rows, embeds=emb, block=block,
               first=first, rules="true")
    seed_rows = block[pos] | (pos == T - 2)
    r = Accumulator({"out": N_LAYERS - 1}, D)
    r_lens_doc(model, ids, r, 0, dim_batch=7, seed_rows=seed_rows, read_rows=rows, embeds=emb, block=block,
               first=first, rules="true")
    src, tp = pos[rows], pos[seed_rows]
    for layer in range(N_LAYERS):
        for kind in ("mid", "out"):
            J = torch.autograd.functional.jacobian(
                lambda x: _run(model, x, masks)[0 if kind == "mid" else 1][layer][0, src].mean(0), emb[:-1])
            want = J[:, src].sum(1)
            assert ((s.mean(kind)[layer] - want).norm() / want.norm()).item() < 1e-5, ("S", kind, layer)
    _, outs = _run(model, emb[:-1], masks)
    for layer in range(N_LAYERS - 1):
        J = torch.autograd.functional.jacobian(
            lambda x: _run(model, emb[:-1], masks, layer, x)[1][N_LAYERS - 1][0, tp].sum(0), outs[layer][0].detach())
        want = J[:, src].mean(1)
        assert ((r.mean("out")[layer] - want).norm() / want.norm()).item() < 1e-5, ("R", layer)
