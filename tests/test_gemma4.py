"""The Gemma 4 layer maps (``neuron_lens.gemma4``) against autograd through HF's own
``Gemma4UnifiedTextModel`` under the RelP rules, on a tiny model shaped like the 12B: sliding
and full layers with different head sizes and key-head counts, keys as values on the full
layers, partial RoPE there, four norms per layer with non-unit weights, layer scalars."""
import pytest
import torch
import torch.autograd.forward_ad as fwAD

gemma = pytest.importorskip("transformers.models.gemma4_unified.modeling_gemma4_unified")
from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig

from neuron_lens import gemma4
from neuron_lens.rules import HALF_RULE_BETA, relp_rules
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
    assert gemma4.is_gemma4(model)
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


def test_primal_matches_the_model():
    """``linearize``'s forward reproduces HF's (attention probabilities and MLP gates feed
    the map, so a wrong primal shows here before it shows in the Jacobians)."""
    model = tiny_model()
    ids = torch.randint(0, V, (SKIP_FIRST + 9,))
    maps = gemma4.linearize(model, ids)
    caught = {}
    hooks = [b.mlp.gate_proj.register_forward_hook(lambda _m, _a, o, i=i: caught.__setitem__(i, o[0]))
             for i, b in enumerate(model.layers)]
    model(input_ids=ids[None, :-1])
    for h in hooks:
        h.remove()
    for i, m in enumerate(maps):
        act = model.layers[i].mlp.act_fn(caught[i][SKIP_FIRST:])
        assert torch.allclose(m.cu, HALF_RULE_BETA * act, atol=1e-5), i
        assert torch.allclose(m.p.sum(-1), torch.ones(()), atol=1e-5)


def test_relp_mlp_tangent_uses_the_gelu_factor():
    model = tiny_model()
    mlp = model.layers[0].mlp
    x, dx = torch.randn(4, D), torch.randn(4, D)
    with relp_rules(model), torch.no_grad(), fwAD.dual_level():
        tangent = fwAD.unpack_dual(mlp(fwAD.make_dual(x, dx))).tangent
    with torch.no_grad():
        g, u = mlp.gate_proj(x), mlp.up_proj(x)
        dg, du = dx @ mlp.gate_proj.weight.T, dx @ mlp.up_proj.weight.T
        phi = mlp.act_fn(g) / g  # identity rule: act(g) = g·φ(g)
        expected = HALF_RULE_BETA * (phi * dg * u + mlp.act_fn(g) * du) @ mlp.down_proj.weight.T
    assert torch.allclose(tangent, expected, atol=1e-5)


def test_relp_rules_keep_values_and_unpatch():
    model = tiny_model()
    e = torch.randn(1, SKIP_FIRST + 4, D)
    with torch.no_grad():
        plain = model(inputs_embeds=e).last_hidden_state
        with relp_rules(model):
            relp = model(inputs_embeds=e).last_hidden_state
    assert torch.equal(plain, relp)
    layer = model.layers[1]
    for module in (layer.mlp, layer.input_layernorm, layer.post_feedforward_layernorm):
        assert "forward" not in module.__dict__ and not hasattr(module, "_rules_orig_forward")


def test_refuses_documents_past_the_sliding_window():
    model = tiny_model()
    model.layers[0].self_attn.sliding_window = SKIP_FIRST + 4
    with pytest.raises(ValueError, match="sliding window"):
        gemma4.linearize(model, torch.randint(0, V, (SKIP_FIRST + 9,)))


def test_input_reads_match_the_s_lens():
    """Covectors pulled down from their layers' MLP inputs equal ``c · K_L`` from the S-lens."""
    from neuron_lens.transport import input_reads_doc
    model = tiny_model(seed=3)
    ids = torch.randint(0, V, (SKIP_FIRST + 9,))
    s = Accumulator({"mid": N_LAYERS}, D)
    s_lens_doc(model, ids, s, 0, dim_batch=5)
    layers = [0, 3, 1, 3, 2, 0]
    c = torch.randn(len(layers), D)
    got = input_reads_doc(model, ids, c, layers)
    want = torch.stack([c[i] @ s.mean("mid")[L] for i, L in enumerate(layers)])
    assert ((got - want).norm() / want.norm()).item() < 1e-5  # fp32 rounding, whatever the scale


def test_input_reads_with_embeds_and_a_bidirectional_block():
    """Seeded and summed on a block of positions that attend to each other both ways in sliding
    layers (an image), with embeddings given: autograd through HF's model under the RelP rules,
    with the same masks passed in, gives the same reads."""
    from neuron_lens.transport import input_reads_doc
    model = tiny_model(seed=5)
    T = SKIP_FIRST + 10
    ids = torch.randint(0, V, (T,))
    torch.manual_seed(6)
    emb = torch.randn(T, D) * 3
    block = torch.zeros(T, dtype=torch.bool)
    block[SKIP_FIRST + 2:SKIP_FIRST + 7] = True
    valid = valid_positions(T)
    rows = block[valid]
    layers = [3, 1, 2, 0]
    c = torch.randn(len(layers), D)
    got = input_reads_doc(model, ids, c, layers, seed_rows=rows, source_rows=rows, embeds=emb, block=block)
    n = T - 1
    causal = torch.full((n, n), float("-inf")).triu(1)
    b = block[:-1]
    sliding = causal.masked_fill(b[:, None] & b[None, :], 0.0)
    masks = {"full_attention": causal[None, None], "sliding_attention": sliding[None, None]}
    with relp_rules(model):
        for i, L in enumerate(layers):
            def f(x, L=L):
                mids = {}
                h = model.layers[L].pre_feedforward_layernorm.register_forward_pre_hook(
                    lambda _m, a: mids.__setitem__(0, a[0]))
                model(inputs_embeds=x[None], attention_mask=masks)
                h.remove()
                return mids[0][0, valid][rows].mean(0) @ c[i]
            g = torch.autograd.functional.jacobian(f, emb[:-1])  # [T-1, d]
            want = g[valid][rows].sum(0)
            assert ((got[i] - want).norm() / want.norm()).item() < 1e-5, L


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


@pytest.mark.parametrize("trim", [False, True])
def test_s_lens_rows_basis_block_match_autograd(trim):
    model, ids, emb, block, masks = _media_case()
    first = int(block.nonzero()[0]) if trim else SKIP_FIRST
    T = len(ids)
    pos = torch.arange(first, T - 1)
    rows = block[pos]
    torch.manual_seed(9)
    basis = torch.linalg.qr(torch.randn(D, 5))[0].T  # [5, D] orthonormal rows
    s = Accumulator({"mid": N_LAYERS, "out": N_LAYERS}, D, d_in=5)
    s_lens_doc(model, ids, s, 0, dim_batch=3, basis=basis, source_rows=rows, target_rows=rows,
               embeds=emb, block=block, first=first)
    src = pos[rows]
    with relp_rules(model):
        for layer in range(N_LAYERS):
            for kind in ("mid", "out"):
                J = torch.autograd.functional.jacobian(
                    lambda x: _run(model, x, masks)[0 if kind == "mid" else 1][layer][0, src].mean(0), emb[:-1])
                want = J[:, src].sum(1) @ basis.T  # [d, k]
                got = s.mean(kind)[layer]
                assert ((got - want).norm() / want.norm()).item() < 1e-5, (kind, layer)


@pytest.mark.parametrize("trim", [False, True])
def test_r_lens_rows_match_autograd(trim):
    model, ids, emb, block, masks = _media_case(seed=11)
    first = int(block.nonzero()[0]) if trim else SKIP_FIRST
    T = len(ids)
    pos = torch.arange(first, T - 1)
    seed_rows = block[pos] | (pos == T - 2)  # t': the block and one later position (not those between)
    read_rows = block[pos]  # t: the block
    r = Accumulator({"out": N_LAYERS - 1, "mid": N_LAYERS}, D)
    r_lens_doc(model, ids, r, 0, dim_batch=7, seed_rows=seed_rows, read_rows=read_rows,
               embeds=emb, block=block, first=first)
    tp, t = pos[seed_rows], pos[read_rows]
    with relp_rules(model):
        _, outs = _run(model, emb[:-1], masks)
        for layer in range(N_LAYERS - 1):
            h = outs[layer][0].detach()
            J = torch.autograd.functional.jacobian(
                lambda x: _run(model, emb[:-1], masks, layer, x)[1][N_LAYERS - 1][0, tp].sum(0), h)
            want = J[:, t].mean(1)
            assert ((r.mean("out")[layer] - want).norm() / want.norm()).item() < 1e-5, ("out", layer)
        mids, _ = _run(model, emb[:-1], masks)
        for layer in range(N_LAYERS):
            blk = model.layers[layer]
            mid = mids[layer][0].detach()
            J = torch.autograd.functional.jacobian(
                lambda x: _run(model, emb[:-1], masks, layer, mlp_out(blk, x[None])[0])[1][N_LAYERS - 1][0, tp].sum(0), mid)
            want = J[:, t].mean(1)
            assert ((r.mean("mid")[layer] - want).norm() / want.norm()).item() < 1e-5, ("mid", layer)


def test_true_rules_match_plain_autograd():
    """``rules="true"``: both lenses against autograd through the model as it is (no RelP)."""
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
