import math
import types

import pytest
import torch
import torch.autograd.forward_ad as fwAD
from transformers import Qwen3Config, Qwen3Model

from neuron_lens.rules import HALF_RULE_BETA, relp_rules
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
@pytest.mark.parametrize("build", [tiny_model, realistic_model])
def test_estimates_match_brute_force(build, n_valid):
    """The hand-written maps against autograd through the model under the RelP rules."""
    model = build()
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


def test_split_half_is_zero_for_equal_halves_and_nan_for_one():
    acc = Accumulator({"out": 2}, 3)
    acc.sums[("out", 0)] += 1
    acc.n_docs = [1, 0]
    assert all(math.isnan(x) for x in acc.split_half("out"))
    acc.sums[("out", 1)] += 2
    acc.n_docs = [1, 2]
    assert acc.split_half("out") == [0.0, 0.0]


def test_r_lens_matches_drowse():
    pytest.importorskip("drowse")
    from drowse.lens.fit import fit_jacobian_lens
    from drowse.lens.relp import relp_backward_rules

    model = tiny_model()
    torch.manual_seed(1)
    rows = [torch.randint(0, 50, (n,)).tolist() for n in (SKIP_FIRST + 9, SKIP_FIRST + 5, SKIP_FIRST + 12)]
    tok = types.SimpleNamespace(pad_token_id=0)
    with relp_backward_rules(model.layers):
        ref = fit_jacobian_lens(model, tok, ["x"] * len(rows), model.layers, dim_batch=4,
                                skip_first=SKIP_FIRST, input_id_rows=rows, max_seq_len=64)
    acc = Accumulator({"out": N_LAYERS - 1}, D)
    for i, row in enumerate(rows):
        r_lens_doc(model, torch.tensor(row), acc, i % 2, dim_batch=6)
    for layer in range(N_LAYERS - 1):
        assert torch.allclose(acc.mean("out")[layer], ref.jacobians[layer], atol=1e-5), layer


def test_relp_rules_keep_values_and_unpatch():
    model = tiny_model()
    e = torch.randn(1, 6, D)
    with torch.no_grad():
        plain = model(inputs_embeds=e).last_hidden_state
        with relp_rules(model):
            relp = model(inputs_embeds=e).last_hidden_state
    assert torch.equal(plain, relp)
    for module in (model.layers[0].mlp, model.layers[0].input_layernorm):
        assert "forward" not in module.__dict__ and not hasattr(module, "_rules_orig_forward")


def test_relp_mlp_tangent_is_half_rule_with_identity_rule():
    model = tiny_model()
    mlp = model.layers[0].mlp
    x, dx = torch.randn(4, D), torch.randn(4, D)
    with relp_rules(model), torch.no_grad(), fwAD.dual_level():
        tangent = fwAD.unpack_dual(mlp(fwAD.make_dual(x, dx))).tangent
    with torch.no_grad():
        g, u = mlp.gate_proj(x), mlp.up_proj(x)
        dg, du = dx @ mlp.gate_proj.weight.T, dx @ mlp.up_proj.weight.T
        expected = HALF_RULE_BETA * (torch.sigmoid(g) * dg * u + torch.nn.functional.silu(g) * du)
        expected = expected @ mlp.down_proj.weight.T
    assert torch.allclose(tangent, expected, atol=1e-5)


def test_relp_norm_tangent_holds_rms_constant():
    model = tiny_model()
    norm = model.layers[1].input_layernorm
    x, dx = torch.randn(3, D), torch.randn(3, D)
    with relp_rules(model), torch.no_grad(), fwAD.dual_level():
        tangent = fwAD.unpack_dual(norm(fwAD.make_dual(x, dx))).tangent
    scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    assert torch.allclose(tangent, norm.weight * dx * scale, atol=1e-6)


def test_r_lens_exports_to_drowse(tmp_path):
    pytest.importorskip("drowse")
    from drowse.lens.store import load_local_lens

    from neuron_lens.lenses import drowse_home, save_drowse_lens

    model = tiny_model()
    acc = Accumulator({"out": N_LAYERS - 1}, D)
    for i in range(3):
        r_lens_doc(model, torch.randint(0, 50, (SKIP_FIRST + 6 + i,)), acc, i % 2, dim_batch=8)
    meta = {"lens": "R", "model": "test/tiny", "layers": N_LAYERS, "model_layers": N_LAYERS,
            "corpus_spec": "file:t.jsonl", "rows_sha256": "0" * 64, "seq_len": 64, "dim_batch": 8}
    save_drowse_lens([meta], acc, tmp_path)
    with drowse_home(tmp_path):
        lens, sidecar = load_local_lens("test/tiny", "relp")
    assert lens.n_prompts == 3 and sidecar["method"] == "relp_cotangent_sum"
    for layer in range(N_LAYERS - 1):
        assert torch.equal(lens.jacobians[layer], acc.mean("out")[layer])


def test_input_reads_match_the_s_lens():
    from neuron_lens.transport import input_reads_doc
    model = tiny_model()
    ids = torch.randint(0, 50, (SKIP_FIRST + 8,))
    s = Accumulator({"mid": N_LAYERS}, D)
    s_lens_doc(model, ids, s, 0, dim_batch=5)
    layers = [2, 0, 1, 2]
    c = torch.randn(len(layers), D)
    got = input_reads_doc(model, ids, c, layers)
    want = torch.stack([c[i] @ s.mean("mid")[L] for i, L in enumerate(layers)])
    assert ((got - want).norm() / want.norm()).item() < 1e-5  # fp32 rounding, whatever the scale
