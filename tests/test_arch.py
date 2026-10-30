"""The pipeline's Gemma 4 geometry (``neuron_lens/arch.py``) against
HF's own modules, on test_gemma4's tiny model: per-layer head sizes and key heads, keys as values
on full layers, each head's write through the post-attention norm, and the RoPE angles and
relative rotation ``head_qk.py`` builds its QK reads from (partial RoPE on full layers)."""
import pytest
import torch

pytest.importorskip("transformers.models.gemma4_unified.modeling_gemma4_unified")
from test_gemma4 import TYPES, tiny_model

from neuron_lens import arch
from neuron_lens.transport import _rotate_half


def _captures(model, ids):
    cap = {}
    hooks = []
    for i, block in enumerate(model.layers):
        at = block.self_attn
        hooks.append(block.input_layernorm.register_forward_hook(lambda _m, _a, o, i=i: cap.__setitem__(("x", i), o)))
        hooks.append(at.q_norm.register_forward_hook(lambda _m, _a, o, i=i: cap.__setitem__(("q", i), o)))
        hooks.append(at.k_norm.register_forward_hook(lambda _m, _a, o, i=i: cap.__setitem__(("k", i), o)))
        hooks.append(at.v_norm.register_forward_hook(lambda _m, a, _o, i=i: cap.__setitem__(("v", i), a[0])))
        hooks.append(at.o_proj.register_forward_pre_hook(lambda _m, a, i=i: cap.__setitem__(("z", i), a[0])))
        hooks.append(at.register_forward_hook(lambda _m, _a, o, i=i: cap.__setitem__(("attn", i), o[0])))
        hooks.append(block.post_attention_layernorm.register_forward_hook(lambda _m, _a, o, i=i: cap.__setitem__(("pa", i), o)))
    with torch.no_grad():
        model(input_ids=ids)
    for h in hooks:
        h.remove()
    return cap


@pytest.mark.parametrize("k_eq_v", [True, False])
def test_layout_matches_the_modules(k_eq_v):
    model = tiny_model(k_eq_v=k_eq_v)
    lay = arch.layout_of(model.config, True)
    assert [x.kind for x in lay] == TYPES
    for x, block in zip(lay, model.layers):
        at = block.self_attn
        assert at.q_proj.weight.shape[0] == x.q_heads * x.head_dim
        assert at.k_proj.weight.shape[0] == x.kv_heads * x.head_dim
        assert (at.v_proj is None) == x.k_eq_v
    assert {x.head_dim for x in lay} == {8, 16} and lay[1].kv_heads == (1 if k_eq_v else 2) and lay[0].kv_heads == 2


@pytest.mark.parametrize("k_eq_v", [True, False])
def test_values_writes_and_slices(k_eq_v):
    """Values are ``W_V γ_in x̂`` (``W_K`` on full layers with keys as values), before the weightless
    v_norm; each head's write ``γ_post ⊙ W_O^h z_h / ρ`` sums to the post-attention norm's output."""
    model = tiny_model(k_eq_v=k_eq_v)
    lay = arch.layout_of(model.config, True)
    ids = torch.randint(0, 50, (1, 12))
    cap = _captures(model, ids)
    for i, (x, block) in enumerate(zip(lay, model.layers)):
        at = block.self_attn
        Wv = (at.k_proj if x.k_eq_v else at.v_proj).weight
        xin = cap[("x", i)][0]  # the input norm's output, γ_in included
        v = (xin @ Wv.T).view(-1, x.kv_heads, x.head_dim)
        torch.testing.assert_close(v, cap[("v", i)][0], rtol=1e-5, atol=1e-5)
        pa = block.post_attention_layernorm
        wo = pa.weight[:, None] * at.o_proj.weight
        rho = torch.rsqrt(cap[("attn", i)][0].pow(2).mean(-1) + arch.norm_eps(pa))
        z = cap[("z", i)][0]
        total = sum((z[:, arch.head_slices(x, h)[1]] @ wo[:, arch.head_slices(x, h)[1]].T) * rho[:, None]
                    for h in range(x.q_heads))
        torch.testing.assert_close(total, cap[("pa", i)][0], rtol=1e-5, atol=1e-5)
        # each query head reads its KV head: the GQA grouping head_slices assumes
        assert [arch.kv_of(x, h) for h in range(x.q_heads)] == [h // (x.q_heads // x.kv_heads) for h in range(x.q_heads)]


def test_rope_angles_and_relative_rotation():
    """``cat([inv_freq] × 2)`` per layer type gives the model's rotary table (identity off the
    rotated quarter on full layers), and ``q_rot·k_rot = q̃ᵀ R̄ k̃`` with
    ``R̄ = diag(cos Δa) − diag(sin Δa) P``, ``Δ = t − s``, as head_qk builds it."""
    model = tiny_model()
    lay = arch.layout_of(model.config, True)
    T = 12
    ids = torch.randint(0, 50, (1, T))
    cap = _captures(model, ids)
    pos = torch.arange(T)[None]
    for i, x in enumerate(lay):
        inv = getattr(model.rotary_emb, f"{x.kind}_inv_freq").float()
        ang = torch.cat([inv] * 2)
        assert len(ang) == x.head_dim
        cos, sin = model.rotary_emb(torch.zeros(1, T, x.head_dim), pos, x.kind)
        torch.testing.assert_close(cos[0], (pos[0, :, None] * ang).cos(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(sin[0], (pos[0, :, None] * ang).sin(), rtol=1e-5, atol=1e-5)
        if x.kind == "full_attention":
            assert (ang == 0).sum() > 0  # partial: the rest is not rotated
        q, k = cap[("q", i)][0], cap[("k", i)][0]  # [T, heads, hd] after q_norm / k_norm
        rot = lambda y, t: y * cos[0, t, None] + _rotate_half(y) * sin[0, t, None]  # noqa: E731
        P = _rotate_half(torch.eye(x.head_dim)).T
        for t, s_ in ((7, 3), (11, 0), (5, 5)):
            d = t - s_
            Rbar = torch.diag((d * ang).cos()) - torch.diag((d * ang).sin()) @ P
            kv = arch.kv_of(x, 1)
            want = rot(q[t], t)[1] @ rot(k[s_], s_)[kv]
            torch.testing.assert_close(q[t, 1] @ Rbar @ k[s_, kv], want, rtol=1e-4, atol=1e-5)
