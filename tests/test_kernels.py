"""The Triton kernels (``neuron_lens.kernels``, CUDA only) against their torch references in
``neuron_lens.layers``, on a one-layer Qwen3.5 DeltaNet with head sizes the kernels' products take
(16 or more) and documents of several chunks, valid rows starting inside the first."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("the kernels run on CUDA", allow_module_level=True)
kernels = pytest.importorskip("neuron_lens.kernels")
q35 = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from neuron_lens import layers
from neuron_lens.transport import SKIP_FIRST


def delta_map(dk, dv, T):
    torch.manual_seed(0)
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=50, hidden_size=64, num_hidden_layers=1, layer_types=["linear_attention"],
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=dk, linear_value_head_dim=dv,
        linear_conv_kernel_dim=4, initializer_range=0.2, num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=16, shared_expert_intermediate_size=16,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True})
    model = q35.Qwen3_5MoeTextModel(cfg).float().eval().requires_grad_(False)
    with torch.no_grad():  # decays and write strengths away from their initialisations
        la = model.layers[0].linear_attn
        la.A_log.copy_(torch.empty_like(la.A_log).uniform_(0.02, 0.5).log())
        la.dt_bias.uniform_(-1, 1)
        maps, _ = layers._run(model.cuda(), torch.randint(0, 50, (T,), device="cuda"))
    return maps[0].mix


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("dk,dv,T", [(128, 128, SKIP_FIRST + 150), (32, 64, SKIP_FIRST + 64), (64, 128, 40)])
def test_delta_kernels_match_reference(dk, dv, T):
    m = delta_map(dk, dv, T)
    (N, hk, _), hv, b = m.q0.shape, m.q.shape[0], 3
    torch.manual_seed(1)
    dq, dkk = torch.randn(b, N, hk, dk, device="cuda"), torch.randn(b, N, hk, dk, device="cuda")
    dvv = torch.randn(b, N, hv, 2 * dv, device="cuda")[..., :dv]  # a strided view, as the projection's split
    db, dg = torch.randn(b, N, hv, device="cuda"), 0.1 * torch.randn(b, N, hv, device="cuda").cumsum(1)
    go = torch.randn(b, N, hv, dv, device="cuda")
    prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        want_do = layers._delta_core_jvp(m, dq, dkk, dvv, db, dg)
        want_g = layers._delta_core_vjp(m, go, hk)
    finally:
        torch.set_float32_matmul_precision(prec)
    do = kernels.delta_jvp(m, dq, dkk, dvv, db, dg)
    g = kernels.delta_vjp(m, go, hk)
    assert rel(do, want_do) < 1e-2
    for name, a, w in zip(("q", "k", "v", "beta", "gc"), g, want_g):
        assert rel(a, w) < 1e-2, name
    # the kernels transpose each other: <jvp(t), go> = <t, vjp(go)>
    lhs = (do * go).sum()
    rhs = sum((a * t).sum() for a, t in zip(g, (dq, dkk, dvv, db, dg)))
    assert abs((lhs - rhs) / lhs).item() < 1e-2
    # compiled, as the layer steps run them
    assert rel(torch.compile(kernels.delta_jvp)(m, dq, dkk, dvv, db, dg), do) < 1e-5
    assert rel(torch.compile(kernels.delta_vjp)(m, go, hk)[1], g[1]) < 1e-5


def test_moe_combine_matches_reference():
    torch.manual_seed(0)
    N, k, b, d = 37, 8, 5, 64
    inv = torch.randperm(N * k, device="cuda").view(N, k)
    w = torch.rand(N, k, device="cuda").softmax(-1)
    y = torch.randn(N * k, b, d, device="cuda").to(torch.bfloat16)
    m = layers.MoE(*[None] * len(layers.MoE._fields))._replace(inv=inv, slot_w=w)
    want = (y[inv] * w[..., None, None]).sum(1).transpose(0, 1)
    assert rel(kernels.moe_combine(inv, w, y), want) < 1e-6
    assert rel(layers._combine(m, y), want) < 1e-6


@pytest.mark.parametrize("jvp", [True, False])
def test_moe_up_matches_reference(jvp):
    """Uneven experts, one empty, rows that end mid-tile."""
    torch.manual_seed(0)
    N, k, b, d, I, E = 45, 4, 5, 64, 32, 8
    idx = torch.randint(0, E - 1, (N * k,), device="cuda")  # expert E − 1 gets no pairs
    order = torch.argsort(idx, stable=True)
    pos, offs = (order // k).contiguous(), torch.bincount(idx, minlength=E).cumsum(0).to(torch.int32)
    cg, cu = (torch.randn(N * k, I, device="cuda").to(torch.bfloat16) for _ in range(2))
    x = torch.randn(b, N, d, device="cuda").to(torch.bfloat16)
    w = torch.randn(E, 2 * I, d, device="cuda").to(torch.bfloat16) if jvp else \
        torch.randn(E, d, I, device="cuda").to(torch.bfloat16)
    rows = x.transpose(0, 1)[pos].reshape(-1, d)
    if jvp:
        gu = torch.nn.functional.grouped_mm(rows, w.transpose(1, 2), offs=offs * b).view(-1, b, 2 * I)
        want = (cg[:, None] * gu[..., :I] + cu[:, None] * gu[..., I:]).reshape(-1, I)
    else:
        gp = torch.nn.functional.grouped_mm(rows, w, offs=offs * b).view(-1, b, I)
        want = torch.cat((cg[:, None] * gp, cu[:, None] * gp), -1).reshape(-1, 2 * I)
    assert rel(kernels.moe_up(pos, offs, cg, cu, x, w, jvp).float(), want.float()) < 1e-2
