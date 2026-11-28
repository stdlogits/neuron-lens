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


@pytest.mark.parametrize("dk,dv,T", [(128, 128, SKIP_FIRST + 150), (32, 64, SKIP_FIRST + 64), (64, 128, 40),
                                     (64, 64, 4 * layers.CHUNK + 1)])  # (whole chunks, no padding)
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
        want_g = layers._delta_core_vjp(m, go)
    finally:
        torch.set_float32_matmul_precision(prec)
    do = kernels.delta_jvp(m, dq, dkk, dvv, db, dg)
    g = kernels.delta_vjp(m, go)
    assert rel(do, want_do) < 1e-2
    for name, a, w in zip(("q", "k", "v", "beta", "gc"), g, want_g):
        assert rel(a, w) < 1e-2, name
    # the kernels transpose each other: <jvp(t), go> = <t, vjp(go)>
    lhs = (do * go).sum()
    rep = lambda x: x.repeat_interleave(hv // hk, 2)  # key-head tangents meet each value head's cotangent
    rhs = sum((a * t).sum() for a, t in zip(g, (rep(dq), rep(dkk), dvv, db, dg)))
    assert abs((lhs - rhs) / lhs).item() < 1e-2
    # compiled, as the layer steps run them
    assert rel(torch.compile(kernels.delta_jvp)(m, dq, dkk, dvv, db, dg), do) < 1e-5
    assert rel(torch.compile(kernels.delta_vjp)(m, go)[1], g[1]) < 1e-5
    # the whole mixer, convolution and norms included, against the torch path
    xn = torch.randn(b, N, m.w_qkv.shape[1], device="cuda")
    G = torch.randn(b, N, m.w_out.shape[0], device="cuda")
    got = layers._delta_jvp(m, xn), layers._delta_vjp(m, G)
    layers.kernels = None
    try:
        torch.set_float32_matmul_precision("highest")
        want = layers._delta_jvp(m, xn), layers._delta_vjp(m, G)
    finally:
        layers.kernels = kernels
        torch.set_float32_matmul_precision(prec)
    assert rel(got[0], want[0]) < 1e-2 and rel(got[1], want[1]) < 1e-2


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


def test_build_rows_bf16_matches_reference():
    torch.manual_seed(0)
    N, F, D = 12, 192, 128
    wg = torch.randn(F, D, device="cuda", dtype=torch.bfloat16)
    wu = torch.randn(F, D, device="cuda", dtype=torch.bfloat16)
    cg = torch.randn(N, F, device="cuda", dtype=torch.bfloat16)
    cu = torch.randn(N, F, device="cuda", dtype=torch.bfloat16)
    n2 = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    got = kernels.build_rows_bf16(wg, wu, cg, cu, n2)
    want = (cg[:, :, None].float() * wg[None].float() + cu[:, :, None].float() * wu[None].float()) * n2[:, None, :].float()
    want = want.to(torch.bfloat16).permute(1, 0, 2).reshape(F, N * D)
    assert rel(got.float(), want.float()) < 1e-3
    # Ungated without norm scale
    got_ug = kernels.build_rows_bf16(wg, None, cg, None, None)
    want_ug = (cg[:, :, None].float() * wg[None].float()).to(torch.bfloat16).permute(1, 0, 2).reshape(F, N * D)
    assert rel(got_ug.float(), want_ug.float()) < 1e-3


def test_contract_fp8_sr_unbiased():
    if torch.cuda.get_device_capability()[0] < 9 and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 scaled_mm requires compute capability >= 8.9")
    torch.manual_seed(0)
    N, F, D = 8, 256, 128
    wg = torch.randn(F, D, device="cuda", dtype=torch.bfloat16)
    wu = torch.randn(F, D, device="cuda", dtype=torch.bfloat16)
    wd = torch.randn(D, F, device="cuda", dtype=torch.bfloat16)
    cg = torch.randn(N, F, device="cuda", dtype=torch.bfloat16)
    cu = torch.randn(N, F, device="cuda", dtype=torch.bfloat16)
    n2 = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    post2 = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    rows = (cg[:, :, None].float() * wg[None].float() + cu[:, :, None].float() * wu[None].float()) * n2[:, None, :].float()
    want = torch.einsum("df,nfd->nd", wd.float(), rows)  # check full matrix via matmul
    want = (wd.float() @ rows.transpose(1, 2).reshape(N * D, F).T).unflatten(1, (N, D)).transpose(0, 1) * post2[:, :, None].float()
    seed = torch.tensor(42, device="cuda", dtype=torch.int64)
    acc = torch.zeros_like(want)
    draws = 64
    for _ in range(draws):
        acc += kernels.contract_fp8_sr(wg, wu, wd, cg, cu, n2, post2, seed).float()
    # Unbiased stochastic rounding converges at O(1/sqrt(draws)): ~0.008 at 64 draws, well under 0.02
    assert rel(acc / draws, want) < 2e-2


def test_bf16_stats_and_hist_exact():
    torch.manual_seed(0)
    P, N, NB = 513, 257, 82
    x = torch.randn(P, N, device="cuda", dtype=torch.bfloat16) * 4.0
    bins = torch.randint(0, 2 * NB, (65536,), device="cuda", dtype=torch.int32)
    s_got = torch.zeros(N, device="cuda", dtype=torch.float32)
    sq_got = torch.zeros(N, device="cuda", dtype=torch.float32)
    cnt_got = torch.zeros(2, NB, N, device="cuda", dtype=torch.int32)
    kernels.bf16_stats_and_hist(x, bins, s_got, sq_got, cnt_got)
    xf = x.float()
    assert rel(s_got, xf.sum(0)) < 1e-5
    assert rel(sq_got, xf.pow(2).sum(0)) < 1e-5
    bits = x.view(torch.uint16).to(torch.int64)
    b = bins[bits].to(torch.int64)
    cnt_want = torch.zeros(2 * NB, N, device="cuda", dtype=torch.int32)
    cnt_want.scatter_add_(0, b, torch.ones_like(b, dtype=torch.int32))
    assert torch.equal(cnt_got.view(2 * NB, N), cnt_want)
