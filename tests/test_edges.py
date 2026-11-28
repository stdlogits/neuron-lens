"""Gauge invariance of head operators (``neuron_lens.heads.operators``) and Haar-rotation floor calibration
(``scripts/16_edges.py``'s ``cos_floor``)."""
import importlib.util
from pathlib import Path

import torch

from neuron_lens import heads

_SPEC = importlib.util.spec_from_file_location("edges16", Path(__file__).resolve().parents[1] / "scripts/16_edges.py")
edges16 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(edges16)


def _rand_orth(n: int, gen: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen))
    return q


def test_sym_sqrt_and_operator_gauge_invariance():
    """Rotating a head's internal value or QK basis leaves its metric-normalized Frobenius norms,
    effective ranks (p2, q2), and composition cosines invariant."""
    g = torch.Generator().manual_seed(0)
    d, hd, kd = 64, 16, 16
    s_in = torch.empty(d).uniform_(0.5, 2.0)
    s_w = torch.empty(d).uniform_(0.5, 2.0)
    O = torch.randn(d, hd, generator=g)
    V = torch.randn(hd, d, generator=g)
    Q = torch.randn(kd, d, generator=g)
    K = torch.randn(kd, d, generator=g)

    def factors(O_, V_, Q_, K_):
        Vs = V_ * s_in[None, :]
        src = O_ @ heads._sym_sqrt(Vs @ Vs.T)
        Os = O_ / s_w[:, None]
        R_v = heads._sym_sqrt(Os.T @ Os) @ V_
        Ks = K_ * s_in[None, :]
        Qs = Q_ * s_in[None, :]
        R_q = heads._sym_sqrt(Ks @ Ks.T) @ Q_
        R_k = heads._sym_sqrt(Qs @ Qs.T) @ K_
        return src, R_v, R_q, R_k

    src1, rv1, rq1, rk1 = factors(O, V, Q, K)
    # Arbitrary invertible GL(hd) gauge transform M on (O, V) -> (O M^-1, M V) leaves O V invariant,
    # and GL(kd) gauge transform N on (Q, K) -> (N^-T Q, N K) leaves Q^T K invariant!
    M = torch.randn(hd, hd, generator=g) + 2.0 * torch.eye(hd)
    N = torch.randn(kd, kd, generator=g) + 2.0 * torch.eye(kd)
    src2, rv2, rq2, rk2 = factors(O @ torch.linalg.inv(M), M @ V, torch.linalg.inv(N).T @ Q, N @ K)

    # Probe with random upstream write A [d, 8] and downstream read B [8, d]
    A = torch.randn(d, 8, generator=g)
    B = torch.randn(8, d, generator=g)
    for x1, x2 in ((src1, src2),):
        assert torch.allclose((B @ x1).norm(), (B @ x2).norm(), rtol=1e-4, atol=1e-4)
        U1, U2 = x1 / s_w[:, None], x2 / s_w[:, None]
        assert torch.allclose(U1.norm(), U2.norm(), rtol=1e-4, atol=1e-4)
        assert torch.allclose((U1.T @ U1).pow(2).sum(), (U2.T @ U2).pow(2).sum(), rtol=1e-4, atol=1e-4)
    for r1, r2 in ((rv1, rv2), (rq1, rq2), (rk1, rk2)):
        assert torch.allclose((r1 @ A).norm(), (r2 @ A).norm(), rtol=1e-4, atol=1e-4)
        Rs1, Rs2 = r1 * s_in[None, :], r2 * s_in[None, :]
        assert torch.allclose(Rs1.norm(), Rs2.norm(), rtol=1e-4, atol=1e-4)
        assert torch.allclose((Rs1 @ Rs1.T).pow(2).sum(), (Rs2 @ Rs2.T).pow(2).sum(), rtol=1e-4, atol=1e-4)


def test_cos_floor_exact_rank1_and_haar_calibration():
    """At rank 1 (p2 = q2 = 1), cos_floor recovers sigmas / sqrt(d) exactly; at higher rank,
    it matches the exact Haar-rotation tail probability."""
    d = 4096
    fl1 = edges16.cos_floor(1.0, 1.0, d, sigmas=5.0)
    # Exact finite-d spherical variance gives nu = (d+2)/(d-1), matching sigmas/sqrt(d) up to O(1/d)
    assert abs(fl1.item() - 5.0 / d ** 0.5) < 5e-5
    # High rank: p2 = 1/32, q2 = 1/32 -> floor shrinks sharply toward 1/sqrt(d) = 0.015625
    fl_hi = edges16.cos_floor(1.0 / 32, 1.0 / 32, d, sigmas=5.0)
    assert 1.0 / d ** 0.5 < fl_hi.item() < fl1.item()
    # Monte Carlo check at moderate dimension d=128, rank r=8, sigmas=2.5 (alpha ~= 0.0124)
    g = torch.Generator().manual_seed(123)
    d_mc, r = 128, 8
    fl_mc = edges16.cos_floor(1.0 / r, 1.0 / r, d_mc, sigmas=2.5).item()
    A = _rand_orth(d_mc, g)[:, :r] / r ** 0.5
    exceed = 0
    trials = 400
    for _ in range(trials):
        B = _rand_orth(d_mc, g)[:r, :] / r ** 0.5
        if (B @ A).norm().item() >= fl_mc:
            exceed += 1
    # Expected ~0.0124 * 400 ~= 5 exceedances; allow generous Monte Carlo margin [0, 15]
    assert exceed <= 15
