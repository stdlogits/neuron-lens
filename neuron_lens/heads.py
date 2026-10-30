"""Attention-head circuits as token maps (findings §11).

A head ``h`` of layer ``L`` reads the normed residual at layer ``L − 1``'s output and writes into
layer ``L``'s MLP input. Reading a token ``v`` there means its S-lens footprint
``X_v = K^out_{L−1}(e_v − ē)`` (the centred embedding at L0); a token ``u`` in the write means the
R-lens direction ``J^{mid⊤}_L γ_f u_u`` (``ov_R``) or the S-lens footprint ``K^mid_L(e_u − ē)``
(``ov_S``). A token map ``M = A Bᵀ`` over the vocabulary is never built: its rows and columns are
read from the 128-dimensional factors.
"""
import torch

from neuron_lens.transport import _rotate_half


def own_token_z(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """``[V, H]``: each token's z in its own column of each head's map ``M[u, v] = A_u · B_v``
    (``A``, ``B``: ``[V, H, k]``), ``(M[v, v] − mean_u M[u, v]) / std_u M[u, v]``, from the mean and
    covariance of ``A``'s rows."""
    abar = A.mean(0)
    Ac = A - abar
    cov = torch.einsum("vhi,vhj->hij", Ac, Ac) / len(A)
    del Ac
    var = torch.einsum("vhi,hij,vhj->vh", B, cov, B)
    return ((A * B).sum(-1) - (B * abar).sum(-1)) / var.clamp_min(1e-30).sqrt()


def copy_score(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """``[H]``: Elhage et al.'s copying score ``Σλ / Σ|λ|`` of each head's map ``A Bᵀ``, from the
    eigenvalues of ``Bᵀ A`` (``k × k``), which are its nonzero ones: +1 copies, −1 suppresses."""
    lam = torch.linalg.eigvals(torch.einsum("vhi,vhj->hij", B.float(), A.float()).cpu().double())
    return (lam.real.sum(-1) / lam.abs().sum(-1)).float()


def rope_mean(offsets: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """``[H, hd, hd]``: RoPE's relative rotation averaged over each head's attention by offset
    (``offsets`` ``[H, T]``, summing to 1 per head), so that a logit is ``q̃ᵀ R̄ k̃``."""
    ang = torch.cat([inv_freq, inv_freq]).to(offsets)
    rot = torch.arange(offsets.shape[1], device=offsets.device)[:, None] * ang  # [T, hd]
    c, s = offsets @ rot.cos(), offsets @ rot.sin()
    P = _rotate_half(torch.eye(len(ang), device=offsets.device)).T  # P x = rotate_half(x)
    return torch.diag_embed(c) - torch.diag_embed(s) @ P


def norm_derivative(weight: torch.Tensor, r: torch.Tensor, m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """``[heads, hd, hd]``: the true derivative of a q/k RMSNorm (and the input norm's scaling),
    averaged over positions: ``diag(w)(E[r/ρ] I − E[r³ x xᵀ/ρ] / hd)`` from the sums ``r``
    ``[heads]`` and ``m`` ``[heads, hd, hd]`` over ``n`` positions."""
    hd = m.shape[-1]
    eye = torch.eye(hd, device=m.device)
    return weight[:, None] * (r[:, None, None] * eye - m / hd) / n
