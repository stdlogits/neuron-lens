"""Attention-head circuits as token maps (findings §10).

A head ``h`` of layer ``L`` reads the normed residual at layer ``L − 1``'s output and writes into
layer ``L``'s MLP input (in a parallel block, GPT-NeoX's, into layer ``L``'s output: ``mid`` becomes ``out``
below, ``arch.head_write_lens``). Reading a token ``v`` there means its S-lens footprint
``X_v = K^out_{L−1}(e_v − ē)`` (the centred embedding at L0); a token ``u`` in the write means the
R-lens direction ``J^{mid⊤}_L γ_f u_u`` (``ov_R``) or the S-lens footprint ``K^mid_L(e_u − ē)``
(``ov_S``). A token map ``M = A Bᵀ`` over the vocabulary is never built: its rows and columns are
read from the 128-dimensional factors.
"""
import torch

from neuron_lens.layers import _rotate_half


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


def rope_mean(offsets: torch.Tensor, inv_freq: torch.Tensor, hd: int | None = None) -> torch.Tensor:
    """``[H, hd, hd]``: RoPE's relative rotation averaged over each head's attention by offset
    (``offsets`` ``[H, T]``, summing to 1 per head), so that a logit is ``q̃ᵀ R̄ k̃``. RoPE on the first
    ``2·len(inv_freq)`` dims of a head of ``hd`` (GPT-NeoX's quarter): the identity on the rest."""
    rd = 2 * len(inv_freq)
    hd = hd or rd
    ang = torch.cat([inv_freq, inv_freq, inv_freq.new_zeros(hd - rd)]).to(offsets)
    rot = torch.arange(offsets.shape[1], device=offsets.device)[:, None] * ang  # [T, hd]
    c, s = offsets @ rot.cos(), offsets @ rot.sin()
    P = torch.zeros(hd, hd, device=offsets.device)
    P[:rd, :rd] = _rotate_half(torch.eye(rd, device=offsets.device)).T  # P x = rotate_half(x) on the rotated dims
    return torch.diag_embed(c) - torch.diag_embed(s) @ P


def norm_derivative(weight: torch.Tensor, r: torch.Tensor, m: torch.Tensor, n: torch.Tensor,
                    width: int | None = None) -> torch.Tensor:
    """``[heads, hd, hd]``: the true derivative of a q/k RMSNorm (and the input norm's scaling),
    averaged over positions: ``diag(w)(E[r/ρ] I − E[r³ x xᵀ/ρ] / D)`` from the sums ``r``
    ``[heads]`` and ``m`` ``[heads, hd, hd]`` over ``n`` positions, ``w`` the scale (``[hd]``, or per head
    ``[heads, hd]``) and ``D`` the width the norm spans (``hd``; OLMo's whole projection, whose derivative this is
    each head's block of, the head read through its own query or key)."""
    hd = m.shape[-1]
    eye = torch.eye(hd, device=m.device)
    return weight[..., :, None] * (r[:, None, None] * eye - m / (width or hd)) / n


def weights(model_id: str, lay, st: dict, dev) -> dict:
    """A layer's head circuits' weights as the reads take them: ``arch.qk``'s (``Wq``, ``Wk``, ``bq``, the q/k norms) and
    ``arch.ov``'s (``Wv``, ``Wo``), with the factors ``06_activations.py``'s sums ``st`` give, averaged over positions: a
    Gated DeltaNet's SiLU ratio ``σ(c)`` per channel after its convolution (the identity rule's; on the queries at the
    queries, on the keys and values at the keys), and an output gate's per channel on ``W_O`` (gated attention's
    ``σ(gate)``, a DeltaNet's output norm's scale and ``silu(z)``), at the queries."""
    from neuron_lens import arch
    out = arch.qk(model_id, lay, dev)
    out["Wv"], out["Wo"] = arch.ov(model_id, lay, dev)
    n_q, n_k = float(st["n_q"]), float(st["n_k"])
    H, kvh, hd, kd = lay.q_heads, lay.kv_heads, lay.head_dim, lay.kd
    if lay.deltanet:
        sig = st["sig"][lay.index].float().to(dev)
        sq, sk, sv = sig[:kvh * kd] / n_q, sig[kvh * kd:2 * kvh * kd] / n_k, sig[2 * kvh * kd:2 * kvh * kd + H * hd] / n_k
        out["Wq"] = out["Wq"] * sq.view(kvh, kd).repeat_interleave(H // kvh, 0).flatten()[:, None]
        out["Wk"], out["Wv"] = out["Wk"] * sk[:, None], out["Wv"] * sv[:, None]
    if lay.gated:
        out["Wo"] = out["Wo"] * (st["og"][lay.index, :H, :hd].float().to(dev) / n_q).flatten()
    return out
