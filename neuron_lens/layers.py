"""Decoder layers as linear maps for the lens estimators (``neuron_lens.transport``), one
implementation per component, composed per layer from what the model's layer holds: a new
architecture built from these parts needs no new map. Qwen3, Gemma 4 and Qwen3.5 (dense and MoE)
are covered; ``check`` refuses what is not.

A layer is read as

    a = mix(n1(h));  h_mid = h + post1(a)                                       (mid: h_mid)
    z = mlp(n2(h_mid));  h_out = (h_mid + post2(z)) · s

with ``post1``, ``post2`` (norms on the sublayers' outputs) and the layer scalar ``s`` optional
(Gemma 4 has them). On a document every part becomes a fixed linear map under RelP's rules, applied
by operation wherever the operation sits (``neuron_lens.rules`` states them for autograd, the
tests' reference):

- **norms** (RMSNorm; scale ``w``, ``1 + w`` for zero-centred norms, or none) take the LN rule,
  their rsqrt held; ``rules="true"`` keeps their true derivative, and the dense MLP's (media fits);
- **activations** ``x · g(x)`` (SiLU, GELU-tanh) take the identity rule, ``g(x) dx``;
- **elementwise gates**, a vector of gates on a vector of features, take the half rule: SwiGLU's,
  DeltaNet's ``norm(o) · silu(z)``, gated attention's ``o · σ(gate)`` (σ's true derivative);
- **scalar weights** that decide how much of a value or a module passes keep their true
  derivative: softmax attention's, DeltaNet's ``A`` (through q, k, β and the decay), a router's
  weights over its chosen experts, a shared expert's sigmoid gate (a router for one expert); so do
  q/k norms, a value norm and DeltaNet's L2 norms.

Mixers:

- **Softmax attention:** grouped query heads, q/k norms, a weightless value norm (Gemma 4), keys as
  values (Gemma 4's full layers), RoPE on all or the first part of each head (the model's own
  tables), an output gate (Qwen3.5: ``q_proj`` gives per head the query, then the gate). Masks are
  causal, within a sliding window where the layer has one (key ``s`` seen from ``t`` if
  ``s > t − window``, as HF's), and both ways inside a ``block`` of positions in sliding layers
  (an image, as Gemma 4's vision mask). Masks enter the forward pass only: a masked key's weight is
  0, and so is its derivative.
- **Gated DeltaNet** (Qwen3.5's ``linear_attention``): ``qkv = silu(conv(W_qkv x))`` (a causal
  depthwise convolution of width 4), q and k L2-normed, q scaled by ``d_k^-½``; per value head (two
  per key head) a delta-rule memory ``S_t = e^{g_t} S_{t−1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ``,
  read as ``o_t = S_t q_t``, with write strength ``β = σ(W_b x)`` and log decay
  ``g = −e^{A_log} softplus(W_a x + dt_bias)``; then ``norm(o) · silu(W_z x)`` per head, and
  ``W_out``. Unrolled, ``o = A v`` with ``A = W L⁻¹ diag(β)``, ``W = (Q Kᵀ) ⊙ Γ`` (lower
  triangle with its diagonal), ``L = I + diag(β) ((K Kᵀ) ⊙ Γ)`` (strictly lower) and
  ``Γ_ts = e^{gc_t − gc_s}`` for the cumulative log decay ``gc``: attention with a structured
  matrix that does not depend on v. The pseudo-values ``U = L⁻¹ diag(β) V`` are the memory's
  columns: ``S_t = Σ_{s≤t} Γ_ts u_s k_sᵀ``. The derivative runs in chunks of ``CHUNK``
  positions, as flash-linear-attention's kernels run the delta rule: within a chunk as small
  ``Γ``-weighted matrices, across chunks through one ``[d_k, d_v]`` state per tangent and head,
  so no ``[b, Hv, N, T]`` matrix is formed. The decay's tangent ``dgc`` moves the queries and
  keys (``Γ_ts`` moves by ``Γ_ts (dgc_t − dgc_s)``), and ``L⁻¹`` is a forward substitution
  over chunks with each chunk's ``(I + diag(β) C)⁻¹`` (``tinv``) held, so the map is one pass
  forward over the chunks and its transpose one pass back (``_delta_core_jvp``,
  ``_delta_core_vjp``; Triton kernels on CUDA, ``neuron_lens.kernels``).

MLPs: **gated** (SwiGLU or GELU-tanh), or **sparse**: routed experts weighted by the router's softmax
renormalised over its top k (Qwen3.5's 256 experts, top 8), plus an optional shared expert behind an
optional sigmoid gate.

Rows are the ``N`` valid positions, as in ``transport``; keys (and DeltaNet's memory) run over all
``T`` positions, padded with masked ones to a multiple of 8 (``Tp``) so that rows of the attention
products stay 16-byte aligned, which cuBLAS's fast kernels need. Tangents enter at valid positions
only, so the convolution and the delta rule's tangents start from zero at the first valid position.
"""
import inspect
from typing import NamedTuple

import torch
import torch.nn.functional as F

from neuron_lens.rules import HALF_RULE_BETA, act_factor, check_layer, is_gated, norm_eps, norm_weight

try:
    from neuron_lens import kernels
except ImportError:  # no Triton (macOS): the cores run in torch
    kernels = None

CHUNK = 32  # DeltaNet's chunk length (positions) for the derivative's chunked passes


class Norm(NamedTuple):
    """An RMSNorm's map at the valid rows: ``a`` ``[N, d]``, its scale times its rsqrt (the LN
    rule); for the true derivative (``rules="true"``) also its input ``x`` and rsqrt ``r``."""
    a: torch.Tensor
    x: torch.Tensor | None
    r: torch.Tensor | None


class Attention(NamedTuple):
    """A softmax-attention mixer's map. Optional parts are ``None`` where the layer lacks them."""
    wq: torch.Tensor  # [Hq·hd, d]; [Hq·2hd, d] with an output gate (per head the query rows, then the gate's)
    wk: torch.Tensor
    wv: torch.Tensor | None  # None: the value is the raw key
    wo: torch.Tensor
    q: torch.Tensor | None  # [N, Hq, hd], [N, Hq, 1], [hd]: queries before q_norm, its rsqrt and scale
    rq: torch.Tensor | None
    wqn: torch.Tensor | None
    k: torch.Tensor | None  # the same for keys
    rk: torch.Tensor | None
    wkn: torch.Tensor | None
    v: torch.Tensor | None  # [N, Hk, hd], [N, Hk, 1]: values before their weightless norm, its rsqrt
    rv: torch.Tensor | None
    cos: torch.Tensor  # [N, 1, rd]: RoPE on the first rd dims of each head
    sin: torch.Tensor
    qr: torch.Tensor  # [Hk, g·N, hd]: normed, rotated queries, grouped by key head
    kr: torch.Tensor  # [Hk, Tp, hd]: normed, rotated keys
    vn: torch.Tensor  # [Hk, Tp, hd]: the values mixed
    p: torch.Tensor  # [Hk, g·N, Tp]: attention probabilities
    o: torch.Tensor | None  # [N, Hq·hd]: the output before its gate, σ(gate), σ'(gate)
    sg: torch.Tensor | None
    dsg: torch.Tensor | None
    scale: float
    s0: int


class DeltaNet(NamedTuple):
    """A Gated DeltaNet mixer's map. Head tensors are ``[Hv, ·, ·]`` (key heads repeated to value
    heads); ``1`` marks valid rows, no suffix all ``T`` positions, ``Tp`` all positions padded to
    whole chunks of ``CHUNK`` (``n`` chunks)."""
    w_qkv: torch.Tensor  # [C, d]: q, k, v channels
    w_z: torch.Tensor  # [Hv·dv, d]
    w_b: torch.Tensor  # [Hv, d]
    w_a: torch.Tensor  # [Hv, d]
    w_out: torch.Tensor  # [d, Hv·dv]
    wc: torch.Tensor  # [C, K]: the depthwise convolution
    sc: torch.Tensor  # [N, C]: σ of the convolution's output (identity rule for its SiLU)
    q0: torch.Tensor  # [N, Hk, dk], [N, Hk, 1]: queries before their L2 norm, and its rsqrt
    rq0: torch.Tensor
    k0: torch.Tensor
    rk0: torch.Tensor
    q: torch.Tensor  # [Hv, Tp, dk]: normed, scaled queries
    k: torch.Tensor  # [Hv, Tp, dk]: normed keys
    u: torch.Tensor  # [Hv, Tp, dv]: pseudo-values
    vmc: torch.Tensor  # [Hv, Tp, dv]: V − C U, C = (K Kᵀ) ⊙ Γ (strictly lower): β's move
    beta: torch.Tensor  # [Hv, Tp]
    gc: torch.Tensor  # [Hv, Tp]: cumulative log decay (constant over the padding)
    hku: torch.Tensor  # [Hv, n, dk, dv]: the memory at each chunk's start, Σ Γ k_s u_sᵀ
    tinv: torch.Tensor  # [Hv, n, C, C]: each chunk's (I + diag(β) C)⁻¹
    qkd: torch.Tensor  # [Hv, n, C, C]: (Q Kᵀ) ⊙ Γ within each chunk; (K Kᵀ) ⊙ Γ, strictly lower
    kkd: torch.Tensor
    qh: torch.Tensor  # [Hv, Tp, dv]: queries, keys through the memory at their chunk's start
    kh: torch.Tensor
    db: torch.Tensor  # [N, Hv]: σ'(b), dβ/db
    dga: torch.Tensor  # [N, Hv]: dg/da
    on: torch.Tensor  # [N, Hv, dv]: the output norm's output (weight · o · rsqrt)
    ao: torch.Tensor  # [N, Hv, 1]: its rsqrt; with ``wn`` the LN rule's scaling
    wn: torch.Tensor  # [dv]
    sz: torch.Tensor  # [N, Hv, dv]: σ(z)
    z: torch.Tensor  # [N, Hv, dv]: silu(z)
    s0: int


class MoE(NamedTuple):
    """A sparse MLP's map: experts (``[E, 2I, d]`` gate and up rows, ``[E, d, I]`` down) at the
    ``M = N·k`` (position, slot) pairs, sorted by expert; an optional shared expert behind an
    optional sigmoid gate."""
    w_gu: torch.Tensor
    w_dn: torch.Tensor
    w_r: torch.Tensor  # [E, d]: the router
    pos: torch.Tensor  # [M]: each pair's row, in expert order
    inv: torch.Tensor  # [N, k]: each (row, slot)'s pair in that order
    offs: torch.Tensor  # [E]: int32, where each expert's pairs end
    slot_w: torch.Tensor  # [N, k], [N, k]: routing weights, and the experts
    idx: torch.Tensor
    cg: torch.Tensor  # [M, I], [M, I]: SwiGLU's gate and up factors (identity, half rules)
    cu: torch.Tensor
    eo: torch.Tensor  # [N, k, d]: each chosen expert's output (the router's term)
    sh: tuple | None  # the shared expert's (w_gate, w_up, w_down, cg [N, Is], cu [N, Is])
    ws: torch.Tensor | None  # [1, d]: the shared expert's gate
    sg: torch.Tensor | None  # [N, 1], [N, 1]: σ(s), σ'(s)
    dsg: torch.Tensor | None
    so: torch.Tensor | None  # [N, d]: the shared expert's output before its gate


class GatedMLP(NamedTuple):
    wg: torch.Tensor
    wu: torch.Tensor
    wd: torch.Tensor
    cg: torch.Tensor  # [N, d_ff]: the gate's factor (identity, half rules; or act'(g)·up)
    cu: torch.Tensor  # [N, d_ff]: the up projection's factor (β·act(g); or act(g))


class Layer(NamedTuple):
    n1: Norm
    mix: Attention | DeltaNet
    post1: Norm | None
    n2: Norm
    mlp: GatedMLP | MoE
    post2: Norm | None
    s: torch.Tensor | None  # []: the layer scalar


# ------------------------------------------------------------------------------------ primal

def _rms(norm, x: torch.Tensor):
    """An RMSNorm as HF runs it (fp32, scaled by ``norm_weight``, then the input's dtype), and its
    rsqrt."""
    r = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm_eps(norm))
    w = norm_weight(norm)
    y = x.float() * r
    return (y if w is None else y * w.float()).to(x.dtype), r


def _norm_map(norm, x, r, S: int, exact: bool) -> Norm:
    w = norm_weight(norm)
    a = r[S:] if w is None else w.float() * r[S:]
    return Norm(a, x[S:].float() if exact else None, r[S:] if exact else None)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


def _rope(x, cos, sin):
    """RoPE on the first ``cos.shape[-1]`` dims of the last axis (all of them, or Qwen3.5's
    quarter); ``_rope(x, cos, -sin)`` is its transpose."""
    rd = cos.shape[-1]
    if rd == x.shape[-1]:
        return x * cos + _rotate_half(x) * sin
    return torch.cat((x[..., :rd] * cos + _rotate_half(x[..., :rd]) * sin, x[..., rd:]), -1)


def _norm_jvp(dx, x, r, w):
    """The true derivative of the RMSNorm ``w·x·r`` over the last dim, applied to ``dx``."""
    return w * r * (dx - x * r.square() * (x * dx).mean(-1, keepdim=True))


def _norm_vjp(gy, x, r, w):
    a = w * gy
    return r * (a - x * r.square() * (x * a).mean(-1, keepdim=True))


def _attention(at, x, cos, sin, mask, S: int, exact: bool):
    """A softmax-attention mixer over a document; its output ``[T, d]`` and map."""
    T, dt = x.shape[0], x.dtype
    tp = mask.shape[1]
    hd, g = at.head_dim, at.num_key_value_groups
    hk = at.k_proj.weight.shape[0] // hd
    gated = is_gated(at)
    if gated and exact:
        raise ValueError("true derivatives are not written out for gated attention")
    qp = F.linear(x, at.q_proj.weight)
    if gated:
        qp = qp.view(T, hk * g, 2 * hd)
        q, gate = qp[..., :hd], qp[..., hd:].reshape(T, -1)
    else:
        q = qp.view(T, hk * g, hd)
    k = F.linear(x, at.k_proj.weight).view(T, hk, hd)
    wv = getattr(at, "v_proj", None)
    v = k if wv is None else F.linear(x, wv.weight).view(T, hk, hd)
    qn_, kn_, vn_ = (getattr(at, n, None) for n in ("q_norm", "k_norm", "v_norm"))
    qn, rq = (q, None) if qn_ is None else _rms(qn_, q)
    kn, rk = (k, None) if kn_ is None else _rms(kn_, k)
    vn, rv = (v, None) if vn_ is None else _rms(vn_, v)
    qr = _rope(qn, cos, sin).transpose(0, 1).reshape(hk, g, T, hd)
    kr = F.pad(_rope(kn, cos, sin).transpose(0, 1), (0, 0, 0, tp - T))  # [Hk, Tp, hd]
    vp = F.pad(vn.transpose(0, 1), (0, 0, 0, tp - T))
    p = torch.softmax((qr @ kr[:, None].mT).float() * at.scaling + mask, -1).to(dt)
    o = (p @ vp[:, None]).reshape(hk * g, T, hd).transpose(0, 1).reshape(T, -1)
    sg = torch.sigmoid(gate.float()) if gated else None
    out = F.linear(o if sg is None else (o.float() * sg).to(dt), at.o_proj.weight)
    part = lambda t: None if t is None else t[S:]
    m = Attention(wq=at.q_proj.weight, wk=at.k_proj.weight, wv=None if wv is None else wv.weight,
                  wo=at.o_proj.weight,
                  q=None if rq is None else q[S:].float(), rq=part(rq),
                  wqn=None if qn_ is None else norm_weight(qn_).float(),
                  k=None if rk is None else k[S:].float(), rk=part(rk),
                  wkn=None if kn_ is None else norm_weight(kn_).float(),
                  v=None if rv is None else v[S:].float(), rv=part(rv),
                  cos=cos[S:].float(), sin=sin[S:].float(),
                  qr=qr[:, :, S:].reshape(hk, -1, hd), kr=kr, vn=vp, p=p[:, :, S:].reshape(hk, -1, tp),
                  o=None if sg is None else o[S:].float(), sg=part(sg),
                  dsg=None if sg is None else sg[S:] * (1 - sg[S:]), scale=float(at.scaling), s0=S)
    return out, m


def _l2(x):
    r = torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + 1e-6)
    return x * r, r


def _decay_mask(gc1, gc, S):
    """``Γ`` at valid rows, ``[Hv, N, T]``: ``e^{gc_t − gc_s}`` for ``s ≤ t``, else 0."""
    N, T = gc1.shape[-1], gc.shape[-1]
    t = torch.arange(S, S + N, device=gc.device)[:, None]
    s = torch.arange(T, device=gc.device)[None]
    diff = (gc1[..., :, None] - gc[..., None, :]).masked_fill(s > t, float("-inf"))
    return diff.exp(), (s < t)


def _delta_primal(la, x, S):
    """The DeltaNet mixer over a document; its output ``[T, d]`` and map."""
    T, dt = x.shape[0], x.dtype
    hk, hv, dk, dv = la.num_k_heads, la.num_v_heads, la.head_k_dim, la.head_v_dim
    wc = la.conv1d.weight.squeeze(1)  # [C, K]
    mixed = F.linear(x, la.in_proj_qkv.weight)  # [T, C]
    c = _conv(mixed[None].float(), wc.float())[0]
    qkv = F.silu(c)  # [T, C] fp32
    q0, k0, v = qkv.split([la.key_dim, la.key_dim, la.value_dim], -1)
    q0, k0, v = q0.view(T, hk, dk), k0.view(T, hk, dk), v.view(T, hv, dv)
    z = F.linear(x, la.in_proj_z.weight).view(T, hv, dv).float()
    b = F.linear(x, la.in_proj_b.weight).float()
    a = F.linear(x, la.in_proj_a.weight).float()
    beta = torch.sigmoid(b)
    g = -la.A_log.float().exp() * F.softplus(a + la.dt_bias.float())
    (qn, rq0), (kn, rk0) = _l2(q0), _l2(k0)
    rep = hv // hk
    qh = (qn * dk ** -0.5).repeat_interleave(rep, 1).transpose(0, 1)  # [Hv, T, dk]
    kh = kn.repeat_interleave(rep, 1).transpose(0, 1)
    vh, beta, g = v.transpose(0, 1), beta.T, g.T  # [Hv, T, dv], [Hv, T]
    gc = g.cumsum(-1)
    gam, strict = _decay_mask(gc, gc, 0)  # [Hv, T, T]
    W = (qh @ kh.mT) * gam
    Cm = (kh @ kh.mT) * gam * strict
    L = torch.eye(T, device=x.device) + beta[..., None] * Cm
    u = torch.linalg.solve_triangular(L, beta[..., None] * vh, upper=False, unitriangular=True)
    o = W @ u  # [Hv, T, dv]
    on, ao = _gated_norm(la.norm, o.transpose(0, 1))
    y = (on * F.silu(z)).reshape(T, -1).to(dt)
    out = F.linear(y, la.out_proj.weight)
    tp = -(-T // CHUNK) * CHUNK
    pad = lambda t: F.pad(t, (0, 0, 0, tp - T)).contiguous()  # (rows first, as the kernels read them)
    gp = torch.cat((gc, gc[:, -1:].expand(hv, tp - T)), -1)  # no decay in the padding
    bp = F.pad(beta, (0, tp - T)).contiguous()
    m = DeltaNet(w_qkv=la.in_proj_qkv.weight, w_z=la.in_proj_z.weight, w_b=la.in_proj_b.weight,
              w_a=la.in_proj_a.weight, w_out=la.out_proj.weight, wc=wc.float(),
              sc=torch.sigmoid(c[S:]),
              q0=q0[S:], rq0=rq0[S:], k0=k0[S:], rk0=rk0[S:],
              q=pad(qh), k=pad(kh), u=pad(u), vmc=pad(vh - Cm @ u), beta=bp, gc=gp,
              **_chunks(gp, pad(qh), pad(kh), pad(u), bp),
              db=(beta * (1 - beta))[:, S:].T, dga=-la.A_log.float().exp() * torch.sigmoid(a + la.dt_bias.float())[S:],
              on=on[S:], ao=ao[S:], wn=la.norm.weight.float(),
              sz=torch.sigmoid(z[S:]), z=F.silu(z[S:]), s0=S)
    return out, m


def _decays(gc):
    """Per chunk, from the cumulative log decay ``[Hv, Tp]``: ``Γ`` within it (``[Hv, n, C, C]``,
    ``s ≤ t``), ``e^{gc_t − gc at the previous chunk's end}`` and ``e^{gc at its end − gc_s}``
    (``[Hv, n, C]``), and ``e^{gc at its end − gc at the previous one's}`` (``[Hv, n]``). Each
    is ``e^{x}`` for some ``x ≤ 0``, as the decay only shrinks."""
    hv = gc.shape[0]
    g = gc.view(hv, -1, CHUNK)
    end = g[..., -1]
    prev = F.pad(end[:, :-1], (1, 0))  # 0 before the first chunk
    lower = torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=gc.device).tril()
    dec = (g[..., :, None] - g[..., None, :]).masked_fill(~lower, float("-inf")).exp()
    return dec, (g - prev[..., None]).exp(), (end[..., None] - g).exp(), (end - prev).exp()


def _chunks(gc, q, k, u, beta) -> dict:
    """``DeltaNet``'s per-document chunk constants (``hku`` to ``kh``) from the padded decay,
    queries, keys, pseudo-values and β."""
    hv, tp = gc.shape
    n = tp // CHUNK
    dec, qd, kd, cd = _decays(gc)
    qc, kc, uc = (x.unflatten(1, (n, CHUNK)) for x in (q, k, u))
    D = (kc * kd[..., None]).mT @ uc  # [Hv, n, dk, dv]: each chunk's own writes, at its end
    hku = [torch.zeros_like(D[:, 0])]
    for c in range(n - 1):
        hku.append(cd[:, c, None, None] * hku[-1] + D[:, c])
    hku = torch.stack(hku, 1)
    kkd = ((kc @ kc.mT) * dec).tril(-1)
    eye = torch.eye(CHUNK, device=gc.device)
    tinv = torch.linalg.solve_triangular(eye + beta.view(hv, n, CHUNK, 1) * kkd, eye.expand_as(kkd),
                                         upper=False, unitriangular=True)
    return dict(hku=hku, tinv=tinv.contiguous(), qkd=(qc @ kc.mT) * dec, kkd=kkd,  # (the solve may return columns first)
                qh=((qc * qd[..., None]) @ hku).flatten(1, 2), kh=((kc * qd[..., None]) @ hku).flatten(1, 2))


def _gated_norm(norm, o):
    """``RMSNormGated``'s norm (before its gate), per head: ``(w · o · r, r)``."""
    r = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    return norm.weight.float() * (o * r), r


def _gmm(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Rows of ``x`` (``[S, in]``, grouped by expert) times their expert's ``w`` (``[E, out, in]``,
    as HF stores expert weights); ``offs`` (``[E]``, int32) ends each expert's rows."""
    return F.grouped_mm(x, w.transpose(1, 2), offs=offs)


def _moe(mlp, x, S: int):
    """The sparse MLP over a document; its output ``[T, d]`` and map."""
    T, d, dt = *x.shape, x.dtype
    ex, router = mlp.experts, mlp.gate
    _, w_slot, idx = router(x)  # [T, k]: renormalised weights, experts
    I, k, E = ex.intermediate_dim, idx.shape[-1], ex.num_experts
    flat = idx.flatten()
    order = torch.argsort(flat, stable=True)  # pairs (position·k + slot) in expert order
    pos_all = order // k
    offs = torch.bincount(flat, minlength=E).cumsum(0).to(torch.int32)
    gu = _gmm(x[pos_all], ex.gate_up_proj, offs)
    gate, up = gu[:, :I], gu[:, I:]
    eo = _gmm(ex.act_fn(gate) * up, ex.down_proj, offs)
    w_all = w_slot.flatten()[order]
    out = torch.zeros(T, d, device=x.device, dtype=torch.float32)
    out.index_add_(0, pos_all, eo.float() * w_all[:, None].float())
    sh, ws, sg1, dsg, so1 = None, None, None, None, None
    shared = getattr(mlp, "shared_expert", None)
    if shared is not None:
        so, scg, scu = _swiglu(shared, x, S, False)
        gate_s = getattr(mlp, "shared_expert_gate", None)
        sg = torch.ones(T, 1, device=x.device) if gate_s is None else \
            torch.sigmoid(F.linear(x, gate_s.weight).float())  # [T, 1]
        out = out + sg * so.float()
        sh = (shared.gate_proj.weight, shared.up_proj.weight, shared.down_proj.weight, scg, scu)
        ws, sg1, so1 = None if gate_s is None else gate_s.weight, sg[S:], so[S:].float()
        dsg = sg1 * (1 - sg1)
    # The valid rows' pairs, still in expert order.
    keep = pos_all >= S
    g, u = gate[keep].float(), up[keep].float()
    eo_slot = torch.empty(T * k, d, device=x.device, dtype=torch.float32)
    eo_slot[order] = eo.float()
    inv = torch.empty(len(order) - S * k, dtype=torch.long, device=x.device)
    inv[order[keep] - S * k] = torch.arange(len(inv), device=x.device)
    m = MoE(w_gu=ex.gate_up_proj, w_dn=ex.down_proj, w_r=router.weight,
            pos=pos_all[keep] - S, inv=inv.view(-1, k),
            offs=torch.bincount(flat[order[keep]], minlength=E).cumsum(0).to(torch.int32),
            slot_w=w_slot[S:].float(), idx=idx[S:],
            cg=(HALF_RULE_BETA * act_factor(ex.act_fn, g) * u).to(dt), cu=(HALF_RULE_BETA * ex.act_fn(g)).to(dt),
            eo=eo_slot.view(T, k, d)[S:], sh=sh, ws=ws, sg=sg1, dsg=dsg, so=so1)
    return out.to(dt), m


def _swiglu(mlp, x, S: int, exact: bool):
    """A gated MLP's output ``[T, d]`` and its factors at rows ``S:``: the identity and half rules'
    (``β·φ(g)·up``, ``β·act(g)``), or the true derivative's (``act'(g)·up``, ``act(g)``)."""
    dt = x.dtype
    gate, up = F.linear(x, mlp.gate_proj.weight), F.linear(x, mlp.up_proj.weight)
    out = F.linear(mlp.act_fn(gate) * up, mlp.down_proj.weight)
    g, u = gate[S:].float(), up[S:].float()
    if exact:
        g = g.requires_grad_(True)
        with torch.enable_grad():
            dact = torch.autograd.grad(mlp.act_fn(g).sum(), g)[0]
        cg, cu = dact * u, mlp.act_fn(g.detach())
    else:
        cg, cu = HALF_RULE_BETA * act_factor(mlp.act_fn, g) * u, HALF_RULE_BETA * mlp.act_fn(g)
    return out, cg.detach().to(dt), cu.detach().to(dt)


_checked = set()


def check(model) -> None:
    """Refuse a decoder stack with parts ``linearize`` does not write out."""
    if id(model) in _checked:
        return
    c = model.config
    for name, ok in (("num_kv_shared_layers", 0), ("hidden_size_per_layer_input", 0),
                     ("enable_moe_block", False)):
        if getattr(c, name, ok) not in (ok, None):
            raise ValueError(f"{name}={getattr(c, name)} is not supported")
    for layer in model.layers:
        check_layer(layer)
        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            if not hasattr(la, "in_proj_qkv"):
                raise ValueError("fused DeltaNet projections (Qwen3-Next's) are not supported")
            if la.conv1d.bias is not None:
                raise ValueError("convolution biases are not supported")
        else:
            at = layer.self_attn
            projections = (at.q_proj, at.k_proj, getattr(at, "v_proj", None), at.o_proj)
            if any(p is not None and p.bias is not None for p in projections):
                raise ValueError("projection biases are not supported")
            if not getattr(at, "is_causal", True):
                raise ValueError("bidirectional text attention is not supported")
        mlp = layer.mlp
        if hasattr(mlp, "experts"):
            if not getattr(mlp.gate, "norm_topk_prob", True):
                raise ValueError("routers that do not renormalise over their top k are not supported")
        elif any(p.bias is not None for p in (mlp.gate_proj, mlp.up_proj, mlp.down_proj)):
            raise ValueError("projection biases are not supported")
    _checked.add(id(model))


def _norms(layer) -> tuple:
    """``(n1, post1, n2, post2)``: Gemma 4's sandwich (input, post-attention, pre- and
    post-feedforward norms), or a pre-norm layer's input and post-attention norms."""
    if hasattr(layer, "pre_feedforward_layernorm"):
        return (layer.input_layernorm, layer.post_attention_layernorm,
                layer.pre_feedforward_layernorm, layer.post_feedforward_layernorm)
    return layer.input_layernorm, None, layer.post_attention_layernorm, None


def _run(model, ids: torch.Tensor, embeds=None, block=None, first=None, rules: str = "relp"):
    """The primal over a document: each layer's map, and the last layer's output ``[T, d]``."""
    from neuron_lens.transport import SKIP_FIRST, valid_positions

    if rules not in ("relp", "true"):
        raise ValueError(f"rules must be relp or true, not {rules}")
    exact = rules == "true"
    check(model)
    valid_positions(len(ids))
    dev, dt = model.embed_tokens.weight.device, model.embed_tokens.weight.dtype
    ids = ids[:-1].to(dev)
    T, S = len(ids), SKIP_FIRST if first is None else first
    if S < SKIP_FIRST or S >= T:
        raise ValueError(f"first valid position {S} outside [{SKIP_FIRST}, {T})")
    tp = -(-T // 8) * 8
    # Gemma scales text embeddings by √d in its embedding module; image and audio embeddings come
    # in unscaled, as the model scatters them.
    h = model.embed_tokens(ids) if embeds is None else embeds[:-1].to(dev, dt)
    pos = torch.arange(T, device=dev)[None]
    types = getattr(model.config, "layer_types", None) or ["full_attention"] * len(model.layers)
    by_type = "layer_type" in inspect.signature(model.rotary_emb.forward).parameters  # Gemma 4: one table per type
    tables = {}
    t_idx = torch.arange(T, device=dev)[:, None]
    s_idx = torch.arange(tp, device=dev)[None]
    causal = torch.full((T, tp), float("-inf"), device=dev).triu(1)  # masks the padding too
    both = None
    if block is not None:
        b = block[:-1].to(dev)
        if b[:S].any():
            raise ValueError(f"the bidirectional block starts before position {S}")
        both = F.pad(b[:, None] & b[None, :], (0, tp - T))
    maps = []
    for i, layer in enumerate(model.layers):
        n1, post1, n2, post2 = _norms(layer)
        x, r1 = _rms(n1, h)
        if hasattr(layer, "linear_attn"):
            if exact:
                raise ValueError("true derivatives are not written out for DeltaNet")
            a, mix = _delta_primal(layer.linear_attn, x, S)
        else:
            at = layer.self_attn
            key = types[i] if by_type else None
            if key not in tables:
                c_, s_ = model.rotary_emb(h[None], pos, key) if by_type else model.rotary_emb(h[None], pos)
                tables[key] = (c_[0, :, None], s_[0, :, None])  # [T, 1, rd]
            window = getattr(at, "sliding_window", None)
            mask = causal
            if window is not None:
                if both is not None:
                    mask = mask.masked_fill(both, 0.0)
                mask = mask.masked_fill(s_idx <= t_idx - window, float("-inf"))
            a, mix = _attention(at, x, *tables[key], mask, S, exact)
        a_in = a
        if post1 is not None:
            a, rp1 = _rms(post1, a_in)
        h_mid = h + a
        x2, r2 = _rms(n2, h_mid)
        if hasattr(layer.mlp, "experts"):
            if exact:
                raise ValueError("true derivatives are not written out for sparse MLPs")
            z, mlp = _moe(layer.mlp, x2, S)
        else:
            z, cg, cu = _swiglu(layer.mlp, x2, S, exact)
            mlp = GatedMLP(layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, layer.mlp.down_proj.weight, cg, cu)
        z_in = z
        if post2 is not None:
            z, rp2 = _rms(post2, z_in)
        h_out = h_mid + z
        scalar = getattr(layer, "layer_scalar", None)
        if scalar is not None:
            h_out = h_out * scalar
        maps.append(Layer(
            n1=_norm_map(n1, h, r1, S, exact), mix=mix,
            post1=None if post1 is None else _norm_map(post1, a_in, rp1, S, exact),
            n2=_norm_map(n2, h_mid, r2, S, exact), mlp=mlp,
            post2=None if post2 is None else _norm_map(post2, z_in, rp2, S, exact),
            s=None if scalar is None else scalar.float().reshape(())))
        h = h_out
    return maps, h


@torch.no_grad()
def linearize(model, ids: torch.Tensor, embeds: torch.Tensor | None = None, block: torch.Tensor | None = None,
              first: int | None = None, rules: str = "relp") -> list[Layer]:
    """Run the primal over a document and keep each layer's linear map.

    ``embeds`` (``[len(ids), d]``) replaces the token embeddings, for documents with image or
    audio tokens (``media.multimodal_embeds``); ``block`` (``[len(ids)]`` bool) marks one image's
    tokens, which attend to each other both ways in sliding layers, as Gemma 4's vision mask has
    it. The block must lie in the valid positions, so that no earlier position attends to a later
    one that carries a tangent. ``first`` (default ``SKIP_FIRST``) moves the first valid position
    later: rows before it carry no tangent or cotangent (exact when every seed and read is at or
    after it), which saves their work. ``rules="true"`` keeps every norm's and the dense MLP's
    true derivative (audio documents read better so)."""
    return _run(model, ids, embeds, block, first, rules)[0]


# ------------------------------------------------------------------------------- attention

def _attn_jvp(m: Attention, xn: torch.Tensor) -> torch.Tensor:
    """Tangents at the mixer's (normed) input to tangents at its output, ``[b, N, d]`` fp32."""
    dt = m.wk.dtype
    b, N, _ = xn.shape
    hk, gn, hd = m.qr.shape
    g = gn // N
    qp = F.linear(xn, m.wq)
    if m.sg is None:
        dq = qp.view(b, N, hk * g, hd).float()
    else:
        qp = qp.view(b, N, hk * g, 2 * hd)
        dq, dgate = qp[..., :hd].float(), qp[..., hd:].reshape(b, N, -1).float()
    if m.wqn is not None:
        dq = _norm_jvp(dq, m.q, m.rq, m.wqn)
    dq = _rope(dq, m.cos, m.sin).to(dt).view(b, N, hk, g, hd).permute(0, 2, 3, 1, 4).reshape(b, hk, gn, hd)
    dk = F.linear(xn, m.wk).view(b, N, hk, hd).float()
    dv = dk if m.wv is None else F.linear(xn, m.wv).view(b, N, hk, hd).float()
    if m.wkn is not None:
        dk = _norm_jvp(dk, m.k, m.rk, m.wkn)
    dk = _rope(dk, m.cos, m.sin)
    if m.rv is not None:
        dv = _norm_jvp(dv, m.v, m.rv, 1.0)
    dk, dv = dk.to(dt).transpose(1, 2), dv.to(dt).transpose(1, 2)  # [b, Hk, N, hd]
    keys = slice(m.s0, m.s0 + N)  # the valid positions, as keys
    # (einsums, where a fixed operand meets the tangents: no copy of it per tangent)
    ds = (dq @ m.kr.mT).float()  # [b, Hk, g·N, Tp]
    ds[..., keys] += torch.einsum("hqd,bhkd->bhqk", m.qr, dk).float()  # keys at valid positions move too
    ds *= m.scale
    dp = (m.p * (ds - (m.p * ds).sum(-1, keepdim=True))).to(dt)
    do = (dp @ m.vn + torch.einsum("hqk,bhkd->bhqd", m.p[..., keys], dv))
    do = do.view(b, hk, g, N, hd).permute(0, 3, 1, 2, 4).reshape(b, N, -1)
    if m.sg is not None:  # the output gate, half rule
        do = HALF_RULE_BETA * (do.float() * m.sg + m.o * m.dsg * dgate)
    return F.linear(do.to(dt), m.wo).float()


def _attn_vjp(m: Attention, G: torch.Tensor) -> torch.Tensor:
    """Cotangents at the mixer's output to cotangents at its (normed) input."""
    dt = m.wk.dtype
    b, N, _ = G.shape
    hk, gn, hd = m.qr.shape
    g = gn // N
    gy = G.to(dt) @ m.wo  # [b, N, Hq·hd]
    if m.sg is not None:
        gy = gy.float()
        ggate = HALF_RULE_BETA * gy * m.o * m.dsg
        gy = HALF_RULE_BETA * gy * m.sg
    go = gy.to(dt).view(b, N, hk, g, hd).permute(0, 2, 3, 1, 4).reshape(b, hk, gn, hd)
    keys = slice(m.s0, m.s0 + N)
    gp = (go @ m.vn.mT).float()  # [b, Hk, g·N, Tp]
    gs = (m.p * (gp - (m.p * gp).sum(-1, keepdim=True)) * m.scale).to(dt)
    gq = (gs @ m.kr).view(b, hk, g, N, hd).permute(0, 3, 1, 2, 4).reshape(b, N, hk * g, hd)
    gk = torch.einsum("bhqk,hqd->bkhd", gs[..., keys], m.qr)  # [b, N, Hk, hd]
    gv = torch.einsum("hqk,bhqd->bkhd", m.p[..., keys], go).float()
    gq = _rope(gq.float(), m.cos, -m.sin)
    if m.wqn is not None:
        gq = _norm_vjp(gq, m.q, m.rq, m.wqn)
    gk = _rope(gk.float(), m.cos, -m.sin)
    if m.wkn is not None:
        gk = _norm_vjp(gk, m.k, m.rk, m.wkn)
    if m.rv is not None:
        gv = _norm_vjp(gv, m.v, m.rv, 1.0)
    if m.sg is not None:
        gq = torch.cat((gq, ggate.view(b, N, hk * g, hd)), -1)
    gx = gq.to(dt).reshape(b, N, -1) @ m.wq
    if m.wv is None:  # the value is the raw key: both cotangents reach k_proj
        gx = gx + (gk + gv).to(dt).reshape(b, N, -1) @ m.wk
    else:
        gx = gx + gk.to(dt).reshape(b, N, -1) @ m.wk + gv.to(dt).reshape(b, N, -1) @ m.wv
    return gx.float()


# --------------------------------------------------------------------------- Gated DeltaNet

def _conv(x: torch.Tensor, wc: torch.Tensor) -> torch.Tensor:
    """The causal depthwise convolution over rows (``[b, N, C]``), zero before the first row:
    ``Σ_j wc[:, j] · x[t − K + 1 + j]``, as K shifted products (``F.conv1d`` takes an FFT path on
    CPU that is orders of magnitude slower)."""
    K, N = wc.shape[-1], x.shape[1]
    xp = F.pad(x, (0, 0, K - 1, 0))
    return sum(wc[:, j] * xp[:, j:j + N] for j in range(K))


def _conv_t(g: torch.Tensor, wc: torch.Tensor) -> torch.Tensor:
    """``_conv``'s transpose: ``Σ_j wc[:, j] · g[t + K − 1 − j]``."""
    K, N = wc.shape[-1], g.shape[1]
    gp = F.pad(g, (0, 0, 0, K - 1))
    return sum(wc[:, j] * gp[:, K - 1 - j:K - 1 - j + N] for j in range(K))


def _l2_jvp(dx, x, r):
    return r * (dx - x * r.square() * (x * dx).sum(-1, keepdim=True))


def _delta_core_jvp(m: DeltaNet, dq, dk, dv, db, dg) -> torch.Tensor:
    """The delta rule's map: tangents of the queries and keys (``[b, N, Hk, dk]``, normed, the
    queries scaled), values (``[b, N, Hv, dv]``), β and the cumulative log decay (``[b, N, Hv]``)
    to the output's, ``[b, N, Hv, dv]``, before its norm:

        do = dW U + W L⁻¹ r,   r = dβ (V − C U) + β (dv − dC U),

    with ``dW = (dQ Kᵀ + Q dKᵀ) ⊙ Γ + (Q Kᵀ) ⊙ dΓ``, ``dΓ_ts = Γ_ts (dgc_t − dgc_s)``, and ``dC``
    alike (strictly lower). One pass forward over the chunks: within each, ``Γ``-weighted
    ``C×C`` blocks; before it, the memory ``H_ku`` (fixed) and one tangent state
    ``Σ Γ ((dk_s − dgc_s k_s) u_sᵀ + k_s x_sᵀ)``, which carries the keys' and the decay's moves
    and ``x = L⁻¹ r`` (forward substitution: ``x = tinv (r − β (k ⊙ qd) state)`` per chunk).
    ``neuron_lens.kernels.delta_jvp`` runs the same pass on CUDA."""
    b, N, hv, dvd = dv.shape
    _, tp, dkd = m.q.shape
    n, S = tp // CHUNK, m.s0
    chunks = lambda x: F.pad(x.repeat_interleave(hv // x.shape[2], 2).transpose(1, 2),
                             (0, 0, S, tp - S - N)).unflatten(2, (n, CHUNK))  # [b, Hv, n, C, ·]
    dq, dk, dv, db, dg = chunks(dq), chunks(dk), chunks(dv), chunks(db[..., None]), chunks(dg[..., None])
    q, k, u, vmc, qh, kh = (x.unflatten(1, (n, CHUNK)) for x in (m.q, m.k, m.u, m.vmc, m.qh, m.kh))
    beta = m.beta.view(hv, n, CHUNK, 1)
    dec, qd, kd, cd = _decays(m.gc)
    H = dv.new_zeros(b, hv, dkd, dvd)
    out = []
    for c in range(n):
        qc, kc, uc, d, qdc, kdc = q[:, c], k[:, c], u[:, c], dec[:, c], qd[:, c, :, None], kd[:, c, :, None]
        dqc, dkc, dgc, hku = dq[:, :, c], dk[:, :, c], dg[:, :, c], m.hku[:, c]
        e = dgc - dgc.mT  # dgc_t − dgc_s
        a0 = (torch.einsum("bhcd,hed->bhce", dqc, kc) + torch.einsum("hcd,bhed->bhce", qc, dkc)) * d + e * m.qkd[:, c]
        a1 = (torch.einsum("bhcd,hed->bhce", dkc, kc) + torch.einsum("hcd,bhed->bhce", kc, dkc)) * d.tril(-1) \
            + e * m.kkd[:, c]
        dcu = (torch.einsum("bhce,hef->bhcf", a1, uc) + torch.einsum("bhcd,hdf->bhcf", dkc * qdc, hku) + dgc * kh[:, c]
               + torch.einsum("hcd,bhdf->bhcf", kc * qdc, H))  # dC U, and C's part of the substitution
        x = torch.einsum("hce,bhef->bhcf", m.tinv[:, c], db[:, :, c] * vmc[:, c] + beta[:, c] * (dv[:, :, c] - dcu))
        out.append(torch.einsum("bhce,hef->bhcf", a0, uc) + torch.einsum("bhcd,hdf->bhcf", dqc * qdc, hku)
                   + dgc * qh[:, c] + torch.einsum("hcd,bhdf->bhcf", qc * qdc, H)
                   + torch.einsum("hce,bhef->bhcf", m.qkd[:, c], x))
        H = cd[:, c, None, None] * H + torch.einsum("bhcd,hcf->bhdf", (dkc - dgc * kc) * kdc, uc) \
            + torch.einsum("hcd,bhcf->bhdf", kc * kdc, x)
    return torch.stack(out, 2).flatten(2, 3)[:, :, S:S + N].transpose(1, 2)


def _delta_core_vjp(m: DeltaNet, go: torch.Tensor):
    """``_delta_core_jvp``'s transpose: the output's cotangents ``[b, N, Hv, dv]`` to the
    queries' and keys' (``[b, N, Hv, dk]``, per value head), values', β's and the cumulative
    decay's. With
    ``gr = L⁻ᵀ Wᵀ go`` and ``gbl = −β gr`` (dC U's cotangent), one pass back over the chunks
    through ``A = Σ_{later} Γ (q_t go_tᵀ + k_t gbl_tᵀ)``, which ``gr`` reads through the keys and
    the keys' cotangent as writers (``gkt``) through the pseudo-values; as readers (``gq`` through
    W, ``gkh`` through C) they need only ``H_ku``. ``dΓ`` gives the decay's,
    ``gq·q + (gkh − gkt)·k``. ``neuron_lens.kernels.delta_vjp`` on CUDA."""
    b, N, hv, dvd = go.shape
    _, tp, dkd = m.q.shape
    n, S = tp // CHUNK, m.s0
    go = F.pad(go.transpose(1, 2), (0, 0, S, tp - S - N)).unflatten(2, (n, CHUNK))  # [b, Hv, n, C, dv]
    valid = torch.zeros(tp, 1, device=go.device)
    valid[S:S + N] = 1
    valid = valid.view(n, CHUNK, 1)
    q, k, u, vmc = (x.unflatten(1, (n, CHUNK)) for x in (m.q, m.k, m.u, m.vmc))
    beta = m.beta.view(hv, n, CHUNK, 1)
    dec, qd, kd, cd = _decays(m.gc)
    A = go.new_zeros(b, hv, dkd, dvd)
    grads = [None] * n
    for c in reversed(range(n)):
        qc, kc, uc, d, qdc, kdc, goc = q[:, c], k[:, c], u[:, c], dec[:, c], qd[:, c, :, None], kd[:, c, :, None], go[:, :, c]
        rhs = torch.einsum("hts,bhtf->bhsf", m.qkd[:, c], goc) \
            + torch.einsum("hsd,bhdf->bhsf", kc * kdc, A)
        gr = torch.einsum("hts,bhtf->bhsf", m.tinv[:, c], rhs) * valid[c]  # Tᵀ rhs
        gbl = -beta[:, c] * gr
        p, pb = torch.einsum("bhtf,hsf->bhts", goc, uc) * d, torch.einsum("bhtf,hsf->bhts", gbl, uc) * d.tril(-1)
        gq = torch.einsum("bhts,hsd->bhtd", p, kc) + torch.einsum("bhtf,hdf->bhtd", goc * qdc, m.hku[:, c])
        gkh = torch.einsum("bhts,hsd->bhtd", pb, kc) + torch.einsum("bhtf,hdf->bhtd", gbl * qdc, m.hku[:, c])
        gkt = torch.einsum("bhts,htd->bhsd", p, qc) + torch.einsum("bhts,htd->bhsd", pb, kc) \
            + torch.einsum("hsf,bhdf->bhsd", uc * kdc, A)
        grads[c] = (gq, gkh + gkt, (gq * qc).sum(-1) + ((gkh - gkt) * kc).sum(-1), beta[:, c] * gr,
                    (gr * vmc[:, c]).sum(-1))
        A = cd[:, c, None, None] * A + torch.einsum("htd,bhtf->bhdf", qc * qdc, goc) \
            + torch.einsum("htd,bhtf->bhdf", kc * qdc, gbl)
    rows = lambda i: torch.stack([g[i] for g in grads], 2).flatten(2, 3)[:, :, S:S + N].transpose(1, 2)
    return rows(0), rows(1), rows(3), rows(4), rows(2)


def _delta_jvp(m: DeltaNet, xn: torch.Tensor) -> torch.Tensor:
    """Tangents at the mixer's (normed) input to tangents at its output, ``[b, N, d]`` fp32."""
    dt = m.w_qkv.dtype
    b, N, _ = xn.shape
    hv, dv = m.u.shape[0], m.u.shape[-1]
    hk, dk = m.q0.shape[1:]
    cuda = kernels is not None and xn.is_cuda
    if cuda:  # the convolution and L2 norms in one kernel
        dq, dkk, dvv = kernels.delta_conv_in(m, F.linear(xn, m.w_qkv))
    else:
        dc = _conv(F.linear(xn, m.w_qkv).float(), m.wc) * m.sc  # [b, N, C] (identity rule)
        dq0, dk0, dvv = dc.split([hk * dk, hk * dk, hv * dv], -1)
        dq = _l2_jvp(dq0.view(b, N, hk, dk), m.q0, m.rq0) * dk ** -0.5
        dkk = _l2_jvp(dk0.view(b, N, hk, dk), m.k0, m.rk0)
    db = m.db * F.linear(xn, m.w_b).float()  # [b, N, Hv]
    dg = (m.dga * F.linear(xn, m.w_a).float()).cumsum(1)
    do = (kernels.delta_jvp if cuda else _delta_core_jvp)(m, dq, dkk, dvv.view(b, N, hv, dv), db, dg)
    dn = do * m.ao * m.wn  # the output norm, LN rule
    dz = F.linear(xn, m.w_z).float().view(b, N, hv, dv)
    dy = HALF_RULE_BETA * (dn * m.z + m.on * m.sz * dz)
    return F.linear(dy.reshape(b, N, -1).to(dt), m.w_out).float()


def _delta_vjp(m: DeltaNet, G: torch.Tensor) -> torch.Tensor:
    """Cotangents at the mixer's output to cotangents at its (normed) input."""
    dt = m.w_qkv.dtype
    b, N, _ = G.shape
    hv, dv = m.u.shape[0], m.u.shape[-1]
    hk, dk = m.q0.shape[1:]
    gy = (G.to(dt) @ m.w_out).float().view(b, N, hv, dv)
    gdn, gz = HALF_RULE_BETA * gy * m.z, HALF_RULE_BETA * gy * m.on * m.sz
    gx = (gz.reshape(b, N, -1).to(dt) @ m.w_z).float()
    cuda = kernels is not None and G.is_cuda
    gdq, gdk, gdv, gbeta, gdgc = (kernels.delta_vjp if cuda else _delta_core_vjp)(m, gdn * m.ao * m.wn)
    gdg = gdgc.flip(1).cumsum(1).flip(1)
    gx = gx + ((gbeta * m.db).to(dt) @ m.w_b + (gdg * m.dga).to(dt) @ m.w_a).float()
    if cuda:  # the L2 norms and the convolution in one kernel
        return gx + (kernels.delta_conv_out(m, gdq, gdk, gdv, dt) @ m.w_qkv).float()
    unheads = lambda x: x.unflatten(2, (hk, hv // hk)).sum(3)  # value heads → their key head
    gq0 = _l2_jvp(unheads(gdq) * dk ** -0.5, m.q0, m.rq0).reshape(b, N, -1)
    gk0 = _l2_jvp(unheads(gdk), m.k0, m.rk0).reshape(b, N, -1)
    gc = torch.cat((gq0, gk0, gdv.reshape(b, N, -1)), -1) * m.sc
    return gx + (_conv_t(gc, m.wc).to(dt) @ m.w_qkv).float()


# --------------------------------------------------------------------------------- the MLPs

def _mlp_jvp(m: GatedMLP, xn):
    return F.linear(m.cg * F.linear(xn, m.wg) + m.cu * F.linear(xn, m.wu), m.wd).float()


def _mlp_vjp(m: GatedMLP, G):
    gp = G.to(m.wd.dtype) @ m.wd
    return ((m.cg * gp) @ m.wg + (m.cu * gp) @ m.wu).float()


def _experts_fwd(m: MoE, x: torch.Tensor) -> torch.Tensor:
    """Tangents ``x`` (``[b, N, d]``) through each (row, slot)'s expert's RelP map, weighted by the
    router and summed over slots; ``[b, N, d]`` fp32. The pairs run in expert order, the tangent
    copies inside each expert's group (``[M·b, d]``)."""
    b, N, d = x.shape
    I = m.cg.shape[-1]
    offs = m.offs * b
    if kernels is not None and x.is_cuda:  # the gather, product and rule in one kernel
        h = kernels.moe_up(m.pos, m.offs, m.cg, m.cu, x, m.w_gu, jvp=True)
    else:
        gu = _gmm(x.transpose(0, 1)[m.pos].reshape(-1, d), m.w_gu, offs).view(-1, b, 2 * I)
        h = (m.cg[:, None] * gu[..., :I] + m.cu[:, None] * gu[..., I:]).reshape(-1, I)
    return _combine(m, _gmm(h, m.w_dn, offs).view(-1, b, d))


def _experts_bwd(m: MoE, g: torch.Tensor) -> torch.Tensor:
    """``_experts_fwd``'s transpose (the router's weight scales each pair's output either side)."""
    b, N, d = g.shape
    I = m.cg.shape[-1]
    offs = m.offs * b
    if kernels is not None and g.is_cuda:
        h = kernels.moe_up(m.pos, m.offs, m.cg, m.cu, g, m.w_dn, jvp=False)
    else:
        gp = _gmm(g.transpose(0, 1)[m.pos].reshape(-1, d), m.w_dn.transpose(1, 2), offs).view(-1, b, I)
        h = torch.cat((m.cg[:, None] * gp, m.cu[:, None] * gp), -1).reshape(-1, 2 * I)
    return _combine(m, _gmm(h, m.w_gu.transpose(1, 2), offs).view(-1, b, d))


def _combine(m: MoE, y: torch.Tensor) -> torch.Tensor:
    """Pairs' outputs (``[M, b, d]``, expert order) to rows: ``Σ_slot w · y``, ``[b, N, d]`` fp32 (a
    gather, fused with the sum, where scattering into rows would take atomic adds;
    ``kernels.moe_combine`` on CUDA)."""
    if kernels is not None and y.is_cuda:
        return kernels.moe_combine(m.inv, m.slot_w, y)
    return (y[m.inv] * m.slot_w[..., None, None]).sum(1).transpose(0, 1)


def _moe_jvp(m: MoE, xn: torch.Tensor) -> torch.Tensor:
    b, N, d = xn.shape
    out = _experts_fwd(m, xn)
    # the router's softmax over the chosen experts
    dl = F.linear(xn, m.w_r).float().gather(-1, m.idx.expand(b, -1, -1))  # [b, N, k]
    dw = m.slot_w * (dl - (m.slot_w * dl).sum(-1, keepdim=True))
    out = out + torch.einsum("bnk,nkd->bnd", dw, m.eo)
    if m.sh is not None:  # the shared expert, behind its gate (a one-expert router)
        wg, wu, wd, scg, scu = m.sh
        out = out + m.sg * F.linear(scg * F.linear(xn, wg) + scu * F.linear(xn, wu), wd).float()
        if m.ws is not None:
            out = out + m.dsg * F.linear(xn, m.ws).float() * m.so
    return out


def _moe_vjp(m: MoE, G: torch.Tensor) -> torch.Tensor:
    dt = m.w_gu.dtype
    b, N, d = G.shape
    gw = torch.einsum("bnd,nkd->bnk", G, m.eo)  # the router's softmax over the chosen experts
    gl = m.slot_w * (gw - (m.slot_w * gw).sum(-1, keepdim=True))
    gx = (torch.zeros(b, N, m.w_r.shape[0], device=G.device).scatter_add(
        -1, m.idx.expand(b, -1, -1), gl).to(dt) @ m.w_r).float()
    if m.sh is not None:
        wg, wu, wd, scg, scu = m.sh
        gp = (m.sg * G).to(dt) @ wd
        gx = gx + ((scg * gp) @ wg + (scu * gp) @ wu).float()
        if m.ws is not None:
            gx = gx + ((m.dsg * (G * m.so).sum(-1, keepdim=True)).to(dt) @ m.ws).float()
    return gx + _experts_bwd(m, G.to(dt))


# --------------------------------------------------------------------------------- the layer

def _nj(n: Norm, dx: torch.Tensor) -> torch.Tensor:
    """A norm's map: the LN rule's scaling, or the true derivative."""
    if n.x is None:
        return dx * n.a
    return n.a * (dx - n.x * n.r.square() * (n.x * dx).mean(-1, keepdim=True))


def _nt(n: Norm, gy: torch.Tensor) -> torch.Tensor:
    """``_nj``'s transpose (the true derivative ``a·(I − x̂x̂ᵀ/d)·r`` is symmetric up to the scale's side)."""
    g = gy * n.a
    if n.x is None:
        return g
    return g - n.x * n.r.square() * (n.x * g).mean(-1, keepdim=True)


def _dtype(m: Layer) -> torch.dtype:
    return (m.mix.wk if isinstance(m.mix, Attention) else m.mix.w_qkv).dtype


def jvp(m: Layer, X: torch.Tensor):
    """Push tangents ``X`` (``[b, N, d]``, fp32) at a layer's input through its map; return
    them at its output and at the MLP's input (``mid``)."""
    dt = _dtype(m)
    xn = _nj(m.n1, X).to(dt)
    a = _attn_jvp(m.mix, xn) if isinstance(m.mix, Attention) else _delta_jvp(m.mix, xn)
    mid = X + (a if m.post1 is None else _nj(m.post1, a))
    xn = _nj(m.n2, mid).to(dt)
    z = _moe_jvp(m.mlp, xn) if isinstance(m.mlp, MoE) else _mlp_jvp(m.mlp, xn)
    out = mid + (z if m.post2 is None else _nj(m.post2, z))
    return (out if m.s is None else out * m.s), mid


def vjp(m: Layer, G: torch.Tensor, mid_seed: torch.Tensor | None = None, row_w: torch.Tensor | None = None):
    """Pull cotangents ``G`` (``[b, N, d]``, fp32) at a layer's output back through its map;
    return them at its input, and their means over positions at the MLP's input (``mid``) and
    at the layer's input (weighted by ``row_w``, ``[N]`` summing to 1, if given). ``mid_seed``
    (``[b, N, d]``) joins the cotangents at the MLP's input."""
    if m.s is not None:
        G = G * m.s
    gz = G if m.post2 is None else _nt(m.post2, G)
    G = G + _nt(m.n2, _moe_vjp(m.mlp, gz) if isinstance(m.mlp, MoE) else _mlp_vjp(m.mlp, gz))
    if mid_seed is not None:
        G = G + mid_seed
    mid = G.mean(1) if row_w is None else (G * row_w[:, None]).sum(1)
    ga = G if m.post1 is None else _nt(m.post1, G)
    G = G + _nt(m.n1, _attn_vjp(m.mix, ga) if isinstance(m.mix, Attention) else _delta_vjp(m.mix, ga))
    return G, mid, G.mean(1) if row_w is None else (G * row_w[:, None]).sum(1)
