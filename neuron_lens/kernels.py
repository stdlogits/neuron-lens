"""Triton kernels for the layer maps' hot spots on CUDA: Gated DeltaNet's derivative (its delta
rule, and the convolution with the L2 norms around it), and the sparse MLP's experts (the first
product, gathering its rows, and the sum over each row's experts).

DeltaNet: ``delta_jvp`` and ``delta_vjp`` run
``layers._delta_core_jvp`` and ``layers._delta_core_vjp`` (the reference, and the path off CUDA).
The map walks the chunks forward, one program per tangent copy, value head and block of value
features, its state in registers; the transpose walks them back for ``gr`` (``_vjp_r``), then takes the
queries' and keys' cotangents chunk by chunk in parallel (``_vjp_k``). The document's
constants are shared by every tangent copy, so they mostly come from L2. Products take TF32, as
``02_fit.py`` sets for torch's.

Tangents arrive in the layout the projections give, ``[b, N, H, ·]`` at valid rows, and leave the
same way; a chunk's rows before the first valid one or past the last load as zeros."""
import math

import torch
import triton
import triton.language as tl

# the experts' first product: rows, columns and inner block, num_warps, num_stages (jvp, vjp)
MOE = {"jvp": (64, 256, 64, 8, 3), "vjp": (128, 256, 64, 8, 4)}
# per DeltaNet kernel: key and value feature blocks, num_warps, num_stages (tuned on an H100; a
# register cap, maxnreg, only slowed them)
CONFIG = {"jvp": (32, 128, 4, 1), "vjp_r": (128, 128, 8, 2), "vjp_k": (64, 64, 4, 1)}
VJPK_COPIES = 16  # tangent copies per program in the chunk-parallel ``_vjp_k``
# the convolution kernels: rows per program, num_warps, tangent copies per program
CONV = {"in": (16, 4, 8), "out": (16, 4, 16)}


@triton.jit
def _jvp(q, k, u, vmc, beta, gc, hku, tinv, qkd, kkd, qh, kh, dq, dk, dv, db, dg, out,
         sqb, sqn, sqh, skb, skn, skh, svb, svn, svh, sbb, sbn, sgb, sgn, sob, son, soh,
         S, N, HV: tl.constexpr, REP: tl.constexpr, NT: tl.constexpr, C: tl.constexpr,
         DK: tl.constexpr, DV: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // HV, i_bh % HV
    i_hk = i_h // REP
    NK: tl.constexpr = DK // BK
    TP: tl.constexpr = NT * C
    oc = tl.arange(0, C)
    obk = tl.arange(0, BK)
    ov = i_v * BV + tl.arange(0, BV)
    lower = oc[:, None] >= oc[None, :]
    strict = oc[:, None] > oc[None, :]
    # the tangent state, Σ Γ ((dk − dgc k) uᵀ + k xᵀ), by blocks of key features
    h0 = tl.zeros([BK, BV], dtype=tl.float32)
    if NK > 1:
        h1 = tl.zeros([BK, BV], dtype=tl.float32)
    if NK > 2:
        h2 = tl.zeros([BK, BV], dtype=tl.float32)
        h3 = tl.zeros([BK, BV], dtype=tl.float32)
    for c in range(NT):
        t = c * C + oc
        r = t - S
        ok = (r >= 0) & (r < N)
        g = tl.load(gc + i_h * TP + t)
        g_end = tl.load(gc + i_h * TP + c * C + C - 1)
        g_prev = tl.load(gc + i_h * TP + c * C - 1, mask=c > 0, other=0.)
        qd = tl.exp(g - g_prev)
        kd = tl.exp(g_end - g)
        cd = tl.exp(g_end - g_prev)
        dec = tl.exp(tl.where(lower, g[:, None] - g[None, :], float("-inf")))
        dgv = tl.load(dg + i_b * sgb + r * sgn + i_h, mask=ok, other=0.)
        dbv = tl.load(db + i_b * sbb + r * sbn + i_h, mask=ok, other=0.)
        a0 = tl.zeros([C, C], dtype=tl.float32)
        a1 = tl.zeros([C, C], dtype=tl.float32)
        o = tl.zeros([C, BV], dtype=tl.float32)
        dcu = tl.zeros([C, BV], dtype=tl.float32)
        for kb in tl.static_range(NK):
            ck = kb * BK + obk
            qq = tl.load(q + (i_h * TP + t[:, None]) * DK + ck[None, :])
            kk = tl.load(k + (i_h * TP + t[:, None]) * DK + ck[None, :])
            dqq = tl.load(dq + i_b * sqb + r[:, None] * sqn + i_hk * sqh + ck[None, :], mask=ok[:, None], other=0.)
            dkk = tl.load(dk + i_b * skb + r[:, None] * skn + i_hk * skh + ck[None, :], mask=ok[:, None], other=0.)
            a0 += tl.dot(dqq, tl.trans(kk)) + tl.dot(qq, tl.trans(dkk))
            a1 += tl.dot(dkk, tl.trans(kk)) + tl.dot(kk, tl.trans(dkk))
            hs = tl.load(hku + ((i_h * NT + c) * DK + ck[:, None]) * DV + ov[None, :])
            o += tl.dot(dqq * qd[:, None], hs)
            dcu += tl.dot(dkk * qd[:, None], hs)
            if kb == 0:
                hb = h0
            elif kb == 1:
                hb = h1
            elif kb == 2:
                hb = h2
            else:
                hb = h3
            o += tl.dot(qq * qd[:, None], hb)
            dcu += tl.dot(kk * qd[:, None], hb)
        sq = (i_h * NT + c) * C * C + oc[:, None] * C + oc[None, :]
        e = dgv[:, None] - dgv[None, :]  # the decay's move, Γ_ts (dgc_t − dgc_s)
        qk = tl.load(qkd + sq)
        a0 = a0 * dec + e * qk
        a1 = tl.where(strict, a1 * dec, 0.) + e * tl.load(kkd + sq)
        fv = (i_h * TP + t[:, None]) * DV + ov[None, :]
        uu = tl.load(u + fv)
        o += dgv[:, None] * tl.load(qh + fv) + tl.dot(a0, uu)
        dcu += dgv[:, None] * tl.load(kh + fv) + tl.dot(a1, uu)
        dvv = tl.load(dv + i_b * svb + r[:, None] * svn + i_h * svh + ov[None, :], mask=ok[:, None], other=0.)
        bt = tl.load(beta + i_h * TP + t)
        x = tl.dot(tl.load(tinv + sq), dbv[:, None] * tl.load(vmc + fv) + bt[:, None] * (dvv - dcu))
        o += tl.dot(qk, x)
        tl.store(out + i_b * sob + r[:, None] * son + i_h * soh + ov[None, :], o, mask=ok[:, None])
        for kb in tl.static_range(NK):
            ck = kb * BK + obk
            kk = tl.load(k + (i_h * TP + t[:, None]) * DK + ck[None, :]) * kd[:, None]
            dkk = tl.load(dk + i_b * skb + r[:, None] * skn + i_hk * skh + ck[None, :], mask=ok[:, None], other=0.)
            upd = tl.dot(tl.trans(dkk * kd[:, None] - dgv[:, None] * kk), uu) + tl.dot(tl.trans(kk), x)
            if kb == 0:
                h0 = h0 * cd + upd
            elif kb == 1:
                h1 = h1 * cd + upd
            elif kb == 2:
                h2 = h2 * cd + upd
            else:
                h3 = h3 * cd + upd


@triton.jit
def _vjp_r(q, k, u, vmc, beta, gc, tinv, qkd, go, gdv, gbp, gks,
           sgb, sgn, sgh, sdb, sdn, sdh, spv, spb, spn, ssb, ssn, ssh,
           S, N, HV: tl.constexpr, NT: tl.constexpr, C: tl.constexpr,
           DK: tl.constexpr, DV: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    """``gr = L⁻ᵀ Wᵀ go``, back over the chunks: stores ``β gr`` (the values' cotangent), this
    value block's part of β's, ``gr · (V − C U)``, and adds its part of the keys' cotangent
    through the state, ``(u ⊙ kd) Aᵀ``, to ``gks``."""
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // HV, i_bh % HV
    NK: tl.constexpr = DK // BK
    TP: tl.constexpr = NT * C
    oc = tl.arange(0, C)
    obk = tl.arange(0, BK)
    ov = i_v * BV + tl.arange(0, BV)
    # A = Σ_{later} Γ (q goᵀ + k gblᵀ), by blocks of key features
    a0 = tl.zeros([BK, BV], dtype=tl.float32)
    if NK > 1:
        a1 = tl.zeros([BK, BV], dtype=tl.float32)
    if NK > 2:
        a2 = tl.zeros([BK, BV], dtype=tl.float32)
        a3 = tl.zeros([BK, BV], dtype=tl.float32)
    for i in range(NT):
        c = NT - 1 - i
        t = c * C + oc
        r = t - S
        ok = (r >= 0) & (r < N)
        g = tl.load(gc + i_h * TP + t)
        g_end = tl.load(gc + i_h * TP + c * C + C - 1)
        g_prev = tl.load(gc + i_h * TP + c * C - 1, mask=c > 0, other=0.)
        qd = tl.exp(g - g_prev)
        kd = tl.exp(g_end - g)
        cd = tl.exp(g_end - g_prev)
        goc = tl.load(go + i_b * sgb + r[:, None] * sgn + i_h * sgh + ov[None, :], mask=ok[:, None], other=0.)
        sq = (i_h * NT + c) * C * C + oc[:, None] * C + oc[None, :]
        rhs = tl.dot(tl.trans(tl.load(qkd + sq)), goc)
        fv = (i_h * TP + t[:, None]) * DV + ov[None, :]
        uu = tl.load(u + fv) * kd[:, None]
        for kb in tl.static_range(NK):
            ck = kb * BK + obk
            kk = tl.load(k + (i_h * TP + t[:, None]) * DK + ck[None, :]) * kd[:, None]
            if kb == 0:
                ab = a0
            elif kb == 1:
                ab = a1
            elif kb == 2:
                ab = a2
            else:
                ab = a3
            rhs = tl.dot(kk, ab, rhs)
            gs = gks + i_b * ssb + r[:, None] * ssn + i_h * ssh + ck[None, :]
            if DV == BV:
                tl.store(gs, tl.dot(uu, tl.trans(ab)), mask=ok[:, None])
            else:
                tl.atomic_add(gs, tl.dot(uu, tl.trans(ab)), mask=ok[:, None])
        x = tl.where(ok[:, None], tl.dot(tl.trans(tl.load(tinv + sq)), rhs), 0.)
        bt = tl.load(beta + i_h * TP + t)
        tl.store(gdv + i_b * sdb + r[:, None] * sdn + i_h * sdh + ov[None, :], bt[:, None] * x, mask=ok[:, None])
        tl.store(gbp + i_v * spv + i_b * spb + r * spn + i_h, tl.sum(x * tl.load(vmc + fv), 1), mask=ok)
        xb = bt[:, None] * x
        for kb in tl.static_range(NK):
            ck = kb * BK + obk
            qq = tl.load(q + (i_h * TP + t[:, None]) * DK + ck[None, :]) * qd[:, None]
            kk = tl.load(k + (i_h * TP + t[:, None]) * DK + ck[None, :]) * qd[:, None]
            upd = tl.dot(tl.trans(kk), -xb, tl.dot(tl.trans(qq), goc))
            if kb == 0:
                a0 = a0 * cd + upd
            elif kb == 1:
                a1 = a1 * cd + upd
            elif kb == 2:
                a2 = a2 * cd + upd
            else:
                a3 = a3 * cd + upd


@triton.jit
def _vjp_k(q, k, u, gc, hku, go, gdv, gks, gq, gk, gcp,
           sgb, sgn, sgh, sdb, sdn, sdh, ssb, ssn, ssh, sqb, sqn, sqh, spk, spb, spn,
           S, N, HV: tl.constexpr, NT: tl.constexpr, C: tl.constexpr,
           DK: tl.constexpr, DV: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
           B, CB: tl.constexpr):
    """One chunk's cotangents of the queries (``gq``, through W) and keys (through C as readers,
    ``gkh``; as writers, ``gkt``, with ``_vjp_r``'s state term) on a block of key features, per
    value head (adding into key heads in the kernel, with atomics, was slower), and this block's
    part of the cumulative decay's."""
    i_k, c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bb, i_h = i_bh // HV, i_bh % HV
    TP: tl.constexpr = NT * C
    oc = tl.arange(0, C)
    obv = tl.arange(0, BV)
    ck = i_k * BK + tl.arange(0, BK)
    t = c * C + oc
    r = t - S
    ok = (r >= 0) & (r < N)
    g = tl.load(gc + i_h * TP + t)
    g_prev = tl.load(gc + i_h * TP + c * C - 1, mask=c > 0, other=0.)
    qd = tl.exp(g - g_prev)
    dec = tl.exp(tl.where(oc[:, None] >= oc[None, :], g[:, None] - g[None, :], float("-inf")))
    qq = tl.load(q + (i_h * TP + t[:, None]) * DK + ck[None, :])
    kk = tl.load(k + (i_h * TP + t[:, None]) * DK + ck[None, :])
    for i_b in range(i_bb * CB, tl.minimum(i_bb * CB + CB, B)):  # CB copies: the chunk's constants stay in L1
        p = tl.zeros([C, C], dtype=tl.float32)
        pb = tl.zeros([C, C], dtype=tl.float32)
        gqc = tl.zeros([C, BK], dtype=tl.float32)
        gkh = tl.zeros([C, BK], dtype=tl.float32)
        for vb in tl.static_range(DV // BV):
            cv = vb * BV + obv
            goc = tl.load(go + i_b * sgb + r[:, None] * sgn + i_h * sgh + cv[None, :], mask=ok[:, None], other=0.)
            gbl = -tl.load(gdv + i_b * sdb + r[:, None] * sdn + i_h * sdh + cv[None, :], mask=ok[:, None], other=0.)
            uu = tl.load(u + (i_h * TP + t[:, None]) * DV + cv[None, :])
            hs = tl.load(hku + ((i_h * NT + c) * DK + ck[None, :]) * DV + cv[:, None])  # [BV, BK]
            p = tl.dot(goc, tl.trans(uu), p)
            pb = tl.dot(gbl, tl.trans(uu), pb)
            gqc = tl.dot(goc * qd[:, None], hs, gqc)
            gkh = tl.dot(gbl * qd[:, None], hs, gkh)
        p = p * dec
        pb = tl.where(oc[:, None] > oc[None, :], pb * dec, 0.)
        gqc = tl.dot(p, kk, gqc)
        gkh = tl.dot(pb, kk, gkh)
        gkt = tl.load(gks + i_b * ssb + r[:, None] * ssn + i_h * ssh + ck[None, :], mask=ok[:, None], other=0.)
        gkt = tl.dot(tl.trans(pb), kk, tl.dot(tl.trans(p), qq, gkt))
        rows = i_b * sqb + r[:, None] * sqn + i_h * sqh + ck[None, :]
        tl.store(gq + rows, gqc, mask=ok[:, None])
        tl.store(gk + rows, gkh + gkt, mask=ok[:, None])
        tl.store(gcp + i_k * spk + i_b * spb + r * spn + i_h, tl.sum(gqc * qq + (gkh - gkt) * kk, 1), mask=ok)


@triton.jit
def _combine(y, inv, w, out, N, B: tl.constexpr, D: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
             BD: tl.constexpr):
    """``out[j, n] = Σ_slot w[n, slot] · y[inv[n, slot], j]`` for one row ``n`` and copy ``j``."""
    n, j = tl.program_id(0), tl.program_id(1)
    os_ = tl.arange(0, KP)
    on = os_ < K
    pairs = tl.load(inv + n * K + os_, mask=on, other=0)
    ws = tl.load(w + n * K + os_, mask=on, other=0.)
    for d0 in tl.static_range(0, D, BD):
        od = d0 + tl.arange(0, BD)
        ys = tl.load(y + (pairs[:, None] * B + j) * D + od[None, :], mask=on[:, None], other=0.)
        tl.store(out + (j * N + n) * D + od, tl.sum(ys.to(tl.float32) * ws[:, None], 0))


def moe_combine(inv, w, y):
    """``layers._combine`` on CUDA: pairs' outputs ``y`` (``[M, b, d]``, any dtype) summed into rows
    with the routing weights, ``[b, N, d]`` fp32."""
    N, K = inv.shape
    _, b, D = y.shape
    assert y.is_contiguous() and inv.is_contiguous() and w.is_contiguous()
    out = torch.empty(b, N, D, device=y.device, dtype=torch.float32)
    _combine[(N, b)](y, inv, w, out, N, B=b, D=D, K=K, KP=triton.next_power_of_2(K), BD=math.gcd(D, 1024), num_warps=4)
    return out


@triton.jit
def _moe_up(x, w, tile_e, tile_r0, rend, pos, cg, cu, h, sxb, sxn, swe, swr, swc, B,
            D: tl.constexpr, I: tl.constexpr, JVP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
            BK: tl.constexpr):
    """The experts' first product on one tile of rows (``pair·B + copy``, in expert order, one
    expert per tile), gathering each row from ``x`` (``[B, N, D]``). JVP: ``h = cg ⊙ (x W_gateᵀ) +
    cu ⊙ (x W_upᵀ)`` (``[·, I]``). VJP (``w`` the down projection, ``[E, D, I]``):
    ``h = (cg ⊙ x W, cu ⊙ x W)`` (``[·, 2I]``)."""
    t, i_n = tl.program_id(0), tl.program_id(1)
    e = tl.load(tile_e + t)
    live = e >= 0
    e = tl.maximum(e, 0)
    r = tl.load(tile_r0 + t) + tl.arange(0, BM)
    okm = live & (r < tl.load(rend + e))
    p, j = r // B, r % B
    a_row = x + j * sxb + tl.load(pos + p, mask=okm, other=0) * sxn
    on = i_n * BN + tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    if JVP:
        acc_u = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, D, BK):
        ok_ = k0 + tl.arange(0, BK)
        a = tl.load(a_row[:, None] + ok_[None, :], mask=okm[:, None], other=0.)
        if JVP:  # w: [E, 2I, D], gate rows then up rows
            acc = tl.dot(a, tl.load(w + e * swe + on[None, :] * swr + ok_[:, None] * swc), acc)
            acc_u = tl.dot(a, tl.load(w + e * swe + (I + on)[None, :] * swr + ok_[:, None] * swc), acc_u)
        else:  # w: [E, D, I]
            acc = tl.dot(a, tl.load(w + e * swe + ok_[:, None] * swr + on[None, :] * swc), acc)
    f = p[:, None] * I + on[None, :]
    g_ = tl.load(cg + f, mask=okm[:, None], other=0.).to(tl.float32)
    u_ = tl.load(cu + f, mask=okm[:, None], other=0.).to(tl.float32)
    if JVP:
        tl.store(h + r[:, None] * I + on[None, :], (g_ * acc + u_ * acc_u).to(h.dtype.element_ty), mask=okm[:, None])
    else:
        tl.store(h + r[:, None] * (2 * I) + on[None, :], (g_ * acc).to(h.dtype.element_ty), mask=okm[:, None])
        tl.store(h + r[:, None] * (2 * I) + I + on[None, :], (u_ * acc).to(h.dtype.element_ty), mask=okm[:, None])


def moe_up(pos, offs, cg, cu, x, w, jvp: bool):
    """``layers._experts_fwd``'s (``jvp``) or ``_experts_bwd``'s first product on CUDA: tangents
    ``x`` (``[b, N, d]``) at each pair's row, through its expert's ``w`` (``w_gu``, or ``w_dn``)
    and the SwiGLU rule's factors; ``[M·b, I]`` (or ``[M·b, 2I]``) in ``x``'s dtype, rows
    ``pair·b + copy`` in expert order."""
    b, N, D = x.shape
    M, I = cg.shape
    E = offs.shape[0]
    BM, BN, BK, nw, ns = MOE["jvp" if jvp else "vjp"]
    BN, BK = math.gcd(BN, I), math.gcd(BK, D)
    assert x.stride(-1) == 1 and w.is_contiguous() and cg.is_contiguous() and cu.is_contiguous()
    # tiles of BM rows, each inside one expert's rows
    rend = offs.long() * b
    rstart = torch.cat((rend.new_zeros(1), rend[:-1]))
    nt = (rend - rstart + BM - 1) // BM
    cum = nt.cumsum(0)
    T = -(-M * b // BM) + E
    t = torch.arange(T, device=x.device)
    te = torch.searchsorted(cum, t, right=True)
    ec = te.clamp(max=E - 1)
    tile_r0 = (rstart[ec] + (t - cum[ec] + nt[ec]) * BM).to(torch.int32)
    tile_e = torch.where(te < E, ec, -1).to(torch.int32)
    h = torch.empty(M * b, I if jvp else 2 * I, device=x.device, dtype=x.dtype)
    _moe_up[(T, I // BN)](x, w, tile_e, tile_r0, rend.to(torch.int32), pos, cg, cu, h, x.stride(0), x.stride(1),
                          *w.stride(), b, D=D, I=I, JVP=jvp, BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
    return h


@triton.jit
def _conv_rows(x, wc, sc, i_b, t, ch, N, CH: tl.constexpr, K: tl.constexpr):
    """The causal convolution of rows ``t`` on channels ``ch``, times σ: ``[len(t), len(ch)]`` fp32."""
    acc = tl.zeros([t.shape[0], ch.shape[0]], dtype=tl.float32)
    for j in tl.static_range(K):
        src = t - (K - 1) + j
        xs = tl.load(x + (i_b * N + src[:, None]) * CH + ch[None, :], mask=((src >= 0) & (src < N))[:, None], other=0.)
        acc += xs.to(tl.float32) * tl.load(wc + ch * K + j)[None, :]
    return acc * tl.load(sc + t[:, None] * CH + ch[None, :], mask=(t < N)[:, None], other=0.)


@triton.jit
def _l2(g, x0, r):
    """An L2 norm's derivative (symmetric, so its own transpose) on rows: ``r (g − x0 r² (x0·g))``."""
    return r * (g - x0 * r * r * tl.sum(x0 * g, 1)[:, None])


@triton.jit
def _conv_in(x, wc, sc, q0, rq0, k0, rk0, oq, okk, ov, N, SCALE: tl.constexpr, CH: tl.constexpr, HK: tl.constexpr,
             HV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, B, CB: tl.constexpr):
    """DeltaNet's convolution of the projected tangents ``x`` (``[b, N, CH]``), times σ (identity
    rule), on one head's channels for ``BT`` rows; the queries' and keys' heads then through their
    L2 norm's derivative (the queries scaled)."""
    i_t, i_c, i_bb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = i_t * BT + tl.arange(0, BT)
    okr = (t < N)[:, None]
    for i_b in range(i_bb * CB, tl.minimum(i_bb * CB + CB, B)):  # CB copies: the factors stay in L1
        if i_c < 2 * HK:
            h = i_c % HK
            od = tl.arange(0, DK)
            g = _conv_rows(x, wc, sc, i_b, t, i_c * DK + od, N, CH, K)
            rows = (t[:, None] * HK + h) * DK + od[None, :]
            out = ((i_b * N + t[:, None]) * HK + h) * DK + od[None, :]
            if i_c < HK:
                y = _l2(g, tl.load(q0 + rows, mask=okr, other=0.), tl.load(rq0 + t * HK + h, mask=t < N, other=0.)[:, None])
                tl.store(oq + out, y * SCALE, mask=okr)
            else:
                y = _l2(g, tl.load(k0 + rows, mask=okr, other=0.), tl.load(rk0 + t * HK + h, mask=t < N, other=0.)[:, None])
                tl.store(okk + out, y, mask=okr)
        else:  # (names apart from the branch above: dk and dv tiles may differ in shape)
            hv_ = i_c - 2 * HK
            odv = tl.arange(0, DV)
            gv_ = _conv_rows(x, wc, sc, i_b, t, 2 * HK * DK + hv_ * DV + odv, N, CH, K)
            tl.store(ov + ((i_b * N + t[:, None]) * HV + hv_) * DV + odv[None, :], gv_, mask=okr)


@triton.jit
def _conv_out(gq, gk, gv, wc, sc, q0, rq0, k0, rk0, out, N, SCALE: tl.constexpr, CH: tl.constexpr, HK: tl.constexpr,
              HV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, B, CB: tl.constexpr):
    """``_conv_in``'s transpose, from the queries', keys' (per value head, summed into their key
    head here) and values' cotangents to the projection's, in ``out``'s dtype."""
    i_t, i_c, i_bb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    REP: tl.constexpr = HV // HK
    t = i_t * BT + tl.arange(0, BT)
    for i_b in range(i_bb * CB, tl.minimum(i_bb * CB + CB, B)):  # CB copies: the factors stay in L1
        if i_c < 2 * HK:
            h = i_c % HK
            od = tl.arange(0, DK)
            ch = i_c * DK + od
            acc = tl.zeros([BT, DK], dtype=tl.float32)
            for j in tl.static_range(K):
                src = t + (K - 1) - j
                oks = (src < N)[:, None]
                g = tl.zeros([BT, DK], dtype=tl.float32)
                for rr in tl.static_range(REP):
                    vrows = ((i_b * N + src[:, None]) * HV + h * REP + rr) * DK + od[None, :]
                    if i_c < HK:
                        g += tl.load(gq + vrows, mask=oks, other=0.)
                    else:
                        g += tl.load(gk + vrows, mask=oks, other=0.)
                rows = (src[:, None] * HK + h) * DK + od[None, :]
                if i_c < HK:
                    g = _l2(g, tl.load(q0 + rows, mask=oks, other=0.), tl.load(rq0 + src * HK + h, mask=src < N, other=0.)[:, None]) * SCALE
                else:
                    g = _l2(g, tl.load(k0 + rows, mask=oks, other=0.), tl.load(rk0 + src * HK + h, mask=src < N, other=0.)[:, None])
                g *= tl.load(sc + src[:, None] * CH + ch[None, :], mask=oks, other=0.)
                acc += g * tl.load(wc + ch * K + j)[None, :]
            tl.store(out + (i_b * N + t[:, None]) * CH + ch[None, :], acc.to(out.dtype.element_ty), mask=(t < N)[:, None])
        else:  # (names apart from the branch above: dk and dv tiles may differ in shape)
            hv_ = i_c - 2 * HK
            odv = tl.arange(0, DV)
            chv = 2 * HK * DK + hv_ * DV + odv
            accv = tl.zeros([BT, DV], dtype=tl.float32)
            for j in tl.static_range(K):
                src = t + (K - 1) - j
                oks = (src < N)[:, None]
                gv_ = tl.load(gv + ((i_b * N + src[:, None]) * HV + hv_) * DV + odv[None, :], mask=oks, other=0.)
                gv_ *= tl.load(sc + src[:, None] * CH + chv[None, :], mask=oks, other=0.)
                accv += gv_ * tl.load(wc + chv * K + j)[None, :]
            tl.store(out + (i_b * N + t[:, None]) * CH + chv[None, :], accv.to(out.dtype.element_ty), mask=(t < N)[:, None])


def delta_conv_in(m, x):
    """``layers._delta_jvp``'s convolution and L2 norms on CUDA: the projected tangents ``x``
    (``[b, N, CH]``) to the queries' and keys' (``[b, N, Hk, dk]``, normed, the queries scaled)
    and values' (``[b, N, Hv, dv]``) tangents, fp32."""
    b, N, CH = x.shape
    _, hk, DK = m.q0.shape
    hv, DV = m.u.shape[0], m.u.shape[-1]
    assert CH == 2 * hk * DK + hv * DV and all(t.is_contiguous() for t in (x, m.wc, m.sc, m.q0, m.rq0, m.k0, m.rk0))
    f32 = dict(device=x.device, dtype=torch.float32)
    oq, ok_, ov = torch.empty(b, N, hk, DK, **f32), torch.empty(b, N, hk, DK, **f32), torch.empty(b, N, hv, DV, **f32)
    BT, nw, CB = CONV["in"]
    _conv_in[(triton.cdiv(N, BT), 2 * hk + hv, triton.cdiv(b, CB))](x, m.wc, m.sc, m.q0, m.rq0, m.k0, m.rk0, oq, ok_, ov, N, SCALE=DK ** -0.5,
                                                   CH=CH, HK=hk, HV=hv, DK=DK, DV=DV, K=m.wc.shape[-1], BT=BT, B=b, CB=CB, num_warps=nw)
    return oq, ok_, ov


def delta_conv_out(m, gq, gk, gv, dtype):
    """``_conv_in``'s transpose on CUDA: the queries' and keys' cotangents (``[b, N, Hv, dk]``, per
    value head) and values' (``[b, N, Hv, dv]``) to the projection's, ``[b, N, CH]`` in ``dtype``."""
    b, N, hv, DV = gv.shape
    _, hk, DK = m.q0.shape
    CH = 2 * hk * DK + hv * DV
    assert all(t.is_contiguous() for t in (gq, gk, gv, m.wc, m.sc, m.q0, m.rq0, m.k0, m.rk0))
    out = torch.empty(b, N, CH, device=gv.device, dtype=dtype)
    BT, nw, CB = CONV["out"]
    _conv_out[(triton.cdiv(N, BT), 2 * hk + hv, triton.cdiv(b, CB))](gq, gk, gv, m.wc, m.sc, m.q0, m.rq0, m.k0, m.rk0, out, N, SCALE=DK ** -0.5,
                                                    CH=CH, HK=hk, HV=hv, DK=DK, DV=DV, K=m.wc.shape[-1], BT=BT, B=b, CB=CB, num_warps=nw)
    return out


def _rows(x):
    """``x`` with unit stride along features (the kernels' loads assume it)."""
    return x if x.stride(-1) == 1 else x.contiguous()


def _shapes(m):
    assert all(x.is_contiguous() for x in (m.q, m.k, m.u, m.vmc, m.beta, m.gc, m.hku, m.tinv, m.qkd, m.kkd, m.qh, m.kh))
    hv, tp, dk = m.q.shape
    C = m.tinv.shape[-1]
    return hv, tp // C, C, dk, m.u.shape[-1]


def _config(name, DK, DV):
    BK, BV, nw, ns = CONFIG[name]
    BK, BV = min(BK, DK), min(BV, DV)
    assert DK % BK == 0 and DV % BV == 0 and DK // BK <= 4, (DK, DV, BK, BV)  # states in at most 4 key blocks
    return BK, BV, dict(num_warps=nw, num_stages=ns)


def delta_jvp(m, dq, dk, dv, db, dg):
    """``layers._delta_core_jvp`` on CUDA."""
    dq, dk, dv, db, dg = map(_rows, (dq, dk, dv, db, dg))
    b, N, hv, DV = dv.shape
    _, NT, C, DK, _ = _shapes(m)
    BK, BV, launch = _config("jvp", DK, DV)
    out = torch.empty(b, N, hv, DV, device=dv.device, dtype=torch.float32)
    _jvp[(DV // BV, b * hv)](
        m.q, m.k, m.u, m.vmc, m.beta, m.gc, m.hku, m.tinv, m.qkd, m.kkd, m.qh, m.kh, dq, dk, dv, db, dg, out,
        *dq.stride()[:3], *dk.stride()[:3], *dv.stride()[:3], *db.stride()[:2], *dg.stride()[:2], *out.stride()[:3],
        m.s0, N, HV=hv, REP=hv // dq.shape[2], NT=NT, C=C, DK=DK, DV=DV, BK=BK, BV=BV, **launch)
    return out


def delta_vjp(m, go):
    """``layers._delta_core_vjp`` on CUDA."""
    go = _rows(go)
    b, N, hv, DV = go.shape
    _, NT, C, DK, _ = _shapes(m)
    f32 = dict(device=go.device, dtype=torch.float32)
    BK, BV, launch = _config("vjp_r", DK, DV)
    assert DV // BV <= 2  # one value block stores to gks; two add onto zeros, either first: deterministic
    gdv, gbp = torch.empty(b, N, hv, DV, **f32), torch.empty(DV // BV, b, N, hv, **f32)
    gks = (torch.empty if DV == BV else torch.zeros)(b, N, hv, DK, **f32)
    _vjp_r[(DV // BV, b * hv)](
        m.q, m.k, m.u, m.vmc, m.beta, m.gc, m.tinv, m.qkd, go, gdv, gbp, gks,
        *go.stride()[:3], *gdv.stride()[:3], *gbp.stride()[:3], *gks.stride()[:3],
        m.s0, N, HV=hv, NT=NT, C=C, DK=DK, DV=DV, BK=BK, BV=BV, **launch)
    BK, BV, launch = _config("vjp_k", DK, DV)
    gq, gk, gcp = torch.empty(b, N, hv, DK, **f32), torch.empty(b, N, hv, DK, **f32), torch.empty(DK // BK, b, N, hv, **f32)
    CB = VJPK_COPIES
    _vjp_k[(DK // BK, NT, triton.cdiv(b, CB) * hv)](
        m.q, m.k, m.u, m.gc, m.hku, go, gdv, gks, gq, gk, gcp,
        *go.stride()[:3], *gdv.stride()[:3], *gks.stride()[:3], *gq.stride()[:3], *gcp.stride()[:3],
        m.s0, N, HV=hv, NT=NT, C=C, DK=DK, DV=DV, BK=BK, BV=BV, B=b, CB=CB, **launch)
    return gq, gk, gdv, gbp.sum(0), gcp.sum(0)
