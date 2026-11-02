"""Triton kernels for Gated DeltaNet's derivative on CUDA. ``delta_jvp`` and ``delta_vjp`` run
``layers._delta_core_jvp`` and ``layers._delta_core_vjp`` (the reference, and the path off CUDA):
one program per tangent copy, value head and block of features walks the chunks, its state in
registers, reading the document's constants (shared by every tangent copy, so mostly from L2) and
each tangent once. Products take TF32, as ``02_fit.py`` sets for torch's.

Tangents arrive in the layout the projections give, ``[b, N, H, ·]`` at valid rows, and leave the
same way; a chunk's rows before the first valid one or past the last load as zeros."""
import torch
import triton
import triton.language as tl

# launch configurations (num_warps, num_stages) per kernel, tuned on an H100
CONFIG = {"jvp": (4, 1), "vjp_r": (4, 1), "vjp_k": (4, 1)}


@triton.jit
def _jvp(q, k, u, vmc, beta, gc, hku, tinv, dq, dk, dv, db, dg, out,
         sqb, sqn, sqh, skb, skn, skh, svb, svn, svh, sbb, sbn, sgb, sgn, sob, son, soh,
         S, N, HV: tl.constexpr, REP: tl.constexpr, NT: tl.constexpr, C: tl.constexpr,
         DK: tl.constexpr, DV: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // HV, i_bh % HV
    i_hk = i_h // REP
    TP: tl.constexpr = NT * C
    oc = tl.arange(0, C)
    obk = tl.arange(0, BK)
    ov = i_v * BV + tl.arange(0, BV)
    lower = oc[:, None] >= oc[None, :]
    strict = oc[:, None] > oc[None, :]
    h0 = tl.zeros([BK, BV], dtype=tl.float32)  # the state, Σ Γ (k̃ uᵀ + k xᵀ), by halves of dk
    h1 = tl.zeros([BK, BV], dtype=tl.float32)
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
        qk = tl.zeros([C, C], dtype=tl.float32)
        dcu = tl.zeros([C, BV], dtype=tl.float32)
        o = tl.zeros([C, BV], dtype=tl.float32)
        for kb in tl.static_range(DK // BK):
            ck = kb * BK + obk
            qq = tl.load(q + (i_h * TP + t[:, None]) * DK + ck[None, :])
            kk = tl.load(k + (i_h * TP + t[:, None]) * DK + ck[None, :])
            dqq = tl.load(dq + i_b * sqb + r[:, None] * sqn + i_hk * sqh + ck[None, :], mask=ok[:, None], other=0.)
            dkk = tl.load(dk + i_b * skb + r[:, None] * skn + i_hk * skh + ck[None, :], mask=ok[:, None], other=0.)
            qt = dqq + dgv[:, None] * qq
            kh = dkk + dgv[:, None] * kk
            kt = dkk - dgv[:, None] * kk
            a0 += tl.dot(qt, tl.trans(kk)) + tl.dot(qq, tl.trans(kt))
            a1 += tl.dot(kh, tl.trans(kk)) + tl.dot(kk, tl.trans(kt))
            qk += tl.dot(qq, tl.trans(kk))
            hs = tl.load(hku + ((i_h * NT + c) * DK + ck[:, None]) * DV + ov[None, :])
            dcu += tl.dot(kh * qd[:, None], hs)
            o += tl.dot(qt * qd[:, None], hs)
            if kb == 0:
                dcu += tl.dot(kk * qd[:, None], h0)
                o += tl.dot(qq * qd[:, None], h0)
            else:
                dcu += tl.dot(kk * qd[:, None], h1)
                o += tl.dot(qq * qd[:, None], h1)
        uu = tl.load(u + (i_h * TP + t[:, None]) * DV + ov[None, :])
        dcu += tl.dot(tl.where(strict, a1 * dec, 0.), uu)
        vm = tl.load(vmc + (i_h * TP + t[:, None]) * DV + ov[None, :])
        bt = tl.load(beta + i_h * TP + t)
        dvv = tl.load(dv + i_b * svb + r[:, None] * svn + i_h * svh + ov[None, :], mask=ok[:, None], other=0.)
        tt = tl.load(tinv + ((i_h * NT + c) * C + oc[:, None]) * C + oc[None, :])
        x = tl.dot(tt, dbv[:, None] * vm + bt[:, None] * (dvv - dcu))  # L⁻¹ r on this chunk
        o += tl.dot(a0 * dec, uu) + tl.dot(qk * dec, x)
        tl.store(out + i_b * sob + r[:, None] * son + i_h * soh + ov[None, :], o, mask=ok[:, None])
        for kb in tl.static_range(DK // BK):
            ck = kb * BK + obk
            kk = tl.load(k + (i_h * TP + t[:, None]) * DK + ck[None, :]) * kd[:, None]
            dkk = tl.load(dk + i_b * skb + r[:, None] * skn + i_hk * skh + ck[None, :], mask=ok[:, None], other=0.)
            upd = tl.dot(tl.trans(dkk * kd[:, None] - dgv[:, None] * kk), uu) + tl.dot(tl.trans(kk), x)
            if kb == 0:
                h0 = h0 * cd + upd
            else:
                h1 = h1 * cd + upd


@triton.jit
def _vjp_r(q, k, vmc, beta, gc, tinv, go, gdv, gbp,
           sgb, sgn, sgh, sdb, sdn, sdh, spv, spb, spn,
           S, N, HV: tl.constexpr, NT: tl.constexpr, C: tl.constexpr,
           DK: tl.constexpr, DV: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    """``gr = L⁻ᵀ Wᵀ go``, back over the chunks; stores ``β gr`` (the values' cotangent) and this
    block's part of β's, ``gr · (V − C U)``."""
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // HV, i_bh % HV
    TP: tl.constexpr = NT * C
    oc = tl.arange(0, C)
    obk = tl.arange(0, BK)
    ov = i_v * BV + tl.arange(0, BV)
    lower = oc[:, None] >= oc[None, :]
    a0 = tl.zeros([BK, BV], dtype=tl.float32)  # Σ_{later} Γ (q goᵀ + k gblᵀ), by halves of dk
    a1 = tl.zeros([BK, BV], dtype=tl.float32)
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
        dec = tl.exp(tl.where(lower, g[:, None] - g[None, :], float("-inf")))
        goc = tl.load(go + i_b * sgb + r[:, None] * sgn + i_h * sgh + ov[None, :], mask=ok[:, None], other=0.)
        qk = tl.zeros([C, C], dtype=tl.float32)
        rhs = tl.zeros([C, BV], dtype=tl.float32)
        for kb in tl.static_range(DK // BK):
            ck = kb * BK + obk
            qq = tl.load(q + (i_h * TP + t[:, None]) * DK + ck[None, :])
            kk = tl.load(k + (i_h * TP + t[:, None]) * DK + ck[None, :])
            qk += tl.dot(qq, tl.trans(kk))
            if kb == 0:
                rhs += tl.dot(kk * kd[:, None], a0)
            else:
                rhs += tl.dot(kk * kd[:, None], a1)
        rhs += tl.dot(tl.trans(qk * dec), goc)
        tt = tl.load(tinv + ((i_h * NT + c) * C + oc[:, None]) * C + oc[None, :])
        gr = tl.where(ok[:, None], tl.dot(tl.trans(tt), rhs), 0.)
        bt = tl.load(beta + i_h * TP + t)
        vm = tl.load(vmc + (i_h * TP + t[:, None]) * DV + ov[None, :])
        tl.store(gdv + i_b * sdb + r[:, None] * sdn + i_h * sdh + ov[None, :], bt[:, None] * gr, mask=ok[:, None])
        tl.store(gbp + i_v * spv + i_b * spb + r * spn + i_h, tl.sum(gr * vm, 1), mask=ok)
        gbl = -bt[:, None] * gr
        for kb in tl.static_range(DK // BK):
            ck = kb * BK + obk
            qq = tl.load(q + (i_h * TP + t[:, None]) * DK + ck[None, :]) * qd[:, None]
            kk = tl.load(k + (i_h * TP + t[:, None]) * DK + ck[None, :]) * qd[:, None]
            upd = tl.dot(tl.trans(qq), goc) + tl.dot(tl.trans(kk), gbl)
            if kb == 0:
                a0 = a0 * cd + upd
            else:
                a1 = a1 * cd + upd


@triton.jit
def _vjp_k(q, k, u, gc, hku, go, gdv, gq, gk, gcp,
           sgb, sgn, sgh, sdb, sdn, sdh, sqb, sqn, sqh, spk, spb, spn,
           S, N, HV: tl.constexpr, NT: tl.constexpr, C: tl.constexpr,
           DK: tl.constexpr, DV: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    """The cotangents of ``q̃``, ``k̂`` and ``k̃`` on one block of key features, back over the
    chunks with ``gbl = −β gr``: the queries' and keys' (``[b, N, Hv, dk]``, per value head) and
    this block's part of the cumulative decay's."""
    i_k, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // HV, i_bh % HV
    TP: tl.constexpr = NT * C
    oc = tl.arange(0, C)
    obv = tl.arange(0, BV)
    ck = i_k * BK + tl.arange(0, BK)
    lower = oc[:, None] >= oc[None, :]
    strict = oc[:, None] > oc[None, :]
    b0 = tl.zeros([BV, BK], dtype=tl.float32)  # Σ_{later} Γ (go qᵀ + gbl kᵀ), by halves of dv
    b1 = tl.zeros([BV, BK], dtype=tl.float32)
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
        dec = tl.exp(tl.where(lower, g[:, None] - g[None, :], float("-inf")))
        qq = tl.load(q + (i_h * TP + t[:, None]) * DK + ck[None, :])
        kk = tl.load(k + (i_h * TP + t[:, None]) * DK + ck[None, :])
        p = tl.zeros([C, C], dtype=tl.float32)
        pb = tl.zeros([C, C], dtype=tl.float32)
        gqc = tl.zeros([C, BK], dtype=tl.float32)
        gkh = tl.zeros([C, BK], dtype=tl.float32)
        gkt = tl.zeros([C, BK], dtype=tl.float32)
        for vb in tl.static_range(DV // BV):
            cv = vb * BV + obv
            goc = tl.load(go + i_b * sgb + r[:, None] * sgn + i_h * sgh + cv[None, :], mask=ok[:, None], other=0.)
            gbl = -tl.load(gdv + i_b * sdb + r[:, None] * sdn + i_h * sdh + cv[None, :], mask=ok[:, None], other=0.)
            uu = tl.load(u + (i_h * TP + t[:, None]) * DV + cv[None, :])
            hs = tl.load(hku + ((i_h * NT + c) * DK + ck[None, :]) * DV + cv[:, None])  # [BV, BK]
            p += tl.dot(goc, tl.trans(uu))
            pb += tl.dot(gbl, tl.trans(uu))
            gqc += tl.dot(goc * qd[:, None], hs)
            gkh += tl.dot(gbl * qd[:, None], hs)
            if vb == 0:
                gkt += tl.dot(uu * kd[:, None], b0)
            else:
                gkt += tl.dot(uu * kd[:, None], b1)
        p = p * dec
        pb = tl.where(strict, pb * dec, 0.)
        gqc += tl.dot(p, kk)
        gkh += tl.dot(pb, kk)
        gkt += tl.dot(tl.trans(p), qq) + tl.dot(tl.trans(pb), kk)
        rows = i_b * sqb + r[:, None] * sqn + i_h * sqh + ck[None, :]
        tl.store(gq + rows, gqc, mask=ok[:, None])
        tl.store(gk + rows, gkh + gkt, mask=ok[:, None])
        tl.store(gcp + i_k * spk + i_b * spb + r * spn + i_h, tl.sum(gqc * qq + (gkh - gkt) * kk, 1), mask=ok)
        for vb in tl.static_range(DV // BV):
            cv = vb * BV + obv
            goc = tl.load(go + i_b * sgb + r[:, None] * sgn + i_h * sgh + cv[None, :], mask=ok[:, None], other=0.)
            gbl = -tl.load(gdv + i_b * sdb + r[:, None] * sdn + i_h * sdh + cv[None, :], mask=ok[:, None], other=0.)
            upd = tl.dot(tl.trans(goc), qq * qd[:, None]) + tl.dot(tl.trans(gbl), kk * qd[:, None])
            if vb == 0:
                b0 = b0 * cd + upd
            else:
                b1 = b1 * cd + upd


def _rows(x):
    """``x`` with unit stride along features (the kernels' loads assume it)."""
    return x if x.stride(-1) == 1 else x.contiguous()


def _shapes(m):
    hv, tp, dk = m.q.shape
    C = m.tinv.shape[-1]
    return hv, tp // C, C, dk, m.u.shape[-1]


def delta_jvp(m, dq, dk, dv, db, dg):
    """``layers._delta_core_jvp`` on CUDA."""
    dq, dk, dv, db, dg = map(_rows, (dq, dk, dv, db, dg))
    b, N, hv, DV = dv.shape
    _, NT, C, DK, _ = _shapes(m)
    BK, BV = min(64, DK), min(64, DV)
    out = torch.empty(b, N, hv, DV, device=dv.device, dtype=torch.float32)
    _jvp[(DV // BV, b * hv)](
        m.q, m.k, m.u, m.vmc, m.beta, m.gc, m.hku, m.tinv, dq, dk, dv, db, dg, out,
        *dq.stride()[:3], *dk.stride()[:3], *dv.stride()[:3], *db.stride()[:2], *dg.stride()[:2], *out.stride()[:3],
        m.s0, N, HV=hv, REP=hv // dq.shape[2], NT=NT, C=C, DK=DK, DV=DV, BK=BK, BV=BV,
        num_warps=CONFIG["jvp"][0], num_stages=CONFIG["jvp"][1])
    return out


def delta_vjp(m, go, hk: int):
    """``layers._delta_core_vjp`` on CUDA."""
    go = _rows(go)
    b, N, hv, DV = go.shape
    _, NT, C, DK, _ = _shapes(m)
    BK, BV = min(64, DK), min(64, DV)
    f32 = dict(device=go.device, dtype=torch.float32)
    gdv, gbp = torch.empty(b, N, hv, DV, **f32), torch.empty(DV // BV, b, N, hv, **f32)
    _vjp_r[(DV // BV, b * hv)](
        m.q, m.k, m.vmc, m.beta, m.gc, m.tinv, go, gdv, gbp,
        *go.stride()[:3], *gdv.stride()[:3], *gbp.stride()[:3],
        m.s0, N, HV=hv, NT=NT, C=C, DK=DK, DV=DV, BK=BK, BV=BV,
        num_warps=CONFIG["vjp_r"][0], num_stages=CONFIG["vjp_r"][1])
    gq, gk, gcp = torch.empty(b, N, hv, DK, **f32), torch.empty(b, N, hv, DK, **f32), torch.empty(DK // BK, b, N, hv, **f32)
    _vjp_k[(DK // BK, b * hv)](
        m.q, m.k, m.u, m.gc, m.hku, go, gdv, gq, gk, gcp,
        *go.stride()[:3], *gdv.stride()[:3], *gq.stride()[:3], *gcp.stride()[:3],
        m.s0, N, HV=hv, NT=NT, C=C, DK=DK, DV=DV, BK=BK, BV=BV,
        num_warps=CONFIG["vjp_k"][0], num_stages=CONFIG["vjp_k"][1])
    unheads = lambda x: x.view(b, N, hk, hv // hk, DK).sum(3)
    return unheads(gq), unheads(gk), gdv, gbp.sum(0), gcp.sum(0)
