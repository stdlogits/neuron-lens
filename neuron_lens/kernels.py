"""Triton kernels for Gated DeltaNet's derivative on CUDA. ``delta_jvp`` and ``delta_vjp`` run
``layers._delta_core_jvp`` and ``layers._delta_core_vjp`` (the reference, and the path off CUDA).
The map walks the chunks forward, one program per tangent copy, value head and block of value
features, its state in registers; the transpose walks them back for ``gr`` (``_vjp_r``), then
takes the queries' and keys' cotangents chunk by chunk in parallel (``_vjp_k``). The document's
constants are shared by every tangent copy, so they mostly come from L2. Products take TF32, as
``02_fit.py`` sets for torch's.

Tangents arrive in the layout the projections give, ``[b, N, H, ·]`` at valid rows, and leave the
same way; a chunk's rows before the first valid one or past the last load as zeros."""
import torch
import triton
import triton.language as tl

# per kernel: key and value feature blocks, num_warps, num_stages (tuned on an H100)
CONFIG = {"jvp": (64, 64, 4, 1), "vjp_r": (64, 64, 4, 1), "vjp_k": (64, 64, 4, 1)}


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
            rhs += tl.dot(kk, ab)
            tl.atomic_add(gks + i_b * ssb + r[:, None] * ssn + i_h * ssh + ck[None, :], tl.dot(uu, tl.trans(ab)),
                          mask=ok[:, None])
        x = tl.where(ok[:, None], tl.dot(tl.trans(tl.load(tinv + sq)), rhs), 0.)
        bt = tl.load(beta + i_h * TP + t)
        tl.store(gdv + i_b * sdb + r[:, None] * sdn + i_h * sdh + ov[None, :], bt[:, None] * x, mask=ok[:, None])
        tl.store(gbp + i_v * spv + i_b * spb + r * spn + i_h, tl.sum(x * tl.load(vmc + fv), 1), mask=ok)
        xb = bt[:, None] * x
        for kb in tl.static_range(NK):
            ck = kb * BK + obk
            qq = tl.load(q + (i_h * TP + t[:, None]) * DK + ck[None, :]) * qd[:, None]
            kk = tl.load(k + (i_h * TP + t[:, None]) * DK + ck[None, :]) * qd[:, None]
            upd = tl.dot(tl.trans(qq), goc) - tl.dot(tl.trans(kk), xb)
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
           DK: tl.constexpr, DV: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    """One chunk's cotangents of the queries (``gq``, through W) and keys (through C as readers,
    ``gkh``; as writers, ``gkt``, with ``_vjp_r``'s state term) on a block of key features, per
    value head, and this block's part of the cumulative decay's."""
    i_k, c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // HV, i_bh % HV
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
        p += tl.dot(goc, tl.trans(uu))
        pb += tl.dot(gbl, tl.trans(uu))
        gqc += tl.dot(goc * qd[:, None], hs)
        gkh += tl.dot(gbl * qd[:, None], hs)
    p = p * dec
    pb = tl.where(oc[:, None] > oc[None, :], pb * dec, 0.)
    qq = tl.load(q + (i_h * TP + t[:, None]) * DK + ck[None, :])
    kk = tl.load(k + (i_h * TP + t[:, None]) * DK + ck[None, :])
    gqc += tl.dot(p, kk)
    gkh += tl.dot(pb, kk)
    gkt = tl.dot(tl.trans(p), qq) + tl.dot(tl.trans(pb), kk) \
        + tl.load(gks + i_b * ssb + r[:, None] * ssn + i_h * ssh + ck[None, :], mask=ok[:, None], other=0.)
    rows = i_b * sqb + r[:, None] * sqn + i_h * sqh + ck[None, :]
    tl.store(gq + rows, gqc, mask=ok[:, None])
    tl.store(gk + rows, gkh + gkt, mask=ok[:, None])
    tl.store(gcp + i_k * spk + i_b * spb + r * spn + i_h, tl.sum(gqc * qq + (gkh - gkt) * kk, 1), mask=ok)


def _rows(x):
    """``x`` with unit stride along features (the kernels' loads assume it)."""
    return x if x.stride(-1) == 1 else x.contiguous()


def _shapes(m):
    assert all(x.is_contiguous() for x in (m.q, m.k, m.u, m.vmc, m.beta, m.gc, m.hku, m.tinv, m.qkd, m.kkd, m.qh, m.kh))
    hv, tp, dk = m.q.shape
    C = m.tinv.shape[-1]
    return hv, tp // C, C, dk, m.u.shape[-1]


def delta_jvp(m, dq, dk, dv, db, dg):
    """``layers._delta_core_jvp`` on CUDA."""
    dq, dk, dv, db, dg = map(_rows, (dq, dk, dv, db, dg))
    b, N, hv, DV = dv.shape
    _, NT, C, DK, _ = _shapes(m)
    BK, BV, nw, ns = CONFIG["jvp"]
    out = torch.empty(b, N, hv, DV, device=dv.device, dtype=torch.float32)
    _jvp[(DV // BV, b * hv)](
        m.q, m.k, m.u, m.vmc, m.beta, m.gc, m.hku, m.tinv, m.qkd, m.kkd, m.qh, m.kh, dq, dk, dv, db, dg, out,
        *dq.stride()[:3], *dk.stride()[:3], *dv.stride()[:3], *db.stride()[:2], *dg.stride()[:2], *out.stride()[:3],
        m.s0, N, HV=hv, REP=hv // dq.shape[2], NT=NT, C=C, DK=DK, DV=DV, BK=BK, BV=BV, num_warps=nw, num_stages=ns)
    return out


def delta_vjp(m, go, hk: int):
    """``layers._delta_core_vjp`` on CUDA."""
    go = _rows(go)
    b, N, hv, DV = go.shape
    _, NT, C, DK, _ = _shapes(m)
    f32 = dict(device=go.device, dtype=torch.float32)
    BK, BV, nw, ns = CONFIG["vjp_r"]
    assert DV // BV <= 2  # two adds onto zero to gks, whichever comes first: deterministic
    gdv, gbp, gks = torch.empty(b, N, hv, DV, **f32), torch.empty(DV // BV, b, N, hv, **f32), torch.zeros(b, N, hv, DK, **f32)
    _vjp_r[(DV // BV, b * hv)](
        m.q, m.k, m.u, m.vmc, m.beta, m.gc, m.tinv, m.qkd, go, gdv, gbp, gks,
        *go.stride()[:3], *gdv.stride()[:3], *gbp.stride()[:3], *gks.stride()[:3],
        m.s0, N, HV=hv, NT=NT, C=C, DK=DK, DV=DV, BK=BK, BV=BV, num_warps=nw, num_stages=ns)
    BK, BV, nw, ns = CONFIG["vjp_k"]
    gq, gk, gcp = torch.empty(b, N, hv, DK, **f32), torch.empty(b, N, hv, DK, **f32), torch.empty(DK // BK, b, N, hv, **f32)
    _vjp_k[(DK // BK, NT, b * hv)](
        m.q, m.k, m.u, m.gc, m.hku, go, gdv, gks, gq, gk, gcp,
        *go.stride()[:3], *gdv.stride()[:3], *gks.stride()[:3], *gq.stride()[:3], *gcp.stride()[:3],
        m.s0, N, HV=hv, NT=NT, C=C, DK=DK, DV=DV, BK=BK, BV=BV, num_warps=nw, num_stages=ns)
    unheads = lambda x: x.view(b, N, hk, hv // hk, DK).sum(3)
    return unheads(gq), unheads(gk), gdv, gbp.sum(0), gcp.sum(0)
