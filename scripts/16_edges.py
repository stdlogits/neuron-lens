"""Direct weight edges between neuron poles and attention heads (pipeline stage 16; findings §12).

The residual stream is a sum: what a neuron or head writes reaches every later reader's input, so the direct
path between any two units is determined by their weights. Two tables are written:

- ``neurons/edges/L<layer>.parquet`` (a row per pole):
  - ``in_*``, ``out_*``: the ``TOP`` earlier/later poles (``_layer``, ``_neuron``, ``_pole``, ``_push``, ``_cos``,
    signed);
  - ``in_head_*``: the ``TOP`` earlier/same-layer heads pushing this pole (``attn -> MLP``; same layer in sequential
    blocks, earlier only in parallel blocks);
  - ``{v,q,k}_out_head_*``: the ``TOP`` later heads this pole pushes per receiving channel (``MLP -> attn``).
- ``heads/edges.parquet`` (a row per query head, via ``head_keys`` / ``write_heads``):
  - ``{v,q,k}_in_*``: the ``TOP`` earlier poles pushing this head's value, query or key channel (``MLP -> attn``);
  - ``out_*``: the ``TOP`` same/later poles this head pushes (``attn -> MLP``);
  - ``{v,q,k}_in_head_*``, ``{v,q,k}_out_head_*``: the ``TOP`` earlier/later heads per receiving channel
    (``attn -> attn``).

Every edge uses the residual's metric at the receiving site (``mid`` for MLPs, ``in`` for heads; ``neuron_lens.metric``,
findings §15). Pole→pole edges keep their signed cosine and ``|cos| ≥ SIGMAS/√d`` floor. Edges involving heads use
gauge-invariant, GQA-distinct effective operators (``neuron_lens.heads.operators``), the unsigned Frobenius
composition score ``cos = ‖B A‖_F / (‖B‖_F ‖A‖_F)``, a Haar-rotation tail-matched floor on ``cos²`` matching the
``SIGMAS`` upper-tail probability for each pair's effective ranks, and modeled RMS ``push = cos · a_t · gain / rms_t``.
Per band into ``data/summary/edges_<model>.{log,json}``.

    python scripts/16_edges.py [--device mps]
"""
import argparse
import json
import math
import time

import pyarrow.parquet as pq
import torch

from neuron_lens import anatomy, arch, heads as head_ops, metric
from neuron_lens.catalog import (
    MODEL,
    TOP,
    TYPICAL,
    acts_path,
    documents,
    fixed,
    gridded,
    head_keys,
    head_table,
    lobe_signs,
    neuron_dtype,
    neuron_keys,
    neuron_table,
    write,
    write_heads,
)
from neuron_lens.neurons import directions, poles as layer_poles
from neuron_lens.paths import SUMMARY

STEM = arch.stem(MODEL)
SIGNS = arch.signs(MODEL)
P = len(SIGNS)  # a neuron's poles
CHANNELS = ("v", "q", "k")
SIGMAS = 5.0
RMS_DOCS = 64
TILE = 2**31  # entries of one [B poles, A poles] block of cosines (8 GB fp32): a dense model's layer pairs are one block
STORE = 2**34  # bytes of lists and writes kept on the device; past it (a sparse model's) they wait on the CPU


@torch.inference_mode()
def residual_stats(dev) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(rms_mid, rms_in, mlp_scale, scalars)``, each ``[L]``: the median RMS of the residual entering each layer's
    MLP (``mid``) and mixer (``in``), the median scale of the norm after its MLP (1 where none follows), over
    ``RMS_DOCS`` documents spread through the catalog's (positions from 1), and each layer's scalar."""
    lm, stack = arch.load_model(MODEL, dev, eager=False)
    seen_mid = [[] for _ in stack.layers]
    seen_in = [[] for _ in stack.layers]
    scales = [[] for _ in stack.layers]
    valid = [None]  # [B, T - 1] bool mask of real positions from 1

    def keep_scale(mod, a, o, i):
        s = arch.mlp_scale(stack.layers[i], o)
        if s is not None:
            scales[i].append(s[:, 1:][valid[0]].cpu())
    hooks = []
    for i, lay in enumerate(stack.layers):
        n1 = anatomy.module(lay, anatomy.parts(lay).n1)
        if n1 is not None:
            hooks.append(n1.register_forward_pre_hook(
                lambda mod, a, i=i: seen_in[i].append(
                    arch.centre(a[0][:, 1:][valid[0]].float(), MODEL).pow(2).mean(-1).sqrt().cpu())))
        else:
            hooks.append(lay.register_forward_pre_hook(
                lambda mod, a, kw, i=i: seen_in[i].append(
                    arch.centre((a[0] if a else kw["hidden_states"])[:, 1:][valid[0]].float(), MODEL).pow(2).mean(-1).sqrt().cpu()),
                with_kwargs=True))
        hooks.append(arch.mid_point(lay).register_forward_pre_hook(
            lambda mod, a, i=i: seen_mid[i].append(
                arch.centre(a[0][:, 1:][valid[0]].float(), MODEL).pow(2).mean(-1).sqrt().cpu())))
        hooks.append(lay.mlp.register_forward_hook(lambda mod, a, o, i=i: keep_scale(mod, a, o, i)))
    docs = documents()
    picked = docs[::max(1, len(docs) // RMS_DOCS)][:RMS_DOCS]
    bs = 16
    for s0 in range(0, len(picked), bs):
        batch = picked[s0:s0 + bs]
        T = max(len(d) for d in batch)
        ids = torch.zeros(len(batch), T, dtype=torch.long)
        mask = torch.zeros(len(batch), T, dtype=torch.long)
        for j, d in enumerate(batch):
            ids[j, :len(d)], mask[j, :len(d)] = d.long(), 1
        ids, mask = ids.to(dev), mask.to(dev)
        valid[0] = mask[:, 1:].bool()
        stack(input_ids=ids, attention_mask=mask)
    for h in hooks:
        h.remove()
    scalars = torch.tensor([arch.layer_scale(lay) for lay in stack.layers])
    del lm, stack
    return (torch.stack([torch.cat(s).median() for s in seen_mid]),
            torch.stack([torch.cat(s).median() for s in seen_in]),
            torch.stack([torch.cat(s).median() if s else torch.tensor(1.0) for s in scales]), scalars)


def cos_floor(p2: torch.Tensor | float, q2: torch.Tensor | float, d: int, sigmas: float = SIGMAS) -> torch.Tensor:
    """Tail-matched significance floor on ``cos = ‖B A‖_F / (‖B‖_F ‖A‖_F)`` under Haar rotations of ``ℝ^d``,
    matching ``P(|Z| ≥ sigmas)`` for the exact null mean ``1/d`` and variance
    ``2 (p2 − 1/d)(q2 − 1/d) / ((d − 1)(d + 2))`` via moment-matched scaled χ²_ν (exact ``sigmas/√d`` at ν = 1)."""
    dev = p2.device if isinstance(p2, torch.Tensor) else (q2.device if isinstance(q2, torch.Tensor) else None)
    p2 = torch.as_tensor(p2, dtype=torch.float64, device=dev)
    q2 = torch.as_tensor(q2, dtype=torch.float64, device=dev)
    p2, q2 = torch.broadcast_tensors(p2, q2)
    mu = 1.0 / d
    var = (2.0 * (p2 - mu).clamp_min(1e-12) * (q2 - mu).clamp_min(1e-12)) / ((d - 1.0) * (d + 2.0))
    nu = (2.0 * mu * mu / var).clamp_min(1.0)
    alpha = math.erfc(sigmas / math.sqrt(2.0))
    log_alpha = math.log(alpha)
    z = math.sqrt(2.0) * torch.special.erfinv(torch.tensor(1.0 - 2.0 * alpha, dtype=torch.float64, device=nu.device))
    a = 0.5 * nu
    x = torch.where(nu <= 1.001, torch.full_like(nu, 0.5 * sigmas * sigmas),
                    a * (1.0 - 2.0 / (9.0 * nu) + z * (2.0 / (9.0 * nu)).sqrt()).pow(3))
    lg = torch.lgamma(a)
    for _ in range(5):
        sf = torch.special.gammaincc(a, x).clamp_min(1e-300)
        log_pdf = (a - 1.0) * x.clamp_min(1e-30).log() - x - lg
        step = (sf.log() - log_alpha) * sf / log_pdf.exp().clamp_min(1e-300)
        x = (x + step).clamp_min(1e-12)
    return (2.0 * x / (nu * d)).sqrt().float()


def poles(layer: int, dev) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A layer's writes ``[d, f]``, each pole's ``act_top`` (``[P·f]``, ``+`` poles first), its read covectors at unit
    length in the metric at its MLP input ``[P·f, d]`` and that metric ``[d]``."""
    p = layer_poles(MODEL, layer, lobe_signs(layer), dev)
    act = torch.from_numpy(pq.read_table(neuron_table("acts", layer), columns=["act_top"]).column(0).to_numpy().copy()).float()
    read = p["cov"] / metric.cov_norm(p["cov"], p["s2"])
    return p["write"], act.to(dev), read, p["s2"]


def merge(best: tuple, new: tuple, k: int) -> tuple:
    """Keep the ``k`` highest scores of two ``(score, id, push, cos)`` lists, row by row."""
    cat = [torch.cat([a, b], 1) for a, b in zip(best, new)]
    top = cat[0].topk(k, dim=1).indices
    return tuple(c.gather(1, top) for c in cat)


def _empty(n: int, k: int, device) -> tuple:
    return (torch.full((n, k), -1.0, device=device), torch.full((n, k), -1, device=device),
            torch.full((n, k), float("nan"), device=device), torch.full((n, k), float("nan"), device=device))


def compute(k: int, dev, stats: tuple | None = None) -> tuple[list, list, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every layer's pole→pole in- and out-lists, ``(score, id, push, cos)`` each ``[P·f, k]``; ids are
    ``layer · P·f + row`` (rows in table order), scores ``|push|`` or −1 below the floor. And the null,
    ``[L, 2, P·f]``: each pole's largest |cos| with the layer before's writes, and with them rotated at random."""
    if str(dev).startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    rms, _rms_in, mlp_scale, scalars = residual_stats(dev) if stats is None else stats
    L = len(rms)
    first = poles(0, dev)
    d, f = first[0].shape
    store = "cpu" if L * P * f * (k * 20 + d * 2) > STORE else dev
    to = lambda x, where: tuple(t.to(where) for t in x)
    units = {0: to(first[:2], store)}
    del first
    unit = lambda w, s2: (w / metric.vec_norm(w.T, s2).T, metric.vec_norm(w.T, s2).squeeze(1))
    floor = SIGMAS / d ** 0.5
    ins, outs = [_empty(P * f, k, store) for _ in range(L)], [_empty(P * f, k, store) for _ in range(L)]
    rot = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(0)))[0].to(dev)
    prev = torch.full((L, 2, P * f), float("nan"))
    cb = f if (P * f) ** 2 <= TILE else max(1, TILE // (P * f * P))
    t0 = time.perf_counter()
    for m in range(1, L):
        wm, am, read, s2 = poles(m, dev)
        units[m] = to((wm, am), store)
        del wm, am
        s = s2.sqrt()
        in_m = to(ins[m], dev)
        for l in range(m):
            w, norm = unit(units[l][0].to(dev), s2)
            size = units[l][1].to(dev) * torch.cat([norm] * P)
            gain = (mlp_scale[l] * scalars[l + 1:m].prod()).item(), rms[m].item()
            out_l = to(outs[l], dev)
            near = torch.full((2, P * f), -torch.inf, device=dev)
            for c0 in range(0, f, cb):
                c1 = min(c0 + cb, f)
                g_scale = gain[0] / gain[1]
                c = read @ w[:, c0:c1]  # [P·f, c1-c0]
                ac = c.abs()
                if l == m - 1:
                    near[0] = torch.maximum(near[0], ac.amax(1))
                    near[1] = torch.maximum(near[1], ((read * s) @ (rot @ (w[:, c0:c1] / s[:, None]))).abs().amax(1))
                ok = ac >= floor
                # Out-edges: for each source neuron n, |push[i, p·f+n]| = |c[i, n]| · size[p·f+n] · g_scale is
                # monotone in |c[i, n]| across all target rows i, so both poles share one contiguous dim=1 topk.
                c_ok = torch.where(ok, ac, -1.0)
                top_out = c_ok.T.contiguous().topk(min(k, c.shape[0]), dim=1).indices  # [c1-c0, k]
                c_out = c.T.gather(1, top_out)  # [c1-c0, k]
                ok_out = c_out.abs() >= floor
                for p_idx, sgn in enumerate(SIGNS):
                    cols_p = torch.arange(c0, c1, device=dev) + p_idx * f
                    sz_p = size[cols_p]  # [c1-c0]
                    # In-edges for this pole sign (contiguous dim=1, no [P·f, P·f] concat)
                    sc_in = torch.where(ok, ac * (sz_p * g_scale), -1.0)
                    top_in = sc_in.topk(min(k, sc_in.shape[1]), dim=1).indices
                    c_in = sgn * c.gather(1, top_in)
                    p_in = c_in * (sz_p[top_in] * g_scale)
                    in_m = merge(in_m, (sc_in.gather(1, top_in), l * P * f + cols_p[top_in], p_in, c_in), k)
                    # Out-edges for this pole sign
                    cos_p = sgn * c_out
                    push_p = cos_p * (sz_p[:, None] * g_scale)
                    sc_out = torch.where(ok_out, push_p.abs(), -1.0)
                    new = merge(tuple(t[cols_p] for t in out_l), (sc_out, m * P * f + top_out, push_p, cos_p), k)
                    for t, v in zip(out_l, new):
                        t[cols_p] = v
                del c, ac, ok, c_ok, top_out, c_out, ok_out
            if l == m - 1:
                prev[m] = near.cpu()
            outs[l] = to(out_l, store)
            del out_l, w
        ins[m] = to(in_m, store)
        del read
        print(f"{time.strftime('%H:%M:%S')} L{m}: in from L0-L{m - 1}, {time.perf_counter() - t0:.0f}s", flush=True)
    return [tuple(t.cpu() for t in x) for x in ins], [tuple(t.cpu() for t in x) for x in outs], rms, mlp_scale, prev


def _head_write_lengths(ops: list[dict], s2: torch.Tensor, h_out: torch.Tensor | None,
                        out_top: torch.Tensor, geo: list, l: int) -> torch.Tensor:
    """``[Hl]``: each head's typical high-firing write length in the receiving metric ``s2``."""
    Hl = geo[l].q_heads
    s = s2.sqrt()
    if h_out is not None:
        o = h_out[l, :Hl, :TYPICAL].to(s.device, torch.float32)  # [Hl, TYPICAL, d]
        return (o / s[None, None, :]).norm(dim=-1).median(dim=-1).values
    src = ops[l]["src"]  # [Hl, d, hd]
    raw_f = src.flatten(1).norm(dim=1).clamp_min(1e-30)
    met_f = (src / s[None, :, None]).flatten(1).norm(dim=1)
    return out_top[l, :Hl].to(s.device) * (met_f / raw_f)


def compute_head_edges(k: int, dev, stats: tuple) -> dict:
    """Direct weight edges involving attention heads: ``attn -> MLP``, ``MLP -> attn`` and ``attn -> attn``,
    using gauge-invariant effective head operators (``neuron_lens.heads.operators``) and Haar tail-matched floors."""
    if str(dev).startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    rms_mid, rms_in, mlp_scale, scalars = stats
    geo = arch.layout(MODEL)
    L, H = len(geo), arch.max_heads(MODEL)
    par = int(arch.parallel(MODEL))
    first = poles(0, dev)
    d, f = first[0].shape
    del first
    st = extra = h_out = None
    if acts_path().exists():
        saved = torch.load(acts_path(), weights_only=False, map_location="cpu", mmap=True)
        st, extra = saved["state"]["total"], saved["state"]["extra"]
        h_out = saved["state"]["top"]["h_out"]  # [L, H, TOP, d]
    out_top = gridded(pq.read_table(head_table("acts"), columns=["out_top"]).column(0).to_numpy().copy()).float()
    ops = [head_ops.operators(MODEL, g, st, extra, dev) for g in geo]
    store = "cpu" if L * P * f * (k * 20 + d * 2) > STORE else dev
    to = lambda x, where: tuple(t.to(where) for t in x)
    n_in_head = [_empty(P * f, k, store) for _ in range(L)]
    n_out_head = {c: [_empty(P * f, k, store) for _ in range(L)] for c in CHANNELS}
    h_in_pole = {c: _empty(L * H, k, dev) for c in CHANNELS}
    h_out_pole = _empty(L * H, k, dev)
    h_in_head = {c: _empty(L * H, k, dev) for c in CHANNELS}
    h_out_head = {c: _empty(L * H, k, dev) for c in CHANNELS}

    # Preload per-layer neuron writes & act_top (on CPU if sparse).
    n_writes, n_acts = [], []
    for l in range(L):
        w, act, _, _ = poles(l, dev)
        n_writes.append(w.to(store))
        n_acts.append(act.to(store))
    t0 = time.perf_counter()

    # 1. attn -> MLP: heads of layer l <= m - par into poles of layer m.
    for m in range(par, L):
        _, _, read, s2 = poles(m, dev)  # [P·f, d]
        s = s2.sqrt()
        in_m = to(n_in_head[m], dev)
        for l in range(m - par + 1):
            Hl, hd = geo[l].q_heads, geo[l].head_dim
            src = ops[l]["src"]  # [Hl, d, hd]
            U = src / s[None, :, None]
            unorm = U.flatten(1).norm(dim=1).clamp_min(1e-30)  # [Hl]
            p2 = (U.mT @ U).flatten(1).pow(2).sum(1) / unorm.pow(4)  # [Hl]
            num = (read @ src.permute(1, 0, 2).reshape(d, Hl * hd)).view(P * f, Hl, hd).norm(dim=-1)  # [P·f, Hl]
            cos = num / unorm[None, :]
            fl = cos_floor(p2, 1.0, d, SIGMAS)[None, :]
            a_t = _head_write_lengths(ops, s2, h_out, out_top, geo, l)
            gain = scalars[l:m].prod().item()
            push = cos * a_t[None, :] * gain / rms_mid[m].item()
            score = torch.where(cos >= fl, push, -1.0)
            h_ids = l * H + torch.arange(Hl, device=dev)
            top = score.topk(min(k, Hl), dim=1).indices
            in_m = merge(in_m, (score.gather(1, top), h_ids[top], push.gather(1, top), cos.gather(1, top)), k)
            top = score.T.contiguous().topk(min(k, P * f), dim=1).indices  # [Hl, k]
            pick = lambda t, top=top: t.T.gather(1, top)
            h_rows = l * H + torch.arange(Hl, device=dev)
            new = merge(tuple(t[h_rows] for t in h_out_pole), (pick(score), m * P * f + top, pick(push), pick(cos)), k)
            for t, v in zip(h_out_pole, new):
                t[h_rows] = v
        n_in_head[m] = to(in_m, store)
        del read

    # 2. MLP -> attn and attn -> attn: sources at layer l < m into heads of layer m, per channel c ∈ {v, q, k}.
    for m in range(1, L):
        Hm = geo[m].q_heads
        s2 = metric.s2(MODEL, "in", m, dev)
        s = s2.sqrt()
        rm = rms_in[m].item()
        h_rows_m = m * H + torch.arange(Hm, device=dev)
        recv = {}
        for c in CHANNELS:
            R = ops[m][c]  # [Hm, r, d]
            r = R.shape[1]
            Rs = R * s[None, None, :]
            rnorm = Rs.flatten(1).norm(dim=1).clamp_min(1e-30)  # [Hm]
            q2 = (Rs @ Rs.mT).flatten(1).pow(2).sum(1) / rnorm.pow(4)  # [Hm]
            fl_mlp = cos_floor(1.0, q2, d, SIGMAS)[:, None]  # [Hm, 1]
            recv[c] = (R.reshape(Hm * r, d), r, rnorm, q2, fl_mlp)
        for l in range(m):
            # 2a. MLP -> attn
            w_l = n_writes[l].to(dev)
            wnorm = metric.vec_norm(w_l.T, s2).squeeze(1).clamp_min(1e-30)
            w_u = w_l / wnorm[None, :]
            size = n_acts[l].to(dev) * torch.cat([wnorm] * P)  # [P·f]
            gain_n = (mlp_scale[l] * scalars[l + 1:m].prod()).item()
            for c in CHANNELS:
                R_flat, r, rnorm, q2, fl = recv[c]
                cos_n = (R_flat @ w_u).view(Hm, r, f).norm(dim=1) / rnorm[:, None]  # [Hm, f]
                cos = cos_n.repeat(1, P)  # [Hm, P·f]
                push = cos * size[None, :] * gain_n / rm
                score = torch.where(cos >= fl, push, -1.0)
                p_ids = l * P * f + torch.arange(P * f, device=dev)
                top = score.topk(min(k, P * f), dim=1).indices
                new = merge(tuple(t[h_rows_m] for t in h_in_pole[c]),
                            (score.gather(1, top), p_ids[top], push.gather(1, top), cos.gather(1, top)), k)
                for t, v in zip(h_in_pole[c], new):
                    t[h_rows_m] = v
                out_l = to(n_out_head[c][l], dev)
                top = score.T.contiguous().topk(min(k, Hm), dim=1).indices  # [P·f, min(k, Hm)]
                pick = lambda t, top=top: t.T.gather(1, top)
                n_out_head[c][l] = to(merge(out_l, (pick(score), h_rows_m[top], pick(push), pick(cos)), k), store)
                del cos_n, cos, push, score
            del w_l, w_u

            # 2b. attn -> attn
            Hl, hd1 = geo[l].q_heads, geo[l].head_dim
            src = ops[l]["src"]  # [Hl, d, hd1]
            U = src / s[None, :, None]
            unorm = U.flatten(1).norm(dim=1).clamp_min(1e-30)  # [Hl]
            p2 = (U.mT @ U).flatten(1).pow(2).sum(1) / unorm.pow(4)  # [Hl]
            a_t = _head_write_lengths(ops, s2, h_out, out_top, geo, l)
            gain_h = scalars[l:m].prod().item()
            src_flat = src.permute(1, 0, 2).reshape(d, Hl * hd1)
            h_rows_l = l * H + torch.arange(Hl, device=dev)
            for c in CHANNELS:
                R_flat, r, rnorm, q2, _ = recv[c]
                num = (R_flat @ src_flat).view(Hm, r, Hl, hd1).pow(2).sum((1, 3)).sqrt()  # [Hm, Hl]
                cos = num / (rnorm[:, None] * unorm[None, :])
                fl = cos_floor(p2[None, :], q2[:, None], d, SIGMAS)
                push = cos * a_t[None, :] * gain_h / rm
                score = torch.where(cos >= fl, push, -1.0)
                top = score.topk(min(k, Hl), dim=1).indices
                new = merge(tuple(t[h_rows_m] for t in h_in_head[c]),
                            (score.gather(1, top), h_rows_l[top], push.gather(1, top), cos.gather(1, top)), k)
                for t, v in zip(h_in_head[c], new):
                    t[h_rows_m] = v
                top = score.topk(min(k, Hm), dim=0).indices.T
                pick = lambda t, top=top: t.T.gather(1, top)
                new = merge(tuple(t[h_rows_l] for t in h_out_head[c]),
                            (pick(score), h_rows_m[top], pick(push), pick(cos)), k)
                for t, v in zip(h_out_head[c], new):
                    t[h_rows_l] = v
    print(f"{time.strftime('%H:%M:%S')} head weight edges (attn↔MLP, attn→attn) in {time.perf_counter() - t0:.1f}s", flush=True)
    cpu = lambda tup: tuple(t.cpu() for t in tup)
    return {
        "n_in_head": [cpu(x) for x in n_in_head],
        "n_out_head": {c: [cpu(x) for x in v] for c, v in n_out_head.items()},
        "h_in_pole": {c: cpu(v) for c, v in h_in_pole.items()},
        "h_out_pole": cpu(h_out_pole),
        "h_in_head": {c: cpu(v) for c, v in h_in_head.items()},
        "h_out_head": {c: cpu(v) for c, v in h_out_head.items()},
        "H": H,
    }


def columns(cols: dict, side: str, lists: tuple, f: int) -> None:
    score, ids, push, cos = lists
    ok = score >= 0
    ids = torch.where(ok, ids, -1)
    layer, row = ids.div(P * f, rounding_mode="floor"), ids % (P * f)
    cols[f"{side}_layer"] = fixed(torch.where(ok, layer, -1).to(torch.int16))
    cols[f"{side}_neuron"] = fixed(torch.where(ok, row % f, -1).to(neuron_dtype(f)))
    cols[f"{side}_pole"] = fixed(torch.where(ok, 1 - 2 * (row // f), 0).to(torch.int8))
    cols[f"{side}_push"] = fixed(torch.where(ok, push, float("nan")))
    cols[f"{side}_cos"] = fixed(torch.where(ok, cos, float("nan")))


def head_columns(cols: dict, side: str, lists: tuple, H: int) -> None:
    score, ids, push, cos = lists
    ok = score >= 0
    ids = torch.where(ok, ids, -1)
    cols[f"{side}_layer"] = fixed(torch.where(ok, ids.div(H, rounding_mode="floor"), -1).to(torch.int16))
    cols[side] = fixed(torch.where(ok, ids % H, -1).to(torch.int16))
    cols[f"{side}_push"] = fixed(torch.where(ok, push, float("nan")))
    cols[f"{side}_cos"] = fixed(torch.where(ok, cos, float("nan")))


def head_summary(he: dict) -> dict:
    grid = arch.head_grid(MODEL).flatten()
    out = {}
    for name, tup in (("attn_to_mlp", he["h_out_pole"]),
                      *((f"mlp_to_attn_{c}", he["h_in_pole"][c]) for c in CHANNELS),
                      *((f"attn_to_attn_{c}", he["h_in_head"][c]) for c in CHANNELS)):
        s, _, p, cs = tup
        s, p, cs = s[grid], p[grid], cs[grid]
        has = s[:, 0] >= 0
        out[name] = {
            "edged": has.float().mean().item(),
            "median_edges": (s >= 0).sum(1).float().median().item(),
            "top_cos": cs[has, 0].median().item() if has.any() else float("nan"),
            "top_push": p[has, 0].median().item() if has.any() else float("nan"),
        }
    return out


def summary(ins: list, outs: list, rms: torch.Tensor, mlp_scale: torch.Tensor, prev: torch.Tensor, k: int, d: int,
            he: dict | None = None) -> None:
    L, f = len(ins), ins[0][0].shape[0] // P
    lines = [f"direct weight edges, {MODEL}: top {k} in and out per pole by |push| among |cos| >= {SIGMAS}/sqrt(d) "
             f"= {SIGMAS / d ** 0.5:.3f}; residual RMS at the MLP input "
             + ", ".join(f"L{i} {r:.1f}" for i, r in enumerate(rms.tolist()) if i % 4 == 0 or i == L - 1), "",
             "band     prev  rotated  edged  top|cos|  top push  in-edges  gap1  gap  pos   hubs | out-edged  out-edges  out gap",
             "  (prev, rotated: median of each pole's largest |cos| with the layer before's writes, and with them rotated at random; edged: share of poles with an in-edge; top |cos|, top push: medians of the strongest in-edge's; "
             "in-edges: median count above the floor, of k; gap1: share of strongest in-edges from the layer before; "
             "gap: median layer gap; pos: share of strongest in-edges that drive the pole; hubs: share of the band's "
             "in-list entries held by the 1% most-listed sources; out-: the same for out-lists)"]
    table = {}
    for lo, hi in arch.bands(MODEL) + [(0, L - 1)]:
        cat = lambda lists, i, a=lo, b=hi: torch.cat([lists[m][i] for m in range(max(a, 0), b + 1)])
        s_in, id_in, p_in, c_in = (cat(ins, i) for i in range(4))
        s_out, id_out = cat(outs, 0), cat(outs, 1)
        own = torch.cat([torch.full((P * f,), m) for m in range(lo, hi + 1)])
        has, has_out = s_in[:, 0] >= 0, s_out[:, 0] >= 0
        listed = id_in[s_in >= 0]
        counts = torch.bincount(listed - listed.min()).sort(descending=True).values if len(listed) else torch.zeros(1)
        n_hub = max(1, int(0.01 * (counts > 0).sum()))
        p = prev[max(lo, 1):hi + 1]
        row = {"poles": len(has), "prev": p[:, 0].median().item(), "rotated": p[:, 1].median().item(), "edged": has.float().mean().item(),
               "top_cos": c_in[has, 0].abs().median().item() if has.any() else float("nan"),
               "top_push": p_in[has, 0].abs().median().item() if has.any() else float("nan"),
               "in_edges": (s_in >= 0).sum(1).float().median().item(),
               "gap1": (own[has] - id_in[has, 0] // (P * f) == 1).float().mean().item() if has.any() else float("nan"),
               "gap": (own[has] - id_in[has, 0] // (P * f)).float().median().item() if has.any() else float("nan"),
               "pos": (p_in[has, 0] > 0).float().mean().item() if has.any() else float("nan"),
               "hubs": (counts[:n_hub].sum() / counts.sum()).item() if len(listed) else float("nan"),
               "out_edged": has_out.float().mean().item(),
               "out_edges": (s_out >= 0).sum(1).float().median().item(),
               "out_gap": (id_out[has_out, 0] // (P * f) - own[has_out]).float().median().item() if has_out.any() else float("nan")}
        table[f"L{lo}-{hi}"] = row
        lines.append(f"L{lo}-{hi}".ljust(8) + f" {row['prev']:4.3f}  {row['rotated']:7.3f}  {row['edged']:5.2f}  {row['top_cos']:8.3f}  {row['top_push']:8.3f}  "
                     f"{row['in_edges']:8.0f}  {row['gap1']:4.2f}  {row['gap']:3.0f}  {row['pos']:4.2f}  {row['hubs']:4.2f} | "
                     f"{row['out_edged']:9.2f}  {row['out_edges']:9.0f}  {row['out_gap']:7.0f}")
    hs = head_summary(he) if he is not None else None
    if hs is not None:
        lines += ["", "head weight edges (tail-matched floor): edged  median-edges  top-cos  top-push"]
        for name, r in hs.items():
            lines.append(f"  {name:18s}  {r['edged']:5.2f}  {r['median_edges']:12.0f}  {r['top_cos']:7.4f}  {r['top_push']:8.3f}")
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"edges_{STEM}.log").write_text(text + "\n")
    (SUMMARY / f"edges_{STEM}.json").write_text(json.dumps({"model": MODEL, "k": k, "sigmas": SIGMAS,
                                                            "rms": rms.tolist(), "mlp_scale": mlp_scale.tolist(),
                                                            "bands": table, **({"heads": hs} if hs is not None else {})}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
    args = ap.parse_args()
    stats = residual_stats(args.device)
    ins, outs, rms, mlp_scale, prev = compute(TOP, args.device, stats=stats)
    he = compute_head_edges(TOP, args.device, stats)
    f, H = ins[0][0].shape[0] // P, he["H"]
    for layer, (i, o) in enumerate(zip(ins, outs)):
        cols = neuron_keys(layer, f)
        columns(cols, "in", i, f)
        columns(cols, "out", o, f)
        head_columns(cols, "in_head", he["n_in_head"][layer], H)
        for c in CHANNELS:
            head_columns(cols, f"{c}_out_head", he["n_out_head"][c][layer], H)
        write(cols, neuron_table("edges", layer))
    hcols = head_keys()
    for c in CHANNELS:
        columns(hcols, f"{c}_in", he["h_in_pole"][c], f)
    columns(hcols, "out", he["h_out_pole"], f)
    for c in CHANNELS:
        head_columns(hcols, f"{c}_in_head", he["h_in_head"][c], H)
        head_columns(hcols, f"{c}_out_head", he["h_out_head"][c], H)
    write_heads(hcols, head_table("edges"))
    summary(ins, outs, rms, mlp_scale, prev, TOP, directions(MODEL, 0)["write"].shape[0], he)


if __name__ == "__main__":
    main()
