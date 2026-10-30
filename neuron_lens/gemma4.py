"""Gemma 4's decoder layers as RelP linear maps, for ``neuron_lens.transport``.

The model is ``gemma4_unified`` (Gemma 4 12B, encoder-free: images and audio enter the same
text stack through linear embedders). Its text layer, as ``Gemma4UnifiedTextDecoderLayer``
runs it:

    x = n_in(h);  a = n_pa(o_proj(attention(x)));  h = h + a                (mid: this h)
    z = n_post(down(act(gate(n_pre(h))) · up(n_pre(h))));  h = (h + z) · layer_scalar

with tanh-GELU, four RMSNorms with plain weights (all under the LN rule), and attention
scaled by 1 (q and k are RMS-normed per head). Sliding layers have ``head_dim`` 256 and 8
key heads; full ("global") layers have 512 and one, keys and values from one projection
(``attention_k_eq_v``: the value is the raw key, before ``k_norm`` and RoPE), and RoPE on a
quarter of the head (the model's rotary table is the identity on the rest, so it is used as
is). Values pass a weightless RMSNorm, which keeps its true derivative, as q/k norms do.

The sliding window (1,024) never binds on lens documents (128 tokens); ``linearize`` refuses
longer ones rather than mask them. Image tokens' bidirectional attention is not modelled:
the lens corpus is text.
"""
from typing import NamedTuple

import torch
import torch.nn.functional as F

from neuron_lens.rules import HALF_RULE_BETA, act_factor, check_layer, norm_eps, norm_weight

_checked = set()


def is_gemma4(model) -> bool:
    """A ``gemma4_unified`` text stack (the 12B); other Gemma 4 text stacks differ."""
    return getattr(model.config, "model_type", "").startswith("gemma4_unified")


class LayerMap(NamedTuple):
    """One Gemma 4 decoder layer's RelP linear map on a document (``transport._LayerMap``'s
    conventions: rows are the ``N`` valid positions, keys run over ``Tp`` padded positions)."""
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor | None  # None: the value is the raw key (full layers)
    wo: torch.Tensor
    wg: torch.Tensor
    wu: torch.Tensor
    wd: torch.Tensor
    a1: torch.Tensor  # [N, d]: input norm weight × its rsqrt (LN rule)
    a2: torch.Tensor  # [N, d]: post-attention norm, on the attention's output
    a3: torch.Tensor  # [N, d]: pre-feedforward norm (the MLP's input norm)
    a4: torch.Tensor  # [N, d]: post-feedforward norm, on the MLP's output
    s: torch.Tensor  # []: layer_scalar
    q: torch.Tensor  # [N, Hq, hd], [N, Hq, 1], [hd]: queries before q_norm, its rsqrt, weight
    rq: torch.Tensor
    wqn: torch.Tensor
    k: torch.Tensor  # the same for keys
    rk: torch.Tensor
    wkn: torch.Tensor
    v: torch.Tensor  # [N, Hk, hd], [N, Hk, 1]: values before their (weightless) norm
    rv: torch.Tensor
    cos: torch.Tensor  # [N, 1, hd]
    sin: torch.Tensor
    qr: torch.Tensor  # [Hk, g·N, hd]: normed, rotated queries, grouped by key head
    kr: torch.Tensor  # [Hk, Tp, hd]: normed, rotated keys
    vn: torch.Tensor  # [Hk, Tp, hd]: normed values
    p: torch.Tensor  # [Hk, g·N, Tp]
    scale: float
    s0: int  # the first valid position
    xs: tuple | None  # exact mode: per residual norm (in, pa, pre, post), its input [N, d] and rsqrt [N, 1]
    cg: torch.Tensor  # [N, d_ff]: β·φ(gate)·up, the gate's factor (identity, half rules)
    cu: torch.Tensor  # [N, d_ff]: β·act(gate), the up projection's factor


def check(model) -> None:
    """Refuse a Gemma 4 text stack with parts ``linearize`` does not write out."""
    if id(model) in _checked:
        return
    c = model.config
    for name, ok in (("num_kv_shared_layers", 0), ("hidden_size_per_layer_input", 0),
                     ("enable_moe_block", False)):
        if getattr(c, name, ok) not in (ok, None):
            raise ValueError(f"Gemma 4 with {name}={getattr(c, name)} is not supported")
    for layer in model.layers:
        check_layer(layer)
        at, mlp = layer.self_attn, layer.mlp
        projections = (at.q_proj, at.k_proj, at.v_proj, at.o_proj,
                       mlp.gate_proj, mlp.up_proj, mlp.down_proj)
        if any(p is not None and p.bias is not None for p in projections):
            raise ValueError("projection biases are not supported")
        if not at.is_causal:
            raise ValueError("bidirectional text attention is not supported")
    _checked.add(id(model))


def _norm(norm, x: torch.Tensor):
    """HF's Gemma 4 RMSNorm (fp32, then the weight, then the input's dtype), and its rsqrt."""
    r = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm_eps(norm))
    w = norm_weight(norm)
    y = x.float() * r
    return (y if w is None else y * w.float()).to(x.dtype), r


@torch.no_grad()
def linearize(model, ids: torch.Tensor, embeds: torch.Tensor | None = None,
              block: torch.Tensor | None = None, first: int | None = None, rules: str = "relp") -> list[LayerMap]:
    """Run the primal over a document and keep each layer's linear map.

    ``embeds`` (``[len(ids), d]``) replaces the token embeddings, for documents with image or
    audio tokens (``multimodal_embeds``); ``block`` (``[len(ids)]`` bool) marks one image's
    tokens, which attend to each other both ways in sliding layers (full layers stay causal),
    as the model's vision mask has it. The block must lie in the valid positions, so that no
    earlier position attends to a later one that carries a tangent. ``first`` (default
    ``SKIP_FIRST``) moves the first valid position later: rows before it carry no tangent or
    cotangent (exact when every seed and read is at or after it), which saves their work.
    ``rules="true"`` keeps every derivative true instead of RelP's: the four residual norms
    with their rank-one term, and the MLP's ``act'(g)·u dg + act(g) du`` (audio documents read
    better so)."""
    from neuron_lens.transport import SKIP_FIRST, _rope, valid_positions

    check(model)
    valid_positions(len(ids))
    dev, dt = model.embed_tokens.weight.device, model.embed_tokens.weight.dtype
    ids = ids[:-1].to(dev)
    T, S = len(ids), SKIP_FIRST if first is None else first
    if S < SKIP_FIRST or S >= T:
        raise ValueError(f"first valid position {S} outside [{SKIP_FIRST}, {T})")
    tp = -(-T // 8) * 8
    # Text tokens are scaled by √d in the embedding, as the model scales them; image and audio
    # embeddings come in unscaled, as the model scatters them.
    h = model.embed_tokens(ids) if embeds is None else embeds[:-1].to(dev, dt)
    pos = torch.arange(T, device=dev)[None]
    tables = {t: model.rotary_emb(h[None], pos, t) for t in set(model.config.layer_types)}
    causal = torch.full((T, tp), float("-inf"), device=dev).triu(1)
    masks = {"full_attention": causal, "sliding_attention": causal}
    if block is not None:
        b = block[:-1].to(dev)
        if b[:S].any():
            raise ValueError(f"the bidirectional block starts before position {S}")
        both = F.pad(b[:, None] & b[None, :], (0, tp - T))
        masks["sliding_attention"] = causal.masked_fill(both, 0.0)
    maps = []
    for i, layer in enumerate(model.layers):
        at, mlp = layer.self_attn, layer.mlp
        if at.sliding_window is not None and T > at.sliding_window:
            raise ValueError(f"{T} tokens exceed layer {i}'s sliding window ({at.sliding_window})")
        cos, sin = (c[0, :, None] for c in tables[model.config.layer_types[i]])  # [T, 1, hd]
        hd, g = at.head_dim, at.num_key_value_groups
        hk = at.k_proj.weight.shape[0] // hd
        h_in = h
        x, r1 = _norm(layer.input_layernorm, h)
        q = F.linear(x, at.q_proj.weight).view(T, hk * g, hd)
        k = F.linear(x, at.k_proj.weight).view(T, hk, hd)
        v = k if at.v_proj is None else F.linear(x, at.v_proj.weight).view(T, hk, hd)
        (qn, rq), (kn, rk), (vn, rv) = _norm(at.q_norm, q), _norm(at.k_norm, k), _norm(at.v_norm, v)
        qr = _rope(qn, cos, sin).transpose(0, 1).reshape(hk, g, T, hd)
        kr = F.pad(_rope(kn, cos, sin).transpose(0, 1), (0, 0, 0, tp - T))  # [Hk, Tp, hd]
        vp = F.pad(vn.transpose(0, 1), (0, 0, 0, tp - T))
        mask = masks[model.config.layer_types[i]]
        p = torch.softmax((qr @ kr[:, None].mT).float() * at.scaling + mask, -1).to(dt)
        o = (p @ vp[:, None]).reshape(hk * g, T, hd).transpose(0, 1).reshape(T, -1)
        a_in = F.linear(o, at.o_proj.weight)
        a, r2 = _norm(layer.post_attention_layernorm, a_in)
        h = h + a
        h_mid = h
        x, r3 = _norm(layer.pre_feedforward_layernorm, h)
        gate, up = F.linear(x, mlp.gate_proj.weight), F.linear(x, mlp.up_proj.weight)
        z_in = F.linear(mlp.act_fn(gate) * up, mlp.down_proj.weight)
        z, r4 = _norm(layer.post_feedforward_layernorm, z_in)
        h = (h + z) * layer.layer_scalar
        gate, up = gate[S:].float(), up[S:].float()
        weight = lambda norm, r: norm.weight.float() * r[S:]
        if rules not in ("relp", "true"):
            raise ValueError(f"rules must be relp or true, not {rules}")
        if rules == "true":
            g_full = gate.requires_grad_(True)
            with torch.enable_grad():
                dact = torch.autograd.grad(mlp.act_fn(g_full).sum(), g_full)[0]
            xs = tuple((x_[S:].float(), r_[S:]) for x_, r_ in ((h_in, r1), (a_in, r2), (h_mid, r3), (z_in, r4)))
        else:
            xs = None
        maps.append(LayerMap(
            wq=at.q_proj.weight, wk=at.k_proj.weight,
            wv=None if at.v_proj is None else at.v_proj.weight, wo=at.o_proj.weight,
            wg=mlp.gate_proj.weight, wu=mlp.up_proj.weight, wd=mlp.down_proj.weight,
            a1=weight(layer.input_layernorm, r1), a2=weight(layer.post_attention_layernorm, r2),
            a3=weight(layer.pre_feedforward_layernorm, r3), a4=weight(layer.post_feedforward_layernorm, r4),
            s=layer.layer_scalar.float().reshape(()),
            q=q[S:].float(), rq=rq[S:], wqn=at.q_norm.weight.float(),
            k=k[S:].float(), rk=rk[S:], wkn=at.k_norm.weight.float(),
            v=v[S:].float(), rv=rv[S:],
            cos=cos[S:].float(), sin=sin[S:].float(),
            qr=qr[:, :, S:].reshape(hk, -1, hd), kr=kr, vn=vp, p=p[:, :, S:].reshape(hk, -1, tp),
            scale=float(at.scaling), s0=S, xs=xs,
            cg=((HALF_RULE_BETA * act_factor(mlp.act_fn, gate) * up) if xs is None else (dact * up)).detach().to(dt),
            cu=((HALF_RULE_BETA if xs is None else 1.0) * mlp.act_fn(gate)).detach().to(dt)))
    return maps


def jvp(m: LayerMap, X: torch.Tensor):
    """Push tangents ``X`` (``[b, N, d]``, fp32) at a layer's input through its map; return
    them at its output and at the MLP's input (``mid``)."""
    from neuron_lens.transport import _norm_jvp, _rope

    dt = m.wq.dtype
    b, N, _ = X.shape
    hk, gn, hd = m.qr.shape
    g = gn // N
    # A residual norm's map: RelP's diagonal scaling, or (exact mode) the true derivative.
    nj = (lambda dx, a, j: dx * a) if m.xs is None else \
        (lambda dx, a, j: a * (dx - m.xs[j][0] * m.xs[j][1].square() * (m.xs[j][0] * dx).mean(-1, keepdim=True)))
    xn = nj(X, m.a1, 0).to(dt)
    dq = _rope(_norm_jvp(F.linear(xn, m.wq).view(b, N, hk * g, hd).float(), m.q, m.rq, m.wqn),
               m.cos, m.sin)
    dk = F.linear(xn, m.wk).view(b, N, hk, hd).float()
    dv = dk if m.wv is None else F.linear(xn, m.wv).view(b, N, hk, hd).float()
    dk = _rope(_norm_jvp(dk, m.k, m.rk, m.wkn), m.cos, m.sin)
    dv = _norm_jvp(dv, m.v, m.rv, 1.0)
    dq = dq.to(dt).view(b, N, hk, g, hd).permute(0, 2, 3, 1, 4).reshape(b, hk, gn, hd)
    dk, dv = dk.to(dt).transpose(1, 2), dv.to(dt).transpose(1, 2)  # [b, Hk, N, hd]
    keys = slice(m.s0, m.s0 + N)
    ds = (dq @ m.kr.mT).float()  # [b, Hk, g·N, Tp]
    ds[..., keys] += (m.qr @ dk.mT).float()
    ds *= m.scale
    dp = (m.p * (ds - (m.p * ds).sum(-1, keepdim=True))).to(dt)
    do = dp @ m.vn + m.p[..., keys] @ dv  # [b, Hk, g·N, hd]
    do = do.view(b, hk, g, N, hd).permute(0, 3, 1, 2, 4).reshape(b, N, -1)
    mid = X + nj(F.linear(do, m.wo).float(), m.a2, 1)
    xn = nj(mid, m.a3, 2).to(dt)
    z = F.linear(m.cg * F.linear(xn, m.wg) + m.cu * F.linear(xn, m.wu), m.wd).float()
    return (mid + nj(z, m.a4, 3)) * m.s, mid


def vjp(m: LayerMap, G: torch.Tensor, mid_seed: torch.Tensor | None = None, row_w: torch.Tensor | None = None):
    """Pull cotangents ``G`` (``[b, N, d]``, fp32) at a layer's output back through its map;
    return them at its input, and their means over positions at the MLP's input (``mid``) and
    at the layer's input (weighted by ``row_w``, ``[N]`` summing to 1, if given). ``mid_seed``
    (``[b, N, d]``) joins the cotangents at the MLP's input."""
    from neuron_lens.transport import _norm_vjp, _rope

    dt = m.wq.dtype
    b, N, _ = G.shape
    hk, gn, hd = m.qr.shape
    g = gn // N
    # A residual norm's transpose: RelP's diagonal scaling, or (exact mode) the true one (the
    # norm's derivative a·(I − x̂x̂ᵀ/d)·r with a = w·r is symmetric up to the weight's side).
    nt = (lambda gy, a, j: gy * a) if m.xs is None else \
        (lambda gy, a, j: (gy * a) - m.xs[j][0] * m.xs[j][1].square() * (m.xs[j][0] * (gy * a)).mean(-1, keepdim=True))
    G = G * m.s
    gp = nt(G, m.a4, 3).to(dt) @ m.wd  # [b, N, d_ff]
    G = G + nt(((m.cg * gp) @ m.wg + (m.cu * gp) @ m.wu).float(), m.a3, 2)
    if mid_seed is not None:
        G = G + mid_seed
    mid = G.mean(1) if row_w is None else (G * row_w[:, None]).sum(1)
    go = (nt(G, m.a2, 1).to(dt) @ m.wo).view(b, N, hk, g, hd).permute(0, 2, 3, 1, 4).reshape(b, hk, gn, hd)
    keys = slice(m.s0, m.s0 + N)
    gp = (go @ m.vn.mT).float()  # [b, Hk, g·N, Tp]
    gs = (m.p * (gp - (m.p * gp).sum(-1, keepdim=True)) * m.scale).to(dt)
    gq = (gs @ m.kr).view(b, hk, g, N, hd).permute(0, 3, 1, 2, 4).reshape(b, N, hk * g, hd)
    gk = (gs[..., keys].mT @ m.qr).transpose(1, 2)  # [b, N, Hk, hd]
    gv = (m.p[..., keys].mT @ go).transpose(1, 2)
    gq = _norm_vjp(_rope(gq.float(), m.cos, -m.sin), m.q, m.rq, m.wqn).to(dt)
    gk = _norm_vjp(_rope(gk.float(), m.cos, -m.sin), m.k, m.rk, m.wkn)
    gv = _norm_vjp(gv.float(), m.v, m.rv, 1.0)
    if m.wv is None:  # the value is the raw key: both cotangents reach k_proj
        gx = gq.reshape(b, N, -1) @ m.wq + (gk + gv).to(dt).reshape(b, N, -1) @ m.wk
    else:
        gx = (gq.reshape(b, N, -1) @ m.wq + gk.to(dt).reshape(b, N, -1) @ m.wk
              + gv.to(dt).reshape(b, N, -1) @ m.wv)
    G = G + nt(gx.float(), m.a1, 0)
    return G, mid, G.mean(1) if row_w is None else (G * row_w[:, None]).sum(1)


@torch.no_grad()
def multimodal_embeds(lm, inputs: dict) -> torch.Tensor:
    """``[T, d]``: the residual stream's input for one processed document (``inputs`` as the
    processor returns them, batch 1): scaled text-token embeddings with image and audio
    embeddings scattered in, as ``Gemma4UnifiedModel.forward`` does."""
    model = lm.model
    ids = inputs["input_ids"]
    image_mask, video_mask, audio_mask = model.get_placeholder_mask(ids)
    text_ids = torch.where(image_mask | video_mask | audio_mask, model.config.text_config.pad_token_id, ids)
    h = model.language_model.embed_tokens(text_ids)
    if "pixel_values" in inputs:
        feats = torch.cat(model.get_image_features(inputs["pixel_values"], inputs["image_position_ids"]).pooler_output)
        h = h.masked_scatter(image_mask[..., None].expand_as(h), feats.to(h.dtype))
    if "input_features" in inputs:
        feats = model.get_audio_features(inputs["input_features"], inputs["input_features_mask"]).pooler_output
        feats = feats[inputs["input_features_mask"]]
        h = h.masked_scatter(audio_mask[..., None].expand_as(h), feats.to(h.dtype))
    return h[0]
