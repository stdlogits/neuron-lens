"""The R-lens and the S-lens: one Jacobian estimator, run in two directions.

Both lenses average a Jacobian of the residual stream over a corpus, summed over offsets as
in the J-lens, and read it with a token matrix:
- **R-lens** (reverse mode): ``J_L = mean_t Σ_{t'≥t} ∂h_final(t') / ∂h_L(t)``, from layer
  ``L`` to the final layer's output (before the final norm), read with ``W_U``: what an
  activation makes the model say, here and later. Its ``out`` lens starts from each layer's
  output (Drowse's estimator, the lens Drowse loads; MLP writes land there), its ``mid`` lens
  from the residual each layer's MLP reads (MLP read directions, and attention writes).
- **S-lens** (forward mode): ``K_L = mean_t Σ_{s≤t} ∂h_L(t) / ∂e(s)``, from the token
  embeddings to each layer, read with ``W_E``: which tokens, here or earlier, made an
  activation. Its targets are ``mid`` and ``out`` too.

``mid`` is the input of ``post_attention_layernorm``; ``out`` is the layer's output.

The derivative is RelP's (``neuron_lens.rules`` states the rules and applies them to the
model for autograd, which the tests use as the reference). Under the rules each decoder layer
is, on a given document, a fixed linear map of the residual: the norms and the MLP become
per-position scalings, and attention keeps its true derivative (softmax, q/k norms, RoPE).
``_linearize`` runs the primal once per document and keeps each layer's constants. The
S-lens pushes tangents up through the maps, the R-lens pulls cotangents down through their
transposes, ``dim_batch`` basis directions at a time, so each layer is a few matrix products
over ``dim_batch × positions`` rows, written out by hand (no autograd). On CUDA the layer steps
are compiled, which fuses the elementwise work between the products.

One end is fixed in each lens, so ``d_model`` basis directions of that end cover every layer
at the other. The seed goes at every valid position and the read is averaged over them.
Valid positions are ``[SKIP_FIRST, T − 1)``: attention sinks and the final position are out,
as in Drowse. Only valid positions ever carry a tangent or a cotangent (attention is causal,
so nothing seeded there reaches an earlier position, and nothing at an earlier position is
read), and the final position is dropped before the primal, since no valid position attends
to it.

**Documents** weigh equally (per-document means) and alternate between two halves, even and
odd, for split-half checks.
"""
import functools
import weakref
from typing import NamedTuple

import torch
import torch.nn.functional as F

from neuron_lens.rules import HALF_RULE_BETA, check_layer

SKIP_FIRST = 16  # Drowse's attention-sink cut; Drowse loads only lenses fitted with it
TARGETS = {"R": ("out", "mid"), "S": ("mid", "out")}
DIM_BATCH = 256


class Accumulator:
    """Sums of per-document estimates, ``[layer, d_out, d_in]`` per target, in two halves
    (even and odd documents). The fit format is ``{"meta", "state", "next_doc",
    "done_sha256"}`` with ``state = {"sums": {(target, half): tensor}, "n_docs": [even, odd]}``."""

    def __init__(self, layers: dict[str, int], d: int, device="cpu", d_in: int | None = None):
        """``d_in``: the S-lens's source width when it is seeded in a subspace (``basis``)."""
        self.sums = {(t, h): torch.zeros(n, d, d if d_in is None else d_in, device=device)
                     for t, n in layers.items() for h in (0, 1)}
        self.n_docs = [0, 0]

    def mean(self, target: str, half: int | None = None) -> torch.Tensor:
        halves = (0, 1) if half is None else (half,)
        n = sum(self.n_docs[h] for h in halves)
        return sum(self.sums[(target, h)] for h in halves) / n

    def split_half(self, target: str) -> list[float]:
        """Per layer, ``‖mean_even − mean_odd‖ / ‖mean‖``: how differently the two halves of
        the corpus read (``nan`` while a half is empty)."""
        a, b = (self.sums[(target, h)] for h in (0, 1))
        n0, n1 = self.n_docs
        if not (n0 and n1):
            return [float("nan")] * a.shape[0]
        return [((a[l] / n0 - b[l] / n1).norm() / ((a[l] + b[l]) / (n0 + n1)).norm()).item()
                for l in range(a.shape[0])]

    def merge(self, other: "Accumulator") -> None:
        for key, tensor in other.sums.items():
            self.sums[key] += tensor.to(self.sums[key].device)
        self.n_docs = [a + b for a, b in zip(self.n_docs, other.n_docs)]

    def state(self) -> dict:
        return {"sums": self.sums, "n_docs": self.n_docs}

    @classmethod
    def from_state(cls, state: dict, device=None) -> "Accumulator":
        acc = cls.__new__(cls)
        acc.sums = {k: v if device is None else v.to(device) for k, v in state["sums"].items()}
        acc.n_docs = list(state["n_docs"])
        return acc


def valid_positions(T: int, device=None) -> torch.Tensor:
    if T - 1 <= SKIP_FIRST:
        raise ValueError(f"{T} tokens leave no valid positions after the first {SKIP_FIRST}")
    return torch.arange(SKIP_FIRST, T - 1, device=device)


class _LayerMap(NamedTuple):
    """One decoder layer's RelP linear map on a document. Rows are the ``N`` valid
    positions; keys run over all ``T`` positions the primal kept, padded with masked ones to a
    multiple of 8 (``Tp``) so that rows of the attention products stay 16-byte aligned, which
    cuBLAS's fast kernels need."""
    wq: torch.Tensor  # projection weights, [out, in]
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    wg: torch.Tensor
    wu: torch.Tensor
    wd: torch.Tensor
    a1: torch.Tensor  # [N, d]: input norm weight × its rsqrt, held constant (LN rule)
    a2: torch.Tensor  # [N, d]: the same for the post-attention norm
    q: torch.Tensor  # [N, Hq, hd]: queries before q_norm (whose true derivative is kept)
    rq: torch.Tensor  # [N, Hq, 1]: their rsqrt
    wqn: torch.Tensor  # [hd]: q_norm's weight
    k: torch.Tensor  # [N, Hk, hd], [N, Hk, 1], [hd]: the same for keys
    rk: torch.Tensor
    wkn: torch.Tensor
    cos: torch.Tensor  # [N, 1, hd]: RoPE
    sin: torch.Tensor
    qr: torch.Tensor  # [Hk, g·N, hd]: normed, rotated queries, grouped by key head
    kr: torch.Tensor  # [Hk, Tp, hd]: normed, rotated keys
    v: torch.Tensor  # [Hk, Tp, hd]
    p: torch.Tensor  # [Hk, g·N, Tp]: attention probabilities
    cg: torch.Tensor  # [N, d_ff]: β·sigmoid(gate)·up, the gate's factor (identity, half rules)
    cu: torch.Tensor  # [N, d_ff]: β·silu(gate), the up projection's factor


_checked = weakref.WeakSet()


def _check(model) -> None:
    """Refuse a decoder stack that is not plain Qwen3 (``_linearize`` writes its forward out)."""
    if model in _checked:
        return
    for layer in model.layers:
        check_layer(layer)
        at, mlp = layer.self_attn, layer.mlp
        projections = (at.q_proj, at.k_proj, at.v_proj, at.o_proj,
                       mlp.gate_proj, mlp.up_proj, mlp.down_proj)
        if any(p.bias is not None for p in projections):
            raise ValueError("projection biases are not supported")
        if getattr(at, "sliding_window", None) or at.scaling != at.head_dim ** -0.5:
            raise ValueError("only full attention scaled by head_dim^-0.5 is supported")
    _checked.add(model)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


def _rope(x, cos, sin):
    """RoPE; ``_rope(x, cos, -sin)`` is its transpose."""
    return x * cos + _rotate_half(x) * sin


def _rms_norm(norm, x: torch.Tensor):
    """HF's RMSNorm, and the rsqrt it scaled by."""
    r = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    return norm.weight * (x.float() * r).to(x.dtype), r


def _norm_jvp(dx, x, r, w):
    """The true derivative of the RMSNorm ``w·x·r`` over the last dim, applied to ``dx``."""
    return w * r * (dx - x * r.square() * (x * dx).mean(-1, keepdim=True))


def _norm_vjp(gy, x, r, w):
    a = w * gy
    return r * (a - x * r.square() * (x * a).mean(-1, keepdim=True))


@torch.no_grad()
def _linearize(model, ids: torch.Tensor) -> list[_LayerMap]:
    """Run the primal over a document and keep each layer's linear map."""
    _check(model)
    valid_positions(len(ids))  # refuses documents too short to have any
    dev, dt = model.embed_tokens.weight.device, model.embed_tokens.weight.dtype
    ids = ids[:-1].to(dev)
    T, S = len(ids), SKIP_FIRST
    tp = -(-T // 8) * 8
    h = model.embed_tokens(ids)
    cos, sin = model.rotary_emb(h[None], torch.arange(T, device=dev)[None])
    cos, sin = cos[0, :, None], sin[0, :, None]  # [T, 1, hd]
    causal = torch.full((T, tp), float("-inf"), device=dev).triu(1)  # masks the padding too
    maps = []
    for layer in model.layers:
        at, mlp = layer.self_attn, layer.mlp
        hd, g = at.head_dim, at.num_key_value_groups
        hk = at.k_proj.weight.shape[0] // hd
        x, r1 = _rms_norm(layer.input_layernorm, h)
        q = F.linear(x, at.q_proj.weight).view(T, hk * g, hd)
        k = F.linear(x, at.k_proj.weight).view(T, hk, hd)
        v = F.pad(F.linear(x, at.v_proj.weight).view(T, hk, hd).transpose(0, 1), (0, 0, 0, tp - T))
        (qn, rq), (kn, rk) = _rms_norm(at.q_norm, q), _rms_norm(at.k_norm, k)
        qr = _rope(qn, cos, sin).transpose(0, 1).reshape(hk, g, T, hd)
        kr = F.pad(_rope(kn, cos, sin).transpose(0, 1), (0, 0, 0, tp - T))  # [Hk, Tp, hd]
        p = torch.softmax((qr @ kr[:, None].mT).float() * at.scaling + causal, -1).to(dt)
        o = (p @ v[:, None]).reshape(hk * g, T, hd).transpose(0, 1).reshape(T, -1)
        h = h + F.linear(o, at.o_proj.weight)
        x, r2 = _rms_norm(layer.post_attention_layernorm, h)
        gate, up = F.linear(x, mlp.gate_proj.weight), F.linear(x, mlp.up_proj.weight)
        h = h + F.linear(F.silu(gate) * up, mlp.down_proj.weight)
        gate, up = gate[S:].float(), up[S:].float()
        maps.append(_LayerMap(
            wq=at.q_proj.weight, wk=at.k_proj.weight, wv=at.v_proj.weight,
            wo=at.o_proj.weight, wg=mlp.gate_proj.weight, wu=mlp.up_proj.weight,
            wd=mlp.down_proj.weight,
            a1=layer.input_layernorm.weight.float() * r1[S:],
            a2=layer.post_attention_layernorm.weight.float() * r2[S:],
            q=q[S:].float(), rq=rq[S:], wqn=at.q_norm.weight.float(),
            k=k[S:].float(), rk=rk[S:], wkn=at.k_norm.weight.float(),
            cos=cos[S:].float(), sin=sin[S:].float(),
            qr=qr[:, :, S:].reshape(hk, -1, hd), kr=kr, v=v, p=p[:, :, S:].reshape(hk, -1, tp),
            cg=(HALF_RULE_BETA * torch.sigmoid(gate) * up).to(dt),
            cu=(HALF_RULE_BETA * F.silu(gate)).to(dt)))
    return maps


def _jvp(m: _LayerMap, X: torch.Tensor):
    """Push tangents ``X`` (``[b, N, d]``, fp32) at a layer's input through its map; return
    them at its output and at the MLP's input (``mid``)."""
    dt = m.wq.dtype
    b, N, _ = X.shape
    hk, gn, hd = m.qr.shape
    g = gn // N
    xn = (X * m.a1).to(dt)
    dq = _rope(_norm_jvp(F.linear(xn, m.wq).view(b, N, hk * g, hd).float(), m.q, m.rq, m.wqn),
               m.cos, m.sin)
    dk = _rope(_norm_jvp(F.linear(xn, m.wk).view(b, N, hk, hd).float(), m.k, m.rk, m.wkn),
               m.cos, m.sin)
    dq = dq.to(dt).view(b, N, hk, g, hd).permute(0, 2, 3, 1, 4).reshape(b, hk, gn, hd)
    dk = dk.to(dt).transpose(1, 2)  # [b, Hk, N, hd]
    dv = F.linear(xn, m.wv).view(b, N, hk, hd).transpose(1, 2)
    keys = slice(SKIP_FIRST, SKIP_FIRST + N)  # the valid positions, as keys
    ds = (dq @ m.kr.mT).float()  # [b, Hk, g·N, Tp]
    ds[..., keys] += (m.qr @ dk.mT).float()  # keys at valid positions move too
    ds *= hd ** -0.5
    dp = (m.p * (ds - (m.p * ds).sum(-1, keepdim=True))).to(dt)
    do = dp @ m.v + m.p[..., keys] @ dv  # [b, Hk, g·N, hd]
    do = do.view(b, hk, g, N, hd).permute(0, 3, 1, 2, 4).reshape(b, N, -1)
    mid = X + F.linear(do, m.wo).float()
    xn = (mid * m.a2).to(dt)
    return mid + F.linear(m.cg * F.linear(xn, m.wg) + m.cu * F.linear(xn, m.wu), m.wd).float(), mid


def _vjp(m: _LayerMap, G: torch.Tensor, mid_seed: torch.Tensor | None = None, row_w: torch.Tensor | None = None):
    """Pull cotangents ``G`` (``[b, N, d]``, fp32) at a layer's output back through its map;
    return them at its input, and their means over positions at the MLP's input (``mid``) and
    at the layer's input (weighted by ``row_w``, ``[N]`` summing to 1, if given). ``mid_seed``
    (``[b, N, d]``) joins the cotangents at the MLP's input."""
    dt = m.wq.dtype
    b, N, _ = G.shape
    hk, gn, hd = m.qr.shape
    g = gn // N
    gp = G.to(dt) @ m.wd  # [b, N, d_ff]
    G = G + ((m.cg * gp) @ m.wg + (m.cu * gp) @ m.wu).float() * m.a2
    if mid_seed is not None:
        G = G + mid_seed
    mid = G.mean(1) if row_w is None else (G * row_w[:, None]).sum(1)
    go = (G.to(dt) @ m.wo).view(b, N, hk, g, hd).permute(0, 2, 3, 1, 4).reshape(b, hk, gn, hd)
    keys = slice(SKIP_FIRST, SKIP_FIRST + N)  # the valid positions, as keys
    gp = (go @ m.v.mT).float()  # [b, Hk, g·N, Tp]
    gs = (m.p * (gp - (m.p * gp).sum(-1, keepdim=True)) * hd ** -0.5).to(dt)
    gq = (gs @ m.kr).view(b, hk, g, N, hd).permute(0, 3, 1, 2, 4).reshape(b, N, hk * g, hd)
    gk = (gs[..., keys].mT @ m.qr).transpose(1, 2)  # [b, N, Hk, hd]
    gv = (m.p[..., keys].mT @ go).transpose(1, 2)
    gq = _norm_vjp(_rope(gq.float(), m.cos, -m.sin), m.q, m.rq, m.wqn).to(dt)
    gk = _norm_vjp(_rope(gk.float(), m.cos, -m.sin), m.k, m.rk, m.wkn).to(dt)
    gx = gq.reshape(b, N, -1) @ m.wq + gk.reshape(b, N, -1) @ m.wk + gv.reshape(b, N, -1) @ m.wv
    G = G + gx.float() * m.a1
    return G, mid, G.mean(1) if row_w is None else (G * row_w[:, None]).sum(1)


def _maps(model):
    """The model family's ``(linearize, jvp, vjp)``: Qwen3 here, Gemma 4 in ``neuron_lens.gemma4``."""
    from neuron_lens import gemma4
    if gemma4.is_gemma4(model):
        return gemma4.linearize, gemma4.jvp, gemma4.vjp
    return _linearize, _jvp, _vjp


def decoder_stack(model):
    """The decoder stack (``.layers``, ``.embed_tokens``, ``.config``) of a loaded model: a
    causal LM's ``.model``, or a multimodal one's ``.model.language_model``."""
    inner = getattr(model, "model", model)
    return getattr(inner, "language_model", inner)


@functools.cache
def _compiled(step):
    return torch.compile(step, dynamic=False)


def _step(step, device: torch.device):
    """``step`` as it runs on ``device``: compiled on CUDA (1.6× on an H100, where unfused it
    spent half its time between the matrix products), eager elsewhere. Shapes stay fixed, so
    full-length documents keep kernels specialized to them: after a shorter document, a
    shape-generic compile made them 10% slower. Past dynamo's recompile limit (8 shapes),
    rarer document lengths run eagerly."""
    return _compiled(step) if device.type == "cuda" else step


def _seed(b: int, N: int, d: int, start: int, device) -> torch.Tensor:
    """``[b, N, d]``: basis direction ``start + i`` at every valid position, in copy ``i``."""
    x = torch.zeros(b, N, d, device=device)
    i = torch.arange(b, device=device)
    x[i, :, start + i] = 1
    return x


@torch.no_grad()
def s_lens_doc(model, ids: torch.Tensor, acc: Accumulator, half: int, *,
               dim_batch: int = DIM_BATCH, basis: torch.Tensor | None = None,
               source_rows: torch.Tensor | None = None, target_rows: torch.Tensor | None = None,
               **linearize_kw) -> None:
    """Add one document's S-lens estimate to ``acc`` (targets ``mid`` and ``out``, as present):
    tangents seeded at the embeddings, pushed up through every layer.

    For documents with image or audio tokens: ``source_rows`` and ``target_rows`` (``[N]`` bool
    over the valid positions) restrict the seeds and the mean over targets to those positions;
    ``basis`` (``[k, d]``, orthonormal rows) seeds only those embedding directions, so the sums
    hold ``K_L Bᵀ`` (``acc`` made with ``d_in=k``): exact for inputs in the span of ``B`` (audio
    frames reach a 640-dimensional subspace); ``linearize_kw`` goes to the model's ``linearize``."""
    linearize, jvp, _ = _maps(model)
    maps = linearize(model, ids, **linearize_kw)
    N, d = maps[0].a1.shape
    dev = maps[0].a1.device
    jvp = _step(jvp, dev)
    kinds = {t for t, _ in acc.sums}
    k = d if basis is None else basis.shape[0]
    src = None if source_rows is None else source_rows.to(dev).float()[None, :, None]
    w = None if target_rows is None else (target_rows.to(dev).float() / target_rows.sum())[None, :, None]
    for start in range(0, k, dim_batch):
        b = min(dim_batch, k - start)
        if basis is None:
            X = _seed(b, N, d, start, dev)
        else:
            X = basis[start:start + b].to(dev, torch.float32)[:, None].expand(b, N, d).contiguous()
        if src is not None:
            X = X * src
        for layer, m in enumerate(maps):
            X, mid = jvp(m, X)
            for kind, est in (("mid", mid), ("out", X)):
                if kind in kinds:
                    e = est.mean(1) if w is None else (est * w).sum(1)
                    s = acc.sums[(kind, half)]
                    s[layer, :, start:start + b] += e.mT.to(s.device)
    acc.n_docs[half] += 1


@torch.no_grad()
def r_lens_doc(model, ids: torch.Tensor, acc: Accumulator, half: int, *,
               dim_batch: int = DIM_BATCH, seed_rows: torch.Tensor | None = None,
               read_rows: torch.Tensor | None = None, **linearize_kw) -> None:
    """Add one document's R-lens estimate to ``acc`` (sources ``out``, the outputs of layers
    ``0 .. n−2``, and ``mid``, the MLP inputs of layers ``0 .. n−1``, as present; the target is
    the output of layer ``n−1``): cotangents seeded at the target, pulled down through the
    layers.

    For documents with image or audio tokens: ``seed_rows`` (``[N]`` bool over the valid
    positions) restricts the target positions ``t'`` seeded, and ``read_rows`` the source
    positions ``t`` averaged; ``linearize_kw`` goes to the model's ``linearize``."""
    linearize, _, vjp = _maps(model)
    maps = linearize(model, ids, **linearize_kw)
    N, d = maps[0].a1.shape
    dev = maps[0].a1.device
    vjp = _step(vjp, dev)
    kinds = {t for t, _ in acc.sums}
    last = 0 if "mid" in kinds else 1  # layer 0's map matters only for its mid
    seeds = None if seed_rows is None else seed_rows.to(dev).float()[None, :, None]
    row_w = None if read_rows is None else read_rows.to(dev).float() / read_rows.sum()
    for start in range(0, d, dim_batch):
        b = min(dim_batch, d - start)
        G = _seed(b, N, d, start, dev)
        if seeds is not None:
            G = G * seeds
        for layer in range(len(maps) - 1, last - 1, -1):
            G, mid, est = vjp(maps[layer], G, None, row_w)
            if "mid" in kinds:
                s = acc.sums[("mid", half)]
                s[layer, start:start + b] += mid.to(s.device)
            if layer > 0 and "out" in kinds:
                s = acc.sums[("out", half)]
                s[layer - 1, start:start + b] += est.to(s.device)
    acc.n_docs[half] += 1


@torch.no_grad()
def input_reads_doc(model, ids: torch.Tensor, covectors: torch.Tensor, layers: list[int], *,
                    seed_rows: torch.Tensor | None = None, source_rows: torch.Tensor | None = None,
                    **linearize_kw) -> torch.Tensor:
    """One document's S-lens reads of chosen covectors, without fitting the S-lens: row ``i``
    is ``c_i · K_{L_i}`` (``[b, d]``, as the S-lens's ``mid`` target reads it), for covector
    ``c_i`` (``covectors[i]``, ``[b, d]``) on the MLP input of layer ``L_i`` (``layers[i]``).
    Each covector is seeded at every valid position of its layer's ``mid``, over ``N`` (the
    mean over ``t``), pulled down to the embeddings with the R-lens's maps (the S-lens's
    transpose), and summed over the source positions. Costs one R-lens block per ``b``
    covectors, whatever their layers.

    ``seed_rows`` and ``source_rows`` (``[N]`` bool over the valid positions) restrict the
    positions read and summed (say, an image's tokens); ``linearize_kw`` goes to the model's
    ``linearize`` (Gemma 4: ``embeds``, ``block``)."""
    linearize, _, vjp = _maps(model)
    maps = linearize(model, ids, **linearize_kw)
    N, d = maps[0].a1.shape
    dev = maps[0].a1.device
    vjp = _step(vjp, dev)
    c = covectors.to(dev, torch.float32)
    at = torch.as_tensor(layers, device=dev)
    rows = torch.ones(N, device=dev) if seed_rows is None else seed_rows.to(dev).float()
    G = torch.zeros(len(c), N, d, device=dev)
    for layer in range(int(at.max()), -1, -1):
        seed = (c * (at == layer)[:, None])[:, None] * (rows / rows.sum())[None, :, None]
        G, _, _ = vjp(maps[layer], G, seed)
    return G.sum(1) if source_rows is None else G[:, source_rows.to(dev)].sum(1)


def target_layers(lens: str, target: str, n: int) -> int:
    """Layers a target holds: the R-lens's ``out`` stops below the last layer, its target."""
    return n - 1 if (lens, target) == ("R", "out") else n


def lens_doc(lens: str, *args, **kwargs) -> None:
    {"R": r_lens_doc, "S": s_lens_doc}[lens](*args, **kwargs)
