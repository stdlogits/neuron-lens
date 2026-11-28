"""Weight-only lens readout of neuron write vectors.

Neuron ``n`` at layer ``L`` adds ``a * W_down[:, n]`` to the residual, and ``a`` takes both
signs (SwiGLU's ``up`` factor is linear), so each neuron has two poles, ``+w`` and ``-w`` (one for an
ungated MLP, ``arch.n_poles``). A lens reads a pole as ``softmax(W_U norm(J_L w))`` (``J_L`` = identity for the
logit lens), with no forward pass. The final norm makes the read independent of ``|w|``, so the probabilities
describe a direction, not an effect size. ``centre``: the final norm is a LayerNorm (``arch.centred``), which
centres each column first; its bias, the same for every column, is left out.
"""
import torch

WHOLE = 2**32  # entries of the largest unembedding product ``layer_reads`` keeps whole (8 GB in fp16)


def rms_norm_columns(t: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6, centre: bool = False) -> torch.Tensor:
    """The model's final norm applied to each column of ``t`` ``[d, n]``: an RMSNorm, or with ``centre`` a LayerNorm."""
    if centre:
        t = t - t.mean(0, keepdim=True)
    return t * torch.rsqrt(t.pow(2).mean(0, keepdim=True) + eps) * gamma[:, None]


def transport(vectors: torch.Tensor, jacobian: torch.Tensor | None) -> torch.Tensor:
    """``J_L @ vectors`` in fp32; ``None`` is the logit lens."""
    vectors = vectors.float()
    return vectors if jacobian is None else jacobian.to(vectors.device, torch.float32) @ vectors


def read(
    vectors: torch.Tensor,
    unembed: torch.Tensor,
    gamma: torch.Tensor,
    jacobian: torch.Tensor | None = None,
    *,
    k: int = 10,
    chunk: int = 1024,
    centre: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``k`` lens probabilities and token ids per column of ``vectors`` ``[d, n]``.

    The unembedding matmul runs in ``unembed``'s dtype, chunked over columns so the
    ``[chunk, vocab]`` logits stay bounded and row-major for fast contiguous ``topk``.
    Returns CPU ``([n, k] probs, [n, k] ids)``.
    """
    t = rms_norm_columns(transport(vectors, jacobian), gamma.to(vectors.device, torch.float32), centre=centre)
    probs, ids = [], []
    for s in range(0, t.shape[1], chunk):
        logits = (t[:, s:s + chunk].T.to(unembed.dtype) @ unembed.T).float()
        top, idx = logits.topk(k, dim=1)
        probs.append((top - logits.logsumexp(1, keepdim=True)).exp().cpu())
        ids.append(idx.cpu())
    return torch.cat(probs), torch.cat(ids)


def _linear_stats(t: torch.Tensor, mat: torch.Tensor, vchunk: int = 16384, floor: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row mean and std of ``mat @ t`` (``mat`` ``[V, d]``, ``t`` ``[d, n]``) using the ``[d, d]`` centred Gram
    matrix of ``t``, avoiding an extra ``[V, n]`` matrix product and avoiding catastrophic cancellation."""
    n = t.shape[1]
    tf = t.float()
    t_mean = tf.mean(1)
    tc = tf - t_mean[:, None]
    G = (tc @ tc.T) / n  # [d, d]
    V = mat.shape[0]
    mean = torch.empty(V, device=mat.device, dtype=torch.float32)
    var = torch.empty(V, device=mat.device, dtype=torch.float32)
    for s in range(0, V, vchunk):
        u = mat[s:s + vchunk].float()
        mean[s:s + vchunk] = u @ t_mean
        var[s:s + vchunk] = ((u @ G) * u).sum(-1)
    return mean, var.clamp_min(floor).sqrt()


def layer_logit_stats(
    vectors: torch.Tensor,
    unembed: torch.Tensor,
    gamma: torch.Tensor,
    jacobian: torch.Tensor | None = None,
    *,
    chunk: int = 1024,
    centre: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token mean and std of the lens logit across a layer's columns, for z-scored
    (layer-contrastive) reads that remove what every neuron of the layer writes."""
    t = rms_norm_columns(transport(vectors, jacobian), gamma.to(vectors.device, torch.float32), centre=centre)
    return _linear_stats(t, unembed)


def vocab_moments(
    vectors: torch.Tensor,
    unembed: torch.Tensor,
    gamma: torch.Tensor,
    *,
    chunk: int = 2048,
    softcap: float | None = None,
    centre: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column skew and kurtosis of ``W_U · norm(w)`` over the vocabulary (no z-stats or topk)."""
    t = rms_norm_columns(vectors.float(), gamma.to(vectors.device, torch.float32), centre=centre)
    skew, kurt = [], []
    for s in range(0, t.shape[1], chunk):
        lg = (t[:, s:s + chunk].T.to(unembed.dtype) @ unembed.T).float()
        if softcap:
            lg = torch.tanh(lg / softcap) * softcap
        c = lg - lg.mean(1, keepdim=True)
        var = c.pow(2).mean(1)
        skew.append((c.pow(3).mean(1) / var.pow(1.5)).cpu())
        kurt.append((c.pow(4).mean(1) / var.pow(2)).cpu())
    return torch.cat(skew), torch.cat(kurt)


def read_z(
    vectors: torch.Tensor,
    unembed: torch.Tensor,
    gamma: torch.Tensor,
    jacobian: torch.Tensor | None,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    k: int = 8,
    centre: bool = False,
) -> torch.Tensor:
    """Top-``k`` token ids per column by lens logit z-scored against ``layer_logit_stats``."""
    t = rms_norm_columns(transport(vectors, jacobian), gamma.to(vectors.device, torch.float32), centre=centre)
    logits = (t.T.to(unembed.dtype) @ unembed.T).float()
    return ((logits - mean) / std).topk(k, dim=1).indices.cpu()


def layer_reads(
    vectors: torch.Tensor,
    unembed: torch.Tensor,
    gamma: torch.Tensor,
    jacobian: torch.Tensor | None = None,
    *,
    signs: tuple[int, ...] = (1, -1),
    k: int = 10,
    kz: int = 8,
    chunk: int = 2048,
    softcap: float | None = None,
    renorm: bool = True,
    centre: bool = False,
) -> dict:
    """Every column's read on each sign (pole), from row-major ``[chunk, vocab]`` unembedding blocks so
    ``topk``, ``logsumexp`` and moments reduce along the contiguous dimension.
    With ``softcap`` (Gemma), the logits are the model's: ``softcap · tanh(logits / softcap)``.
    ``renorm=False`` keeps each column's size (columns already on the final norm's scale, before its weight, and
    centred where it centres).

    Per sign ``s``: the top-``k`` probabilities and ids of ``softmax(s · logits)`` and the
    top-``kz`` ids and values by layer-contrastive z (each token's logit z-scored across the
    columns, as in ``read_z``; the ``−`` pole's z is minus the ``+`` pole's). Also each column's
    skew and kurtosis of its logits over the vocabulary, for sign ``+`` (a pole's skew is ``s``
    times it). Returns CPU tensors ``{"ids": {s: [n, k]}, "probs", "zids", "z", "skew": [n], "kurt"}``.
    """
    gamma = gamma.to(vectors.device, torch.float32)
    t = rms_norm_columns(transport(vectors, jacobian), gamma, centre=centre) if renorm else \
        transport(vectors, jacobian) * gamma[:, None]
    n = t.shape[1]
    if softcap is None:
        # Linear logits: exact per-token mean/std via the [d, d] centred Gram matrix of t, so each column chunk
        # runs t.T @ unembed.T only once and no whole [n, V] logits matrix is materialized.
        mean, std = _linear_stats(t, unembed)
        def block(s: int) -> torch.Tensor:
            return (t[:, s:s + chunk].T.to(unembed.dtype) @ unembed.T).float()
    else:
        # Softcap is nonlinear: keep the product whole where it fits WHOLE, else chunk twice.
        logits = t.T.to(unembed.dtype) @ unembed.T if unembed.shape[0] * n <= WHOLE else None
        if logits is not None:
            for s in range(0, n, chunk):
                logits[s:s + chunk] = (torch.tanh(logits[s:s + chunk].float() / softcap) * softcap).to(logits.dtype)

        def block(s: int) -> torch.Tensor:
            if logits is not None:
                return logits[s:s + chunk].float()
            lg = t[:, s:s + chunk].T.to(unembed.dtype) @ unembed.T
            return (torch.tanh(lg.float() / softcap) * softcap).to(lg.dtype).float()
        total = torch.zeros(unembed.shape[0], device=unembed.device)
        sq = torch.zeros_like(total)
        for s in range(0, n, chunk):
            lg = block(s)
            total += lg.sum(0)
            sq += lg.pow(2).sum(0)
        mean = total / n
        std = (sq / n - mean.pow(2)).clamp_min(1e-8).sqrt()
    out = {"ids": {s: [] for s in signs}, "probs": {s: [] for s in signs},
           "zids": {s: [] for s in signs}, "z": {s: [] for s in signs}, "skew": [], "kurt": []}
    for s in range(0, n, chunk):
        lg = block(s)  # [chunk, V] contiguous
        z = (lg - mean) / std
        for sign in signs:
            slg = lg if sign == 1 else -lg
            top, idx = slg.topk(k, dim=1)
            out["probs"][sign].append((top - slg.logsumexp(1, keepdim=True)).exp().cpu())
            out["ids"][sign].append(idx.cpu())
            sz = z if sign == 1 else -z
            zt = sz.topk(kz, dim=1)
            out["zids"][sign].append(zt.indices.cpu())
            out["z"][sign].append(zt.values.cpu())
        c = lg - lg.mean(1, keepdim=True)
        var = c.pow(2).mean(1)
        out["skew"].append((c.pow(3).mean(1) / var.pow(1.5)).cpu())
        out["kurt"].append((c.pow(4).mean(1) / var.pow(2)).cpu())
    return {key: {s: torch.cat(v[s]) for s in signs} if isinstance(v, dict) else torch.cat(v)
            for key, v in out.items()}


def s_tops(cov: torch.Tensor, emb: torch.Tensor, signs: tuple[int, ...], k: int, chunk: int = 2048) -> dict:
    """S-lens reads of unit covector rows ``cov`` ``[n, d]`` over the kept embeddings: per sign, the
    top raw scores and the top layer-contrastive z (each token's score z-scored across the rows)."""
    n = len(cov)
    # Exact per-token mean/std from the [d, d] centred Gram of cov, halving the [n, V] GEMMs.
    mean, std = _linear_stats(cov.T, emb, floor=1e-12)
    out = {(kind, sign): [] for kind in ("raw", "z") for sign in signs}
    for s in range(0, n, chunk):
        sc = (cov[s:s + chunk].half() @ emb.T).float()
        for sign in signs:
            out["raw", sign].append((sign * sc).topk(k, dim=1))
            out["z", sign].append((sign * (sc - mean) / std).topk(k, dim=1))
    return {key: (torch.cat([x.indices for x in v]).cpu(), torch.cat([x.values for x in v]).cpu()) for key, v in out.items()}
