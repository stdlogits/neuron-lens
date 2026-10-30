"""Weight-only lens readout of neuron write vectors.

Neuron ``n`` at layer ``L`` adds ``a * W_down[:, n]`` to the residual, and ``a`` takes both
signs (SwiGLU's ``up`` factor is linear), so each neuron has two poles, ``+w`` and ``-w``. A
lens reads a pole as ``softmax(W_U norm(J_L w))`` (``J_L`` = identity for the logit lens),
with no forward pass. RMSNorm makes the read independent of ``|w|``, so the probabilities
describe a direction, not an effect size.
"""
import torch


def rms_norm_columns(t: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """The model's final RMSNorm applied to each column of ``t`` ``[d, n]``."""
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``k`` lens probabilities and token ids per column of ``vectors`` ``[d, n]``.

    The unembedding matmul runs in ``unembed``'s dtype, chunked over columns so the
    ``[vocab, chunk]`` logits stay bounded. Returns CPU ``([n, k] probs, [n, k] ids)``.
    """
    t = rms_norm_columns(transport(vectors, jacobian), gamma.to(vectors.device, torch.float32))
    probs, ids = [], []
    for s in range(0, t.shape[1], chunk):
        logits = (unembed @ t[:, s:s + chunk].to(unembed.dtype)).float()
        top, idx = logits.topk(k, dim=0)
        probs.append((top - logits.logsumexp(0, keepdim=True)).exp().T.cpu())
        ids.append(idx.T.cpu())
    return torch.cat(probs), torch.cat(ids)


def layer_logit_stats(
    vectors: torch.Tensor,
    unembed: torch.Tensor,
    gamma: torch.Tensor,
    jacobian: torch.Tensor | None = None,
    *,
    chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token mean and std of the lens logit across a layer's columns, for z-scored
    (layer-contrastive) reads that remove what every neuron of the layer writes."""
    t = rms_norm_columns(transport(vectors, jacobian), gamma.to(vectors.device, torch.float32))
    total = torch.zeros(unembed.shape[0], device=unembed.device)
    sq = torch.zeros_like(total)
    for s in range(0, t.shape[1], chunk):
        logits = (unembed @ t[:, s:s + chunk].to(unembed.dtype)).float()
        total += logits.sum(1)
        sq += logits.pow(2).sum(1)
    n = t.shape[1]
    mean = total / n
    return mean, (sq / n - mean.pow(2)).clamp_min(1e-8).sqrt()


def read_z(
    vectors: torch.Tensor,
    unembed: torch.Tensor,
    gamma: torch.Tensor,
    jacobian: torch.Tensor | None,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    k: int = 8,
) -> torch.Tensor:
    """Top-``k`` token ids per column by lens logit z-scored against ``layer_logit_stats``."""
    t = rms_norm_columns(transport(vectors, jacobian), gamma.to(vectors.device, torch.float32))
    logits = (unembed @ t.to(unembed.dtype)).float()
    return ((logits - mean[:, None]) / std[:, None]).topk(k, dim=0).indices.T.cpu()


def layer_reads(
    vectors: torch.Tensor,
    unembed: torch.Tensor,
    gamma: torch.Tensor,
    jacobian: torch.Tensor | None = None,
    *,
    signs: tuple[int, ...] = (1, -1),
    k: int = 10,
    kz: int = 8,
    chunk: int = 1024,
    softcap: float | None = None,
) -> dict:
    """Every column's read on each sign (pole), from one unembedding product kept whole.
    With ``softcap`` (Gemma), the logits are the model's: ``softcap · tanh(logits / softcap)``.

    Per sign ``s``: the top-``k`` probabilities and ids of ``softmax(s · logits)`` and the
    top-``kz`` ids and values by layer-contrastive z (each token's logit z-scored across the
    columns, as in ``read_z``; the ``−`` pole's z is minus the ``+`` pole's). Also each column's
    skew and kurtosis of its logits over the vocabulary, for sign ``+`` (a pole's skew is ``s``
    times it). Returns CPU tensors ``{"ids": {s: [n, k]}, "probs", "zids", "z", "skew": [n], "kurt"}``.
    """
    t = rms_norm_columns(transport(vectors, jacobian), gamma.to(vectors.device, torch.float32))
    logits = unembed @ t.to(unembed.dtype)
    n = logits.shape[1]
    if softcap:  # in place, a chunk of columns at a time (a whole fp32 copy is 7 GB for Gemma 4)
        for s in range(0, n, chunk):
            logits[:, s:s + chunk] = (torch.tanh(logits[:, s:s + chunk].float() / softcap) * softcap).to(logits.dtype)
    total = torch.zeros(logits.shape[0], device=logits.device)
    sq = torch.zeros_like(total)
    for s in range(0, n, chunk):
        lg = logits[:, s:s + chunk].float()
        total += lg.sum(1)
        sq += lg.pow(2).sum(1)
    mean = total / n
    std = (sq / n - mean.pow(2)).clamp_min(1e-8).sqrt()
    out = {"ids": {s: [] for s in signs}, "probs": {s: [] for s in signs},
           "zids": {s: [] for s in signs}, "z": {s: [] for s in signs}, "skew": [], "kurt": []}
    for s in range(0, n, chunk):
        lg = logits[:, s:s + chunk].float()
        z = (lg - mean[:, None]) / std[:, None]
        for sign in signs:
            top, idx = (sign * lg).topk(k, dim=0)
            out["probs"][sign].append((top - (sign * lg).logsumexp(0, keepdim=True)).exp().T.cpu())
            out["ids"][sign].append(idx.T.cpu())
            zt = (sign * z).topk(kz, dim=0)
            out["zids"][sign].append(zt.indices.T.cpu())
            out["z"][sign].append(zt.values.T.cpu())
        c = lg - lg.mean(0, keepdim=True)
        var = c.pow(2).mean(0)
        out["skew"].append((c.pow(3).mean(0) / var.pow(1.5)).cpu())
        out["kurt"].append((c.pow(4).mean(0) / var.pow(2)).cpu())
    return {key: {s: torch.cat(v[s]) for s in signs} if isinstance(v, dict) else torch.cat(v)
            for key, v in out.items()}
