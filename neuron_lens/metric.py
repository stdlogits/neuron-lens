"""The residual's metric: how a covector (what a unit reads) becomes a vector (what a lens transports), and back.

Every layer reads the residual through an RMSNorm, ``y = γ ⊙ x / rms(x)``. Rescaling residual coordinate ``i`` by ``k``,
with every norm's ``γ_i`` divided by ``k`` and every write into the residual multiplied by it, leaves the model nearly
unchanged (only the rms sees it), so the raw basis carries a unit per coordinate: Gemma 4's coordinate scales span
80-1,600× (99th over 1st percentile), Qwen3-8B's 3-6×. A read that pairs a covector with a vector needs no metric
(``read_S``, ``write_R``, OV, edges); one that turns a covector into a vector (a neuron's read row, or the R-lens row
of a token, fed to a lens or a projection that takes vectors) does, and only a
per-coordinate ``s²`` that moves with the units keeps the read free of them: ``v = s² ⊙ c``. Here ``s² = σ / |γ|``,
with ``σ`` the residual's standard deviation per coordinate where a norm reads it (``x / rms``, over the lens corpus,
``02_fit.py --lens scale``) and ``γ`` that norm's weight: in the norm's frame, the read row times the residual's
scale (``σ ⊙ w`` for a neuron's row ``w``), between the row itself (``w / γ``, which needs a floor where ``γ ≈ 0``) and
the diagonal Haufe pattern (``σ² γ ⊙ w``). Findings §13 compares them. A length the model takes itself stays
Euclidean: every RMSNorm divides by the raw basis's rms (a QK footprint's size, a write's energy at the final norm).

A read that sets a vector against vectors, a write (or the residual, or a head's output) against the S-lens's token
footprints ``K ē``, asks which input token would put it there: an inverse problem, which the transpose (``ē · Kᵀ w``,
Euclidean) answers only where ``K`` doesn't stretch. ``polar(K)`` takes the midpoint between the transpose and the
inverse, ``ē · Kᵀ (K Kᵀ)^-½ w = (U ē) · w`` with ``K = (K Kᵀ)^½ U``: the footprints with the lens's stretch removed,
by its nearest rotation ``U``. Of the powers ``(K Kᵀ)^-p``, ``p = ½`` reads best on both models (findings §13).

Points, each the input of a norm that reads the residual: ``in`` ``[L + 1, d]``, layer ``L``'s input (its input norm;
``in[L]`` for the last index is the final norm's input, the last layer's output), so layer ``L``'s output is
``in[L + 1]``; ``mid`` ``[L, d]``, layer ``L``'s MLP input (the norm that feeds the MLP).
"""
import functools

import torch

from neuron_lens import arch
from neuron_lens.paths import FITS
from neuron_lens.weights import tensors

TINY = 1e-12
FLOOR = 0.01  # polar: K Kᵀ's eigenvalues floored at this share of their mean (Gemma 4's S-lenses have a near-null bulk)


def fit_name(model_id: str) -> str:
    return f"scale-{arch.stem(model_id)}-L{arch.text_config(model_id).num_hidden_layers}-dclm"


class Sums:
    """Even- and odd-document sums of ``x / rms(x)`` and its square per coordinate at every point, and position counts."""

    def __init__(self, n_layers: int, d: int, device="cpu"):
        z = lambda *s: torch.zeros(*s, device=device)
        self.sums = {"in": z(2, n_layers + 1, d), "in_sq": z(2, n_layers + 1, d), "mid": z(2, n_layers, d),
                     "mid_sq": z(2, n_layers, d), "n": z(2)}
        self.n_docs = [0, 0]

    @classmethod
    def from_state(cls, state: dict, device="cpu"):
        self = cls.__new__(cls)
        self.sums = {k: v.to(device) for k, v in state["sums"].items()}
        self.n_docs = list(state["n_docs"])
        return self

    def add(self, point: str, layer: int, x: torch.Tensor, half: int) -> None:
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True).clamp_min(TINY))
        self.sums[point][half, layer] += v.sum(0)
        self.sums[point + "_sq"][half, layer] += v.pow(2).sum(0)

    def sigma(self, halves=(0, 1)) -> dict:
        n = self.sums["n"][list(halves)].sum()
        out = {}
        for p in ("in", "mid"):
            m = self.sums[p][list(halves)].sum(0) / n
            out[p] = (self.sums[p + "_sq"][list(halves)].sum(0) / n - m.pow(2)).clamp_min(0).sqrt()
        return out

    def split_half(self, point: str) -> list[float]:
        """Per layer, ``‖σ_even − σ_odd‖ / ‖σ‖``."""
        a, b, m = self.sigma((0,))[point], self.sigma((1,))[point], self.sigma()[point]
        return ((a - b).norm(dim=1) / m.norm(dim=1)).tolist()


@torch.no_grad()
def scale_doc(stack, ids: torch.Tensor, acc: Sums, half: int, first: int) -> None:
    """One text document's residual at every point into ``acc``, at positions ``[first, T − 1)``, where the
    lenses seed (``transport.SKIP_FIRST``)."""
    dev = stack.embed_tokens.weight.device
    hooks, n = [], len(stack.layers)
    keep = slice(first, len(ids) - 1)

    def at(point, layer):
        def f(module, args):
            acc.add(point, layer, args[0][0, keep], half)
        return f

    for layer, lay in enumerate(stack.layers):
        hooks.append(lay.input_layernorm.register_forward_pre_hook(at("in", layer)))
        hooks.append(arch.mid_norm(lay).register_forward_pre_hook(at("mid", layer)))
    hooks.append(stack.norm.register_forward_pre_hook(at("in", n)))
    try:
        stack(input_ids=ids[None].to(dev))
    finally:
        for h in hooks:
            h.remove()
    acc.sums["n"][half] += len(ids) - 1 - first
    acc.n_docs[half] += 1


@functools.cache
def scales(model_id: str) -> dict:
    """``{"in": [L + 1, d], "mid": [L, d]}``, the residual's per-coordinate σ (fp32, CPU): from the scale fit in
    ``data/fits/`` when it is there, else the release (``stdlogits/rs-lenses``, ``<model>/scale.pt``)."""
    path = FITS / f"{fit_name(model_id)}.pt"
    if path.exists():
        return Sums.from_state(torch.load(path, weights_only=False, map_location="cpu")["state"]).sigma()
    from huggingface_hub import hf_hub_download
    x = torch.load(hf_hub_download("stdlogits/rs-lenses", f"{arch.stem(model_id)}/scale.pt"), weights_only=True,
                   map_location="cpu")
    return {"in": x["in"].float(), "mid": x["mid"].float()}


def norm_weight(model_id: str, point: str, layer: int) -> torch.Tensor:
    """The weight of the norm that reads ``point`` at ``layer`` (``in``: the layer's input norm, or the final norm
    past the last layer; ``mid``: the norm that feeds the MLP)."""
    n = arch.text_config(model_id).num_hidden_layers
    if point == "in" and layer == n:
        name = arch.names(model_id)["norm"]
    elif point == "in":
        name = arch.layout(model_id)[layer].prefix + "input_layernorm.weight"
    else:
        from neuron_lens.neurons import mlp_names
        name = mlp_names(model_id, layer)["norm"]
    return tensors(model_id, [name])[name].float()


def s2(model_id: str, point: str, layer: int, dev="cpu") -> torch.Tensor:
    """``[d]``: ``σ / |γ|`` at ``point`` (``in``, ``mid``, or ``out``, layer ``layer``'s output, ``in[layer + 1]``)."""
    if point == "out":
        point, layer = "in", layer + 1
    sigma = scales(model_id)[point][layer]
    return (sigma / norm_weight(model_id, point, layer).abs().clamp_min(TINY)).to(dev)


def vector(c: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """Covectors ``[n, d]`` to vectors."""
    return c * s2


def covector(v: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """Vectors ``[n, d]`` to covectors."""
    return v / s2


def cov_norm(c: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """``[n, 1]``: each covector row's length in the metric, ``√(c · s² c)``."""
    return (c.pow(2) * s2).sum(-1, keepdim=True).sqrt()


def vec_norm(v: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """``[n, 1]``: each vector row's length in the metric, ``√(v · v / s²)``."""
    return (v.pow(2) / s2).sum(-1, keepdim=True).sqrt()


def polar(K: torch.Tensor) -> torch.Tensor:
    """``[d, d]`` (on ``K``'s device): ``(K Kᵀ)^-½`` for an S-lens ``K`` ``[d, d_in]``, its eigenvalues floored at
    ``FLOOR`` of their mean, at unit mean diagonal. Vectors ``w`` ``[n, d]`` read against the footprints as
    ``w @ polar(K) @ K`` (covectors on the embeddings) rather than ``w @ K``."""
    kap, U = torch.linalg.eigh((K.float() @ K.float().T).cpu().double())
    kap = kap.clamp_min(0)
    M = (U * (kap + FLOOR * kap.mean()).rsqrt()) @ U.T
    return (M / M.diagonal().mean()).float().to(K.device)
