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

``mid`` is the residual after the layer's mixer (the input of the MLP's norm); ``out`` is the
layer's output.

The derivative is RelP's (``neuron_lens.rules`` states the rules and applies them to the
model for autograd, which the tests use as the reference). Under the rules each decoder layer
is, on a given document, a fixed linear map of the residual; ``neuron_lens.layers`` writes those
maps out by component (norms, softmax attention, Gated DeltaNet, gated and sparse MLPs), runs the
primal once per document and keeps each layer's constants. The S-lens pushes tangents up through
the maps, the R-lens pulls cotangents down through their transposes, ``dim_batch`` basis
directions at a time, so each layer is a few matrix products over ``dim_batch × positions`` rows,
written out by hand (no autograd). On CUDA the layer steps are compiled, which fuses the
elementwise work between the products.

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

import torch

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


def _maps(model):
    """``(linearize, jvp, vjp)`` for a decoder stack (``neuron_lens.layers``, every family)."""
    from neuron_lens import layers
    return layers.linearize, layers.jvp, layers.vjp


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
    N, d = maps[0].n1.a.shape
    dev = maps[0].n1.a.device
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
    N, d = maps[0].n1.a.shape
    dev = maps[0].n1.a.device
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
    N, d = maps[0].n1.a.shape
    dev = maps[0].n1.a.device
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
