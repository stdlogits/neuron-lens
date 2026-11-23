"""A sparse MLP's neurons and experts (Qwen3.5-35B-A3B, Qwen3-30B-A3B, OLMoE): their activations, as the catalog
stages (``06``, ``07``, ``17``) collect them, and their rows and writes, without the dense matrices a stage would build
for a dense MLP.

A sparse layer's neurons are its experts' side by side, then its shared expert's (``neurons.n_neurons``): neuron
``e · I + i`` is neuron ``i`` of expert ``e`` (``I`` an expert's width), ``E · I + j`` the shared expert's ``j``. A neuron's
activation is the coefficient of its write in the MLP's output, so that the output is ``Σ_n a_n w_n``, as for a dense
MLP's neurons:

- a routed neuron's is ``ρ · act(g) · u`` at a position its expert is chosen for, ``ρ`` the routing weight there (the
  top ``k``'s softmax renormalised over them, Qwen3-MoE and Qwen3.5; the softmax's own value, OLMoE's, whose router
  doesn't renormalise), and 0 at every other position;
- a shared neuron's is ``σ(s) · act(g) · u`` at every position, ``σ(s)`` the shared expert's sigmoid gate (1 without one).

A routed pole fires only where its expert is chosen, so its lists of firings (top documents, drivers in ``07`` aside)
hold routed positions alone: a document whose positions never route to the expert has no extreme for the pole, and a
list with fewer than ``TOP`` such documents is padded (``−1``, NaN), as every shorter list is. Its rates and quantiles
count every position, the unrouted ones at 0.

``Tap`` hooks a decoder layer's sparse MLP and hands each forward pass's activations over as a ``Routed`` (the routed
pairs, and the shared expert's dense block), recomputing the experts' gate and up projections at the pairs it is asked
for (the experts' forward keeps them to itself); the routing is the model's own, read from the experts' inputs.
``Rows`` gathers per-neuron rows of a layer's expert weights (gate, up, write) by neuron id, without building them.
"""
from typing import NamedTuple

import torch
import torch.nn.functional as F

from neuron_lens import anatomy
from neuron_lens.rules import act_factor


class Routed(NamedTuple):
    """One forward pass of a sparse MLP over ``n`` rows (its flattened input's positions): ``M`` routed (row, expert)
    pairs, sorted by expert, and the shared expert's activations at every row. fp32."""
    rows: torch.Tensor  # [M]: each pair's row
    expert: torch.Tensor  # [M]
    weight: torch.Tensor  # [M]: the routing weight ρ
    gate: torch.Tensor  # [M, I]: the gate's pre-activation g
    act: torch.Tensor  # [M, I]: ρ · act(g) · u
    shared: torch.Tensor | None  # [n, Is]: σ(s) · act(g) · u
    shared_gate: torch.Tensor | None  # [n, Is]: the shared expert's g
    shared_weight: torch.Tensor | None  # [n]: σ(s)
    n: int
    E: int
    I: int

    @property
    def width(self) -> int:
        return self.E * self.I + (0 if self.shared is None else self.shared.shape[1])

    def cols(self) -> torch.Tensor:
        """``[M, I]``: each pair's neuron ids, ``e · I + i``."""
        return self.expert[:, None] * self.I + torch.arange(self.I, device=self.act.device)

    def dense(self, keep: torch.Tensor | None = None, fill: float = 0.0) -> torch.Tensor:
        """``[rows, f]``: every neuron's activation at the rows ``keep`` (a mask over the ``n`` rows; all by default),
        ``fill`` where a routed neuron's expert isn't chosen (0, its activation; NaN to tell the unrouted apart)."""
        keep = torch.ones(self.n, dtype=torch.bool, device=self.act.device) if keep is None else keep
        at = torch.full((self.n,), -1, dtype=torch.long, device=self.act.device)
        at[keep] = torch.arange(int(keep.sum()), device=self.act.device)
        out = torch.full((int(keep.sum()), self.width), fill, device=self.act.device)
        sel = at[self.rows] >= 0
        out.view(-1).index_copy_(0, (at[self.rows[sel]][:, None] * self.width + self.cols()[sel]).flatten(),
                                 self.act[sel].flatten())
        if self.shared is not None:
            out[:, self.E * self.I:] = self.shared[keep]
        return out


def routed(mlp, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, rows: torch.Tensor | None = None,
           shared: tuple | None = None) -> Routed:
    """The activations of a sparse MLP's neurons from its experts' inputs: ``x`` ``[n, d]``, the chosen experts ``idx``
    and their weights ``w`` ``[n, k]``, the gate and up projections recomputed expert by expert in ``x``'s dtype (as the
    model computes them); with ``rows`` (a mask over the ``n`` rows), the pairs at those rows alone. ``shared``: the
    shared expert's ``(g, act(g)·u, s)`` at every row (``s`` its gate's logit, ``None`` without one)."""
    ex = mlp.experts
    E, I, n, k = ex.num_experts, ex.intermediate_dim, len(x), idx.shape[1]
    flat = idx.flatten()
    pair = torch.arange(n * k, device=x.device)
    if rows is not None:
        sel = rows.repeat_interleave(k)
        flat, pair = flat[sel], pair[sel]
    order = torch.argsort(flat, stable=True)
    flat, pair = flat[order], pair[order]
    counts = torch.bincount(flat, minlength=E).tolist()
    r, gate, act = pair // k, [], []
    start = 0
    for e, c in enumerate(counts):
        if c:
            g, u = F.linear(x[r[start:start + c]], ex.gate_up_proj[e]).chunk(2, dim=-1)
            gate.append(g.float())
            act.append(ex.act_fn(g.float()) * u.float())
        start += c
    d0 = lambda: torch.zeros(0, I, device=x.device)
    gate, act = (torch.cat(v) if v else d0() for v in (gate, act))
    rho = w.flatten()[pair].float()
    sh = sg = sw = None
    if shared is not None:
        g_s, a_s, s = shared
        sw = torch.ones(n, device=x.device) if s is None else torch.sigmoid(s.float()).reshape(n)
        sh, sg = a_s.float() * sw[:, None], g_s.float()
    return Routed(r, flat, rho, gate, act * rho[:, None], sh, sg, sw, n, E, I)


class Tap:
    """Hooks on a decoder layer's sparse MLP that call ``sink(Routed)`` after each of its forward passes. ``rows``
    (optional): a callable giving, for the pass, the mask of the rows whose pairs to compute (17 keeps the positions some
    firing sits at)."""

    def __init__(self, mlp, sink, rows=None):
        self.mlp, self.sink, self.rows, self.cap = mlp, sink, rows, {}
        self.hooks = [mlp.experts.register_forward_pre_hook(self._experts, with_kwargs=True),
                      mlp.register_forward_hook(self._done)]
        sh = getattr(mlp, "shared_expert", None)
        if sh is not None:
            self.hooks += [sh.gate_proj.register_forward_hook(lambda m, a, o: self.cap.__setitem__("g_s", o)),
                           sh.down_proj.register_forward_pre_hook(lambda m, a: self.cap.__setitem__("a_s", a[0]))]
            gate = getattr(mlp, "shared_expert_gate", None)
            if gate is not None:
                self.hooks.append(gate.register_forward_hook(lambda m, a, o: self.cap.__setitem__("s", o)))

    def _experts(self, m, args, kwargs):
        names = ("hidden_states", "top_k_index", "top_k_weights")
        self.cap["x"], self.cap["idx"], self.cap["w"] = (*args, *(kwargs[n] for n in names[len(args):]))

    def _done(self, m, args, out):
        c, self.cap = self.cap, {}
        shared = None if "a_s" not in c else (c["g_s"].reshape(-1, c["g_s"].shape[-1]),
                                              c["a_s"].reshape(-1, c["a_s"].shape[-1]), c.get("s"))
        x = c["x"].reshape(-1, c["x"].shape[-1])
        self.sink(routed(self.mlp, x, c["idx"].reshape(len(x), -1), c["w"].reshape(len(x), -1),
                         None if self.rows is None else self.rows(), shared))

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()


def sparse(layer) -> bool:
    return anatomy.parts(layer).sparse


def width(model_id: str) -> int:
    """The MLP neurons a layer has, the same at every layer (``neurons.n_neurons``); the catalog's pole ids
    (``layer · P·f + row``) need that, and a model whose layers differ (dense layers between sparse ones) is refused."""
    from neuron_lens.arch import text_config
    from neuron_lens.neurons import n_neurons
    f = {n_neurons(model_id, l) for l in range(text_config(model_id).num_hidden_layers)}
    if len(f) != 1:
        raise NotImplementedError(f"{model_id}: layers of {sorted(f)} neurons; the catalog needs one width")
    return f.pop()


def stack_width(stack) -> int:
    """``width`` from a loaded decoder stack."""
    f = set()
    for lay in stack.layers:
        mlp = lay.mlp
        if sparse(lay):
            sh = getattr(mlp, "shared_expert", None)
            f.add(mlp.experts.num_experts * mlp.experts.intermediate_dim + (0 if sh is None else sh.down_proj.in_features))
        else:
            f.add(anatomy.module(lay, anatomy.parts(lay).down).in_features)
    if len(f) != 1:
        raise NotImplementedError(f"layers of {sorted(f)} neurons; the catalog needs one width")
    return f.pop()


class Rows:
    """A sparse layer's per-neuron rows, ``rows[n]`` ``[len(n), d]`` for neuron ids ``n``, gathered from ``experts``
    ``[E, I, d]`` (views of the model's weights) and ``shared`` ``[Is, d]`` (``None`` without a shared expert), times
    ``scale`` ``[d]`` (``None``: 1)."""

    def __init__(self, experts: torch.Tensor, shared: torch.Tensor | None, scale: torch.Tensor | None = None):
        self.experts, self.shared, self.scale = experts, shared, scale
        self.E, self.I, self.d = experts.shape
        self.n = self.E * self.I + (0 if shared is None else len(shared))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, n: torch.Tensor) -> torch.Tensor:
        n = n.long()
        out = torch.empty(len(n), self.d, device=self.experts.device)
        ex = n < self.E * self.I
        out[ex] = self.experts[n[ex] // self.I, n[ex] % self.I].float()
        if self.shared is not None:
            out[~ex] = self.shared[n[~ex] - self.E * self.I].float()
        return out if self.scale is None else out * self.scale

    def blocks(self, size: int = 2**14):
        """``(ids, rows)`` over every neuron, ``size`` at a time."""
        for s in range(0, self.n, size):
            ids = torch.arange(s, min(s + size, self.n), device=self.experts.device)
            yield ids, self[ids]


def layer_rows(lay, scale: torch.Tensor | None = None) -> dict:
    """A loaded sparse layer's ``gate``, ``up`` (times ``scale``, the MLP input norm's) and ``write`` (each neuron's
    write as the residual sees it: times the norm after the MLP's weight where one follows) as ``Rows``."""
    from neuron_lens.arch import mlp_norm
    mlp = lay.mlp
    ex, I = mlp.experts, mlp.experts.intermediate_dim
    sh = getattr(mlp, "shared_expert", None)
    post = mlp_norm(lay)
    gu = ex.gate_up_proj  # [E, 2I, d]
    out = {"gate": Rows(gu[:, :I], None if sh is None else sh.gate_proj.weight, scale),
           "up": Rows(gu[:, I:], None if sh is None else sh.up_proj.weight, scale),
           "write": Rows(ex.down_proj.transpose(1, 2), None if sh is None else sh.down_proj.weight.T,
                         None if post is None else post.weight.float())}
    return out


def layer_write_reads(lay, r: torch.Tensor) -> torch.Tensor:
    """``[n, f]``: each read row ``r`` ``[n, d]`` against every neuron's write of a loaded sparse layer, ``r · w_m``,
    without the ``[d, f]`` write matrix (one product per expert block of the model's own weights)."""
    from neuron_lens.arch import mlp_norm
    mlp = lay.mlp
    post = mlp_norm(lay)
    r = r.float() if post is None else r.float() * post.weight.float()
    W = mlp.experts.down_proj  # [E, d, I]
    out = torch.einsum("nd,edi->nei", r, W.float() if W.dtype != torch.float32 else W).flatten(1)
    sh = getattr(mlp, "shared_expert", None)
    return out if sh is None else torch.cat([out, r @ sh.down_proj.weight.float()], 1)


def factors(r: Routed, keep: torch.Tensor, act_fn) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """At the pairs whose row is in ``keep``: ``(cols, cg, cu)``, ``[M, I]`` each, the neuron ids and RelP's factors of
    the gate's and up's ways times the routing weight, ``ρ · ½ φ(g) u = ½ a / g`` and ``ρ · ½ act(g)`` (``φ = act / g``,
    the identity rule, and the half rule), whose means over the routed positions make an expert's mean map
    (``08_reads.py``). ``act_fn``: the experts' activation."""
    sel = keep[r.rows]
    g, a, rho = r.gate[sel], r.act[sel], r.weight[sel][:, None]
    cg = torch.where(g != 0, 0.5 * a / g.where(g != 0, torch.ones_like(g)), torch.zeros_like(g))
    return r.cols()[sel], cg, 0.5 * rho * g * act_factor(act_fn, g)
