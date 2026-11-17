"""What is in the residual at each neuron pole's top firings, for ``17_firings.py``.

One forward pass over the activation documents. At each firing (``acts``' ``top_doc``, ``top_pos``,
where ``top_val > 0``), per layer and pole, a running sum of

- ``before``: the residual entering the norm that feeds the layer's MLP (the input ``mid`` targets read);
- ``mlp``: the layer's MLP output as it reaches the residual;
- ``pred``: the final residual, normalized as the final norm does: through the unembedding, the mean of it is the
  model's mean logit there (for Gemma 4, the softcap is applied to the mean, not to each firing's logits).

``before`` and ``mlp`` share the firing's scale, ``1 / rms(before)``, and so stay additive: the residual
after the layer is ``before · s + mlp`` (``s`` Gemma 4's ``layer_scalar``, 1 for Qwen3), plus the layer's attention in
a parallel block (GPT-NeoX's). Where the norms are LayerNorms (``arch.centred``) all three are centred, as every reader
sees them, and the scale is ``1 / std``.

Also per firing, what the pole does to the part of the residual it reads. Under RelP's rules (the norm's scale ``k = 1 / rms``
and ``φ(g) / g`` held) the activation is exactly linear in the residual ``x``, ``a = c · x`` with
``c = ½ (φ(g) / g) k (u r_gate + g r_up)``, and the neuron's whole contribution is the rank-one map ``x → w (c · x)`` (``w`` its
write as the residual sees it). Along ``c`` that map multiplies the residual by ``parallel = c · w``, positive where the
pole amplifies what it reads, negative where it suppresses it; off ``c`` it adds ``orthogonal = √(‖c‖²‖w‖² − (c · w)²)`` of the
same size. Both are numbers per unit of what the pole reads, unchanged by rescaling ``up`` against ``down``. An ungated MLP
(GPT-NeoX's) reads ``g = k r · x + b`` with ``r`` its centred input row and ``b = W_in β + b_in`` its constant (the norm's
and the projection's biases), and ``c = φ(g) k r``.
"""
import torch

from neuron_lens import arch
from neuron_lens.rules import final_norm, is_centred, mlp_act

SITES = ("before", "mlp", "pred")


def firing_plan(doc: torch.Tensor, pos: torch.Tensor, val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A layer's firings from its ``[P·f, K]`` ``top_doc``, ``top_pos``, ``top_val``: ``(doc, pos, pole row)``,
    flat, sorted by document, those with a positive value."""
    row, k = (val > 0).nonzero(as_tuple=True)
    d = doc[row, k]
    order = d.argsort(stable=True)
    return d[order].int(), pos[row, k][order].int(), row[order].int()


def weights(lay) -> dict:
    """What ``parallel`` and ``orthogonal`` need from a decoder layer's weights: the read rows (``gate``, ``up``, each times
    ``norm``; centred where the norm is a LayerNorm, through each row's mean ``gm``, ``um``), the activation, and per neuron
    the inner products of those rows and its write ``w``, in fp32. Ungated (GPT-NeoX): ``gate`` is the input row, ``up``
    is None, and ``b`` the pre-activation's constant ``W_in β + b_in``."""
    norm_m = arch.mid_norm(lay)
    norm, centred = norm_m.weight.float(), is_centred(norm_m)
    mlp = lay.mlp
    gated = hasattr(mlp, "gate_proj")
    gate = (mlp.gate_proj if gated else mlp.dense_h_to_4h).weight.float()
    w = arch.mlp_write(lay).T
    dot = lambda a, b: (a * b).sum(1)
    rg = gate * norm
    gm = rg.mean(1) if centred else torch.zeros(len(rg), device=rg.device)
    rg = rg - gm[:, None]
    out = {"gate": gate, "norm": norm, "gm": gm, "act": mlp_act(mlp), "gw": dot(rg, w), "gg": dot(rg, rg), "ww": dot(w, w),
           "centred": centred}
    if gated:
        up = mlp.up_proj.weight.float()
        ru = up * norm
        um = ru.mean(1) if centred else torch.zeros(len(ru), device=ru.device)
        ru = ru - um[:, None]
        return out | {"up": up, "um": um, "uw": dot(ru, w), "uu": dot(ru, ru), "gu": dot(rg, ru)}
    b = mlp.dense_h_to_4h.bias
    beta = getattr(norm_m, "bias", None)
    const = torch.zeros(len(gate), device=gate.device) if b is None else b.float().clone()
    if beta is not None:
        const += gate @ beta.float()
    return out | {"up": None, "b": const}


class Accumulator:
    """Running sums ``[rows, d]`` per site and layer (``rows`` a layer's poles, ``P·f``), on the CPU; ``hooks`` adds to them
    during a forward pass. Between ``begin`` and the end of each batch, documents ``a..b`` are rows ``0..`` of the batch."""

    def __init__(self, plan: dict[int, tuple], rows: int, d: int, device):
        self.plan = {l: tuple(t.to(device) for t in p) for l, p in plan.items()}
        self.layers = sorted(plan)
        self.dev = device
        self.sum = {site: {l: torch.zeros(rows, d) for l in self.layers} for site in SITES}
        self.count = {l: torch.zeros(rows) for l in self.layers}
        self.gain = {l: torch.zeros(rows, 2) for l in self.layers}  # parallel, orthogonal, summed
        self.scale, self.weights, self.map = {}, {}, {}
        self.a = self.b = 0

    def begin(self, a: int, b: int) -> None:
        self.a, self.b = a, b

    def _firings(self, l: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        doc, pos, row = self.plan[l]
        lo, hi = torch.searchsorted(doc, self.a), torch.searchsorted(doc, self.b)
        return doc[lo:hi] - self.a, pos[lo:hi], row[lo:hi]

    def before(self, l: int, x: torch.Tensor) -> None:
        bi, p, r = self._firings(l)
        v = x[bi, p].float()
        if self.weights[l]["centred"]:
            v = v - v.mean(-1, keepdim=True)
        self.scale[l] = torch.rsqrt(v.pow(2).mean(-1, keepdim=True).clamp_min(1e-12))
        self.map[l] = self.self_map(l, v, r.long(), self.scale[l][:, 0])
        r = r.long().cpu()
        self.sum["before"][l].index_add_(0, r, (v * self.scale[l]).cpu())
        self.count[l].index_add_(0, r, torch.ones(len(r)))

    def self_map(self, l: int, x: torch.Tensor, row: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """``[n, 2]``: at firings of poles ``row`` with residual ``x`` and norm scale ``k``, ``parallel`` and ``orthogonal``."""
        w = self.weights[l]
        n = row % (len(w["ww"]))
        xs = x.sum(-1)  # (the rows' means times it: their centring, where the norm centres)
        g = ((w["gate"][n] * w["norm"] * x).sum(-1) - w["gm"][n] * xs) * k
        if w["up"] is None:  # ungated: the identity rule, c = φ(g) k r
            g = g + w["b"][n]
            c = w["act"](g) / g * k
            par = c * w["gw"][n]
            return torch.stack([par, (c.square() * w["gg"][n] * w["ww"][n] - par.square()).clamp_min(0).sqrt()], 1)
        u = ((w["up"][n] * w["norm"] * x).sum(-1) - w["um"][n] * xs) * k
        s = w["act"](g) / g
        cg, cu = 0.5 * s * u * k, 0.5 * s * g * k  # the read at this firing: c = cg r_gate + cu r_up
        par = cg * w["gw"][n] + cu * w["uw"][n]
        size2 = (cg.square() * w["gg"][n] + cu.square() * w["uu"][n] + 2 * cg * cu * w["gu"][n]) * w["ww"][n]
        return torch.stack([par, (size2 - par.square()).clamp_min(0).sqrt()], 1)

    def mlp(self, l: int, out: torch.Tensor, scalar, post) -> None:
        bi, p, r = self._firings(l)
        v = out[bi, p].float() * (1 if scalar is None else scalar.float().reshape(()))
        if self.weights[l]["centred"]:
            v = v - v.mean(-1, keepdim=True)
        r = r.long().cpu()
        self.sum["mlp"][l].index_add_(0, r, (v * self.scale[l]).cpu())
        self.gain[l].index_add_(0, r, (self.map[l] if post is None else self.map[l] * post[bi, p].float()[:, None]).cpu())

    def pred(self, x: torch.Tensor) -> None:
        for l in self.layers:
            bi, p, r = self._firings(l)
            v = x[bi, p].float()
            if self.weights[l]["centred"]:
                v = v - v.mean(-1, keepdim=True)
            self.sum["pred"][l].index_add_(0, r.long().cpu(), (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True).clamp_min(1e-12))).cpu())

    def hooks(self, stack) -> list:
        out = []
        for l in self.layers:
            lay = stack.layers[l]
            module, scalar = arch.mlp_output(lay)
            self.weights[l] = weights(lay)
            out.append(arch.mid_norm(lay).register_forward_pre_hook(lambda m, a, l=l: self.before(l, a[0])))
            out.append(module.register_forward_hook(lambda m, a, o, l=l, s=scalar, lay=lay: self.mlp(l, o, s, arch.mlp_scale(lay, a[0]))))
        out.append(final_norm(stack).register_forward_pre_hook(lambda m, a: self.pred(a[0])))
        return out

    def means(self, l: int) -> dict[str, torch.Tensor]:
        """The layer's means ``[P·f, d]`` (zero rows for poles without a firing)."""
        n = self.count[l].clamp_min(1)[:, None]
        return {site: self.sum[site][l] / n for site in SITES}

    def gains(self, l: int) -> torch.Tensor:
        """``[P·f, 2]``: the layer's ``parallel`` and ``orthogonal``, each pole's mean over its firings (NaN without any)."""
        return torch.where(self.count[l] > 0, self.gain[l].T / self.count[l].clamp_min(1), torch.tensor(float("nan"))).T

    def state(self) -> dict:
        return {"sum": self.sum, "count": self.count, "gain": self.gain}

    def load(self, state: dict) -> None:
        self.sum, self.count, self.gain = state["sum"], state["count"], state["gain"]
