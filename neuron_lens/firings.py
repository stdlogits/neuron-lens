"""What is in the residual at each neuron pole's top firings, for ``17_firings.py``.

One forward pass over the activation documents. At each firing (``acts``' ``top_doc``, ``top_pos``,
where ``top_val > 0``), per layer and pole, a running sum of

- ``before``: the residual entering the norm that feeds the layer's MLP (the input ``mid`` targets read);
- ``mlp``: the layer's MLP output as it reaches the residual;
- ``pred``: the final residual, normalized as the final norm does: through the unembedding, the mean of it is the
  model's mean logit there (for Gemma 4, the softcap is applied to the mean, not to each firing's logits).

``before`` and ``mlp`` share the firing's scale, ``1 / rms(before)``, and so stay additive: the residual
after the layer is ``before · s + mlp`` (``s`` Gemma 4's ``layer_scalar``, 1 for Qwen3).
"""
import torch

from neuron_lens import arch

SITES = ("before", "mlp", "pred")


def firing_plan(doc: torch.Tensor, pos: torch.Tensor, val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A layer's firings from its ``[2f, K]`` ``top_doc``, ``top_pos``, ``top_val``: ``(doc, pos, pole row)``,
    flat, sorted by document, those with a positive value."""
    row, k = (val > 0).nonzero(as_tuple=True)
    d = doc[row, k]
    order = d.argsort(stable=True)
    return d[order].int(), pos[row, k][order].int(), row[order].int()


class Accumulator:
    """Running sums ``[2f, d]`` per site and layer, on the CPU; ``hooks`` adds to them during a forward pass.
    Between ``begin`` and the end of each batch, documents ``a..b`` are rows ``0..`` of the batch."""

    def __init__(self, plan: dict[int, tuple], f: int, d: int, device):
        self.plan = {l: tuple(t.to(device) for t in p) for l, p in plan.items()}
        self.layers = sorted(plan)
        self.dev = device
        self.sum = {site: {l: torch.zeros(2 * f, d) for l in self.layers} for site in SITES}
        self.count = {l: torch.zeros(2 * f) for l in self.layers}
        self.scale, self.scalar = {}, {l: 1.0 for l in self.layers}
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
        self.scale[l] = torch.rsqrt(v.pow(2).mean(-1, keepdim=True).clamp_min(1e-12))
        r = r.long().cpu()
        self.sum["before"][l].index_add_(0, r, (v * self.scale[l]).cpu())
        self.count[l].index_add_(0, r, torch.ones(len(r)))

    def mlp(self, l: int, out: torch.Tensor, scalar) -> None:
        bi, p, r = self._firings(l)
        v = out[bi, p].float() * (1 if scalar is None else scalar.float().reshape(()))
        self.sum["mlp"][l].index_add_(0, r.long().cpu(), (v * self.scale[l]).cpu())

    def pred(self, x: torch.Tensor) -> None:
        for l in self.layers:
            bi, p, r = self._firings(l)
            v = x[bi, p].float()
            self.sum["pred"][l].index_add_(0, r.long().cpu(), (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True).clamp_min(1e-12))).cpu())

    def hooks(self, stack) -> list:
        out = []
        for l in self.layers:
            lay = stack.layers[l]
            module, scalar = arch.mlp_output(lay)
            self.scalar[l] = 1.0 if scalar is None else float(scalar)
            out.append(arch.mid_norm(lay).register_forward_pre_hook(lambda m, a, l=l: self.before(l, a[0])))
            out.append(module.register_forward_hook(lambda m, a, o, l=l, s=scalar: self.mlp(l, o, s)))
        out.append(stack.norm.register_forward_pre_hook(lambda m, a: self.pred(a[0])))
        return out

    def means(self, l: int) -> dict[str, torch.Tensor]:
        """The layer's means ``[2f, d]`` (zero rows for poles without a firing), with ``after``."""
        n = self.count[l].clamp_min(1)[:, None]
        m = {site: self.sum[site][l] / n for site in SITES}
        m["after"] = m["before"] * self.scalar[l] + m["mlp"]
        return m

    def state(self) -> dict:
        return {"sum": self.sum, "count": self.count}

    def load(self, state: dict) -> None:
        self.sum, self.count = state["sum"], state["count"]
