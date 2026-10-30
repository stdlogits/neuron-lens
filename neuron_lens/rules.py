"""The RelP derivative rules, which both lens estimators take derivatives under.

RelP (arXiv:2508.21258, as in the R-lens and Drowse's ``core.relp``) replaces three local
derivatives in each decoder layer:
- **LN rule:** the decoder layers' RMSNorms hold their ``rsqrt`` constant (Qwen3: the input
  and post-attention norms; Gemma 4: those and the pre- and post-feedforward norms, the
  post-attention one normalizing the attention's output);
- **identity rule:** the MLP activation ``act(g) = g·φ(g)`` is linearised as ``φ(g)·dg``:
  ``φ = sigmoid`` for SiLU (Qwen3), ``φ = (1 + tanh(√(2/π)(g + 0.044715 g³)))/2`` for
  tanh-GELU (Gemma 4);
- **half rule:** ``act(g)·up`` passes half of each factor's change (β = 0.5).

Each rule is written as ``value.detach() + (lin - lin.detach())``: the value is the module's
own output, unchanged, while the derivative is that of ``lin``. Both forward-mode tangents
and reverse-mode gradients follow it, so the S-lens and the R-lens share one rule set, and
the R-lens's backward map is exactly the transpose of the S-lens's forward one. q/k norms
inside attention keep their true derivative (Drowse's ``include_qk_norms: false``), and so
does Gemma 4's weightless value norm, also inside attention.

RelP is the only rule set here: it beat the true Jacobian on the write side (the R-lens
against the J-lens) and on the read side (findings §1, §6).

``neuron_lens.transport`` writes the linear maps these rules give out by hand, for speed;
``relp_rules`` applies them to the model for autograd, which the tests use as the reference.
"""
import contextlib
import math
import types

import torch

HALF_RULE_BETA = 0.5
_GELU_C = math.sqrt(2 / math.pi)
RESIDUAL_NORMS = ("input_layernorm", "post_attention_layernorm",
                  "pre_feedforward_layernorm", "post_feedforward_layernorm")


def norm_eps(norm) -> float:
    return norm.variance_epsilon if hasattr(norm, "variance_epsilon") else norm.eps


def norm_weight(norm) -> torch.Tensor | None:
    """The norm's weight, or ``None`` for a weightless one (Gemma 4's value norm)."""
    return getattr(norm, "weight", None)


def act_factor(act_fn, g: torch.Tensor) -> torch.Tensor:
    """``φ(g) = act(g) / g`` (identity rule), for the activations ``check_layer`` accepts."""
    name = type(act_fn).__name__.casefold()
    if name in ("siluactivation", "silu"):
        return torch.sigmoid(g)
    if name == "gelutanh":
        return 0.5 * (1 + torch.tanh(_GELU_C * (g + 0.044715 * g.pow(3))))
    raise ValueError(f"RelP identity rule has no factor for {type(act_fn).__name__}")


def residual_norms(layer) -> list:
    return [getattr(layer, n) for n in RESIDUAL_NORMS if hasattr(layer, n)]


def _norm(self, x: torch.Tensor) -> torch.Tensor:
    y = self._rules_orig_forward(x)
    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm_eps(self)).detach()
    w = norm_weight(self)
    lin = ((1 if w is None else w.float()) * xf * scale).to(y.dtype)
    return y.detach() + (lin - lin.detach())


def _mlp(self, x: torch.Tensor) -> torch.Tensor:
    g, u = self.gate_proj(x), self.up_proj(x)
    act = self.act_fn(g)
    act_lin = g * act_factor(self.act_fn, g).detach()
    da, du = act_lin - act_lin.detach(), u - u.detach()
    prod = (act * u).detach() + HALF_RULE_BETA * (da * u.detach() + act.detach() * du)
    return self.down_proj(prod)


def check_layer(layer) -> None:
    """Refuse a layer whose norms or activation are not the shapes the rules assume."""
    act_factor(layer.mlp.act_fn, torch.zeros(1))  # raises for an activation without a rule
    at = layer.self_attn
    norms = residual_norms(layer) + [at.q_norm, getattr(at, "k_norm", None), getattr(at, "v_norm", None)]
    for norm in filter(None, norms):
        w = norm_weight(norm)
        dim = w.shape[0] if w is not None else 8
        p = next(layer.parameters())
        x = torch.randn(2, dim, device=p.device, dtype=p.dtype)
        with torch.no_grad():
            y = norm(x).float()
            xf = x.float()
            lin = (1 if w is None else w.float()) * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm_eps(norm))
        if not torch.allclose(lin, y, rtol=1e-2, atol=1e-2):
            raise ValueError(f"{type(norm).__name__} is not a plain-weight RMSNorm")


@contextlib.contextmanager
def relp_rules(model):
    """Apply the RelP rules to a decoder stack (``model.layers``, Qwen3 or Gemma 4) for the
    duration."""
    patched = []
    for layer in model.layers:
        check_layer(layer)
        patched += [(norm, _norm) for norm in residual_norms(layer)] + [(layer.mlp, _mlp)]
    for module, _ in patched:
        if "forward" in module.__dict__:
            raise ValueError(f"{type(module).__name__} already has an instance-level forward")
    for module, fn in patched:
        module._rules_orig_forward = module.forward
        module.forward = types.MethodType(fn, module)
    try:
        yield
    finally:
        for module, _ in patched:
            del module.forward  # the instance attribute; the class's forward is back
            del module._rules_orig_forward
