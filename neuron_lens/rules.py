"""The RelP derivative rules, which both lens estimators take derivatives under, stated for
autograd: ``relp_rules`` applies them to a decoder stack, and the tests use it as the reference for
``neuron_lens.layers``, which writes the same linear maps out by hand, for speed.

RelP (arXiv:2508.21258, as in the R-lens and Drowse's ``lens.relp``) replaces local derivatives by
operation, wherever the operation sits:
- **LN rule:** RMSNorms hold their ``rsqrt`` constant: the residual stream's (Qwen3: the input and
  post-attention norms; Gemma 4: those and the pre- and post-feedforward norms, the post-attention
  one normalizing the attention's output) and Qwen3.5 DeltaNet's per-head output norm;
- **identity rule:** an activation ``act(g) = g·φ(g)`` is linearised as ``φ(g)·dg``: ``φ = sigmoid``
  for SiLU (in MLPs and experts, after DeltaNet's convolution, in its ``silu(z)``),
  ``φ = (1 + tanh(√(2/π)(g + 0.044715 g³)))/2`` for tanh-GELU (Gemma 4);
- **half rule:** an elementwise gate on a vector of features passes half of each factor's change
  (β = 0.5): ``act(g)·up``, DeltaNet's ``norm(o)·silu(z)``, gated attention's ``o·σ(gate)``.

Scalar weights that decide how much of a value or a module passes keep their true derivative:
softmax attention's, DeltaNet's mixing matrix (its q, k, β and decay), a router's weights over its
chosen experts and a shared expert's sigmoid gate; so do q/k norms (Drowse's
``include_qk_norms: false``), Gemma 4's weightless value norm and DeltaNet's L2 norms.

Each rule is written as ``value.detach() + (lin - lin.detach())``: the value is the module's
own output, unchanged, while the derivative is that of ``lin``. Both forward-mode tangents
and reverse-mode gradients follow it, so the S-lens and the R-lens share one rule set, and
the R-lens's backward map is exactly the transpose of the S-lens's forward one.

RelP is the only rule set here: it beat the true Jacobian on the write side (the R-lens
against the J-lens) and on the read side (findings §1, §6).
"""
import contextlib
import math
import sys
import types

import torch

HALF_RULE_BETA = 0.5
_GELU_C = math.sqrt(2 / math.pi)
RESIDUAL_NORMS = ("input_layernorm", "post_attention_layernorm",
                  "pre_feedforward_layernorm", "post_feedforward_layernorm")


def norm_eps(norm) -> float:
    return norm.variance_epsilon if hasattr(norm, "variance_epsilon") else norm.eps


ZERO_CENTRED = ("Qwen3_5RMSNorm", "Qwen3_5MoeRMSNorm")  # these scale by 1 + weight


def norm_weight(norm) -> torch.Tensor | None:
    """The norm's scale: its weight (``1 + weight`` for Qwen3.5's zero-centred norms), or
    ``None`` for a weightless one (Gemma 4's value norm)."""
    w = getattr(norm, "weight", None)
    return 1.0 + w if w is not None and type(norm).__name__ in ZERO_CENTRED else w


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


def is_gated(at) -> bool:
    """Gated attention (Qwen3.5): ``q_proj`` gives, per head, the query and an output gate."""
    return at.q_proj.weight.shape[0] == 2 * at.config.num_attention_heads * at.head_dim


def _check_norm(norm, p) -> None:
    w = norm_weight(norm)
    dim = w.shape[0] if w is not None else 8
    x = torch.randn(2, dim, device=p.device, dtype=p.dtype)
    with torch.no_grad():
        y = norm(x).float()
        xf = x.float()
        lin = (1 if w is None else w.float()) * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm_eps(norm))
    if not torch.allclose(lin, y, rtol=1e-2, atol=1e-2):
        raise ValueError(f"{type(norm).__name__} is not an RMSNorm the rules know")


def check_layer(layer) -> None:
    """Refuse a layer whose norms or activations are not the shapes the rules assume."""
    mlp = layer.mlp
    acts = [mlp.experts.act_fn] + ([mlp.shared_expert.act_fn] if getattr(mlp, "shared_expert", None) else []) \
        if hasattr(mlp, "experts") else [mlp.act_fn]
    for act in acts:
        act_factor(act, torch.zeros(1))  # raises for an activation without a rule
    norms = residual_norms(layer)
    if hasattr(layer, "linear_attn"):
        if layer.linear_attn.activation != "silu":
            raise ValueError(f"DeltaNet activation {layer.linear_attn.activation} has no rule")
    else:
        at = layer.self_attn
        norms += [getattr(at, n, None) for n in ("q_norm", "k_norm", "v_norm")]
    p = next(layer.parameters())
    for norm in filter(None, norms):
        _check_norm(norm, p)


# ------------------------------------------------------------- gates, DeltaNet, sparse MLPs

def _half(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a · b`` whose derivative passes half of each factor's change (the half rule)."""
    da, db = a - a.detach(), b - b.detach()
    return (a * b).detach() + HALF_RULE_BETA * (da * b.detach() + a.detach() * db)


def _silu_id(x: torch.Tensor) -> torch.Tensor:
    """SiLU under the identity rule: derivative ``σ(x)``."""
    lin = x * torch.sigmoid(x).detach()
    return torch.nn.functional.silu(x).detach() + (lin - lin.detach())


def _hf(module, name: str):
    """A helper function from the module's own modeling file (its RoPE, attention, delta rule)."""
    return getattr(sys.modules[type(module).__module__], name)


def _gated_attn(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    apply_rotary_pos_emb, eager_attention_forward = _hf(self, "apply_rotary_pos_emb"), _hf(self, "eager_attention_forward")
    shape = hidden_states.shape[:-1]
    q, gate = torch.chunk(self.q_proj(hidden_states).view(*shape, -1, self.head_dim * 2), 2, dim=-1)
    q = self.q_norm(q.view(*shape, -1, self.head_dim)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(*shape, -1, self.head_dim)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(*shape, -1, self.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
    o, _ = eager_attention_forward(self, q, k, v, attention_mask, scaling=self.scaling)
    o = o.reshape(*shape, -1)
    return self.o_proj(_half(o, torch.sigmoid(gate.reshape(*shape, -1)))), None


def _delta(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    torch_chunk_gated_delta_rule = _hf(self, "torch_chunk_gated_delta_rule")
    F = torch.nn.functional
    B, T, _ = hidden_states.shape
    mixed = self.in_proj_qkv(hidden_states).transpose(1, 2)
    K = self.conv_kernel_size
    mixed = _silu_id(F.conv1d(F.pad(mixed, (K - 1, 0)), self.conv1d.weight, groups=self.conv_dim)).transpose(1, 2)
    q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    q, k = q.reshape(B, T, -1, self.head_k_dim), k.reshape(B, T, -1, self.head_k_dim)
    v = v.reshape(B, T, -1, self.head_v_dim)
    z = self.in_proj_z(hidden_states).reshape(B, T, -1, self.head_v_dim)
    beta = self.in_proj_b(hidden_states).sigmoid()
    g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)
    rep = self.num_v_heads // self.num_k_heads
    q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
    o, _ = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    o = o.float()
    r = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.norm.variance_epsilon).detach()
    y = _half(self.norm.weight.float() * o * r, _silu_id(z.float()))
    return self.out_proj(y.reshape(B, T, -1).to(hidden_states.dtype))


def _moe(self, hidden_states):
    F = torch.nn.functional
    B, T, D = hidden_states.shape
    x = hidden_states.view(-1, D)
    _, w, idx = self.gate(x)
    out = torch.zeros_like(x)
    ex = self.experts
    for e in idx.unique().tolist():
        tok, slot = torch.where(idx == e)
        gate, up = F.linear(x[tok], ex.gate_up_proj[e]).chunk(2, dim=-1)
        act_lin = gate * act_factor(ex.act_fn, gate).detach()
        act = ex.act_fn(gate).detach() + (act_lin - act_lin.detach())
        eo = F.linear(_half(act, up), ex.down_proj[e])
        out = out.index_add(0, tok, eo * w[tok, slot, None])
    if getattr(self, "shared_expert", None) is not None:
        shared = self.shared_expert(x)
        if getattr(self, "shared_expert_gate", None) is not None:
            shared = torch.sigmoid(self.shared_expert_gate(x)) * shared  # a one-expert router
        out = out + shared
    return out.view(B, T, D)


@contextlib.contextmanager
def relp_rules(model):
    """Apply the RelP rules to a decoder stack (``model.layers``) for the duration: patch each
    residual norm, MLP or sparse MLP (and its shared expert), DeltaNet mixer and gated attention."""
    patched = []
    for layer in model.layers:
        check_layer(layer)
        patched += [(norm, _norm) for norm in residual_norms(layer)]
        if hasattr(layer, "linear_attn"):
            patched.append((layer.linear_attn, _delta))
        elif is_gated(layer.self_attn):
            patched.append((layer.self_attn, _gated_attn))
        if hasattr(layer.mlp, "experts"):
            patched.append((layer.mlp, _moe))
            if getattr(layer.mlp, "shared_expert", None) is not None:
                patched.append((layer.mlp.shared_expert, _mlp))
        else:
            patched.append((layer.mlp, _mlp))
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
