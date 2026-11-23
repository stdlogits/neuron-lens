"""Each MLP neuron's read and write directions as the residual sees them, across models, for the
catalog stages (``08_reads.py``, ``14_media_tables.py``, ``16_edges.py``) and the checks (``checks/reads_at_firings.py``).
``poles`` gives them in the residual's metric (``neuron_lens.metric``): the covector a pole reads with and its vector.

A neuron ``n`` of layer ``L`` reads the MLP input through its gate and up rows times the norm that
feeds the MLP (Qwen3: ``post_attention_layernorm``; Gemma 4: ``pre_feedforward_layernorm``), and
writes column ``n`` of ``down_proj``. Gemma 4's MLP output then passes a post-feedforward RMSNorm,
whose scale the LN rule holds constant, and the layer's residual is multiplied by its
``layer_scalar``: what reaches the residual is ``s · γ_post ⊙ W_down[:, n]``, the write the
catalog reads. Qwen3's is ``W_down[:, n]`` itself. GPT-NeoX's MLP is ungated, ``W_out GELU(W_in x + b)``: its input
row plays the gate's part (``gate``; no ``up``), each neuron has one pole (``arch.n_poles``), and its rows are
``centre``d, as its LayerNorm reads the centred residual.
"""

import torch

from neuron_lens import anatomy, metric
from neuron_lens.arch import centre, layout, norm_scale
from neuron_lens.weights import tensors


def mlp_names(model_id: str, layer: int) -> dict:
    """A dense MLP's tensor names in the checkpoint (``anatomy``): ``gate``, ``up`` (gated MLPs), ``down``, and the
    norm that feeds it (``norm``; none where it reads the residual itself, OLMo 2 and 3), the norm on its output
    (``post``: Gemma 4, OLMo) and the layer's scalar (``scalar``: Gemma 4), where the layer has them."""
    p = anatomy.layer_parts(model_id, layer)
    w = lambda path, what="weight": anatomy.weight_name(model_id, layer, path, what)
    names = {"gate": w(p.gate), "down": w(p.down)}
    for key, path in (("up", p.up), ("norm", p.n2), ("post", p.post2)):
        if path is not None:
            names[key] = w(path)
    if p.scalar is not None:
        names["scalar"] = w(p.scalar, "")
    return names


def directions(model_id: str, layer: int, dev="cpu") -> dict:
    """``write`` ``[d, f]`` (the direction each neuron adds to the residual; where a norm follows the MLP, Gemma 4's
    and OLMo's, its size at a position also scales with that norm there), and the read rows ``gate``, ``up``
    ``[f, d]`` times the MLP input norm's scale (``arch.norm_scale``; none in OLMo 2 and 3), ``centre``d, in fp32; no
    ``up`` for an ungated MLP."""
    p, lay = anatomy.layer_parts(model_id, layer), layout(model_id)[layer]
    nm = mlp_names(model_id, layer)
    w = {k: v.float().to(dev) for k, v in tensors(model_id, sorted(nm.values())).items()}
    g = norm_scale(model_id, lay, p.n2, dev)
    write = w[nm["down"]]
    if p.post2 is not None:
        write = norm_scale(model_id, lay, p.post2, dev)[:, None] * write
    if "scalar" in nm:
        write = w[nm["scalar"]].reshape(()) * write
    rows = {k: w[nm[k]] if g is None else w[nm[k]] * g for k in ("gate", "up") if k in nm}
    return {"write": centre(write, model_id, 0), **{k: centre(v, model_id) for k, v in rows.items()}}


def softcap(model_id: str) -> float | None:
    """The final logits' softcap (Gemma: ``30 · tanh(x / 30)``), or ``None``."""
    from neuron_lens.arch import text_config
    return getattr(text_config(model_id), "final_logit_softcapping", None)


def read_rows(a: torch.Tensor, b: torch.Tensor, lobe: torch.Tensor) -> torch.Tensor:
    """``[2f, d]``, the poles' read directions, ``+`` poles then ``−``, from the unit gate and up rows
    ``a``, ``b`` ``[f, d]``: ``â ± b̂``, and the up row alone, ``∓b̂``, for the poles on SiLU's negative lobe
    (``lobe``, ``catalog.lobe_signs``). There SiLU is nearly flat, so the activation follows ``up``, opposite
    in sign to the pole, and which way of the gate raises it varies from pole to pole (findings §7)."""
    return torch.where(lobe.to(a)[:, None] < 0, torch.cat([-b, b]), torch.cat([a + b, a - b]))


def poles(model_id: str, layer: int, lobe: torch.Tensor, dev="cpu") -> dict:
    """A layer's neurons in the residual's metric (``neuron_lens.metric``, findings §15): ``write`` ``[d, f]``; ``a``, ``b``
    ``[f, d]``, the gate and up rows times the MLP input norm (covectors) at unit length in the metric; ``cov`` ``[2f, d]``,
    the poles' read covectors (``read_rows``), what the S-lens pulls back and a write meets; ``vec`` ``[2f, d]``, each
    one's vector ``s² ⊙ cov`` at unit length, what the R-lens transports (``σ ⊙ (ŵ_gate ± ŵ_up)``, the raw rows times
    the residual's scale); ``s2`` ``[d]``, the metric at the MLP input. An ungated MLP has no ``b``, and its poles' rows
    are ``a`` (``[f, d]``: one pole a neuron)."""
    w = directions(model_id, layer, dev)
    s2 = metric.s2(model_id, "mid", layer, dev)
    a, b = (w[k] / metric.cov_norm(w[k], s2) if k in w else None for k in ("gate", "up"))
    cov = a if b is None else read_rows(a, b, lobe)
    vec = metric.vector(cov, s2)
    return {"write": w["write"], "a": a, "b": b, "cov": cov, "vec": vec / vec.norm(dim=1, keepdim=True), "s2": s2}


def write_cosines(write: torch.Tensor, a: torch.Tensor, b: torch.Tensor, lobe: torch.Tensor,
                  s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos_gate, cos_up)``, ``[2f]`` each: the cosine, in the metric ``s2``, of each pole's write with the gate and up
    rows (``poles``' unit ``a``, ``b``), each on the side its firings sit (the gate below zero on the negative lobe, ``up``
    opposite the pole there), positive where the pole writes toward them. Ungated (``b`` None): ``[f]`` with the input
    row, and NaN for ``up``."""
    w = write.T / metric.vec_norm(write.T, s2)
    x = (w * a).sum(1)
    if b is None:
        return x, torch.full_like(x, float("nan"))
    y = (w * b).sum(1)
    return torch.cat([x, -x]) * lobe.to(x), torch.cat([y, y]) * lobe.to(x)


def orthogonal_write(write: torch.Tensor, a: torch.Tensor, b: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """``[d, f]``: each write minus its parts along its neuron's gate and up patterns (``s² ⊙ a``, ``s² ⊙ b``), so that
    neither row reads it, what it adds beyond anything it reads (``a``, ``b`` unit in the metric ``s2``, as ``poles``;
    ungated, ``b`` None: minus its part along ``s² ⊙ a`` alone)."""
    pa, w = metric.vector(a, s2), write.T
    if b is None:
        return (w - (a * w).sum(1, keepdim=True) * pa).T
    pb = metric.vector(b, s2)
    g = (a * pb).sum(1)
    ra, rb = (a * w).sum(1), (b * w).sum(1)
    det = (1 - g.pow(2)).clamp_min(1e-6)
    alpha, beta = (ra - g * rb) / det, (rb - g * ra) / det
    return (w - alpha[:, None] * pa - beta[:, None] * pb).T
