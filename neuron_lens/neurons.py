"""Each MLP neuron's read and write directions as the residual sees them, across models, for the
catalog stages (``08_reads.py``, ``14_media_tables.py``, ``16_edges.py``) and the checks (``checks/reads_at_firings.py``).
``poles`` gives them in the residual's metric (``neuron_lens.metric``): the covector a pole reads with and its vector.

A neuron ``n`` of layer ``L`` reads the MLP input through its gate and up rows times the norm that
feeds the MLP (Qwen3: ``post_attention_layernorm``; Gemma 4: ``pre_feedforward_layernorm``), and
writes column ``n`` of ``down_proj``. Gemma 4's MLP output then passes a post-feedforward RMSNorm,
whose scale the LN rule holds constant, and the layer's residual is multiplied by its
``layer_scalar``: what reaches the residual is ``s · γ_post ⊙ W_down[:, n]``, the write the
catalog reads. Qwen3's is ``W_down[:, n]`` itself.
"""

import torch

from neuron_lens import metric
from neuron_lens.arch import is_gemma4, layout
from neuron_lens.weights import tensors


def mlp_names(model_id: str, layer: int) -> dict:
    p = layout(model_id)[layer].prefix
    names = {"gate": p + "mlp.gate_proj.weight", "up": p + "mlp.up_proj.weight", "down": p + "mlp.down_proj.weight"}
    if is_gemma4(model_id):
        return names | {"norm": p + "pre_feedforward_layernorm.weight", "post": p + "post_feedforward_layernorm.weight",
                        "scalar": p + "layer_scalar"}
    return names | {"norm": p + "post_attention_layernorm.weight"}


def directions(model_id: str, layer: int, dev="cpu") -> dict:
    """``write`` ``[d, f]`` (the direction each neuron adds to the residual; for Gemma 4 its size at a
    position also scales with the post-feedforward norm there), and the
    read rows ``gate``, ``up`` ``[f, d]`` times the MLP input norm's weight, in fp32."""
    nm = mlp_names(model_id, layer)
    w = {k: v.float().to(dev) for k, v in tensors(model_id, sorted(nm.values())).items()}
    g = w[nm["norm"]]
    write = w[nm["down"]]
    if "post" in nm:
        write = w[nm["scalar"]].reshape(()) * w[nm["post"]][:, None] * write
    return {"write": write, "gate": w[nm["gate"]] * g, "up": w[nm["up"]] * g}


def softcap(model_id: str) -> float | None:
    """The final logits' softcap (Gemma: ``30 · tanh(x / 30)``), or ``None``."""
    from neuron_lens.arch import text_config
    return getattr(text_config(model_id), "final_logit_softcapping", None)


def read_rows(a: torch.Tensor, b: torch.Tensor, lobe: torch.Tensor) -> torch.Tensor:
    """``[2f, d]``, the poles' read directions, ``+`` poles then ``−``, from the unit gate and up rows
    ``a``, ``b`` ``[f, d]``: ``â ± b̂``, and the up row alone, ``∓b̂``, for the poles on SiLU's negative lobe
    (``lobe``, ``catalog.lobe_signs``). There SiLU is nearly flat, so the activation follows ``up``, opposite
    in sign to the pole, and which way of the gate raises it varies from pole to pole (findings §17)."""
    return torch.where(lobe.to(a)[:, None] < 0, torch.cat([-b, b]), torch.cat([a + b, a - b]))


def poles(model_id: str, layer: int, lobe: torch.Tensor, dev="cpu") -> dict:
    """A layer's neurons in the residual's metric (``neuron_lens.metric``, findings §13): ``write`` ``[d, f]``; ``a``, ``b``
    ``[f, d]``, the gate and up rows times the MLP input norm (covectors) at unit length in the metric; ``cov`` ``[2f, d]``,
    the poles' read covectors (``read_rows``), what the S-lens pulls back and a write meets; ``vec`` ``[2f, d]``, each
    one's vector ``s² ⊙ cov`` at unit length, what the R-lens transports (``σ ⊙ (ŵ_gate ± ŵ_up)``, the raw rows times
    the residual's scale); ``s2`` ``[d]``, the metric at the MLP input."""
    w = directions(model_id, layer, dev)
    s2 = metric.s2(model_id, "mid", layer, dev)
    a, b = (w[k] / metric.cov_norm(w[k], s2) for k in ("gate", "up"))
    cov = read_rows(a, b, lobe)
    vec = metric.vector(cov, s2)
    return {"write": w["write"], "a": a, "b": b, "cov": cov, "vec": vec / vec.norm(dim=1, keepdim=True), "s2": s2}


def write_cosines(write: torch.Tensor, a: torch.Tensor, b: torch.Tensor, lobe: torch.Tensor,
                  s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos_gate, cos_up)``, ``[2f]`` each: the cosine, in the metric ``s2``, of each pole's write with the gate and up
    rows (``poles``' unit ``a``, ``b``), each on the side its firings sit (the gate below zero on the negative lobe, ``up``
    opposite the pole there), positive where the pole writes toward them."""
    w = write.T / metric.vec_norm(write.T, s2)
    x, y = (w * a).sum(1), (w * b).sum(1)
    return torch.cat([x, -x]) * lobe.to(x), torch.cat([y, y]) * lobe.to(x)


def orthogonal_write(write: torch.Tensor, a: torch.Tensor, b: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """``[d, f]``: each write minus its parts along its neuron's gate and up patterns (``s² ⊙ a``, ``s² ⊙ b``), so that
    neither row reads it, what it adds beyond anything it reads (``a``, ``b`` unit in the metric ``s2``, as ``poles``)."""
    pa, pb = metric.vector(a, s2), metric.vector(b, s2)
    w = write.T
    g = (a * pb).sum(1)
    ra, rb = (a * w).sum(1), (b * w).sum(1)
    det = (1 - g.pow(2)).clamp_min(1e-6)
    alpha, beta = (ra - g * rb) / det, (rb - g * ra) / det
    return (w - alpha[:, None] * pa - beta[:, None] * pb).T
