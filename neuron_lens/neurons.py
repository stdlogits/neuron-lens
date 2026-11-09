"""Each MLP neuron's read and write directions as the residual sees them, across models, for the
catalog stages (``08_reads.py``, ``09_legible.py``, ``14_media_tables.py``) and the checks.

A neuron ``n`` of layer ``L`` reads the MLP input through its gate and up rows times the norm that
feeds the MLP (Qwen3: ``post_attention_layernorm``; Gemma 4: ``pre_feedforward_layernorm``), and
writes column ``n`` of ``down_proj``. Gemma 4's MLP output then passes a post-feedforward RMSNorm,
whose scale the LN rule holds constant, and the layer's residual is multiplied by its
``layer_scalar``: what reaches the residual is ``s · γ_post ⊙ W_down[:, n]``, the write the
catalog reads. Qwen3's is ``W_down[:, n]`` itself.
"""

import torch

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
    ``a``, ``b`` ``[f, d]``: ``â ± b̂``, and its negative for the poles on SiLU's negative lobe (``lobe``,
    ``catalog.lobe_signs``). There the gate is below zero, and the pole's sign needs ``up`` opposite."""
    return torch.cat([a + b, a - b]) * lobe.to(a)[:, None]


def write_cosines(write: torch.Tensor, a: torch.Tensor, b: torch.Tensor, lobe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos_gate, cos_up)``, ``[2f]`` each: the cosine of each pole's write with the gate and up
    sides of what it reads (``read_rows``), positive where the pole writes toward them."""
    w = (write / write.norm(dim=0, keepdim=True)).T
    x, y = (w * a).sum(1), (w * b).sum(1)
    return torch.cat([x, -x]) * lobe.to(x), torch.cat([y, y]) * lobe.to(x)


def orthogonal_write(write: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``[d, f]``: each write with its components along its neuron's gate and up rows removed, what it
    adds beyond anything it reads. The rows are nearly orthogonal; the second is made exactly so."""
    b2 = b - (a * b).sum(1, keepdim=True) * a
    b2 = b2 / b2.norm(dim=1, keepdim=True)
    w = write.T
    return (w - (w * a).sum(1, keepdim=True) * a - (w * b2).sum(1, keepdim=True) * b2).T
