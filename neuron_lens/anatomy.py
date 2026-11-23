"""A decoder layer's anatomy, read from its module tree: its norms, its mixer and its MLP, the weights each unit reads
and writes, and their names in the checkpoint. The lens maps (``layers``), the rules (``rules``) and the catalog stages
read a layer through it, so a model family built from known parts needs no branch of its own.

Three layouts of norms (``Parts``):

- **pre-norm**: the mixer reads ``input_layernorm``, the MLP ``post_attention_layernorm`` (Qwen3, Qwen3.5, OLMoE,
  GPT-NeoX's parallel blocks, Olmo Hybrid's DeltaNet layers);
- **sandwich** (Gemma 4): ``post_attention_layernorm`` norms the attention's output, ``pre_feedforward_layernorm`` the
  MLP's input and ``post_feedforward_layernorm`` its output, and a ``layer_scalar`` multiplies the layer's output;
- **outputs only** (OLMo 2 and 3, Olmo Hybrid's attention layers): ``post_attention_layernorm`` and
  ``post_feedforward_layernorm`` norm the sublayers' outputs, and each sublayer reads the residual itself.

The same name, ``post_attention_layernorm``, is the MLP's input norm in the first and the attention output's norm in
the others, which is why nothing outside this module should read a norm by name.

A part is an attribute path within the decoder layer (``"mlp.down_proj"``). ``module`` resolves it on a loaded layer, or
on the layers of ``skeleton``'s meta-device copy of the model (the module tree, no weights); ``weight_name`` turns it
into the checkpoint's tensor name, for the stages that read weights without loading the model (``weights.tensors``):
the path below the decoder stack, after the checkpoint's own prefix for the stack (Qwen3.5's checkpoints keep the text
model under ``model.language_model.``, where its causal LM holds it at ``model.``).
"""
import functools
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Parts:
    n1: str | None  # the mixer's input norm; None: the mixer reads the residual itself
    post1: str | None  # the norm on the mixer's output
    n2: str | None  # the MLP's input norm; None: the MLP reads the residual itself
    post2: str | None  # the norm on the MLP's output
    mixer: str  # "self_attn", GPT-NeoX's "attention", or a Gated DeltaNet's "linear_attn"
    deltanet: bool
    out: str  # the mixer's output projection, whose input is the heads' outputs side by side
    sparse: bool  # routed experts ("mlp.experts", "mlp.gate" the router)
    gate: str | None  # a dense MLP's rows: its gate (GPT-NeoX's ungated input rows), up (None ungated) and down
    up: str | None
    down: str | None
    scalar: str | None  # what multiplies the layer's output (Gemma 4's "layer_scalar")
    parallel: bool  # both sublayers read the layer's input (GPT-NeoX's use_parallel_residual)

    @property
    def norms(self) -> list[str]:
        return [n for n in (self.n1, self.post1, self.n2, self.post2) if n is not None]


def parts(layer) -> Parts:
    """A decoder layer's parts by attribute path (module docstring)."""
    has = lambda path: module(layer, path) is not None
    if has("pre_feedforward_layernorm"):
        norms = ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm")
    elif not has("input_layernorm"):
        norms = (None, "post_attention_layernorm", None, "post_feedforward_layernorm")
    else:
        norms = ("input_layernorm", None, "post_attention_layernorm", None)
    mixer = next(m for m in ("linear_attn", "self_attn", "attention") if has(m))
    out = next(f"{mixer}.{p}" for p in ("o_proj", "out_proj", "dense") if has(f"{mixer}.{p}"))
    sparse = has("mlp.experts")
    first = lambda *ps: next((f"mlp.{p}" for p in ps if has(f"mlp.{p}")), None)
    return Parts(*norms, mixer=mixer, deltanet=mixer == "linear_attn", out=out, sparse=sparse,
                 gate=None if sparse else first("gate_proj", "dense_h_to_4h"), up=None if sparse else first("up_proj"),
                 down=None if sparse else first("down_proj", "dense_4h_to_h"),
                 scalar="layer_scalar" if has("layer_scalar") else None,
                 parallel=bool(getattr(layer, "use_parallel_residual", False)))


def module(layer, path: str | None):
    """The part at ``path`` (``None`` where the layer has none, or for ``path=None``)."""
    if path is None:
        return None
    for name in path.split("."):
        layer = getattr(layer, name, None)
        if layer is None:
            return None
    return layer


def mlp_reads(layer):
    """The module whose input is the residual the layer's MLP reads (its input norm, or the MLP where it has none):
    where a forward pre-hook sees ``mid``."""
    p = parts(layer)
    return module(layer, p.n2) if p.n2 is not None else layer.mlp


def mlp_writes(layer):
    """``(module, scalar)``: the module whose output is the layer's MLP write as it joins the residual (the norm on the
    MLP's output, or the MLP), and what then multiplies it (Gemma 4's ``layer_scalar``; ``None``)."""
    p = parts(layer)
    return (module(layer, p.post2) if p.post2 is not None else layer.mlp), module(layer, p.scalar)


def norm_scale(layer, path: str | None, dim: int) -> torch.Tensor:
    """A norm's scale as the rules read it (``rules.norm_weight``: ``1 + w`` for zero-centred norms), in fp32, or ones
    where the layer has no norm there."""
    from neuron_lens.rules import norm_weight
    norm = module(layer, path)
    w = None if norm is None else norm_weight(norm)
    p = next(layer.parameters())
    return torch.ones(dim, device=p.device) if w is None else w.float()


# ---------------------------------------------------------------------------------- weight-only reads

@functools.cache
def skeleton(model_id: str):
    """``(causal LM, decoder stack)`` of the model on the meta device: its module tree without weights (about a second
    to build, any size)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from neuron_lens.transport import decoder_stack
    with torch.device("meta"):
        lm = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_id))
    return lm, decoder_stack(lm)


@functools.cache
def checkpoint_keys(model_id: str) -> frozenset[str]:
    """The checkpoint's tensor names, from its safetensors headers (no weights fetched)."""
    from huggingface_hub import get_safetensors_metadata
    return frozenset(get_safetensors_metadata(model_id).weight_map)


def _path(root, target) -> str:
    return next(n for n, m in root.named_modules() if m is target)


@functools.cache
def stack_prefix(model_id: str) -> str:
    """The checkpoint's prefix for the decoder stack's tensors (``model.``, ``model.language_model.``, ``gpt_neox.``):
    where its token embedding sits, less the embedding's own name in the stack."""
    _, stack = skeleton(model_id)
    rel = _path(stack, stack.get_input_embeddings()) + ".weight"
    found = sorted((k for k in checkpoint_keys(model_id) if k == rel or k.endswith("." + rel)), key=len)
    if not found:
        raise KeyError(f"{model_id}: no {rel} in the checkpoint")
    return found[0][:-len(rel)]


def weight_name(model_id: str, layer: int, path: str, what: str = "weight") -> str:
    """The checkpoint's name for a part's tensor (``what``: ``"weight"``, ``"bias"``, or ``""`` for a parameter itself,
    such as ``layer_scalar``) in decoder layer ``layer``."""
    name = f"{stack_prefix(model_id)}layers.{layer}.{path}" + (f".{what}" if what else "")
    if name not in checkpoint_keys(model_id):
        raise KeyError(f"{model_id}: no {name} in the checkpoint")
    return name


def layer_parts(model_id: str, layer: int) -> Parts:
    """``parts`` of decoder layer ``layer``, from the skeleton."""
    return parts(skeleton(model_id)[1].layers[layer])


@functools.cache
def top_names(model_id: str) -> dict:
    """The token embedding's, unembedding's and final norm's tensor names (``embed``, ``unembed``, ``norm``); a tied
    unembedding is the embedding (Gemma 4, Qwen3 to 4B, Qwen3.5 to 4B)."""
    from neuron_lens.rules import final_norm
    lm, stack = skeleton(model_id)
    pre = stack_prefix(model_id)
    embed = pre + _path(stack, stack.get_input_embeddings()) + ".weight"
    keys = checkpoint_keys(model_id)
    tied = lm.get_output_embeddings().weight is stack.get_input_embeddings().weight
    head = next((k for k in (_path(lm, lm.get_output_embeddings()) + ".weight", "embed_out.weight") if k in keys), None)
    return {"embed": embed, "unembed": embed if tied or head is None else head,
            "norm": pre + _path(stack, final_norm(stack)) + ".weight"}
