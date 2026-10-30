"""Read individual weight tensors from a Hub checkpoint's safetensors shards, no model load."""
import json

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open


def tensors(repo: str, names: list[str], revision: str | None = None) -> dict[str, torch.Tensor]:
    """``{name: tensor}`` in the checkpoint's dtype, fetching shards into the HF cache (or the
    one ``model.safetensors`` of an unsharded checkpoint, as Gemma 4 12B's)."""
    try:
        index = hf_hub_download(repo, "model.safetensors.index.json", revision=revision)
        with open(index) as f:
            shard_of = json.load(f)["weight_map"]
    except EntryNotFoundError:
        shard_of = {name: "model.safetensors" for name in names}
    out = {}
    for name in names:
        with safe_open(hf_hub_download(repo, shard_of[name], revision=revision), "pt") as f:
            out[name] = f.get_tensor(name)
    return out


def down_proj(layer: int) -> str:
    """Neuron ``n`` of layer ``L`` writes column ``n`` of this matrix (Qwen/Llama naming)."""
    return f"model.layers.{layer}.mlp.down_proj.weight"


def readout_names() -> list[str]:
    """The unembedding and the final norm, which every lens readout ends with."""
    return ["lm_head.weight", "model.norm.weight"]
