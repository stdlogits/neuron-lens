"""Read individual weight tensors from a Hub checkpoint's safetensors shards, no model load."""
from collections import defaultdict
import functools
import json

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open


@functools.lru_cache(maxsize=None)
def _shard_path(repo: str, filename: str, revision: str | None) -> str:
    return hf_hub_download(repo, filename, revision=revision)


@functools.lru_cache(maxsize=None)
def _weight_map(repo: str, revision: str | None) -> dict[str, str] | None:
    try:
        with open(_shard_path(repo, "model.safetensors.index.json", revision)) as f:
            return json.load(f)["weight_map"]
    except EntryNotFoundError:
        return None


def tensors(repo: str, names: list[str], revision: str | None = None) -> dict[str, torch.Tensor]:
    """``{name: tensor}`` in the checkpoint's dtype, fetching shards into the HF cache (or the
    one ``model.safetensors`` of an unsharded checkpoint, as Gemma 4 12B's)."""
    shard_of = _weight_map(repo, revision)
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name in names:
        by_shard["model.safetensors" if shard_of is None else shard_of[name]].append(name)
    out = {}
    for shard, keys in by_shard.items():
        with safe_open(_shard_path(repo, shard, revision), "pt") as f:
            for name in keys:
                out[name] = f.get_tensor(name)
    return out


def down_proj(layer: int) -> str:
    """Neuron ``n`` of layer ``L`` writes column ``n`` of this matrix (Qwen/Llama naming)."""
    return f"model.layers.{layer}.mlp.down_proj.weight"


def readout_names() -> list[str]:
    """The unembedding and the final norm, which every lens readout ends with."""
    return ["lm_head.weight", "model.norm.weight"]
