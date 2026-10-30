"""Lens loading and export: this project's fits (``data/fits/``, the authoritative format),
R-lenses in Drowse homes, Neuronpedia J-lenses, merging. Drowse (the ``drowse`` extra) is
imported only where a Drowse lens is read or written."""
import contextlib
import hashlib
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from neuron_lens.paths import FITS, LENS_HOMES
from neuron_lens.transport import SKIP_FIRST, Accumulator

NEURONPEDIA_REPO = "neuronpedia/jacobian-lens"
NEURONPEDIA_FILES = {
    "Qwen/Qwen3-8B": "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
}


@contextlib.contextmanager
def drowse_home(path: Path | None):
    """Point Drowse at ``path`` for the duration; ``None`` keeps the user's default home."""
    old = os.environ.get("DROWSE_HOME")
    if path is not None:
        os.environ["DROWSE_HOME"] = str(path)
    try:
        yield
    finally:
        if path is not None:
            if old is None:
                os.environ.pop("DROWSE_HOME", None)
            else:
                os.environ["DROWSE_HOME"] = old


def load_r_lens(model_id: str, fit: str | None):
    """The R-lens of ``data/lenses/<fit>/`` (``None``: the default Drowse home's)."""
    from drowse.lens.store import load_local_lens

    with drowse_home(None if fit is None else LENS_HOMES / fit):
        loaded = load_local_lens(model_id, "r")
    if loaded is None:
        raise FileNotFoundError(f"no R-lens for {model_id} in fit {fit!r}")
    return loaded


def merge(lenses: list):
    """Prompt-weighted mean of lenses over the same layers: exactly the lens a single fit
    over the union of their (disjoint) corpora would give, since the estimator is a mean
    of per-prompt Jacobians. The lenses must be the same kind at the same site; the merge
    keeps that key."""
    from drowse.lens.jlens import JacobianLens

    key = lenses[0].key
    if any(lens.key != key for lens in lenses):
        raise ValueError("lenses are of different kinds: " + ", ".join(sorted({lens.key.label for lens in lenses})))
    layers = lenses[0].source_layers
    if any(lens.source_layers != layers for lens in lenses):
        raise ValueError("lenses cover different source layers")
    total = sum(lens.n_prompts for lens in lenses)
    jac = {l: sum(lens.jacobians[l] * (lens.n_prompts / total) for lens in lenses) for l in layers}
    return JacobianLens(jac, n_prompts=total, d_model=lenses[0].d_model, key=key)


def load_lens(spec: str, model_id: str) -> dict | None:
    """A lens spec's Jacobians by layer: ``logit`` (``None``, the identity), ``J`` (Neuronpedia's
    J-lens), ``R`` (the default Drowse home's R-lens), ``R:<fit>`` (``data/lenses/<fit>``),
    ``R:<fit1>+<fit2>`` (their prompt-weighted merge) or ``fit:<stem>[@mid]`` (an R-lens fit file
    ``data/fits/<stem>.pt`` itself, both halves: its ``out`` target, from each layer's output, or
    with ``@mid`` from each layer's MLP input, keyed by that layer) or ``hub[@mid]`` (the released
    R-lens in ``stdlogits/rs-lenses``, ``12_release.py``: from each layer's output, the last
    layer the identity, or from each MLP input). ``logit``, ``J``, ``fit:`` and ``hub`` need no Drowse."""
    if spec == "logit":
        return None
    if spec.partition("@")[0] == "hub":
        from neuron_lens.arch import stem
        f = "r-lens/mid.pt" if spec.endswith("@mid") else "r-lens/lens.pt"
        return torch.load(hf_hub_download("stdlogits/rs-lenses", f"{stem(model_id)}/{f}"), map_location="cpu",
                          weights_only=True, mmap=True)["J"]
    if spec == "J":
        payload = torch.load(hf_hub_download(NEURONPEDIA_REPO, NEURONPEDIA_FILES[model_id]),
                             map_location="cpu", weights_only=True)
        return payload["J"]
    if spec.startswith("fit:"):
        stem, _, target = spec.removeprefix("fit:").partition("@")
        _, acc = load_fits([FITS / f"{stem}.pt"], "R", model_id)
        J = acc.mean(target or "out")
        return {l: J[l] for l in range(J.shape[0])}
    if spec == "R":
        return load_r_lens(model_id, None)[0].jacobians
    return merge([load_r_lens(model_id, f)[0] for f in spec.removeprefix("R:").split("+")]).jacobians


def load_fits(paths: list[Path], lens: str, model_id: str | None = None) -> tuple[list[dict], Accumulator]:
    """Fits of one lens (``02_fit.py`` files, e.g. shards or corpora), summed. Their metas
    must agree except for corpus, shard and ``dim_batch``. Each must be a ``lens`` fit (``R`` or
    ``S``: both are ``[layers, d, d]``, so one would load silently as the other) of every layer of
    ``model_id`` (a truncated fit would leave later layers to the logit lens)."""
    metas, acc = [], None
    for path in paths:
        saved = torch.load(path, weights_only=False, map_location="cpu")
        meta = saved["meta"]
        if meta["lens"] != lens or meta["layers"] != meta["model_layers"] or model_id not in (None, meta["model"]):
            raise ValueError(f"{path} is a {meta['lens']} fit of {meta['layers']}/{meta['model_layers']} layers of "
                             f"{meta['model']}, not a whole {lens} fit of {model_id or 'its model'}")
        metas.append(meta)
        part = Accumulator.from_state(saved["state"])
        if acc is None:
            acc = part
        else:
            acc.merge(part)
    skip = {"corpus", "corpus_spec", "rows_sha256", "docs", "shard", "dim_batch"}
    if len({repr({k: v for k, v in m.items() if k not in skip}) for m in metas}) > 1:
        raise ValueError("fits disagree on more than corpus and shard")
    return metas, acc


def save_drowse_lens(metas: list[dict], acc: Accumulator, home: Path, half: int | None = None) -> Path:
    """Write an R-lens fit as a Drowse lens (``local/r``) under the Drowse home ``home``: all
    its documents, or one half of them (0 even, 1 odd) for a split-half comparison."""
    from drowse.lens.jlens import JacobianLens
    from drowse.lens.key import LensKey
    from drowse.lens.store import LensProvenance, save_lens

    meta = metas[0]
    if meta["lens"] != "R" or meta["layers"] != meta["model_layers"]:
        raise ValueError("Drowse holds R-lenses of the whole model only")
    J = acc.mean("out", half)
    n = sum(acc.n_docs) if half is None else acc.n_docs[half]
    lens = JacobianLens({l: J[l] for l in range(J.shape[0])}, n_prompts=n, d_model=J.shape[-1], key=LensKey("R"))
    corpora = sorted({(m["corpus_spec"], m["rows_sha256"]) for m in metas})
    sha = corpora[0][1] if len(corpora) == 1 else hashlib.sha256(repr(corpora).encode()).hexdigest()
    provenance = LensProvenance(
        corpus_spec="+".join(c for c, _ in corpora), corpus_sha256=sha, corpus_hash_kind="token_ids_v1",
        seq_len=meta["seq_len"], dim_batch=meta["dim_batch"], skip_first=SKIP_FIRST,
        usable_prompt_count=n, model_layer_count=meta["model_layers"])
    with drowse_home(home):
        return save_lens(lens, meta["model"], provenance)
