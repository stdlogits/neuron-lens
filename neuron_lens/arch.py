"""Model geometry across Qwen3, Gemma 4 and GPT-NeoX (Pythia): attention heads' sizes, weights and lenses, the
corpora and token counts, for the pipeline's Gemma path and the media checks.

Three facts about a model's layers that the stages read instead of branching on the model: ``n_poles``, a
neuron's poles (two for a gated MLP, whose ``up`` takes both signs; one for GPT-NeoX's ungated GELU, whose
negative side is a dip to −0.17, not a second direction); ``parallel``, whether both sublayers read the layer's
input (GPT-NeoX: the MLP reads the layer's input, so ``mid`` is the layer below's ``out``, and the heads write
``out``); and ``centred``, whether the residual's norms are LayerNorms (a read row then acts on the centred
residual, so its mean is subtracted).

Per layer, ``layout(model_id)`` gives the query and key-value head counts and the head size (Gemma 4
varies them by layer: sliding layers 8 KV heads of 256, full layers 1 of 512), whether the value
is the raw key (Gemma 4's full layers, ``attention_k_eq_v``), and the weight names. ``ov(...)``
gives a layer's OV weights as the residual sees them: ``W_V diag(γ_in)`` (rows: KV heads × head
size) and ``diag(γ_post) W_O``: Gemma 4's attention output passes a post-attention RMSNorm before
the residual, whose scale the LN rule holds constant, so its weight multiplies the write (Qwen3
has none). Gemma 4's values also pass a weightless RMSNorm per head, ``v / rms(v)``: a token's
value keeps its direction and loses its size, which scales each column of a token map and leaves
its per-token z unchanged; the eigenvalue scores leave it out, as they leave the input norm's
scale out.

``lens_means(model_id, ...)`` loads the lens means the checks read: from fit files (``02_fit.py``)
when present, else the released lenses on the Hub (``stdlogits/rs-lenses``, ``12_release.py``).
"""
import dataclasses
import functools
import json
from dataclasses import dataclass

import torch
from transformers import AutoConfig

from neuron_lens.paths import DATA, FITS, SUMMARY
from neuron_lens.rules import attention, is_centred, norm_eps
from neuron_lens.weights import tensors

QWEN = "Qwen/Qwen3-8B"


@dataclass
class Layer:
    q_heads: int
    kv_heads: int
    head_dim: int
    k_eq_v: bool  # the value is the raw key (no v_proj)
    kind: str  # "sliding_attention", "full_attention" (Gemma 4) or "attention"
    prefix: str  # the layer's weight name prefix


def stem(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


@functools.cache
def is_gemma4(model_id: str) -> bool:
    return AutoConfig.from_pretrained(model_id).model_type.startswith("gemma4")


@functools.cache  # AutoConfig asks the Hub each time: these sit in inner loops
def text_config(model_id: str):
    return AutoConfig.from_pretrained(model_id).get_text_config()


@functools.cache
def is_neox(model_id: str) -> bool:
    """GPT-NeoX (Pythia): its own weight names, a fused QKV projection, LayerNorms, an ungated MLP."""
    return text_config(model_id).model_type == "gpt_neox"


def n_poles(model_id: str) -> int:
    return 1 if is_neox(model_id) else 2


def signs(model_id: str) -> tuple[int, ...]:
    """The poles' signs, in catalog order: ``(1, −1)``, or ``(1,)`` for an ungated MLP."""
    return (1, -1)[:n_poles(model_id)]


def parallel(model_id: str) -> bool:
    return bool(getattr(text_config(model_id), "use_parallel_residual", False))


def centred(model_id: str) -> bool:
    return is_neox(model_id)


def centre(x: torch.Tensor, model_id: str, dim: int = -1) -> torch.Tensor:
    """Rows ``[n, d]`` (or columns, ``dim=0``) as the residual's readers see them: a LayerNorm reads the centred
    residual, so a read row and the row minus its mean read it alike, and a write's mean reaches no reader."""
    return x - x.mean(dim, keepdim=True) if centred(model_id) else x


@functools.cache
def layout(model_id: str) -> list[Layer]:
    return layout_of(text_config(model_id), is_gemma4(model_id))


def layout_of(tc, gemma: bool) -> list[Layer]:
    if tc.model_type == "gpt_neox":  # no grouping; the weights under gpt_neox.
        hd = tc.hidden_size // tc.num_attention_heads
        return [Layer(tc.num_attention_heads, tc.num_attention_heads, hd, False, "attention", f"gpt_neox.layers.{i}.")
                for i in range(tc.num_hidden_layers)]
    if gemma:
        return [Layer(tc.num_attention_heads, c.num_key_value_heads, c.head_dim, bool(t == "full_attention" and tc.attention_k_eq_v),
                      t, f"model.language_model.layers.{i}.")
                for i, (t, c) in enumerate(zip(tc.layer_types, tc.per_layer_config))]
    return [Layer(tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim, False, "attention", f"model.layers.{i}.")
            for i in range(tc.num_hidden_layers)]


def names(model_id: str) -> dict:
    """The embedding, unembedding and final norm weights' names."""
    if is_neox(model_id):  # untied
        return {"embed": "gpt_neox.embed_in.weight", "unembed": "embed_out.weight", "norm": "gpt_neox.final_layer_norm.weight"}
    if is_gemma4(model_id):
        e = "model.language_model.embed_tokens.weight"
        return {"embed": e, "unembed": e, "norm": "model.language_model.norm.weight"}  # tied
    e = "model.embed_tokens.weight"  # Qwen3 to 4B ties the unembedding to it, and stores no lm_head
    return {"embed": e, "unembed": e if text_config(model_id).tie_word_embeddings else "lm_head.weight",
            "norm": "model.norm.weight"}


def chat_model(model_id: str) -> str:
    """The model whose chat template renders the chat slice (``05_act_corpus.py``): its own, except that a Qwen3 base
    reads its post-trained sibling's transcripts, so the two meet the same text (the bases' own template differs)."""
    return model_id.removesuffix("-Base") if model_id.startswith("Qwen/Qwen3-") else model_id


def half_dtype(config) -> torch.dtype:
    """The half dtype a checkpoint is stored in, which GPU passes run it in: bf16 (Qwen3, Gemma 4), or fp16 (Pythia,
    trained in fp16, whose bf16-rounded weights move late layers' attention by up to 0.2 in probability)."""
    dt = getattr(config, "dtype", None)
    dt = getattr(torch, dt) if isinstance(dt, str) else dt
    return dt if dt in (torch.float16, torch.bfloat16) else torch.bfloat16


def embed_scale(model_id: str) -> float:
    """What the embedding layer multiplies token embeddings by: ``√d`` for Gemma, else 1."""
    return text_config(model_id).hidden_size ** 0.5 if is_gemma4(model_id) else 1.0


def _qkv(model_id: str, lay: Layer, dev) -> dict:
    """GPT-NeoX's fused projection as ``Wq``, ``Wk``, ``Wv`` ``[H·hd, d]`` and ``bq`` ``[H·hd]``, in fp32."""
    p = lay.prefix + "attention.query_key_value."
    w = tensors(model_id, [p + "weight", p + "bias"])
    W = w[p + "weight"].float().to(dev).view(lay.q_heads, 3, lay.head_dim, -1)
    return {"Wq": W[:, 0].reshape(-1, W.shape[-1]), "Wk": W[:, 1].reshape(-1, W.shape[-1]),
            "Wv": W[:, 2].reshape(-1, W.shape[-1]),
            "bq": w[p + "bias"].float().to(dev).view(lay.q_heads, 3, lay.head_dim)[:, 0].reshape(-1)}


def ov(model_id: str, lay: Layer, dev="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """``(W_V diag(γ_in) [kv·hd, d], diag(γ_post) W_O [d, H·hd])`` in fp32 (the first ``centre``d)."""
    p = lay.prefix
    if is_neox(model_id):
        ln, wo = p + "input_layernorm.weight", p + "attention.dense.weight"
        w = {k: v.float().to(dev) for k, v in tensors(model_id, [ln, wo]).items()}
        return centre(_qkv(model_id, lay, dev)["Wv"] * w[ln], model_id), centre(w[wo], model_id, 0)
    wv = p + ("self_attn.k_proj.weight" if lay.k_eq_v else "self_attn.v_proj.weight")
    want = [wv, p + "self_attn.o_proj.weight", p + "input_layernorm.weight"]
    gemma = is_gemma4(model_id)
    if gemma:
        want.append(p + "post_attention_layernorm.weight")
    w = {k: v.float().to(dev) for k, v in tensors(model_id, want).items()}
    Wo = w[p + "self_attn.o_proj.weight"]
    if gemma:
        Wo = w[p + "post_attention_layernorm.weight"][:, None] * Wo
    return w[wv] * w[p + "input_layernorm.weight"], Wo


def qk(model_id: str, lay: Layer, dev="cpu") -> dict:
    """``Wq``, ``Wk`` (times ``γ_in``, ``centre``d) and the q/k norm weights (``None`` without q/k norms), in fp32; GPT-NeoX's
    query bias ``bq`` too (the key bias adds the same score to every key of a query, and cancels)."""
    p = lay.prefix
    if is_neox(model_id):
        ln = p + "input_layernorm.weight"
        g1 = tensors(model_id, [ln])[ln].float().to(dev)
        w = _qkv(model_id, lay, dev)
        return {"Wq": centre(w["Wq"] * g1, model_id), "Wk": centre(w["Wk"] * g1, model_id), "q_norm": None,
                "k_norm": None, "bq": w["bq"]}
    w = tensors(model_id, [p + f"self_attn.{n}" for n in ("q_proj.weight", "k_proj.weight", "q_norm.weight", "k_norm.weight")]
                + [p + "input_layernorm.weight"])
    w = {k: v.float().to(dev) for k, v in w.items()}
    g1 = w[p + "input_layernorm.weight"]
    return {"Wq": w[p + "self_attn.q_proj.weight"] * g1, "Wk": w[p + "self_attn.k_proj.weight"] * g1,
            "q_norm": w[p + "self_attn.q_norm.weight"], "k_norm": w[p + "self_attn.k_norm.weight"], "bq": None}


def kv_of(lay: Layer, h: int) -> int:
    return h // (lay.q_heads // lay.kv_heads)


def head_slices(lay: Layer, h: int) -> tuple[slice, slice]:
    """Query head ``h``'s rows of ``W_V`` (its KV head) and columns of ``W_O``."""
    kv, hd = kv_of(lay, h), lay.head_dim
    return slice(kv * hd, (kv + 1) * hd), slice(h * hd, (h + 1) * hd)


def lens_means(model_id: str, r_fit: str | None = None, s_fit: str | None = None, sub: str = "",
               parts=("K_out", "K_mid", "J_out", "J_mid")) -> dict:
    """``K_out``, ``K_mid`` (``[L, d, d_in]``: embeddings, or for the image and audio S-lenses
    ``basis`` directions, to each layer's output and MLP input), ``J_out`` (``[L − 1, d, d]``, each
    layer's output below the last) and ``J_mid`` (``[L, d, d]``). From the fit files when given and
    present, else the release; ``sub``: "" (text), "image/" or "audio/" (released media lenses)."""
    fit = {"K": s_fit, "J": r_fit}
    if all(fit[k[0]] and (FITS / f"{fit[k[0]]}.pt").exists() for k in parts):
        from neuron_lens.lenses import load_fits
        out = {}
        for lens in ("K", "J"):
            want = [k for k in parts if k[0] == lens]
            if want:
                _, acc = load_fits([FITS / f"{fit[lens]}.pt"], "S" if lens == "K" else "R", model_id)
                out |= {k: acc.mean(k[2:]) for k in want}
                del acc
        return out
    from huggingface_hub import hf_hub_download
    files = {"K_out": "s-lens/out.pt", "K_mid": "s-lens/mid.pt", "J_out": "r-lens/lens.pt", "J_mid": "r-lens/mid.pt"}
    n = text_config(model_id).num_hidden_layers
    out = {}
    for k in parts:
        x = torch.load(hf_hub_download("stdlogits/rs-lenses", f"{stem(model_id)}/{sub}{files[k]}"),
                       weights_only=True, map_location="cpu", mmap=True)
        m = x["K" if k.startswith("K") else "J"]
        out[k] = torch.stack([m[l].float() for l in range(n - 1 if k == "J_out" else n)])
        if "basis" in x:
            out["basis"] = x["basis"].float()
    return out


def counts_path(model_id: str):
    """``03_token_counts.py``'s output for this model's tokenizer."""
    return DATA / f"token_counts-{stem(model_id)}.pt"


EAGER_FP32 = "eager_fp32"


def _eager_fp32(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Eager attention with its scores and softmax in fp32, returning the weights: HF's eager attention forms q·k in
    the model's dtype, which fp16 overflows on Pythia (its loss turns NaN)."""
    g = query.shape[1] // key.shape[1]
    if g > 1:
        key, value = key.repeat_interleave(g, 1), value.repeat_interleave(g, 1)
    w = torch.matmul(query.float(), key.float().transpose(2, 3)) * scaling
    if attention_mask is not None:
        m = attention_mask[:, :, :, :key.shape[-2]]
        w = w.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else w + m.float()
    w = torch.softmax(w, dim=-1)
    return torch.matmul(w, value.float()).to(query.dtype).transpose(1, 2).contiguous(), w.to(query.dtype)


@functools.cache
def _register_eager_fp32() -> None:
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, eager_mask
    AttentionInterface.register(EAGER_FP32, _eager_fp32)
    AttentionMaskInterface.register(EAGER_FP32, eager_mask)


def attn_implementation(dtype: torch.dtype) -> str:
    """Eager attention (for its patterns): HF's, or for fp16 its scores in fp32 (``_eager_fp32``)."""
    if dtype != torch.float16:
        return "eager"
    _register_eager_fp32()
    return EAGER_FP32


def load_model(model_id: str, dev: str, eager: bool = True):
    """``(lm, stack)``: the model in its half dtype (``half_dtype``; eager attention, for its patterns) and its decoder
    stack."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from neuron_lens.transport import decoder_stack
    dt = half_dtype(AutoConfig.from_pretrained(model_id))
    kw = {"attn_implementation": attn_implementation(dt)} if eager else {}
    lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, **kw).to(dev).eval().requires_grad_(False)
    return lm, decoder_stack(lm)


def logits(lm, hidden: torch.Tensor) -> torch.Tensor:
    """Final logits in fp32 from the stack's last (normed) hidden state, with Gemma's softcap."""
    x = lm.get_output_embeddings()(hidden).float()
    cap = getattr(lm.config.get_text_config(), "final_logit_softcapping", None)
    return torch.tanh(x / cap) * cap if cap else x


# ---------------------------------------------------------------- depth: the reading plans and the bands

@dataclass(frozen=True)
class Plan:
    """A model's reading plan (findings §8): which read a label card shows at each depth, and where its prompt's
    lines change. The input edges were set by reading: two raters of different lineages rated each weight read's
    coherence, blind to the lens, at 15-16 layers of each model (``checks/depth_reads.py``), and the plan takes the most
    coherent. The write read is the R-lens's at every depth (it also predicts steering, §3, §19). The head lines follow
    each model's census of head reads (§10, §11, §19); the next-word edge is Qwen3-8B's L21 at the same relative depth
    in the other Qwen3 sizes, and Gemma 4's L33 (§15). ``10_labels.py`` writes the lines; the depth bands (``bands``)
    and the site's are cut from the same edges."""
    layers: int
    s_last: int  # the input read is the S-lens's through this layer (-1: at no depth)
    assoc_last: int  # then the R-lens's, with the line that it often names an associated concept, through this layer
    write_edge: int  # writes before this layer get the line that steering follows them less closely (0: none)
    late: int  # writes from this layer often name the next word, and a peaked write gets its note (§5)
    heads_early: int  # the early heads' line through this layer, their OV read ``heads_early_read``'s
    heads_late: int  # the late heads' line from this layer
    s_read: str = "read_S"  # the S-lens input read: plain, or its z read (Qwen3-4B)
    heads_early_read: str = "ov_S"


PLANS = {
    "qwen3-0.6b": Plan(28, s_last=5, assoc_last=8, write_edge=0, late=16, heads_early=3, heads_late=20),
    "qwen3-1.7b": Plan(28, s_last=9, assoc_last=18, write_edge=0, late=16, heads_early=3, heads_late=20),
    "qwen3-4b": Plan(36, s_last=5, assoc_last=8, write_edge=3, late=21, heads_early=4, heads_late=26, s_read="read_S_z"),
    "qwen3-8b": Plan(36, s_last=3, assoc_last=8, write_edge=4, late=21, heads_early=8, heads_late=28),
    "qwen3-14b": Plan(40, s_last=8, assoc_last=11, write_edge=7, late=23, heads_early=4, heads_late=28),
    "gemma-4-12b": Plan(48, s_last=-1, assoc_last=19, write_edge=11, late=33, heads_early=13, heads_late=33,
                        heads_early_read="ov_R"),
}
ROLES = ("sensory", "association", "workspace", "motor")


def plan(model_id: str) -> Plan | None:
    """The model's reading plan; a base model takes its post-trained model's (§19: the same neurons, close reads)."""
    return PLANS.get(stem(model_id).removesuffix("-base"))


def depth_bands(model_id: str) -> dict[str, tuple[int, int]]:
    """The model's depth bands, ``{role: (first, last)}``, cut from its reading plan at the edges where a card's read or
    its line changes: sensory while the input read is the S-lens's, association while the R-lens's carries the
    association line, workspace after, and motor from where writes often name the next word. Motor holds where it
    starts first (Qwen3-1.7B's association line runs to L18 and its next-word writes from L16: no workspace band); a
    model without an S-lens input read (Gemma 4) has no sensory band. A model without a plan (Pythia, Qwen3-32B, Qwen3.5)
    gets quarters, named by their layers."""
    p = plan(model_id)
    if p is None:
        n = text_config(model_id).num_hidden_layers
        q = n // 4
        return {f"L{lo}-L{hi}": (lo, hi) for lo, hi in ((i * q, (i + 1) * q - 1 if i < 3 else n - 1) for i in range(4))}
    role = lambda L: ("motor" if L >= p.late else "sensory" if L <= p.s_last else "association" if L <= p.assoc_last
                      else "workspace")
    out: dict[str, tuple[int, int]] = {}
    for L in range(p.layers):
        out[role(L)] = (out.get(role(L), (L, L))[0], L)
    assert list(out) == [r for r in ROLES if r in out], f"{model_id}: bands out of order"
    return out


def bands(model_id: str) -> list[tuple[int, int]]:
    """The model's depth bands, ``[(first, last)]`` in depth order (``depth_bands``), for the summaries."""
    return list(depth_bands(model_id).values())


def write_depth(path=None) -> None:
    """Every plan and its bands as JSON (``data/summary/depth.json``), for what reads them outside Python: the site's
    catalog build and its words for each band. Rerun after changing ``PLANS``: ``python -m neuron_lens.arch``."""
    out = {m: {"plan": dataclasses.asdict(p),
               "bands": [{"role": r, "first": a, "last": b} for r, (a, b) in depth_bands(m).items()]}
           for m, p in PLANS.items()}
    path = path or SUMMARY / "depth.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{path}: " + "; ".join(f"{m} " + " ".join(f"{b['role']} L{b['first']}-{b['last']}" for b in v["bands"])
                                  for m, v in out.items()))


# ---------------------------------------------------------------- the pipeline's Gemma path

def groups(model_id: str) -> list[int]:
    """Query heads per key-value head, per layer."""
    return [x.q_heads // x.kv_heads for x in layout(model_id)]


def padded(model_id: str) -> tuple[int, int]:
    """``(key-value heads, head size)``, the largest over layers: per-layer statistics are kept in
    buffers this size, each layer in its leading block (Gemma 4: 8 heads of 256, or 1 of 512)."""
    lay = layout(model_id)
    return max(x.kv_heads for x in lay), max(x.head_dim for x in lay)


def attn_norm(layer):
    """The norm between a decoder layer's attention output and the residual (Gemma 4's post-attention
    norm), or ``None`` where the heads write the residual directly (Qwen3)."""
    return layer.post_attention_layernorm if hasattr(layer, "pre_feedforward_layernorm") else None


def mlp_norm(layer):
    """The norm between a decoder layer's MLP output and the residual (Gemma 4's post-feedforward
    norm), or ``None`` where the neurons write the residual directly (Qwen3)."""
    return getattr(layer, "post_feedforward_layernorm", None)


def _scale(norm, x: torch.Tensor) -> torch.Tensor | None:
    return None if norm is None else torch.rsqrt(x.float().pow(2).mean(-1) + norm_eps(norm))


def head_write_lens(model_id: str, side: str, fit: str | None) -> torch.Tensor:
    """Per layer, the lens (``side`` ``R`` or ``S``) at the point the layer's heads write: its MLP input (``mid``), or in
    a parallel block its output (``out``, the R-lens's last one the identity: the last layer's output is its target)."""
    if not parallel(model_id):
        return lens(model_id, f"{side}_mid", fit)
    out = lens(model_id, f"{side}_out", fit)
    return torch.cat([out, torch.eye(out.shape[-1])[None]]) if side == "R" else out


def head_write(layer) -> torch.Tensor:
    """``[d, H·hd]``: a decoder layer's ``W_O`` as its heads' outputs reach the residual, in fp32:
    times the norm weight where a norm follows the attention (its scale per position: ``attn_scale``)."""
    from neuron_lens.layers import _out_proj
    w = _out_proj(attention(layer)).weight.float()
    if is_centred(layer.input_layernorm):  # a write's mean reaches no reader (``centre``)
        w = w - w.mean(0)
    norm = attn_norm(layer)
    return w if norm is None else norm.weight.float()[:, None] * w


def attn_scale(layer, attn_out: torch.Tensor) -> torch.Tensor | None:
    """``[B, T]``: the scale of the norm after the attention at each position (held constant, the LN
    rule), from the attention output; ``None`` where no norm follows it."""
    return _scale(attn_norm(layer), attn_out)


def mlp_write(layer) -> torch.Tensor:
    """``[d, f]``: a decoder layer's ``W_down`` as its neurons' outputs reach the residual, in fp32:
    times the norm weight where a norm follows the MLP (its scale per position: ``mlp_scale``)."""
    w = (layer.mlp.down_proj if hasattr(layer.mlp, "down_proj") else layer.mlp.dense_4h_to_h).weight.float()
    if is_centred(layer.input_layernorm):
        w = w - w.mean(0)
    norm = mlp_norm(layer)
    return w if norm is None else norm.weight.float()[:, None] * w


def mlp_scale(layer, mlp_out: torch.Tensor) -> torch.Tensor | None:
    """``[B, T]``: the scale of the norm after the MLP at each position, from the MLP's output; ``None``
    where no norm follows it."""
    return _scale(mlp_norm(layer), mlp_out)


def layer_scale(layer) -> float:
    """What multiplies a decoder layer's whole output (Gemma 4's ``layer_scalar``; 1 otherwise)."""
    s = getattr(layer, "layer_scalar", None)
    return 1.0 if s is None else float(s)


def mid_norm(layer):
    """The norm that feeds a decoder layer's MLP (its input is the residual the ``mid`` lens targets read)."""
    return layer.pre_feedforward_layernorm if hasattr(layer, "pre_feedforward_layernorm") else layer.post_attention_layernorm


def mlp_output(layer):
    """``(module, scalar)``: the module whose output is the layer's MLP write, and what multiplies it on
    the way to the residual (Gemma 4's ``layer_scalar``; ``None`` for Qwen3)."""
    if hasattr(layer, "pre_feedforward_layernorm"):
        return layer.post_feedforward_layernorm, layer.layer_scalar
    return layer.mlp, None


def bos(tok) -> list[int]:
    """``[bos]`` if the tokenizer starts every document with it (Gemma), else ``[]`` (Qwen3)."""
    return [tok.bos_token_id] if tok.bos_token_id is not None and tok("a")["input_ids"][0] == tok.bos_token_id else []


def chat_open(model_id: str, tok) -> int:
    """The token a chat transcript opens with: Qwen3's ``<|im_start|>``; for a model without a chat
    template, the first token of ``05_act_corpus.py``'s plain ``User:`` turns."""
    if tok.chat_template:
        return tok.convert_tokens_to_ids("<|im_start|>") if "<|im_start|>" in tok.get_vocab() else \
            tok.apply_chat_template([{"role": "user", "content": "a"}], tokenize=True)[len(bos(tok))]
    return tok("User:", add_special_tokens=False)["input_ids"][0]


def inv_freq(stack, model_id: str):
    """RoPE's angle per position: one tensor (Qwen3), or one per layer type (Gemma 4)."""
    rot = stack.rotary_emb
    if is_gemma4(model_id):
        return {t: getattr(rot, f"{t}_inv_freq").float().cpu() for t in {x.kind for x in layout(model_id)}}
    return rot.inv_freq.float().cpu()


def layer_inv_freq(freqs, model_id: str, layer: int) -> torch.Tensor:
    return freqs[layout(model_id)[layer].kind] if isinstance(freqs, dict) else freqs


def cap(model_id: str, logits: torch.Tensor) -> torch.Tensor:
    """The final softcap on logits (Gemma: ``30 · tanh(x / 30)``); unchanged for Qwen3."""
    c = getattr(text_config(model_id), "final_logit_softcapping", None)
    return (torch.tanh(logits.float() / c) * c).to(logits.dtype) if c else logits


def lens(model_id: str, kind: str, fit: str | None):
    """A lens mean by layer, indexable by layer: ``kind`` ``R_out``, ``R_mid``, ``S_out`` or
    ``S_mid``, from the fit ``data/fits/<fit>.pt`` or, for ``hub``/``None``, the release."""
    lens_kind = {"R_out": "J_out", "R_mid": "J_mid", "S_out": "K_out", "S_mid": "K_mid"}[kind]
    if fit and fit != "hub":
        from neuron_lens.lenses import load_fits
        _, acc = load_fits([FITS / f"{fit}.pt"], kind[0], model_id)
        return acc.mean(kind[2:])
    return lens_means(model_id, parts=(lens_kind,))[lens_kind]


def fits(model_id: str) -> tuple[str, str]:
    """``(R fit, S fit)`` of record: the model's DCLM fit files in ``data/fits/`` (``02_fit.py``),
    else, where they are not on this machine, the released lenses (``hub``)."""
    names = tuple(f"{lens}-{stem(model_id)}-L{text_config(model_id).num_hidden_layers}-dclm" for lens in "rs")
    return names if all((FITS / f"{x}.pt").exists() for x in names) else ("hub", "hub")


if __name__ == "__main__":
    write_depth()
