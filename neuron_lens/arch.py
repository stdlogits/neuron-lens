"""Model geometry across Qwen3 and Gemma 4: attention heads' sizes, weights and lenses, the
corpora and token counts, for the pipeline's Gemma path and the media checks.

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
import functools
from dataclasses import dataclass

import torch
from transformers import AutoConfig

from neuron_lens.paths import DATA, FITS
from neuron_lens.rules import norm_eps
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
def layout(model_id: str) -> list[Layer]:
    return layout_of(text_config(model_id), is_gemma4(model_id))


def layout_of(tc, gemma: bool) -> list[Layer]:
    if gemma:
        return [Layer(tc.num_attention_heads, c.num_key_value_heads, c.head_dim, bool(t == "full_attention" and tc.attention_k_eq_v),
                      t, f"model.language_model.layers.{i}.")
                for i, (t, c) in enumerate(zip(tc.layer_types, tc.per_layer_config))]
    return [Layer(tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim, False, "attention", f"model.layers.{i}.")
            for i in range(tc.num_hidden_layers)]


def names(model_id: str) -> dict:
    """The embedding, unembedding and final norm weights' names."""
    if is_gemma4(model_id):
        e = "model.language_model.embed_tokens.weight"
        return {"embed": e, "unembed": e, "norm": "model.language_model.norm.weight"}  # tied
    return {"embed": "model.embed_tokens.weight", "unembed": "lm_head.weight", "norm": "model.norm.weight"}


def embed_scale(model_id: str) -> float:
    """What the embedding layer multiplies token embeddings by: ``√d`` for Gemma, else 1."""
    return text_config(model_id).hidden_size ** 0.5 if is_gemma4(model_id) else 1.0


def ov(model_id: str, lay: Layer, dev="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """``(W_V diag(γ_in) [kv·hd, d], diag(γ_post) W_O [d, H·hd])`` in fp32."""
    p = lay.prefix
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
    """``Wq``, ``Wk`` (times ``γ_in``) and the q/k norm weights, in fp32."""
    p = lay.prefix
    w = tensors(model_id, [p + f"self_attn.{n}" for n in ("q_proj.weight", "k_proj.weight", "q_norm.weight", "k_norm.weight")]
                + [p + "input_layernorm.weight"])
    w = {k: v.float().to(dev) for k, v in w.items()}
    g1 = w[p + "input_layernorm.weight"]
    return {"Wq": w[p + "self_attn.q_proj.weight"] * g1, "Wk": w[p + "self_attn.k_proj.weight"] * g1,
            "q_norm": w[p + "self_attn.q_norm.weight"], "k_norm": w[p + "self_attn.k_norm.weight"]}


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
    return DATA / "token_counts.pt" if model_id == QWEN else DATA / f"token_counts-{stem(model_id)}.pt"


def load_model(model_id: str, dev: str, eager: bool = True):
    """``(lm, stack)``: the model in bf16 (eager attention, for its patterns) and its decoder stack."""
    from transformers import AutoModelForCausalLM

    from neuron_lens.transport import decoder_stack
    kw = {"attn_implementation": "eager"} if eager else {}
    lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, **kw).to(dev).eval().requires_grad_(False)
    return lm, decoder_stack(lm)


def logits(lm, hidden: torch.Tensor) -> torch.Tensor:
    """Final logits in fp32 from the stack's last (normed) hidden state, with Gemma's softcap."""
    x = lm.lm_head(hidden).float()
    cap = getattr(lm.config.get_text_config(), "final_logit_softcapping", None)
    return torch.tanh(x / cap) * cap if cap else x


def bands(n_layers: int) -> list[tuple[int, int]]:
    """Layer bands for the summaries: Qwen3-8B's (36 layers), else quarters."""
    if n_layers == 36:
        return [(0, 8), (9, 12), (13, 20), (21, 27), (28, 35)]
    q = n_layers // 4
    return [(i * q, (i + 1) * q - 1 if i < 3 else n_layers - 1) for i in range(4)]


# ---------------------------------------------------------------- the pipeline's Gemma path

def groups(model_id: str) -> list[int]:
    """Query heads per key-value head, per layer."""
    return [x.q_heads // x.kv_heads for x in layout(model_id)]


def padded(model_id: str) -> tuple[int, int]:
    """``(key-value heads, head size)``, the largest over layers: per-layer statistics are kept in
    buffers this size, each layer in its leading block (Gemma 4: 8 heads of 256, or 1 of 512)."""
    lay = layout(model_id)
    return max(x.kv_heads for x in lay), max(x.head_dim for x in lay)


def head_write(layer) -> torch.Tensor:
    """``[d, H·hd]``: a decoder layer's ``W_O`` as its heads' outputs reach the residual, in fp32:
    times Gemma 4's post-attention norm weight (whose scale ``attn_scale`` gives per position)."""
    w = layer.self_attn.o_proj.weight.float()
    pa = getattr(layer, "post_attention_layernorm", None)
    return pa.weight.float()[:, None] * w if hasattr(layer, "pre_feedforward_layernorm") else w


def attn_scale(layer, attn_out: torch.Tensor) -> torch.Tensor | None:
    """``[B, T]``: Gemma 4's post-attention norm's scale at each position (held constant, the LN
    rule), from the attention output; ``None`` for Qwen3, whose heads write the residual directly."""
    if not hasattr(layer, "pre_feedforward_layernorm"):
        return None
    return torch.rsqrt(attn_out.float().pow(2).mean(-1) + norm_eps(layer.post_attention_layernorm))


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
