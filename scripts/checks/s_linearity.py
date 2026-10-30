"""How much of a token's footprint can any linear lens predict on Qwen3-8B? Is the S-lens's limit the
model's nonlinearity or its own estimator (summing over offsets, averaging over contexts)?

96 sampled tokens (``catalog.kept``) are put at
position 16, the first the lens seeds, of 4 DCLM documents. Their exact footprint on the MLP input
(centred over the tokens, averaged over the documents; at offset 0, and summed over the positions
the lens reads, 16-126) is set against:
- the single-position linearization ``J·δ`` in each document, with the tangent
  ``δ = e_v − e_orig`` at position 16 only: under RelP (``transport``'s layer maps) and under the
  true derivative (a central finite difference in fp32), the best a linear lens could do there;
- the S-lens, ``K_L (e_v − ē)``;
- the embedding (``K = I``).

R² with one fitted scale, and the median per-token cosine (findings §6). Fp32 on MPS (MPS has no forward-mode
AD for ``linear``, hence the finite difference). Log in ``data/summary/s_linearity_<model>.log``.

    python scripts/checks/s_linearity.py
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_lens.catalog import kept
from neuron_lens.corpus import lens_docs
from neuron_lens.lenses import load_fits
from neuron_lens.paths import CORPUS_DIR, FITS, SUMMARY
from neuron_lens.transport import SKIP_FIRST, _jvp, _linearize

MODEL = "Qwen/Qwen3-8B"
S_FIT = "s-qwen3-8b-L36-dclm"
N_TOKENS, N_DOCS, SEQ = 96, 4, 128
LAYERS = (0, 1, 2, 5, 10, 15, 25)
EPS = 0.01
CHUNK = 16


def r2(dh: torch.Tensor, p: torch.Tensor) -> float:
    alpha = (dh * p).sum() / (p * p).sum()
    return (1 - (dh - alpha * p).pow(2).sum() / dh.pow(2).sum()).item()


def main() -> None:
    dev = "mps" if torch.backends.mps.is_available() else "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    keep = kept()
    ids = keep[torch.randperm(len(keep), generator=torch.Generator().manual_seed(0))[:N_TOKENS]].to(dev)
    docs = torch.tensor(lens_docs(CORPUS_DIR / "act-dclm.jsonl", tok, 4 * N_DOCS, SEQ, SEQ)[:N_DOCS], device=dev)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    model.requires_grad_(False)
    s0, end = SKIP_FIRST, SEQ - 1  # the seeded position, and the end of the read positions
    cap = {}
    for l in LAYERS:
        model.model.layers[l].post_attention_layernorm.register_forward_pre_hook(
            lambda m, x, l=l: cap.__setitem__(l, x[0]))

    def reduce(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # offset 0, summed over read positions
        return h[:, s0], h[:, s0:end].sum(1)
    E = model.model.embed_tokens.weight
    res = {k: {l: [torch.zeros(N_TOKENS, E.shape[1], device=dev) for _ in range(2)] for l in LAYERS}
           for k in ("exact", "J true", "J relp")}
    with torch.no_grad():
        for doc in docs:
            for c in range(0, N_TOKENS, CHUNK):
                cid = ids[c:c + CHUNK]
                batch = doc[None].repeat(len(cid), 1)
                batch[:, s0] = cid
                model.model(batch)
                for l in LAYERS:
                    for j, v in enumerate(reduce(cap[l])):
                        res["exact"][l][j][c:c + CHUNK] += v / N_DOCS
                emb = E[doc][None].repeat(len(cid), 1, 1)
                delta = E[cid] - E[doc[s0]]
                sides = []
                for sign in (1, -1):
                    shifted = emb.clone()
                    shifted[:, s0] += sign * EPS * delta
                    model.model(inputs_embeds=shifted)
                    sides.append({l: reduce(cap[l]) for l in LAYERS})
                for l in LAYERS:
                    for j in range(2):
                        res["J true"][l][j][c:c + CHUNK] += (sides[0][l][j] - sides[1][l][j]) / (2 * EPS * N_DOCS)
            X = torch.zeros(N_TOKENS, end - s0, E.shape[1], device=dev)
            X[:, 0] = E[ids] - E[doc[s0]]
            for l, m in enumerate(_linearize(model.model, doc)):
                X, mid = _jvp(m, X)
                if l in LAYERS:
                    for j, v in enumerate(reduce(F.pad(mid, (0, 0, s0, 0)))):
                        res["J relp"][l][j] += v / N_DOCS
                if l == max(LAYERS):
                    break
    _, acc = load_fits([FITS / f"{S_FIT}.pt"], "S")
    K = acc.mean("mid")
    del acc
    e = E[ids] - E[ids].mean(0)
    lines = [f"S-lens linearity, {MODEL}: {N_TOKENS} tokens at position {s0} of {N_DOCS} DCLM documents; footprint on "
             f"the MLP input at offset 0 and summed over positions {s0}-{end - 1}; R² (one fitted scale) / median cosine "
             "of each predictor: J true (finite difference), J relp (transport maps), both at position 16 only in each "
             f"document; K ({S_FIT}); embed", ""]
    for l in LAYERS:
        parts = []
        for j, name in enumerate(("offset 0", "summed")):
            dh = res["exact"][l][j] - res["exact"][l][j].mean(0)
            preds = {"J true": res["J true"][l][j], "J relp": res["J relp"][l][j], "K": e @ K[l].to(dev).T, "embed": e}
            parts.append(f"{name}: " + ", ".join(
                f"{k} {r2(dh, p - p.mean(0)):.2f}/{F.cosine_similarity(dh, p - p.mean(0), dim=1).median():.2f}"
                for k, p in preds.items()))
        lines.append(f"L{l:2d}  " + "   ".join(parts))
        print(lines[-1], flush=True)
    (SUMMARY / f"s_linearity_{MODEL.split('/')[-1].lower()}.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
