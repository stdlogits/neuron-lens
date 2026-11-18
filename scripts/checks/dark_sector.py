"""The dark sector: the part of each layer's residual the R-lens barely reads, what it carries, and what removing it does
to the model, early, in the middle and late (findings §21). Qwen3-8B, CUDA, the model in bf16 with fp32 statistics.

**Sectors.** At each layer's output (and MLP input), the R-lens readout in the residual's metric (``neuron_lens.metric``),
``B = W̄_U diag(γ_f) J_L diag(s)`` (``W̄_U``: the kept unembedding, centred over tokens, since softmax ignores a shift
shared by every token), has eigenvectors ``V`` ordered by gain. The **bright** sector is the top ``K_L`` of them, the
fewest that carry 90% of the lens energy (``Σ gain · (v·x)²``) of the layer's neuron writes, as a mean over the unit
writes (at the MLP input: over the gate and up rows as vectors, ``s ⊙ c``). The **dark** sector is the rest, orthogonal in
the metric. A sector of a vector is its part in the sector, mapped back to raw coordinates.

**Ablation.** One sector of the MLP output, over a band of layers, at positions ≥ 1, replaced by its mean over the
positions of held-out DCLM documents (the first position, the sink, keeps its output). Controls: random subspaces of the
dark and bright dimensions (``rand_d``, ``rand_b``; random orthonormal bases in the metric) and the whole MLP output
(``mlp``).

Parts:

- ``trace``: per layer, ``K_L``; how loud each layer's dark and bright sectors are to every later layer's lens (the
  mean gain of the sector's directions over a random direction's: 1 = as loud as random); in context, the dark share
  of the residual, attention and MLP outputs (raw and centred; against the share of dimensions and a per-coordinate
  null) and of what drives firing neurons (``½ (g_d/g + u_d/u)`` over the top 1% |activations|, against a random subspace
  of the dark dimension); single-layer ablations (every layer, every sector): ΔNLL, KL to the clean model and the KL the
  lens predicts (``W_U (γ_f ⊙ J_L Δm / rms)``, the R-lens's LN rule), the perturbation's energy and bright share at each
  later layer's output, what later attention and MLPs add to it, the later neurons that move most and the next tokens
  that suffer most.
- ``bands``: ablations over bands (L0, L1-L8, L9-L20, L21-L35, L1-L35): NLL, KL and top-1 agreement on the four
  activation slices, and greedy generations from ten prompts.
- ``comprehend``: who-did-it questions about one sentence, seven types sharing (agent, patient, verb) triples (reversible:
  active, subject relative, passive, object relative; non-reversible: active, passive, object relative), forced choice
  A/B in both orders, scored by the log-prob margin of the agent's letter. A band's sector is ablated on the sentence's
  tokens only (``encode``), at the answer only (``answer``) or everywhere (``all``).
- ``patch``: one sector of one layer's residual (``h``), or of a band's MLP outputs (``m``), at chosen positions, taken
  from a source prompt's clean run: facts (prompts "The capital of X is" and "The official language of X is" of one
  length that differ in the country, patched at every position, the country's or the last), language (a capital
  prompt's last position from one about another country in the other language, Chinese ↔ English, with four outcomes:
  either country's capital in either language) and word pieces (DCLM prefixes ending inside a two-piece word or after a
  whole word: which piece comes next, and whether the word goes on). Scored by the share of the clean target-to-source
  gap the patch moves (the two answers' log-prob difference; the source language's and the source country's log-odds;
  the mass on word-continuing pieces), against random subspaces of each sector's dimension; also the sectors' shares of
  the source-target difference, and greedy generations.

Outputs: ``data/summary/dark_sector_<part>_<model>.json`` and ``.log``.

    python scripts/checks/dark_sector.py trace
    python scripts/checks/dark_sector.py bands
    python scripts/checks/dark_sector.py comprehend
    python scripts/checks/dark_sector.py patch
"""
import argparse
import json
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from neuron_lens import arch, metric
from neuron_lens.catalog import MODEL, kept, tokenizer
from neuron_lens.corpus import act_file, lens_docs
from neuron_lens.paths import SUMMARY
from neuron_lens.weights import tensors

DEV = "cuda"
SHARE = 0.9  # the bright sector's share of the writes' lens energy
FIRST = 16  # positions scored from here (the lenses' sink cut, transport.SKIP_FIRST)
T = 128
STEM = arch.stem(MODEL)
SECTORS = ("dark", "bright", "rand_d", "rand_b", "mlp")
BANDS = {"L0": range(1), "L1-8": range(1, 9), "L9-20": range(9, 21), "L21-35": range(21, 36), "L1-35": range(1, 36)}
LOG = []


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.append(line)


def save(part: str, res: dict) -> None:
    (SUMMARY / f"dark_sector_{part}_{STEM}.json").write_text(json.dumps(res, ensure_ascii=False))
    (SUMMARY / f"dark_sector_{part}_{STEM}.log").write_text("\n".join(LOG) + "\n")


def sectors(mid: bool = False) -> dict:
    """``s`` (the metric's √s² per layer), ``V`` (bright bases, whitened, fp16), ``K``, ``R`` (random ``K``-dim bases) at
    each layer's output; with ``mid`` the same at the MLP input (``s_mid``, ``V_mid``, ``K_mid``, ``R_mid``), plus the raw
    readout Grams ``GM`` at each output (for ``trace``'s loudness table), and ``J_out`` (CPU)."""
    nm = arch.names(MODEL)
    n = arch.text_config(MODEL).num_hidden_layers
    head = tensors(MODEL, [nm["unembed"], nm["norm"]])
    gamma = head[nm["norm"]].float().to(DEV)
    wk = head[nm["unembed"]][kept()]
    mu = wk.float().mean(0).to(DEV)
    d = wk.shape[1]
    G = torch.zeros(d, d, device=DEV)
    for i in range(0, len(wk), 16384):
        c = (wk[i:i + 16384].to(DEV).float() - mu) * gamma
        G += c.T @ c
    del wk, head
    parts = ("J_out", "J_mid") if mid else ("J_out",)
    lm = arch.lens_means(MODEL, *arch.fits(MODEL), parts=parts)
    gen = torch.Generator(device=DEV).manual_seed(0)
    out = {k: [] for k in ("s", "V", "K", "R", "s_mid", "V_mid", "K_mid", "R_mid", "GM")}

    def top(Gr, s, X):
        gain, V = torch.linalg.eigh(s[:, None] * Gr * s[None, :])
        gain, V = gain.flip(0).clamp_min(0), V.flip(1)
        e = (X @ V).pow(2) * gain
        k = int(((e.cumsum(1) / e.sum(1, keepdim=True)).mean(0) < SHARE).sum()) + 1
        return k, V[:, :k].half().contiguous(), torch.linalg.qr(torch.randn(d, k, device=DEV, generator=gen))[0].half()

    for L in range(n):
        p = arch.layout(MODEL)[L].prefix
        w = tensors(MODEL, [p + "mlp.down_proj.weight", p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight",
                            p + "post_attention_layernorm.weight"])
        J = lm["J_out"][L].to(DEV).float() if L < lm["J_out"].shape[0] else torch.eye(d, device=DEV)
        Gr = J.T @ G @ J
        s = metric.s2(MODEL, "out", L, DEV).sqrt()
        X = w[p + "mlp.down_proj.weight"].to(DEV).float().T / s
        k, V, R = top(Gr, s, X / X.norm(dim=1, keepdim=True))
        out["s"].append(s), out["K"].append(k), out["V"].append(V), out["R"].append(R)
        if mid:
            out["GM"].append(Gr)
            s = metric.s2(MODEL, "mid", L, DEV).sqrt()
            Jm = lm["J_mid"][L].to(DEV).float()
            g = w[p + "post_attention_layernorm.weight"].to(DEV).float()
            rows = torch.cat([w[p + "mlp.gate_proj.weight"].to(DEV).float(), w[p + "mlp.up_proj.weight"].to(DEV).float()])
            Xm = rows * g * s  # covector rows c as whitened vectors s ⊙ c
            k, V, R = top(Jm.T @ G @ Jm, s, Xm / Xm.norm(dim=1, keepdim=True))
            out["s_mid"].append(s), out["K_mid"].append(k), out["V_mid"].append(V), out["R_mid"].append(R)
            del Jm, Xm, rows
        del w, J, X
    out["J_out"] = lm["J_out"]
    log("sectors: K_out " + " ".join(map(str, out["K"])) + (" | K_mid " + " ".join(map(str, out["K_mid"])) if mid else ""))
    return out


def project(x: torch.Tensor, s: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """The part of raw vectors ``x`` in the span of the whitened orthonormal ``basis``, in raw coordinates."""
    return (((x.float() / s).half() @ basis) @ basis.T).float() * s


def model_and_means(sec: dict, extra_hooks=None):
    """The model, and each layer's mean MLP output over positions ≥ 1 of 64 held-out DCLM documents (the 97th-160th
    full-length ones; the first 64 are the scored documents of ``trace``, the first 32 those of ``bands``)."""
    from transformers import AutoModelForCausalLM
    tok = tokenizer()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEV).eval()
    layers = model.model.layers
    docs = torch.tensor(lens_docs(act_file("dclm"), tok, 160, T, T, full=True))[-64:]
    sums = [torch.zeros(sec["s"][0].shape[0], device=DEV) for _ in layers]
    count = [0]

    def cap(M):
        def h(mod, args, out):
            sums[M] += out[:, 1:].float().sum((0, 1))
            if M == 0:
                count[0] += out[:, 1:].shape[0] * out[:, 1:].shape[1]
        return h

    hooks = [lay.mlp.register_forward_hook(cap(M)) for M, lay in enumerate(layers)]
    with torch.inference_mode():
        for b in range(0, len(docs), 4):
            model(input_ids=docs[b:b + 4].to(DEV), use_cache=False)
    for h in hooks:
        h.remove()
    return tok, model, [m / count[0] for m in sums]


def part(x: torch.Tensor, M: int, sector: str, sec: dict) -> torch.Tensor:
    """The part of raw vectors ``x`` at layer ``M``'s output in ``sector`` (``mlp``, ``full``: all of it)."""
    if sector in ("mlp", "full"):
        return x.float()
    p = project(x, sec["s"][M], sec["V" if sector in ("dark", "bright") else "R"][M])
    return p if sector in ("bright", "rand_b") else x.float() - p


def ablated(out: torch.Tensor, M: int, sector: str, sec: dict, mean: torch.Tensor, mask=None) -> torch.Tensor:
    """``out`` (an MLP output ``[B, T, d]``) with its part in ``sector`` replaced by that part of ``mean``: at positions
    ≥ 1 (with ``mask`` ``[B, T]``, where it is set)."""
    dev = out.float() - mean
    if mask is None and out.shape[1] > 1:
        dev[:, 0] = 0
    p = part(dev, M, sector, sec)
    if mask is not None:
        p = p * mask[..., None]
    return (out.float() - p).to(out.dtype)


# ------------------------------------------------------------------------------------------------------- trace

def trace(n_docs: int, batch: int) -> None:
    sec = sectors(mid=True)
    nL, d = len(sec["K"]), sec["s"][0].shape[0]
    # loudness of each layer's sectors to each later layer's lens
    rel_dark = torch.full((nL, nL), float("nan"))
    rel_bright = torch.full((nL, nL), float("nan"))
    for L in range(nL):
        s, Vb, K = sec["s"][L], sec["V"][L].float(), sec["K"][L]
        for M in range(L, nL):
            A = s[:, None] * sec["GM"][M] * s[None, :]
            trA, trB = A.trace(), ((A @ Vb) * Vb).sum()
            rel_dark[L, M] = ((trA - trB) / (d - K) / (trA / d)).item()
            rel_bright[L, M] = (trB / K / (trA / d)).item()
    del sec["GM"]
    torch.cuda.empty_cache()
    tok, model, means = model_and_means(sec)
    layers = model.model.layers
    W_U, gamma = model.lm_head.weight, model.model.norm.weight.float()
    f = layers[0].mlp.down_proj.weight.shape[1]
    docs = torch.tensor(lens_docs(act_file("dclm"), tok, n_docs, T, T, full=True))
    v = slice(FIRST, T - 1)
    sV, sVm = sec["V"], sec["V_mid"]

    def energies(x, M, mid=False):
        s, V = (sec["s_mid"][M], sVm[M]) if mid else (sec["s"][M], sV[M])
        xw = x.float() / s
        return (xw.half() @ V).float().pow(2).sum(-1), xw.pow(2).sum(-1)

    Z = lambda *sh: torch.zeros(*sh, device=DEV, dtype=torch.float64)
    cen = {k: {"tot": Z(nL), "bright": Z(nL), "sum_w": Z(nL, d), "sum_b": [Z(K) for K in sec["K"]], "sq_w": Z(nL, d)}
           for k in ("h", "a", "m")}
    drive = {"dark": Z(nL), "rand": Z(nL), "n": Z(nL), "x_dark": Z(nL), "x": Z(nL)}
    act_sq = Z(nL, f)
    st = {m: {"nll": Z(nL), "kl": Z(nL), "kl_pred": Z(nL), "inj": Z(nL), "inj_dark": Z(nL), "dh": Z(nL, nL),
              "dh_bright": Z(nL, nL), "da": Z(nL, nL), "dm": Z(nL, nL)} for m in SECTORS}
    listen = {m: torch.zeros(nL, nL, f, device=DEV) for m in ("dark", "bright")}
    pos_dnll = {m: torch.zeros(nL, len(docs), T - 1 - FIRST) for m in ("dark", "bright")}
    nll_clean, n_pos = Z(1), 0
    S = {"clean": True, "L": None, "mode": None}
    cache = {}

    def attn_hook(M):
        def h(mod, args, out):
            if S["clean"]:
                cache["a"][M] = out[0]
            elif M > S["L"]:
                da = (out[0] - cache["a"][M])[:, v].reshape(-1, d).float() / sec["s"][M]
                st[S["mode"]]["da"][S["L"], M] += da.pow(2).sum().double()
        return h

    def mid_hook(M):
        def h(mod, args):
            if not S["clean"]:
                return
            x = args[0][:, v].reshape(-1, d)
            mlp = layers[M].mlp
            r = x.float().pow(2).mean(-1, keepdim=True).add(mod.variance_epsilon).rsqrt()
            g = mod.weight.float()
            xn = (x.float() * r * g).to(x.dtype)
            gp, up = mlp.gate_proj(xn).float(), mlp.up_proj(xn).float()
            act = F.silu(gp) * up
            sel = act.abs() >= act.abs().flatten().kthvalue(int(0.99 * act.numel())).values
            for name, basis in (("dark", sVm[M]), ("rand", sec["R_mid"][M])):
                xd = ((x.float() - project(x, sec["s_mid"][M], basis)) * r * g).to(x.dtype)
                share = (0.5 * (mlp.gate_proj(xd).float() / gp + mlp.up_proj(xd).float() / up))[sel].clamp(-2, 3)
                drive[name][M] += share.sum().double()
            drive["n"][M] += sel.sum()
            be, te = energies(x, M, mid=True)
            drive["x_dark"][M] += (te - be).sum().double()
            drive["x"][M] += te.sum().double()
        return h

    def mlp_hook(M):
        def h(mod, args, out):
            if S["clean"]:
                cache["m"][M] = out
                return None
            L, mode = S["L"], S["mode"]
            if M == L:
                new = ablated(out, M, mode, sec, means[M])
                dm = (new - out)[:, v].reshape(-1, d)
                be, te = energies(dm, M)
                st[mode]["inj"][L] += te.sum().double()
                st[mode]["inj_dark"][L] += (te - be).sum().double()
                cache["dm"] = (new - out).float()
                return new
            if M > L:
                dm = (out - cache["m"][M])[:, v].reshape(-1, d).float() / sec["s"][M]
                st[mode]["dm"][L, M] += dm.pow(2).sum().double()
            return None
        return h

    def down_hook(M):
        def h(mod, args):
            if S["clean"]:
                cache["act"][M] = args[0]
                act_sq[M] += args[0][:, v].float().pow(2).sum((0, 1)).double()
            elif S["mode"] in listen and M > S["L"]:
                listen[S["mode"]][S["L"], M] += (args[0][:, v].float() - cache["act"][M][:, v].float()).abs().sum((0, 1))
        return h

    def layer_hook(M):
        def h(mod, args, out):
            hs = out[0] if isinstance(out, tuple) else out
            if S["clean"]:
                cache["h"][M] = hs
            elif M >= S["L"]:
                be, te = energies((hs - cache["h"][M])[:, v].reshape(-1, d), M)
                st[S["mode"]]["dh"][S["L"], M] += te.sum().double()
                st[S["mode"]]["dh_bright"][S["L"], M] += be.sum().double()
        return h

    hooks = []
    for M, lay in enumerate(layers):
        hooks += [lay.self_attn.register_forward_hook(attn_hook(M)),
                  lay.post_attention_layernorm.register_forward_pre_hook(mid_hook(M)),
                  lay.mlp.register_forward_hook(mlp_hook(M)), lay.mlp.down_proj.register_forward_pre_hook(down_hook(M)),
                  lay.register_forward_hook(layer_hook(M))]
    with torch.inference_mode():
        for b0 in range(0, len(docs), batch):
            tb = time.perf_counter()
            ids = docs[b0:b0 + batch].to(DEV)
            S["clean"] = True
            cache.update(h=[None] * nL, a=[None] * nL, m=[None] * nL, act=[None] * nL)
            logits = model(input_ids=ids, use_cache=False).logits[:, v].float()
            lp_clean = logits.log_softmax(-1)
            tgt = ids[:, FIRST + 1:T]
            nll_c = -lp_clean.gather(-1, tgt[..., None])[..., 0]
            nll_clean += nll_c.sum().double()
            n_pos += nll_c.numel()
            rms = cache["h"][-1][:, v].float().pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            for k in ("h", "a", "m"):
                for M in range(nL):
                    x = cache[k][M][:, v].reshape(-1, d)
                    be, te = energies(x, M)
                    xw = x.float() / sec["s"][M]
                    c = cen[k]
                    c["tot"][M] += te.sum().double()
                    c["bright"][M] += be.sum().double()
                    c["sum_w"][M] += xw.sum(0).double()
                    c["sum_b"][M] += (xw.half() @ sV[M]).float().sum(0).double()
                    c["sq_w"][M] += xw.pow(2).sum(0).double()
            S["clean"] = False
            for L in range(nL):
                JL = sec["J_out"][L].to(DEV, torch.bfloat16) if L < sec["J_out"].shape[0] else None
                for mode in SECTORS:
                    S["L"], S["mode"] = L, mode
                    lp = model(input_ids=ids, use_cache=False).logits[:, v].float().log_softmax(-1)
                    nll = -lp.gather(-1, tgt[..., None])[..., 0]
                    st[mode]["nll"][L] += nll.sum().double()
                    st[mode]["kl"][L] += (lp_clean.exp() * (lp_clean - lp)).sum().double()
                    dh = cache["dm"][:, v]
                    if JL is not None:
                        dh = (dh.to(torch.bfloat16) @ JL.T).float()
                    lpp = (logits + ((dh / rms * gamma).to(torch.bfloat16) @ W_U.T).float()).log_softmax(-1)
                    st[mode]["kl_pred"][L] += (lp_clean.exp() * (lp_clean - lpp)).sum().double()
                    if mode in pos_dnll:
                        pos_dnll[mode][L, b0:b0 + len(ids)] = (nll - nll_c).cpu()
                    del lp, lpp
            log(f"batch {b0 // batch + 1}/{(len(docs) + batch - 1) // batch} in {time.perf_counter() - tb:.0f}s")
    for h in hooks:
        h.remove()

    n = n_pos
    res = {"K_out": sec["K"], "K_mid": sec["K_mid"], "n_positions": n, "nll_clean": nll_clean.item() / n,
           "loud_dark": rel_dark.tolist(), "loud_bright": rel_bright.tolist(), "census": {}, "drive": {}, "single": {},
           "listeners": {}, "tokens": {}}
    for M in range(nL):
        row = {}
        for k, c in cen.items():
            tot, br = c["tot"][M].item() / n, c["bright"][M].item() / n
            mw, mb = c["sum_w"][M] / n, c["sum_b"][M] / n
            q = c["sq_w"][M] / n
            diag = (q * (1 - sV[M].double().pow(2).sum(1))).sum().item() / q.sum().item()
            row[k] = {"dark": 1 - br / tot, "dark_centred": 1 - (br - mb.pow(2).sum().item()) / (tot - mw.pow(2).sum().item()),
                      "dark_dims": 1 - sec["K"][M] / d, "diag_null": diag}
        res["census"][M] = row
        nd = max(drive["n"][M].item(), 1)
        res["drive"][M] = {"dark": drive["dark"][M].item() / nd, "rand": drive["rand"][M].item() / nd,
                           "x_dark": drive["x_dark"][M].item() / drive["x"][M].item(), "dark_dims_mid": 1 - sec["K_mid"][M] / d}
    for mode, s_ in st.items():
        res["single"][mode] = {
            "dnll": [(s_["nll"][L].item() - nll_clean.item()) / n for L in range(nL)],
            "kl": [s_["kl"][L].item() / n for L in range(nL)], "kl_pred": [s_["kl_pred"][L].item() / n for L in range(nL)],
            "inj": [s_["inj"][L].item() / n for L in range(nL)],
            "inj_dark": [s_["inj_dark"][L].item() / max(s_["inj"][L].item(), 1e-12) for L in range(nL)],
            "dh": (s_["dh"] / n).tolist(), "dh_bright": (s_["dh_bright"] / s_["dh"].clamp_min(1e-12)).tolist(),
            "da": (s_["da"] / n).tolist(), "dm": (s_["dm"] / n).tolist()}
    sd = (act_sq / n).sqrt().float().clamp_min(1e-6)
    tg = docs[:, FIRST + 1:T].flatten().numpy()
    ids_, inv, cnt = np.unique(tg, return_inverse=True, return_counts=True)
    for mode in ("dark", "bright"):
        res["listeners"][mode], res["tokens"][mode] = {}, {}
        for L in range(nL):
            z = listen[mode][L] / n / sd
            z[: L + 1] = 0
            top = torch.topk(z.flatten(), 16)
            res["listeners"][mode][L] = {"by_layer": z.sum(1).tolist(),
                                         "top": [[int(i) // f, int(i) % f, float(x)] for x, i in zip(top.values, top.indices)]}
            mean = np.bincount(inv, pos_dnll[mode][L].flatten().numpy()) / cnt
            o = np.argsort(-np.where(cnt >= 15, mean, -np.inf))[:12]
            res["tokens"][mode][L] = [[tok.decode([int(ids_[i])]), float(mean[i]), int(cnt[i])] for i in o]
    save("trace", res)
    log("TRACE DONE")


# ------------------------------------------------------------------------------------------------------- bands

PROMPTS = ["The capital of France is", "def fibonacci(n):\n", "Once upon a time, in a small village,",
           "Q: What is 17 + 25?\nA:", "中国的首都是", "Monday, Tuesday, Wednesday,",
           "The mitochondria is the powerhouse of the cell because"]
CHAT = ["Explain why the sky is blue in two sentences.", "Who are you?", "Write a haiku about autumn."]


def bands(n_eval: int, new: int, batch: int) -> None:
    sec = sectors()
    tok, model, means = model_and_means(sec)
    evals = {k: torch.tensor(lens_docs(act_file(k), tok, n_eval, T, T, full=True)) for k in ("dclm", "code", "zh", "chat")}
    S = {"sector": None, "band": None}

    def hook(M):
        def h(mod, args, out):
            if S["sector"] is None or M not in BANDS[S["band"]]:
                return None
            return ablated(out, M, S["sector"], sec, means[M])
        return h

    hooks = [lay.mlp.register_forward_hook(hook(M)) for M, lay in enumerate(model.model.layers)]
    prompts = [tok(p, return_tensors="pt").input_ids for p in PROMPTS]
    for c in CHAT:
        x = tok.apply_chat_template([{"role": "user", "content": c}], add_generation_prompt=True, enable_thinking=False,
                                    return_tensors="pt")
        prompts.append(x if isinstance(x, torch.Tensor) else x["input_ids"])
    conds = [("clean", None)] + [(m, b) for b in ("L0", "L1-8", "L9-20", "L21-35", "L1-35") for m in ("dark", "bright")] \
        + [("rand_d", "L1-35"), ("rand_b", "L1-35"), ("mlp", "L1-35"), ("mlp", "L9-20")]
    clean_lp = {}
    res = {"K_out": sec["K"], "prompts": PROMPTS + [f"[chat] {c}" for c in CHAT], "conds": []}
    with torch.inference_mode():
        for sector, band in conds:
            S["sector"], S["band"] = (None, None) if sector == "clean" else (sector, band)
            row = {"sector": sector, "band": band, "slices": {}, "gen": []}
            for name, ids in evals.items():
                nll, kl, top1, n = 0.0, 0.0, 0.0, 0
                for b in range(0, len(ids), batch):
                    x = ids[b:b + batch].to(DEV)
                    lp = model(input_ids=x, use_cache=False).logits[:, FIRST:T - 1].float().log_softmax(-1)
                    nll += -lp.gather(-1, x[:, FIRST + 1:T, None]).sum().item()
                    n += x[:, FIRST + 1:T].numel()
                    if sector == "clean":
                        clean_lp[name, b] = lp.to(torch.bfloat16).cpu()
                    else:
                        c = clean_lp[name, b].to(DEV).float()
                        kl += (c.exp() * (c - lp)).sum().item()
                        top1 += (c.argmax(-1) == lp.argmax(-1)).sum().item()
                row["slices"][name] = {"nll": nll / n, "kl": kl / n, "top1_agree": 1.0 if sector == "clean" else top1 / n}
            for p in prompts:
                out = model.generate(p.to(DEV), max_new_tokens=new, do_sample=False, pad_token_id=tok.eos_token_id)
                row["gen"].append(tok.decode(out[0, p.shape[1]:]))
            res["conds"].append(row)
            log(f"{sector:6s} {band or '':6s} | NLL " + " ".join(f"{k} {v['nll']:.3f}" for k, v in row["slices"].items())
                + f" | {row['gen'][0][:60]!r}")
            save("bands", res)
    for h in hooks:
        h.remove()
    log("BANDS DONE")


# ------------------------------------------------------------------------------------------------------- comprehend

ANIMATE = ["boy", "girl", "man", "woman", "doctor", "nurse", "teacher", "student", "king", "queen", "dog", "cat", "lion",
           "tiger", "farmer", "soldier"]
REVERSIBLE = [("kissed", "kissed", "kissing"), ("chased", "chased", "chasing"), ("pushed", "pushed", "pushing"),
              ("hugged", "hugged", "hugging"), ("called", "called", "calling"), ("followed", "followed", "following"),
              ("helped", "helped", "helping"), ("hit", "hit", "hitting"), ("visited", "visited", "visiting"),
              ("praised", "praised", "praising"), ("blamed", "blamed", "blaming"), ("thanked", "thanked", "thanking"),
              ("pulled", "pulled", "pulling"), ("warned", "warned", "warning"), ("greeted", "greeted", "greeting"),
              ("watched", "watched", "watching")]
AGENTS = ["boy", "girl", "man", "woman", "teacher", "farmer", "doctor", "student"]
FIXED = [("ate", "eaten", "eating", "apple"), ("read", "read", "reading", "book"), ("drove", "driven", "driving", "car"),
         ("painted", "painted", "painting", "fence"), ("wrote", "written", "writing", "letter"),
         ("opened", "opened", "opening", "door"), ("fixed", "fixed", "fixing", "bike"), ("cooked", "cooked", "cooking", "soup"),
         ("built", "built", "building", "house"), ("baked", "baked", "baking", "cake"),
         ("planted", "planted", "planting", "tree"), ("sang", "sung", "singing", "song"),
         ("kicked", "kicked", "kicking", "ball"), ("drank", "drunk", "drinking", "juice"), ("cut", "cut", "cutting", "bread"),
         ("carried", "carried", "carrying", "box")]
ADJ = ["tall", "happy", "tired", "old", "young", "quiet"]
TEMPLATES = {"active_rev": "The {a} {past} the {p}.", "subjrel_rev": "The {a} that {past} the {p} was {adj}.",
             "passive_rev": "The {p} was {pp} by the {a}.", "objrel_rev": "The {p} that the {a} {past} was {adj}.",
             "active_nonrev": "The {a} {past} the {p}.", "passive_nonrev": "The {p} was {pp} by the {a}.",
             "objrel_nonrev": "The {p} that the {a} {past} was {adj}."}
HEAD = "Read the sentence and answer the question with A or B.\nSentence: "


def items(n: int) -> list[dict]:
    rng = random.Random(0)
    rev, fixed = [], []
    for _ in range(n):
        a, p = rng.sample(ANIMATE, 2)
        past, pp, ger = rng.choice(REVERSIBLE)
        rev.append({"a": a, "p": p, "past": past, "pp": pp, "ger": ger, "adj": rng.choice(ADJ)})
        past, pp, ger, o = rng.choice(FIXED)
        fixed.append({"a": rng.choice(AGENTS), "p": o, "past": past, "pp": pp, "ger": ger, "adj": rng.choice(ADJ)})
    out = []
    for kind, tpl in TEMPLATES.items():
        for i, it in enumerate(rev if kind.endswith("_rev") else fixed):
            s = tpl.format(**it)
            for order in (0, 1):
                o1, o2 = (it["a"], it["p"]) if order == 0 else (it["p"], it["a"])
                text = f"{HEAD}{s}\nQuestion: Who did the {it['ger']}?\nA) the {o1}\nB) the {o2}\nAnswer:"
                out.append({"kind": kind, "text": text, "span": (len(HEAD), len(HEAD) + len(s)), "agent_letter": "AB"[order]})
    return out


def comprehend(n_items: int, batch: int) -> None:
    sec = sectors()
    tok, model, means = model_and_means(sec)
    idA, idB = (tok(x).input_ids for x in (" A", " B"))
    assert len(idA) == len(idB) == 1
    its = items(n_items)
    enc = [tok(x["text"], return_offsets_mapping=True) for x in its]
    n_tok = max(len(e["input_ids"]) for e in enc)
    ids = torch.full((len(its), n_tok), tok.pad_token_id)
    att = torch.zeros(len(its), n_tok, dtype=torch.long)
    masks = {k: torch.zeros(len(its), n_tok, dtype=torch.bool) for k in ("encode", "answer", "all")}
    last = torch.zeros(len(its), dtype=torch.long)
    for i, (x, e) in enumerate(zip(its, enc)):
        n = len(e["input_ids"])
        ids[i, :n], att[i, :n], last[i] = torch.tensor(e["input_ids"]), 1, n - 1
        a, b = x["span"]
        masks["encode"][i] = torch.tensor([c1 > a and c0 < b for c0, c1 in e["offset_mapping"]] + [False] * (n_tok - n))
        masks["answer"][i, n - 1] = True
        masks["all"][i, 1:n] = True
    S = {"sector": None, "band": None, "mask": None}

    def hook(M):
        def h(mod, args, out):
            if S["sector"] is None or M not in BANDS[S["band"]]:
                return None
            return ablated(out, M, S["sector"], sec, means[M], S["mask"])
        return h

    hooks = [lay.mlp.register_forward_hook(hook(M)) for M, lay in enumerate(model.model.layers)]
    conds = [("clean", None, None)] + [(s, b, sc) for b in ("L0", "L1-8", "L9-20", "L21-35") for s in SECTORS
                                       for sc in ("encode", "answer", "all")]
    res = {"n_items": n_items, "examples": [its[i]["text"] for i in (0, 6 * n_items, 10 * n_items)], "conds": []}
    clean = None
    with torch.inference_mode():
        for sector, band, scope in conds:
            S["sector"], S["band"] = (None, None) if sector == "clean" else (sector, band)
            margin = torch.zeros(len(its))
            for b in range(0, len(its), batch):
                sl = slice(b, b + batch)
                if scope:
                    S["mask"] = masks[scope][sl].to(DEV)
                lg = model(input_ids=ids[sl].to(DEV), attention_mask=att[sl].to(DEV), use_cache=False).logits
                lp = lg[torch.arange(lg.shape[0]), last[sl].to(DEV)].float().log_softmax(-1)[:, [idA[0], idB[0]]].cpu()
                agent_a = torch.tensor([x["agent_letter"] == "A" for x in its[sl]])
                margin[sl] = torch.where(agent_a, lp[:, 0] - lp[:, 1], lp[:, 1] - lp[:, 0])
            row = {"sector": sector, "band": band, "scope": scope, "types": {}}
            for kind in TEMPLATES:
                mg = margin[[i for i, x in enumerate(its) if x["kind"] == kind]]
                item = mg.view(-1, 2).mean(1)  # both option orders, which cancels a bias to one letter
                row["types"][kind] = {"item_acc": (item > 0).float().mean().item(), "margin": mg.mean().item()}
            if clean is None:
                clean = row["types"]
            for kind, t in row["types"].items():
                t["kept"] = t["margin"] / clean[kind]["margin"]
            res["conds"].append(row)
            log(f"{sector:6s} {band or '':6s} {scope or '':6s} | " + " ".join(
                f"{k.replace('_rev', 'R').replace('_nonrev', 'N')} {t['item_acc']:.2f}({t['kept']:+.2f})"
                for k, t in row["types"].items()))
            save("comprehend", res)
    for h in hooks:
        h.remove()
    log("COMPREHEND DONE")


# ------------------------------------------------------------------------------------------------------- patch

COUNTRIES = [  # (country, in Chinese, capital, in Chinese, official language)
    ("France", "法国", "Paris", "巴黎", "French"), ("Italy", "意大利", "Rome", "罗马", "Italian"),
    ("China", "中国", "Beijing", "北京", "Chinese"), ("Germany", "德国", "Berlin", "柏林", "German"),
    ("Spain", "西班牙", "Madrid", "马德里", "Spanish"), ("Japan", "日本", "Tokyo", "东京", "Japanese"),
    ("Russia", "俄罗斯", "Moscow", "莫斯科", "Russian"), ("Egypt", "埃及", "Cairo", "开罗", "Arabic"),
    ("Greece", "希腊", "Athens", "雅典", "Greek"), ("Poland", "波兰", "Warsaw", "华沙", "Polish"),
    ("Austria", "奥地利", "Vienna", "维也纳", "German"), ("Canada", "加拿大", "Ottawa", "渥太华", "English"),
    ("South Korea", "韩国", "Seoul", "首尔", "Korean"), ("Thailand", "泰国", "Bangkok", "曼谷", "Thai"),
    ("Vietnam", "越南", "Hanoi", "河内", "Vietnamese"), ("Portugal", "葡萄牙", "Lisbon", "里斯本", "Portuguese"),
    ("Norway", "挪威", "Oslo", "奥斯陆", "Norwegian"), ("Sweden", "瑞典", "Stockholm", "斯德哥尔摩", "Swedish"),
    ("Ireland", "爱尔兰", "Dublin", "都柏林", "Irish"), ("Turkey", "土耳其", "Ankara", "安卡拉", "Turkish"),
    ("Iran", "伊朗", "Tehran", "德黑兰", "Persian"), ("India", "印度", "New Delhi", "新德里", "Hindi"),
    ("Australia", "澳大利亚", "Canberra", "堪培拉", "English"), ("Argentina", "阿根廷", "Buenos Aires", "布宜诺斯艾利斯", "Spanish"),
    ("Cuba", "古巴", "Havana", "哈瓦那", "Spanish"), ("Peru", "秘鲁", "Lima", "利马", "Spanish"),
    ("Kenya", "肯尼亚", "Nairobi", "内罗毕", "Swahili"), ("the Netherlands", "荷兰", "Amsterdam", "阿姆斯特丹", "Dutch"),
    ("Belgium", "比利时", "Brussels", "布鲁塞尔", "Dutch"), ("Denmark", "丹麦", "Copenhagen", "哥本哈根", "Danish"),
    ("Finland", "芬兰", "Helsinki", "赫尔辛基", "Finnish"), ("Hungary", "匈牙利", "Budapest", "布达佩斯", "Hungarian"),
    ("the Czech Republic", "捷克", "Prague", "布拉格", "Czech"), ("the Philippines", "菲律宾", "Manila", "马尼拉", "Filipino"),
    ("Indonesia", "印度尼西亚", "Jakarta", "雅加达", "Indonesian"), ("Saudi Arabia", "沙特阿拉伯", "Riyadh", "利雅得", "Arabic"),
    ("New Zealand", "新西兰", "Wellington", "惠灵顿", "English"), ("Chile", "智利", "Santiago", "圣地亚哥", "Spanish"),
    ("Colombia", "哥伦比亚", "Bogotá", "波哥大", "Spanish"), ("Nigeria", "尼日利亚", "Abuja", "阿布贾", "English"),
    ("Morocco", "摩洛哥", "Rabat", "拉巴特", "Arabic"), ("Ukraine", "乌克兰", "Kyiv", "基辅", "Ukrainian"),
    ("Pakistan", "巴基斯坦", "Islamabad", "伊斯兰堡", "Urdu"), ("Bangladesh", "孟加拉国", "Dhaka", "达卡", "Bengali"),
    ("Malaysia", "马来西亚", "Kuala Lumpur", "吉隆坡", "Malay"), ("Romania", "罗马尼亚", "Bucharest", "布加勒斯特", "Romanian"),
    ("Mongolia", "蒙古", "Ulaanbaatar", "乌兰巴托", "Mongolian"), ("Nepal", "尼泊尔", "Kathmandu", "加德满都", "Nepali"),
    ("Ethiopia", "埃塞俄比亚", "Addis Ababa", "亚的斯亚贝巴", "Amharic"), ("Venezuela", "委内瑞拉", "Caracas", "加拉加斯", "Spanish"),
    ("Iraq", "伊拉克", "Baghdad", "巴格达", "Arabic"), ("Syria", "叙利亚", "Damascus", "大马士革", "Arabic"),
    ("Afghanistan", "阿富汗", "Kabul", "喀布尔", "Pashto"), ("Cambodia", "柬埔寨", "Phnom Penh", "金边", "Khmer"),
    ("Ghana", "加纳", "Accra", "阿克拉", "English")]
PATCH = ("dark", "bright", "rand_d", "rand_b", "full")
SHOW = (2, 6, 12, 18, 24, 30)  # layers of the residual patches in the generations


def patch(n_words: int, batch: int, new: int) -> None:
    sec = sectors()
    from transformers import AutoModelForCausalLM
    tok = tokenizer()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEV).eval()
    layers = model.model.layers
    nL, d = len(layers), sec["s"][0].shape[0]
    n_vocab = model.lm_head.weight.shape[0]
    strs = tok.batch_decode([[i] for i in range(len(tok))]) + [""] * (n_vocab - len(tok))
    cont = torch.tensor([bool(re.fullmatch(r"[a-z]+", x)) for x in strs], device=DEV)  # pieces that continue a word
    cjk = torch.tensor([bool(re.search(r"[一-鿿]", x)) for x in strs], device=DEV)
    rng = random.Random(0)
    S = {"mode": None}
    cap = {"h": [None] * nL, "m": [None] * nL}

    def hook(M, kind):
        def h(mod, args, out):
            x = out[0] if isinstance(out, tuple) else out
            if S["mode"] is None or x.shape[1] != S["T"]:  # decode steps of a generation are left alone
                return None
            b, p = S["b"], S["p"]
            if S["mode"] == "cap":
                cap[kind][M] = x[b, p].clone()
                return None
            if S["mode"] != kind or M not in S["layers"]:
                return None
            x = x.clone()
            t = x[b, p]
            x[b, p] = (t.float() + part(S["src"][kind][M].float() - t.float(), M, S["sector"], sec)).to(x.dtype)
            return (x, *out[1:]) if isinstance(out, tuple) else x
        return h

    hooks = [h for M, lay in enumerate(layers)
             for h in (lay.register_forward_hook(hook(M, "h")), lay.mlp.register_forward_hook(hook(M, "m")))]

    def run(rows, pos, mode=None, src=None, **kw):
        """Log-probs of the next token after each row; ``pos``: each row's positions, captured (``cap``) or patched
        (``h``: the layer's output, ``m``: its MLP output) from ``src`` (``[nL, P, d]``, rows' positions in order)."""
        off = np.cumsum([0] + [len(q) for q in pos])
        lps, caps = [], {"h": [], "m": []}
        for b0 in range(0, len(rows), batch):
            rs, ps = rows[b0:b0 + batch], pos[b0:b0 + batch]
            n = max(map(len, rs))
            ids = torch.full((len(rs), n), tok.pad_token_id)
            att = torch.zeros_like(ids)
            for i, r in enumerate(rs):
                ids[i, :len(r)], att[i, :len(r)] = torch.tensor(r), 1
            S.update(mode=mode, T=n, b=torch.tensor([i for i, q in enumerate(ps) for _ in q], dtype=torch.long, device=DEV),
                     p=torch.tensor([j for q in ps for j in q], dtype=torch.long, device=DEV), **kw)
            if src is not None:
                S["src"] = {k: v[:, off[b0]:off[b0 + len(rs)]] for k, v in src.items()}
            hs = model.model(input_ids=ids.to(DEV), attention_mask=att.to(DEV), use_cache=False).last_hidden_state
            last = torch.tensor([len(r) - 1 for r in rs], device=DEV)
            lps.append(model.lm_head(hs[torch.arange(len(rs), device=DEV), last]).float().log_softmax(-1))
            if mode == "cap":
                for k, v in caps.items():
                    v.append(torch.stack(cap[k]))
        S["mode"] = None
        lp = torch.cat(lps)
        return (lp, {k: torch.cat(v, 1) for k, v in caps.items()}) if mode == "cap" else lp

    def generate(row, pos, mode=None, src=None, **kw):
        S.update(mode=mode, T=len(row), b=torch.zeros(len(pos), dtype=torch.long, device=DEV),
                 p=torch.tensor(pos, dtype=torch.long, device=DEV), src=src, **kw)
        with torch.inference_mode():
            out = model.generate(torch.tensor([row], device=DEV), max_new_tokens=new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        S["mode"] = None
        return tok.decode(out[0, len(row):])

    def first(words):
        """First tokens of each spelling, leaving out bare whitespace."""
        return sorted({i for w in words for i in tok(w).input_ids[:1] if strs[i].strip()})

    # metrics, per pair, of the next-token log-probs
    def m_fact(lp, pairs):
        i = torch.arange(len(pairs), device=DEV)
        at, as_ = (torch.tensor([p[k] for p in pairs], device=DEV) for k in ("at", "as"))
        top = lp.argmax(-1)
        return {"ld": lp[i, at] - lp[i, as_], "flip": (top == as_).float(), "keep": (top == at).float()}

    def m_lang(lp, pairs):
        pr, top = lp.exp(), lp.argmax(-1).tolist()
        cells = {c: torch.stack([pr[i, p["cells"][c]].sum() for i, p in enumerate(pairs)]) for c in ("tt", "ts", "st", "ss")}
        lg = lambda *cs: (sum(cells[c] for c in cs) + 1e-9).log()
        which = [next((c for c in cells if top[i] in p["cells"][c]), "other") for i, p in enumerate(pairs)]
        out = {"lang": lg("ts", "ss") - lg("tt", "st"), "content": lg("st", "ss") - lg("tt", "ts"), "cjk": (pr * cjk).sum(-1)}
        out |= {f"p_{c}": v for c, v in cells.items()}
        return out | {f"top_{c}": torch.tensor([w == c for w in which], dtype=torch.float) for c in (*cells, "other")}

    def m_word(lp, pairs):
        top = lp.argmax(-1)
        out = {"cont": (lp.exp() * cont).sum(-1), "top_cont": cont[top].float()}
        if "tn" in pairs[0]:
            i = torch.arange(len(pairs), device=DEV)
            tn, sn = (torch.tensor([p[k] for p in pairs], device=DEV) for k in ("tn", "sn"))
            out |= {"ld": lp[i, tn] - lp[i, sn], "flip": (top == sn).float(), "keep": (top == tn).float()}
        return out

    summ = lambda m: {k: v.float().mean().item() for k, v in m.items()}
    res = {"K_out": sec["K"], "tasks": {}}

    def task(name, pairs, metric, primary, shows=1):
        tr, sr = [p["t"] for p in pairs], [p["s"] for p in pairs]
        tp, sp = [p["tp"] for p in pairs], [p["sp"] for p in pairs]
        lp_s, src = run(sr, sp, "cap")
        lp_t, tgt = run(tr, tp, "cap")
        out = {"n": len(pairs), "clean_t": summ(metric(lp_t, pairs)), "clean_s": summ(metric(lp_s, pairs)),
               "examples": [[tok.decode(p["t"]), tok.decode(p["s"])] for p in pairs[:4]], "share": {}, "resid": {}, "mlp": {}}
        for k in ("h", "m"):  # where the source-target difference sits, by sector
            sh = {"bright": [], "rand_b": [], "dims": [K / d for K in sec["K"]]}
            for L in range(nL):
                dw = (src[k][L].float() - tgt[k][L].float()) / sec["s"][L]
                tot = dw.pow(2).sum().item()
                for name_, B in (("bright", sec["V"][L]), ("rand_b", sec["R"][L])):
                    sh[name_].append((dw.half() @ B).float().pow(2).sum().item() / max(tot, 1e-12))
            out["share"][k] = sh

        def cond(**kw):
            r = summ(metric(run(tr, tp, src=src, **kw), pairs))
            for k in primary:
                r["frac_" + k] = (r[k] - out["clean_t"][k]) / (out["clean_s"][k] - out["clean_t"][k])
            return r

        for sector in PATCH:
            rows = [cond(mode="h", layers={L}, sector=sector) for L in range(nL)]
            out["resid"][sector] = {k: [r[k] for r in rows] for k in rows[0]}
        for band, lay in BANDS.items():
            out["mlp"][band] = {sector: cond(mode="m", layers=set(lay), sector=sector) for sector in PATCH}
        off = np.cumsum([0] + [len(q) for q in tp])
        out["gen"] = []
        for j in range(shows):
            sj = {k: v[:, off[j]:off[j + 1]] for k, v in src.items()}
            g = {"target": tok.decode(tr[j]), "source": tok.decode(sr[j]), "clean": generate(tr[j], tp[j]),
                 "clean_source": generate(sr[j], sp[j]), "resid": {}, "mlp": {}}
            for L in SHOW:
                g["resid"][L] = {s_: generate(tr[j], tp[j], "h", sj, layers={L}, sector=s_) for s_ in ("dark", "bright", "full")}
            for band in ("L0", "L1-8", "L9-20", "L21-35"):
                g["mlp"][band] = {s_: generate(tr[j], tp[j], "m", sj, layers=set(BANDS[band]), sector=s_)
                                  for s_ in ("dark", "bright", "full")}
            out["gen"].append(g)
        res["tasks"][name] = out
        show = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 35)
        log(f"[{name}] n={len(pairs)} clean target " + " ".join(f"{k} {out['clean_t'][k]:.2f}" for k in primary)
            + " | source " + " ".join(f"{k} {out['clean_s'][k]:.2f}" for k in primary))
        log("  layer        " + " ".join(f"{L:5d}" for L in show))
        for k in primary:
            for sector in PATCH:
                log(f"  {k[:5]:5s} {sector:6s} " + " ".join(f"{out['resid'][sector]['frac_' + k][L]:+5.2f}" for L in show))
            log(f"  {k[:5]:5s} MLP bands " + " | ".join(
                f"{b} " + " ".join(f"{s_[:2]} {out['mlp'][b][s_]['frac_' + k]:+.2f}" for s_ in PATCH) for b in BANDS))
        log("  bright share of the difference (h) " + " ".join(f"{out['share']['h']['bright'][L]:.2f}" for L in show))
        log("  random K-dim share             " + " ".join(f"{out['share']['h']['rand_b'][L]:.2f}" for L in show))
        save("patch", res)

    with torch.inference_mode():
        # facts: aligned prompts that differ in the country, patched at every position, the country's or the last
        rows = []
        for en, _, capital, _, lang in COUNTRIES:
            for rel, text, ans in (("capital", f"The capital of {en} is", capital),
                                   ("language", f"The official language of {en} is", lang)):
                rows.append({"rel": rel, "en": en, "ids": tok(text).input_ids, "a": tok(" " + ans).input_ids[0]})
        lp = run([r["ids"] for r in rows], [[] for _ in rows])
        rows = [r for r, t in zip(rows, lp.argmax(-1).tolist()) if t == r["a"]]
        groups = {}
        for r in rows:
            groups.setdefault((r["rel"], len(r["ids"])), []).append(r)
        pairs = []
        for g in groups.values():
            g.sort(key=lambda r: (r["en"] not in ("France", "Italy"), rng.random()))
            for i, r in enumerate(g):
                o = next((g[(i + k) % len(g)] for k in range(1, len(g)) if g[(i + k) % len(g)]["a"] != r["a"]), None)
                if o is not None:
                    pairs.append({"t": r["ids"], "s": o["ids"], "at": r["a"], "as": o["a"], "rel": r["rel"]})
        pairs.sort(key=lambda p: p["rel"] != "capital")
        log(f"facts: {len(rows)} of {2 * len(COUNTRIES)} prompts answered right, {len(pairs)} pairs")
        for scope in ("all", "subject", "last"):
            for p in pairs:
                n = len(p["t"])
                p["tp"] = p["sp"] = {"all": list(range(1, n)), "last": [n - 1],
                                     "subject": [k for k in range(n) if p["t"][k] != p["s"][k]]}[scope]
            task(f"fact_{scope}", pairs, m_fact, ("ld",), shows=1 if scope == "all" else 0)

        # language: the last position of a capital prompt in one language, from one about another country in the other
        ok = []
        for en, zh, capital, capital_zh, _ in COUNTRIES:
            c = {"en": tok(f"The capital of {en} is").input_ids, "zh": tok(f"{zh}的首都是").input_ids,
                 "a_en": first([" " + capital, capital]), "a_zh": first([capital_zh, " " + capital_zh]), "name": en}
            ok.append(c)
        lp = run([c[k] for c in ok for k in ("en", "zh")], [[] for _ in range(2 * len(ok))]).argmax(-1).tolist()
        ok = [c for i, c in enumerate(ok) if lp[2 * i] in c["a_en"] and lp[2 * i + 1] in c["a_zh"]]
        ok.sort(key=lambda c: (c["name"] not in ("China", "France"), c["name"] != "China", rng.random()))
        log(f"language: {len(ok)} of {len(COUNTRIES)} countries answered right in both languages")
        for tl, sl in (("zh", "en"), ("en", "zh")):
            pairs = []
            for i, x in enumerate(ok):
                for k in range(1, len(ok)):
                    y = ok[(i + k) % len(ok)]
                    sets = [x["a_" + tl], x["a_" + sl], y["a_" + tl], y["a_" + sl]]
                    if len({t for s_ in sets for t in s_}) == sum(map(len, sets)):
                        break
                else:
                    continue
                pairs.append({"t": x[tl], "s": y[sl], "tp": [len(x[tl]) - 1], "sp": [len(y[sl]) - 1],
                              "cells": {"tt": x["a_" + tl], "ts": x["a_" + sl], "st": y["a_" + tl], "ss": y["a_" + sl]}})
            task(f"lang_{tl}_from_{sl}", pairs, m_lang, ("lang", "content"), shows=1)

        # word pieces: DCLM prefixes ending inside a two-piece word, or after a whole word
        docs = lens_docs(act_file("dclm"), tok, 600, T, T, full=True)
        mid, bound = [], []
        for doc in docs:
            m_, b_ = [], []
            for i in range(24, T - 2):
                a, b, c = strs[doc[i]], strs[doc[i + 1]], strs[doc[i + 2]]
                if re.fullmatch(r" [A-Za-z]+", a) and re.fullmatch(r"[a-z]{2,}", b) and c[:1] and not c[:1].isalpha():
                    m_.append({"ids": doc[:i + 1], "next": doc[i + 1]})
                elif re.fullmatch(r" [a-z]{3,}", a) and re.fullmatch(r" [A-Za-z]+", b):
                    b_.append({"ids": doc[:i + 1], "next": doc[i + 1]})
            mid += rng.sample(m_, min(2, len(m_)))
            bound += rng.sample(b_, min(1, len(b_)))
        mid, bound = mid[:4 * n_words], bound[:4 * n_words]
        lp = run([r["ids"] for r in mid + bound], [[] for _ in mid + bound])
        top, p_top, pc = lp.argmax(-1).tolist(), lp.max(-1).values.exp().tolist(), (lp.exp() * cont).sum(-1).tolist()
        n_mid = len(mid)
        mid = [r for i, r in enumerate(mid) if top[i] == r["next"] and p_top[i] >= 0.5][:n_words]
        bound = [r for j, r in enumerate(bound) if pc[n_mid + j] < 0.05][:n_words]
        log(f"words: {len(mid)} mid-word and {len(bound)} word-final prefixes kept")
        last = lambda r: [len(r["ids"]) - 1]
        ident = []
        for i, r in enumerate(mid):
            o = next((mid[(i + k) % len(mid)] for k in range(1, len(mid)) if mid[(i + k) % len(mid)]["next"] != r["next"]), None)
            ident.append({"t": r["ids"], "s": o["ids"], "tp": last(r), "sp": last(o), "tn": r["next"], "sn": o["next"]})
        task("word_ident", ident, m_word, ("ld",), shows=2)
        k = min(len(mid), len(bound))
        task("word_end", [{"t": mid[i]["ids"], "s": bound[i]["ids"], "tp": last(mid[i]), "sp": last(bound[i])}
                          for i in range(k)], m_word, ("cont",), shows=2)
        task("word_start", [{"t": bound[i]["ids"], "s": mid[i]["ids"], "tp": last(bound[i]), "sp": last(mid[i])}
                            for i in range(k)], m_word, ("cont",), shows=1)
    for h in hooks:
        h.remove()
    log("PATCH DONE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("trace", "bands", "comprehend", "patch"))
    ap.add_argument("--docs", type=int, default=64, help="trace: scored DCLM documents")
    ap.add_argument("--eval-docs", type=int, default=32, help="bands: documents per slice")
    ap.add_argument("--new-tokens", type=int, default=48, help="bands: generated tokens")
    ap.add_argument("--items", type=int, default=48, help="comprehend: items per sentence type")
    ap.add_argument("--words", type=int, default=64, help="patch: word-piece prefixes of each kind")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    if a.part == "trace":
        trace(a.docs, a.batch)
    elif a.part == "bands":
        bands(a.eval_docs, a.new_tokens, a.batch)
    elif a.part == "comprehend":
        comprehend(a.items, 64)
    else:
        patch(a.words, 32, 20)


if __name__ == "__main__":
    main()
