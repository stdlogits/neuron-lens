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

Outputs: ``data/summary/dark_sector_<part>_<model>.json`` and ``.log``.

    python scripts/checks/dark_sector.py trace
    python scripts/checks/dark_sector.py bands
    python scripts/checks/dark_sector.py comprehend
"""
import argparse
import json
import random
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


def ablated(out: torch.Tensor, M: int, sector: str, sec: dict, mean: torch.Tensor, mask=None) -> torch.Tensor:
    """``out`` (an MLP output ``[B, T, d]``) with its part in ``sector`` replaced by that part of ``mean``: at positions
    ≥ 1 (with ``mask`` ``[B, T]``, where it is set)."""
    dev = out.float() - mean
    if mask is None and out.shape[1] > 1:
        dev[:, 0] = 0
    s = sec["s"][M]
    part = {"mlp": lambda: dev, "dark": lambda: dev - project(dev, s, sec["V"][M]),
            "bright": lambda: project(dev, s, sec["V"][M]), "rand_d": lambda: dev - project(dev, s, sec["R"][M]),
            "rand_b": lambda: project(dev, s, sec["R"][M])}[sector]()
    if mask is not None:
        part = part * mask[..., None]
    return (out.float() - part).to(out.dtype)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("trace", "bands", "comprehend"))
    ap.add_argument("--docs", type=int, default=64, help="trace: scored DCLM documents")
    ap.add_argument("--eval-docs", type=int, default=32, help="bands: documents per slice")
    ap.add_argument("--new-tokens", type=int, default=48, help="bands: generated tokens")
    ap.add_argument("--items", type=int, default=48, help="comprehend: items per sentence type")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    if a.part == "trace":
        trace(a.docs, a.batch)
    elif a.part == "bands":
        bands(a.eval_docs, a.new_tokens, a.batch)
    else:
        comprehend(a.items, 64)


if __name__ == "__main__":
    main()
