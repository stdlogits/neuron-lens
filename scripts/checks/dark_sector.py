"""The dark sector: the part of each layer's residual the R-lens barely reads, what it carries, and what removing it does
to the model, early, in the middle and late (findings §21). Qwen3-8B, CUDA, the model in bf16 with fp32 statistics.

**Sectors.** At each layer's output (and MLP input), the R-lens readout in the residual's metric (``neuron_lens.metric``),
``B = W̄_U diag(γ_f) J_L diag(s)`` (``W̄_U``: the kept unembedding, centred over tokens, since softmax ignores a shift
shared by every token), has eigenvectors ``V`` ordered by gain. The **bright** sector is the top ``K_L`` of them, the
fewest that carry 90% of the lens energy (``Σ gain · (v·x)²``) of the layer's neuron writes, as a mean over the unit
writes (at the MLP input: over the gate and up rows as vectors, ``s ⊙ c``). The **dark** sector is the rest, orthogonal in
the metric. A sector of a vector is its part in the sector, mapped back to raw coordinates. The S-lens's sectors
(``patch``: ``s_bright``, ``s_dark``) take the same rule to the input side: the Gram of the token footprints at each
output, ``K_L ē`` (the S-lens's map from the embedding, on the kept tokens' embeddings, centred), as whitened vectors, in
place of the readout's.

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
  the mass on word-continuing pieces), against random subspaces of each sector's dimension. Frames: the same content in
  two settings that differ in what comes next, patched at the tokens they share or the last position: a question
  plainly or in the chat template (does the assistant's turn open a ``<think>``?), the assistant's turn with thinking on
  or off (does it open with "Okay"?), a sentence cut before a word as written or in capitals (is the next piece in capitals?), a shopping list
  after commas or bulleted (does a comma or a new line follow the last item?); scored by the log-odds of that mass.
  Every task also gets the KL from the clean target to the patched run, at the next token and over the target's greedy
  continuation of ``FOLLOW`` tokens (teacher-forced), and how far the patch moves both toward the source
  (``1 − KL(source ‖ patched) / KL(source ‖ target)``, the source given the same continuation); the sectors' shares of the
  source-target difference; and greedy generations.
- ``middle``: the layers where neither lens's bright sector holds what the pairs differ in. For the countries (at the
  subject's tokens), the answer's language, the next word piece and thinking mode, the pairs are split in two halves
  (the countries: the capital prompts and the official-language ones, since one country's direction can't carry
  another's; the rest share no country, document or question); at each of ``MIDDLE_LAYERS`` and ``K`` in ``MIDDLE_K``, the part of the
  source-target difference in a ``K``-dim subspace is patched on the second half: the R-lens's top directions, the
  S-lens's, those of their average (the two whitened Grams at unit trace), points a quarter, half and three quarters of
  the way along the shortest rotation from the S-lens's to the R-lens's (``geodesic``), the top directions of the first
  half's differences and of the gradients of its targets' score (an exact R-lens of each context), the union of the two
  lenses' top directions (``2K`` dims) against each lens's top ``2K``, and random subspaces of ``K`` and ``2K`` dims; scored as ``patch`` is, with the
  share of the second half's difference energy each subspace holds. The R-lens's top ``K`` with ``K`` random directions,
  and its next ``K`` alone, tell what the S-lens's top ``K`` adds to the R-lens's from what any further directions would.
- ``far``: two prompts with nothing in common but the chat template (thinking off): a story asked for, and small talk.
  Each sector is swapped at every position (aligned from the end, the first left out) or at the template's tail after the
  message, both ways round, scored as ``patch`` is (the two replies' first tokens; the KL toward the source over the
  target's own reply), with greedy generations.

Outputs: ``data/summary/dark_sector_<part>_<model>.json`` and ``.log``.

    python scripts/checks/dark_sector.py trace
    python scripts/checks/dark_sector.py bands
    python scripts/checks/dark_sector.py comprehend
    python scripts/checks/dark_sector.py patch
    python scripts/checks/dark_sector.py middle
    python scripts/checks/dark_sector.py far
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


def sectors(mid: bool = False, s_lens: bool = False, keep: int = 0) -> dict:
    """``s`` (the metric's √s² per layer), ``V`` (bright bases, whitened, fp16), ``K``, ``R`` (random ``K``-dim bases) at
    each layer's output; with ``mid`` the same at the MLP input (``s_mid``, ``V_mid``, ``K_mid``, ``R_mid``), plus the raw
    readout Grams ``GM`` at each output (for ``trace``'s loudness table), and ``J_out`` (CPU); with ``s_lens`` the
    S-lens's bright sector at each output (``VS``, ``KS``, ``RS``); with ``keep`` (and ``s_lens``) each output's top
    ``keep`` directions of the R-lens (``VRt``), the S-lens (``VSt``) and their average (``VMt``: the mean of the two
    whitened Grams, each scaled to unit trace), and a random ``keep``-dim basis (``Rt``)."""
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
    if s_lens:  # the token embeddings' Gram, centred over the kept tokens
        we = tensors(MODEL, [nm["embed"]])[nm["embed"]][kept()]
        mu_e = we.float().mean(0).to(DEV)
        GE = torch.zeros(d, d, device=DEV)
        for i in range(0, len(we), 16384):
            c = we[i:i + 16384].to(DEV).float() - mu_e
            GE += c.T @ c
        del we
        parts += ("K_out",)
    lm = arch.lens_means(MODEL, *arch.fits(MODEL), parts=parts)
    gen = torch.Generator(device=DEV).manual_seed(0)
    out = {k: [] for k in ("s", "V", "K", "R", "s_mid", "V_mid", "K_mid", "R_mid", "GM", "VS", "KS", "RS", "overlap",
                           "VRt", "VSt", "VMt", "Rt")}

    def top(Gr, s, X):
        gain, V = torch.linalg.eigh(s[:, None] * Gr * s[None, :])
        gain, V = gain.flip(0).clamp_min(0), V.flip(1)
        e = (X @ V).pow(2) * gain
        k = int(((e.cumsum(1) / e.sum(1, keepdim=True)).mean(0) < SHARE).sum()) + 1
        return k, V[:, :k].half().contiguous(), torch.linalg.qr(torch.randn(d, k, device=DEV, generator=gen))[0].half(), V

    for L in range(n):
        p = arch.layout(MODEL)[L].prefix
        w = tensors(MODEL, [p + "mlp.down_proj.weight", p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight",
                            p + "post_attention_layernorm.weight"])
        J = lm["J_out"][L].to(DEV).float() if L < lm["J_out"].shape[0] else torch.eye(d, device=DEV)
        Gr = J.T @ G @ J
        s = metric.s2(MODEL, "out", L, DEV).sqrt()
        X = w[p + "mlp.down_proj.weight"].to(DEV).float().T / s
        k, V, R, VR = top(Gr, s, X / X.norm(dim=1, keepdim=True))
        out["s"].append(s), out["K"].append(k), out["V"].append(V), out["R"].append(R)
        if s_lens:  # the token footprints K ē at this output, as whitened vectors: their Gram, and its top directions
            Kl = lm["K_out"][L].to(DEV).float()
            s2_ = s.pow(2)
            GS = Kl @ GE @ Kl.T / s2_[:, None] / s2_[None, :]
            k, VS, RS, VSf = top(GS, s, X / X.norm(dim=1, keepdim=True))
            out["KS"].append(k), out["VS"].append(VS), out["RS"].append(RS)
            out["overlap"].append((V.float().T @ VS.float()).pow(2).sum().item() / k)  # S-bright's share in R-bright
            if keep:
                wr, ws = s[:, None] * Gr * s[None, :], s[:, None] * GS * s[None, :]
                VM = torch.linalg.eigh(wr / wr.trace() + ws / ws.trace())[1].flip(1)
                out["VRt"].append(VR[:, :keep].half().contiguous()), out["VSt"].append(VSf[:, :keep].half().contiguous())
                out["VMt"].append(VM[:, :keep].half().contiguous())
                out["Rt"].append(torch.linalg.qr(torch.randn(d, keep, device=DEV, generator=gen))[0].half())
                del wr, ws, VM
            del Kl, GS
        if mid:
            out["GM"].append(Gr)
            s = metric.s2(MODEL, "mid", L, DEV).sqrt()
            Jm = lm["J_mid"][L].to(DEV).float()
            g = w[p + "post_attention_layernorm.weight"].to(DEV).float()
            rows = torch.cat([w[p + "mlp.gate_proj.weight"].to(DEV).float(), w[p + "mlp.up_proj.weight"].to(DEV).float()])
            Xm = rows * g * s  # covector rows c as whitened vectors s ⊙ c
            k, V, R, _ = top(Jm.T @ G @ Jm, s, Xm / Xm.norm(dim=1, keepdim=True))
            out["s_mid"].append(s), out["K_mid"].append(k), out["V_mid"].append(V), out["R_mid"].append(R)
            del Jm, Xm, rows
        del w, J, X
    out["J_out"] = lm["J_out"]
    log("sectors: K_out " + " ".join(map(str, out["K"])) + (" | K_mid " + " ".join(map(str, out["K_mid"])) if mid else ""))
    if s_lens:
        log("S-lens sectors: K " + " ".join(map(str, out["KS"])))
        log("  their share in the R-lens's bright sector " + " ".join(f"{x:.2f}" for x in out["overlap"])
            + " (random: " + " ".join(f"{k / d:.2f}" for k in out["K"]) + ")")
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


SPLIT = {"bright": ("V", True), "dark": ("V", False), "rand_b": ("R", True), "rand_d": ("R", False),
         "s_bright": ("VS", True), "s_dark": ("VS", False), "rand_sb": ("RS", True), "rand_sd": ("RS", False)}


def part(x: torch.Tensor, M: int, sector: str, sec: dict) -> torch.Tensor:
    """The part of raw vectors ``x`` at layer ``M``'s output in ``sector`` (``mlp``, ``full``: all of it; the ``s_``
    sectors and their random controls are the S-lens's)."""
    if sector in ("mlp", "full"):
        return x.float()
    basis, inside = SPLIT[sector]
    p = project(x, sec["s"][M], sec[basis][M])
    return p if inside else x.float() - p


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
PATCH = ("dark", "bright", "rand_d", "rand_b", "s_dark", "s_bright", "rand_sd", "rand_sb", "full")
ABBR = {"dark": "dk", "bright": "br", "rand_d": "rd", "rand_b": "rb", "s_dark": "Sd", "s_bright": "Sb", "rand_sd": "rSd",
        "rand_sb": "rSb", "full": "fu"}
SHOW = (2, 6, 12, 18, 24, 30)  # layers of the residual patches in the generations
TOPK = 1024  # a reference distribution's kept tokens in the KLs (the rest pooled into one)
MIDDLE_K = (8, 32, 128, 512)  # subspace dimensions of the middle part
MIDDLE_LAYERS = (0, 1, 2, 4, 6, 8, 10, 12, 16, 20)
MIDDLE_TASKS = ("fact_subject", "lang_zh_from_en", "word_ident", "frame_think_last")
GEO = (0.25, 0.5, 0.75)  # points on the shortest rotation from the S-lens's top directions to the R-lens's
FOLLOW = 20  # tokens of the target's greedy continuation, scored teacher-forced
STORIES = [
    "Write a short story about a dragon who learns to bake bread.",
    "Tell me a story about a lighthouse keeper who finds a message in a bottle.",
    "Write a bedtime story about a fox and an owl who become friends.",
    "Write a story about a robot exploring an abandoned city.",
    "Tell me a short story about a girl who can talk to trees.",
    "Write a story about a sailor lost at sea during a storm.",
    "Write a fairy tale about a king who loses his crown.",
    "Tell me a story about a detective solving a mystery on a night train.",
    "Write a short story about a cat who becomes the mayor of a small town.",
    "Write a story about two astronauts stranded on the moon.",
    "Tell me a story about a boy who finds a key that opens any door.",
    "Write a short story about an old man and the garden he planted for his wife.",
    "Write a story about a village that wakes up with no memories.",
    "Tell me a story about a violinist who plays for the stars.",
    "Write a short story about a ghost who is afraid of people.",
    "Write a story about a baker whose cakes grant wishes.",
]
CHATS = [
    "Hey! How was your weekend? Mine was pretty relaxing.",
    "Hi there, I just got back from the gym and I'm exhausted.",
    "Good morning! Did you sleep well last night?",
    "Ugh, my coffee machine broke this morning. How's your day going?",
    "Hey, what are you up to tonight? I'm thinking of ordering pizza.",
    "Hi! I just finished a really long meeting. How are you?",
    "Hey, long time no talk! What have you been doing lately?",
    "Hello! It's raining here again. Is the weather nice where you are?",
    "Hi, I'm bored at work. Want to chat for a bit?",
    "Hey! I just adopted a puppy and I'm so excited.",
    "Morning! I can't decide what to have for breakfast.",
    "Hi there! My sister is visiting this weekend. Any plans for yours?",
    "Hey, I finally finished that book you recommended. Loved it.",
    "Hi! Just wanted to say hello and see how you're doing.",
    "Ugh, Mondays. How was your commute today?",
    "Hey! I'm learning to cook. Last night I made pasta from scratch.",
]
ITEMS = [  # one token each after a space
    "apple", "banana", "cherry", "grape", "lemon", "orange", "peach", "pear", "plum", "mango", "onion", "garlic",
    "carrot", "potato", "tomato", "pepper", "rice", "bread", "cheese", "butter", "milk", "sugar", "salt", "flour",
    "eggs", "chicken", "beef", "pork", "fish", "salmon", "tuna", "beans", "corn", "peas", "lettuce", "spinach",
    "cabbage", "soap", "paper", "pens", "tape", "glue", "string", "nails", "screws", "towels", "candles",
    "coffee", "tea", "juice", "water", "soda", "cereal", "pasta", "noodles", "yogurt", "honey", "jam", "nuts"
]


def geodesic(U1: torch.Tensor, U2: torch.Tensor, t: float) -> torch.Tensor:
    """The subspace a share ``t`` of the way along the shortest rotation (the Grassmann geodesic) from the span of
    orthonormal ``U1`` to that of ``U2`` (``[d, K]`` each), as an orthonormal basis."""
    A, cos, Bt = torch.linalg.svd(U1.T @ U2)
    theta = torch.arccos(cos.clamp(-1, 1))
    X, Y = U1 @ A, U2 @ Bt.T  # principal vectors: x_i · y_j = δ_ij cos θ_i
    Q = (Y - X * cos) / torch.sin(theta).clamp_min(1e-6)
    return torch.linalg.qr(X * torch.cos(t * theta) + Q * torch.sin(t * theta))[0]


def patch(n_words: int, n_frame: int, batch: int, new: int, which: str = "patch") -> None:
    middle = which == "middle"
    sec = sectors(s_lens=True, keep=2 * max(MIDDLE_K) if middle else 0)
    from transformers import AutoModelForCausalLM
    tok = tokenizer()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEV).eval()
    layers = model.model.layers
    nL, d = len(layers), sec["s"][0].shape[0]
    n_vocab = model.lm_head.weight.shape[0]
    strs = tok.batch_decode([[i] for i in range(len(tok))]) + [""] * (n_vocab - len(tok))
    enc = lambda text: tok(text).input_ids
    mask = lambda f: torch.tensor([bool(f(x)) for x in strs], device=DEV)
    letters = lambda x: re.sub(r"[^A-Za-z]", "", x)
    cont = mask(lambda x: re.fullmatch(r"[a-z]+", x))  # pieces that continue a word
    cjk = mask(lambda x: re.search(r"[一-鿿]", x))
    caps = mask(lambda x: letters(x).isupper())
    lower = mask(lambda x: re.search(r"[a-z]", x))
    nl, comma = mask(lambda x: x.startswith("\n")), mask(lambda x: x.startswith(","))
    think = mask(lambda x: x == "<think>")
    rng = random.Random(0)
    S = {"mode": None, "basis": None}
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
            dt = S["src"][kind][M].float() - t.float()
            add = project(dt, sec["s"][M], S["basis"]) if S["basis"] is not None else part(dt, M, S["sector"], sec)
            x[b, p] = (t.float() + add).to(x.dtype)
            return (x, *out[1:]) if isinstance(out, tuple) else x
        return h

    hooks = [h for M, lay in enumerate(layers)
             for h in (lay.register_forward_hook(hook(M, "h")), lay.mlp.register_forward_hook(hook(M, "m")))]
    tail = lambda v: (1 - v.exp().sum(-1)).clamp_min(1e-9).log()

    def kl(lp, i, v, t):
        """KL from a reference (log-probs ``v`` of its top tokens ``i``, ``t`` of the rest pooled) to ``lp``."""
        q = lp.gather(-1, i)
        return (v.exp() * (v - q)).sum(-1) + t.exp() * (t - tail(q))

    def run(rows, pos=None, mode=None, src=None, start=None, n_pos=1, refs=(), keep=False, **kw):
        """Log-probs after each row's prompt (row i's ends at ``start[i]``, by default its last token); the next
        token's in ``lp``. ``pos``: each row's positions, captured (``cap``) or patched (``h``: the layer's output,
        ``m``: its MLP output) from ``src`` (``[nL, P, d]``, the rows' positions in order). Over ``n_pos`` positions from
        the prompt's end (the next token, then a continuation's): ``kl``, the KL from each reference to this run
        (``[R, refs, n_pos]``), and with ``keep``, this run's top tokens there as a reference (``top``)."""
        pos = pos or [[] for _ in rows]
        start = start or [len(r) - 1 for r in rows]
        off = np.cumsum([0] + [len(q) for q in pos])
        acc = {"lp": [], "kl": [], "top": [], "h": [], "m": []}
        for b0 in range(0, len(rows), batch):
            sl = slice(b0, b0 + batch)
            rs, ps = rows[sl], pos[sl]
            n = max(map(len, rs))
            ids = torch.full((len(rs), n), tok.pad_token_id)
            att = torch.zeros_like(ids)
            for i, r in enumerate(rs):
                ids[i, :len(r)], att[i, :len(r)] = torch.tensor(r), 1
            S.update({"mode": mode, "T": n, "basis": None, "p": torch.tensor([j for q in ps for j in q], dtype=torch.long, device=DEV),
                      "b": torch.tensor([i for i, q in enumerate(ps) for _ in q], dtype=torch.long, device=DEV), **kw})
            if src is not None:
                S["src"] = {k: v[:, off[b0]:off[b0 + len(rs)]] for k, v in src.items()}
            hs = model.model(input_ids=ids.to(DEV), attention_mask=att.to(DEV), use_cache=False).last_hidden_state
            at = torch.tensor(start[sl], device=DEV)[:, None] + torch.arange(n_pos, device=DEV)
            lp = model.lm_head(hs[torch.arange(len(rs), device=DEV)[:, None], at]).float().log_softmax(-1)
            acc["lp"].append(lp[:, 0])
            if keep:
                v, i = lp.topk(TOPK, -1)
                acc["top"].append((i, v, tail(v)))
            if refs:
                acc["kl"].append(torch.stack([kl(lp, *(x[sl] for x in ref)) for ref in refs], 1))
            if mode == "cap":
                for k in ("h", "m"):
                    acc[k].append(torch.stack(cap[k]))
        S["mode"] = None
        out = {"lp": torch.cat(acc["lp"])}
        if keep:
            out["top"] = tuple(torch.cat(x) for x in zip(*acc["top"]))
        if refs:
            out["kl"] = torch.cat(acc["kl"])
        if mode == "cap":
            out["cap"] = {k: torch.cat(acc[k], 1) for k in ("h", "m")}
        return out

    def greedy(rows, n):
        rows = [list(r) for r in rows]
        for _ in range(n):
            for r, t in zip(rows, run(rows)["lp"].argmax(-1).tolist()):
                r.append(t)
        return rows

    def generate(row, pos, mode=None, src=None, **kw):
        S.update({"mode": mode, "T": len(row), "b": torch.zeros(len(pos), dtype=torch.long, device=DEV), "basis": None,
                  "p": torch.tensor(pos, dtype=torch.long, device=DEV), "src": src, **kw})
        with torch.inference_mode():
            out = model.generate(torch.tensor([row], device=DEV), max_new_tokens=new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        S["mode"] = None
        return tok.decode(out[0, len(row):])

    def first(words):
        """First tokens of each spelling, leaving out bare whitespace."""
        return sorted({i for w in words for i in enc(w)[:1] if strs[i].strip()})

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

    def m_mass(a, b=None):
        """The log-odds of the next token's mass on vocabulary mask ``a`` against ``b`` (by default the rest)."""
        def m(lp, pairs):
            pr = lp.exp()
            pa = (pr * a.float()).sum(-1)
            pb = (pr * b.float()).sum(-1) if b is not None else 1 - pa
            return {"form": (pa + 1e-9).log() - (pb.clamp_min(0) + 1e-9).log(), "top_a": a[lp.argmax(-1)].float()}
        return m

    summ = lambda m: {k: v.float().mean().item() for k, v in m.items()}
    res = {"K_out": sec["K"], "K_S": sec["KS"], "S_in_R": sec["overlap"], "tasks": {}}

    def task(name, pairs, metric, primary, shows=1):
        tr, sr = [p["t"] for p in pairs], [p["s"] for p in pairs]
        tp, sp = [p["tp"] for p in pairs], [p["sp"] for p in pairs]
        # the target's greedy continuation, after the target's prompt and after the source's
        cont_ = [r[len(t):] for r, t in zip(greedy(tr, FOLLOW), tr)]
        tc, sc = [t + c for t, c in zip(tr, cont_)], [s + c for s, c in zip(sr, cont_)]
        at_t, at_s = [len(t) - 1 for t in tr], [len(s) - 1 for s in sr]
        ct = run(tc, tp, "cap", start=at_t, n_pos=FOLLOW, keep=True)
        cs = run(sc, sp, "cap", start=at_s, n_pos=FOLLOW, keep=True, refs=(ct["top"],))
        src, tgt, refs = cs["cap"], ct["cap"], (ct["top"], cs["top"])
        base = run(tc, start=at_t, n_pos=FOLLOW, refs=refs)["kl"]
        mean = lambda x: x.mean().item()
        out = {"n": len(pairs), "clean_t": summ(metric(ct["lp"], pairs)), "clean_s": summ(metric(cs["lp"], pairs)),
               "kl": {"t_s_next": mean(cs["kl"][:, 0, 0]), "t_s_follow": mean(cs["kl"][:, 0, 1:]),
                      "s_t_next": mean(base[:, 1, 0]), "s_t_follow": mean(base[:, 1, 1:]), "floor": mean(base[:, 0])},
               "examples": [[tok.decode(p["t"]), tok.decode(p["s"]), tok.decode(c)] for p, c in zip(pairs[:4], cont_)],
               "share": {}, "resid": {}, "mlp": {}}
        for k in ("h", "m"):  # where the source-target difference sits, by sector
            sh = {"bright": [], "rand_b": [], "s_bright": [], "rand_sb": [], "dims": [K / d for K in sec["K"]],
                  "s_dims": [K / d for K in sec["KS"]]}
            for L in range(nL):
                dw = (src[k][L].float() - tgt[k][L].float()) / sec["s"][L]
                tot = dw.pow(2).sum().item()
                for name_, B in (("bright", sec["V"][L]), ("rand_b", sec["R"][L]), ("s_bright", sec["VS"][L]),
                                 ("rand_sb", sec["RS"][L])):
                    sh[name_].append((dw.half() @ B).float().pow(2).sum().item() / max(tot, 1e-12))
            out["share"][k] = sh

        def cond(**kw):
            o = run(tc, tp, src=src, start=at_t, n_pos=FOLLOW, refs=refs, **kw)
            r = summ(metric(o["lp"], pairs))
            kl_ = o["kl"]
            r |= {"kl_next": mean(kl_[:, 0, 0]), "kl_follow": mean(kl_[:, 0, 1:]),
                  "frac_next": 1 - mean(kl_[:, 1, 0]) / out["kl"]["s_t_next"],
                  "frac_follow": 1 - mean(kl_[:, 1, 1:]) / out["kl"]["s_t_follow"]}
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
                g["resid"][L] = {s_: generate(tr[j], tp[j], "h", sj, layers={L}, sector=s_)
                                 for s_ in ("dark", "bright", "s_bright", "full")}
            for band in ("L0", "L1-8", "L9-20", "L21-35"):
                g["mlp"][band] = {s_: generate(tr[j], tp[j], "m", sj, layers=set(BANDS[band]), sector=s_)
                                  for s_ in ("dark", "bright", "full")}
            out["gen"].append(g)
        res["tasks"][name] = out
        show = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 35)
        kq = out["kl"]
        log(f"[{name}] n={len(pairs)} clean target " + " ".join(f"{k} {out['clean_t'][k]:.2f}" for k in primary)
            + " | source " + " ".join(f"{k} {out['clean_s'][k]:.2f}" for k in primary)
            + f" | KL t→s next {kq['t_s_next']:.2f} follow {kq['t_s_follow']:.2f}, s→t next {kq['s_t_next']:.2f}"
            f" follow {kq['s_t_follow']:.2f} (floor {kq['floor']:.1e})")
        log("  layer" + " " * 12 + " ".join(f"{L:5d}" for L in show))
        for k in (*("frac_" + k for k in primary), "frac_next", "frac_follow", "kl_next", "kl_follow"):
            for sector in PATCH:
                log(f"  {k[:12]:12s} {ABBR[sector]:3s} " + " ".join(f"{out['resid'][sector][k][L]:+5.2f}" for L in show))
            log(f"  {k[:12]:12s} MLP bands " + " | ".join(
                f"{b} " + " ".join(f"{ABBR[s_]} {out['mlp'][b][s_][k]:+.2f}" for s_ in PATCH) for b in BANDS))
        log("  bright share of the difference (h) " + " ".join(f"{out['share']['h']['bright'][L]:.2f}" for L in show))
        log("  random K-dim share             " + " ".join(f"{out['share']['h']['rand_b'][L]:.2f}" for L in show))
        log("  S-bright share                 " + " ".join(f"{out['share']['h']['s_bright'][L]:.2f}" for L in show))
        log("  random K_S-dim share           " + " ".join(f"{out['share']['h']['rand_sb'][L]:.2f}" for L in show))
        save(which, res)

    def frame(name, pairs, a, b=None, scopes=("shared", "last"), shows=1):
        """Pairs that differ in frame around the same content, kept where the clean source's next token is in ``a`` and
        the target's isn't; patched at the shared tokens (``tshare``, ``sshare``) or the last position."""
        top_t = run([p["t"] for p in pairs])["lp"].argmax(-1).tolist()
        top_s = run([p["s"] for p in pairs])["lp"].argmax(-1).tolist()
        kept_ = [p for p, x, y in zip(pairs, top_t, top_s) if a[y] and not a[x]][:n_frame]
        log(f"frame {name}: {len(kept_)} of {len(pairs)} pairs kept")
        for scope in scopes:
            for p in kept_:
                p["tp"], p["sp"] = ((p["tshare"], p["sshare"]) if scope == "shared"
                                    else ([len(p["t"]) - 1], [len(p["s"]) - 1]))
            do(f"frame_{name}_{scope}", kept_, m_mass(a, b), ("form",), shows=shows if scope == scopes[0] else 0)

    def gradients(pairs, metric, key):
        """Each layer's output gradient of the clean target's ``key`` at each pair's patched positions, as raw covectors:
        ``[nL]`` of ``[P, d]``."""
        store, out = {}, [[] for _ in range(nL)]

        def grab(M):
            def h(mod, args, o):
                x = o[0] if isinstance(o, tuple) else o
                x.retain_grad()
                store[M] = x
            return h

        model.requires_grad_(False)
        hs_ = [lay.register_forward_hook(grab(M)) for M, lay in enumerate(layers)]
        with torch.inference_mode(False), torch.enable_grad():
            for b0 in range(0, len(pairs), 8):
                bp = pairs[b0:b0 + 8]
                n = max(len(q["t"]) for q in bp)
                ids = torch.full((len(bp), n), tok.pad_token_id)
                att = torch.zeros_like(ids)
                for i, q in enumerate(bp):
                    ids[i, :len(q["t"])], att[i, :len(q["t"])] = torch.tensor(q["t"]), 1
                emb = model.model.embed_tokens(ids.to(DEV)).detach().requires_grad_(True)
                hs = model.model(inputs_embeds=emb, attention_mask=att.to(DEV), use_cache=False).last_hidden_state
                last = torch.tensor([len(q["t"]) - 1 for q in bp], device=DEV)
                lp = model.lm_head(hs[torch.arange(len(bp), device=DEV), last]).float().log_softmax(-1)
                metric(lp, bp)[key].sum().backward()
                for M in range(nL):
                    g = store[M].grad
                    out[M].append(torch.cat([g[i, q["tp"]] for i, q in enumerate(bp)]).float())
                store.clear()
        for h in hs_:
            h.remove()
        return [torch.cat(g) for g in out]

    SWITCH = {"fact_subject": "flip", "lang_zh_from_en": "top_ts", "word_ident": "flip", "frame_think_last": "top_a"}

    def middle_task(name, pairs, metric, primary, shows=0):
        """At each of ``MIDDLE_LAYERS``, ``K``-dim subspaces of each family patched (their part of the source-target
        difference) on the pairs of half 1: the R-lens's and the S-lens's top directions, their average, points on the
        rotation from the S-lens's to the R-lens's, the top directions of half 0's differences and of the gradients of
        its targets' score, and a random subspace. Scored by the share of the primary score's gap, how often the top
        token switches, the KL to the clean target, and the share of half 1's difference energy each subspace holds."""
        if name not in MIDDLE_TASKS:
            return
        fit, test = [p for p in pairs if p["half"] == 0], [p for p in pairs if p["half"] == 1]
        cap_ = lambda ps, k: run([p[k] for p in ps], [p[k + "p"] for p in ps], "cap")
        sf, tf, ss, ts = cap_(fit, "s"), cap_(fit, "t"), cap_(test, "s"), cap_(test, "t")
        rows_t, pos_t = [p["t"] for p in test], [p["tp"] for p in test]
        ref = run(rows_t, keep=True)["top"]
        clean_t, clean_s = summ(metric(ts["lp"], test)), summ(metric(ss["lp"], test))
        grads = gradients(fit, metric, primary[0])
        out = {"n_fit": len(fit), "n_test": len(test), "clean_t": clean_t, "clean_s": clean_s, "layers": {}}
        k0, sw = primary[0], SWITCH[name]
        log(f"[{name}] fit {len(fit)} pairs ({len(grads[0])} positions), test {len(test)}; clean target {k0} "
            f"{clean_t[k0]:.2f} {sw} {clean_t[sw]:.2f} | source {clean_s[k0]:.2f} {clean_s[sw]:.2f}")
        for L in MIDDLE_LAYERS:
            s_ = sec["s"][L]
            dfit = (sf["cap"]["h"][L].float() - tf["cap"]["h"][L].float()) / s_
            dtest = (ss["cap"]["h"][L].float() - ts["cap"]["h"][L].float()) / s_
            fam = {"R": sec["VRt"][L].float(), "S": sec["VSt"][L].float(), "mix": sec["VMt"][L].float(),
                   "diff": torch.linalg.svd(dfit, full_matrices=False).Vh.T,
                   "grad": torch.linalg.svd(grads[L] * s_, full_matrices=False).Vh.T, "rand": sec["Rt"][L].float()}
            row = {}
            for K in MIDDLE_K:
                bases = {f: B[:, :K] for f, B in fam.items()}
                bases |= {f"geo{t}": geodesic(bases["S"], bases["R"], t) for t in GEO}
                bases["union"] = torch.linalg.qr(torch.cat([bases["S"], bases["R"]], 1))[0]  # 2K dims
                bases |= {"R2": fam["R"][:, :2 * K], "S2": fam["S"][:, :2 * K], "rand2": fam["rand"][:, :2 * K],
                          "Rnext": fam["R"][:, K:2 * K],  # the R-lens's next K directions alone, and its top K with K random
                          "R+rand": torch.linalg.qr(torch.cat([bases["R"], fam["rand"][:, K:2 * K]], 1))[0]}
                for f, B in bases.items():
                    o = run(rows_t, pos_t, "h", src=ss["cap"], layers={L}, basis=B.half().contiguous(), refs=(ref,))
                    r = summ(metric(o["lp"], test))
                    cell = {"dim": B.shape[1], "kl": o["kl"].mean().item(), "switch": r[sw],
                            "energy": (dtest @ B).pow(2).sum().item() / dtest.pow(2).sum().item()}
                    for k in primary:
                        cell["frac_" + k] = (r[k] - clean_t[k]) / (clean_s[k] - clean_t[k])
                    if f in ("diff", "grad"):
                        cell |= {f"in_{g}": (B.T @ bases[g]).pow(2).sum().item() / B.shape[1] for g in ("R", "S")}
                    row[f"{f}@{K}"] = cell
                log(f"  L{L:<2d} K {K:3d} " + " ".join(
                    f"{f} {row[f'{f}@{K}']['frac_' + k0]:+.2f}/{row[f'{f}@{K}']['switch']:.2f}" for f in bases))
            log(f"  L{L:<2d} energy@128 " + " ".join(f"{f} {row[f'{f}@128']['energy']:.2f}" for f in bases)
                + " | diff in R/S " + "/".join(f"{row['diff@128'][f'in_{g}']:.2f}" for g in ("R", "S"))
                + ", grad in R/S " + "/".join(f"{row['grad@128'][f'in_{g}']:.2f}" for g in ("R", "S"))
                + f" (random {128 / d:.2f})")
            out["layers"][L] = row
        res.setdefault("middle", {})[name] = out
        save("middle", res)

    do = middle_task if middle else task
    halves = (0, 1) if middle else (None,)

    with torch.inference_mode():
        if which == "far":  # a story asked for against small talk, in the chat template with thinking off
            user, turn = enc("<|im_start|>user\n"), enc("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
            sides = {k: sorted((user + enc(x) + turn for x in v), key=len) for k, v in (("story", STORIES), ("chat", CHATS))}
            tops = {k: run(v)["lp"].argmax(-1).tolist() for k, v in sides.items()}
            for tl, sl in (("story", "chat"), ("chat", "story")):
                n_ = len(sides[sl])
                pairs = [{"t": sides[tl][i], "s": sides[sl][(i + k) % n_], "at": tops[tl][i], "as": tops[sl][(i + k) % n_]}
                         for k in (0, 1) for i in range(len(sides[tl])) if tops[tl][i] != tops[sl][(i + k) % n_]]
                log(f"far {tl} from {sl}: {len(pairs)} pairs whose replies open differently")
                for scope in ("all", "tail"):  # every position from the end (not the first), or the template's tail
                    for q in pairs:
                        m_ = min(len(q["t"]), len(q["s"])) - 1 if scope == "all" else len(turn)
                        q["tp"], q["sp"] = (list(range(len(q[k]) - m_, len(q[k]))) for k in ("t", "s"))
                    do(f"far_{tl}_from_{sl}_{scope}", pairs, m_fact, ("ld",), shows=2 if scope == "all" else 0)
        if which != "far":
            # facts: aligned prompts that differ in the country, patched at every position, the country's or the last
            rows = []
            for en, _, capital, _, lang in COUNTRIES:
                for rel, text, ans in (("capital", f"The capital of {en} is", capital),
                                       ("language", f"The official language of {en} is", lang)):
                    rows.append({"rel": rel, "en": en, "ids": enc(text), "a": enc(" " + ans)[0], "half": int(rel == "language")})
            lp = run([r["ids"] for r in rows])["lp"]
            rows = [r for r, t in zip(rows, lp.argmax(-1).tolist()) if t == r["a"]]
            groups = {}
            for r in rows:
                groups.setdefault((r["rel"], len(r["ids"])), []).append(r)
            pairs = []
            for g in groups.values():
                g.sort(key=lambda r: (r["en"] not in ("France", "Italy"), rng.random()))
                for h_ in halves:  # the middle part fits on the capital prompts and scores on the official-language ones
                    gh = [r for r in g if h_ in (None, r["half"])]
                    for i, r in enumerate(gh):
                        o = next((gh[(i + k) % len(gh)] for k in range(1, len(gh)) if gh[(i + k) % len(gh)]["a"] != r["a"]),
                                 None)
                        if o is not None:
                            pairs.append({"t": r["ids"], "s": o["ids"], "at": r["a"], "as": o["a"], "rel": r["rel"], "half": h_})
            pairs.sort(key=lambda p: p["rel"] != "capital")
            log(f"facts: {len(rows)} of {2 * len(COUNTRIES)} prompts answered right, {len(pairs)} pairs")
            for scope in ("all", "subject", "last"):
                for p in pairs:
                    n = len(p["t"])
                    p["tp"] = p["sp"] = {"all": list(range(1, n)), "last": [n - 1],
                                         "subject": [k for k in range(n) if p["t"][k] != p["s"][k]]}[scope]
                do(f"fact_{scope}", pairs, m_fact, ("ld",), shows=1 if scope == "all" else 0)

            # language: the last position of a capital prompt in one language, from one about another country in the other
            ok = []
            for ci, (en, zh, capital, capital_zh, _) in enumerate(COUNTRIES):
                c = {"en": enc(f"The capital of {en} is"), "zh": enc(f"{zh}的首都是"), "half": ci % 2,
                     "a_en": first([" " + capital, capital]), "a_zh": first([capital_zh, " " + capital_zh]), "name": en}
                ok.append(c)
            lp = run([c[k] for c in ok for k in ("en", "zh")])["lp"].argmax(-1).tolist()
            ok = [c for i, c in enumerate(ok) if lp[2 * i] in c["a_en"] and lp[2 * i + 1] in c["a_zh"]]
            ok.sort(key=lambda c: (c["name"] not in ("China", "France"), c["name"] != "China", rng.random()))
            log(f"language: {len(ok)} of {len(COUNTRIES)} countries answered right in both languages")
            for tl, sl in (("zh", "en"), ("en", "zh")):
                pairs = []
                for h_ in halves:
                    okh = [c for c in ok if h_ in (None, c["half"])]
                    for i, x in enumerate(okh):
                        for k in range(1, len(okh)):
                            y = okh[(i + k) % len(okh)]
                            sets = [x["a_" + tl], x["a_" + sl], y["a_" + tl], y["a_" + sl]]
                            if len({t for s_ in sets for t in s_}) == sum(map(len, sets)):
                                break
                        else:
                            continue
                        pairs.append({"t": x[tl], "s": y[sl], "tp": [len(x[tl]) - 1], "sp": [len(y[sl]) - 1], "half": h_,
                                      "cells": {"tt": x["a_" + tl], "ts": x["a_" + sl], "st": y["a_" + tl], "ss": y["a_" + sl]}})
                do(f"lang_{tl}_from_{sl}", pairs, m_lang, ("lang", "content"), shows=1)

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
            lp = run([r["ids"] for r in mid + bound])["lp"]
            top, p_top, pc = lp.argmax(-1).tolist(), lp.max(-1).values.exp().tolist(), (lp.exp() * cont).sum(-1).tolist()
            n_mid = len(mid)
            mid = [r for i, r in enumerate(mid) if top[i] == r["next"] and p_top[i] >= 0.5][:n_words]
            bound = [r for j, r in enumerate(bound) if pc[n_mid + j] < 0.05][:n_words]
            log(f"words: {len(mid)} mid-word and {len(bound)} word-final prefixes kept")
            last = lambda r: [len(r["ids"]) - 1]
            ident = []
            for h_ in halves:
                mh = [r for j, r in enumerate(mid) if h_ in (None, j % 2)]
                for i, r in enumerate(mh):
                    o = next((mh[(i + k) % len(mh)] for k in range(1, len(mh)) if mh[(i + k) % len(mh)]["next"] != r["next"]),
                             None)
                    ident.append({"t": r["ids"], "s": o["ids"], "tp": last(r), "sp": last(o), "tn": r["next"], "sn": o["next"],
                                  "half": h_})
            do("word_ident", ident, m_word, ("ld",), shows=2)
            k = min(len(mid), len(bound))
            do("word_end", [{"t": mid[i]["ids"], "s": bound[i]["ids"], "tp": last(mid[i]), "sp": last(bound[i])}
                              for i in range(k)], m_word, ("cont",), shows=2)
            do("word_start", [{"t": bound[i]["ids"], "s": mid[i]["ids"], "tp": last(bound[i]), "sp": last(mid[i])}
                                for i in range(k)], m_word, ("cont",), shows=1)

            # frames: the same content in two settings, which differ in what comes next
            qs = [f"What is the capital of {en}?" for en, *_ in COUNTRIES] + [
                f"What is the official language of {en}?" for en, *_ in COUNTRIES]
            rng.shuffle(qs)
            user, turn = enc("<|im_start|>user\n"), enc("<|im_end|>\n<|im_start|>assistant\n")
            chat, mode_ = [], []
            for qi, q in enumerate(qs):  # a plain question and answer against the chat template: does the turn open a thought?
                Q, pre = enc(q), enc("Question:\n")
                chat.append({"t": pre + Q + enc("\nAnswer:"), "s": user + Q + turn,
                             "tshare": list(range(len(pre), len(pre) + len(Q))),
                             "sshare": list(range(len(user), len(user) + len(Q)))})
                mode_.append({"t": user + Q + turn + enc("<think>\n\n</think>\n\n"), "s": user + Q + turn + enc("<think>\n"),
                              "half": qi % 2})
            frame("chat", chat, think)
            frame("think", mode_, mask(lambda x: x in ("Okay", "Alright", "Hmm", "Well")), scopes=("last",))
            sents = []  # one sentence of each document, 9-30 tokens, on one line
            for doc in docs:
                for i in range(1, T - 32):
                    if strs[doc[i - 1]].endswith(".") and re.fullmatch(r" [A-Z][a-z]+", strs[doc[i]]):
                        j = next((j for j in range(i + 8, i + 30) if strs[doc[j]] == "."), None)
                        if j is not None and not any("\n" in strs[x] for x in doc[i:j + 1]):
                            sents.append(tok.decode(doc[i:j + 1]).strip())
                            break
            upper = []
            for s_ in sents:
                t = enc(s_)
                cut = [k for k in range(5, len(t)) if re.fullmatch(r" [a-z]{3,}", strs[t[k]])]
                if cut:  # a sentence cut before a lowercase word, as written and in capitals
                    text = tok.decode(t[:cut[len(cut) // 2]])
                    upper.append({"t": enc(text), "s": enc(text.upper())})
            frame("caps", upper, caps, lower, scopes=("last",))
            lists = []
            for _ in range(4 * n_frame):  # a list after commas against a bulleted list: what follows the last item?
                t, s_, tsh, ssh = enc("Shopping list:"), enc("Shopping list:\n-"), [], []
                for k, w in enumerate(rng.sample(ITEMS, rng.randint(4, 7))):
                    if k:
                        t, s_ = t + enc(","), s_ + enc("\n-")
                    tsh.append(len(t))
                    ssh.append(len(s_))
                    t, s_ = t + enc(" " + w), s_ + enc(" " + w)
                lists.append({"t": t, "s": s_, "tshare": tsh, "sshare": ssh})
            frame("list", lists, nl, comma)
    for h in hooks:
        h.remove()
    log("PATCH DONE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("trace", "bands", "comprehend", "patch", "middle", "far"))
    ap.add_argument("--docs", type=int, default=64, help="trace: scored DCLM documents")
    ap.add_argument("--eval-docs", type=int, default=32, help="bands: documents per slice")
    ap.add_argument("--new-tokens", type=int, default=48, help="bands: generated tokens")
    ap.add_argument("--items", type=int, default=48, help="comprehend: items per sentence type")
    ap.add_argument("--words", type=int, default=64, help="patch: word-piece prefixes of each kind")
    ap.add_argument("--frames", type=int, default=64, help="patch: pairs of each frame")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    if a.part == "trace":
        trace(a.docs, a.batch)
    elif a.part == "bands":
        bands(a.eval_docs, a.new_tokens, a.batch)
    elif a.part == "comprehend":
        comprehend(a.items, 64)
    else:
        patch(a.words, a.frames, 32, 20, which=a.part)


if __name__ == "__main__":
    main()
