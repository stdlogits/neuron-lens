"""The residual's coordinate sectors (findings §22): where the R-lens and the S-lens read the residual on its own
coordinates, what each part carries, and what neurons and heads do with it. Qwen3-8B and Gemma 4 12B (``MODEL``), the
lenses of record; the model in its half dtype, statistics in fp32; CUDA, else MPS.

**Documents.** 224 full-length DCLM documents of 128 tokens (``lens_docs``): the first 64 are scored, the next 96 define
the sectors (split by document into even and odd halves where a result is held out), the last 64 give the means that
ablations put in. Positions start at 1; positions whose residual norm passes 5× the median at any layer (the attention
sinks' massive activations, §9) are left out of the statistics, but never out of the model.

**Per coordinate**, at each layer's output: the signed share of the R-lens readout, ``p_i = Σ_t h_ti (A h_t)_i /
Σ_t h_tᵀ A h_t`` with ``A = Jᵀ W̄ᵀ W̄ J`` (``W̄``: the kept unembedding, centred, times the final norm's weight; it sums to
one and needs no metric); the S-lens token share, the squared correlation over positions of ``h_i`` with the current
token's footprint ``K ē``; the lens-free token share (held-out R² of ``h_i`` predicted by its token's mean); persistence,
the correlation of the standardised coordinate at ``L`` and ``L + 2`` (or ``+ 4``); write shielding, ``−log`` of the
layer's MLP and attention write energy at ``i`` in the residual's σ frame over its mean (weights only); and the share of
the residual's energy.

**Sectors.** ``R`` = the top 11% of coordinates by readout share, ``S`` = the top 11% by S-lens token share, both outside
``n``: c = R∩S (the core), r = R∖S, s = S∖R, d = the rest, n = the norm coordinates (above-fair energy that the next
layer's input norm reads at under 1/1000 of it, or over 1% of the energy with the sink positions included). n is never
ablated. To *ablate* a set is to replace the residual's coordinates in it by their means, at every position but the
first (or at position 100 alone).

Parts (outputs in ``data/summary/sectors_<part>_<model>.json``):

- ``profile``: the per-coordinate measures and sectors at every layer, the held-out readout share of the top 11% against
  random coordinates and the best 11% of a random basis, the sectors' shares of energy and readout, the valleys of
  write shielding and the coordinate groups they cut, the core against the most persistent coordinates, and the write,
  norm and scale factors behind the top 11% (geometric means over it against the rest).
- ``basis``: inside the top 11% at three layers, coordinates against random rotations of them and against ICA sources
  (96, fitted on even documents): held-out kurtosis, the top 32 firings' modal token, whether the S-lens read names the
  firing token and the R-lens read the next one, persistence, and how many coordinates hold 90% of a direction; the
  readout's concentration in coordinates against a random basis and its own eigenbasis; per-neuron kurtosis of writes
  and gate rows against a rotation; how many coordinates rebuild a pole's write read to cosine 0.5.
- ``unions`` (Qwen3-8B): every union of c, r, s and d ablated at L4, L8, L14, L22 and L28, and findings §21's sectors in
  the eigenbasis (R-bright, S-bright outside it, doubly dark) the same way: ΔNLL and KL on the scored documents, at one
  position too for coordinate sets, random coordinate sets of each sector's size, greedy rollouts of §21's ten prompts.
- ``depth`` (Qwen3-8B): c, r+s+d and d ablated at every layer; greedy rollouts (64 tokens) of 30 prompts at seven layers.
- ``late`` (Qwen3-8B): the late shielded group (from ``profile``) and d at L20-L35: the R-lens read of the residual with
  only one part kept (the rest at its mean), ranked against the true next token; d's readout in its own eigenbasis;
  ablations; rollouts; the late group's ICA sources and d's top readout directions read through the R-lens.
- ``units``: every neuron's and head's shares of read and write on the five sectors (reads: the share of the activation
  energy a row draws, ``Σ_S (c_i σ_i)²``, from the sectors of the layer's input; writes: energy in the σ frame, and for n
  the raw share), by layer, against what a row with equal weight on every coordinate would put on each (``chance``); the
  units that put half or more on one sector, with their catalog labels; correlations with the catalog's label
  confidences and head scores.
"""
import argparse
import itertools
import json
import time
import warnings

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from scipy.stats import gaussian_kde, spearmanr

from neuron_lens import arch, metric
from neuron_lens.catalog import (
    MODEL,
    head_table,
    kept,
    kept_index,
    neuron_table,
    tokenizer,
    vocab,
)
from neuron_lens.corpus import act_file, lens_docs
from neuron_lens.neurons import directions, mlp_names
from neuron_lens.paths import SUMMARY
from neuron_lens.weights import tensors

warnings.filterwarnings("ignore")
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
T, FIRST, TPOS, TOPQ = 128, 16, 100, 0.11
NAMES = "crsdn"
STEM = arch.stem(MODEL)
UNIONS = ["".join(c) for n in (1, 2, 3) for c in itertools.combinations("crsd", n)]
PROMPTS = ["The capital of France is", "def fibonacci(n):\n", "Once upon a time, in a small village,",
           "Q: What is 17 + 25?\nA:", "中国的首都是", "Monday, Tuesday, Wednesday,",
           "The mitochondria is the powerhouse of the cell because"]
CHAT = ["Explain why the sky is blue in two sentences.", "Who are you?", "Write a haiku about autumn."]
PLAIN30 = PROMPTS + ["Translate to French: The cat is sleeping on the warm windowsill.\nFrench:",
                     "The Eiffel Tower was completed in the year", "SELECT name, email FROM users WHERE",
                     "To be, or not to be,", "The three primary colors are", "function add(a, b) {\n  return",
                     "In 1969, Neil Armstrong became the first person to",
                     "Water boils at 100 degrees Celsius at sea level, but on a high mountain it boils",
                     "My favorite recipe for pancakes starts with", "La capitale de l'Espagne est", "東京は日本の",
                     "Dear Professor Smith,\n\nI am writing to ask whether", '{"name": "Alice", "age": 30, "city":']
CHAT30 = CHAT + ["If I have 3 apples and eat one, how many are left? Answer briefly.", "What is the opposite of 'ancient'?",
                 "Give me three tips for sleeping better.", "Translate 'good morning' into Spanish, German and Japanese.",
                 "Write a Python one-liner that reverses a string.", "What year did World War II end?",
                 "Tell me a joke about cats."]
LOG = []
T0 = time.time()


def log(msg: str) -> None:
    line = f"[{time.time() - T0:6.0f}s] {msg}"
    print(line, flush=True)
    LOG.append(line)


def save(part: str, res: dict) -> None:
    (SUMMARY / f"sectors_{part}_{STEM}.json").write_text(json.dumps(res, ensure_ascii=False))
    (SUMMARY / f"sectors_{part}_{STEM}.log").write_text("\n".join(LOG) + "\n")


def load(part: str) -> dict:
    return json.loads((SUMMARY / f"sectors_{part}_{STEM}.json").read_text())


class Ctx:
    """The model's readout, embeddings, lenses, scales and documents; the model itself on first use."""

    def __init__(self, lenses=("J_out", "K_out")):
        self.nm = arch.names(MODEL)
        self.nL = arch.text_config(MODEL).num_hidden_layers
        self.gemma = arch.is_gemma4(MODEL)
        self.kid, self.kix, self.voc, self.tok = kept(MODEL), kept_index(MODEL), vocab(MODEL), tokenizer(MODEL)
        head = tensors(MODEL, [self.nm["unembed"], self.nm["norm"], self.nm["embed"]])
        wk = head[self.nm["unembed"]][self.kid].float()
        self.WU = ((wk - wk.mean(0)) * head[self.nm["norm"]].float()).to(DEV)
        self.G = self.WU.T @ self.WU
        we = head[self.nm["embed"]][self.kid].float()
        self.E = (we - we.mean(0)).to(DEV)
        self.d = self.G.shape[0]
        del wk, we, head
        self.lens = arch.lens_means(MODEL, *arch.fits(MODEL), parts=lenses) if lenses else {}
        self.sig = metric.scales(MODEL)
        docs = torch.tensor(lens_docs(act_file("dclm", MODEL), self.tok, 224, T, T, full=True))
        self.scored, self.pick, self.meand = docs[:64], docs[64:160], docs[160:224]
        self._lm = None

    def J(self, L):
        return self.lens["J_out"][L].to(DEV).float() if L < self.lens["J_out"].shape[0] else torch.eye(self.d, device=DEV)

    def K(self, L):
        return self.lens["K_out"][min(L, self.lens["K_out"].shape[0] - 1)].to(DEV).float()

    def lm(self):
        if self._lm is None:
            self._lm = arch.load_model(MODEL, DEV, eager=False)[0]
            mods = dict(self._lm.named_modules())
            self.layer_mod = [mods[arch.layout(MODEL)[L].prefix.rstrip(".")] for L in range(self.nL)]
        return self._lm

    def next_norm(self, L):
        name = mlp_names(MODEL, L + 1)["norm"] if L + 1 < self.nL else self.nm["norm"]
        return tensors(MODEL, [name])[name].float()

    def sig_out(self, L):
        return self.sig["in"][min(L + 1, self.sig["in"].shape[0] - 1)]


def residuals(C: Ctx, docs, layers):
    """Each layer's output at positions 1+ (``{L: [N, d]}`` bf16 on the CPU), the tokens and each row's document."""
    lm = C.lm()
    cap = {}
    hs = [C.layer_mod[L].register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach()))
          for L in layers]
    H = {L: [] for L in layers}
    with torch.inference_mode():
        for b in range(0, len(docs), 4):
            lm(input_ids=docs[b:b + 4].to(DEV), use_cache=False)
            for L in layers:
                H[L].append(cap[L][:, 1:].reshape(-1, C.d).to(torch.bfloat16).cpu())
    for h in hs:
        h.remove()
    return ({L: torch.cat(v) for L, v in H.items()}, docs[:, 1:].reshape(-1),
            torch.arange(len(docs)).repeat_interleave(docs.shape[1] - 1))


def not_sink(H):
    ok = None
    for h in H.values():
        n = h.float().norm(dim=1)
        o = n < 5 * n.median()
        ok = o if ok is None else ok & o
    return ok


def valley(x, peak_ratio=0.9):
    """The antimode of a 1-D sample (Gaussian KDE, Silverman bandwidth): (cut, valley/smaller peak), or None."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    kde = gaussian_kde(x, bw_method="silverman")
    g = np.linspace(np.quantile(x, 0.002), np.quantile(x, 0.998), 600)
    f = kde(g)
    mx = [i for i in range(1, len(g) - 1) if f[i] > f[i - 1] and f[i] >= f[i + 1]]
    if len(mx) < 2:
        return None
    a, b = sorted(sorted(mx, key=lambda i: -f[i])[:2])
    lo = a + int(np.argmin(f[a:b + 1]))
    r = f[lo] / min(f[a], f[b])
    return (float(g[lo]), float(r)) if r < peak_ratio else None


def kurt(a, dim=0):
    a = a - a.mean(dim, keepdim=True)
    return a.pow(4).mean(dim) / a.pow(2).mean(dim).pow(2)


def modal(ix):
    return torch.tensor([torch.unique(c, return_counts=True)[1].max().item() / len(c) for c in ix.T])


def hits(C, read, ids, valid=None):
    """Share of ``ids`` [k, m] among the top 64 kept tokens of each column of ``read`` [V, m] (``valid``: which count)."""
    top = read.topk(64, dim=0).indices.cpu()
    k = C.kix[ids]
    ok = (k >= 0) if valid is None else (k >= 0) & valid
    return ((k[:, None, :] == top[None]).any(1) & ok).float().mean(0)


def pieces(C, ids):
    return " ".join(repr(C.voc[i])[1:-1] for i in ids)


def kept_pieces(C, ix):
    return pieces(C, C.kid[ix.cpu()].tolist())


def write_energy(C, L):
    """``[d]``: the layer's MLP and attention write energy per coordinate in the residual's σ frame."""
    w = directions(MODEL, L)["write"].float()
    p = arch.layout(MODEL)[L].prefix
    names = [p + "self_attn.o_proj.weight"] + ([p + "post_attention_layernorm.weight"] if C.gemma else [])
    t = tensors(MODEL, names)
    Wo = t[names[0]].float()
    if C.gemma:
        Wo = t[names[1]].float()[:, None] * Wo
    return w.pow(2).sum(1) / C.sig_out(L).pow(2) + Wo.pow(2).sum(1) / C.sig["mid"][L].pow(2)


def chance(C, L, sect):
    """What a row with equal weight on every coordinate would put on each sector, the baseline for ``units``: a read, the
    share of the norm's output energy, ``(γ σ)²``, on the sectors of the layer's input; a write, the share of the σ
    frame, ``(γ_post / σ)²`` (``γ_post`` Gemma's post-norm, else 1), on the sectors of where it writes. For the MLP and
    the heads."""
    p = arch.layout(MODEL)[L].prefix
    nm = mlp_names(MODEL, L)
    post_attn = p + "post_attention_layernorm.weight" if "post" in nm else None
    want = [nm["norm"], p + "input_layernorm.weight"] + ([nm["post"], post_attn] if post_attn else [])
    w = {k: v.float() for k, v in tensors(MODEL, want).items()}
    one = torch.ones(C.d)
    lr, lw = sect[max(0, L - 1)], sect[L]

    def sh(e, lab):
        return [(e[lab == k].sum() / e.sum()).item() for k in range(5)]
    return {"mlp_read": sh((w[nm["norm"]] * C.sig["mid"][L]) ** 2, lr),
            "mlp_write": sh((w.get(nm.get("post"), one) / C.sig_out(L)) ** 2, lw),
            "head_read": sh((w[p + "input_layernorm.weight"] * C.sig["in"][L]) ** 2, lr),
            "head_write": sh((w.get(post_attn, one) / C.sig["mid"][L]) ** 2, lw)}


def norm_gap(ratio, big):
    """Among the coordinates with above-fair energy: log10 of the next norm's read share over the energy share, the
    largest among the norm coordinates (under 1e-3) and the smallest among the others."""
    lr = torch.log10(ratio.clamp_min(1e-30))
    ns = big & (ratio < 1e-3)
    return {"norm_max_log10": lr[ns].max().item() if ns.any() else None, "others_min_log10": lr[big & ~ns].min().item(),
            "n_norm": int(ns.sum())}


def sectors_of(prof):
    return torch.tensor([[NAMES.index(ch) for ch in s] for s in prof["sectors"]])


# ------------------------------------------------------------------------------------------------------------ profile

def profile() -> None:
    C = Ctx()
    layers = list(range(C.nL))
    H, tok, doc = residuals(C, C.pick, layers)
    ok = not_sink(H)
    log(f"{len(tok)} positions, {int((~ok).sum())} at sinks")
    t, dc = tok[ok], doc[ok]
    ev = dc % 2 == 0
    kk = int(TOPQ * C.d)
    res = {"model": MODEL, "layers": [], "sectors": [], "groups": {}}
    Z, shield = {}, {}
    for L in layers:
        hall = H[L].float()
        h = hall[ok]
        mu, sd = h.mean(0), h.std(0).clamp_min(1e-8)
        Z[L] = ((h - mu) / sd).to(torch.bfloat16)
        hc = (h - mu).to(DEV)
        J = C.J(L)
        A = J.T @ C.G @ J
        Ah = hc @ A
        q = (hc * Ah)
        pR = (q.sum(0) / q.sum()).cpu()
        evd = ev.to(DEV)
        pe = (q[evd].sum(0) / q[evd].sum()).cpu()
        po = (q[~evd].sum(0) / q[~evd].sum()).cpu()
        r = C.kix[t]
        m = r >= 0
        foot = C.E[r[m].to(DEV)] @ C.K(L).T
        y = hc[m.to(DEV)]
        tokS = (((foot - foot.mean(0)) * (y - y.mean(0))).mean(0) / (foot.std(0) * y.std(0)).clamp_min(1e-12)).pow(2).cpu()
        del foot, y
        # lens-free token share: each token's mean on even documents predicting odd ones
        te, to = t[ev], t[~ev]
        uniq, inv, cnt = torch.unique(te, return_inverse=True, return_counts=True)
        means = torch.zeros(len(uniq), C.d).index_add_(0, inv, h[ev] - mu) / cnt[:, None]
        lut = torch.full((int(t.max()) + 1,), -1, dtype=torch.long)
        lut[uniq[cnt >= 3]] = torch.arange(len(uniq))[cnt >= 3]
        j = lut[to]
        yo = h[~ev][j >= 0] - mu
        eta = 1 - (yo - means[j[j >= 0]]).pow(2).sum(0) / (yo - yo.mean(0)).pow(2).sum(0)
        e2_all, e2 = hall.pow(2).mean(0), h.pow(2).mean(0)
        n_sh = e2 / e2.sum()
        rd = C.next_norm(L).pow(2) * e2
        rd = rd / rd.sum()
        nset = ((n_sh > 1 / C.d) & (rd / n_sh < 1e-3)) | (e2_all / e2_all.sum() > 0.01)
        okc = ~nset
        def top(x, okc=okc):
            return torch.zeros(C.d, dtype=torch.bool).index_fill_(0, x.masked_fill(~okc, -1e9).topk(kk).indices, True)

        Rm, Sm = top(pR), top(tokS)
        lab = torch.full((C.d,), 3)
        lab[Rm & Sm], lab[Rm & ~Sm], lab[Sm & ~Rm], lab[nset] = 0, 1, 2, 4
        res["sectors"].append("".join(NAMES[i] for i in lab.tolist()))
        we_ = write_energy(C, L)
        shield[L] = -torch.log(we_.clamp_min(1e-30) / we_.mean())
        e = {"layer": L, "n_coords": nset.nonzero().squeeze(1).tolist(),
             "held_out_top": po[pe.topk(kk).indices].sum().item(),
             "random_coords": po[torch.randperm(C.d)[:kk]].sum().item(),
             "norm_gap": norm_gap(rd / n_sh, n_sh > 1 / C.d)}
        for k in range(5):
            mk = lab == k
            e[NAMES[k]] = {"count": int(mk.sum()), "energy": n_sh[mk].sum().item(),
                           "energy_all": (e2_all[mk].sum() / e2_all.sum()).item(), "readout": pR[mk].sum().item(),
                           "token_S": tokS[mk].mean().item() if mk.any() else 0.0,
                           "token_free": eta[mk].median().item() if mk.any() else 0.0}
        if L % 4 == 0:  # the best 11% of a random basis of the standardised residual
            Q = torch.linalg.qr(torch.randn(C.d, C.d, device=DEV))[0]
            sdd = sd.to(DEV)
            hz, Az = hc / sdd, Ah * sdd
            qe = ((hz[evd] @ Q) * (Az[evd] @ Q)).sum(0) / q[evd].sum()
            qo = ((hz[~evd] @ Q) * (Az[~evd] @ Q)).sum(0) / q[~evd].sum()
            e["random_basis_best"] = qo[qe.topk(kk).indices].sum().item()
            del Q, hz, Az
        # the write, norm and scale factors behind the top 11% by readout share (geometric means, top / rest)
        so = C.sig_out(L)
        p = arch.layout(MODEL)[L].prefix
        mn = mlp_names(MODEL, L)
        want = [mn["down"], p + "self_attn.o_proj.weight"] + ([mn["post"], p + "post_attention_layernorm.weight"] if C.gemma else [])
        w = tensors(MODEL, want)
        gm = lambda x, mk: (x.clamp_min(1e-30).log()[mk].mean() - x.clamp_min(1e-30).log()[~mk].mean()).exp().item()
        e["top_factors"] = {"sigma2": gm(so.pow(2), Rm), "mlp_rows2": gm(w[mn["down"]].float().pow(2).sum(1), Rm),
                            "attn_rows2": gm(w[p + "self_attn.o_proj.weight"].float().pow(2).sum(1), Rm),
                            "post_gamma2": gm(w[mn["post"]].float().pow(2), Rm) if C.gemma else None,
                            "final_gamma2": gm(C.next_norm(C.nL - 1).pow(2), Rm),
                            "write_rel": gm(torch.exp(-shield[L]), Rm)}
        e["_pR"], e["_tokS"], e["_eta"] = pR, tokS, eta
        res["layers"].append(e)
        del hc, Ah, q
        log(f"L{L}: " + " ".join(f"{k} {e[k]['count']}" for k in NAMES) + f" | top 11% {e['held_out_top']:.2f}"
            + (f" (random basis {e['random_basis_best']:.2f})" if "random_basis_best" in e else ""))
    # persistence, shielding valleys, groups, and the core against the most persistent coordinates
    sect = sectors_of(res)
    for g in (2, 4):
        for L in layers:
            res["layers"][L][f"persist{g}"] = ((Z[L].float() * Z[L + g].float()).mean(0)).tolist() if L + g < C.nL else None
    runs = []
    for L in layers:  # layers whose shielding has a valley, in runs that skip at most one layer
        v = valley(shield[L].numpy())
        res["layers"][L]["shield_valley"] = v
        res["layers"][L]["shielded"] = int((shield[L] > v[0]).sum()) if v else 0
        if v:
            if runs and L - runs[-1][-1] <= 2:
                runs[-1].append(L)
            else:
                runs.append([L])
    for rr in [r for r in runs if len(r) >= 3]:
        votes = torch.stack([shield[L] > res["layers"][L]["shield_valley"][0] for L in rr]).float().mean(0)
        grp = votes >= 0.5
        res["groups"][f"L{rr[0]}-L{rr[-1]}"] = {"layers": rr, "coords": grp.nonzero().squeeze(1).tolist(),
                                                "sectors_at": {str(L): {NAMES[k]: int(((sect[L] == k) & grp).sum()) for k in range(5)}
                                                               for L in range(rr[0], C.nL, max(1, C.nL // 12))}}
    for L in layers:
        e = res["layers"][L]
        P = torch.tensor(e["persist4"]) if e["persist4"] is not None else None
        nset = sect[L] == 4
        def top(x, nset=nset):
            return torch.zeros(C.d, dtype=torch.bool).index_fill_(0, x.masked_fill(nset, -1e9).topk(kk).indices, True)

        R11, S11 = top(e["_pR"]), top(e["_tokS"])
        core = R11 & S11
        row = {}
        for name, mk in (("R11", R11), ("S11", S11), ("core", core)):
            row[name] = {"n": int(mk.sum()), "readout": e["_pR"][mk].sum().item(), "token_free": e["_eta"][mk].median().item(),
                         "persist4": P[mk].median().item() if P is not None else None}
        if P is not None:
            mp = torch.zeros(C.d, dtype=torch.bool).index_fill_(0, P.masked_fill(nset, -9).topk(int(core.sum())).indices, True)
            row["core_vs_most_persistent_jaccard"] = (core & mp).sum().item() / (core | mp).sum().item()
        e["joint"] = row
        for k in ("_pR", "_tokS", "_eta"):
            del e[k]
        e["persist4_by_sector"] = {NAMES[k]: torch.tensor(e["persist4"])[sect[L] == k].median().item() for k in range(4)} if e["persist4"] else None
        del e["persist4"]
        if e["persist2"] is not None:
            e["persist2_by_sector"] = {NAMES[k]: torch.tensor(e["persist2"])[sect[L] == k].median().item() for k in range(4)}
        del e["persist2"]
    log("groups: " + "; ".join(f"{k}: {len(v['coords'])}" for k, v in res["groups"].items()))
    save("profile", res)


# ------------------------------------------------------------------------------------------------------------ basis

def basis(layers) -> None:
    from sklearn.decomposition import FastICA
    C = Ctx()
    prof = load("profile")
    sect = sectors_of(prof)
    want = sorted(set(layers) | {L + 4 for L in layers})
    H, tok, doc = residuals(C, C.pick, want)
    ok = not_sink(H)
    t2 = torch.stack([tok, torch.cat([tok[1:], tok[:1]])], 1)[ok]  # current and next token (the next is wrong at a
    nxt_ok = torch.cat([doc[1:] == doc[:-1], torch.tensor([False])])[ok]  # document's end: masked below)
    dc = doc[ok]
    ev = dc % 2 == 0
    kk = int(TOPQ * C.d)
    res = {"model": MODEL, "layers": {}}
    for L in layers:
        h = H[L].float()[ok]
        mu, sd = h.mean(0), h.std(0).clamp_min(1e-8)
        z = (h - mu) / sd
        h8 = H[L + 4].float()[ok]
        z8 = (h8 - h8.mean(0)) / h8.std(0).clamp_min(1e-8)
        J, K = C.J(L), C.K(L)
        A = J.T @ C.G @ J
        he = (h[ev] - mu).to(DEV)
        Ae = he @ A
        pe = ((he * Ae).sum(0) / (he * Ae).sum()).cpu()
        Cs = pe.masked_fill(sect[L] == 4, -9).topk(kk).indices
        out = {}
        # readout concentration inside C: coordinates, a random basis, the eigenbasis of its share matrix
        hz = [(z[ev])[:, Cs], (z[~ev])[:, Cs]]
        Az = [(((h[ev] - mu).to(DEV) @ A).cpu() * sd)[:, Cs], (((h[~ev] - mu).to(DEV) @ A).cpu() * sd)[:, Cs]]
        k10 = kk // 10

        def conc(Q, hz=hz, Az=Az, k10=k10):
            se = ((hz[0] @ Q) * (Az[0] @ Q)).sum(0)
            so_ = ((hz[1] @ Q) * (Az[1] @ Q)).sum(0)
            return (so_[se.topk(k10).indices].sum() / so_.sum()).item()

        M = hz[0].T @ Az[0]
        M = (M + M.T) / 2
        c_, r_, e_ = conc(torch.eye(kk)), conc(torch.linalg.qr(torch.randn(kk, kk))[0]), conc(torch.linalg.eigh(M)[1])
        out["readout_top10pct"] = {"coords": c_, "random": r_, "eigen": e_, "index": (c_ - r_) / (e_ - r_)}
        # directions: coordinates, rotations of them, ICA sources (held out)
        ica = FastICA(n_components=96, whiten="unit-variance", max_iter=3000, tol=1e-4, random_state=0).fit(hz[0].numpy().astype(np.float64))
        Mz, Wz = torch.tensor(ica.mixing_).float(), torch.tensor(ica.components_).float().T
        Q = torch.linalg.qr(torch.randn(kk, 96))[0]
        I = torch.zeros(kk, 96)
        I[torch.arange(96), torch.arange(96)] = 1
        cur, nxt = t2[~ev][:, 0], t2[~ev][:, 1]
        nok = nxt_ok[~ev]
        for name, (Vc, Wc) in {"coordinates": (I, I), "rotations": (Q, Q), "ica": (Mz, Wz)}.items():
            a = z[~ev][:, Cs] @ Wc
            sg = torch.sign(((a - a.mean(0)) ** 3).mean(0))
            sg[sg == 0] = 1
            a, Vc, Wc = a * sg, Vc * sg, Wc * sg
            a8 = z8[~ev][:, Cs] @ Wc
            tp = a.topk(32, dim=0).indices
            V = torch.zeros(C.d, 96)
            V[Cs] = sd[Cs, None] * Vc
            Cv = torch.zeros(C.d, 96)
            Cv[Cs] = Wc / sd[Cs, None]
            Rr, Sr = C.WU @ (J @ V.to(DEV)), C.E @ (K.T @ Cv.to(DEV))
            w = Vc.pow(2) / Vc.pow(2).sum(0, keepdim=True)
            n90 = ((w.sort(0, descending=True).values.cumsum(0) < 0.9).sum(0) + 1).float()
            q = lambda x: [x.median().item(), x.quantile(0.9).item()]
            k_ = kurt(a)
            out[name] = {"kurtosis": q(k_), "modal_token": q(modal(cur[tp])), "S_hits_current": q(hits(C, Sr, cur[tp])),
                         "R_hits_next": q(hits(C, Rr, nxt[tp], nok[tp])), "persist4": torch.nn.functional.cosine_similarity(
                             a - a.mean(0), a8 - a8.mean(0), dim=0).median().item(), "coords_for_90pct": n90.median().item()}
            if name == "ica":
                out["ica_examples"] = [{"kurtosis": k_[j].item(), "fires": pieces(C, cur[tp[:8, j]].tolist()),
                                        "R": kept_pieces(C, Rr[:, j].topk(6).indices), "S": kept_pieces(C, Sr[:, j].topk(6).indices)}
                                       for j in k_.argsort(descending=True)[:12].tolist()]
        # neurons: per-neuron kurtosis of writes and gate rows (each coordinate standardised across neurons), vs a rotation
        D = directions(MODEL, L)
        Qd = torch.linalg.qr(torch.randn(C.d, C.d))[0]
        for key, X in (("write", D["write"].T.float()), ("gate", D["gate"].float())):
            Zx = X / X.std(0, keepdim=True).clamp_min(1e-12)
            out[f"neuron_{key}_kurtosis"] = {"coords": kurt(Zx, 1).median().item(), "rotated": kurt(Zx @ Qd, 1).median().item()}
        # how many coordinates rebuild a pole's write read to cosine 0.5 (24 random poles), against a random basis
        s_out = metric.s2(MODEL, "out", L, DEV).sqrt()
        Qm = torch.linalg.qr(torch.randn(C.d, C.d, device=DEV))[0]

        def k_for(v, Ag, at=0.5):
            size = (v.pow(2) * Ag.diagonal()).clamp_min(0).sqrt()
            o = size.argsort(descending=True)
            tot = v @ Ag @ v
            for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, C.d):
                vk = torch.zeros_like(v)
                vk[o[:k]] = v[o[:k]]
                if (vk @ Ag @ v) / (vk @ Ag @ vk).sqrt() / tot.sqrt() >= at:
                    return k
            return C.d

        Wd = D["write"].float().to(DEV)
        At = Qm.T @ (A * s_out[:, None] * s_out[None, :]) @ Qm
        ks, kr = [], []
        for n in torch.randint(0, Wd.shape[1], (24,)).tolist():
            v = Wd[:, n]
            ks.append(k_for(v, A))
            kr.append(k_for(Qm.T @ (v / s_out), At))
        out["write_read_coords_for_cos05"] = {"coords": float(np.median(ks)), "random_basis": float(np.median(kr))}
        res["layers"][str(L)] = out
        log(f"L{L}: index {out['readout_top10pct']['index']:.2f}; kurtosis coords {out['coordinates']['kurtosis'][0]:.2f} "
            f"rot {out['rotations']['kurtosis'][0]:.2f} ica {out['ica']['kurtosis'][0]:.2f}; S hits ica {out['ica']['S_hits_current'][1]:.2f}")
        save("basis", res)


# ------------------------------------------------------------------------------------------------------------ ablations

class Ablator:
    """Mean ablation of a set of the residual's coordinates (or a subspace) at one layer's output, scored on the scored
    documents against the clean model, and greedy rollouts."""

    def __init__(self, C: Ctx, layers, prompts_plain, prompts_chat, new_tokens):
        self.C = C
        lm = C.lm()
        self.state = {"L": None, "fn": None, "first": None, "pos": None}
        for L in layers:
            C.layer_mod[L].register_forward_hook(self._hook(L))
        cap = {}
        hs = [C.layer_mod[L].register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach()))
              for L in layers]
        msum = {L: torch.zeros(C.d, dtype=torch.float64) for L in layers}
        cnt = 0
        with torch.inference_mode():
            for b in range(0, len(C.meand), 4):
                lm(input_ids=C.meand[b:b + 4].to(DEV), use_cache=False)
                for L in layers:
                    msum[L] += cap[L][:, 1:].float().sum((0, 1)).cpu().double()
                cnt += cap[layers[0]][:, 1:].shape[0] * cap[layers[0]][:, 1:].shape[1]
        for h in hs:
            h.remove()
        self.means = {L: (msum[L] / cnt).float().to(DEV) for L in layers}
        tok = C.tok
        ids = [tok(p).input_ids for p in prompts_plain]
        for c in prompts_chat:
            x = tok.apply_chat_template([{"role": "user", "content": c}], add_generation_prompt=True, enable_thinking=False)
            ids.append(x if isinstance(x, list) else x["input_ids"])
        self.Lmax = max(len(x) for x in ids)
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.inp = torch.tensor([[self.pad] * (self.Lmax - len(x)) + x for x in ids])
        self.att = torch.tensor([[0] * (self.Lmax - len(x)) + [1] * len(x) for x in ids])
        self.first = torch.tensor([self.Lmax - len(x) for x in ids])
        self.stops = {tok.convert_tokens_to_ids(s) for s in ("<|im_end|>", "<|endoftext|>", "<end_of_turn>", "<eos>")} - {None, tok.unk_token_id}
        self.new = new_tokens
        self.clean_nll, self.clean_lp = self._score()
        self.clean_t_nll, self.clean_t_lp = self._score(pos=TPOS)

    def _hook(self, L):
        def h(mod, args, out):
            st = self.state
            if st["L"] != L or st["fn"] is None:
                return None
            o = out[0] if isinstance(out, tuple) else out
            x = o.float()
            new = st["fn"](x)
            B, Tn, _ = x.shape
            keep = torch.zeros(B, Tn, 1, dtype=torch.bool, device=x.device)
            if st["pos"] is not None:
                keep[:] = True
                keep[:, st["pos"]] = False
            elif Tn > 1:
                first = st["first"] if st["first"] is not None else torch.zeros(B, dtype=torch.long)
                keep[(torch.arange(Tn)[None, :] <= first[:, None]).to(x.device)] = True
            new = torch.where(keep, x, new).to(o.dtype)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return h

    @torch.no_grad()
    def _score(self, pos=None, ref=None):
        nll, lps, kls = [], [], []
        for i, b in enumerate(range(0, len(self.C.scored), 8)):
            x = self.C.scored[b:b + 8].to(DEV)
            lp = F.log_softmax(self.C.lm()(input_ids=x, use_cache=False).logits.float(), -1)
            sl = slice(FIRST - 1, T - 2) if pos is None else slice(pos, pos + 1)
            lp = lp[:, sl]
            nll.append((-lp.gather(2, x[:, 1:][:, sl][..., None])[..., 0]).cpu())
            if ref is None:
                lps.append(lp.half().cpu())
            else:
                rr = ref[i].to(DEV).float()
                kls.append((rr.exp() * (rr - lp)).sum(-1).cpu())
        return torch.cat(nll), (lps if ref is None else torch.cat(kls).mean().item())

    def score(self, L, fn, one_position=False):
        self.state.update(L=L, fn=fn, pos=None)
        nll, kl = self._score(ref=self.clean_lp)
        row = {"dnll": (nll.mean() - self.clean_nll.mean()).item(), "kl": kl}
        if one_position:
            self.state["pos"] = TPOS
            n1, k1 = self._score(pos=TPOS, ref=self.clean_t_lp)
            row.update({"dnll_pos": (n1.mean() - self.clean_t_nll.mean()).item(), "kl_pos": k1})
        self.state.update(L=None, fn=None, pos=None)
        return row

    @torch.no_grad()
    def rollouts(self, L=None, fn=None):
        self.state.update(L=L, fn=fn, pos=None)
        outs = []
        for b in range(0, len(self.inp), 10):
            self.state["first"] = self.first[b:b + 10]
            o = self.C.lm().generate(self.inp[b:b + 10].to(DEV), attention_mask=self.att[b:b + 10].to(DEV),
                                     max_new_tokens=self.new, do_sample=False, pad_token_id=self.pad)
            for row in o[:, self.Lmax:].tolist():
                cut = next((i for i, t in enumerate(row) if t in self.stops), len(row))
                outs.append(self.C.tok.decode(row[:cut]).replace("�", "▯"))
        self.state.update(L=None, fn=None, first=None)
        return outs

    def mask_fn(self, L, mask):
        mk, mean = mask.to(DEV), self.means[L]
        return lambda x: torch.where(mk, mean, x)


def unions(layers) -> None:
    C = Ctx()
    prof = load("profile")
    sect = sectors_of(prof)
    ab = Ablator(C, layers, PROMPTS, CHAT, 48)
    Gd, GEd = C.G.cpu().double(), (C.E.T @ C.E).cpu().double()
    res = {"model": MODEL, "layers": layers, "prompts": PROMPTS + [f"[chat] {c}" for c in CHAT],
           "clean_nll": ab.clean_nll.mean().item(), "clean_gen": ab.rollouts(), "per_layer": {}}

    def top(Gr, s, X):
        gain, V = torch.linalg.eigh(s[:, None] * Gr * s[None, :])
        gain, V = gain.flip(0).clamp_min(0), V.flip(1)
        e = (X @ V).pow(2) * gain
        return V[:, :int(((e.cumsum(1) / e.sum(1, keepdim=True)).mean(0) < 0.9).sum()) + 1]

    for L in layers:
        coord = {NAMES[k]: sect[L] == k for k in range(4)}
        s = metric.s2(MODEL, "out", L, "cpu").sqrt().double()
        p = arch.layout(MODEL)[L].prefix
        X = tensors(MODEL, [p + "mlp.down_proj.weight"])[p + "mlp.down_proj.weight"].double().T / s
        X = X / X.norm(dim=1, keepdim=True)
        Jd, Kd = C.lens["J_out"][L].double(), C.lens["K_out"][L].double()
        V = top(Jd.T @ Gd @ Jd, s, X)
        VS = top(Kd @ GEd @ Kd.T / s.pow(2)[:, None] / s.pow(2)[None, :], s, X)
        Qs, sv, _ = torch.linalg.svd(VS - V @ (V.T @ VS), full_matrices=False)
        SO = Qs[:, sv > 0.1]
        cosines = torch.linalg.svdvals(V.T @ VS)
        U = torch.cat([V, SO], 1)
        DE = torch.linalg.qr(torch.cat([U, torch.randn(C.d, C.d - U.shape[1], dtype=torch.float64)], 1))[0][:, U.shape[1]:]
        eig = {"r": V.float().to(DEV), "s": SO.float().to(DEV), "d": DE.float().to(DEV)}
        ax = {k: B.pow(2).sum(1).cpu() for k, B in eig.items()}
        out = {"sizes_coord": {k: int(v.sum()) for k, v in coord.items()}, "sizes_eig": {k: int(v.shape[1]) for k, v in eig.items()},
               "n": int((sect[L] == 4).sum()), "rs_cos": {"max": cosines.max().item(), "median": cosines.median().item()},
               "axes_in_eig": {a: {b: ax[b][coord[a]].mean().item() for b in eig} for a in coord},
               "eig_dims_on_coord": {b: {a: (ax[b][coord[a]].sum() / eig[b].shape[1]).item() for a in coord} for b in eig},
               "abl": {}, "gen": {}}
        sf = s.float().to(DEV)
        for u in UNIONS:
            mk = torch.zeros(C.d, dtype=torch.bool)
            for ch in u:
                mk |= coord[ch]
            fn = ab.mask_fn(L, mk)
            out["abl"][f"coord:{u}"] = ab.score(L, fn, one_position=True)
            out["gen"][f"coord:{u}"] = ab.rollouts(L, fn)
        for u in [x for x in UNIONS if "c" not in x]:
            B = torch.cat([eig[ch] for ch in u], 1)
            mean = ab.means[L]
            fn = lambda x, B=B, mean=mean, sf=sf: x - ((((x - mean) / sf) @ B) @ B.T) * sf
            out["abl"][f"eig:{u}"] = ab.score(L, fn)
            out["gen"][f"eig:{u}"] = ab.rollouts(L, fn)
        okix = (sect[L] != 4).nonzero().squeeze(1)
        for ch in "crsd":
            mk = torch.zeros(C.d, dtype=torch.bool)
            mk[okix[torch.randperm(len(okix))[:int(coord[ch].sum())]]] = True
            out["abl"][f"rand:{ch}"] = ab.score(L, ab.mask_fn(L, mk))
        res["per_layer"][str(L)] = out
        log(f"L{L}: " + " ".join(f"{u} {out['abl'][f'coord:{u}']['dnll']:+.2f}" for u in ("c", "r", "s", "d", "rsd", "crs"))
            + " | eig " + " ".join(f"{u} {out['abl'][f'eig:{u}']['dnll']:+.2f}" for u in ("r", "s", "d", "sd")))
        save("unions", res)


def depth(gen_layers) -> None:
    C = Ctx(lenses=None)
    sect = sectors_of(load("profile"))
    layers = list(range(C.nL))
    ab = Ablator(C, layers, PLAIN30, CHAT30, 64)
    res = {"model": MODEL, "gen_layers": gen_layers, "prompts": PLAIN30 + [f"[chat] {c}" for c in CHAT30],
           "clean_nll": ab.clean_nll.mean().item(), "abl": {}, "gen": {"clean": ab.rollouts()}}
    for L in layers:
        for name in ("c", "rsd", "d"):
            mk = torch.zeros(C.d, dtype=torch.bool)
            for ch in name:
                mk |= sect[L] == NAMES.index(ch)
            fn = ab.mask_fn(L, mk)
            res["abl"][f"{name}@{L}"] = ab.score(L, fn, one_position=True)
            if L in gen_layers:
                res["gen"][f"{name}@{L}"] = ab.rollouts(L, fn)
        log(f"L{L}: " + " ".join(f"{k} {res['abl'][f'{k}@{L}']['dnll']:+.2f}" for k in ("c", "rsd", "d")))
        save("depth", res)


# ------------------------------------------------------------------------------------------------------------ late

def late(layers, read_layer) -> None:
    from sklearn.decomposition import FastICA
    C = Ctx(lenses=("J_out",))
    prof = load("profile")
    sect = sectors_of(prof)
    late_key = max(prof["groups"], key=lambda k: int(k.split("-")[0][1:]))
    LATE = torch.zeros(C.d, dtype=torch.bool)
    LATE[prof["groups"][late_key]["coords"]] = True
    early = [k for k in prof["groups"] if k != late_key]
    EARLY = torch.zeros(C.d, dtype=torch.bool)
    for k in early:
        EARLY[prof["groups"][k]["coords"]] = True
    ab = Ablator(C, sorted(set(layers) | {read_layer}), PLAIN30, CHAT30, 64)
    res = {"model": MODEL, "late_group": late_key, "late_n": int(LATE.sum()), "late_new_n": int((LATE & ~EARLY).sum()),
           "context": {}, "rotated": {}, "abl": {}, "gen": {}, "reads": {}}
    H, _, _ = residuals(C, C.scored, sorted(set(layers) | {read_layer}))
    nxt = C.scored[:, FIRST:T - 1].reshape(-1)
    sl = torch.zeros(C.scored.shape[0], T - 1, dtype=torch.bool)
    sl[:, FIRST - 2:T - 3] = True  # rows are positions 1+: the positions FIRST-1..T-3, which predict tokens FIRST..T-2
    sl = sl.reshape(-1)
    for L in layers:
        h = H[L][sl].float()
        J = C.J(L)
        dm, cm, crs = sect[L] == 3, sect[L] == 0, sect[L] <= 2
        okix = (sect[L] != 4).nonzero().squeeze(1)
        rnd = torch.zeros(C.d, dtype=torch.bool)
        rnd[okix[torch.randperm(len(okix))[:int(LATE.sum())]]] = True
        row = {}
        for name, mk in {"full": torch.ones(C.d, dtype=torch.bool), "late group": LATE, "d": dm, "c": cm, "c+r+s": crs, "random": rnd}.items():
            ranks = []
            mkd = mk.to(DEV)
            for i in range(0, len(h), 1024):
                x = torch.where(mkd, h[i:i + 1024].to(DEV), ab.means[L])
                lg = (x @ J.T) @ C.WU.T
                k = C.kix[nxt[i:i + 1024]]
                tv = lg.gather(1, k.clamp_min(0).to(DEV)[:, None])
                rk = (lg > tv).sum(1).float().cpu()
                rk[k < 0] = float("nan")
                ranks.append(rk)
            rk = torch.cat(ranks)
            rk = rk[~rk.isnan()]
            row[name] = {"median_rank": rk.median().item(), "top10": (rk < 10).float().mean().item()}
        res["context"][str(L)] = row
        A = J.T @ C.G @ J
        hc = (h.to(DEV) - ab.means[L])
        half = len(hc) // 2
        di = dm.nonzero().squeeze(1).to(DEV)

        def dshare(x, B=None, A=A, di=di):
            Ax = x @ A
            tot = (x * Ax).sum()
            xd, Axd = x[:, di], Ax[:, di]
            return ((xd * Axd).sum(0) if B is None else ((xd @ B) * (Axd @ B)).sum(0)) / tot

        M = hc[:half][:, di].T @ (hc[:half] @ A)[:, di]
        B = torch.linalg.eigh(((M + M.T) / 2).cpu())[1].flip(1).to(DEV)
        se, so_ = dshare(hc[:half], B), dshare(hc[half:], B)
        oe = se.argsort(descending=True)
        cs, oc = dshare(hc[half:]), dshare(hc[:half]).argsort(descending=True)
        res["rotated"][str(L)] = {"total": cs.sum().item(), **{str(k): {"eigen": so_[oe[:k]].sum().item(), "coords": cs[oc[:k]].sum().item()}
                                                             for k in (16, 64, 256)}}
        drows = torch.zeros(C.d, dtype=torch.bool)
        dl = dm.nonzero().squeeze(1)
        drows[dl[torch.randperm(len(dl))[:int(LATE.sum())]]] = True
        for name, mk in (("late group", LATE & (sect[L] != 4)), ("late group, new members", LATE & ~EARLY & (sect[L] != 4)),
                         ("d", dm), ("random in d", drows), ("all but the late group", (sect[L] != 4) & ~LATE)):
            res["abl"][f"{name}@{L}"] = ab.score(L, ab.mask_fn(L, mk), one_position=True) | {"n": int(mk.sum())}
        if L in (28, 34):
            for name, mk in (("late group", LATE & (sect[L] != 4)), ("all but the late group", (sect[L] != 4) & ~LATE)):
                res["gen"][f"{name}@{L}"] = ab.rollouts(L, ab.mask_fn(L, mk))
        log(f"L{L}: ranks " + " ".join(f"{k} {v['median_rank']:.0f}" for k, v in row.items())
            + " | abl " + " ".join(f"{k} {res['abl'][f'{k}@{L}']['dnll']:+.2f}" for k in ("late group", "d", "random in d")))
        save("late", res)
    # what the late group and d's late readout read as
    L = read_layer
    Hp, tp, dp = residuals(C, C.pick, [L])
    okp = not_sink(Hp)
    h = Hp[L].float()[okp]
    t2 = torch.stack([tp, torch.cat([tp[1:], tp[:1]])], 1)[okp]
    nvalid = torch.cat([dp[1:] == dp[:-1], torch.tensor([False])])[okp]
    ev = dp[okp] % 2 == 0
    mu, sd = h.mean(0), h.std(0).clamp_min(1e-8)
    J = C.J(L)
    A = J.T @ C.G @ J
    cur, nx, nv = t2[~ev][:, 0], t2[~ev][:, 1], nvalid[~ev]

    def show(a, v):
        sk = torch.sign(((a - a.mean()) ** 3).mean())
        sk = sk if sk != 0 else torch.tensor(1.0)
        a, v = a * sk, v * sk
        tpk = a.topk(24).indices
        Rr = C.WU @ (J @ v.to(DEV))
        return {"kurtosis": kurt(a[:, None])[0].item(), "R_hits_next": hits(C, Rr[:, None], nx[tpk][:, None], nv[tpk][:, None])[0].item(),
                "R+": kept_pieces(C, Rr.topk(8).indices), "R-": kept_pieces(C, (-Rr).topk(8).indices),
                "fires": pieces(C, cur[tpk[:10]].tolist()), "next": pieces(C, nx[tpk[:10]].tolist())}

    ci = (LATE & (sect[L] != 4)).nonzero().squeeze(1)
    z = (h - mu) / sd
    ica = FastICA(n_components=48, whiten="unit-variance", max_iter=3000, tol=1e-4, random_state=0).fit(z[ev][:, ci].numpy().astype(np.float64))
    src = torch.tensor(ica.transform(z[~ev][:, ci].numpy().astype(np.float64))).float()
    mix = torch.tensor(ica.mixing_).float()
    kz = kurt(src)
    reads = []
    for j in kz.argsort(descending=True)[:12].tolist():
        v = torch.zeros(C.d)
        v[ci] = sd[ci] * mix[:, j]
        reads.append(show(src[:, j], v))
    di = (sect[L] == 3).nonzero().squeeze(1)
    he = (h[ev] - mu).to(DEV)
    M = he[:, di.to(DEV)].T @ (he @ A)[:, di.to(DEV)]
    ev_, B = torch.linalg.eigh(((M + M.T) / 2).cpu())
    B, ev_ = B.flip(1), ev_.flip(0)
    dirs = []
    for j in range(8):
        v = torch.zeros(C.d)
        v[di] = B[:, j]
        dirs.append(show((h[~ev] - mu) @ (v / v.norm()), v) | {"share": (ev_[j] / M.diagonal().sum().cpu()).item()})
    res["reads"] = {"layer": L, "late_ica": reads, "d_directions": dirs}
    save("late", res)


# ------------------------------------------------------------------------------------------------------------ units

def units() -> None:
    import pandas as pd
    C = Ctx(lenses=None)
    sect = sectors_of(load("profile"))
    nL = C.nL
    W, R, NR = [], [], []
    for L in range(nL):
        D = directions(MODEL, L)
        w = D["write"].float()
        w2 = (w / C.sig_out(L)[:, None]).pow(2)
        W.append(torch.stack([w2[sect[L] == k].sum(0) / w2.sum(0) for k in range(5)]))
        raw = w.pow(2)
        NR.append(raw[sect[L] == 4].sum(0) / raw.sum(0))
        lr = sect[max(0, L - 1)]
        rs = []
        for key in ("gate", "up"):
            x = (D[key].float() * C.sig["mid"][L]).pow(2)
            rs.append(torch.stack([x[:, lr == k].sum(1) / x.sum(1) for k in range(5)]))
        R.append((rs[0] + rs[1]) / 2)
    W, R, NR = torch.stack(W), torch.stack(R), torch.stack(NR)
    frac = torch.stack([torch.stack([(sect[L] == k).float().mean() for k in range(5)]) for L in range(nL)])
    f = W.shape[2]
    res = {"model": MODEL, "neurons": {"by_layer": [], "half_on_one": {}, "top": {}, "vs_labels": {}}, "heads": {}}
    ch = [chance(C, L, sect) for L in range(nL)]
    for L in range(nL):
        lg = torch.log(R[L, 4].clamp(1e-6, 1 - 1e-6) / (1 - R[L, 4].clamp(1e-6, 1 - 1e-6)))
        v = valley(lg.numpy())
        past = lg > v[0] if v else None
        res["neurons"]["by_layer"].append({"layer": L, "frac": frac[L].tolist(), "write_median": W[L].median(1).values.tolist(),
                                            "write_n_raw_median": NR[L].median().item(), "read_median": R[L].median(1).values.tolist(),
                                            "read_n_valley": v, "read_n_past": past.float().mean().item() if v else None,
                                            "read_n_groups_median": [R[L, 4][~past].median().item(), R[L, 4][past].median().item()] if v and past.any() and (~past).any() else None,
                                            "read_n_half": int((R[L, 4] >= 0.5).sum()), "read_c_half": int((R[L, 0] >= 0.5).sum()),
                                            "write_n_raw_half": int((NR[L] >= 0.5).sum()),
                                            "chance_read": ch[L]["mlp_read"], "chance_write": ch[L]["mlp_write"]})
    for name, X in (("write", W), ("read", R)):
        for k in range(5):
            Xk = NR if (name == "write" and k == 4) else X[:, k]
            res["neurons"]["half_on_one"][f"{name}_{NAMES[k]}"] = int((Xk >= 0.5).sum())
            top = Xk.flatten().topk(6).indices.tolist()
            entries = []
            for i in top:
                L, n = divmod(i, f)
                lab = pq.read_table(neuron_table("labels", L), columns=["fires", "writes"]).to_pydict()
                entries.append({"layer": L, "neuron": n, "share": Xk.flatten()[i].item(), "sector_frac": frac[L, k].item(),
                                "+": [lab["fires"][n], lab["writes"][n]], "-": [lab["fires"][f + n], lab["writes"][f + n]]})
            res["neurons"]["top"][f"{name}_{NAMES[k]}"] = entries
    cors = []
    for L in range(nL):
        lab = pq.read_table(neuron_table("labels", L), columns=["writes_conf", "fires_conf"]).to_pydict()
        rd = pq.read_table(neuron_table("reads", L), columns=["vocab_kurt"]).to_pydict()
        g = lambda x: np.array([np.nan if v is None else float(v) for v in x])
        lens_w = np.tile((W[L, 0] + W[L, 1] + W[L, 2]).numpy(), 2)
        rc = np.tile(R[L, 0].numpy(), 2)
        sp = lambda a, b: spearmanr(a, b, nan_policy="omit").statistic
        cors.append([sp(lens_w, g(lab["writes_conf"])), sp(lens_w, g(rd["vocab_kurt"])), sp(rc, g(lab["fires_conf"]))])
        cm = pq.read_table(neuron_table("acts", L), columns=["cur_mode"]).to_pydict()["cur_mode"]
        cm = np.nanmax(np.stack([g(cm)[:f], g(cm)[f:]]), 0)
        cors[-1].append(sp(R[L, 0].numpy(), cm))
    cors = np.array(cors)
    bands = [(0, nL // 4), (nL // 4, nL // 2), (nL // 2, 3 * nL // 4), (3 * nL // 4, nL)]
    res["neurons"]["vs_labels"] = {f"L{a}-L{b - 1}": dict(zip(["write_lens_sectors~writes_conf", "write_lens_sectors~vocab_kurt",
                                                              "read_c~fires_conf", "read_c~cur_mode"], np.nanmedian(cors[a:b], 0).tolist()))
                                   for a, b in bands}
    # heads
    rows = []
    for L in range(nL):
        ly = arch.layout(MODEL)[L]
        q = arch.qk(MODEL, ly)
        Wv, Wo = arch.ov(MODEL, ly)
        si, sm = C.sig["in"][L], C.sig["mid"][L]
        lr, lw = sect[max(0, L - 1)], sect[L]
        hd = ly.head_dim
        for hh in range(ly.q_heads):
            vs, os_ = arch.head_slices(ly, hh)
            kv = arch.kv_of(ly, hh)
            o = {"layer": L, "head": hh}
            for name, Wr in (("Q", q["Wq"][hh * hd:(hh + 1) * hd]), ("K", q["Wk"][kv * hd:(kv + 1) * hd]), ("V", Wv[vs])):
                e = (Wr * si).pow(2).sum(0)
                for k in range(5):
                    o[f"{name}_{NAMES[k]}"] = (e[lr == k].sum() / e.sum()).item()
            wo = Wo[:, os_]
            e = (wo / sm[:, None]).pow(2).sum(1)
            raw = wo.pow(2).sum(1)
            for k in range(5):
                o[f"O_{NAMES[k]}"] = (e[lw == k].sum() / e.sum()).item()
            o["O_n_raw"] = (raw[lw == 4].sum() / raw.sum()).item()
            rows.append(o)
    df = pd.DataFrame(rows)
    acts = pq.read_table(head_table("acts"), columns=["layer", "head", "sink", "prev", "self", "induction", "duplicate"]).to_pandas()
    rds = pq.read_table(head_table("reads"), columns=["layer", "head", "ov_R_copy", "qk_self_S"]).to_pandas()
    labs = pq.read_table(head_table("labels"), columns=["layer", "head", "attends", "writes"]).to_pandas()
    for tb in (acts, rds, labs):
        tb["layer"], tb["head"] = tb["layer"].astype(int), tb["head"].astype(int)
    m = df.merge(acts, on=["layer", "head"]).merge(rds, on=["layer", "head"]).merge(labs, on=["layer", "head"])
    mid = m[(m.layer >= nL // 4) & (m.layer < 3 * nL // 4)]
    hres = {"by_layer": [], "vs_catalog_mid": {}, "top": {}}
    for L in range(nL):
        g_ = m[m.layer == L]
        hres["by_layer"].append({"layer": L, "frac_c": frac[L, 0].item(), **{c: float(g_[c].median()) for c in ("Q_c", "K_c", "V_c", "O_c", "O_n_raw", "K_n")},
                                 "chance_read": ch[L]["head_read"], "chance_write": ch[L]["head_write"]})
    for c in ("Q_c", "K_c", "V_c", "O_c", "K_n", "O_n_raw"):
        hres["vs_catalog_mid"][c] = {x: float(spearmanr(mid[c], mid[x], nan_policy="omit").statistic)
                                     for x in ("sink", "prev", "self", "induction", "duplicate", "ov_R_copy", "qk_self_S")}
        hres["vs_catalog_mid"][c]["all_prev"] = float(spearmanr(m[c], m["prev"], nan_policy="omit").statistic)
        hres["vs_catalog_mid"][c]["all_sink"] = float(spearmanr(m[c], m["sink"], nan_policy="omit").statistic)
        hres["top"][c] = [{"layer": int(r["layer"]), "head": int(r["head"]), "share": float(r[c]), "sink": float(r["sink"]),
                           "prev": float(r["prev"]), "self": float(r["self"]), "duplicate": float(r["duplicate"]),
                           "attends": str(r["attends"]), "writes": str(r["writes"])}
                          for _, r in m.sort_values(c, ascending=False).head(8).iterrows()]
    hres["core_qk_heads_by_quarter"] = {f"L{a}-L{b - 1}": {s: float(np.mean(m[(m.layer >= a) & (m.layer < b)][f"{s}_c"] >= 0.3)) for s in "QKV"}
                                        for a, b in bands}
    res["heads"] = hres
    log("units: " + " ".join(f"{k} {v}" for k, v in res["neurons"]["half_on_one"].items()))
    save("units", res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("profile", "basis", "unions", "depth", "late", "units"))
    ap.add_argument("--layers", default=None, help="basis: three layers; unions: the ablated layers; late: the layers")
    a = ap.parse_args()
    gemma = arch.is_gemma4(MODEL)
    lay = [int(x) for x in a.layers.split(",")] if a.layers else None
    if a.part == "profile":
        profile()
    elif a.part == "basis":
        basis(lay or ([12, 20, 28] if gemma else [8, 14, 22]))
    elif a.part == "unions":
        unions(lay or [4, 8, 14, 22, 28])
    elif a.part == "depth":
        depth([2, 4, 8, 14, 22, 28, 34])
    elif a.part == "late":
        late(lay or [20, 24, 28, 32, 34, 35], 30)
    else:
        units()


if __name__ == "__main__":
    main()
