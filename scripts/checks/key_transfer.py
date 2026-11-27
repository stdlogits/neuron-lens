"""Which sector carries a factual key, within a model and from one model to another (findings §22): Qwen3-8B and
Gemma 4 12B, their half dtypes, maps in fp64; CUDA, else MPS.

Plato's Key (Cheng, Liu, Yau and Ré, Hazy Research 2026) swaps a donor model's residual at the last subject token of a
factual prompt, mapped by a centred orthogonal Procrustes rotation and one global scale, into a recipient's prompt for
another subject of the same relation, and counts how often the recipient then gives the donor subject's answer. Here
the swap is cut by §22's coordinate sectors (``sectors_profile_<model>.json``: c, r, s, d, n at each layer's output).

**Facts.** CounterFact (``NeelNanda/counterfact-tracing``), 21 relations, one template each (the most common, or
``TEMPLATES`` where it invites another continuation), prefixed ``Fact:``. A fact is kept when both models give its answer
under teacher forcing (every answer token the argmax). Split by subject (crc32): train 60% (the fits), val 15% (layer
choice, 256 pairs), test 25% (1,000 pairs; ``split`` uses every test donor). A pair (recipient prompt's subject s2,
donor subject s1) shares a relation and differs in its answer and its first answer token.

**Swaps.** The residual at s2's last token at layer ℓ_B's output is replaced; success is every token of s1's answer the
argmax (``full``; ``first``: its first token; ``kept``: the first token of s2's own answer). Within the recipient
(``self_*``): s1's own state, or with one sector from s1 and the rest at the recipient's training mean (``_mean``) or its
own state (``_hyb``). Across models (``x_*``): ``μ_B + s R (z_A − μ_A)`` fitted in each model's metric frame (``× 1/s``,
``s² = σ/|γ|`` at the next norm; ``--frame raw`` fits raw coordinates, which fail into Gemma 4, whose norm gains reach
~360); the full map fed one sector of the donor's centred key (``x_full_cin`` ...); maps fitted between two sectors
(c_A → c_B ...); top-256 PCs; controls (a random rotation, a fit on shuffled pairs, random coordinate sets of c's size,
d's top-|c| PCs).

Stages (outputs ``data/summary/key_transfer_<stage>_<recipient>.json``; facts and keys in ``data/key_transfer/``):

- ``facts``: both models in turn, recall and keys at every other layer.
- ``swap --recipient <qwen|gemma>``: the layer grid on val (full map), every variant at the best pair on test, and a
  depth profile (each recipient layer with its best donor layer); where the full map sends each donor sector.
- ``frames``: the full map fitted in four frames (raw, the next norm's gain, metric, the rms of all later gains).
- ``split``: per-pair success of the main variants at the best pair, by the donor subject's token count.
- ``landing``: where the full map sends each donor sector (its share on the recipient's core) against a fit on shuffled
  pairs, and by whether both tokenizers end the subject on the same token.
- ``sparse``: at the best layer, the recipient's own key with only k of d's coordinates from the donor (the rest at the
  mean), ranked by variance, by attribution (gradient × change at the midpoint between the mean and the donor's key, of
  the donor answer's logit over the recipient's), at random, or d's top-k PCs; and d without its top-k by attribution.
"""
import argparse
import json
import time
import zlib

import numpy as np
import pandas as pd
import torch

from neuron_lens import arch, metric
from neuron_lens.catalog import tokenizer
from neuron_lens.paths import DATA, SUMMARY
from neuron_lens.transport import decoder_stack

KT = DATA / "key_transfer"
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MODELS = {"qwen": "Qwen/Qwen3-8B", "gemma": "google/gemma-4-12B"}
LAYERS = {"qwen": list(range(0, 36, 2)), "gemma": list(range(0, 48, 2))}
GRID = {"qwen": list(range(4, 26, 2)), "gemma": list(range(6, 32, 2))}
DGRID = {m: g[::2] for m, g in GRID.items()}  # donor layers (the donor axis is a broad plateau)
RELATIONS = ["P101", "P103", "P108", "P131", "P136", "P1412", "P159", "P17", "P176", "P178", "P19", "P20",
             "P27", "P276", "P36", "P364", "P37", "P407", "P449", "P495", "P937"]
TEMPLATES = {"P101": "{} is known for work in the field of", "P1412": "The language spoken by {} is",
             "P136": "The genre of {} is", "P20": "{} died in the city of", "P37": "The official language of {} is",
             "P937": "{} worked in the city of", "P19": "{} was born in the city of",
             "P17": "{} is located in the country of", "P364": "The original language of {} is"}
PREFIX = "Fact:"
COUNTERFACT = ("NeelNanda/counterfact-tracing", "data/train-00000-of-00001-36693f2cad948c42.parquet")
KS = [16, 32, 64, 128, 256, 512, 1024, 2048]
BATCH = 96
FRAME = "metric"
T0 = time.time()
LOG = []


def log(msg):
    line = f"[{time.time() - T0:6.0f}s] {msg}"
    print(line, flush=True)
    LOG.append(line)


def out_path(stage, m, ext="json"):
    tag = "" if FRAME == "metric" or stage in ("frames", "sparse", "landing") else f"{FRAME}_"
    return SUMMARY / f"key_transfer_{stage}_{tag}{arch.stem(MODELS[m])}.{ext}"


def save(stage, m, res):
    out_path(stage, m).write_text(json.dumps(res))
    out_path(stage, m, "log").write_text("\n".join(LOG) + "\n")


# ------------------------------------------------------------------------------------------------ facts and keys
def fact_table():
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download(COUNTERFACT[0], COUNTERFACT[1], repo_type="dataset"))
    rows = []
    for rid in RELATIONS:
        sub = df[df.relation_id == rid]
        pre, suf = TEMPLATES.get(rid, sub.relation.value_counts().index[0]).split("{}")
        head = PREFIX + " " + pre
        for s, a in zip(sub.subject, sub.target_true):
            s, a = s.strip(), a.strip()
            rows.append(dict(rel=rid, subject=s, answer=a, prompt=head + s + suf, subj_end=len(head) + len(s)))
    return pd.DataFrame(rows).drop_duplicates(["rel", "subject"]).reset_index(drop=True)


def tokenize(tok, f):
    """Per fact: prompt ids, answer ids (clean boundaries only), the subject's last token, else None."""
    out = []
    for p, a, se in zip(f.prompt, f.answer, f.subj_end):
        e = tok(p + " " + a, return_offsets_mapping=True)
        ids, off = e["input_ids"], e["offset_mapping"]
        bound = [i for i, (s, t) in enumerate(off) if s >= len(p) and t > s]
        if not bound or off[bound[0] - 1][1] != len(p):
            out.append(None)
            continue
        b = bound[0]
        last = [i for i, (s, t) in enumerate(off[:b]) if s < se <= t]
        if len(last) != 1 or off[last[0]][1] != se:
            out.append(None)
            continue
        out.append(dict(prompt=ids[:b], answer=ids[b:], pos=last[0]))
    return out


def load(m):
    lm = arch.load_model(MODELS[m], DEV, eager=False)[0]
    return lm, decoder_stack(lm)


def batch_ids(seqs):
    B, T = len(seqs), max(len(s) for s in seqs)
    ids = torch.zeros(B, T, dtype=torch.long)
    att = torch.zeros(B, T, dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor(s)
        att[b, :len(s)] = 1
    return ids.to(DEV), att.to(DEV)


def forward(lm, stack, seqs, pos, ans_start, ans_len, capture=(), swap=None):
    """Right-padded batch; returns (argmax ids per answer position [list], {L: [B, d] at pos}) with an optional
    ``swap = (L, vecs [B, d])`` written into layer L's output at ``pos``."""
    ids, att = batch_ids(seqs)
    B = len(seqs)
    bi = torch.arange(B, device=DEV)
    pt = torch.tensor(pos, device=DEV)
    cap, hooks = {}, []
    for L in capture:
        hooks.append(stack.layers[L].register_forward_hook(
            lambda m_, i, o, L=L: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[bi, pt].float().cpu())))
    if swap is not None:
        Ls, V = swap
        V = V.to(DEV)

        def put(m_, i, o):
            h = o[0] if isinstance(o, tuple) else o
            h[bi, pt] = V.to(h.dtype)
            return o
        hooks.append(stack.layers[Ls].register_forward_hook(put))
    try:
        with torch.inference_mode():
            hid = stack(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
            rows = [hid[b, ans_start[b] - 1 + k] for b in range(B) for k in range(ans_len[b])]
            pred = arch.logits(lm, torch.stack(rows)).argmax(-1).cpu().tolist()
    finally:
        for h in hooks:
            h.remove()
    out, i = [], 0
    for b in range(B):
        out.append(pred[i:i + ans_len[b]])
        i += ans_len[b]
    return out, cap


def stage_facts():
    KT.mkdir(parents=True, exist_ok=True)
    f = fact_table()
    log(f"{len(f)} facts in {len(RELATIONS)} relations")
    toks = {m: tokenize(tokenizer(MODELS[m]), f) for m in MODELS}
    ok = np.array([toks["qwen"][i] is not None and toks["gemma"][i] is not None for i in range(len(f))])
    f = f[ok].reset_index(drop=True)
    toks = {m: [t for t, k in zip(toks[m], ok) if k] for m in MODELS}
    log(f"{len(f)} with clean token boundaries in both")
    for m in MODELS:
        lm, stack = load(m)
        t = toks[m]
        K = torch.zeros(len(f), len(LAYERS[m]), stack.config.hidden_size, dtype=torch.float16)
        rec = np.zeros(len(f), bool)
        order = np.argsort([len(x["prompt"]) + len(x["answer"]) for x in t])
        for s in range(0, len(f), BATCH):
            idx = order[s:s + BATCH]
            pred, cap = forward(lm, stack, [t[i]["prompt"] + t[i]["answer"] for i in idx], [t[i]["pos"] for i in idx],
                                [len(t[i]["prompt"]) for i in idx], [len(t[i]["answer"]) for i in idx],
                                capture=LAYERS[m])
            for j, i in enumerate(idx):
                rec[i] = pred[j] == t[i]["answer"]
            K[torch.from_numpy(idx)] = torch.stack([cap[L] for L in LAYERS[m]], 1).half()
        torch.save({"keys": K, "recall": torch.tensor(rec), "layers": LAYERS[m], "toks": t}, KT / f"keys_{m}.pt")
        log(f"{m}: recall {rec.mean():.3f}")
        del lm, stack, K
        if DEV == "mps":
            torch.mps.empty_cache()
    f.to_parquet(KT / "facts.parquet")


# ------------------------------------------------------------------------------------------------ maps
def procrustes(ZA, ZB):
    M = ZB.T @ ZA
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    if not (torch.isfinite(U).all() and torch.isfinite(Vh).all()):  # gesdd can fail silently; gesvd is sturdier
        import scipy.linalg
        U, _, Vh = (torch.from_numpy(x) for x in scipy.linalg.svd(M.numpy(), full_matrices=False, lapack_driver="gesvd"))
    return U @ Vh, (ZB.norm() / ZA.norm()).item()


def rand_orth(dB, dA, g):
    Q, _ = torch.linalg.qr(torch.randn(max(dA, dB), min(dA, dB), generator=g, dtype=torch.float64))
    return Q if dB >= dA else Q.T


def sectors(m):
    prof = json.loads((SUMMARY / f"sectors_profile_{arch.stem(MODELS[m])}.json").read_text())
    return [np.array(list(s)) for s in prof["sectors"]]


def frame(m, L, kind):
    """``[d]`` per-coordinate factor that takes layer L's output into the frame a map is fitted in: ``raw`` (ones),
    ``gain`` (the next layer's input-norm gain), ``metric`` (``1/s``, ``s² = σ/|γ|`` at that norm), ``read`` (the rms
    of every later norm's gain: each later layer's input and MLP norms and the final norm)."""
    mid = MODELS[m]
    if kind == "raw":
        return torch.ones(arch.text_config(mid).hidden_size, dtype=torch.float64)
    if kind == "gain":
        return metric.norm_weight(mid, "in", L + 1).abs().double()
    if kind == "metric":
        return metric.s2(mid, "out", L).double().rsqrt()
    n = arch.text_config(mid).num_hidden_layers
    gs = ([metric.norm_weight(mid, "in", l) for l in range(L + 1, n + 1)]
          + [metric.norm_weight(mid, "mid", l) for l in range(L + 1, n)])
    return torch.stack(gs).double().pow(2).mean(0).sqrt()


class Side:
    """One model's keys at one layer: raw keys and their train mean (``mraw``); in the fitting frame (``F = K ⊙ f``)
    the train mean ``mu`` and centred train keys ``Z``; its sector index sets."""

    def __init__(self, m, keys, L, sect, train, g, kind=None):
        self.m, self.L = m, L
        self.K = keys["keys"][:, keys["layers"].index(L)].double()
        self.mraw = self.K[train].mean(0)
        self.f = frame(m, L, kind or FRAME)
        self.F = self.K * self.f
        self.mu = self.F[train].mean(0)
        self.Z = self.F[train] - self.mu
        s = sect[L]
        self.idx = {k: torch.tensor(np.where(s == k)[0]) for k in "crsdn"}
        self.idx["rs"] = torch.cat([self.idx["r"], self.idx["s"]])
        self.idx["crs"] = torch.cat([self.idx["c"], self.idx["rs"]])
        notn = torch.tensor(np.where(s != "n")[0])
        self.idx["randc"] = notn[torch.randperm(len(notn), generator=g)[:len(self.idx["c"])]]
        self.d = self.K.shape[1]


def sub_map(A, B, ka, kb, k_pc=None):
    """Procrustes between A's coordinates ``ka`` and B's ``kb`` (optionally each side's top ``k_pc`` PCs within them)."""
    ia, ib = A.idx[ka] if isinstance(ka, str) else ka, B.idx[kb] if isinstance(kb, str) else kb
    ZA, ZB = A.Z[:, ia], B.Z[:, ib]
    PA = PB = None
    if k_pc:
        PA = torch.linalg.svd(ZA, full_matrices=False)[2][:k_pc].T
        PB = torch.linalg.svd(ZB, full_matrices=False)[2][:k_pc].T
        ZA, ZB = ZA @ PA, ZB @ PB
    R, s = procrustes(ZA, ZB)
    return dict(ia=ia, ib=ib, PA=PA, PB=PB, R=R, s=s)


def apply_sub(mp, A, B, donors, base):
    """B's raw vectors: ``base`` [n, dB] with B's coordinates ``ib`` set from A's donors through the sub-map."""
    z = A.F[donors][:, mp["ia"]] - A.mu[mp["ia"]]
    if mp["PA"] is not None:
        z = z @ mp["PA"]
    y = mp["s"] * z @ mp["R"].T
    if mp["PB"] is not None:
        y = y @ mp["PB"].T
    out = base.clone()
    out[:, mp["ib"]] = (B.mu[mp["ib"]] + y) / B.f[mp["ib"]]
    return out


def mapped(A, B, R, s, donors, mask=None):
    """B's raw vectors from A's donors through the full map (``mask``: only these coordinates of A's centred key)."""
    z = A.F[donors] - A.mu
    if mask is not None:
        keep = torch.zeros(A.d, dtype=torch.bool)
        keep[mask] = True
        z = z * keep
    return (B.mu + s * z @ R.T) / B.f


# ------------------------------------------------------------------------------------------------ swaps
def pairs(f, toks_b, split_ix, n, g):
    """(recipient j, donor i) in the same relation with different answers whose first tokens differ under B."""
    rng = np.random.default_rng(int(torch.randint(0, 2 ** 31, (1,), generator=g)))
    by_rel = {}
    for i in split_ix:
        by_rel.setdefault(f.rel[i], []).append(i)
    out = []
    for i in rng.permutation(split_ix):
        pool = by_rel[f.rel[i]]
        for _ in range(20):
            j = pool[rng.integers(len(pool))]
            if f.answer[j] != f.answer[i] and toks_b[j]["answer"][0] != toks_b[i]["answer"][0]:
                out.append((j, i))
                break
        if len(out) >= n:
            break
    return out


def run(lm, stack, toks_b, L, prs, V, per=False):
    """Success rates of swapping rows of V into the recipient prompts at L (``per``: also each pair's full success)."""
    full = first = kept = 0
    each = []
    for s in range(0, len(prs), BATCH):
        pp = prs[s:s + BATCH]
        pred, _ = forward(lm, stack, [toks_b[j]["prompt"] + toks_b[i]["answer"] for j, i in pp],
                          [toks_b[j]["pos"] for j, _ in pp], [len(toks_b[j]["prompt"]) for j, _ in pp],
                          [len(toks_b[i]["answer"]) for _, i in pp], swap=(L, V[s:s + BATCH].float()))
        for (j, i), p in zip(pp, pred):
            full += p == toks_b[i]["answer"]
            first += p[0] == toks_b[i]["answer"][0]
            kept += p[0] == toks_b[j]["answer"][0]
            each.append(int(p == toks_b[i]["answer"]))
    n = len(prs)
    out = dict(full=full / n, first=first / n, kept=kept / n, n=n)
    return (out, each) if per else out


class Setup:
    """Facts, keys, splits, val and test pairs for one recipient (the same pairs in every stage)."""

    def __init__(self, recipient, n_val=256, n_test=1000):
        self.recipient = recipient
        self.donor = "gemma" if recipient == "qwen" else "qwen"
        self.f = pd.read_parquet(KT / "facts.parquet")
        self.keys = {m: torch.load(KT / f"keys_{m}.pt", weights_only=False) for m in MODELS}
        both = (self.keys["qwen"]["recall"] & self.keys["gemma"]["recall"]).numpy()
        sub = np.array([zlib.crc32(s.encode()) % 100 for s in self.f.subject])
        self.train = np.where(both & (sub < 60))[0]
        self.val = np.where(both & (sub >= 60) & (sub < 75))[0]
        self.test = np.where(both & (sub >= 75))[0]
        log(f"recall: qwen {self.keys['qwen']['recall'].float().mean():.3f}, gemma "
            f"{self.keys['gemma']['recall'].float().mean():.3f}, both {both.mean():.3f}; train {len(self.train)}, "
            f"val {len(self.val)}, test {len(self.test)}")
        self.g = torch.Generator().manual_seed(0)
        self.tb = self.keys[recipient]["toks"]
        self.sect = {m: sectors(m) for m in MODELS}
        self.pv = pairs(self.f, self.tb, self.val, n_val, self.g)
        self.pt = pairs(self.f, self.tb, self.test, n_test, self.g)

    def side(self, m, L, kind=None):
        return Side(m, self.keys[m], L, self.sect[m], self.train, self.g, kind)

    def best(self):
        return json.loads(out_path("swap", self.recipient).read_text())["best"]


def block_shares(R, A, B, test):
    """Where the full map sends each sector of A's centred test keys (in the fitting frame): share of the mapped energy
    on each of B's sectors, against B's own share of energy there and the coordinate share."""
    zA = A.F[test] - A.mu
    zB = B.F[test] - B.mu
    res = {}
    for ka in "cdrs":
        za = torch.zeros_like(zA)
        za[:, A.idx[ka]] = zA[:, A.idx[ka]]
        y = za @ R.T
        e = y.pow(2).sum()
        res[ka] = {kb: (y[:, B.idx[kb]].pow(2).sum() / e).item() for kb in "crsdn"}
    res["B_own"] = {kb: (zB[:, B.idx[kb]].pow(2).sum() / zB.pow(2).sum()).item() for kb in "crsdn"}
    res["coords"] = {kb: len(B.idx[kb]) / B.d for kb in "crsdn"}
    res["A_energy"] = {ka: (zA[:, A.idx[ka]].pow(2).sum() / zA.pow(2).sum()).item() for ka in "crsdn"}
    return res


def variants(A, B, prs, g, which="all"):
    """{name: [n, dB] raw vectors} for the pairs ``prs`` (recipient j, donor i). A is the donor side at ℓ_A, B the
    recipient side at ℓ_B (B also supplies the self swaps)."""
    J = torch.tensor([j for j, _ in prs])
    I = torch.tensor([i for _, i in prs])
    own, donor_self = B.K[J], B.K[I]
    mean = B.mraw.expand(len(prs), -1)
    V = {"self_full": donor_self}

    def mix(base, k):
        out = base.clone()
        out[:, B.idx[k]] = donor_self[:, B.idx[k]]
        return out
    for k in ("c", "d"):
        V[f"self_{k}_hyb"] = mix(own, k)
        V[f"self_{k}_mean"] = mix(mean, k)
    R, s = procrustes(A.Z, B.Z)
    V["x_full"] = mapped(A, B, R, s, I)
    for k in ("c", "d"):
        V[f"x_full_{k}in"] = mapped(A, B, R, s, I, A.idx[k])
    if which == "depth":
        return V, R
    for k in ("rs", "randc"):
        V[f"self_{k}_hyb"] = mix(own, k)
    V["self_mean"] = mean.clone()
    V["self_own"] = own.clone()
    V["x_full_rsin"] = mapped(A, B, R, s, I, A.idx["rs"])
    V["x_rand"] = mapped(A, B, rand_orth(B.d, A.d, g), s, I)
    perm = torch.randperm(A.Z.shape[0], generator=g)
    V["x_shuf"] = mapped(A, B, *procrustes(A.Z[perm], B.Z), I)
    V["x_pc256"] = apply_sub(sub_map(A, B, torch.arange(A.d), torch.arange(B.d), 256), A, B, I, mean)
    for k in ("c", "d", "crs", "randc"):
        mp = sub_map(A, B, k, k)
        V[f"x_{k}_mean"] = apply_sub(mp, A, B, I, mean)
        V[f"x_{k}_hyb"] = apply_sub(mp, A, B, I, own)
    kc = min(len(A.idx["c"]), len(B.idx["c"]))
    V["x_dpc_mean"] = apply_sub(sub_map(A, B, "d", "d", kc), A, B, I, mean)
    return V, R


def stage_swap(recipient, n_val, n_test, n_depth):
    S = Setup(recipient, n_val, n_test)
    donor, tb = S.donor, S.tb
    lm, stack = load(recipient)
    res = dict(recipient=recipient, donor=donor, frame=FRAME, n_train=len(S.train), n_val=len(S.val),
               n_test=len(S.test), grid={})
    side = {}
    for Lb in GRID[recipient]:
        B = S.side(recipient, Lb)
        res["grid"][Lb] = {"self": run(lm, stack, tb, Lb, S.pv, B.K[torch.tensor([i for _, i in S.pv])])}
        for La in DGRID[donor]:
            A = side.get(La) or S.side(donor, La)
            side[La] = A
            R, s = procrustes(A.Z, B.Z)
            res["grid"][Lb][La] = run(lm, stack, tb, Lb, S.pv, mapped(A, B, R, s, torch.tensor([i for _, i in S.pv])))
        g = res["grid"][Lb]
        log(f"val L{Lb}: self {g['self']['full']:.3f} " + " ".join(f"{La}:{g[La]['full']:.2f}" for La in DGRID[donor]))
        save("swap", recipient, res)
    Lb, La = max(((b, a) for b in GRID[recipient] for a in DGRID[donor]), key=lambda p: res["grid"][p[0]][p[1]]["full"])
    res["best"] = [Lb, La]
    log(f"best pair: recipient L{Lb}, donor L{La}")
    A, B = S.side(donor, La), S.side(recipient, Lb)
    V, R = variants(A, B, S.pt, S.g)
    res["sizes"] = {"A": {k: len(v) for k, v in A.idx.items()}, "B": {k: len(v) for k, v in B.idx.items()}}
    res["blocks"] = block_shares(R, A, B, torch.tensor(S.test))
    res["test"] = {}
    for name, v in V.items():
        res["test"][name] = run(lm, stack, tb, Lb, S.pt, v)
        log(f"test {name:14s} full {res['test'][name]['full']:.3f} kept {res['test'][name]['kept']:.3f}")
    save("swap", recipient, res)
    res["depth"] = {}
    pd_ = S.pt[:n_depth]
    for Lb in GRID[recipient]:
        La = max(DGRID[donor], key=lambda x: res["grid"][Lb][x]["full"])
        A, B = S.side(donor, La), S.side(recipient, Lb)
        V, R = variants(A, B, pd_, S.g, which="depth")
        res["depth"][Lb] = {"donor_layer": La, "blocks": block_shares(R, A, B, torch.tensor(S.test))}
        for name, v in V.items():
            res["depth"][Lb][name] = run(lm, stack, tb, Lb, pd_, v)
        log(f"depth L{Lb} (donor L{La}): " + " ".join(f"{k}={res['depth'][Lb][k]['full']:.2f}" for k in V))
        save("swap", recipient, res)


def stage_frames(recipient, layer_pairs, n_val):
    S = Setup(recipient, n_val)
    I = torch.tensor([i for _, i in S.pv])
    lm, stack = load(recipient)
    res = {}
    for Lb, La in layer_pairs:
        cell = {"self": run(lm, stack, S.tb, Lb, S.pv, S.side(S.recipient, Lb, "raw").K[I])}
        for kind in ("raw", "gain", "metric", "read"):
            A, B = S.side(S.donor, La, kind), S.side(S.recipient, Lb, kind)
            R, s = procrustes(A.Z, B.Z)
            cell[kind] = run(lm, stack, S.tb, Lb, S.pv, mapped(A, B, R, s, I))
        res[f"{Lb}:{La}"] = cell
        log(f"frames L{Lb} <- L{La}: " + " ".join(f"{k} {v['full']:.3f}" for k, v in cell.items()))
        save("frames", recipient, res)


def stage_split(recipient):
    S = Setup(recipient, n_test=100000)
    Lb, La = S.best()
    A, B = S.side(S.donor, La), S.side(recipient, Lb)
    V, _ = variants(A, B, S.pt, S.g)
    keep = ["self_full", "self_c_mean", "self_d_mean", "self_c_hyb", "self_d_hyb", "self_mean", "x_full", "x_full_cin",
            "x_full_din", "x_full_rsin", "x_c_mean", "x_d_mean", "x_crs_mean"]
    ntok = {m: np.array([len(tokenizer(MODELS[m])(" " + s, add_special_tokens=False)["input_ids"])
                         for s in S.f.subject]) for m in MODELS}
    lm, stack = load(recipient)
    I = np.array([i for _, i in S.pt])
    res = dict(recipient=recipient, donor=S.donor, frame=FRAME, best=[Lb, La], n=len(S.pt), per={},
               ntok={m: ntok[m][I].tolist() for m in MODELS})
    one = ntok[recipient][I] == 1
    for name in keep:
        out, each = run(lm, stack, S.tb, Lb, S.pt, V[name], per=True)
        res["per"][name] = each
        e = np.array(each)
        log(f"split {name:12s} all {e.mean():.3f}  one-token {e[one].mean():.3f} (n {one.sum()})  "
            f"multi {e[~one].mean():.3f} (n {(~one).sum()})")
    save("split", recipient, res)


def stage_landing(recipient):
    """Where the full map at the best pair sends each donor sector's content (share of its mapped energy on the
    recipient's core), against a fit on shuffled pairs, and for subjects whose last token is or isn't the same string
    in both tokenizers. CPU only."""
    S = Setup(recipient)
    Lb, La = S.best()
    A, B = S.side(S.donor, La), S.side(recipient, Lb)
    R, _ = procrustes(A.Z, B.Z)
    Rs, _ = procrustes(A.Z[torch.randperm(A.Z.shape[0], generator=S.g)], B.Z)
    T = torch.tensor(S.test)
    last = {}
    for m in MODELS:
        tk, t = tokenizer(MODELS[m]), S.keys[m]["toks"]
        last[m] = np.array([tk.decode([t[i]["prompt"][t[i]["pos"]]]).strip().lower() for i in S.test])
    same = torch.tensor(last["qwen"] == last["gemma"])
    fit, shuf = block_shares(R, A, B, T), block_shares(Rs, A, B, T)
    res = dict(best=[Lb, La], fit={k: fit[k]["c"] for k in "cdrs"}, shuffled={k: shuf[k]["c"] for k in "cdrs"},
               same_last_token=block_shares(R, A, B, T[same])["c"]["c"],
               different_last_token=block_shares(R, A, B, T[~same])["c"]["c"], n_same=int(same.sum()), n=len(T),
               coords=fit["coords"]["c"], own=fit["B_own"]["c"])
    log(f"landing on the core: {res}")
    save("landing", recipient, res)


# ------------------------------------------------------------------------------------------------ sparsity of d's part
def attribution(lm, stack, tb, L, prs, B):
    """``[d]``: mean over pairs of gradient × change at the midpoint between B's training mean and the donor's key, of
    the donor answer's first-token logit over the recipient's (at the last prompt position)."""
    tot = torch.zeros(B.d, dtype=torch.float64)
    for s in range(0, len(prs), 24):
        pp = prs[s:s + 24]
        ids, att = batch_ids([tb[j]["prompt"] for j, _ in pp])
        bi = torch.arange(len(pp), device=DEV)
        pt = torch.tensor([tb[j]["pos"] for j, _ in pp], device=DEV)
        delta = B.K[torch.tensor([i for _, i in pp])] - B.mraw
        X = (B.mraw + delta / 2).float().to(DEV).requires_grad_(True)

        def put(m_, i, o):
            h = o[0] if isinstance(o, tuple) else o
            h2 = h.clone()
            h2[bi, pt] = X.to(h.dtype)
            return (h2,) + tuple(o[1:]) if isinstance(o, tuple) else h2
        hook = stack.layers[L].register_forward_hook(put)
        try:
            with torch.enable_grad():
                hid = stack(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
                last = torch.tensor([len(tb[j]["prompt"]) - 1 for j, _ in pp], device=DEV)
                lg = arch.logits(lm, hid[bi, last])
                dt = torch.tensor([tb[i]["answer"][0] for _, i in pp], device=DEV)
                ot = torch.tensor([tb[j]["answer"][0] for j, _ in pp], device=DEV)
                (lg[bi, dt] - lg[bi, ot]).sum().backward()
        finally:
            hook.remove()
        tot += (X.grad.cpu().double() * delta).sum(0)
    return tot / len(prs)


def stage_sparse(recipient, n_train):
    S = Setup(recipient)
    ptr = pairs(S.f, S.tb, S.train, n_train, S.g)
    Lb = S.best()[0]
    B = S.side(recipient, Lb, "metric")
    lm, stack = load(recipient)
    d = B.idx["d"]
    notn = torch.tensor(np.where(S.sect[recipient][Lb] != "n")[0])
    score = attribution(lm, stack, S.tb, Lb, ptr, B)
    log(f"attribution over {len(ptr)} train pairs: d holds {score[d].sum() / score[notn].sum():.3f} of the total")
    I = torch.tensor([i for _, i in S.pt])
    donor, mean = B.K[I], B.mraw.expand(len(S.pt), -1)
    order = {"var": d[torch.argsort(B.Z[:, d].pow(2).sum(0), descending=True)],
             "attr": d[torch.argsort(score[d], descending=True)],
             "rand": d[torch.randperm(len(d), generator=S.g)],
             "attr_all": notn[torch.argsort(score[notn], descending=True)]}
    P = torch.linalg.svd(B.Z[:, d], full_matrices=False)[2]
    lab = S.sect[recipient][Lb]
    res = dict(recipient=recipient, layer=Lb, n_d=len(d), n_train_pairs=len(ptr), ks=KS, curves={},
               attr_share_d=(score[d].sum() / score[notn].sum()).item(),
               attr_top={k: (score[order["attr"][:k]].sum() / score[d].sum()).item() for k in KS},
               attr_all_top_sectors={k: {c: int((lab[order["attr_all"][:k].numpy()] == c).sum()) for c in "crsd"}
                                     for k in KS},
               attr_share_by_sector={c: (score[torch.tensor(np.where(lab == c)[0])].sum() / score[notn].sum()).item()
                                     for c in "crsd"},
               score=score.tolist())
    log(f"top-k by attribution over all coordinates, by sector: {res['attr_all_top_sectors']}")

    def keep(ix):
        v = mean.clone()
        v[:, ix] = donor[:, ix]
        return v
    res["all_d"] = run(lm, stack, S.tb, Lb, S.pt, keep(d))
    res["none"] = run(lm, stack, S.tb, Lb, S.pt, mean.clone())
    log(f"all of d {res['all_d']['full']:.3f}, none {res['none']['full']:.3f}")
    for name, o in order.items():
        res["curves"][name] = [run(lm, stack, S.tb, Lb, S.pt, keep(o[:k]))["full"] for k in KS]
        log(f"{name:8s} " + " ".join(f"{k}:{v:.3f}" for k, v in zip(KS, res["curves"][name])))
    z = (B.F[I][:, d] - B.mu[d])
    pcs = []
    for k in KS:
        v = mean.clone()
        v[:, d] = (B.mu[d] + z @ P[:k].T @ P[:k]) / B.f[d]
        pcs.append(run(lm, stack, S.tb, Lb, S.pt, v)["full"])
    res["curves"]["pc"] = pcs
    log("pc       " + " ".join(f"{k}:{v:.3f}" for k, v in zip(KS, pcs)))
    drop = []
    for k in KS:
        v = keep(d)
        v[:, order["attr"][:k]] = mean[:, order["attr"][:k]]
        drop.append(run(lm, stack, S.tb, Lb, S.pt, v)["full"])
    res["curves"]["attr_drop"] = drop
    log("drop     " + " ".join(f"{k}:{v:.3f}" for k, v in zip(KS, drop)))
    save("sparse", recipient, res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["facts", "swap", "frames", "split", "landing", "sparse"])
    ap.add_argument("--recipient", choices=list(MODELS))
    ap.add_argument("--frame", choices=["raw", "metric"], default="metric")
    ap.add_argument("--val", type=int, default=256)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--depth", type=int, default=1000)
    ap.add_argument("--train-pairs", type=int, default=1000)
    ap.add_argument("--pairs", default="", help="frames: recipient:donor layer pairs, e.g. 18:12,14:12")
    a = ap.parse_args()
    FRAME = a.frame
    if a.stage == "facts":
        stage_facts()
    elif a.stage == "swap":
        stage_swap(a.recipient, a.val, a.test, a.depth)
    elif a.stage == "frames":
        stage_frames(a.recipient, [tuple(map(int, x.split(":"))) for x in a.pairs.split(",")], a.val)
    elif a.stage == "split":
        stage_split(a.recipient)
    elif a.stage == "landing":
        stage_landing(a.recipient)
    else:
        stage_sparse(a.recipient, a.train_pairs)
