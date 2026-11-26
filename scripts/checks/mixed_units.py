"""Mixed units (findings §23): poles whose top documents fall into two topics, each coherent, as unlike as random
documents. Any catalog model (``MODEL``); the embeddings on CUDA or MPS.

§23's two cases (Qwen3-32B's Colorado and Tornado unit, Qwen3-14B's baseball and blood glucose unit) were found by
hand. This scan looks for more, in any model with an ``acts`` table, so a model with training checkpoints (OLMo 3 7B)
can have its cases traced back.

- ``embed``: every activation document (``catalog.documents``, up to 256 tokens) embedded once, Qwen3-Embedding-0.6B
  (``encode``). The embedding is anisotropic (random documents at cosine 0.46 ± 0.08 in Qwen3-14B's), so every part
  centres it on the documents' mean and renormalises (``load_docs``); the cosine of ``PAIRS`` random pairs of documents
  is then the scale (0.00 ± 0.08).
- ``scan``: for every pole with ``act_top`` ≥ ``FLOOR``, its top ``TOP`` documents split in two (spherical 2-means from
  the sign of their first principal component and from ``RESTARTS`` − 1 random splits, the best fit kept), and per group
  its size, its coherence (the mean cosine between its documents) and the mean cosine across the groups, each in z units
  of the random pairs. A pole is *mixed* when both groups hold ``MIN_MODE``+ distinct documents (counted down the
  ranking, a document within cosine ``DUP`` of one already counted is a near-copy: templated pages, such as hotel
  listings, otherwise make two tight groups for any token they repeat), each group's coherence is ``W_MIN``+ and the
  groups are no more alike than ``B_MAX`` (random documents are 0): §23's Qwen3-14B L33 n4648+ splits 51/13 (baseball,
  glucose; 39/12 distinct) at 2.0/1.9 z, across 0.1. Per group of a mixed pole, the token it peaks on most and that
  token's share: a *token* mix when either group's share is ``TOKEN_SHARE``+ (a job on a token in each); else a *code*
  mix when ``CODE_HI``+ of one group's documents and ``CODE_LO`` or less of the other's are from the code slice (code
  against prose, which the embedder parts as far as two topics: a list of imports and a list of ingredients make one); else a
  *topic* mix (n4648+: peak shares 0.10/0.23, no code). Groups that peak on the same token are a homonym.
- ``show [n]``: the counts by depth band, and the ``n`` most mixed topic mixes (by the weaker group's coherence less
  the cross-group likeness), then a quarter as many code and token mixes, with three contexts from each group.
- ``history <revs>... [--n n]``: the mixed poles across training checkpoints (``relict.py tops`` at each revision, in
  training order, the final last), against *controls*: poles whose two groups are each as coherent but alike (``cross``
  ≥ ``W_MIN``: one topic split in two), up to ``CONTROLS`` a mixed pole from its layer. Each document a pole holds at a
  checkpoint is its final group *a* or *b* when its likeness to that group's centroid (z over random documents'
  likeness to it) is past that of all but ``FALSE_PASS`` of random documents and beats the other's, else *other* (a
  cutoff at the group's least typical document passed half of random documents for mixed poles, a quarter for
  controls). A group's *onset* is the first checkpoint from which it holds ``MIN_MODE // 2``+ documents at every later
  one; a pole *forms* at the first checkpoint where either group does. A *transient* mode: ``TRANSIENT``+ other
  documents at one checkpoint between formation and the end, coherent (``W_MIN``+) and a mode of its own, not a fringe
  of the final groups (its documents' mean cosine with each group's, in z, at most half that group's coherence).
  Relict generalization predicts mixed poles pass through transients more often than controls do, and that the other
  documents before *b*'s onset are like both groups (a bridge) more than like a matched stranger's *b* (``bridge``: the
  mean, over a pole's other documents before its *b* onset, of the smaller of their two likenesses, less the same with
  another mixed pole's *b* at a similar distance from *a*); a bridge through a shared token, that *b*'s firing tokens
  are *a*'s early on (``shared``) and part later. Per kind and checkpoint, the means of *b*'s share, ``shared`` and the
  other documents' share (these mix cohorts: few poles hold both groups early); the onset order by *b*'s final size, and
  *solo* phases, where one group holds the pole alone before the other arrives (``order``); the bridge tests aligned on
  the later group's onset, each pole against itself at the end (``aligned``); per pole for the ``n`` most mixed: each
  checkpoint's counts, ``act_top`` and ``shared``, and three documents of its largest transient. Saved in
  ``history.pt``.
- ``writes``: at the final checkpoint, whether a mixed pole's write names what follows its *a* firings, its *b*
  firings, or both, against random poles' writes and the controls (``writes``).
- ``panel``, then ``probe <revs>...``, then ``live <revs>...``: whether a pole's later group was there below its top
  documents before its onset. A fixed panel per pole (its own *a* and *b* documents, held-out documents like each, random
  documents), each pole's peak on it at each revision (no ``tops`` needed, so the revisions can be dense:
  download-bound), and per group the revision from which its held-out documents stay above ``LIVE`` of random
  documents, against its onset in the top documents (``live``; the onsets from ``history`` over the same revisions).

Data in ``data/mixed_units/<model>/``, the summary in ``data/summary/mixed_units_<model>.log``.

    NEURON_LENS_MODEL=allenai/Olmo-3-1025-7B python scripts/checks/mixed_units.py embed|scan|show [n]
    NEURON_LENS_MODEL=allenai/Olmo-3-1025-7B python scripts/checks/mixed_units.py history stage1-step1000 ... main
    NEURON_LENS_MODEL=allenai/Olmo-3-1025-7B python scripts/checks/mixed_units.py writes|panel
    NEURON_LENS_MODEL=allenai/Olmo-3-1025-7B python scripts/checks/mixed_units.py probe|live stage1-step1000 ... main
"""
import argparse
import collections
import functools
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import (
    MODEL,
    TOP,
    catalog_dir,
    context,
    documents,
    neuron_table,
    tokenizer,
)
from neuron_lens.paths import DATA, SUMMARY

STEM = arch.stem(MODEL)
OUT = DATA / "mixed_units" / STEM
DEV = "cuda" if torch.cuda.is_available() else "mps"
EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"
FLOOR = 0.05  # act_top of a firing pole (secondary_modes.py's sample)
MIN_MODE, W_MIN, B_MAX, DUP, TOKEN_SHARE = 8, 1.5, 0.5, 0.4, 0.5
CODE_HI, CODE_LO = 0.5, 0.2
KINDS = ("topic", "code", "token")
PAIRS, KMEANS_STEPS, RESTARTS = 200_000, 10, 8
CONTROLS, TRANSIENT, RANDOM, FALSE_PASS = 4, 16, 4096, 0.01
PANEL_OWN, PANEL_HELD, PANEL_RANDOM, PANEL_CONTROLS, LIVE = 12, 12, 1024, 400, 0.8
SETS = ("own_a", "own_b", "held_a", "held_b")
WRITE_RANDOM = 32
STAGE1, STAGE2 = 1_413_814, 47_684  # OLMo 3 7B's stage-1 and stage-2 steps, for ``step``
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


@torch.no_grad()
def encode(texts: list[str], batch: int = 128, max_len: int = 512) -> torch.Tensor:
    """Qwen3-Embedding-0.6B as its card runs it: each text with its end token, left-padded, the last position's state,
    normalised; documents take no instruction."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMBEDDER)
    m = AutoModel.from_pretrained(EMBEDDER, dtype=torch.bfloat16).to(DEV).eval()
    ids = [x[:max_len - 1] for x in tok(texts, add_special_tokens=False)["input_ids"]]
    ids = [x + [tok.eos_token_id] for x in ids]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = torch.empty(len(ids), m.config.hidden_size, dtype=torch.float16)
    for k in range(0, len(order), batch):
        b = order[k:k + batch]
        n = max(len(ids[i]) for i in b)
        x = torch.tensor([[tok.pad_token_id] * (n - len(ids[i])) + ids[i] for i in b], device=DEV)
        mask = torch.tensor([[0] * (n - len(ids[i])) + [1] * len(ids[i]) for i in b], device=DEV)
        h = m(input_ids=x, attention_mask=mask).last_hidden_state[:, -1]
        out[b] = torch.nn.functional.normalize(h.float(), dim=-1).half().cpu()
    return out


def embed() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok, docs, t0 = tokenizer(), documents(), time.time()
    E = encode([tok.decode(d.tolist()) for d in docs])
    torch.save({"emb": E}, OUT / "docs.pt")
    _, mu, sd = load_docs()
    say(f"{len(E):,} documents embedded in {time.time() - t0:.0f} s; centred, random pairs' cosine {mu:.3f} ± {sd:.3f}")


def load_docs() -> tuple[torch.Tensor, float, float]:
    """The documents' embeddings centred on their mean and renormalised, and random pairs' cosine (mean, sd)."""
    E = torch.load(OUT / "docs.pt")["emb"].float()
    E = torch.nn.functional.normalize(E - E.mean(0), dim=-1)
    a, b = torch.randint(len(E), (2, PAIRS), generator=torch.Generator().manual_seed(0))
    cos = (E[a[a != b]] * E[b[a != b]]).sum(1)
    return E, cos.mean().item(), cos.std().item()


def split(X: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """``[P, K]`` bool: each pole's two groups of documents (``X`` ``[P, K, d]`` normalised; ``valid`` the real ones):
    spherical 2-means from the first principal component's sign and from random splits, the best fit (the summed
    cosine of each document to its group's centroid) kept."""
    w = valid.float().unsqueeze(-1)
    mean = (X * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1)
    Xc = (X - mean) * w
    gen = torch.Generator().manual_seed(0)
    starts = [torch.linalg.eigh(Xc @ Xc.transpose(1, 2))[1][..., -1] > 0]
    starts += [(torch.rand(valid.shape, generator=gen) > 0.5).to(X.device) for _ in range(RESTARTS - 1)]
    best = fit = None
    for g in starts:
        for _ in range(KMEANS_STEPS):
            c = torch.stack([(X * ((g == s) & valid).float().unsqueeze(-1)).sum(1) for s in (True, False)], 1)
            sim = X @ torch.nn.functional.normalize(c, dim=-1).transpose(1, 2)
            g = sim.argmax(-1) == 0
        f = (sim.amax(-1) * valid).sum(1)
        if best is None:
            best, fit = g, f
        else:
            better = f > fit
            best, fit = torch.where(better.unsqueeze(1), g, best), torch.maximum(f, fit)
    return best & valid


def stats(X: torch.Tensor, valid: torch.Tensor, g: torch.Tensor) -> dict:
    """Per pole: the groups' sizes, coherences (the mean cosine within) and the mean cosine across, the larger group
    first (``group``: True for its documents)."""
    S = X @ X.transpose(1, 2)
    eye = torch.eye(X.shape[1], dtype=torch.bool, device=X.device)
    b = ~g & valid
    swap = b.sum(1) > g.sum(1)
    a, b = torch.where(swap.unsqueeze(1), b, g), torch.where(swap.unsqueeze(1), g, b)
    mean = lambda m: (S * m).sum((1, 2)) / m.sum((1, 2)).clamp_min(1)
    pair = lambda m: m.unsqueeze(2) & m.unsqueeze(1) & ~eye
    keep = torch.zeros_like(valid)  # distinct documents, down the ranking
    for j in range(X.shape[1]):
        keep[:, j] = valid[:, j] & ~(keep[:, :j] & (S[:, :j, j] >= DUP)).any(1)
    return {"n_a": a.sum(1), "n_b": b.sum(1), "d_a": (keep & a).sum(1), "d_b": (keep & b).sum(1),
            "w_a": mean(pair(a)), "w_b": mean(pair(b)), "cross": mean(a.unsqueeze(2) & b.unsqueeze(1)), "group": a}


@torch.no_grad()
def scan() -> None:
    from neuron_lens.catalog import firing_tokens, flat_tokens
    flat, start = flat_tokens()
    E, mu, sd = load_docs()
    E = E.to(DEV)
    out, t0 = [], time.time()
    for layer in range(arch.text_config(MODEL).num_hidden_layers):
        t = pq.read_table(neuron_table("acts", layer), columns=["neuron", "pole", "act_top", "top_doc", "top_pos"])
        act_top = torch.tensor(t["act_top"].to_numpy(zero_copy_only=False))
        grid = lambda c: torch.tensor(t[c].combine_chunks().flatten().to_numpy().reshape(len(t), -1)[:, :TOP]).long()
        doc, pos = grid("top_doc"), grid("top_pos")
        keep = (act_top >= FLOOR).nonzero().squeeze(1)
        acc = collections.defaultdict(list)
        for c in keep.split(4096):
            d = doc[c].to(DEV)
            valid = d >= 0
            X = E[d.clamp_min(0)] * valid.unsqueeze(-1)
            for k, v in stats(X, valid, split(X, valid)).items():
                acc[k].append(v.cpu())
        r = {k: torch.cat(v) for k, v in acc.items()}
        for k in ("w_a", "w_b", "cross"):
            r[k] = ((r[k] - mu) / sd).half()
        r |= {"layer": layer, "neuron": torch.tensor(t["neuron"].to_numpy())[keep],
              "pole": torch.tensor(t["pole"].to_numpy())[keep], "act_top": act_top[keep], "doc": doc[keep], "pos": pos[keep],
              "poles": len(t)}
        for k in ("share_a", "share_b"):
            r[k] = torch.full((len(keep),), float("nan"))
        for k in ("tok_a", "tok_b"):
            r[k] = torch.full((len(keep),), -1)
        for i in mixed(r).nonzero().squeeze(1).tolist():
            v = r["doc"][i] >= 0
            peaks = firing_tokens(flat, start, r["doc"][i][v], r["pos"][i][v])
            for side, sel in (("a", r["group"][i][v]), ("b", ~r["group"][i][v])):
                r["tok_" + side][i], r["share_" + side][i] = peak_token(peaks[sel])
        out.append(r)
        say(f"L{layer}: {len(keep):,} of {len(t):,} poles fire, {int(mixed(r).sum()):,} mixed; {time.time() - t0:.0f} s")
    torch.save({"layers": out, "mu": mu, "sd": sd, "floor": FLOOR}, OUT / "scan.pt")


def mixed(r: dict) -> torch.Tensor:
    return ((torch.minimum(r["d_a"], r["d_b"]) >= MIN_MODE) & (torch.minimum(r["w_a"], r["w_b"]) >= W_MIN)
            & (r["cross"] <= B_MAX))


def kind(r: dict, i: int) -> str:
    """A mixed pole's kind: a job on one token in either group (``token``), code against prose (``code``), or a
    ``topic`` mix."""
    if max(float(r["share_a"][i]), float(r["share_b"][i])) >= TOKEN_SHARE:
        return "token"
    v = r["doc"][i] >= 0
    c, g = code_docs()[r["doc"][i][v]].float(), r["group"][i][v]
    lo, hi = sorted((float(c[g].mean()), float(c[~g].mean())))
    return "code" if hi >= CODE_HI and lo <= CODE_LO else "topic"


@functools.cache
def code_docs() -> torch.Tensor:
    """Whether each activation document is from the code slice (``docs.parquet``)."""
    sl = pq.read_table(catalog_dir(MODEL) / "docs.parquet", columns=["slice"])["slice"].to_pylist()
    return torch.tensor([s == "code" for s in sl])


def peak_token(ids: torch.Tensor) -> tuple[int, float]:
    """The most common token among a group's peaks, and its share."""
    if len(ids) == 0:
        return -1, 0.0
    t, c = collections.Counter(ids.tolist()).most_common(1)[0]
    return t, c / len(ids)


def show(n: int) -> None:
    S = torch.load(OUT / "scan.pt", weights_only=False)
    tok, docs = tokenizer(), documents()
    say(f"{MODEL}: poles with act_top ≥ {S['floor']}; mixed: both groups {MIN_MODE}+ distinct of {TOP} documents, each "
        f"coherent ≥ {W_MIN} z, across ≤ {B_MAX} z (z over random documents' cosine, {S['mu']:.3f} ± {S['sd']:.3f}); a "
        f"token mix peaks on one token in {TOKEN_SHARE:.0%}+ of a group")
    cands = [{"layer": r["layer"], "i": i, "kind": kind(r, i), "homonym": bool(r["tok_a"][i] == r["tok_b"][i]),
              "score": float(torch.minimum(r["w_a"][i], r["w_b"][i]) - r["cross"][i])}
             for r in S["layers"] for i in mixed(r).nonzero().squeeze(1).tolist()]
    for name, (lo, hi) in arch.depth_bands(MODEL).items():
        fire = sum(len(r["act_top"]) for r in S["layers"] if lo <= r["layer"] <= hi)
        m = [c for c in cands if lo <= c["layer"] <= hi]
        say(f"  {name:12s} L{lo}-L{hi}: {len(m):,} of {fire:,} firing poles mixed ({len(m) / max(1, fire):.2%}): "
            + ", ".join(f"{sum(c['kind'] == k for c in m):,} {k}" for k in KINDS) + " mixes, "
            f"{sum(c['homonym'] for c in m):,} peak on one token in both groups")
    ranked = sorted(cands, key=lambda c: -c["score"])
    ranked = [c for k in KINDS for c in [c for c in ranked if c["kind"] == k][:n if k == "topic" else n // 4]]
    say(f"\nthe most mixed (the weaker group's coherence less the cross likeness): {n} topic mixes, then {n // 4} code "
        f"and {n // 4} token mixes")
    for c in ranked:
        r, i = S["layers"][c["layer"]], c["i"]
        sign = "+" if r["pole"][i] > 0 else "−"
        say(f"\nL{c['layer']} n{int(r['neuron'][i])}{sign} act_top {float(r['act_top'][i]):.2f}: groups "
            f"{int(r['n_a'][i])}/{int(r['n_b'][i])} ({int(r['d_a'][i])}/{int(r['d_b'][i])} distinct), coherence "
            f"{float(r['w_a'][i]):.1f}/{float(r['w_b'][i]):.1f} z, across {float(r['cross'][i]):.1f} z; peaks "
            f"{tok.decode([int(r['tok_a'][i])])!r} {float(r['share_a'][i]):.0%} / {tok.decode([int(r['tok_b'][i])])!r} "
            f"{float(r['share_b'][i]):.0%}")
        for side, flag in (("a", True), ("b", False)):
            ctx = [(int(d), int(q)) for d, q, g in zip(r["doc"][i], r["pos"][i], r["group"][i]) if d >= 0 and bool(g) == flag]
            for d, q in ctx[:3]:
                say(f"   {side}: {context(tok, docs[d].tolist(), q)}")
    (SUMMARY / f"mixed_units_{STEM}.log").write_text("\n".join(LOG) + "\n")


def centroid(X: torch.Tensor, R: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Documents ``X``' normalised centroid, and random documents' (``R``) likeness to it, mean and sd: the scale
    of ``like``."""
    c = torch.nn.functional.normalize(X.sum(0), dim=0)
    r = R @ c
    return c, float(r.mean()), float(r.std())


def like(X: torch.Tensor, c: tuple) -> torch.Tensor:
    """Each document's likeness to centroid ``c`` (``centroid``'s), z over random documents'."""
    return (X @ c[0] - c[1]) / c[2]


def coherence(X: torch.Tensor, mu: float, sd: float) -> float:
    """The mean cosine between documents ``X`` ``[n, d]``, in z units of random pairs."""
    n = len(X)
    return float((((X @ X.T).sum() - n) / (n * (n - 1)) - mu) / sd) if n > 1 else float("nan")


SIZES = ((8, 23), (24, 64))  # b's final size, for the order by size
SOLO, NEAR, OFFSETS = 8, 0.9, range(-8, 9)


def order(units: list[dict], kinds: tuple) -> None:
    """Onset order by *b*'s final size (*b* is the smaller group, so it reaches ``MIN_MODE // 2`` later by size alone),
    and *solo* phases: a checkpoint before the other group's onset where one group holds ``SOLO``+ documents and the
    other one or none."""
    say("\nonset order by b's final size (b before a / together / after; by 4+ checkpoints), and a solo phase before the "
        f"other's onset ({SOLO}+ documents, the other 1 or none)")
    last = len(units[0]["hist"]) - 1
    for k_ in kinds:
        for lo, hi in SIZES:
            us = [u for u in units if u["kind"] == k_ and u["on_a"] is not None and u["on_b"] is not None
                  and lo <= len(u["docs_b"]) <= hi]
            if len(us) < 5:
                continue
            g = torch.tensor([u["on_b"] - u["on_a"] for u in us], dtype=torch.float)
            solo = lambda s, o, u: any(u["hist"][k][s] >= SOLO and u["hist"][k][o] <= 1 for k in range(min(u["on_" + o], last)))
            say(f"  {k_:8s} n_b {lo}-{hi}: {len(us):5d} poles; {(g < 0).float().mean():.2f} / {(g == 0).float().mean():.2f} "
                f"/ {(g > 0).float().mean():.2f} (by 4+: {(g <= -4).float().mean():.2f} / {(g >= 4).float().mean():.2f}); "
                f"b solo first {sum(solo('b', 'a', u) for u in us) / len(us):.2f}, a solo first "
                f"{sum(solo('a', 'b', u) for u in us) / len(us):.2f}")


def letters(tok, i: int, cache: dict) -> str:
    if i not in cache:
        cache[i] = "".join(c for c in tok.decode([i]).lower() if c.isalpha())
    return cache[i]


def string_share(ts: torch.Tensor, tr: torch.Tensor, tok, cache: dict) -> float:
    """The share of firing tokens ``ts`` sharing four letters in a row with one of ``tr`` that is another token."""
    words = {letters(tok, int(x), cache) for x in tr.tolist()}
    hit = []
    for x in ts.tolist():
        w = letters(tok, int(x), cache)
        hit.append(len(w) >= 4 and any(len(v) >= 4 and v != w and any(w[j:j + 4] in v for j in range(len(w) - 3))
                                       for v in words))
    return sum(hit) / len(hit)


def aligned(mixed_u: list[dict], revs: list[str], E: torch.Tensor, R: torch.Tensor) -> None:
    """The bridge tests aligned on the later group's onset. The *root* is the group with the earlier onset, the
    *second* the other. Per checkpoint and pole:
    - ``near``: of the documents it holds that are like the second (past random documents' ``NEAR`` quantile), the share
      also like the root; the same for a stranger's second group at the root's distance (a pole's own root documents
      spill into any centroid near it, more while the root holds the pole alone);
    - ``tok``: the share of the second's firing tokens among the root's (both 4+), and ``str``: that share sharing four
      letters with a root firing token (mongo, Mongolia), each as a cohort, before the onset against the end;
    - ``drift``: the second's documents' mean likeness to the root. A weak mode reaches a pole's top documents only where
      some of the other mode's features help it, so a growing mode drifts away from the root with no chain."""
    half, nan, tok, cache = MIN_MODE // 2, float("nan"), tokenizer(), {}
    us = [u for u in mixed_u if u["on_a"] is not None and u["on_b"] is not None and u["on_a"] != u["on_b"]]
    if len(us) < 2:
        return
    for u in us:
        u["root"], u["sec"] = ("a", "b") if u["on_a"] < u["on_b"] else ("b", "a")
        for s in "ab":
            u["q_" + s] = float(like(R, u[s]).quantile(NEAR))
    C = {s: torch.stack([u[s][0] for u in us]) for s in "ab"}
    for j, u in enumerate(us):
        Rt, Sc = C[u["root"]], C[u["sec"]]
        d = (Rt[j] @ Sc.T - Rt[j] @ Sc[j]).abs()
        d[j] = float("inf")
        u["c_s"] = us[int(d.argmin())][u["sec"]]
        u["q_s"] = float(like(R, u["c_s"]).quantile(NEAR))
        u["al"] = []
        for x in u["hist"]:
            z = {"a": x["za"], "b": x["zb"], "s": like(E[x["doc"]], u["c_s"])}
            rl = z[u["root"]] >= u["q_" + u["root"]]
            sl, ss = z[u["sec"]] >= u["q_" + u["sec"]], z["s"] >= u["q_s"]
            ir, is_ = x["is_" + u["root"]], x["is_" + u["sec"]]
            tr, ts = x["tk"][ir], x["tk"][is_]
            both4 = min(len(tr), len(ts)) >= half
            u["al"].append({"n_sec": int(sl.sum()), "both": int((sl & rl).sum()), "n_s": int(ss.sum()),
                            "both_s": int((ss & rl).sum()),
                            "tok": float(torch.isin(ts, tr).float().mean()) if both4 else nan,
                            "str": string_share(ts, tr, tok, cache) if both4 else nan,
                            "drift": float(z[u["root"]][is_].mean()) if len(ts) >= half else nan})
    K = len(revs)
    at = lambda u, o: u["al"][u["on_" + u["sec"]] + o] if 0 <= u["on_" + u["sec"]] + o < K else None
    say(f"\naligned on the second group's onset (checkpoint offsets): near, of the documents like the second (past "
        f"random's q{NEAR * 100:.0f}) the share also like the root, own / a stranger's second at the root's distance; "
        f"the cohorts compare a pole before the onset with the end")
    for k_ in KINDS:
        s_ = [u for u in us if u["kind"] == k_]
        if not s_:
            continue
        say(f"  {k_} ({len(s_)} poles; root a {sum(u['root'] == 'a' for u in s_)}, root b {sum(u['root'] == 'b' for u in s_)})")
        for name, num, den in (("near own", "both", "n_sec"), ("near str", "both_s", "n_s")):
            row = []
            for o in OFFSETS:
                p = torch.tensor([x[num] / x[den] for u in s_ if (x := at(u, o)) and x[den] >= half])
                row.append(f"{o:+d}:{p.mean():.2f}" if len(p) else f"{o:+d}:-")
            say(f"    {name:9s} " + " ".join(row))
        for name, before in (("tok", (-3, -2)), ("str", (-3, -2)), ("drift", (0, 1))):
            e, z = [], []
            for u in s_:
                b = [x[name] for o in before if (x := at(u, o)) and x[name] == x[name]]
                if b and u["al"][-1][name] == u["al"][-1][name] and (name != "drift" or u["on_" + u["sec"]] + 1 < K - 1):
                    e.append(sum(b) / len(b))
                    z.append(u["al"][-1][name])
            if len(e) > 1:
                dd = torch.tensor(e) - torch.tensor(z)
                say(f"    {name:9s} cohort {len(e)}: {'at the onset' if name == 'drift' else 'before the onset'} "
                    f"{sum(e) / len(e):.3f}, at the end {sum(z) / len(z):.3f}, paired {dd.mean():+.3f} ± "
                    f"{dd.std() / len(dd) ** 0.5:.3f}")
        c = torch.bincount(torch.tensor([u["on_" + u["sec"]] for u in s_]), minlength=K)
        say("    second's onsets: " + " ".join(f"{rev.replace('stage1-step', '')}:{int(c[j])}" for j, rev in enumerate(revs) if c[j]))


@torch.no_grad()
def history(revs: list[str], n: int) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    import relict

    from neuron_lens.catalog import firing_tokens, flat_tokens
    S = torch.load(OUT / "scan.pt", weights_only=False)
    E, mu, sd = load_docs()
    flat, start = flat_tokens()
    g = torch.Generator().manual_seed(0)
    R = E[torch.randint(len(E), (RANDOM,), generator=g)]
    f = arch.text_config(MODEL).intermediate_size
    half, nan = MIN_MODE // 2, float("nan")
    units = []
    for r in S["layers"]:
        m = mixed(r).nonzero().squeeze(1)
        alike = ((torch.minimum(r["d_a"], r["d_b"]) >= MIN_MODE) & (torch.minimum(r["w_a"], r["w_b"]) >= W_MIN)
                 & (r["cross"] >= W_MIN)).nonzero().squeeze(1)
        alike = alike[torch.randperm(len(alike), generator=g)[:CONTROLS * len(m)]]
        for ctl, idx in ((False, m), (True, alike)):
            for i in idx.tolist():
                v = r["doc"][i] >= 0
                u = {"kind": "control" if ctl else kind(r, i), "layer": r["layer"], "neuron": int(r["neuron"][i]),
                     "pole": int(r["pole"][i]), "w_a": float(r["w_a"][i]), "w_b": float(r["w_b"][i]),
                     "score": float(torch.minimum(r["w_a"][i], r["w_b"][i]) - r["cross"][i])}
                for side, sel in (("a", r["group"][i] & v), ("b", ~r["group"][i] & v)):
                    u["docs_" + side] = r["doc"][i][sel]
                    u[side] = centroid(E[u["docs_" + side]], R)
                    u["tau_" + side] = float(like(R, u[side]).quantile(1 - FALSE_PASS))
                units.append(u)
    say(f"{MODEL}: " + ", ".join(f"{sum(u['kind'] == k for u in units)} {k}" for k in KINDS) + " mixes, "
        f"{sum(u['kind'] == 'control' for u in units)} controls, over {len(revs)} checkpoints ({', '.join(revs)})")
    for u in units:
        u["hist"] = []
    for rev in revs:
        t = torch.load(relict.OUT / f"tops-{rev}.pt")
        for u in units:
            c = u["neuron"] + f * (u["pole"] < 0)
            doc, val, pos = t["doc"][u["layer"], :, c], t["val"][u["layer"], :, c].float(), t["pos"][u["layer"], :, c]
            ok = (doc >= 0) & (val > 0)
            doc, pos = doc[ok].long(), pos[ok].long()
            X = E[doc]
            za, zb = like(X, u["a"]), like(X, u["b"])
            is_a = (za >= u["tau_a"]) & (za >= zb)
            is_b = (zb >= u["tau_b"]) & (zb > za) & ~is_a
            o = ~is_a & ~is_b
            tk = firing_tokens(flat, start, doc, pos)
            ta, tb = tk[is_a], tk[is_b]
            x = {"a": int(is_a.sum()), "b": int(is_b.sum()), "o": int(o.sum()),
                 "act_top": float(t["act_top"][u["layer"], c]), "coh_o": coherence(X[o], mu, sd),
                 "shared": float(torch.isin(tb, ta).float().mean()) if min(len(ta), len(tb)) >= half else nan,
                 "o_doc": doc[o], "o_pos": pos[o], "o_za": za[o], "o_zb": zb[o],
                 "doc": doc, "za": za, "zb": zb, "tk": tk, "is_a": is_a, "is_b": is_b}
            if x["o"] >= TRANSIENT and x["coh_o"] >= W_MIN:
                x["cross_a"], x["cross_b"] = (float(((X[o] @ E[u["docs_" + s]].T).mean() - mu) / sd) for s in "ab")
            u["hist"].append(x)
        say(f"  {rev}: read")
        del t
    last = len(revs) - 1

    def onset(h, side):
        k = last
        while k > 0 and h[k - 1][side] >= half:
            k -= 1
        return k if h[last][side] >= half else None

    for u in units:
        h = u["hist"]
        u["on_a"], u["on_b"] = onset(h, "a"), onset(h, "b")
        u["form"] = next((k for k, x in enumerate(h) if x["a"] >= half or x["b"] >= half), last)
        u["transient"] = [k for k in range(u["form"], last) if "cross_a" in h[k]
                          and h[k]["cross_a"] <= u["w_a"] / 2 and h[k]["cross_b"] <= u["w_b"] / 2]
    mixed_u = [u for u in units if u["kind"] != "control"]
    if len(mixed_u) > 1:  # a stranger's b at a similar distance from a, for the bridge null
        A = torch.stack([u["a"][0] for u in mixed_u])
        Bc = torch.stack([u["b"][0] for u in mixed_u])
        d = A @ Bc.T  # [own a, other's b]
        d = (d - d.diagonal()[:, None]).abs().fill_diagonal_(float("inf"))
        for u, j in zip(mixed_u, d.argmin(1).tolist()):
            u["stranger"] = mixed_u[j]
    kinds = (*KINDS, "control")
    say(f"\nat {revs[-1]}, the share of each final group that passes as its own (a / b, median), against {FALSE_PASS:.0%} "
        f"of random documents by construction")
    for k_ in kinds:
        us = [u for u in units if u["kind"] == k_]
        if us:
            ra = torch.tensor([u["hist"][last]["a"] / len(u["docs_a"]) for u in us]).median()
            rb = torch.tensor([u["hist"][last]["b"] / len(u["docs_b"]) for u in us]).median()
            say(f"  {k_:8s} {len(us):5d} poles: {ra:.2f} / {rb:.2f}")
    say("\nper checkpoint, the mean over poles of b's share of a + b (where 4+), the share of b's firing tokens among "
        "a's (where both hold 4+) and the other documents' share")
    for k_ in kinds:
        us = [u for u in units if u["kind"] == k_]
        for name, fn in (("b share", lambda x: x["b"] / (x["a"] + x["b"]) if x["a"] + x["b"] >= half else nan),
                         ("shared", lambda x: x["shared"]),
                         ("other", lambda x: x["o"] / max(1, x["a"] + x["b"] + x["o"]))):
            v = torch.tensor([[fn(x) for x in u["hist"]] for u in us]) if us else torch.zeros(0, len(revs))
            say(f"  {k_:8s} {name:8s} " + " ".join(f"{x:.2f}" for x in v.nanmean(0).tolist())
                + ("" if name == "other" else "   (n " + " ".join(str(int(c)) for c in (~v.isnan()).sum(0)) + ")"))
    say("\nonsets (checkpoint indices), formation and transients")
    for k_ in kinds:
        us = [u for u in units if u["kind"] == k_ and u["on_a"] is not None and u["on_b"] is not None]
        if not us:
            continue
        gap = torch.tensor([u["on_b"] - u["on_a"] for u in us], dtype=torch.float)
        born = sum(u["on_b"] <= u["form"] for u in us) / len(us)
        tr = torch.tensor([bool(u["transient"]) for u in us], dtype=torch.float)
        cr = [(u["hist"][k]["cross_a"], u["hist"][k]["cross_b"]) for u in us for k in u["transient"]]
        crs = f"; transients' likeness to a / b {sum(c[0] for c in cr) / len(cr):.2f} / {sum(c[1] for c in cr) / len(cr):.2f} z" if cr else ""
        say(f"  {k_:8s} {len(us):5d} poles with both onsets: b before a {(gap < 0).float().mean():.2f}, together "
            f"{(gap == 0).float().mean():.2f}, after {(gap > 0).float().mean():.2f} (median gap {gap.median():.0f}); "
            f"b there at formation {born:.2f}; a transient {tr.mean():.3f} ± {tr.std() / len(tr) ** 0.5:.3f}{crs}")
    br, adj, bk = [], [], []
    for u in mixed_u:
        if u["on_b"] is None or "stranger" not in u:
            continue
        ks = range(u["form"], u["on_b"])
        x = [(u["hist"][k]["o_doc"], u["hist"][k]["o_za"], u["hist"][k]["o_zb"]) for k in ks if u["hist"][k]["o"]]
        if not x:
            continue
        docs_ = torch.cat([d for d, _, _ in x])
        za, zb = torch.cat([a for _, a, _ in x]), torch.cat([b for _, _, b in x])
        zs = like(E[docs_], u["stranger"]["b"])
        br.append(float(torch.minimum(za, zb).mean() - torch.minimum(za, zs).mean()))
        adj.append(float(zb.mean() - zs.mean()))
        bk.append(u["kind"])
    for k_ in KINDS:
        sel = [j for j, x in enumerate(bk) if x == k_]
        if len(sel) > 1:
            b_, a_ = torch.tensor(br)[sel], torch.tensor(adj)[sel]
            say(f"  bridge, {k_}: {len(sel)} poles with other documents before b's onset: min(z_a, z_b) less the "
                f"stranger's {b_.mean():+.3f} ± {b_.std() / len(b_) ** 0.5:.3f}; z_b less the stranger's "
                f"{a_.mean():+.3f} ± {a_.std() / len(a_) ** 0.5:.3f}")
    order(units, kinds)
    aligned(mixed_u, revs, E, R)
    tok, docs = tokenizer(), documents()
    say(f"\nthe {n} most mixed: per checkpoint a/b/other documents (of {TOP}), act_top; t marks a transient; then the "
        f"share of b's firing tokens among a's")
    ranked = sorted(mixed_u, key=lambda u: -u["score"])
    for u in [u for k in KINDS for u in [u for u in ranked if u["kind"] == k][:n if k == "topic" else n // 4]]:
        sign = "+" if u["pole"] > 0 else "−"
        say(f"\nL{u['layer']} n{u['neuron']}{sign} ({u['kind']} mix): onsets a {u['on_a']} b {u['on_b']}, forms {u['form']}")
        say("   " + "  ".join(f"{rev}:{x['a']}/{x['b']}/{x['o']}{'t' if k in u['transient'] else ''} {x['act_top']:.3g}"
                              for k, (rev, x) in enumerate(zip(revs, u["hist"]))))
        say("   shared: " + " ".join(f"{x['shared']:.2f}" for x in u["hist"]))
        if u["transient"]:
            k = max(u["transient"], key=lambda k: u["hist"][k]["o"])
            x = u["hist"][k]
            say(f"   transient at {revs[k]}: {x['o']} documents, coherence {x['coh_o']:.1f} z, like a {x['cross_a']:.1f} z, "
                f"like b {x['cross_b']:.1f} z")
            for d, q in list(zip(x["o_doc"].tolist(), x["o_pos"].tolist()))[:3]:
                say(f"   o: {context(tok, docs[d].tolist(), q)}")
    keep = ("kind", "layer", "neuron", "pole", "score", "w_a", "w_b", "on_a", "on_b", "form", "transient")
    torch.save({"revs": revs, "units": [
        {k: u[k] for k in keep} | {"n_a": len(u["docs_a"]), "n_b": len(u["docs_b"])}
        | {k: [x[k] for x in u["hist"]] for k in ("a", "b", "o", "shared", "act_top")}
        | ({"o_doc": [x["o_doc"] for x in u["hist"]]} if u["kind"] != "control" else {}) for u in units]},
        OUT / "history.pt")
    (SUMMARY / f"mixed_units_history_{STEM}.log").write_text("\n".join(LOG) + "\n")


def panel() -> None:
    """The probe's fixed panel: every mixed pole and ``PANEL_CONTROLS`` controls, each with its final *a* and *b*
    documents (up to ``PANEL_OWN`` each), held-out documents like each group (the ``PANEL_HELD`` nearest its centroid
    outside the pole's top documents and short of the other group's cutoff), and ``PANEL_RANDOM`` random documents."""
    S = torch.load(OUT / "scan.pt", weights_only=False)
    H = torch.load(OUT / "history.pt", weights_only=False)
    E, _, _ = load_docs()
    R = E[torch.randint(len(E), (RANDOM,), generator=torch.Generator().manual_seed(0))]
    L_ = {r["layer"]: r for r in S["layers"]}
    mixed_u = [u for u in H["units"] if u["kind"] != "control"]
    ctl = [u for u in H["units"] if u["kind"] == "control"]
    ctl = [ctl[i] for i in torch.randperm(len(ctl), generator=torch.Generator().manual_seed(2))[:PANEL_CONTROLS].tolist()]
    units = []
    for u in mixed_u + ctl:
        r = L_[u["layer"]]
        i = int(((r["neuron"] == u["neuron"]) & (r["pole"] == u["pole"])).nonzero()[0, 0])
        v = r["doc"][i] >= 0
        c, tau, x = {}, {}, {k: u[k] for k in ("kind", "layer", "neuron", "pole")}
        for side, sel in (("a", r["group"][i] & v), ("b", ~r["group"][i] & v)):
            c[side] = centroid(E[r["doc"][i][sel]], R)
            tau[side] = float(like(R, c[side]).quantile(1 - FALSE_PASS))
            x["own_" + side] = r["doc"][i][sel][:PANEL_OWN].tolist()
        for side, other in (("a", "b"), ("b", "a")):
            z = like(E, c[side])
            z[like(E, c[other]) >= tau[other]] = -torch.inf
            z[r["doc"][i][v].long()] = -torch.inf
            h = z.topk(PANEL_HELD)
            x["held_" + side], x["held_past_" + side] = h.indices.tolist(), float(h.values.median()) - tau[side]
        units.append(x)
    rand = torch.randperm(len(E), generator=torch.Generator().manual_seed(1))[:PANEL_RANDOM].tolist()
    ids = sorted(set(rand) | {d for x in units for s in SETS for d in x[s]})
    torch.save({"units": units, "rand": rand, "ids": ids}, OUT / "panel.pt")
    past = torch.tensor([[x["held_past_a"], x["held_past_b"]] for x in units])
    say(f"{MODEL}: {len(units)} poles ({len(mixed_u)} mixed, {len(ctl)} controls) on {len(ids):,} documents; held-out "
        f"documents' median likeness past their group's cutoff a {past[:, 0].median():+.2f}, b {past[:, 1].median():+.2f} z")


@torch.inference_mode()
def probe(revs: list[str]) -> None:
    """Each panel pole's peak (its sign, positions from 1) on each panel document at each revision, run as whole
    documents. The revisions need no ``tops``, so they can be dense (download-bound)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import relict
    P = torch.load(OUT / "panel.pt")
    ids, docs = P["ids"], documents()
    T = [(x["layer"], x["neuron"], x["pole"]) for x in P["units"]]
    for rev in revs:
        path = OUT / f"probe-{rev}.pt"
        if path.exists():
            continue
        stack, t0 = relict.load(rev), time.time()
        cap, hs = {}, []
        for l in sorted({l for l, _, _ in T}):
            js = [j for j, (l2, _, _) in enumerate(T) if l2 == l]
            ns = torch.tensor([T[j][1] for j in js], device=DEV)
            sg = torch.tensor([float(T[j][2]) for j in js], device=DEV)
            hs.append(relict._down(stack.layers[l].mlp).register_forward_pre_hook(
                lambda m, a, js=js, ns=ns, sg=sg: cap.__setitem__(js[0], (js, a[0][..., ns].float() * sg))))
        peak = torch.empty(len(ids), len(T))
        for s in range(0, len(ids), relict.BATCH):
            b = [docs[d] for d in ids[s:s + relict.BATCH]]
            n = max(map(len, b))
            x = torch.zeros(len(b), n, dtype=torch.long)
            mask = torch.zeros(len(b), n, dtype=torch.bool)
            for k, d in enumerate(b):
                x[k, :len(d)], mask[k, :len(d)] = d, True
            cap.clear()
            stack(input_ids=x.to(DEV), attention_mask=mask.long().to(DEV))
            m = mask.to(DEV).clone()
            m[:, 0] = False
            for js, a in cap.values():
                peak[s:s + len(b), js] = a.masked_fill(~m[..., None], -torch.inf).amax(1).cpu()
        for h in hs:
            h.remove()
        torch.save({"ids": ids, "targets": T, "peak": peak.half()}, path)
        say(f"  {rev}: {time.time() - t0:.0f} s")
        del stack
        torch.cuda.empty_cache() if DEV == "cuda" else None


def step(rev: str) -> int:
    """Training steps seen at a revision (stage 2 after stage 1's last, the final model after both)."""
    if rev == "main":
        return STAGE1 + STAGE2 + 1
    n = int(rev.split("step")[1])
    return n if rev.startswith("stage1") else STAGE1 + n


def live(revs: list[str]) -> None:
    """Whether a pole's later group was there below its top documents before its onset (two responses of unequal
    strength from the start) or absent until it joined. Per pole, revision and panel set: the set's median peak as a
    percentile of the pole's random documents' peaks. A group is *live* from the first revision from which its held-out
    documents' percentile stays ``LIVE``+; its onset (``history``'s, over the same revisions) needs its documents in the
    pole's top ``TOP``."""
    P = torch.load(OUT / "panel.pt")
    H = torch.load(OUT / "history.pt", weights_only=False)
    hrev = H["revs"]
    on = {(u["layer"], u["neuron"], u["pole"]): u for u in H["units"]}
    revs = [r for r in hrev if r in revs]
    K = len(revs)
    pct = {s: torch.full((len(P["units"]), K), float("nan")) for s in SETS}
    for k, rev in enumerate(revs):
        d = torch.load(OUT / f"probe-{rev}.pt")
        row = {x: j for j, x in enumerate(d["ids"])}
        pk = d["peak"].float()
        rnd = pk[[row[x] for x in P["rand"]]].sort(0).values
        for s in SETS:
            med = torch.stack([pk[[row[q] for q in x[s]], j].median() for j, x in enumerate(P["units"])])
            pct[s][:, k] = torch.searchsorted(rnd.T.contiguous(), med[:, None]).squeeze(1).float() / len(rnd)
    rows = []
    for j, x in enumerate(P["units"]):
        u = on[(x["layer"], x["neuron"], x["pole"])]
        if u["on_a"] is None or u["on_b"] is None or u["on_a"] == u["on_b"]:
            continue
        ia, ib = revs.index(hrev[u["on_a"]]), revs.index(hrev[u["on_b"]])
        rt, sc = ("a", "b") if ia < ib else ("b", "a")
        first = lambda h: next((k for k in range(K) if (h[k:] >= LIVE).all()), K)
        rows.append({"kind": x["kind"], "j": j, "rt": rt, "sc": sc, "on_rt": min(ia, ib), "on_sc": max(ia, ib),
                     "live_rt": first(pct["held_" + rt][j]), "live_sc": first(pct["held_" + sc][j]),
                     "form": revs.index(hrev[u["form"]])})
    say(f"{MODEL}: {len(rows)} panel poles with ordered onsets over {K} revisions ({revs[0]} to {revs[-1]}); a set is "
        f"live where its median peak passes {LIVE:.0%} of random documents' peaks")
    for k_ in (*KINDS, "control"):
        rs = [r for r in rows if r["kind"] == k_]
        if not rs:
            continue
        say(f"\n{k_} ({len(rs)} poles): mean percentile, aligned on the later group's onset (revision offsets)")
        for name, s, side in (("later held", "held", "sc"), ("later own", "own", "sc"), ("earlier held", "held", "rt"),
                              ("earlier own", "own", "rt")):
            line = []
            for o in OFFSETS:
                v = torch.tensor([float(pct[s + "_" + r[side]][r["j"], r["on_sc"] + o]) for r in rs if 0 <= r["on_sc"] + o < K])
                line.append(f"{o:+d}:{v.mean():.2f}" if len(v) else f"{o:+d}:-")
            say(f"  {name:12s} " + " ".join(line))
        for name, side in (("later", "sc"), ("earlier", "rt")):
            lag = torch.tensor([r["on_" + side] - r["live_" + side] for r in rs if r["live_" + side] < K], dtype=torch.float)
            ratio = torch.tensor([step(revs[r["on_" + side]]) / step(revs[r["live_" + side]]) for r in rs if r["live_" + side] < K])
            at_form = sum(r["live_" + side] <= r["form"] for r in rs) / len(rs)
            say(f"  {name} group live before its onset by 0 revisions {(lag <= 0).float().mean():.2f}, 1-2 "
                f"{((lag >= 1) & (lag <= 2)).float().mean():.2f}, 3+ {(lag >= 3).float().mean():.2f}; never "
                f"{sum(r['live_' + side] >= K for r in rs)}; live by the pole's formation {at_form:.2f}; median steps at "
                f"onset over steps when live {ratio.median():.2f}")
    torch.save({"revs": revs, "pct": pct, "rows": rows}, OUT / "live.pt")
    (SUMMARY / f"mixed_units_live_{STEM}.log").write_text("\n".join(LOG) + "\n")


def writes() -> None:
    """At the final checkpoint, whether a pole's write (``reads``' ``write_R``, its top ``TOP`` tokens) names what
    follows its *a* firings, its *b* firings, or both: the share of next tokens in its write, less the same in
    ``WRITE_RANDOM`` random poles' writes from its layer; also over next tokens unique to their group (*b*'s that never
    follow an *a* firing, and the reverse). For controls the two groups are one job, so their ratio is 1 by
    construction; a fossil *b* (firing, serving nothing) would give *b* none."""
    from neuron_lens.catalog import firing_tokens, flat_tokens
    S = torch.load(OUT / "scan.pt", weights_only=False)
    H = torch.load(OUT / "history.pt", weights_only=False)
    flat, start = flat_tokens()
    g = torch.Generator().manual_seed(0)
    by_layer = collections.defaultdict(list)
    for u in H["units"]:
        by_layer[u["layer"]].append(u)
    rows = []
    for l in sorted(by_layer):
        r = next(r for r in S["layers"] if r["layer"] == l)
        key = {(n, p): i for i, (n, p) in enumerate(zip(r["neuron"].tolist(), r["pole"].tolist()))}
        t = pq.read_table(neuron_table("reads", l), columns=["neuron", "pole", "write_R_ids"])
        ids = torch.tensor(t["write_R_ids"].to_pylist())
        wk = {(n, p): j for j, (n, p) in enumerate(zip(t["neuron"].to_pylist(), t["pole"].to_pylist()))}
        for u in by_layer[l]:
            i = key[(u["neuron"], u["pole"])]
            v = r["doc"][i] >= 0
            nx = firing_tokens(flat, start, r["doc"][i][v].long(), r["pos"][i][v].long(), 1)
            ga = r["group"][i][v]
            W, Wr = ids[wk[(u["neuron"], u["pole"])]], ids[torch.randint(len(ids), (WRITE_RANDOM,), generator=g)]
            uniq = {"a": ga.clone(), "b": ~ga}
            uniq["a"][ga] = ~torch.isin(nx[ga], nx[~ga])
            uniq["b"][~ga] = ~torch.isin(nx[~ga], nx[ga])
            oa, ob = u["on_a"], u["on_b"]
            x = {"kind": u["kind"], "layer": l,
                 "first": "=" if oa is None or ob is None or oa == ob else "a" if oa < ob else "b"}
            for name, m in (("a", ga), ("b", ~ga), ("a1", uniq["a"]), ("b1", uniq["b"])):
                x[name] = (float(torch.isin(nx[m], W).float().mean()
                                 - torch.stack([torch.isin(nx[m], w) for w in Wr]).float().mean()) if m.any() else float("nan"))
            rows.append(x)
    say(f"{MODEL}: next tokens after a's and b's firings in the pole's write_R (top {TOP}), less {WRITE_RANDOM} random "
        f"poles' writes; b / a; then next tokens unique to their group")
    for k_ in (*KINDS, "control"):
        for lo, hi in arch.depth_bands(MODEL).values():
            x = [r for r in rows if r["kind"] == k_ and lo <= r["layer"] <= hi]
            if len(x) < 5:
                continue
            T = {f: torch.tensor([r[f] for r in x]) for f in ("a", "b", "a1", "b1")}
            se = lambda z: float(z[~z.isnan()].std() / (~z.isnan()).sum() ** 0.5)
            say(f"  {k_:8s} L{lo}-L{hi}: {len(x):5d}  a {T['a'].nanmean():+.3f} ± {se(T['a']):.3f}  b {T['b'].nanmean():+.3f} "
                f"± {se(T['b']):.3f}  b/a {T['b'].nanmean() / T['a'].nanmean():.2f};  unique: a {T['a1'].nanmean():+.3f} b "
                f"{T['b1'].nanmean():+.3f}  b/a {T['b1'].nanmean() / T['a1'].nanmean():.2f}")
    say("  by which group's onset came first (mixed kinds pooled, the latter half of the layers)")
    half_l = arch.text_config(MODEL).num_hidden_layers // 2
    for grp in ("mixed", "control"):
        for first in ("a", "b"):
            x = [r for r in rows if (r["kind"] == "control") == (grp == "control") and r["first"] == first and r["layer"] >= half_l]
            T = {f: torch.tensor([r[f] for r in x]) for f in ("a", "b")}
            say(f"    {grp:7s} {first} first: {len(x):5d}  a {T['a'].nanmean():+.3f}  b {T['b'].nanmean():+.3f}")
    (SUMMARY / f"mixed_units_writes_{STEM}.log").write_text("\n".join(LOG) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("part", choices=("embed", "scan", "show", "history", "writes", "panel", "probe", "live"))
    ap.add_argument("args", nargs="*", help="show: n; history, probe, live: the revisions in training order, the final last")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    if args.part == "history":
        history(args.args, args.n)
    elif args.part == "probe":
        probe(args.args)
    elif args.part == "live":
        live(args.args)
    else:
        {"embed": embed, "scan": scan, "writes": writes, "panel": panel,
         "show": lambda: show(int(args.args[0]) if args.args else args.n)}[args.part]()


if __name__ == "__main__":
    main()
