"""Relict generalization across training checkpoints (findings §23, §24): does an unrelated secondary mode's link form
through an intermediate that later falls away? Pythia (``MODEL``, one data order, checkpoints as revisions), fp16 on
MPS or CUDA.

Parts (data in ``data/relict/<model>/``, summaries in ``data/summary/relict_<part>_<model>.log``):

- ``tops <rev>...``: every neuron's top documents at each checkpoint, over the catalog's activation documents as
  ``06_activations.py`` keeps them (per document its peak position from 1, the ``TOP`` highest documents, ``act_top``
  the median of the first ``TYPICAL``).
- ``matrix <rev>...``: at each checkpoint, every census target's (``secondary_modes.py``'s plan, run on the final
  checkpoint) peak over positions [t − 2, t + 8] on the census's contexts (the targets' own, the confirmed sources'
  and the control) and on the ``N_CTX`` top contexts at that checkpoint of the targets and their confirmed sources;
  raw values, ``[contexts × targets]``.
- ``step <rev>...``: ``tops`` and ``matrix`` on one load of each checkpoint.
- ``cross <rev> <revs>...``: at one checkpoint (``main``), every target on the contexts every checkpoint in ``revs``
  gave the census's units: whether a job a unit had earlier still drives the target.
- ``score <revs>...``: how alike each confirmed pair's two units are at each checkpoint (the mean cosine of their top
  contexts' embeddings, as z over every pair of baseline units at that checkpoint), by thirds of their likeness at the
  final checkpoint on their first two contexts each; the final value retested on the other three (free of that
  selection). *Onset*: the first checkpoint where the source's own top contexts drive the target (≥ max(0.3, the
  control's q95 at that checkpoint) on two of five). Relict generalization predicts pairs unlike at the end that were
  alike at onset; the null is unlinked pairs of baseline units with the same final selection, layer gap and final
  retest, so a difference earlier in training is one the end doesn't carry. Last, an absolute share for comparing
  models: confirmed pairs as unlike at the end as a typical random pair at their layer gap.
- ``judge cards <revs>... | test [n] | submit end|onset | collect | score``: the census's judge (``secondary_modes.py``:
  Sonnet 5.5 at maximum effort, Batch API, decoys on every card) with five own contexts in place of a label (prompt
  ``relict_judge.md``), for ``N_JUDGE`` targets: each one's confirmed sources at the end, the same card again (the
  judge's reproducibility), and the sources at their link onset; ``score`` sets onset against the end and the retest,
  and the free rulers (likeness, a shared marked token) against the judge.
- ``embed <revs>...``: Qwen3-Embedding-0.6B embeddings of every top context, at every checkpoint, of the units in the
  census's confirmed pairs (``confirmed``) and of ``N_BASE`` random census units (the baseline), as
  ``secondary_modes.show`` prints them.

    NEURON_LENS_MODEL=EleutherAI/pythia-160m python scripts/checks/relict.py tops main step64000 ...
"""
import argparse
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import MODEL, TOP, TYPICAL, documents, neuron_table
from neuron_lens.paths import DATA, SUMMARY

sys.path.insert(0, str(Path(__file__).parent))
import secondary_modes as sm  # noqa: E402

STEM = arch.stem(MODEL)
OUT = DATA / "relict" / STEM
DEV = "cuda" if torch.cuda.is_available() else "mps"
BATCH, CTX_BATCH = (64, 256) if torch.cuda.is_available() else (32, 32)  # documents a batch (tops), contexts a batch
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def _down(mlp):
    return mlp.down_proj if hasattr(mlp, "down_proj") else mlp.dense_4h_to_h


def load(rev: str):
    return arch.load_model(MODEL, DEV, eager=False, revision=None if rev == "main" else rev)[1]


@torch.inference_mode()
def tops(rev: str, stack=None) -> dict:
    """``{"val", "doc", "pos"}`` ``[L, TOP, f]`` and ``act_top`` ``[L, f]`` at checkpoint ``rev`` (one pole a neuron)."""
    path = OUT / f"tops-{rev}.pt"
    if path.exists():
        return torch.load(path)
    stack = stack or load(rev)
    docs = documents()
    L, f = len(stack.layers), arch.text_config(MODEL).intermediate_size
    val = torch.full((L, TOP, f), -torch.inf, device=DEV)
    doc = torch.full((L, TOP, f), -1, dtype=torch.int32, device=DEV)
    pos = torch.full((L, TOP, f), -1, dtype=torch.int16, device=DEV)
    ctx = {}

    def hook(l):
        def f_(m, a):
            x = a[0].float().masked_fill(~ctx["mask"][..., None], -torch.inf)
            ext, at = x.max(1)  # [B, f]
            v, i = torch.cat([val[l], ext]).topk(TOP, dim=0)
            val[l] = v
            doc[l] = torch.cat([doc[l], ctx["doc"][:, None].expand(-1, f)]).gather(0, i)
            pos[l] = torch.cat([pos[l], at.to(torch.int16)]).gather(0, i)
        return f_
    hs = [_down(lay.mlp).register_forward_pre_hook(hook(l)) for l, lay in enumerate(stack.layers)]
    t0 = time.time()
    for s in range(0, len(docs), BATCH):
        b = docs[s:s + BATCH]
        T = max(map(len, b))
        ids = torch.zeros(len(b), T, dtype=torch.long)
        mask = torch.zeros(len(b), T, dtype=torch.bool)
        for r, d in enumerate(b):
            ids[r, :len(d)], mask[r, :len(d)] = d, True
        am = mask.clone()
        mask[:, 0] = False  # statistics from position 1 (06's NEURON_FROM)
        ctx.update(mask=mask.to(DEV), doc=torch.arange(s, s + len(b), dtype=torch.int32, device=DEV))
        stack(input_ids=ids.to(DEV), attention_mask=am.to(DEV).long())
        if s // BATCH % (12800 // BATCH) == 0:
            say(f"  {rev} {s}/{len(docs)} docs, {(s + len(b)) / (time.time() - t0):.0f} docs/s")
    for h in hs:
        h.remove()
    out = {"val": val.half().cpu(), "doc": doc.cpu(), "pos": pos.cpu(),
           "act_top": val[:, :TYPICAL].median(1).values.clamp_min(0).cpu()}
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    say(f"{rev}: {len(docs)} docs in {time.time() - t0:.0f} s")
    return out


def units(P: dict) -> list[tuple]:
    """The census's units: its targets, then every proposed source not already listed."""
    us = {tuple(t[1:]): None for t in P["targets"]}
    for srcs in P["sources"]:
        for src, _, _ in srcs:
            us.setdefault(tuple(src), None)
    return list(us)


def top_contexts(t: dict, u: tuple, k: int = sm.N_CTX) -> list[tuple[int, int]]:
    l, n, _ = u
    return [(int(d), int(q)) for d, q, v in zip(t["doc"][l, :k, n], t["pos"][l, :k, n], t["val"][l, :k, n]) if d >= 0 and v > 0]


@torch.inference_mode()
def peaks(rev: str, C: list, T: list, stack=None) -> torch.Tensor:
    """Every target's raw window peak (``secondary_modes.batch_ids``' window) on contexts ``C`` at ``rev``,
    ``[len(C), len(T)]`` fp16."""
    stack = stack or load(rev)
    docs = documents()
    cap, by_layer = {}, {}
    for j, (_, l, n, p) in enumerate(T):
        by_layer.setdefault(l, []).append((j, n, p))
    hs = []
    for l, items in by_layer.items():
        js = torch.tensor([j for j, _, _ in items], device=DEV)
        ns = torch.tensor([n for _, n, _ in items], device=DEV)
        ps = torch.tensor([p for _, _, p in items], dtype=torch.float32, device=DEV)
        hs.append(_down(stack.layers[l].mlp).register_forward_pre_hook(
            lambda m, a, js=js, ns=ns, ps=ps: cap.__setitem__(int(js[0]), (js, a[0][..., ns].float() * ps))))
    peak = torch.empty(len(C), len(T), dtype=torch.float16)
    for i in range(0, len(C), CTX_BATCH):
        ids, mask, win = sm.batch_ids(docs, C[i:i + CTX_BATCH])
        cap.clear()
        stack(input_ids=ids.to(DEV), attention_mask=mask.to(DEV))
        row = torch.empty(len(ids), len(T), device=DEV)
        w = win.to(DEV)[..., None]
        for js, a in cap.values():
            row[:, js] = a.masked_fill(~w, -1e4).amax(1)
        peak[i:i + len(ids)] = row.half().cpu()
    for h in hs:
        h.remove()
    return peak


def matrix(rev: str, stack=None) -> dict:
    """``{"contexts", "plan_row", "unit_ctx", "peak"}``: the contexts run, every target's raw window peak on them
    ``[contexts × targets]`` (fp16), the row of each of the census's contexts (−1 where not run) and each unit's rows.
    Run: of the census's contexts, the targets' own, the confirmed sources' and the control's; and the top contexts at
    ``rev`` of the targets and their confirmed sources (the census's confirmed pairs, ``confirmed``)."""
    path = OUT / f"matrix-{rev}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    P, t = sm.load()[0], tops(rev, stack)
    win = torch.load(sm.OUT / "state.pt")["win"].float().nan_to_num(0)
    pairs = confirmed(P, win)
    keep = sorted({i for o in P["own"] for i in o} | {i for _, _, rows in pairs for i in rows} | set(P["control"]))
    idx = {P["contexts"][i]: r for r, i in enumerate(keep)}
    plan_row = torch.full((len(P["contexts"]),), -1, dtype=torch.long)
    plan_row[keep] = torch.arange(len(keep))
    us = list(dict.fromkeys([tuple(x[1:]) for x in P["targets"]] + [s_ for _, s_, _ in pairs]))
    unit_ctx = {u: [idx.setdefault(c, len(idx)) for c in top_contexts(t, u)] for u in us}
    C = sorted(idx, key=idx.get)
    t0 = time.time()
    out = {"contexts": C, "plan_row": plan_row, "unit_ctx": unit_ctx, "peak": peaks(rev, C, P["targets"], stack)}
    torch.save(out, path)
    say(f"{rev}: {len(C)} contexts ({len(keep)} the census's) in {time.time() - t0:.0f} s")
    return out


def cross(rev: str, revs: list[str]) -> dict:
    """At ``rev``, every target on the contexts every checkpoint in ``revs`` gave its units (``matrix``): whether a
    unit's earlier job still drives the target. ``{"contexts", "peak"}``."""
    path = OUT / f"cross-{rev}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    P = sm.load()[0]
    idx = {}
    for k in revs:
        m = matrix(k)
        for rows in m["unit_ctx"].values():
            for i in rows:
                idx.setdefault(m["contexts"][i], len(idx))
    C = sorted(idx, key=idx.get)
    t0 = time.time()
    out = {"contexts": C, "peak": peaks(rev, C, P["targets"])}
    torch.save(out, path)
    say(f"cross at {rev}: {len(C)} contexts from {len(revs)} checkpoints in {time.time() - t0:.0f} s")
    return out


def confirmed(P: dict, win: torch.Tensor) -> list[tuple[int, tuple, list[int]]]:
    """The census's confirmed pairs, ``(target, source, rows)``, as ``secondary_modes.py`` confirms them on
    ``win`` (the target's peaks over its ``act_top``): ≥ 2 of the source's contexts at max(0.3, the target's control q95)."""
    q95 = win[P["control"]].quantile(0.95, dim=0)
    out = []
    for j, srcs in enumerate(P["sources"]):
        thr = max(sm.THRESH, float(q95[j]))
        out += [(j, tuple(src), rows) for src, _, rows in srcs if (win[rows, j] >= thr).sum() >= 2]
    return out


def link_at(P: dict, pairs: list, t: dict, m: dict) -> dict:
    """At one checkpoint (its ``tops`` and ``matrix``), for each confirmed pair: whether the source's own top contexts
    there drive the target (≥ max(0.3, the control's q95 there) on two), and those contexts in order of the target's
    peak on them."""
    T = P["targets"]
    a = torch.tensor([float(t["act_top"][l, n]) for _, l, n, _ in T]).clamp_min(1e-3)
    pk = m["peak"].float() / a
    row = m.get("plan_row", torch.arange(len(P["contexts"])))
    thr = torch.maximum(torch.full((len(T),), sm.THRESH), pk[row[P["control"]]].quantile(0.95, dim=0))
    out = {}
    for j, X, _ in pairs:
        uk = m["unit_ctx"].get(X, [])
        v = pk[uk, j]
        out[(j, X)] = (len(uk) >= 2 and bool((v >= thr[j]).sum() >= 2),
                       [m["contexts"][uk[i]] for i in v.argsort(descending=True).tolist()])
    return out


N_BASE = 500


def embed(revs: list[str]) -> dict:
    """``{rev: {unit: [n, d] fp16}}`` and the baseline units: see the module docstring."""
    from sentence_transformers import SentenceTransformer

    from neuron_lens.catalog import tokenizer
    path = OUT / "embed.pt"
    E = torch.load(path, weights_only=False) if path.exists() else {"emb": {}}
    P = sm.load()[0]
    win = torch.load(sm.OUT / "state.pt")["win"].float().nan_to_num(0)
    us = sorted({tuple(P["targets"][j][1:]) for j, _, _ in confirmed(P, win)} | {s for _, s, _ in confirmed(P, win)})
    import random
    rest = sorted(set(units(P)) - set(us))
    base = random.Random(0).sample(rest, min(N_BASE, len(rest)))
    E["units"], E["base"] = us, base
    m = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=DEV)
    tok, docs = tokenizer(), documents()
    for rev in revs:
        if rev in E["emb"]:
            continue
        t, t0 = tops(rev), time.time()
        ctx = {u: top_contexts(t, u) for u in us + base}
        flat = [c for cs in ctx.values() for c in cs]
        text = [sm.show(tok, docs, d, q).replace("⟦", "[").replace("⟧", "]") for d, q in flat]
        V = m.encode(text, batch_size=128, normalize_embeddings=True, convert_to_tensor=True).half().cpu()
        out, i = {}, 0
        for u, cs in ctx.items():
            out[u], i = V[i:i + len(cs)], i + len(cs)
        E["emb"][rev] = out
        torch.save(E, path)
        say(f"embed {rev}: {len(flat)} contexts of {len(ctx)} units in {time.time() - t0:.0f} s")
    return E


STEPS = lambda r: 143000 if r == "main" else int(r.removeprefix("step"))
GAPS = (1, 2, 3, 5, 9)  # layer-gap bins (lower edges) for matching random pairs to the linked ones
MATCH = 3  # random pairs drawn per linked pair from its cell
FIRST = 4  # a pole's strongest proposed inputs, for a per-pole share that doesn't grow with the number proposed


def score(revs: list[str]) -> None:
    import bisect
    import random
    P = sm.load()[0]
    T = P["targets"]
    win = torch.load(sm.OUT / "state.pt")["win"].float().nan_to_num(0)
    pairs = confirmed(P, win)
    E = torch.load(OUT / "embed.pt", weights_only=False)
    revs = sorted((r for r in revs if (OUT / f"matrix-{r}.pt").exists() and r in E["emb"]), key=STEPS)
    mid = [r for r in revs if 4000 <= STEPS(r) < 143000]
    rng = random.Random(0)
    us = sorted(set(E["units"]) | set(E["base"]))
    ix = {u: i for i, u in enumerate(us)}
    d = next(iter(E["emb"]["main"].values())).shape[1]

    def means(r, a=0, b=None, need=1):
        """Each unit's mean context embedding at ``r`` (over its contexts [a:b]; NaN with fewer than ``need``):
        the mean cosine of two units' contexts is the dot product of these."""
        M = torch.full((len(us), d), float("nan"))
        for u, i in ix.items():
            e = E["emb"][r][u][a:b]
            if len(e) >= need:
                M[i] = e.float().mean(0)
        return M

    M = {r: means(r) for r in revs}
    M2, M3 = means("main", 0, 2, 2), means("main", 2, 5, 3)  # the final selecting and retest contexts
    bi = torch.tensor([ix[u] for u in E["base"]])
    bs = {}  # every pair of baseline units at each checkpoint
    for r in revs:
        S = M[r][bi] @ M[r][bi].T
        x = S[torch.triu_indices(len(bi), len(bi), 1).unbind()]
        x = x[~x.isnan()]
        bs[r] = (x.mean().item(), x.std().item())
    dot = lambda A, i, k: (A[i] * A[k]).sum(-1)
    z = lambda v, r: (v - bs[r][0]) / bs[r][1]
    say(f"{MODEL}: {len(pairs)} confirmed pairs, checkpoints {', '.join(revs)}")
    say("random-pair baseline (mean cosine): " + ", ".join(f"{STEPS(r) // 1000}k {bs[r][0]:.3f}" for r in revs))
    L, D = {}, {}  # links; each unit's top documents (the embedded contexts' order) at each checkpoint
    for r in revs:
        t, m = tops(r), matrix(r)
        D[r] = {u: [d_ for d_, _ in top_contexts(t, u)] for u in us}
        for (j, X), (lk_, _) in link_at(P, pairs, t, m).items():
            L[(j, X, r)] = lk_
    n5 = torch.tensor([len(E["emb"]["main"][u]) >= 5 for u in us])
    keep = [(j, X) for j, X, _ in pairs if n5[ix[tuple(T[j][1:])]] and n5[ix[X]]]
    pi, xi = torch.tensor([ix[tuple(T[j][1:])] for j, _ in keep]), torch.tensor([ix[X] for _, X in keep])
    sel, ret = z(dot(M2, pi, xi), "main"), z(dot(M3, pi, xi), "main")
    zk = {r: z(dot(M[r], pi, xi), r) for r in revs}
    rows = [dict(u=tuple(T[j][1:]), v=X, band=T[j][0], gap=T[j][1] - X[0], k0=next((r for r in revs if L[(j, X, r)]), None), sel=float(sel[i]),
                 retest=float(ret[i]), zk={r: float(zk[r][i]) for r in revs}) for i, (j, X) in enumerate(keep)]
    lo, hi = torch.tensor([x["sel"] for x in rows]).quantile(torch.tensor([1 / 3, 2 / 3])).tolist()
    groups = (("unlike at the end", lambda x: x["sel"] < lo), ("middle", lambda x: lo <= x["sel"] < hi),
              ("alike at the end", lambda x: x["sel"] >= hi))
    med = lambda v: torch.tensor(v).nanmedian().item()
    say("\nz (likeness over the random-pair baseline), medians; onset as in the docstring")
    say("group                 n   final retest  at onset  onset − retest (mean ± se)   onset step")
    for name, f in groups:
        g = [x for x in rows if f(x) and x["k0"] not in (None, "main")]
        dd = torch.tensor([x["zk"][x["k0"]] - x["retest"] for x in g])
        say(f"{name:18s} {len(g):5d}  {med([x['retest'] for x in g]):11.2f}  {med([x['zk'][x['k0']] for x in g]):8.2f}  "
            f"{dd.mean():8.2f} ± {dd.std() / len(dd) ** 0.5:.2f}            {med([STEPS(x['k0']) for x in g]):7.0f}")
    say("\nmedian z by checkpoint (the final column includes the selecting contexts)")
    say("group".ljust(18) + "".join(f"{STEPS(r) // 1000:>6}k" for r in revs))
    for name, f in groups:
        g = [x for x in rows if f(x)]
        say(name.ljust(18) + "".join(f"{med([x['zk'][r] for x in g]):7.2f}" for r in revs))
    for b in dict.fromkeys(x["band"] for x in rows):
        out = []
        for name, f in groups:
            g = [x for x in rows if f(x) and x["band"] == b and x["k0"] not in (None, "main")]
            dd = torch.tensor([x["zk"][x["k0"]] - x["retest"] for x in g])
            out.append(f"{name} {dd.mean():+.2f} ± {dd.std() / max(len(dd), 1) ** 0.5:.2f} (n={len(g)})")
        say(f"  band {b}: onset − retest: " + "; ".join(out))

    # the null: unlinked pairs of baseline units in different layers, with the linked pairs' final selection, drawn
    # MATCH to a linked pair from its cell of layer gap and final retest (deciles of the linked pairs' retest)
    ok = n5[bi] & torch.stack([~M[r][bi].isnan().any(-1) for r in mid]).all(0)
    bu = [E["base"][i] for i in range(len(bi)) if ok[i]]
    ui, vi = map(torch.tensor, zip(*[(ix[u], ix[v]) for u in bu for v in bu if u[0] < v[0]]))
    nsel, nret = z(dot(M2, ui, vi), "main"), z(dot(M3, ui, vi), "main")
    nzk = {r: z(dot(M[r], ui, vi), r) for r in mid}
    ngap = torch.tensor([us[k][0] - us[i][0] for i, k in zip(ui.tolist(), vi.tolist())])
    un = [x for x in rows if x["sel"] < lo]
    edges = torch.tensor([x["retest"] for x in un]).quantile(torch.linspace(0, 1, 11)[1:-1]).tolist()
    gbin = lambda g: bisect.bisect_right(GAPS, int(g)) - 1
    cell = {}
    for i in torch.nonzero(nsel < lo).flatten().tolist():
        cell.setdefault((gbin(ngap[i]), bisect.bisect(edges, float(nret[i]))), []).append(i)
    lk, nu = [], []
    for x in un:
        c = cell.get((gbin(x["gap"]), bisect.bisect(edges, x["retest"])))
        if c:
            lk.append(x)
            nu += [dict(u=us[ui[i]], v=us[vi[i]], retest=float(nret[i]), zk={r: float(nzk[r][i]) for r in mid})
                   for i in rng.choices(c, k=MATCH)]
    say(f"\nunlike at the end: linked pairs (n={len(lk)} of {len(un)} with a cell) against random unlinked pairs in other "
        f"layers with the same final selection, layer gap and final retest (n={len(nu)}, from {len(ui)})")
    peak = lambda x: max((x["zk"][r] for r in mid if x["zk"][r] == x["zk"][r]), default=float("nan"))
    for c in (1.0, 1.5, 2.0):
        a_, b_ = (sum(peak(x) >= c for x in xs) / len(xs) for xs in (lk, nu))
        se = (a_ * (1 - a_) / len(lk) + b_ * (1 - b_) / len(nu)) ** 0.5
        say(f"  z ≥ {c} at some checkpoint {STEPS(mid[0]) // 1000}k-{STEPS(mid[-1]) // 1000}k: linked {a_:.3f}, random {b_:.3f} "
            f"(difference {a_ - b_:+.3f} ± {se:.3f})")
    say("  median z (linked/random): " + ", ".join(f"{STEPS(r) // 1000}k {med([x['zk'][r] for x in lk]):.2f}/"
                                                 f"{med([x['zk'][r] for x in nu]):.2f}" for r in mid)
        + f"; retest {med([x['retest'] for x in lk]):.2f}/{med([x['retest'] for x in nu]):.2f}")
    early, late = [r for r in mid if STEPS(r) <= 16000], [r for r in mid if STEPS(r) >= 64000]
    mz = lambda xs, rs: torch.tensor([torch.tensor([x["zk"][r] for r in rs]).nanmean() for x in xs]).nan_to_num(0)
    for name, f in (("early", lambda xs: mz(xs, early)), ("late", lambda xs: mz(xs, late)),
                    ("early − late, per pair (a history the end doesn't share)", lambda xs: mz(xs, early) - mz(xs, late))):
        a_, b_ = f(lk), f(nu)
        say(f"  mean z {name}: linked {a_.mean():.3f}, random {b_.mean():.3f} "
            f"(difference {a_.mean() - b_.mean():+.3f} ± {(a_.var() / len(a_) + b_.var() / len(b_)) ** 0.5:.3f})")
    say(f"    (early {', '.join(f'{STEPS(r) // 1000}k' for r in early)}; late {', '.join(f'{STEPS(r) // 1000}k' for r in late)})")

    def apart(u, v, r):
        """z of the mean cosine over context pairs from different documents (NaN when there are none)."""
        a, b = E["emb"][r][u].float(), E["emb"][r][v].float()
        msk = torch.tensor(D[r][u])[:, None] != torch.tensor(D[r][v])[None, :]
        return z(float((a @ b.T)[msk].mean()), r) if len(a) and len(b) and msk.any() else float("nan")
    share = lambda xs, r: sum(bool(set(D[r][x["u"]]) & set(D[r][x["v"]])) for x in xs) / len(xs)
    say("  sharing a top document (linked/random): " + ", ".join(f"{STEPS(r) // 1000}k {share(lk, r):.3f}/{share(nu, r):.3f}"
                                                               for r in mid))
    ap = {id(x): {r: apart(x["u"], x["v"], r) for r in mid} for x in lk + nu}
    say("  median z over contexts from different documents (linked/random): "
        + ", ".join(f"{STEPS(r) // 1000}k {med([ap[id(x)][r] for x in lk]):.2f}/{med([ap[id(x)][r] for x in nu]):.2f}" for r in mid))
    for name, rs in (("early", [r for r in mid if STEPS(r) <= 16000]), ("late", [r for r in mid if STEPS(r) >= 64000])):
        mz = lambda xs: torch.tensor([torch.tensor([ap[id(x)][r] for r in rs]).nanmean() for x in xs]).nan_to_num(0)
        a_, b_ = mz(lk), mz(nu)
        say(f"  same, mean z {name}: linked {a_.mean():.3f}, random {b_.mean():.3f} "
            f"(difference {a_.mean() - b_.mean():+.3f} ± {(a_.var() / len(a_) + b_.var() / len(b_)) ** 0.5:.3f})")

    # absolute, for comparing models: each confirmed pair's likeness at the end on all its contexts, against random
    # pairs in other layers at the same layer gap (their median: as unlike as a typical random pair) and against a
    # unit's own contexts split (2 against 3: the same unit)
    full = z(dot(M["main"], pi, xi), "main")
    nfull = z(dot(M["main"], ui, vi), "main")
    ref = {g: float(nfull[torch.tensor([gbin(x) == g for x in ngap.tolist()])].median()) for g in range(len(GAPS))}
    own = z((M2[bi] * M3[bi]).sum(-1), "main")
    own = own[~own.isnan()]
    floor = (own < nfull.median()).float().mean().item()  # how often the ruler calls a unit unlike itself
    own = own.median().item()
    below = [float(full[i]) < ref[gbin(x["gap"])] for i, x in enumerate(rows)]
    rel = torch.tensor([(float(full[i]) - ref[gbin(x["gap"])]) / (own - ref[gbin(x["gap"])]) for i, x in enumerate(rows)])
    say(f"\nabsolute (all contexts at the end): random pairs' median z by layer gap "
        + ", ".join(f"{GAPS[g]}+ {v:.2f}" for g, v in ref.items()) + f"; a unit against itself (2 contexts against 3) {own:.2f}, "
        f"below the random median {floor:.3f} (the ruler's floor)")
    sz = []
    for r in revs:
        x = z((means(r, 0, 2, 2)[bi] * means(r, 2, 5, 3)[bi]).sum(-1), r)
        sz.append(f"{STEPS(r) // 1000}k {x[~x.isnan()].median():.2f}")
    say("  a unit against itself, median z by checkpoint (baseline units with five contexts): " + ", ".join(sz))
    say(f"  confirmed pairs: median z {full.median():.2f}; as unlike as a typical random pair (below its gap's median) "
        f"{sum(below) / len(below):.3f}; likeness from random (0) to the same unit (1): quartiles "
        + ", ".join(f"{v:.2f}" for v in rel.quantile(torch.tensor([0.25, 0.5, 0.75])).tolist()))
    unlike = {keep[i] for i in range(len(rows)) if below[i]}
    for b in dict.fromkeys(x["band"] for x in rows):
        g = [i for i, x in enumerate(rows) if x["band"] == b]
        js = [j for j, t_ in enumerate(T) if t_[0] == b]
        first = [{tuple(src) for src, _, _ in P["sources"][j][:FIRST]} for j in js]
        say(f"  band {b}: confirmed inputs below random {sum(below[i] for i in g) / len(g):.3f} (n={len(g)}); median likeness "
            f"(0-1) {rel[g].median():.2f}; poles with one among their {FIRST} strongest inputs "
            f"{sum(any((j, X) in unlike for X in f) for j, f in zip(js, first)) / len(js):.3f} of {len(js)} "
            f"({sum(len(f) == FIRST for f in first) / len(js):.2f} with {FIRST} proposed)")


# the judge: secondary_modes.py's (Sonnet 5.5 at maximum effort, Batch API, decoys on every card), with five own
# contexts in place of a label (Pythia's catalog has none; prompt relict_judge.md)

N_JUDGE = 200  # targets with a confirmed source judged per model, as many from each band (seed 0)
PROMPT = Path(__file__).with_name("relict_judge.md")


def judge_cards(revs: list[str]) -> None:
    """``judge-cards.jsonl`` and ``judge-key.json``. For each of N_JUDGE targets with a confirmed source: an *end* card
    (its confirmed sources, each as the two of the source's contexts where the target fires most, with the target's own
    sixth and seventh contexts and two random poles' contexts as decoys, shuffled; five own contexts above them), a
    *retest* card (the same, shuffled again, other random decoys: the judge's reproducibility), and an *onset* card for
    each checkpoint before the last where one of its links begins (the target's and those sources' top contexts there)."""
    import json
    import random

    from neuron_lens.catalog import tokenizer
    P = sm.load()[0]
    T, C = P["targets"], P["contexts"]
    win = torch.load(sm.OUT / "state.pt")["win"].float().nan_to_num(0)
    pairs = confirmed(P, win)
    revs = sorted((r for r in revs if (OUT / f"matrix-{r}.pt").exists()), key=STEPS)
    by = {}
    for j, X, rows in pairs:
        by.setdefault(j, []).append((X, rows))
    rng = random.Random(0)
    bands = list(dict.fromkeys(x[0] for x in T))
    pick = []
    for b in bands:
        js = sorted(j for j in by if T[j][0] == b)
        pick += rng.sample(js, min(N_JUDGE // len(bands), len(js)))
    tok, docs = tokenizer(), documents()
    cards, key = [], {}

    def card(cid, j, own, srcs, **info):
        groups = [("src", X, cs) for X, cs in srcs[:sm.MAX_SRC]]
        if len(own) >= 7:
            groups.append(("own", None, own[5:7]))
        groups.append(("random", None, [C[rng.choice(P["control"])] for _ in range(2)]))
        rng.shuffle(groups)
        letters = [chr(65 + i) for i in range(len(groups))]
        text = [f"Target unit: {sm.name(T[j][1:])}", "Its own top contexts:"] + [f"  {sm.show(tok, docs, *c)}" for c in own[:5]]
        for a_, (_, _, cs) in zip(letters, groups):
            text += [f"\nGroup {a_}:"] + [f"  {sm.show(tok, docs, *c)}" for c in cs]
        cards.append({"custom_id": cid, "text": "\n".join(text)})
        key[cid] = dict(target=j, groups={a_: {"kind": k, "src": x} for a_, (k, x, _) in zip(letters, groups)}, **info)

    t = tops("main")
    for j in pick:
        srcs = sorted(by[j], key=lambda x: -float(win[x[1], j].mean()))
        srcs = [(X, [C[rows[i]] for i in win[rows, j].argsort(descending=True)[:2].tolist()]) for X, rows in srcs]
        own = top_contexts(t, tuple(T[j][1:]), 7)
        card(f"e{j:04d}", j, own, srcs, stage="end", rev="main")
        card(f"r{j:04d}", j, own, srcs, stage="retest", rev="main")
    sel, onset, own = [(j, X, r_) for j, X, r_ in pairs if j in set(pick)], {}, {}
    for r in revs:
        t, m = tops(r), matrix(r)
        own.update({(j, r): top_contexts(t, tuple(T[j][1:]), 7) for j in pick})
        for (j, X), (ok, cs) in link_at(P, sel, t, m).items():
            if ok and (j, X) not in onset:
                onset[(j, X)] = (r, cs[:2])
    for j in pick:
        at = {}
        for X, _ in sorted(by[j], key=lambda x: -float(win[x[1], j].mean())):
            if (j, X) in onset and onset[(j, X)][0] != "main":
                at.setdefault(onset[(j, X)][0], []).append((X, onset[(j, X)][1]))
        for r, srcs in at.items():
            card(f"o{j:04d}-{r}", j, own[(j, r)], srcs, stage="onset", rev=r)
    (OUT / "judge-cards.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cards))
    (OUT / "judge-key.json").write_text(json.dumps(key))
    n = {st: sum(k["stage"] == st for k in key.values()) for st in ("end", "retest", "onset")}
    say(f"{len(pick)} targets ({len(sel)} confirmed pairs, {sum(v[0] != 'main' for v in onset.values())} with an onset before "
        f"the last checkpoint): {n['end']} end cards, {n['retest']} retest, {n['onset']} onset")


def judge_key() -> dict:
    import json
    return json.loads((OUT / "judge-key.json").read_text())


def judged() -> list[dict]:
    """Every judged group: card, stage, checkpoint, target, kind, source, normalized p_same, p_related, p_unrelated."""
    key = judge_key()
    J = torch.load(OUT / "judged.pt", weights_only=False) if (OUT / "judged.pt").exists() else {}
    out = []
    for cid, gs in J.items():
        k = key[cid]
        for g in gs:
            a = g["group"].strip().removeprefix("Group ").strip()
            if a not in k["groups"]:
                continue
            s_ = g["p_same"] + g["p_related"] + g["p_unrelated"] or 1
            info = k["groups"][a]
            out.append(dict(cid=cid, stage=k["stage"], rev=k["rev"], j=k["target"], kind=info["kind"],
                            src=tuple(info["src"]) if info["src"] else None, p_same=g["p_same"] / s_,
                            p_rel=g["p_related"] / s_, p_un=g["p_unrelated"] / s_, shared=g["shared"]))
    return out


def params(card: dict) -> dict:
    p = sm.params(card)
    p["system"] = [{"type": "text", "text": PROMPT.read_text(), "cache_control": {"type": "ephemeral"}}]
    return p


def judge_test(n: int) -> None:
    import json
    import random
    cards = [json.loads(x) for x in (OUT / "judge-cards.jsonl").read_text().split("\n")]
    key, c = judge_key(), sm.client()
    for card in random.Random(1).sample(cards, n):
        with c.messages.stream(**params(card)) as st:
            r = st.get_final_message()
        say(card["text"], f"\n--- {key[card['custom_id']]}\n" + next(b.text for b in r.content if b.type == "text"))
        say(f"usage: {r.usage.input_tokens} in, {r.usage.cache_read_input_tokens} cache read, {r.usage.output_tokens} out\n")


def judge_submit(stage: str) -> None:
    """``end``: every end card; ``onset``: the onset and retest cards of the targets with a source judged unrelated
    (p ≥ 0.5) at the end."""
    import json

    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    cards = [json.loads(x) for x in (OUT / "judge-cards.jsonl").read_text().split("\n")]
    key = judge_key()
    if stage == "end":
        cards = [c for c in cards if key[c["custom_id"]]["stage"] == "end"]
    else:
        un = {v["j"] for v in judged() if v["stage"] == "end" and v["kind"] == "src" and v["p_un"] >= 0.5}
        cards = [c for c in cards if key[c["custom_id"]]["stage"] in ("onset", "retest") and key[c["custom_id"]]["target"] in un]
    b = sm.client().messages.batches.create(
        requests=[Request(custom_id=c["custom_id"], params=MessageCreateParamsNonStreaming(**params(c))) for c in cards])
    path = OUT / "judge-batches.json"
    ids = json.loads(path.read_text()) if path.exists() else {}
    ids[stage] = b.id
    path.write_text(json.dumps(ids))
    say(f"{stage}: {len(cards)} cards, batch {b.id}")


def judge_collect() -> None:
    import json
    c = sm.client()
    path = OUT / "judged.pt"
    J = torch.load(path, weights_only=False) if path.exists() else {}
    for stage, bid in json.loads((OUT / "judge-batches.json").read_text()).items():
        while (b := c.messages.batches.retrieve(bid)).processing_status != "ended":
            print(time.strftime("%H:%M:%S"), stage, b.request_counts, flush=True)
            time.sleep(60)
        n, bad, tk = 0, 0, [0, 0, 0, 0]
        for r in c.messages.batches.results(bid):
            try:
                J[r.custom_id] = json.loads(next(x.text for x in r.result.message.content if x.type == "text"))["groups"]
                u = r.result.message.usage
                tk = [a + (b_ or 0) for a, b_ in zip(tk, (u.input_tokens, u.cache_creation_input_tokens,
                                                         u.cache_read_input_tokens, u.output_tokens))]
                n += 1
            except Exception:
                bad += 1
        # the Batch API's prices for Sonnet 5.5: half of $2 in, $2.50 cache write, $0.10 cache read, $10 out per MTok
        cost = (tk[0] * 1 + tk[1] * 1.25 + tk[2] * 0.05 + tk[3] * 5) / 1e6
        say(f"{stage}: {n} judged, {bad} failed; tokens in {tk[0]}, cache write {tk[1]}, cache read {tk[2]}, out {tk[3]} "
            f"(~${cost:.2f})")
    torch.save(J, path)


def judge_score() -> None:
    """The judge's verdicts: decoys, the share of judged inputs and targets with an unrelated source (with the floor the
    judge's false 'unrelated' on own contexts puts under it), the free rulers against the judge, and each pair's verdict
    at its onset against its verdict at the end and at the retest."""
    from neuron_lens.stats import auroc
    P = sm.load()[0]
    T = P["targets"]
    V, key = judged(), judge_key()
    un = lambda v: v["p_un"] >= 0.5
    say(f"{MODEL}: {len({v['cid'] for v in V})} cards judged")
    fpr = {}
    for st in ("end", "retest", "onset"):
        d = {k: [v for v in V if v["stage"] == st and v["kind"] == k] for k in ("own", "random")}
        if not d["own"]:
            continue
        fpr[st] = sum(map(un, d["own"])) / len(d["own"])
        say(f"  {st} decoys: own contexts judged same {sum(v['p_same'] >= max(v['p_rel'], v['p_un']) for v in d['own']) / len(d['own']):.2f}, "
            f"unrelated {fpr[st]:.3f} (n={len(d['own'])}); random poles' contexts unrelated {sum(map(un, d['random'])) / len(d['random']):.2f}")
    end = {(v["j"], v["src"]): v for v in V if v["stage"] == "end" and v["kind"] == "src"}
    say(f"\nend: {len(end)} judged inputs, unrelated {sum(map(un, end.values())) / len(end):.3f}; "
        f"argmax same/related/unrelated " + "/".join(f"{sum(max(('p_same', 'p_rel', 'p_un'), key=v.get) == k for v in end.values()) / len(end):.2f}"
                                                for k in ("p_same", "p_rel", "p_un")))
    js = sorted({v["j"] for v in V if v["stage"] == "end"})
    for b in dict.fromkeys(T[j][0] for j in js):
        jb = [j for j in js if T[j][0] == b]
        n = {j: sum(1 for (j_, _) in end if j_ == j) for j in jb}
        has = {j for (j, _), v in end.items() if un(v) and T[j][0] == b}
        floor = sum(1 - (1 - fpr["end"]) ** n[j] for j in jb) / len(jb)
        say(f"  band {b}: {len(jb)} targets, inputs unrelated {sum(un(v) for (j, _), v in end.items() if T[j][0] == b) / sum(n.values()):.3f}; "
            f"targets with one {len(has) / len(jb):.3f} (floor {floor:.3f})")
    # the free rulers against the judge
    E = torch.load(OUT / "embed.pt", weights_only=False)
    em, base = E["emb"]["main"], E["base"]
    M = torch.stack([em[u].float().mean(0) for u in base if len(em[u])])
    S_ = M @ M.T
    x = S_[torch.triu_indices(len(M), len(M), 1).unbind()]
    docs, t = documents(), tops("main")
    marked = lambda u: {int(docs[d][q]) for d, q in top_contexts(t, u, 20)}
    ys, zs, ws = [], [], []
    for (j, X), v in end.items():
        U = tuple(T[j][1:])
        if len(em.get(U, [])) and len(em.get(X, [])):
            ys.append(un(v))
            zs.append(-(((em[U].float().mean(0) @ em[X].float().mean(0)).item() - x.mean().item()) / x.std().item()))
            ws.append(-float(len(marked(U) & marked(X)) > 0))
    ys = torch.tensor(ys)
    say(f"  free rulers against the judge's 'unrelated' (AUROC, n={len(ys)}, {int(ys.sum())} unrelated): topical likeness "
        f"{auroc(torch.tensor(zs), ys):.2f}, a shared marked token in the top 20 {auroc(torch.tensor(ws), ys):.2f}")
    # onset against end and retest
    ret = {(v["j"], v["src"]): v for v in V if v["stage"] == "retest" and v["kind"] == "src"}
    ons = {(v["j"], v["src"]): v for v in V if v["stage"] == "onset" and v["kind"] == "src"}
    if not ons:
        return
    say("\nthe targets with an unrelated input at the end: their inputs' verdicts at the retest and at onset")
    for name, f in (("unrelated at the end", un), ("not unrelated at the end", lambda v: not un(v))):
        g = [k for k, v in end.items() if f(v) and k[0] in {j for j, _ in ons}]
        r_ = [un(ret[k]) for k in g if k in ret]
        o_ = [un(ons[k]) for k in g if k in ons]
        say(f"  {name:26s} n={len(g):4d}: unrelated at the retest {sum(r_) / max(len(r_), 1):.3f} (n={len(r_)}), at onset "
            f"{sum(o_) / max(len(o_), 1):.3f} (n={len(o_)}; median onset step "
            f"{torch.tensor([STEPS(ons[k]['rev']) for k in g if k in ons]).float().median():.0f})")
    say("  (relict generalization: inputs unrelated at the end and at their retest that were not at their onset)")
    for v in sorted((v for k, v in ons.items() if un(end[k]) and k in ret and un(ret[k]) and not un(v)), key=lambda v: v["j"])[:12]:
        e = end[(v["j"], v["src"])]
        say(f"    {sm.name(T[v['j']][1:])} ← {sm.name(v['src'])}: at {v['rev']} '{v['shared'][:50]}' ({v['p_un']:.2f}); at the end "
            f"'{e['shared'][:50]}' ({e['p_un']:.2f})")


def check_main(t: dict) -> None:
    """The pass at ``main`` against the catalog's ``acts`` tables: top-document overlap and act_top."""
    for l in (0, len(t["val"]) // 2, len(t["val"]) - 1):
        a = pq.read_table(neuron_table("acts", l), columns=["top_doc", "act_top"]).to_pydict()
        ov = [len(set(x[:TOP]) & set(t["doc"][l, :, n].tolist())) / TOP for n, x in enumerate(a["top_doc"])]
        r = torch.tensor(a["act_top"]) / t["act_top"][l].clamp_min(1e-6)
        say(f"  L{l}: top-doc overlap with the catalog median {torch.tensor(ov).median():.3f}, "
            f"act_top ratio median {r.median():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["tops", "matrix", "step", "cross", "embed", "score", "judge"])
    ap.add_argument("revs", nargs="*")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    if args.part == "tops":
        for rev in args.revs:
            t = tops(rev)
            if rev == "main":
                check_main(t)
    if args.part == "matrix":
        for rev in args.revs:
            matrix(rev)
    if args.part == "step":
        for rev in args.revs:
            if not (OUT / f"matrix-{rev}.pt").exists():
                stack = load(rev)
                tops(rev, stack)
                matrix(rev, stack)
                del stack
                torch.cuda.empty_cache() if DEV == "cuda" else None
    if args.part == "cross":
        cross(args.revs[0], args.revs[1:])
    if args.part == "embed":
        embed(args.revs)
    if args.part == "score":
        score(args.revs)
    part = args.part
    if args.part == "judge":
        act, rest = args.revs[0], args.revs[1:]
        {"cards": lambda: judge_cards(rest), "test": lambda: judge_test(int(rest[0]) if rest else 2),
         "submit": lambda: judge_submit(rest[0]), "collect": judge_collect, "score": judge_score}[act]()
        part = "judge_score" if act == "score" else "judge"
    with open(SUMMARY / f"relict_{part}_{STEM}.log", "w" if part in ("score", "judge_score") else "a") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
