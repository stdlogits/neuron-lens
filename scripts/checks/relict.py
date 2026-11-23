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
        a = torch.tensor([float(t["act_top"][l, n]) for _, l, n, _ in T]).clamp_min(1e-3)
        pk = m["peak"].float() / a
        row = m.get("plan_row", torch.arange(len(P["contexts"])))
        thr = torch.maximum(torch.full((len(T),), sm.THRESH), pk[row[P["control"]]].quantile(0.95, dim=0))
        for j, X, _ in pairs:
            uk = m["unit_ctx"].get(X, [])
            L[(j, X, r)] = len(uk) >= 2 and bool((pk[uk, j] >= thr[j]).sum() >= 2)
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
    ap.add_argument("part", choices=["tops", "matrix", "step", "cross", "embed", "score"])
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
    with open(SUMMARY / f"relict_{args.part}_{STEM}.log", "w" if args.part == "score" else "a") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
