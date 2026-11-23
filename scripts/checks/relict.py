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
    ap.add_argument("part", choices=["tops", "matrix", "step", "cross", "embed"])
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
    with open(SUMMARY / f"relict_{args.part}_{STEM}.log", "a") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
