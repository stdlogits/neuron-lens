"""A census of secondary modes (findings §24): weights propose, contexts check, a judge names the relation. Qwen3-8B
(``MODEL``; any catalog, labelled for the judge), in its half dtype on CUDA or MPS.

**Sample.** 500 firing poles (``act_top`` ≥ 0.05) per depth band, layers 1+ (seed 0). **Proposal.** Each pole's first
16 positive in-edges (the ``edges`` table, weights only). **Check.** Every source's five top contexts (``acts``'
``top_doc``, ``top_pos``: the 64 tokens before the firing, the firing and the 8 after), each target's own five (the
reference) and one top context each of 400 random firing poles (the control) are run once, and every target's peak over
positions [t − 2, t + 8] around the source's firing is recorded as a share of its ``act_top``: a [contexts × targets]
matrix. A source is *confirmed* for a target when the target reaches max(0.3, its q95 over the control contexts) on at
least two of the source's five. Chance: five random control contexts pass that 0.1% of the time.

**Judge.** Each target with a confirmed source gets a card: its label and three of its own contexts, then up to 12
confirmed sources, each as the two of the source's contexts where the target fired most, with two decoys shuffled in (the
target's own fourth and fifth contexts, which should read "same", and two random poles' contexts). No activation values
are shown. Sonnet 5.5 at maximum effort (prompt ``secondary_modes_judge.md``, Batch API) gives each group probabilities of
"same", "related" and "unrelated", ranked rather than calibrated (findings §13).

Parts (data in ``data/secondary_modes/<model>/``, summaries in ``data/summary/secondary_modes_<part>_<model>.log``):

- ``measure``: the plan and the matrix (resumable; Qwen3-8B on an M5 Max, MPS: 47 minutes for 97,511 contexts).
- ``cards``, ``test [n]``, ``submit``, ``collect``: the judge's cards, n of them run synchronously, the batch, the results.
- ``score``: confirmation by band and by in-edge rank, the decoys, the verdicts, the share of targets with an unrelated
  source (and the floor the judge's false "unrelated" calls on own contexts put under it), the strongest unrelated cases.
- ``inherit``: every pole of the model on each unrelated mode's contexts, the target's own and the control: *job-sharers*
  are poles that fire (≥ 0.3) on ≥ 3 of the target's own five and on ≤ 2% of 200 control contexts; *carriers* are
  job-sharers that also fire on ≥ 2 of the mode's contexts; chance replaces those by random control contexts (20 draws).
- ``weights``: weight-only relatedness scores for each judged source, against the judge (AUROC; low score = unrelated):
  ``push``, the edge ``cos``; ``orth``, the source's write minus its part along the target's read (residual metric at the
  target's MLP input) against the push-weighted mean of the target's other proposed sources' remainders; ``wr_R``, the
  write (R-lens ``out``) against the target's read (R-lens ``mid``); ``rr_R``, ``rr_S``, read against read through either
  lens; ``sR``, the source's S read against the target's R read; ``rr_Rd``, ``rr_Sd``, the reads with the layer's top 64
  read directions (PCs of 1,024 sampled read rows) removed; ``ss_w``, ``ss_rR``, ``ss_S``, the source against the target's
  other confirmed sources (leave-one-out). Token reads are z-scored per token over 1,024 sampled rows of the layer, and
  compared by Pearson r. Then per-target weight signatures (``cos_gate_up``, read peakedness, ``w_norm``, ...) against
  whether a target has an unrelated source.
- ``labels``: the cosine of the target's and the source's catalog labels (``fires``) under Qwen3-Embedding-0.6B, against
  the judge, with controls: shuffled verdicts, pairs within one target, clear target labels only, word overlap, pairs
  sharing no word, random poles' labels, and weight-strength quartiles.

    NEURON_LENS_MODEL=Qwen/Qwen3-8B python scripts/checks/secondary_modes.py measure|cards|test|submit|collect|score|inherit|weights|labels
"""
import argparse
import functools
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

from neuron_lens import arch, metric
from neuron_lens.catalog import MODEL, documents, kept, lobe_signs, neuron_table, tokenizer
from neuron_lens.catalog import entry as _entry
from neuron_lens.neurons import poles
from neuron_lens.paths import DATA, SUMMARY
from neuron_lens.stats import auroc
from neuron_lens.weights import tensors

STEM = MODEL.split("/")[-1].lower()
OUT = DATA / "secondary_modes" / STEM
DEV = "cuda" if torch.cuda.is_available() else "mps"
WIN, AFTER, THRESH = 64, 8, 0.3
PER_BAND, SOURCES, N_CTX, N_CONTROL = 500, 16, 5, 400
JUDGE, EFFORT, MAX_TOKENS, MAX_SRC, BEFORE = "claude-sonnet-5-5", "max", 32000, 12, 40
BANDS = list(arch.depth_bands(MODEL))
LOG = []
entry = functools.lru_cache(maxsize=None)(_entry)


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def name(u):
    return f"L{u[0]} n{u[1]} {'+' if u[2] > 0 else '−'}"


def load():
    P = torch.load(OUT / "plan.pt", weights_only=False)
    key = json.loads((OUT / "key.json").read_text()) if (OUT / "key.json").exists() else {}
    J = torch.load(OUT / "judged.pt") if (OUT / "judged.pt").exists() else {}
    return P, key, J


def verdicts(key, J):
    """Per judged group: target, kind (src, own, random), source, normalized p_same, p_related, p_unrelated."""
    out = []
    for cid, k in key.items():
        g = {x["group"].strip().removeprefix("Group ").strip(): x for x in J.get(cid, [])}
        for a, info in k["groups"].items():
            if a in g:
                v = g[a]
                s = v["p_same"] + v["p_related"] + v["p_unrelated"] or 1
                src = tuple(info["src"]) if info["src"] else None
                out.append(dict(j=k["target"], kind=info["kind"], src=src, p_same=v["p_same"] / s, p_rel=v["p_related"] / s,
                                p_un=v["p_unrelated"] / s, shared=v["shared"],
                                share=k["confirmed"][str(src)]["share"] if src else None))
    return out


# measure

def in_edges(floor: float) -> list[dict]:
    """Each layer's in-lists as the ``edges`` table holds them, recomputed by ``16_edges.py`` with the floor
    ``|cos| ≥ floor/√d`` in place of its 5/√d: Pythia's weight graph is sparse (the median pole has no in-edge past
    5/√d at 160M), and the context check, not the floor, keeps chance proposals out."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("edges16", Path(__file__).parents[1] / "16_edges.py")
    e16 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e16)
    e16.SIGMAS = floor
    ins = e16.compute(64, DEV)[0]
    out = []
    for score, ids, push, _ in ins:
        f = score.shape[0] // e16.P
        ok = score >= 0
        row = ids % (e16.P * f)
        out.append({"in_layer": torch.where(ok, ids.div(e16.P * f, rounding_mode="floor"), -1).tolist(),
                    "in_neuron": torch.where(ok, row % f, -1).tolist(), "in_pole": torch.where(ok, 1 - 2 * (row // f), 0).tolist(),
                    "in_push": torch.where(ok, push, float("nan")).tolist()})
    return out


def plan(rng, floor=None):
    L = arch.text_config(MODEL).num_hidden_layers
    rows = {}
    edges = in_edges(floor) if floor else None
    for l in range(L):
        a = pq.read_table(neuron_table("acts", l), columns=["neuron", "pole", "act_top", "top_doc", "top_pos", "top_val"]).to_pydict()
        e = edges[l] if edges else pq.read_table(neuron_table("edges", l), columns=["in_layer", "in_neuron", "in_pole", "in_push"]).to_pydict()
        rows[l] = {(n, p): dict(act_top=t, doc=d, pos=q, val=v, ins=list(zip(il, inn, ip, pu)))
                   for n, p, t, d, q, v, il, inn, ip, pu in zip(a["neuron"], a["pole"], a["act_top"], a["top_doc"], a["top_pos"],
                                                                 a["top_val"], e["in_layer"], e["in_neuron"], e["in_pole"], e["in_push"])}
    top = lambda r, k: [(int(d), int(q)) for d, q, v in zip(r["doc"], r["pos"], r["val"]) if d >= 0 and v > 0][:k]
    targets = []
    for role, (lo, hi) in arch.depth_bands(MODEL).items():
        pool = [(l, n, p) for l in range(max(lo, 1), hi + 1) for (n, p), r in rows[l].items() if r["act_top"] >= 0.05]
        targets += [(role, *u) for u in rng.sample(pool, min(PER_BAND, len(pool)))]
    ctx = {}
    add = lambda c: ctx.setdefault(c, len(ctx))
    own, sources = [], []
    for role, l, n, p in targets:
        r = rows[l][(n, p)]
        own.append([add(c) for c in top(r, N_CTX)])
        s = []
        for il, inn, ip, pu in r["ins"]:
            if il >= 0 and pu > 0:
                s.append(((il, inn, ip), float(pu), [add(c) for c in top(rows[il][(inn, ip)], N_CTX)]))
            if len(s) == SOURCES:
                break
        sources.append(s)
    firing = [(l, n, p) for l in range(L) for (n, p), r in rows[l].items() if r["act_top"] >= 0.05]
    control = []
    while len(control) < N_CONTROL:
        c = top(rows[(u := rng.choice(firing))[0]][u[1:]], 1)
        if c:
            control.append(add(c[0]))
    return dict(targets=targets, act_top=[rows[l][(n, p)]["act_top"] for _, l, n, p in targets], own=own, sources=sources,
                control=control, contexts=sorted(ctx, key=ctx.get), floor=floor or 5.0)


def batch_ids(docs, batch):
    seqs, fire = [], []
    for d, q in batch:
        lo = max(1, q - WIN)
        seqs.append(docs[d][lo:q + AFTER + 1].long())
        fire.append(q - lo)
    Lm = max(map(len, seqs))
    ids = torch.zeros(len(seqs), Lm, dtype=torch.long)
    mask = torch.zeros_like(ids)
    win = torch.zeros(len(seqs), Lm, dtype=torch.bool)
    for r, (s, f) in enumerate(zip(seqs, fire)):
        ids[r, :len(s)], mask[r, :len(s)] = s, 1
        win[r, max(1, f - 2):min(len(s), f + 9)] = True
    return ids, mask, win


def lm_half():
    """The decoder stack in the model's half dtype (bf16; Pythia fp16, its checkpoints' dtype)."""
    torch.set_grad_enabled(False)
    return arch.load_model(MODEL, DEV, eager=False)[1]


def down(stack, l):
    mlp = stack.layers[l].mlp
    return mlp.down_proj if hasattr(mlp, "down_proj") else mlp.dense_4h_to_h


def measure(args):
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "plan.pt").exists():
        torch.save(plan(random.Random(0), args.floor), OUT / "plan.pt")
    P = torch.load(OUT / "plan.pt", weights_only=False)
    T, C = P["targets"], P["contexts"]
    say(f"{len(T)} targets, {len(C)} unique contexts, {sum(len(s) for s in P['sources'])} proposed sources")
    sf = OUT / "state.pt"
    st = torch.load(sf) if sf.exists() else dict(done=0, win=torch.full((len(C), len(T)), float("nan"), dtype=torch.float16))
    docs, lm = documents(), lm_half()
    cap, by_layer = {}, {}
    for j, (_, l, n, p) in enumerate(T):
        by_layer.setdefault(l, []).append((j, n, p))
    scale = torch.tensor(P["act_top"], device=DEV)
    for l, items in by_layer.items():
        js, ns = torch.tensor([j for j, _, _ in items], device=DEV), torch.tensor([n for _, n, _ in items], device=DEV)
        ps = torch.tensor([p for _, _, p in items], dtype=torch.float32, device=DEV)
        down(lm, l).register_forward_pre_hook(lambda m, a, js=js, ns=ns, ps=ps: cap.__setitem__(int(js[0]), (js, a[0][..., ns].float() * ps)))
    class Stop(Exception):
        pass
    def stop(m, a, o):
        raise Stop
    lm.layers[max(by_layer)].mlp.register_forward_hook(stop)
    t0, saved, i0 = time.time(), time.time(), st["done"]
    while st["done"] < len(C):
        i = st["done"]
        ids, mask, win = batch_ids(docs, C[i:i + 32])
        cap.clear()
        try:
            lm(input_ids=ids.to(DEV), attention_mask=mask.to(DEV))
        except Stop:
            pass
        row = torch.empty(len(ids), len(T), device=DEV)
        w = win.to(DEV)[..., None]
        for js, a in cap.values():
            row[:, js] = a.masked_fill(~w, -1e4).amax(1)
        st["win"][i:i + len(ids)] = (row / scale).half().cpu()
        st["done"] = i + len(ids)
        if time.time() - saved > 600 or st["done"] >= len(C):
            torch.save(st, sf)
            saved = time.time()
            say(f"{time.strftime('%H:%M:%S')} {st['done']}/{len(C)} contexts, {(st['done'] - i0) / (time.time() - t0):.1f}/s")


# the judge

def show(tok, docs, d, q):
    ids = docs[d][max(0, q - BEFORE):q + AFTER + 1].tolist()
    k = q - max(0, q - BEFORE)
    s = lambda xs: tok.decode(xs).replace("\n", "⏎")
    return f"…{s(ids[:k])}⟦{s(ids[k:k + 1])}⟧{s(ids[k + 1:])}…"


def cards(args):
    P, _, _ = load()
    win = torch.load(OUT / "state.pt")["win"].float().nan_to_num(0)
    C, tok, docs, rng = P["contexts"], tokenizer(), documents(), random.Random(0)
    q95 = win[P["control"]].quantile(0.95, dim=0)
    out, key = [], {}
    for j, (role, l, n, p) in enumerate(P["targets"]):
        thr = max(THRESH, float(q95[j]))
        conf = []
        for src, push, rows in P["sources"][j]:
            v = win[rows, j]
            if (v >= thr).sum() >= 2:
                conf.append((src, push, float(v.mean()), [rows[i] for i in v.argsort(descending=True)[:2].tolist()]))
        if not conf:
            continue
        conf = sorted(conf, key=lambda c: -c[2])[:MAX_SRC]
        own = P["own"][j]
        groups = [("src", c[0], c[3]) for c in conf] + ([("own", None, own[3:5])] if len(own) >= 5 else [])
        groups.append(("random", None, [rng.choice(P["control"]), rng.choice(P["control"])]))
        rng.shuffle(groups)
        letters = [chr(65 + i) for i in range(len(groups))]
        text = [f"Target unit: {name((l, n, p))}", f"Main job (label): {entry(l, n, p)['fires']}", "Its own contexts:"]
        text += [f"  {show(tok, docs, *C[i])}" for i in own[:3]]
        for a, (kind, src, rows) in zip(letters, groups):
            text += [f"\nGroup {a}:"] + [f"  {show(tok, docs, *C[i])}" for i in rows]
        cid = f"t{j:04d}"
        out.append({"custom_id": cid, "text": "\n".join(text)})
        key[cid] = {"target": j, "groups": {a: {"kind": kind, "src": src} for a, (kind, src, _) in zip(letters, groups)},
                    "confirmed": {str(c[0]): {"push": c[1], "share": c[2]} for c in conf}}
    (OUT / "cards.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in out))
    (OUT / "key.json").write_text(json.dumps(key))
    say(f"{len(out)} of {len(P['targets'])} targets have a confirmed source; {sum(len(k['confirmed']) for k in key.values())} confirmed")


SCHEMA = {"type": "object", "properties": {"groups": {"type": "array", "items": {
    "type": "object", "properties": {"group": {"type": "string"}, "shared": {"type": "string"}, "p_same": {"type": "number"},
                                     "p_related": {"type": "number"}, "p_unrelated": {"type": "number"}},
    "required": ["group", "shared", "p_same", "p_related", "p_unrelated"], "additionalProperties": False}}},
    "required": ["groups"], "additionalProperties": False}


def client():
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY") or (Path.home() / ".config/anthropic/api_key").read_text().strip()
    return anthropic.Anthropic(api_key=key)


def params(card):
    system = Path(__file__).with_name("secondary_modes_judge.md").read_text()
    return {"model": JUDGE, "max_tokens": MAX_TOKENS, "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": card["text"]}]}


def all_cards():
    return [json.loads(x) for x in (OUT / "cards.jsonl").read_text().splitlines()]


def test(args):
    c = client()
    for card in random.Random(1).sample(all_cards(), args.n):
        with c.messages.stream(**params(card)) as s:
            r = s.get_final_message()
        say(card["text"], "\n---\n" + next(b.text for b in r.content if b.type == "text"))


def submit(args):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    b = client().messages.batches.create(requests=[Request(custom_id=x["custom_id"], params=MessageCreateParamsNonStreaming(**params(x))) for x in all_cards()])
    (OUT / "batch.json").write_text(json.dumps({"id": b.id}))
    say(b.id, b.processing_status)


def collect(args):
    c = client()
    bid = json.loads((OUT / "batch.json").read_text())["id"]
    while (b := c.messages.batches.retrieve(bid)).processing_status != "ended":
        print(time.strftime("%H:%M:%S"), b.request_counts, flush=True)
        time.sleep(60)
    out, bad = {}, 0
    for r in c.messages.batches.results(bid):
        try:
            out[r.custom_id] = json.loads(next(x.text for x in r.result.message.content if x.type == "text"))["groups"]
        except Exception:
            bad += 1
    torch.save(out, OUT / "judged.pt")
    say(f"{len(out)} judged, {bad} failed")


# results

def score(args):
    P, key, J = load()
    T = P["targets"]
    win = torch.load(OUT / "state.pt")["win"].float().nan_to_num(0)
    ctrl = win[P["control"]]
    q95 = ctrl.quantile(0.95, dim=0)
    rng = random.Random(0)
    null = torch.tensor([sum((ctrl[torch.tensor(rng.sample(range(len(ctrl)), 5)), j] >= max(THRESH, float(q95[j]))).sum() >= 2
                             for _ in range(50)) / 50 for j in range(len(T))])
    own = torch.tensor([win[r, j].mean().item() for j, r in enumerate(P["own"]) if r])
    V = verdicts(key, J)
    src = [v for v in V if v["kind"] == "src"]
    dec = {k: torch.tensor([[v["p_same"], v["p_rel"], v["p_un"]] for v in V if v["kind"] == k]) for k in ("own", "random")}
    fpr = float((dec["own"][:, 2] >= 0.5).float().mean())
    say(f"{MODEL}: {len(T)} targets, {sum(len(s) for s in P['sources'])} proposed sources, {sum(len(k['confirmed']) for k in key.values())} confirmed, "
        f"{len(J)} cards judged; own contexts {own.median():.2f} of act_top at the median; random contexts' q95 {q95.median():.3f}")
    say(f"judge on decoys: own held-out contexts argmax same {(dec['own'].argmax(1) == 0).float().mean():.2f}, judged unrelated {fpr:.3f}; "
        f"random poles' contexts argmax unrelated {(dec['random'].argmax(1) == 2).float().mean():.2f}; AUROC of p_unrelated, random vs own "
        f"{auroc(torch.cat([dec['random'][:, 2], dec['own'][:, 2]]), torch.cat([torch.ones(len(dec['random'])), torch.zeros(len(dec['own']))]).bool()):.2f}")
    rank_conf = {}
    for j in range(len(T)):
        conf = key.get(f"t{j:04d}", {}).get("confirmed", {})
        for r, (s, _, _) in enumerate(P["sources"][j]):
            rank_conf.setdefault(min(r, 15) // 4, []).append(str(tuple(s)) in conf)
    say("confirmed share by in-edge rank 1-4, 5-8, 9-12, 13-16: " + ", ".join(f"{sum(v) / len(v):.3f}" for _, v in sorted(rank_conf.items())))
    shares = torch.tensor([v["share"] for v in src])
    say(f"confirmed sources' strength (share of the target's act_top on their contexts): median {shares.median():.2f}, quartiles "
        f"{shares.quantile(0.25):.2f}-{shares.quantile(0.75):.2f}")
    say(f"\n{'band':12s} {'targets':>7s} {'proposed':>8s} {'confirmed':>9s} {'chance':>6s} {'≥1 conf':>7s} | {'same':>5s} {'rel':>5s} {'unrel':>5s} | "
        f"{'≥1 unrel':>8s} {'floor':>6s} {'≥1 unrel ≥0.5x':>14s}")
    for b in BANDS:
        js = [j for j, t in enumerate(T) if t[0] == b]
        prop = sum(len(P["sources"][j]) for j in js)
        nconf = {j: len(key[f"t{j:04d}"]["confirmed"]) for j in js if f"t{j:04d}" in key}
        r = [v for v in src if T[v["j"]][0] == b]
        arg = torch.tensor([max(range(3), key=lambda i: (v["p_same"], v["p_rel"], v["p_un"])[i]) for v in r])
        un = {v["j"] for v in r if v["p_un"] >= 0.5}
        strong = {v["j"] for v in r if v["p_un"] >= 0.5 and v["share"] >= 0.5}
        floor = sum(1 - (1 - fpr) ** n for n in nconf.values()) / len(js)
        say(f"{b:12s} {len(js):7d} {prop:8d} {sum(nconf.values()) / max(prop, 1):9.3f} {null[js].mean():6.3f} {len(nconf) / len(js):7.2f} | "
            f"{(arg == 0).float().mean():5.2f} {(arg == 1).float().mean():5.2f} {(arg == 2).float().mean():5.2f} | "
            f"{len(un) / len(js):8.3f} {floor:6.3f} {len(strong) / len(js):14.3f}")
    say("(floor: the share of targets the judge's false 'unrelated' rate on own contexts would give, from each target's number of "
        "confirmed sources)")
    say("\nstrongest unrelated sources (target ← source):")
    for v in sorted([v for v in src if v["p_un"] >= 0.5], key=lambda v: -v["share"])[:30]:
        t = T[v["j"]][1:]
        say(f"  {v['share']:.2f}  {name(t)} [{entry(*t)['fires'][:60]}] ← {name(v['src'])} [{entry(*v['src'])['fires'][:50]}]  shared: {v['shared'][:60]}")


def inherit(args):
    P, key, J = load()
    T, C = P["targets"], P["contexts"]
    win = torch.load(OUT / "state.pt")["win"].float().nan_to_num(0)
    rng = random.Random(0)
    modes = []
    for v in verdicts(key, J):
        if v["kind"] == "src" and v["p_un"] >= 0.5:
            j = v["j"]
            rows = next(r for s, _, r in P["sources"][j] if tuple(s) == v["src"])
            thr = max(THRESH, float(win[P["control"], j].quantile(0.95)))
            modes.append((j, v["src"], [r for r in rows if win[r, j] >= thr]))
    rows = sorted({r for _, _, m in modes for r in m} | {r for j, _, _ in modes for r in P["own"][j]} | set(P["control"]))
    L = arch.text_config(MODEL).num_hidden_layers
    act_top = torch.stack([torch.from_numpy(pq.read_table(neuron_table("acts", l), columns=["act_top"]).column(0).to_numpy().copy())
                           for l in range(L)]).float()
    f = act_top.shape[1] // 2
    recf = OUT / "inherit_rec.pt"  # [contexts, L, 2f] fp16, every pole's peak share
    if recf.exists():
        rec = torch.load(recf)
    else:
        docs, lm = documents(), lm_half()
        cap = {}
        for l in range(L):
            down(lm, l).register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(l, a[0]))
        rec = torch.zeros(len(rows), L, 2 * f, dtype=torch.float16)
        scale = act_top.clamp_min(1e-3).to(DEV)
        for i in range(0, len(rows), 16):
            ids, mask, w = batch_ids(docs, [C[r] for r in rows[i:i + 16]])
            lm(input_ids=ids.to(DEV), attention_mask=mask.to(DEV))
            wd = w.to(DEV)[..., None]
            for l in range(L):
                a = cap[l].float()
                rec[i:i + len(ids), l] = (torch.cat([a.masked_fill(~wd, -1e4).amax(1), (-a).masked_fill(~wd, -1e4).amax(1)], 1) / scale[l]).half().cpu()
        torch.save(rec, recf)
    idx = {r: n for n, r in enumerate(rows)}
    ctrl = list(P["control"])
    rng.shuffle(ctrl)
    broad = (rec[[idx[r] for r in ctrl[:200]]].float() >= THRESH).float().mean(0) > 0.02
    fires = lambda rs, need: (rec[[idx[r] for r in rs]].float() >= THRESH).sum(0) >= need
    lay = torch.arange(L)[:, None].expand(L, 2 * f)
    out = []
    for j, src, m in modes:
        _, lb, nb, pb = T[j]
        if len(m) < 2:
            continue
        share = fires(P["own"][j], 3) & ~broad
        share[lb, nb if pb > 0 else f + nb] = False
        carry = share & fires(m, 2)
        chance = sum((share & fires(rng.sample(ctrl[200:], len(m)), 2)).float() for _ in range(20)) / 20
        out.append(dict(j=j, src=src, n_down=int(share[lay > lb].sum()), n_up=int(share[lay < lb].sum()), c_down=int(carry[lay > lb].sum()),
                        c_up=int(carry[lay < lb].sum()), h_down=float(chance[lay > lb].sum()), h_up=float(chance[lay < lb].sum())))
    torch.save(out, OUT / "inherit.pt")
    say(f"{len(out)} unrelated modes with two or more contexts; job-sharers, carriers and chance per mode, pooled by the target's band")
    for b in BANDS:
        o = [x for x in out if T[x["j"]][0] == b]
        dn = [x for x in o if x["n_down"]]
        up = [x for x in o if x["n_up"]]
        tot = lambda k: sum(x[k] for x in o)
        say(f"  {b:12s} modes {len(o):3d} | downstream: {tot('n_down') / len(o):5.1f} sharers a mode, carriers {tot('c_down') / max(tot('n_down'), 1):.3f} of them "
            f"(chance {tot('h_down') / max(tot('n_down'), 1):.3f}), modes with a carrier over chance {sum(x['c_down'] - x['h_down'] >= 1 for x in dn) / max(len(dn), 1):.2f}"
            f" | upstream: {tot('n_up') / len(o):5.1f} sharers, carriers {tot('c_up') / max(tot('n_up'), 1):.3f} (chance {tot('h_up') / max(tot('n_up'), 1):.3f}), "
            f"modes with one {sum(x['c_up'] - x['h_up'] >= 1 for x in up) / max(len(up), 1):.2f}")


def weights(args):
    P, key, J = load()
    T = P["targets"]
    items = [v for v in verdicts(key, J) if v["kind"] == "src"]
    targets = sorted({v["j"] for v in items})
    L = arch.text_config(MODEL).num_hidden_layers
    need_read = {tuple(T[j][1:]) for j in targets} | {v["src"] for v in items}
    need_write = {tuple(s) for j in targets for s, _, _ in P["sources"][j]}
    keep = kept()
    nm = arch.names(MODEL)
    hd = tensors(MODEL, sorted({nm["unembed"], nm["norm"], nm["embed"]}))
    W_U, gam = hd[nm["unembed"]][keep].to(DEV, torch.float16), hd[nm["norm"]].to(DEV, torch.float32)
    emb = hd[nm["embed"]][keep].float()
    emb = (emb - emb.mean(0)).to(DEV, torch.float16)
    R_FIT, S_FIT = arch.fits(MODEL)
    J_out, J_mid, K_mid = (torch.as_tensor(arch.lens(MODEL, k, fit)).to(DEV, torch.float32) for k, fit in (("R_out", R_FIT), ("R_mid", R_FIT), ("S_mid", S_FIT)))
    def rl(vecs, Jl):
        t = vecs.to(DEV).float() if Jl is None else vecs.to(DEV).float() @ Jl.T
        return ((t / t.norm(dim=1, keepdim=True)) * gam).half() @ W_U.T
    def sl(covs, Kl):
        c = covs.to(DEV).float() @ Kl
        return (c / c.norm(dim=1, keepdim=True)).half() @ emb.T
    defl = lambda c, U: c.float() - (c.float() @ U.T) @ U
    zst = lambda x: (x.float().mean(0), x.float().std(0).clamp_min(1e-4))
    z = lambda x, st: (x.float() - st[0]) / st[1]
    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm()))
    reads, writes, stats, s2, pcs = {}, {}, {}, {}, {}
    g = torch.Generator().manual_seed(0)
    for l in range(L):
        p = poles(MODEL, l, lobe_signs(l))
        f = p["write"].shape[1]
        s2[l] = metric.s2(MODEL, "mid", l)
        for u in need_read:
            if u[0] == l:
                r = u[1] if u[2] > 0 else f + u[1]
                reads[u] = (p["cov"][r].clone(), p["vec"][r].clone())
        for u in need_write:
            if u[0] == l:
                writes[u] = (u[2] * p["write"][:, u[1]]).clone()
        smp = torch.randint(0, 2 * f, (1024,), generator=g)
        X = p["cov"][smp].float()
        X = X / X.norm(dim=1, keepdim=True)
        pcs[l] = torch.linalg.svd(X - X.mean(0), full_matrices=False)[2][:64].to(DEV)
        dc = defl(p["cov"][smp].to(DEV), pcs[l])
        stats[l] = dict(rr=zst(rl(p["vec"][smp], J_mid[l])), ss=zst(sl(p["cov"][smp], K_mid[l])),
                        rrd=zst(rl(metric.vector(dc, s2[l].to(DEV)), J_mid[l])), ssd=zst(sl(dc, K_mid[l])),
                        ww=zst(rl((p["write"][:, smp % f] * torch.where(smp < f, 1.0, -1.0)).T, J_out[l] if l < len(J_out) else None)))
        del p
    res, vecs = [], {}
    for j in targets:
        M = T[j][1]
        cB, vB = reads[tuple(T[j][1:])]
        S2 = s2[M]
        ip = lambda u, w: (u * w / S2).sum(-1)
        vb = vB / ip(vB, vB).sqrt()
        rem = {}
        for s, push, _ in P["sources"][j]:
            w = writes[tuple(s)]
            w = w / ip(w, w).sqrt()
            rem[tuple(s)] = (w - ip(w, vb) * vb, push, float((cB * w).sum() / metric.cov_norm(cB[None], S2)[0, 0]))
        zB = z(rl(vB[None], J_mid[M])[0], stats[M]["rr"])
        zBs = z(sl(cB[None], K_mid[M])[0], stats[M]["ss"])
        dB = defl(cB[None].to(DEV), pcs[M])
        zBd = z(rl(metric.vector(dB, S2.to(DEV)), J_mid[M])[0], stats[M]["rrd"])
        zBsd = z(sl(dB, K_mid[M])[0], stats[M]["ssd"])
        for it in [v for v in items if v["j"] == j]:
            s = it["src"]
            r, push, cos = rem[s]
            m = sum(pu * rr for t_, (rr, pu, _) in rem.items() if t_ != s) if len(rem) > 1 else None
            la = s[0]
            cA, vA = reads[s]
            zw = z(rl(writes[s][None], J_out[la] if la < len(J_out) else None)[0], stats[la]["ww"])
            zr = z(rl(vA[None], J_mid[la])[0], stats[la]["rr"])
            zs_ = z(sl(cA[None], K_mid[la])[0], stats[la]["ss"])
            dA = defl(cA[None].to(DEV), pcs[la])
            vecs[(j, s)] = (zw.half(), zr.half(), zs_.half())
            res.append(dict(it, push=push, cos=cos, orth=float(ip(r, m) / (ip(r, r).sqrt() * ip(m, m).sqrt())) if m is not None else float("nan"),
                            wr_R=corr(zw, zB), rr_R=corr(zr, zB), rr_S=corr(zs_, zBs), sR=corr(zs_, zB),
                            rr_Rd=corr(z(rl(metric.vector(dA, s2[la].to(DEV)), J_mid[la])[0], stats[la]["rrd"]), zBd),
                            rr_Sd=corr(z(sl(dA, K_mid[la])[0], stats[la]["ssd"]), zBsd)))
    for x in res:
        others = [y for y in res if y["j"] == x["j"] and y["src"] != x["src"]]
        for i, k in enumerate(("ss_w", "ss_rR", "ss_S")):
            x[k] = sum(corr(vecs[(x["j"], x["src"])][i].float(), vecs[(y["j"], y["src"])][i].float()) for y in others) / len(others) if others else float("nan")
    torch.save(res, OUT / "weight_scores.pt")
    names = ["push", "cos", "orth", "wr_R", "rr_R", "rr_S", "sR", "rr_Rd", "rr_Sd", "ss_w", "ss_rR", "ss_S"]
    for title, sel, cut in [("all judged sources (unrelated: p ≥ 0.5)", lambda x: True, lambda x: x["p_un"] >= 0.5),
                            ("extremes (p_unrelated ≥ 0.8 against p_same ≥ 0.6)", lambda x: x["p_un"] >= 0.8 or x["p_same"] >= 0.6, lambda x: x["p_un"] >= 0.8)]:
        say(f"\nAUROC for 'unrelated', low score = unrelated; {title}")
        say(f"{'band':12s} {'n':>5s} {'unrel':>6s} " + " ".join(f"{k:>6s}" for k in names))
        for b in BANDS + ["all"]:
            r = [x for x in res if (b == "all" or T[x["j"]][0] == b) and sel(x)]
            pos = torch.tensor([cut(x) for x in r])
            say(f"{b:12s} {len(r):5d} {pos.float().mean():6.2f} " + " ".join(f"{auroc(-torch.tensor([x[k] for x in r]).nan_to_num(0), pos):6.2f}" for k in names))
    un = {x["j"] for x in res if x["p_un"] >= 0.5}
    js = sorted({x["j"] for x in res})
    feats = lambda e: dict(cos_gate_up=e["cos_gate_up"], read_R_top=e["read_R_val"][0], read_Rz_top=e["read_R_z_val"][0], read_S_top=e["read_S_val"][0],
                           write_R_top=e["write_R_val"][0], vocab_kurt=e["vocab_kurt"], w_norm=e["w_norm"], null_share=e["null_share"])
    F = {j: feats(entry(*T[j][1:])) for j in js}
    say("\nper-target weight signatures: AUROC for having an unrelated source (high value = more likely)")
    for b in BANDS + ["all"]:
        sel = [j for j in js if b == "all" or T[j][0] == b]
        pos = torch.tensor([j in un for j in sel])
        say(f"  {b:12s} n={len(sel):4d}  " + "  ".join(f"{k} {auroc(torch.tensor([float(F[j][k]) for j in sel]).nan_to_num(0), pos):.2f}" for k in F[js[0]]))


def labels(args):
    from sentence_transformers import SentenceTransformer
    P, key, J = load()
    T = P["targets"]
    res = [v for v in verdicts(key, J) if v["kind"] == "src"]
    lab = lambda u: entry(*u)["fires"]
    m = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=DEV)
    emb = lambda xs: m.encode(xs, batch_size=64, normalize_embeddings=True, convert_to_tensor=True).float().cpu()
    us = sorted({tuple(T[x["j"]][1:]) for x in res} | {x["src"] for x in res})
    E = dict(zip(us, emb([lab(u) for u in us])))
    for x in res:
        x["cos"] = float(E[tuple(T[x["j"]][1:])] @ E[x["src"]])
    A = lambda sc, pos: auroc(torch.tensor(sc, dtype=torch.float32), torch.tensor(pos))
    un = [x["p_un"] >= 0.5 for x in res]
    say(f"label cosine (Qwen3-Embedding-0.6B) against the judge's 'unrelated', AUROC, low cosine = unrelated; n={len(res)}")
    for b in BANDS + ["all"]:
        r = [x for x in res if b == "all" or T[x["j"]][0] == b]
        ext = [x for x in r if x["p_un"] >= 0.8 or x["p_same"] >= 0.6]
        say(f"  {b:12s} n={len(r):4d}  all {A([-x['cos'] for x in r], [x['p_un'] >= 0.5 for x in r]):.3f}   extremes "
            f"{A([-x['cos'] for x in ext], [x['p_un'] >= 0.8 for x in ext]):.3f}")
    rng = random.Random(0)
    sh = un[:]
    rng.shuffle(sh)
    say(f"controls: shuffled verdicts {A([-x['cos'] for x in res], sh):.3f}")
    wins = tot = 0
    by = {}
    for x in res:
        by.setdefault(x["j"], []).append(x)
    for xs in by.values():
        for a in xs:
            for b_ in xs:
                if a["p_un"] >= 0.5 > b_["p_un"]:
                    tot += 1
                    wins += (a["cos"] < b_["cos"]) + 0.5 * (a["cos"] == b_["cos"])
    say(f"  within one target (unrelated against the target's other sources): {wins / tot:.3f} over {tot} pairs")
    clear = [x for x in res if entry(*T[x["j"]][1:])["fires_conf"] >= 0.5 and not re.search("shadow|unclear", lab(T[x["j"]][1:]), re.I)]
    say(f"  clear target labels only (n={len(clear)}): {A([-x['cos'] for x in clear], [x['p_un'] >= 0.5 for x in clear]):.3f}")
    stop = {"and", "the", "with", "words", "word", "tokens", "token", "text", "like"}
    words = lambda s: {w for w in re.findall(r"\w+", s.lower()) if len(w) > 2 and w not in stop}
    jac = [len(words(lab(T[x["j"]][1:])) & words(lab(x["src"]))) / max(len(words(lab(T[x["j"]][1:])) | words(lab(x["src"]))), 1) for x in res]
    say(f"  word overlap (Jaccard) {A([-v for v in jac], un):.3f}; the embedding on pairs sharing no word {A([-x['cos'] for x, v in zip(res, jac) if v == 0], [u for u, v in zip(un, jac) if v == 0]):.3f} "
        f"(n={sum(v == 0 for v in jac)})")
    pairs = []
    for x in rng.sample(res, 600):
        for _ in range(3):
            u = (x["src"][0], rng.randrange(arch.text_config(MODEL).intermediate_size), rng.choice((1, -1)))
            if entry(*u)["act_top"] > 0.05:
                pairs.append((lab(T[x["j"]][1:]), lab(u)))
                break
    nc = (emb([a for a, _ in pairs]) * emb([b for _, b in pairs])).sum(1)
    cs = torch.tensor([x["cos"] for x in res])
    q = lambda t: f"{t.median():.3f} ({t.quantile(0.25):.3f}-{t.quantile(0.75):.3f})"
    say(f"  median cosine (IQR): random poles {q(nc)} | judged unrelated {q(cs[torch.tensor(un)])} | related "
        f"{q(torch.tensor([x['cos'] for x in res if x['p_un'] < 0.5 and x['p_same'] < 0.5]))} | same {q(torch.tensor([x['cos'] for x in res if x['p_same'] >= 0.5]))}")
    push = {(x["j"], x["src"]): next(pu for s, pu, _ in P["sources"][x["j"]] if tuple(s) == x["src"]) for x in res}
    pp = torch.tensor(list(push.values())).quantile(torch.tensor([0.25, 0.5, 0.75]))
    for qi in range(4):
        lo, hi = (-1e9 if qi == 0 else float(pp[qi - 1])), (1e9 if qi == 3 else float(pp[qi]))
        r = [x for x in res if lo <= push[(x["j"], x["src"])] < hi]
        say(f"  push quartile {qi + 1} (n={len(r)}): {A([-x['cos'] for x in r], [x['p_un'] >= 0.5 for x in r]):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("measure", "cards", "test", "submit", "collect", "score", "inherit", "weights", "labels"))
    ap.add_argument("n", nargs="?", type=int, default=2, help="test: cards to run")
    ap.add_argument("--floor", type=float, default=None, help="measure: propose from in-edges past floor/√d (16_edges.py's "
                    "computation) in place of the edges table's 5/√d")
    args = ap.parse_args()
    globals()[args.part](args)
    (SUMMARY / f"secondary_modes_{args.part}_{STEM}.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
