"""Do the firing labels predict a pole's activation token by token (findings §8)? Simulation scoring.

Bills et al. (2023)'s test of an explanation, on the poles ``relabel_check.py --poles all`` scored by detection
(``data/summary/label_check_<model>.json``), so each pole gets both scores. A simulator (Claude Sonnet) sees only a
``fires`` label and predicts the pole's activation, 0 to 10, at every token of ``3 × PER_KIND`` excerpts of ``WIN``
tokens, shuffled: around held-out top firings (ranks 6-24, off every card), around weak firings (a peak in ``WEAK`` of
``act_top``, under the top regime the detection test draws from) and at random, the last two from random documents
outside the pole's top 64. True activations come from a forward pass over each excerpt's whole document, the pole's sign
applied and clipped at zero. The score is the correlation of simulated with true activations, over all excerpts (Bills's
top-and-random score) and within each kind. Three labels are simulated on the same excerpts: the pole's own, its label
before the lean card (the release at ``OLD``), and another firing pole of the layer's (the control). Beside the
correlation, coarser reads of the same predictions: the correlation of each excerpt's predicted and true peaks (which
excerpts fire, and how strongly, wherever in them), whether an excerpt's predicted peak ranks top or weak excerpts above
random ones without a firing (AUROC), and how often a top excerpt's predicted peak falls within a token of the true one.

    python scripts/checks/simulate_labels.py excerpts [--device mps]    # true activations: a forward pass, minutes
    python scripts/checks/simulate_labels.py run [--poles 10]           # the simulator (Batch API), then the scores
    python scripts/checks/simulate_labels.py run --batch msgbatch_...   # resume a submitted batch
"""
import argparse
import importlib.util
import json
import math
import random
import re
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from neuron_lens import arch
from neuron_lens.catalog import (
    HELD_OUT,
    MODEL,
    N_CONTEXTS,
    documents,
    neuron_table,
    tokenizer,
)
from neuron_lens.paths import DATA, SUMMARY

spec = importlib.util.spec_from_file_location("labels", Path(__file__).resolve().parent.parent / "10_labels.py")
labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels)

NAME = arch.stem(MODEL)
OUT = DATA / "simulate" / NAME
POOL = 2000  # random documents the weak and random excerpts come from
WIN = 64  # tokens an excerpt shows (Bills's length)
PER_KIND = 5
KINDS = ("top", "weak", "random")
WEAK = (0.25, 0.5)  # a weak excerpt's peak, as a share of act_top (a quarter is where the catalog's rate counts a firing)
FIRES = 0.25  # an excerpt holds a firing when its true peak passes this share of act_top
PRICE = {"input": 1.0, "output": 5.0}  # $/MTok, Claude Sonnet 5.5's batch prices
ARMS = ("own", "earlier", "control")
RELEASE, OLD = "stdlogits/neuron-lens", "37da15370817"  # the release before the lean card's run (findings §8)
SYSTEM = """You simulate one neuron of a language model. Given a description of where the neuron fires, predict its \
activation at every token of some text excerpts, from 0 to 10: 0 where it is silent, 10 for its strongest typical firing. \
A neuron is silent on most tokens, and only some excerpts hold a firing. Its activation at a token depends only on that \
token and the text before it, never on the text after. Each excerpt comes as its text, then its tokens, numbered from 0 \
(newlines shown as ⏎). Reply with one JSON object and nothing else, a key per excerpt number: \
{"1": [[i, a], ...], "2": [], ...}, the tokens i you predict above 0 with their activation a (1 to 10); an excerpt you \
predict silent gets []."""


def pole_key(pid: str) -> tuple[int, int, int]:
    m = re.match(r"L(\d+)_n(\d+)_([pm])$", pid)
    return int(m[1]), int(m[2]), 1 if m[3] == "p" else -1


def checked_poles() -> list[str]:
    return list(json.loads((SUMMARY / f"label_check_{NAME}.json").read_text())["poles"])


def acts_rows(poles: list[str]) -> dict:
    out = {}
    for l in sorted({pole_key(p)[0] for p in poles}):
        t = pq.read_table(neuron_table("acts", l), columns=["neuron", "pole", "act_top", "top_doc", "top_pos", "top_val"]).to_pylist()
        by = {(r["neuron"], r["pole"]): r for r in t}
        for p in poles:
            L, n, s = pole_key(p)
            if L == l:
                out[p] = by[n, s]
    return out


def window(rng, pos: int, length: int) -> int:
    """A window's start holding ``pos``, at a random offset from 16 tokens in to 4 before the end."""
    if length <= WIN:
        return 0
    return min(max(pos - rng.randint(16, WIN - 4), 0), length - WIN)


def excerpts(args) -> None:
    poles = checked_poles()
    rows = acts_rows(poles)
    rng = random.Random(1117)
    docs = documents()
    tops = {p: [(rows[p]["top_doc"][j], rows[p]["top_pos"][j], rows[p]["top_val"][j]) for j in
                sorted(rng.sample(range(N_CONTEXTS, HELD_OUT), PER_KIND))] for p in poles}
    pool = rng.sample(range(len(docs)), POOL)
    run = sorted(set(pool) | {d for t in tops.values() for d, _, _ in t})
    cols = {l: [p for p in poles if pole_key(p)[0] == l] for l in sorted({pole_key(p)[0] for p in poles})}
    dev = torch.device(args.device)
    _, stack = arch.load_model(MODEL, args.device, eager=False)
    stack.layers = stack.layers[:max(cols) + 1]
    got = {}
    for l, ps in cols.items():
        idx = torch.tensor([pole_key(p)[1] for p in ps], device=dev)
        sign = torch.tensor([pole_key(p)[2] for p in ps], device=dev, dtype=torch.float32)
        stack.layers[l].mlp.down_proj.register_forward_pre_hook(
            lambda m, x, l=l, idx=idx, sign=sign: got.__setitem__(l, (x[0][..., idx].float() * sign).clamp_min(0).half().cpu()))
    act = {}  # doc -> {pole: [T] activations}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(run), args.batch):
            ds = run[i:i + args.batch]
            T = max(len(docs[d]) for d in ds)
            ids = torch.zeros((len(ds), T), dtype=torch.long)
            for k, d in enumerate(ds):  # right-padded: a causal model's earlier positions don't see the padding
                ids[k, :len(docs[d])] = docs[d]
            stack(input_ids=ids.to(dev))
            for k, d in enumerate(ds):
                act[d] = {p: got[l][k, :len(docs[d]), j] for l, ps in cols.items() for j, p in enumerate(ps)}
            if i // args.batch % 20 == 0:
                print(f"{time.strftime('%H:%M:%S')} {i + len(ds)}/{len(run)} documents, {time.time() - t0:.0f}s", flush=True)
    out = {}
    for p in poles:
        r, top = rows[p], set(rows[p]["top_doc"])
        a_top = r["act_top"]
        ex, check = [], []
        for d, pos, val in tops[p]:
            check.append(float(act[d][p][pos]) / val if val > 0 else math.nan)
            s = window(rng, pos, len(docs[d]))
            ex.append(("top", d, s))
        free = [d for d in pool if d not in top and len(docs[d]) >= WIN]
        weak = []
        for d in rng.sample(free, len(free)):
            if len(weak) == PER_KIND:
                break
            v = act[d][p].float()
            at = ((v >= WEAK[0] * a_top) & (v < WEAK[1] * a_top)).nonzero().flatten().tolist()
            if not at:
                continue
            s = window(rng, rng.choice(at), len(v))
            if v[s:s + WIN].max() < WEAK[1] * a_top:
                weak.append(("weak", d, s))
        ex += weak
        for d in rng.sample(free, PER_KIND):
            ex.append(("random", d, rng.randint(0, len(docs[d]) - WIN)))
        out[p] = {"act_top": a_top, "top_check": check, "excerpts": [
            {"kind": k, "doc": d, "start": s, "ids": docs[d][s:s + WIN].tolist(), "act": act[d][p][s:s + WIN].tolist()}
            for k, d, s in ex]}
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT / "excerpts.pt")
    ratio = [x for v in out.values() for x in v["top_check"] if not math.isnan(x)]
    n_weak = [sum(e["kind"] == "weak" for e in v["excerpts"]) for v in out.values()]
    print(f"{len(out)} poles; top firings recomputed / catalog: median {sorted(ratio)[len(ratio) // 2]:.3f}, "
          f"range {min(ratio):.3f}-{max(ratio):.3f}; weak excerpts per pole: {sum(n_weak) / len(n_weak):.2f} "
          f"(none for {sum(n == 0 for n in n_weak)}); {time.time() - t0:.0f}s", flush=True)


def controls(poles: list[str], rng) -> dict:
    """Per pole, the label of another firing pole of its layer, drawn at random."""
    out = {}
    for l in sorted({pole_key(p)[0] for p in poles}):
        lab = {(r["neuron"], r["pole"]): r for r in pq.read_table(neuron_table("labels", l)).to_pylist()}
        a = pq.read_table(neuron_table("acts", l), columns=["neuron", "pole", "act_top"]).to_pylist()
        firing = [(r["neuron"], r["pole"]) for r in a if r["act_top"] >= labels.SILENT and lab[r["neuron"], r["pole"]]["fires"]]
        for p in poles:
            L, n, s = pole_key(p)
            if L == l:
                out[p] = (lab[n, s], lab[rng.choice([k for k in firing if k != (n, s)])])
    return out


def earlier(layer: int) -> dict:
    """The release's labels of this layer at ``OLD``, the labels the lean card replaced (as ``relabel_check.py --old``)."""
    path = hf_hub_download(RELEASE, f"{NAME}/neurons/labels/L{layer:02d}.parquet", repo_type="dataset", revision=OLD)
    return {(r["neuron"], r["pole"]): r for r in pq.read_table(path).to_pylist()}


def shown(tok, e: dict) -> str:
    text = tok.decode(e["ids"]).replace("\n", "⏎")
    toks = " | ".join(f"{i} {json.dumps(tok.decode([t]).replace(chr(10), '⏎'), ensure_ascii=False)}" for i, t in enumerate(e["ids"]))
    return f"text: {text}\ntokens: {toks}"


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sy == 0:
        return math.nan  # nothing to predict
    if sx == 0:
        return 0.0  # predicted flat where the truth varies: no information
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman(x: list[float], y: list[float]) -> float:
    rank = lambda v: [sorted(v).index(a) + (sorted(v).count(a) - 1) / 2 for a in v]
    return pearson(rank(x), rank(y))


def last_object(text: str) -> dict | None:
    """The reply's last JSON object: a simulator that corrects itself writes the corrected object last."""
    dec, out, i = json.JSONDecoder(), None, text.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            out = obj
        i = text.find("{", end)
    return out


def predictions(obj, n: int) -> list[list[float]] | None:
    """Each excerpt's predicted activations from the simulator's reply, or None if it can't be read."""
    if not isinstance(obj, dict):
        return None
    out = []
    for k in range(1, n + 1):
        row = [0.0] * WIN
        for item in obj.get(str(k), []) or []:
            try:
                i, a = int(item[0]), float(item[1])
            except (TypeError, ValueError, IndexError):
                return None
            if 0 <= i < WIN:
                row[i] = a
        out.append(row)
    return out


def score(pred: list[list[float]], ex: list[dict], a_top: float) -> dict:
    true = [e["act"] + [0.0] * (WIN - len(e["act"])) for e in ex]
    flat = lambda rows: [x for r in rows for x in r]
    out = {"all": pearson(flat(pred), flat(true)), "excerpt": pearson([max(r) for r in pred], [max(t) for t in true])}
    for k in KINDS:
        sel = [i for i, e in enumerate(ex) if e["kind"] == k]
        if sel:
            out[k] = pearson(flat(pred[i] for i in sel), flat(true[i] for i in sel))
    quiet = [max(pred[i]) for i, e in enumerate(ex) if e["kind"] == "random" and max(true[i]) < FIRES * a_top]
    for k in ("top", "weak"):
        hot = [max(pred[i]) for i, e in enumerate(ex) if e["kind"] == k]
        if hot and quiet:
            out[f"finds_{k}"] = labels.auroc(hot + quiet, [1] * len(hot) + [0] * len(quiet))
    lands = {"on": 0, "after_1_2": 0, "before_1_2": 0, "off_3_5": 0, "far": 0, "silent": 0}  # top excerpts' predicted peak
    hit = 0
    for i, e in enumerate(ex):
        if e["kind"] != "top":
            continue
        if max(pred[i]) == 0:
            lands["silent"] += 1
            continue
        d = max(range(WIN), key=pred[i].__getitem__) - max(range(WIN), key=true[i].__getitem__)
        lands["on" if d == 0 else "after_1_2" if 0 < d <= 2 else "before_1_2" if -2 <= d < 0 else "off_3_5" if abs(d) <= 5 else "far"] += 1
        hit += abs(d) <= 1
    n_top = sum(lands.values())
    out["peak"] = hit / n_top if n_top else math.nan
    out["lands"] = lands
    return out


def run(args) -> None:
    ex = torch.load(OUT / "excerpts.pt", weights_only=False)
    rng = random.Random(1118)
    poles = checked_poles()[:args.poles] if args.poles else checked_poles()
    poles = [p for p in poles if p in ex]
    pairs = controls(poles, rng)
    old = {l: earlier(l) for l in sorted({pole_key(p)[0] for p in poles})}
    tok = tokenizer()
    order, reqs, conf = {}, [], {}
    for p in poles:
        own, ctl = pairs[p]
        if not own["fires"]:  # a pole the labeller left empty: no label to score
            continue
        o = list(range(len(ex[p]["excerpts"])))
        rng.shuffle(o)
        order[p], conf[p] = o, own["fires_conf"]
        body = "\n\n".join(f"Excerpt {k + 1}\n{shown(tok, ex[p]['excerpts'][i])}" for k, i in enumerate(o))
        prev = old[pole_key(p)[0]].get(pole_key(p)[1:])
        for arm, desc in (("own", own), ("earlier", prev), ("control", ctl)):
            if not desc or not desc["fires"]:
                continue
            reqs.append({"custom_id": f"{arm}_{p}", "params": {
                "model": labels.SCORER, "max_tokens": 16000, "system": SYSTEM,
                "messages": [{"role": "user", "content": f"The neuron fires: {desc['fires']}\n\n{body}"}]}})
    c = labels.client()
    if args.batch:
        batch_id = args.batch
    else:
        batch_id = c.messages.batches.create(requests=reqs).id
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "batches.jsonl").open("a").write(json.dumps({"batch": batch_id, "poles": len(order), "requests": len(reqs)}) + "\n")
    sent = {r["custom_id"] for r in reqs}
    print(f"simulation batch {batch_id}: {len(reqs)} requests", flush=True)
    labels.wait(c, batch_id)
    got, usage, failed = {}, {"input_tokens": 0, "output_tokens": 0}, 0
    with open(OUT / f"results-{batch_id}.jsonl", "w") as fh:
        for res in c.messages.batches.results(batch_id):
            if res.result.type != "succeeded":
                failed += 1
                continue
            msg = res.result.message
            for k in usage:
                usage[k] += getattr(msg.usage, k)
            text = labels.text_of(msg)
            fh.write(json.dumps({"id": res.custom_id, "stop": msg.stop_reason, "text": text}, ensure_ascii=False) + "\n")
            got[res.custom_id] = last_object(text)
    cost = (usage["input_tokens"] * PRICE["input"] + usage["output_tokens"] * PRICE["output"]) / 1e6
    detect = json.loads((SUMMARY / f"label_check_{NAME}.json").read_text())["poles"]
    per, unread = {}, 0
    for p, o in order.items():
        e = [ex[p]["excerpts"][i] for i in o]
        out = {"layer": pole_key(p)[0], "n_weak": sum(x["kind"] == "weak" for x in e), "conf": conf[p],
               "detection": detect[p].get("new")}
        for arm in ARMS:
            if f"{arm}_{p}" not in sent:
                continue
            pred = predictions(got.get(f"{arm}_{p}"), len(e))
            if pred is None:
                unread += 1
                continue
            out[arm] = score(pred, e, ex[p]["act_top"])
        per[p] = out
    stat = lambda xs: (sum(xs) / len(xs), *labels.boot(xs)) if xs else None
    vals = lambda arm, k: [v[arm][k] for v in per.values() if arm in v and not math.isnan(v[arm].get(k, math.nan))]
    summary = {arm: {k: stat(vals(arm, k)) for k in ("all", "excerpt", *KINDS, "finds_top", "finds_weak", "peak")} for arm in ARMS}
    for arm in ("control", "earlier"):
        both = [v for v in per.values() if "own" in v and arm in v]
        summary[f"own_minus_{arm}"] = stat([v["own"]["all"] - v[arm]["all"] for v in both])
    hi = [v["own"]["all"] for v in per.values() if "own" in v and v["conf"] >= 0.7]
    lo = [v["own"]["all"] for v in per.values() if "own" in v and v["conf"] < 0.4]
    summary["own_all_by_conf"] = {"0.7+": stat(hi), "under 0.4": stat(lo)}
    lands = [v["own"]["lands"] for v in per.values() if "own" in v]
    total = sum(sum(x.values()) for x in lands)
    summary["own_top_peak_lands"] = {k: sum(x[k] for x in lands) / total for k in lands[0]}
    summary["own_poles_by_top_exact"] = {h: sum(x["on"] == h for x in lands) for h in range(PER_KIND + 1)}
    near = [round(v["own"]["peak"] * PER_KIND) for v in per.values() if "own" in v]
    summary["own_poles_by_top_within_1"] = {h: near.count(h) for h in range(PER_KIND + 1)}
    quarter = lambda layer: min(4 * layer // labels.LAYERS, 3)
    summary["own_all_by_quarter"] = {q: stat(xs) for q in range(4) if (xs := [v["own"]["all"] for v in per.values()
                                                                                if "own" in v and quarter(v["layer"]) == q])}
    paired = [(v["own"]["all"], v["detection"]) for v in per.values() if "own" in v and v["detection"] is not None]
    summary["spearman_with_detection"] = spearman(*map(list, zip(*paired))) if len(paired) > 2 else None
    summary["cost_usd"], summary["usage"], summary["failed"], summary["unread"] = cost, usage, failed, unread
    path = SUMMARY / f"label_simulation_{NAME}.json"
    path.write_text(json.dumps({"model": MODEL, "batch": batch_id, "simulator": labels.SCORER, "window": WIN,
                                "per_kind": PER_KIND, "weak": WEAK, "summary": summary, "poles": per}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("excerpts")
    a.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
    a.add_argument("--batch", type=int, default=16)
    b = sub.add_parser("run")
    b.add_argument("--poles", type=int, default=None, help="the first this many of the checked poles (a pilot)")
    b.add_argument("--batch", default=None, help="a submitted batch to resume")
    args = ap.parse_args()
    excerpts(args) if args.cmd == "excerpts" else run(args)


if __name__ == "__main__":
    main()
