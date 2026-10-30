"""Blind interpretation of random poles: how far do the lenses alone go? (findings §8)

A stratified random sample of poles (``PER_BAND`` from distinct neurons in each of §7's bands) is
described once per arm (``ARMS``: the arm's card kind and its explainers' model):

- **lens:** the weight-only reads of the lenses of record and nothing else: the R-lens and S-lens
  reads of the write, the forward R-lens (mid target) and S-lens reads of the read direction,
  plain and layer-contrastive;
- **contexts:** the pole's top ``N_CONTEXTS`` contexts and firing rates, with no lens read (the
  reference: what activations alone give on the same poles);
- **both** and **opus:** both on one card, explained by Sonnet and by Opus (how far an explainer
  blind to steering gets). Scorers and judges are Sonnet in every arm.

Each explainer says what the pole fires on and what it promotes when its write is added. Two tests
on the model check both labels, the second on a CUDA GPU:

- **firing** (detection, as in §8's description pilot): ``N_POS`` held-out top contexts (top
  documents past the first ``N_CONTEXTS``, which no card shows) against quiet positions, both random
  ones (``N_NEG_RAND``, as in the pilot) and ones in the same documents as the positives
  (``N_NEG_SAME``: same topic and genre, so a topic label does not separate them). A scorer who
  sees only a label gives each marked token a probability of firing; the score is the AUROC,
  against a control label from another pole of the band and arm (a derangement).
- **steering:** greedy continuations of ``PROMPTS`` with the pole's write ``s·a·W_down[:, n]`` added
  to the layer's MLP output at every position, at the doses ``a`` where the mean next-token KL over
  the prompts' positions reaches each of ``KL_TARGETS`` (the same visible change for every pole,
  silent ones included; a pilot of 8 poles found 0.5 nats subtle, 2 often degenerate, 4 gibberish).
  A judge sees the unsteered and steered text and four "promotes" labels, the pole's and three
  other poles' of its band and arm, and gives each a probability of being the pole's own. Chance
  is 0.25. Judge-free beside it: how many steered tokens are in the R-lens's top 64 for the write,
  against the same three other poles' top 64.

Stages (``data/blind/<model>/``):
- ``cards`` (any device): the sample, ``cards/{lens,contexts,both}_XX.md`` in one shuffled order
  (each explainer sees mixed bands), ``poles.json``, and ``gpu_in.pt`` (documents and positives)
  for ``measure``;
- ``measure`` (GPU): activations of the sampled poles over the positives' documents and random
  ones, the quiet negatives, and the steered continuations, to ``measured.pt``;
- explainers (Claude Code subagents briefed with ``lens_blind_explain.md``) write
  ``cards/labels_<arm>_*.jsonl``;
- ``prompt``: the explainers' labels and the measurements, to scorer tasks
  ``tasks/score_<arm>_XX.md`` and judge tasks ``tasks/judge_<arm>_XX.md`` (one arm per file, so no
  scorer sees a pole under two arms' labels) and ``tasks/key.json``. Arms run in ``ARMS`` order
  from one seeded generator and an arm with no labels draws nothing from it, so adding an arm
  leaves the earlier arms' tasks, ids and answers as they were. Scorers (``lens_blind_score.md``)
  and judges (``lens_blind_judge.md``) write ``scored_*.jsonl`` and ``judged_*.jsonl`` beside them;
- ``grade``: ``scored_*.jsonl`` and ``judged_*.jsonl`` in ``tasks/`` against the key, to
  ``data/summary/lens_blind_<model>.{log,json}`` and a row per pole in ``..._poles.jsonl``.

    python scripts/checks/lens_blind.py cards
    python scripts/checks/lens_blind.py measure --device cuda   # on a CUDA GPU
    python scripts/checks/lens_blind.py prompt
    python scripts/checks/lens_blind.py grade
"""
import argparse
import json
import random
import time
from collections import defaultdict
from statistics import mean, median

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_lens.catalog import (
    MODEL,
    N_CONTEXTS,
    acts_path,
    context,
    doc_tokens,
    load_layer,
)
from neuron_lens.paths import DATA, SUMMARY

NAME = MODEL.split("/")[-1].lower()
OUT = DATA / "blind" / NAME


def pole_id(l: int, n: int, s: int) -> str:
    return f"L{l} n{n} {'+' if s > 0 else '−'}"


def auroc(p: list[float], truth: list[int]) -> float:
    pos = [x for x, t in zip(p, truth) if t]
    neg = [x for x, t in zip(p, truth) if not t]
    return mean((a > b) + 0.5 * (a == b) for a in pos for b in neg)


BANDS = [(0, 8), (9, 12), (13, 20), (21, 27), (28, 35)]
PER_BAND = 32
SILENT = 0.05  # act_top under this: no firing test (§7, §8)
N_SHOW = 12  # tokens per read on a lens card
N_POS, N_NEG_RAND, N_NEG_SAME = 6, 3, 3
N_RANDOM_DOCS = 160
MIN_POS = 8  # negatives need some context before them
CARDS_PER_FILE, SCORE_PER_FILE, JUDGE_PER_FILE = 40, 75, 40
ARMS = {"lens": ("lens", "sonnet"), "contexts": ("contexts", "sonnet"), "both": ("both", "sonnet"),
        "opus": ("both", "opus")}  # arm -> (card kind, explainer model)
LENS_READS = (
    ("write_R_ids", "writes (R-lens of record: what the write promotes downstream)"),
    ("write_R_z_ids", "writes, layer-contrastive (R-lens z)"),
    ("write_S_ids", "writes, as input tokens (S-lens of the write: tokens whose footprint it resembles)"),
    ("write_S_z_ids", "writes, as input tokens, layer-contrastive (S-lens z)"),
    ("read_R_ids", "reads (forward R-lens of the read direction, from the layer's MLP input)"),
    ("read_R_z_ids", "reads, layer-contrastive (forward R-lens z)"),
    ("read_S_ids", "reads, as input tokens (S-lens: tokens here or earlier that drive it)"),
    ("read_S_z_ids", "reads, as input tokens, layer-contrastive (S-lens z)"),
)
PROMPTS = (
    ("narrative", "I woke up this morning and"),
    ("word", 'My favorite word is "'),
    ("chinese", "今天我想写一写"),
    ("chat", None),  # a chat turn, built from CHAT_TURN with the model's template
)
CHAT_TURN = "Tell me about something you find interesting."
KL_TARGETS = (0.5, 1.0, 2.0)  # nats, mean over the prompts' positions
DOSE_TAGS = ("light", "moderate", "strong")
DOSE_GRID = torch.logspace(-1, 4.5, 34)  # in activation units (× W_down[:, n])
NEW_TOKENS = 40


def band_of(layer: int) -> str:
    lo, hi = next(b for b in BANDS if b[0] <= layer <= b[1])
    return f"L{lo}-L{hi}"


def cards(args) -> None:
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    show = lambda ids: " | ".join(tok.decode([i]).replace("\n", "⏎") for i in ids[:N_SHOW])
    f = 12288
    sample = []
    for lo, hi in BANDS:
        seen = set()
        while len(seen) < PER_BAND:
            l, n = rng.randint(lo, hi), rng.randrange(f)
            if (l, n) not in seen:
                seen.add((l, n))
                sample.append((l, n, rng.choice((1, -1))))
    tokens = doc_tokens(torch.load(acts_path(), weights_only=False, map_location="cpu", mmap=True))
    by_layer = defaultdict(list)
    for i, (l, n, s) in enumerate(sample):
        by_layer[l].append(i)
    poles = [None] * len(sample)
    cards = {kind: [None] * len(sample) for kind in ("lens", "contexts", "both")}
    for l, idx in sorted(by_layer.items()):
        t = load_layer(l)
        rows = [sample[i][1] if sample[i][2] > 0 else len(t) // 2 + sample[i][1] for i in idx]
        for i, e in zip(idx, t.take(rows).to_pylist()):
            _, n, s = sample[i]
            pid = pole_id(l, n, s)
            head = [f"### {pid}", f"- layer {l} of 36"]
            lens = [f"- {title}: {show(e[key])}" for key, title in LENS_READS]
            cards["lens"][i] = "\n".join(head + lens)
            silent = e["act_top"] < SILENT
            lines = head + [f"- activation: typical high {e['act_top']:.2f}, max {e['act_max']:.2f}; fires (past a "
                            f"quarter of typical) at {e['rate']:.1e} of positions; by slice: "
                            + ", ".join(f"{k} {e[f'rate_{k}']:.1e}" for k in ("dclm", "zh", "code", "chat"))]
            if silent:
                lines.append("- near-silent: no top context recorded")
            else:
                lines += [f"  - context: {c}" for c in str(e["contexts"]).split(" ‖ ")[:N_CONTEXTS] if c]
            cards["contexts"][i] = "\n".join(lines)
            cards["both"][i] = "\n".join(head + lens + lines[len(head):])
            fired = [j for j, v in enumerate(e["top_val"]) if v > 0]
            held = fired[N_CONTEXTS:]  # the contexts card showed the first N_CONTEXTS
            pos = [] if silent else [(int(e["top_doc"][j]), int(e["top_pos"][j]), float(e["top_val"][j]))
                                     for j in sorted(rng.sample(held, min(N_POS, len(held))))]
            poles[i] = {"id": pid, "layer": l, "neuron": n, "sign": s, "band": band_of(l),
                        "act_top": e["act_top"], "silent": silent, "pos": pos,
                        "write_R_top": e["write_R_ids"]}
        print(f"L{l}: {len(idx)} poles", flush=True)
    order = list(range(len(sample)))
    rng.shuffle(order)
    (OUT / "cards").mkdir(parents=True, exist_ok=True)
    for old in (OUT / "cards").glob("*.md"):
        old.unlink()
    for kind, cs in cards.items():
        for b in range(0, len(order), CARDS_PER_FILE):
            (OUT / "cards" / f"{kind}_{b // CARDS_PER_FILE:02d}.md").write_text(
                "\n\n".join(cs[i] for i in order[b:b + CARDS_PER_FILE]) + "\n")
    (OUT / "poles.json").write_text(json.dumps(poles, ensure_ascii=False))
    docs = {d for p in poles for d, _, _ in p["pos"]}
    long_docs = [d for d in range(len(tokens)) if len(tokens[d]) >= 64]
    random_docs = sorted(rng.sample(long_docs, N_RANDOM_DOCS))
    torch.save({"poles": poles, "random_docs": random_docs,
                "docs": {d: tokens[d].tolist() for d in sorted(docs | set(random_docs))}}, OUT / "gpu_in.pt")
    n_silent = sum(p["silent"] for p in poles)
    print(f"{len(poles)} poles ({n_silent} silent), {len(docs)} positive documents, {N_RANDOM_DOCS} random -> {OUT}")


def prompt_ids(tok) -> list[list[int]]:
    out = []
    for name, text in PROMPTS:
        if text is None:
            text = tok.apply_chat_template([{"role": "user", "content": CHAT_TURN}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        out.append(tok(text, add_special_tokens=False)["input_ids"])
    return out


def measure(args) -> None:
    device = args.device
    inp = torch.load(OUT / "gpu_in.pt", weights_only=False)
    poles, docs = inp["poles"], inp["docs"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()
    lm.requires_grad_(False)
    layers = lm.model.layers
    rng = random.Random(args.seed)
    out = {"meta": {"model": MODEL, "kl_targets": KL_TARGETS, "prompts": PROMPTS, "chat_turn": CHAT_TURN,
                    "new_tokens": NEW_TOKENS, "dtype": "bfloat16"}}

    # Activations of every sampled pole (signed) at every position of the documents.
    at = defaultdict(list)  # layer -> [(pole index, neuron, sign)]
    for i, p in enumerate(poles):
        at[p["layer"]].append((i, p["neuron"], p["sign"]))
    ids = sorted(docs)
    acts = {}
    grab = {}

    def hook(layer):
        cols = torch.tensor([n for _, n, _ in at[layer]], device=device)
        signs = torch.tensor([s for _, _, s in at[layer]], device=device, dtype=torch.float32)

        def f(mod, a):
            grab[layer] = (a[0][..., cols].float() * signs).cpu()
        return f
    hs = [layers[l].mlp.down_proj.register_forward_pre_hook(hook(l)) for l in at]
    pad = tok.pad_token_id
    t0 = time.perf_counter()
    for b in range(0, len(ids), args.batch):
        part = ids[b:b + args.batch]
        T = max(len(docs[d]) for d in part)
        x = torch.full((len(part), T), pad)
        mask = torch.zeros(len(part), T, dtype=torch.long)
        for j, d in enumerate(part):
            x[j, :len(docs[d])] = torch.tensor(docs[d])
            mask[j, :len(docs[d])] = 1
        with torch.inference_mode():
            lm.model(input_ids=x.to(device), attention_mask=mask.to(device))
        for j, d in enumerate(part):
            a = torch.zeros(len(docs[d]), len(poles))
            for l, cols in at.items():
                a[:, [i for i, _, _ in cols]] = grab[l][j, :len(docs[d])]
            acts[d] = a.half()
    for h in hs:
        h.remove()
    print(f"{time.strftime('%H:%M:%S')} activations of {len(poles)} poles over {len(ids)} documents, "
          f"{time.perf_counter() - t0:.0f}s", flush=True)

    sets = {}
    for i, p in enumerate(poles):
        if p["silent"]:
            continue
        quiet = p["act_top"] / 10

        def quiet_at(ds, k):
            q = [(d, t) for d in ds for t in (acts[d][MIN_POS:, i] < quiet).nonzero().squeeze(1).add(MIN_POS).tolist()]
            return rng.sample(q, min(k, len(q)))
        pos = [(d, t, v, float(acts[d][t, i])) for d, t, v in p["pos"]]
        same = quiet_at(sorted({d for d, _, _ in p["pos"]}), N_NEG_SAME)
        rand = quiet_at(inp["random_docs"], N_NEG_RAND)
        items = [(context(tok, torch.tensor(docs[d]), t, after=0), "pos", v) for d, t, v, _ in pos]
        items += [(context(tok, torch.tensor(docs[d]), t, after=0), kind, float(acts[d][t, i]))
                  for kind, picked in (("same", same), ("rand", rand)) for d, t in picked]
        rng.shuffle(items)
        sets[p["id"]] = {"items": items, "recheck": [(v, r) for _, _, v, r in pos]}
    out["sets"] = sets
    rc = [(v, r) for s in sets.values() for v, r in s["recheck"]]
    print(f"positives rechecked: {len(rc)}, median ratio recomputed/catalog "
          f"{torch.tensor([r / v for v, r in rc]).median():.3f}", flush=True)

    # Steering: calibrate each pole's dose on the prompts, then generate at each target.
    prompts = prompt_ids(tok)
    tok.padding_side = "left"
    gen_kw = dict(max_new_tokens=NEW_TOKENS, do_sample=False, temperature=None, top_p=None, top_k=None,
                  pad_token_id=pad)

    def batch_left(rows):
        T = max(len(r) for r in rows)
        x = torch.full((len(rows), T), pad)
        mask = torch.zeros(len(rows), T, dtype=torch.long)
        for j, r in enumerate(rows):
            x[j, T - len(r):] = torch.tensor(r)
            mask[j, T - len(r):] = 1
        return x.to(device), mask.to(device)

    def generate(layer, delta, rows):
        h = layers[layer].mlp.register_forward_hook(lambda mod, a, o: o + delta[:, None, :].to(o.dtype))
        x, mask = batch_left(rows)
        with torch.inference_mode():
            y = lm.generate(input_ids=x, attention_mask=mask, **gen_kw)[:, x.shape[1]:]
        h.remove()
        return [r[r != pad].tolist() for r in y.cpu()]

    d_model = layers[0].mlp.down_proj.weight.shape[0]
    base = generate(0, torch.zeros(len(prompts), d_model, device=device), prompts)
    out["base"] = base
    steer = {}
    grid = DOSE_GRID.to(device)
    t0 = time.perf_counter()
    for i, p in enumerate(poles):
        l, n, s = p["layer"], p["neuron"], p["sign"]
        w = s * layers[l].mlp.down_proj.weight[:, n].float()
        kl = torch.zeros(len(grid), device=device)
        for pr in prompts:
            delta = torch.cat([torch.zeros(1, d_model, device=device), grid[:, None] * w])  # row 0: unsteered
            h = layers[l].mlp.register_forward_hook(lambda mod, a, o: o + delta[:, None, :].to(o.dtype))
            with torch.inference_mode():
                lp = torch.log_softmax(lm(input_ids=torch.tensor([pr] * len(delta), device=device)).logits.float(), -1)
            h.remove()
            kl += (lp[0].exp() * (lp[0] - lp[1:])).sum(-1).mean(-1) / len(prompts)
        lg, lk = grid.log(), kl.clamp_min(1e-9).log()
        doses = []
        for target in KL_TARGETS:
            j = int((kl >= target).nonzero()[0]) if (kl >= target).any() else len(grid) - 1
            if 0 < j and kl[j] >= target:  # interpolate log dose in log KL between the bracketing grid points
                u = (torch.tensor(target).log().to(device) - lk[j - 1]) / (lk[j] - lk[j - 1])
                doses.append(float((lg[j - 1] + u * (lg[j] - lg[j - 1])).exp()))
            else:
                doses.append(float(grid[j]))
        rows = [pr for pr in prompts for _ in doses]
        delta = torch.stack([d * w for _ in prompts for d in doses])
        gen = generate(l, delta, rows)
        steer[p["id"]] = {"doses": doses, "kl_grid": kl.cpu().tolist(),
                          "gen": [[gen[k * len(doses) + j] for j in range(len(doses))] for k in range(len(prompts))]}
        if i % 10 == 0:
            print(f"{time.strftime('%H:%M:%S')} steer {i + 1}/{len(poles)} {p['id']}: doses "
                  f"{', '.join(f'{d:.3g}' for d in doses)} (act_top {p['act_top']:.3g}), "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    out["steer"], out["dose_grid"] = steer, DOSE_GRID.tolist()
    torch.save(out, OUT / "measured.pt")
    print(f"-> {OUT / 'measured.pt'}")


def read_jsonl(paths) -> dict:
    got = {}
    for path in sorted(paths):
        for line in path.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                got[o.get("id") or o["task"]] = o
    return got


def prompt(args) -> None:
    poles = {p["id"]: p for p in json.loads((OUT / "poles.json").read_text())}
    m = torch.load(OUT / "measured.pt", weights_only=False)
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = lambda ids: tok.decode(ids).replace("\n", "⏎")
    labels = {arm: read_jsonl((OUT / "cards").glob(f"labels_{arm}_*.jsonl")) for arm in ARMS}
    rng = random.Random(args.seed)
    band = defaultdict(list)
    for pid, p in poles.items():
        band[p["band"]].append(pid)
    control, others = {}, {}
    for ids in band.values():
        order = ids[:]
        rng.shuffle(order)
        control |= {a: order[(i + 1) % len(order)] for i, a in enumerate(order)}
        for pid in ids:
            others[pid] = rng.sample([x for x in ids if x != pid], 3)
    tasks_dir = OUT / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    files, key = {}, {}  # written at the end, only where changed: scorers may be reading them
    names = [name for name, _ in PROMPTS]
    shown = [t if t is not None else f"[chat] user: {CHAT_TURN} / assistant:" for _, t in PROMPTS]
    n_score = n_judge = 0
    for arm in ARMS:
        lab = labels[arm]
        score = []
        for pid in sorted(set(m["sets"]) & set(lab)):
            items = m["sets"][pid]["items"]
            for kind, src in (("matched", pid), ("control", control[pid])):
                if src in lab:
                    score.append({"pole": pid, "kind": kind, "label": lab[src]["fires"]["label"],
                                  "conf": lab[pid]["fires"]["confidence"], "items": items})
        rng.shuffle(score)
        for b in range(0, len(score), SCORE_PER_FILE):
            blocks = []
            for t in score[b:b + SCORE_PER_FILE]:
                tid = f"d{n_score:04d}"
                n_score += 1
                key[tid] = {"test": "fires", "arm": arm, "pole": t["pole"], "kind": t["kind"], "conf": t["conf"],
                            "truth": [x[1] for x in t["items"]]}
                blocks.append(f"### {tid}\nlabel: {t['label']}\n"
                              + "\n".join(f"{j + 1}. {x[0]}" for j, x in enumerate(t["items"])))
            files[f"score_{arm}_{b // SCORE_PER_FILE:02d}.md"] = "\n\n".join(blocks) + "\n"
        judge = [pid for pid in poles if pid in lab and all(o in lab for o in others[pid])]
        rng.shuffle(judge)
        for b in range(0, len(judge), JUDGE_PER_FILE):
            blocks = []
            for pid in judge[b:b + JUDGE_PER_FILE]:
                tid = f"s{n_judge:04d}"
                n_judge += 1
                opts = [pid] + others[pid]
                rng.shuffle(opts)
                key[tid] = {"test": "writes", "arm": arm, "pole": pid, "options": opts,
                            "conf": lab[pid]["writes"]["confidence"]}
                st = m["steer"][pid]
                lines = [f"### {tid}"]
                for k, name in enumerate(names):
                    lines.append(f"prompt {k + 1}: {shown[k]}")
                    lines.append(f"  unsteered: {text(m['base'][k])}")
                    for j, tag in enumerate(DOSE_TAGS):
                        lines.append(f"  {tag}: {text(st['gen'][k][j])}")
                lines += [f"{c}. {lab[o]['writes']['label']}" for c, o in zip("ABCD", opts)]
                blocks.append("\n".join(lines))
            files[f"judge_{arm}_{b // JUDGE_PER_FILE:02d}.md"] = "\n\n".join(blocks) + "\n"
        if lab:
            print(f"{arm}: labels for {len(lab)} poles, {len(score)} scoring tasks, {len(judge)} judging tasks")
    files["key.json"] = json.dumps(key, ensure_ascii=False)
    for old in tasks_dir.glob("*.md"):
        if old.name not in files:
            old.unlink()
    for name, body in files.items():
        if not (tasks_dir / name).exists() or (tasks_dir / name).read_bytes() != body.encode():
            (tasks_dir / name).write_bytes(body.encode())
            print(f"  wrote {name}")


def grade(args) -> None:
    poles = {p["id"]: p for p in json.loads((OUT / "poles.json").read_text())}
    key = json.loads((OUT / "tasks" / "key.json").read_text())
    got = read_jsonl((OUT / "tasks").glob("scored_*.jsonl")) | read_jsonl((OUT / "tasks").glob("judged_*.jsonl"))
    m = torch.load(OUT / "measured.pt", weights_only=False)
    labels = {arm: read_jsonl((OUT / "cards").glob(f"labels_{arm}_*.jsonl")) for arm in ARMS}
    labels = {arm: lab for arm, lab in labels.items() if lab}
    res = defaultdict(dict)  # (arm, pole) -> measures
    for tid, k in key.items():
        if tid not in got:
            continue
        p, r = got[tid]["p"], res[(k["arm"], k["pole"])]
        r["conf_" + k["test"]] = k["conf"]
        if k["test"] == "fires":
            if len(p) != len(k["truth"]):
                continue
            for neg in ("all", "rand", "same"):
                keep = [j for j, t in enumerate(k["truth"]) if t == "pos" or neg in ("all", t)]
                r[f"{k['kind']}_{neg}"] = auroc([p[j] for j in keep], [k["truth"][j] == "pos" for j in keep])
        elif len(p) == 4:
            q = [max(x, 0.0) for x in p]
            q = [x / sum(q) for x in q] if sum(q) > 0 else [0.25] * 4
            true = k["options"].index(k["pole"])
            r["p_true"] = q[true]
            r["top1"] = (q[true] == max(q)) / sum(x == max(q) for x in q)

    # Judge-free: steered tokens in the pole's own R-lens top 64, against three other poles' of its band.
    base = [t for g in m["base"] for t in g]
    for pid, st in m["steer"].items():
        gen = [[t for g in st["gen"] for t in g[j]] for j in range(len(KL_TARGETS))]
        own = set(poles[pid]["write_R_top"])
        hit = lambda toks, s: sum(t in s for t in toks) / max(1, len(toks))
        band = [q for q in poles if poles[q]["band"] == poles[pid]["band"] and q != pid]
        others = random.Random(pid).sample(band, 3)
        r = res[("lens", pid)]
        r["lens_hit"] = [hit(g, own) - hit(base, own) for g in gen]
        r["lens_hit_other"] = [mean(hit(g, set(poles[o]["write_R_top"])) - hit(base, set(poles[o]["write_R_top"]))
                                    for o in others) for g in gen]

    rng = random.Random(0)

    def interval(xs) -> tuple[float, float]:
        boot = sorted(mean(rng.choices(xs, k=len(xs))) for _ in range(2000))
        return boot[50], boot[1949]

    def ci(xs, row, key):
        row[f"{key}_ci"] = interval(xs)
        return f"{mean(xs):.2f} [{row[key + '_ci'][0]:.2f}, {row[key + '_ci'][1]:.2f}]"

    lines = [f"blind interpretation of random poles, {MODEL}: {len(poles)} poles, {PER_BAND} a band, "
             f"{sum(p['silent'] for p in poles.values())} silent; {len(got)} of {len(key)} tasks answered",
             "fires: AUROC of the scorer's probabilities on held-out top contexts against quiet ones, all / random "
             "/ same-document negatives, matched (control: another pole's label of the band and arm); "
             "holds = matched AUROC against all negatives >= 0.8",
             "writes: the judge's probability on the pole's own label among four of its band and arm (chance "
             "0.25), and top-1 accuracy; means with 95% bootstrap intervals over poles"]
    summary = {}
    for name, sel in [("all", lambda p: True)] + [(f"L{lo}-L{hi}", lambda p, lo=lo, hi=hi: lo <= p["layer"] <= hi)
                                                  for lo, hi in BANDS]:
        lines.append(f"\n{name}")
        for arm in labels:
            rs = [res[(arm, pid)] for pid, p in poles.items() if sel(p)]
            f = [r for r in rs if "matched_all" in r and "control_all" in r]
            w = [r for r in rs if "p_true" in r]
            row = {"n_fires": len(f), "n_writes": len(w)}
            if f:
                for neg in ("all", "rand", "same"):
                    row[f"fires_{neg}"] = mean(r[f"matched_{neg}"] for r in f)
                    row[f"fires_{neg}_control"] = mean(r[f"control_{neg}"] for r in f)
                row["fires_holds"] = mean(r["matched_all"] >= 0.8 for r in f)
                row["fires_holds_control"] = mean(r["control_all"] >= 0.8 for r in f)
                lines.append(f"  {arm:8s} fires  n {len(f):3d}  all {ci([r['matched_all'] for r in f], row, 'fires_all')} "
                             f"(control {row['fires_all_control']:.2f})  rand {row['fires_rand']:.2f} "
                             f"({row['fires_rand_control']:.2f})  same {row['fires_same']:.2f} "
                             f"({row['fires_same_control']:.2f})  holds {row['fires_holds']:.2f} "
                             f"({row['fires_holds_control']:.2f})")
            if w:
                row["writes_p"] = mean(r["p_true"] for r in w)
                row["writes_top1"] = mean(r["top1"] for r in w)
                lines.append(f"  {arm:8s} writes n {len(w):3d}  p(own) {ci([r['p_true'] for r in w], row, 'writes_p')}  "
                             f"top-1 {ci([r['top1'] for r in w], row, 'writes_top1')}  (chance 0.25)")
            fw = [r for r in f if "top1" in r]
            if fw:  # firing poles with both tests: how many labels hold on each side
                both = [(r["matched_all"] >= 0.8, r["top1"] == 1) for r in fw]
                row |= {"both": mean(a and b for a, b in both), "either": mean(a or b for a, b in both),
                        "neither": mean(not (a or b) for a, b in both)}
                lines.append(f"  {arm:8s} firing poles, fires holds and writes top-1: both {row['both']:.2f}, "
                             f"either {row['either']:.2f}, neither {row['neither']:.2f}")
            summary[f"{name}/{arm}"] = row
        firing = [pid for pid, p in poles.items() if sel(p) and not p["silent"]]
        if firing:
            ratio = [median(m["steer"][pid]["doses"][j] / poles[pid]["act_top"] for pid in firing)
                     for j in range(len(KL_TARGETS))]
            summary[f"{name}/dose_over_act_top"] = ratio
            lines.append("  doses, median multiple of act_top (firing poles): "
                         + ", ".join(f"{t:g} nats {r:.3g}×" for t, r in zip(KL_TARGETS, ratio)))
        hits = [res[("lens", pid)] for pid, p in poles.items() if sel(p) and "lens_hit" in res[("lens", pid)]]
        if hits:
            for j, tag in enumerate(DOSE_TAGS):
                own = [r["lens_hit"][j] for r in hits]
                oth = [r["lens_hit_other"][j] for r in hits]
                gt = mean((a > b) + 0.5 * (a == b) for a, b in zip(own, oth))  # ties (often both 0) count half
                summary[f"{name}/lens_hit_{tag}"] = {"own": mean(own), "other": mean(oth), "own_gt_other": gt,
                                                     "both_zero": mean(a == b == 0 for a, b in zip(own, oth))}
                lines.append(f"  judge-free, {tag}: steered tokens in the R top 64, over unsteered: own "
                             f"{mean(own):+.3f}, other poles' {mean(oth):+.3f}; own > other {gt:.2f} (ties half; "
                             f"chance 0.5)")
    lines.append("\nby the explainer's confidence (all bands)")
    for arm in labels:
        for test, metric in (("fires", "matched_all"), ("writes", "p_true")):
            for lo, hi in ((0, 0.4), (0.4, 0.7), (0.7, 1.01)):
                xs = [r[metric] for (a, _), r in res.items() if a == arm and metric in r
                      and lo <= r.get(f"conf_{test}", -1) < hi]
                if xs:
                    lines.append(f"  {arm:8s} {test:6s} confidence {lo:.1f}-{min(hi, 1):.1f}: n {len(xs):3d}, "
                                 f"{metric} {mean(xs):.2f}")
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"lens_blind_{NAME}.log").write_text(text + "\n")
    (SUMMARY / f"lens_blind_{NAME}.json").write_text(json.dumps(summary, indent=1))
    with open(SUMMARY / f"lens_blind_{NAME}_poles.jsonl", "w") as fh:
        for pid, p in poles.items():
            row = {k: p[k] for k in ("id", "band", "act_top", "silent")}
            row["doses"] = m["steer"][pid]["doses"]
            for arm in labels:
                lab = labels[arm].get(pid, {})
                row[arm] = {"fires": lab.get("fires"), "writes": lab.get("writes")} | {
                    k: v for k, v in res.get((arm, pid), {}).items()}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("cards", "measure", "prompt", "grade"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"cards": cards, "measure": measure, "prompt": prompt, "grade": grade}[args.what](args)


if __name__ == "__main__":
    main()
