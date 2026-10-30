"""Label every neuron pole and attention head with Claude Haiku 5.5 from its lens reads and top
contexts: the catalog's ``labels`` tables (pipeline stage 10).

**Neurons** (``--side neurons``): the evidence card per neuron (both poles) is the "both" card of
``checks/lens_blind.py`` (findings §8) made depth-aware: of the eight reads in ``neurons/reads``,
only those that carry signal at the neuron's layer (§6, §7), primary first, each to 64 tokens, in
rank tiers (1-8 ‖ 9-24 ‖ 25-64) with variants merged (``release×9``), the layer's hub tokens (in 3%+
of its raw top 64s) dropped from raw reads, a strength percentile among the layer's poles, the
tokens two lenses or the two sides share, and from L21 a note on peaked (high-kurtosis) writes
(§5); then ``neurons/acts``'
activation line and first 5 top contexts. Labels: ``fires`` and ``writes``, each with a confidence.
This arm was checked blind on 100 poles with ``checks/haiku_blind.py`` (§8). The full run
(2026-10-29, on the first release's tables): 44,244 requests of 10 neurons, $105.41 at batch prices
with thinking on (effort low), 1,071 poles' labels recovered from malformed JSON.

**Heads** (``--side heads``): the card per head shows its attention pattern scores, its output
scale, its OV reads (what it copies and suppresses; through the S-lens too through L8, §11), its
QK reads (what it attends to and avoids, the tokens that seek themselves), its exact drivers as the
query and as the key, and its first 5 top contexts with the query in ``⟦ ⟧`` and the attended token
in ``⟪ ⟫``. Labels: ``fires``, ``attends`` and ``writes``. Like the neuron cards, nothing on a card
summarizes the held-out top firings (ranks 6-24): no token counts or tests over them, and no
in-context output read. ``check`` tests the labels blind on ``CHECK_PER_BAND`` random heads a band:
- **firing:** a scorer (Claude Sonnet) sees a head's ``fires`` and ``attends`` labels and 8
  contexts, 4 of its held-out firings and 4 of another head of its band, shuffled, and gives each
  the probability that it is the described head's: the AUROC per head, against the same contexts
  scored under a third head's labels;
- **writes:** a judge sees what a head's firings actually write (``heads/acts``' ``out_R_z``, the
  in-context output read, not on the card) and picks the head's ``writes`` label out of four from
  its band (chance 25%).

The prompts are ``scripts/prompts/<model>/<side>.md``. Stages (under ``data/labels/<model>/<side>/``;
nothing is sent to the API without an explicit stage):

- ``cards``: ``cards/L<layer>.jsonl``, one card per neuron or head;
- ``pilot``: one batch of ``--n`` random requests; measures tokens (thinking included) and the parse
  rate and projects the full run's cost into ``state.json``;
- ``submit``: every request, in batches under the API's limits; refuses unless a pilot at the same
  thinking setting projected under ``--cap`` dollars and parsed 90%;
- ``status``, ``collect`` (labels to ``labels/L<layer>.jsonl``), ``retry`` (failed or unparsed
  requests, once);
- ``shards``: the ``labels`` table, ``neurons/labels/L<layer>.parquet`` or ``heads/labels.parquet``;
- ``check`` (heads): the blind check, into ``data/summary/labels_heads_<model>.{json,md}`` (the
  card's section).

    python scripts/10_labels.py cards --side heads
    python scripts/10_labels.py pilot --side heads [--thinking on|off]
    python scripts/10_labels.py submit --side heads --cap 5
    python scripts/10_labels.py collect --side heads; ... shards --side heads; ... check

The API key comes from ``ANTHROPIC_API_KEY`` or ``~/.config/anthropic/api_key``.
"""
import argparse
import bisect
import collections
import json
import os
import random
import re
import time
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from neuron_lens import arch
from neuron_lens.catalog import (
    MODEL,
    N_CONTEXTS,
    acts_path,
    context,
    corpus_doc_tokens,
    doc_tokens,
    head_keys,
    head_table,
    load_heads,
    load_layer,
    neuron_keys,
    neuron_table,
    tokenizer,
    write,
)
from neuron_lens.paths import DATA, ROOT, SUMMARY

NAME = arch.stem(MODEL)
LAYERS = arch.text_config(MODEL).num_hidden_layers
OUT = {side: DATA / "labels" / NAME / side for side in ("neurons", "heads")}
PROMPT = {side: ROOT / "scripts" / "prompts" / NAME / f"{side}.md" for side in OUT}
LABELLER, SCORER = "claude-haiku-5-5", "claude-sonnet-5-5"
PRICE = {"input": 0.05, "output": 0.25, "cache_read": 0.005, "cache_write": 0.0625}  # $/MTok, Haiku's batch prices
PER_REQUEST = {"neurons": 10, "heads": 8}  # cards per request
FIELDS = {"neurons": ("fires", "writes"), "heads": ("fires", "attends", "writes")}
MAX_TOKENS = 12000
MAX_REQUESTS, MAX_BYTES = 3000, 200_000_000  # per batch, under the API's 100,000 and 256 MB
SILENT = 0.05  # act_top under this: near-silent (§7, §8)
HUB = 0.03  # a raw read's token in this share of the layer's top 64s is a hub (§6)
READ = {"read_S": "in·S", "read_S_z": "in·S z", "read_R": "in·R", "read_R_z": "in·R z",
        "write_S": "out·S", "write_S_z": "out·S z", "write_R": "out·R", "write_R_z": "out·R z"}
ID = {"neurons": re.compile(r"^L(\d+) n(\d+) ([+−-])$"), "heads": re.compile(r"^L(\d+)h(\d+)$")}
# The reading plan, per model: by depth, (last layer, the reads shown primary first, the card's line on
# them), for the input and output side of a neuron and the OV side of a head; the layer from which a
# peaked write gets its note (§5); and the bands of the head check. Qwen3-8B's follows §6, §7 and §11,
# Gemma 4's its census in §13, in four bands: token (L0-L4), early (L5-L13), middle (L14-L32), late.
PLANS = {
    "qwen3-8b": {
        "in": [
            (4, ["read_S", "read_S_z"], "in·S names the literal tokens it fires on; in·R carries nothing here"),
            (7, ["read_S_z", "read_S"], "in·S is weaker than at L0-L4 but still the input read; in·R carries almost nothing"),
            (8, ["read_S", "read_S_z", "read_R", "read_R_z"], "the crossover: in·S and in·R both carry signal"),
            (12, ["read_R", "read_R_z"], "in·R is the input read from here; in·S has gone to noise"),
            (20, ["read_R_z", "read_R"], "the weakest band for every input read: expect little, say unclear freely"),
            (35, ["read_R", "read_R_z", "read_S_z"], "in·R is often the word about to come next; in·S z names concepts"),
        ],
        "out": [
            (4, ["write_R_z", "write_S_z"], "raw out·R is hub noise here; writes are weak this early"),
            (20, ["write_R", "write_R_z", "write_S_z"],
             "out·R is what its push makes the model say (it predicts steering for 92-100% of poles from L5)"),
            (35, ["write_R", "write_R_z"], "out·R is usually the next word it predicts"),
        ],
        "heads": [
            (3, ["ov_S", "ov_R"], ("the write's footprint (OV·S) is the output read here; OV·R is weak this early. "
                                  "QK reads track real attention best at layer 0")),
            (8, ["ov_S", "ov_R"], "OV·S is the record through layer 8, OV·R carries signal from about layer 4"),
            (12, ["ov_R"], "OV·R is the output read from here"),
            (20, ["ov_R"], "the weakest band for every head read: QK reads are dark here, expect little, say unclear freely"),
            (35, ["ov_R"], "OV·R and the QK reads are most reliable here; late heads often attend to what they copy"),
        ],
        "late": 21,
        "check_bands": ((0, 8), (9, 12), (13, 20), (21, 27), (28, 35)),
    },
    "gemma-4-12b": {
        "in": [
            (4, ["read_S", "read_S_z"], ("in·S names the literal tokens it fires on (it holds for 32-88% of poles here); "
                                        "in·R carries nothing")),
            (13, ["read_S_z", "read_S"], ("in·S fades by layer 9 and returns at layers 10-13 (23-57% of poles); "
                                         "in·R carries nothing")),
            (32, ["read_S_z", "read_R_z"], ("the weakest band for every input read (most poles here pass no test): "
                                           "expect little, say unclear freely")),
            (47, ["read_S_z", "read_R"], ("in·R sometimes names the word about to come next and in·S z a concept, "
                                         "but both are weak: lean on the contexts")),
        ],
        "out": [
            (4, ["write_R_z", "write_S_z"], "writes are weak this early: neither lens predicts a natural-size push before layer 11"),
            (13, ["write_S", "write_R", "write_R_z"], ("out·S holds at layers 11-13 (10-22% of poles), where neurons write back "
                                                      "the tokens they detect; out·R predicts what an amplified push does from layer 11")),
            (32, ["write_R", "write_R_z"], ("out·R is what its push makes the model say (r 0.5-0.8 with steering), "
                                           "though it rarely names the next word here")),
            (47, ["write_R", "write_R_z"], ("out·R is often the next word it predicts (it holds for 21-37% of poles at "
                                           "layers 33-40, 8-14% after)")),
        ],
        "heads": [
            (13, ["ov_S", "ov_R"], ("neither OV read passes its test this early (with 16 heads a layer the tests are weak); "
                                   "early heads often attend by token identity or position: lean on the pattern and the contexts")),
            (32, ["ov_R"], "the weakest band for every head read: 70% of heads here pass no test, expect little, say unclear freely"),
            (47, ["ov_R"], "OV·R is most reliable here: it holds for about a fifth of the heads at layers 36-47"),
        ],
        "late": 33,
        "check_bands": ((0, 4), (5, 13), (14, 32), (33, 47)),
    },
}[NAME]
BANDS = PLANS["check_bands"]
CHECK_PER_BAND = 20
CHECK_SHOWN = 4  # held-out firings per head in a firing check


def lenient(side: str) -> re.Pattern:
    """Recovers a label line whose JSON is malformed (an unescaped quote inside a label, say)."""
    return re.compile(r'"id"\s*:\s*"([^"]+)"' + "".join(
        rf'.*?"{k}"\s*:\s*\{{\s*"label"\s*:\s*"(.*?)"\s*,\s*"confidence"\s*:\s*([0-9.]+)' for k in FIELDS[side]))


def at_depth(side: str, layer: int) -> tuple[list[str], str]:
    return next((reads, line) for last, reads, line in PLANS[side] if layer <= last)


def plan(layer: int) -> tuple[list[str], list[str], str, str]:
    """The reads shown at this depth (input, output), primary first, with a line on each side."""
    (i, ni), (o, no) = at_depth("in", layer), at_depth("out", layer)
    return i, o, ni, no


def stem(t: str) -> str:
    """Variant key: case, leading punctuation and common English suffixes removed (release, Released, -releases)."""
    k = t.strip().lower().lstrip("_-.").lstrip("/([")
    for suf in ("ing", "ed", "es", "s", "ly"):
        if k.endswith(suf) and len(k) - len(suf) >= 4:
            k = k[: -len(suf)]
            break
    return k[:-1] if k.endswith("e") and len(k) > 4 else k


def grouped(tokens: list[str], skip=()) -> tuple[str, set]:
    """Tokens in rank tiers, variants collapsed into their first occurrence (×count); skip: keys shown already."""
    order, count, rank = [], collections.Counter(), {}
    for r, t in enumerate(tokens):
        k = stem(t) or t
        if k in skip:
            continue
        if k not in count:
            order.append((k, t))
            rank[k] = r
        count[k] += 1
    tiers = [[], [], []]
    for k, t in order:
        tiers[0 if rank[k] < 8 else 1 if rank[k] < 24 else 2].append(t + (f"×{count[k]}" if count[k] > 1 else ""))
    return " ‖ ".join(" | ".join(x) for x in tiers if x), set(count)


def cards_layer(layer: int) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    cache = {}

    def dec(i: int) -> str:
        if i not in cache:
            t = tok.decode([i], skip_special_tokens=True).replace("\n", "⏎").replace("\r", "␍")  # as the labelled cards
            cache[i] = t if t.strip() else repr(t)
        return cache[i]

    ins, outs, ni, no = plan(layer)
    shown = ins + outs
    t = load_layer(layer)
    f = t.num_rows // 2  # rows: the + poles of neurons 0..f-1, then the − poles
    strength, hub = {}, {}
    for b in shown:
        strength[b] = sorted(sum(v[:8]) / 8 for v in t.column(f"{b}_val").to_pylist())
        c = collections.Counter()
        for ids in t.column(b + "_ids").to_pylist():
            c.update(set(ids))
        hub[b] = set() if b.endswith("_z") else {k for k, v in c.items() if v / t.num_rows >= HUB}
    kurt = sorted(t.column("vocab_kurt").to_pylist())
    keep = ["vocab_skew", "vocab_kurt", "act_top", "act_max", "rate", "rate_dclm", "rate_zh", "rate_code",
            "rate_chat", "contexts"] + [c for b in shown for c in (b + "_ids", f"{b}_val")]
    rows = t.select(keep).to_pylist()
    with open(OUT["neurons"] / "cards" / f"L{layer:02d}.jsonl", "w", encoding="utf-8") as fh:
        for n in range(f):
            lines = [f"## layer {layer}, neuron {n}", f"input reads at this depth: {ni}.", f"output reads at this depth: {no}."]
            for pole, r in ((1, rows[n]), (-1, rows[f + n])):
                lines.append(f"\n### L{layer} n{n} {'+' if pole > 0 else '−'}")
                toks, keys = {}, {}
                for b in shown:
                    toks[b] = [dec(i) for i in r[b + "_ids"] if i not in hub[b]]
                    pct = 100 * bisect.bisect_left(strength[b], sum(r[f"{b}_val"][:8]) / 8) / len(strength[b])
                    twin = next((c for c in keys if c.removesuffix("_z") == b.removesuffix("_z")), None)
                    body, keys[b] = grouped(toks[b], keys[twin] if twin else ())
                    lines.append(f"{READ[b] + ' adds' if twin else READ[b]} (strength: {pct:.0f}th percentile of the layer): "
                                 + (body or "(nothing new)" if twin else body or "(only the layer's hub tokens)"))
                for side in (ins, outs):
                    for j, a in enumerate(side):
                        for c in side[j + 1:]:
                            if a.removesuffix("_z") == c.removesuffix("_z"):
                                continue  # a read and its own z version share most tokens by construction
                            sh = {stem(x) for x in toks[a]} & {stem(x) for x in toks[c]} - {""}
                            if sh:
                                lines.append(f"shared by {READ[a]} and {READ[c]}: " + " | ".join(sorted(sh)[:16]))
                sh = {stem(x) for b in ins for x in toks[b]} & {stem(x) for b in outs for x in toks[b]} - {""}
                lines.append("shared by the input and output sides: " + (" | ".join(sorted(sh)[:16]) if sh else "none"))
                if layer >= PLANS["late"]:
                    kp = 100 * bisect.bisect_left(kurt, r["vocab_kurt"]) / len(kurt)
                    if kp >= 90:
                        lines.append(f"write shape: peaked (kurtosis {kp:.0f}th percentile), skew {r['vocab_skew']:+.2f}: "
                                     + ("a next-word predictor" if r["vocab_skew"] > 0 else "its bottom tokens may be the next word it suppresses"))
                lines.append(f"- activation: typical high {r['act_top']:.2f}, max {r['act_max']:.2f}; fires (past a "
                             f"quarter of typical) at {r['rate']:.1e} of positions; by slice: "
                             + ", ".join(f"{s} {r[f'rate_{s}']:.1e}" for s in ("dclm", "zh", "code", "chat")))
                if r["act_top"] < SILENT:
                    lines.append("- near-silent: no top context recorded")
                else:
                    lines += [f"  - context: {c}" for c in str(r["contexts"]).split(" ‖ ")[:N_CONTEXTS] if c]
            fh.write(json.dumps({"layer": layer, "unit": n, "card": "\n".join(lines)}, ensure_ascii=False) + "\n")
    return layer


def head_plan(layer: int) -> tuple[list[str], str]:
    """The OV reads shown at this depth, primary first, with a line on what to trust."""
    return at_depth("heads", layer)


# Gemma 4's layers alternate sliding-window and global attention (design.md); Qwen3-8B's are all global.
SPAN = {"sliding_attention": ", a sliding-window layer (1,024 tokens)", "full_attention": ", a global layer"}


def head_cards() -> None:
    tok = tokenizer()
    cache = {}

    def dec(i: int) -> str:
        if i not in cache:
            t = tok.decode([i], skip_special_tokens=True).replace("\n", "⏎").replace("\r", "␍")
            cache[i] = t if t.strip() else repr(t)
        return cache[i]

    t = load_heads()
    rows = t.to_pylist()
    L = max(r["layer"] for r in rows) + 1
    pct = {}
    for k in ("ov_R_copy", "ov_S_copy", "qk_self", "out_top"):
        for layer in range(L):
            pct[k, layer] = sorted(r[k] for r in rows if r["layer"] == layer)

    def rank(r: dict, k: str) -> str:
        s = pct[k, r["layer"]]
        return f"percentile {100 * bisect.bisect_left(s, r[k]) / len(s):.0f} of the layer"

    def tokens(r: dict, name: str, skip=()) -> tuple[str, set]:
        return grouped([dec(i) for i in r[f"{name}_ids"]], skip)

    (OUT["heads"] / "cards").mkdir(parents=True, exist_ok=True)
    by_layer = collections.defaultdict(list)
    for r in rows:
        layer, h = r["layer"], r["head"]
        ov, note = head_plan(layer)
        span = SPAN.get(arch.layout(MODEL)[layer].kind, "")
        lines = [f"## L{layer}h{h} (layer {layer} of {L}{span}, query head {h}, key-value head {r['kv_head']})",
                 f"at this depth: {note}.",
                 (f"- attention pattern: to position 0 (the sink) {r['sink']:.2f}, to the previous token {r['prev']:.2f}, "
                  f"to itself {r['self']:.2f}; on random text said twice, from the second copy to the token after the "
                  f"first occurrence (induction) {r['induction']:.2f} and to the occurrence itself (duplicate) {r['duplicate']:.2f}"),
                 (f"- output: typical high norm {r['out_top']:.2f} ({rank(r, 'out_top')}), past a quarter of it at "
                  f"{100 * r['rate']:.2g}% of positions")]
        for name in ov:
            tag = "OV·R (what reading a token makes it say)" if name == "ov_R" else "OV·S (the write's footprint: as if the token were there)"
            lines.append(f"- {tag}: copying score {r[f'{name}_copy']:+.2f} (+1 copies what it reads, −1 suppresses; "
                         f"{rank(r, f'{name}_copy')})")
            body, keys = tokens(r, name)
            lines.append(f"  - copies most: {body}")
            body_z, _ = tokens(r, f"{name}_z", keys)
            lines.append(f"  - against the layer's heads, also: {body_z or '(nothing new)'}")
            body, keys = tokens(r, f"{name}_neg")
            lines.append(f"  - suppresses most: {body}")
        body, keys = tokens(r, "qk_S")
        lines.append(f"- QK·S (what it attends to under its mean query, whatever the query): {body}")
        body_z, _ = tokens(r, "qk_S_z", keys)
        lines.append(f"  - against the layer's heads, also: {body_z or '(nothing new)'}")
        lines.append(f"  - avoids: {tokens(r, 'qk_S_neg')[0]}")
        lines.append(f"- self-seeking (a token as the query favours the same token as the key): median z "
                     f"{r['qk_self']:+.2f} ({rank(r, 'qk_self')}); most: {tokens(r, 'qk_self_S')[0]}")
        lines.append(f"- attends to what it copies (Spearman of QK·S with OV·R over tokens): {r['qk_ov_rho']:+.2f}")
        lines.append(f"- drivers in real text: as the query, the tokens that make its output largest: {tokens(r, 'drivers')[0]}")
        lines.append(f"  - as the key, the tokens it attends to most from the 16 after them: {tokens(r, 'keys')[0]}")
        lines += [f"  - context: {c}" for c in str(r["contexts"]).split(" ‖ ")[:N_CONTEXTS] if c]
        by_layer[layer].append({"layer": layer, "unit": h, "card": "\n".join(lines)})
    for layer, cs in by_layer.items():
        with open(OUT["heads"] / "cards" / f"L{layer:02d}.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in cs)
    print(f"{time.strftime('%H:%M:%S')} {len(rows)} head cards -> {OUT['heads'] / 'cards'}")


def cards(args) -> None:
    if args.side == "heads":
        head_cards()
        return
    (OUT["neurons"] / "cards").mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        for layer in pool.imap_unordered(cards_layer, range(LAYERS)):
            print(f"{time.strftime('%H:%M:%S')} L{layer:02d} cards", flush=True)


# The API side.

def client():
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    path = Path.home() / ".config" / "anthropic" / "api_key"
    if not key and path.exists():
        key = path.read_text().strip()
    if not key:
        raise SystemExit("no API key: set ANTHROPIC_API_KEY or put it in ~/.config/anthropic/api_key")
    return anthropic.Anthropic(api_key=key)


def load_state(side: str) -> dict:
    p = OUT[side] / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"batches": {}}


def save_state(side: str, s: dict) -> None:
    (OUT[side] / "state.json").write_text(json.dumps(s, indent=1))


def chunks(side: str) -> dict:
    """custom_id -> [(layer, unit, card)], ``PER_REQUEST`` cards in layer order."""
    out, k = {}, PER_REQUEST[side]
    for path in sorted((OUT[side] / "cards").glob("L*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        for i in range(0, len(rows), k):
            part = rows[i:i + k]
            out[f"L{part[0]['layer']:02d}_{i // k:04d}"] = [(r["layer"], r["unit"], r["card"]) for r in part]
    return out


def request(side: str, cid: str, part: list, system: str, thinking: str) -> dict:
    extra = {} if thinking == "on" else {"thinking": {"type": "disabled"}}  # allowed at effort high or below
    what = f"{len(part)} neurons, {2 * len(part)} poles" if side == "neurons" else f"{len(part)} heads"
    return {"custom_id": cid, "params": {
        "model": LABELLER, "max_tokens": MAX_TOKENS, "output_config": {"effort": "low"}, **extra,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": f"Evidence cards ({what}):\n\n" + "\n\n".join(c for _, _, c in part)}]}}


def cost(usage: dict, price: dict = PRICE) -> float:
    return (usage.get("input_tokens", 0) * price["input"] + usage.get("output_tokens", 0) * price["output"]
            + usage.get("cache_read_input_tokens", 0) * price["cache_read"]
            + usage.get("cache_creation_input_tokens", 0) * price["cache_write"]) / 1e6


def parse(side: str, text: str, part: list):
    """The request's labels, one per pole or head, or None if any is missing."""
    want = ({(layer, n, s) for layer, n, _ in part for s in (1, -1)} if side == "neurons"
            else {(layer, h) for layer, h, _ in part})
    fields, got = FIELDS[side], {}
    for line in text.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:  # recover the fields
            m = lenient(side).search(line)
            if not m:
                continue
            o = {"id": m[1], "note": "(recovered from malformed JSON)",
                 **{k: {"label": m[2 + 2 * i], "confidence": m[3 + 2 * i]} for i, k in enumerate(fields)}}
        m = ID[side].match(str(o.get("id", "")).strip())
        if not m or not all(isinstance(o.get(k), dict) and "label" in o[k] for k in fields):
            continue
        key = (int(m[1]), int(m[2]), 1 if m[3] == "+" else -1) if side == "neurons" else (int(m[1]), int(m[2]))
        rec = dict(zip(("layer", "neuron", "pole") if side == "neurons" else ("layer", "head"), key))
        for k in fields:
            rec[k] = str(o[k]["label"])
            try:
                rec[f"{k}_conf"] = float(o[k]["confidence"])
            except (KeyError, TypeError, ValueError):
                rec[f"{k}_conf"] = float("nan")
        rec["note"] = str(o.get("note", ""))
        got[key] = rec
    return [got[k] for k in sorted(want)] if want <= set(got) else None


def usage_of(message) -> dict:
    return {k: v for k, v in message.usage.model_dump().items() if isinstance(v, int)}


def text_of(message) -> str:
    return "".join(x.text for x in message.content if x.type == "text")


def wait(c, batch_id: str) -> None:
    while c.messages.batches.retrieve(batch_id).processing_status != "ended":
        time.sleep(60)


def pilot(args) -> None:
    side = args.side
    c, system, allc = client(), PROMPT[side].read_text(), chunks(side)
    pick = random.Random(1031).sample(sorted(allc), min(args.n, len(allc)))
    b = c.messages.batches.create(requests=[request(side, cid, allc[cid], system, args.thinking) for cid in pick])
    print("pilot batch", b.id, flush=True)
    wait(c, b.id)
    use, ok, n = collections.Counter(), 0, 0
    for r in c.messages.batches.results(b.id):
        n += 1
        if r.result.type == "succeeded":
            use.update(usage_of(r.result.message))
            ok += parse(side, text_of(r.result.message), allc[r.custom_id]) is not None
    s = load_state(side)  # another pilot may have saved meanwhile
    s[f"pilot_{args.thinking}"] = {"batch": b.id, "requests": n, "parsed": ok, "usage": dict(use), "spent": cost(use),
                                   "projected_total": cost(use) / max(n, 1) * len(allc), "requests_total": len(allc)}
    save_state(side, s)
    print(json.dumps(s[f"pilot_{args.thinking}"], indent=1))


def submit(args) -> None:
    side = args.side
    s = load_state(side)
    p = s.get(f"pilot_{args.thinking}")
    if not p or p["projected_total"] > args.cap or p["parsed"] < 0.9 * p["requests"]:
        raise SystemExit(f"refusing: run a pilot at --thinking {args.thinking} that projects under ${args.cap} and parses 90%")
    c, system, allc = client(), PROMPT[side].read_text(), chunks(side)
    sent = {x for b in s["batches"].values() for x in b["ids"]}
    batch, size = [], 0
    for cid in [x for x in allc if x not in sent] + [None]:
        rq = None if cid is None else request(side, cid, allc[cid], system, args.thinking)
        n = 0 if rq is None else len(json.dumps(rq, ensure_ascii=False).encode())
        if batch and (rq is None or len(batch) >= MAX_REQUESTS or size + n > MAX_BYTES):
            b = c.messages.batches.create(requests=batch)
            s["batches"][b.id] = {"ids": [r["custom_id"] for r in batch], "collected": False, "thinking": args.thinking}
            save_state(side, s)
            print(f"{time.strftime('%H:%M:%S')} submitted {b.id}: {len(batch)} requests", flush=True)
            batch, size = [], 0
        if rq is not None:
            batch.append(rq)
            size += n


def status(args) -> None:
    s, c = load_state(args.side), client()
    for i, info in s["batches"].items():
        b = c.messages.batches.retrieve(i)
        print(i, b.processing_status, vars(b.request_counts), "collected" if info["collected"] else "")
    print(f"spent (collected): ${s.get('spent', 0):.2f}")


def collect(args) -> None:
    side = args.side
    s, c, allc = load_state(side), client(), chunks(side)
    (OUT[side] / "labels").mkdir(exist_ok=True)
    fails = set(s.get("failed", []))
    for i, info in s["batches"].items():
        if info["collected"] or c.messages.batches.retrieve(i).processing_status != "ended":
            continue
        out = collections.defaultdict(list)
        for r in c.messages.batches.results(i):
            if r.result.type != "succeeded":
                fails.add(r.custom_id)
                continue
            s["spent"] = s.get("spent", 0) + cost(usage_of(r.result.message))
            labs = parse(side, text_of(r.result.message), allc[r.custom_id])
            if labs is None:
                fails.add(r.custom_id)
            else:
                fails.discard(r.custom_id)
                out[labs[0]["layer"]] += labs
        for layer, labs in out.items():
            with open(OUT[side] / "labels" / f"L{layer:02d}.jsonl", "a", encoding="utf-8") as fh:
                fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in labs)
        info["collected"], s["failed"] = True, sorted(fails)
        save_state(side, s)
        print(f"{time.strftime('%H:%M:%S')} collected {i}", flush=True)
    print(f"{len(fails)} requests failed or unparsed; spent ${s.get('spent', 0):.2f}")


def retry(args) -> None:
    side = args.side
    s, c, system, allc = load_state(side), client(), PROMPT[side].read_text(), chunks(side)
    if s.get("failed"):
        b = c.messages.batches.create(requests=[request(side, cid, allc[cid], system, args.thinking) for cid in s["failed"]])
        s["batches"][b.id] = {"ids": s["failed"], "collected": False, "thinking": args.thinking, "retry": True}
        save_state(side, s)
        print("retry batch", b.id, len(s["failed"]), "requests")


def shards(args) -> None:
    """The labels table in catalog row order; a missing pole or head is a KeyError."""
    side, fields = args.side, FIELDS[args.side]
    got = {}
    for path in sorted((OUT[side] / "labels").glob("L*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                o = json.loads(line)
                got[(o["layer"], o["neuron"], o["pole"]) if side == "neurons" else (o["layer"], o["head"])] = o

    def columns(rows: list[dict]) -> dict:
        cols = {}
        for k in fields:
            cols[k] = pa.array([r[k] for r in rows])
            cols[f"{k}_conf"] = pa.array([r[f"{k}_conf"] for r in rows], pa.float32())
        cols["note"] = pa.array([r["note"] for r in rows])
        return cols
    if side == "neurons":
        for layer in sorted({k[0] for k in got}):
            f = sum(1 for k in got if k[0] == layer) // 2
            rows = [got[(layer, n, 1)] for n in range(f)] + [got[(layer, n, -1)] for n in range(f)]
            write(neuron_keys(layer, f) | columns(rows), neuron_table("labels", layer))
    else:
        keys = pq.read_table(head_table("reads"), columns=["layer", "head", "kv_head"])
        L, H = max(keys.column("layer").to_pylist()) + 1, keys.num_rows // (max(keys.column("layer").to_pylist()) + 1)
        rows = [got[(layer, h)] for layer in range(L) for h in range(H)]
        write(head_keys(L, H, H // (max(keys.column("kv_head").to_pylist()) + 1)) | columns(rows), head_table("labels"))
    print(f"{time.strftime('%H:%M:%S')} {side}/labels written")


# The blind check of the head labels.

SCORE_SYSTEM = """You test descriptions of attention heads in a language model against text. Each context shows \
a position where some head's output was large: the query token in ⟦ ⟧, where the context ends, and the token that \
head attended to most in ⟪ ⟫ (when it attended to the query itself, only ⟦ ⟧ is shown). Given one head's description, say for each context \
how likely it is that this head produced it: some of the contexts are this head's, the others another head's. \
Reply with one JSON line and nothing else: {"p": [p1, ..., pn]}, a probability from 0 to 1 per context, in order."""

JUDGE_SYSTEM = """You match descriptions of attention heads to what the heads write. You see the tokens a head's \
output most promotes when it fires in real text (each token's score compared with the other heads of its layer), \
and four descriptions of what a head writes, A to D. Exactly one describes this head. Reply with one JSON line and \
nothing else: {"pick": "A"} (or B, C, D)."""


def check(args) -> None:
    rows = load_heads().to_pylist()
    lab = {(r["layer"], r["head"]): r for r in pq.read_table(head_table("labels")).to_pylist()}
    tok = tokenizer()
    docs = doc_tokens(torch_load(acts_path())) if acts_path().exists() else corpus_doc_tokens()
    rng = random.Random(1030)
    by_band = {b: [r for r in rows if b[0] <= r["layer"] <= b[1]] for b in BANDS}
    reqs, meta = [], {}
    for b, pool in by_band.items():
        for r in rng.sample(pool, CHECK_PER_BAND):
            other, third = rng.sample([x for x in pool if x is not r], 2)
            ctx = []
            for src, truth in ((r, 1), (other, 0)):
                held = list(range(N_CONTEXTS, len(src["top_doc"])))
                for j in rng.sample(held, CHECK_SHOWN):
                    ctx.append((context(tok, docs[src["top_doc"][j]], src["top_pos"][j], src["src_pos"][j], after=0), truth))
            rng.shuffle(ctx)
            hid = f"L{r['layer']}h{r['head']}"
            text = "\n".join(f"{i + 1}. {c}" for i, (c, _) in enumerate(ctx))
            for arm, desc in (("own", lab[r["layer"], r["head"]]), ("control", lab[third["layer"], third["head"]])):
                reqs.append({"custom_id": f"fire_{arm}_{hid}", "params": {
                    "model": SCORER, "max_tokens": 2000, "system": SCORE_SYSTEM,
                    "messages": [{"role": "user", "content": f"The head fires: {desc['fires']}\nIt attends to: "
                                  f"{desc['attends']}\n\nContexts:\n{text}"}]}})
            options = [r] + rng.sample([x for x in pool if x is not r], 3)
            rng.shuffle(options)
            answer = "ABCD"[options.index(r)]
            written = " | ".join(r["out_R_z_str"].split(" | ")[:8])
            more = " | ".join(tok.decode([i]).replace("\n", "⏎") for i in r["out_R_z_ids"][8:24])
            reqs.append({"custom_id": f"write_{hid}", "params": {
                "model": SCORER, "max_tokens": 2000, "system": JUDGE_SYSTEM,
                "messages": [{"role": "user", "content": f"What the head's output promotes: {written} ‖ {more}\n\n"
                              + "\n".join(f"{k}. {lab[o['layer'], o['head']]['writes']}" for k, o in zip("ABCD", options))}]}})
            meta[hid] = {"band": f"L{b[0]}-L{b[1]}", "truth": [t for _, t in ctx], "answer": answer,
                         "conf": {k: lab[r["layer"], r["head"]][f"{k}_conf"] for k in FIELDS["heads"]}}
    c = client()
    batch = c.messages.batches.create(requests=reqs)
    print(f"check batch {batch.id}: {len(reqs)} requests", flush=True)
    wait(c, batch.id)
    got = {}
    for res in c.messages.batches.results(batch.id):
        if res.result.type == "succeeded":
            m = re.search(r"\{.*\}", text_of(res.result.message), re.DOTALL)
            try:
                got[res.custom_id] = json.loads(m[0]) if m else None
            except json.JSONDecodeError:
                got[res.custom_id] = None
    per = {}
    for hid, mt in meta.items():
        out = {"band": mt["band"]}
        for arm in ("own", "control"):
            p = (got.get(f"fire_{arm}_{hid}") or {}).get("p")
            if isinstance(p, list) and len(p) == len(mt["truth"]):
                out[arm] = auroc([float(x) for x in p], mt["truth"])
        pick = (got.get(f"write_{hid}") or {}).get("pick")
        if pick in tuple("ABCD"):
            out["writes"] = float(pick == mt["answer"])
        out["conf"] = mt["conf"]
        per[hid] = out
    result = {"heads": per, "batch": batch.id, "scorer": SCORER, "per_band": CHECK_PER_BAND, "shown": CHECK_SHOWN}
    lines = summarize(per, result)
    (SUMMARY / f"labels_heads_{NAME}.json").write_text(json.dumps(result, indent=1))
    (SUMMARY / f"labels_heads_{NAME}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def boot(xs: list[float], n: int = 2000) -> tuple[float, float]:
    rng = random.Random(0)
    ms = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def auroc(p: list[float], truth: list[int]) -> float:
    pos = [x for x, t in zip(p, truth) if t]
    neg = [x for x, t in zip(p, truth) if not t]
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def summarize(per: dict, result: dict) -> list[str]:
    """The card's section on the head labels, from the check."""
    def stat(key, band=None):
        xs = [v[key] for v in per.values() if key in v and (band is None or v["band"] == band)]
        return (sum(xs) / len(xs), *boot(xs)) if xs else (float("nan"),) * 3
    own, ctl, wr = stat("own"), stat("control"), stat("writes")
    result["summary"] = {"firing": own, "control": ctl, "writes": wr,
                         "bands": {b: {k: stat(k, b)[0] for k in ("own", "control", "writes")}
                                   for b in sorted({v["band"] for v in per.values()}, key=lambda s: int(s[1:].split("-")[0]))}}
    lines = ["Every head described by Claude Haiku 5.5 from its card (batch API, effort low, adaptive thinking):",
             "`fires` (where its output is large), `attends` (what it attends to there) and `writes` (what its",
             "output promotes), each with a confidence 0-1, and `note`. The cards show the head's pattern scores,",
             "output scale, OV and QK reads, drivers and first 5 top contexts; nothing about its other top firings.",
             f"The prompt is `scripts/prompts/{NAME}/heads.md` on GitHub.", "",
             f"**Checked blind** on {len(per)} random heads, {result['per_band']} a band, by Claude Sonnet 5.5:", "",
             (f"- **Firing:** given a head's `fires` and `attends` labels and {2 * result['shown']} contexts, "
              f"{result['shown']} of its held-out top firings (ranks 6-24) and {result['shown']} of another head of its band, "
              f"the scorer separates them at AUROC {own[0]:.2f} [{own[1]:.2f}, {own[2]:.2f}]; a third head's labels on the "
              f"same contexts score {ctl[0]:.2f}."),
             (f"- **Writes:** given what the head's firings actually promote in context (`out_R_z`, not on the card), a "
              f"judge picks its `writes` label out of four from its band {100 * wr[0]:.0f}% of the time "
              f"[{100 * wr[1]:.0f}, {100 * wr[2]:.0f}] (chance 25%)."), "",
             "| band | firing AUROC | control | writes picked |", "|---|---|---|---|"]
    for b, v in result["summary"]["bands"].items():
        lines.append(f"| {b} | {v['own']:.2f} | {v['control']:.2f} | {100 * v['writes']:.0f}% |")
    return lines


def torch_load(path):
    import torch
    return torch.load(path, weights_only=False, map_location="cpu", mmap=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("cards", "pilot", "submit", "status", "collect", "retry", "shards", "check"))
    ap.add_argument("--side", choices=("neurons", "heads"), default="neurons")
    ap.add_argument("--thinking", choices=("on", "off"), default="on")
    ap.add_argument("--n", type=int, default=60, help="pilot requests")
    ap.add_argument("--cap", type=float, default=90.0, help="most the projected full run may cost, dollars")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.what == "check":
        args.side = "heads"
    OUT[args.side].mkdir(parents=True, exist_ok=True)
    {"cards": cards, "pilot": pilot, "submit": submit, "status": status, "collect": collect,
     "retry": retry, "shards": shards, "check": check}[args.what](args)


if __name__ == "__main__":
    main()
