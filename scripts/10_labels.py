"""Label every neuron pole and attention head with Claude Haiku 5.5 from its lens reads and top
contexts: the catalog's ``labels`` tables (pipeline stage 10).

**Neurons** (``--side neurons``): the evidence card per neuron (both poles) shows, for each pole, one input read and one
write read of ``neurons/reads``, the ones its model's reading plan (``arch.PLANS``) picks for the neuron's layer: the read
raters found most coherent there, blind to the lens (``checks/depth_reads.py``; the write read is the R-lens's at every
depth, which also predicts steering, §3). Each shows its first 24 tokens in rank tiers (1-8 ‖ 9-24), variants merged
(``release×9``, runs of line breaks as ``⏎``), the layer's hub tokens (in 3%+ of its raw top 64s) dropped from raw reads;
then the tokens the other lens's reads agree on, those both sides share, from the plan's late layer a note on peaked
(high-kurtosis) writes (§5), a line marking a faint shadow of the other pole (§7, which then drops its input read), the
pole's firing rate with the corpus slices where it fires at least twice or under half as often, and ``neurons/acts``'
first 5 top contexts. No strengths or activation scale: a label's confidence is how clearly the card shows one theme.
The prompt is shared by every model (``scripts/prompts/neurons.md``); a model's own file says what the model is and gives
its worked examples, and its plan fills the guide to the reads by depth. Labels: ``fires`` and ``writes``, each with a
confidence, and a note. Thinking is off by default. The first full run (2026-10-29, on the first release's tables and an
earlier card: up to eight reads of 64 tokens with strength percentiles): 44,244 requests of 10 neurons, $105.41 at batch
prices with thinking on (effort low), checked blind on 100 poles (``checks/haiku_blind.py``, §13).

**Heads** (``--side heads``): the card per head shows its attention pattern scores, how often it fires, one OV read
(what it copies and suppresses: through the S-lens early, the R-lens after, as its model's plan says; §10), its QK reads
(what it attends to, the tokens that seek themselves, and from L1 those a query about to say them seeks), its exact
drivers as the query and as the key, each list to 24 tokens, without layer percentiles, and its first 5 top contexts with the query in ``⟦ ⟧`` and the attended token
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

The prompts are ``scripts/prompts/<side>.md`` with ``scripts/prompts/<model>/<side>.md`` filled in. Stages (under ``data/labels/<model>/<side>/``;
nothing is sent to the API without an explicit stage):

- ``cards``: ``cards/L<layer>.jsonl``, one card per neuron or head; ``--flipped``, only the neurons with a pole on SiLU's
  negative lobe (``acts``' ``neg_lobe`` past a half), whose input reads turned round when those poles began to be read
  the right way (findings §7), for relabelling just those: ``shards`` lands the new labels on the table already there;
- ``pilot``: one batch of ``--n`` random requests; measures tokens (thinking included, with ``--thinking on``) and the parse
  rate and projects the full run's cost into ``state.json``;
- ``submit``: every request, in batches under the API's limits; refuses unless a pilot at the same thinking setting
  (this model's, else the newest of another's, scaled to this model's cards) projects under ``--cap`` dollars and
  parsed 90%;
- ``status``, ``collect`` (labels to ``labels/L<layer>.jsonl``), ``retry`` (failed or unparsed
  requests: as they were the first time, card by card once a request has failed twice, so a card the API refuses
  costs only its own labels);
- ``shards``: the ``labels`` table, ``neurons/labels/L<layer>.parquet`` or ``heads/labels.parquet``; a pole or head
  left without a label gets empty labels, NaN confidences and a note saying why;
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
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from neuron_lens import arch
from neuron_lens.catalog import (
    MODEL,
    HELD_OUT,
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
TEMPLATE = {side: ROOT / "scripts" / "prompts" / f"{side}.md" for side in OUT}  # shared by every model
PROMPT = {side: ROOT / "scripts" / "prompts" / NAME / f"{side}.md" for side in OUT}  # the model's own: about it, examples
LABELLER, SCORER = "claude-haiku-5-5", "claude-sonnet-5-5"
PRICE = {"input": 0.05, "output": 0.25, "cache_read": 0.005, "cache_write": 0.0625}  # $/MTok, Haiku's batch prices
PER_REQUEST = {"neurons": 10, "heads": 8}  # cards per request
FIELDS = {"neurons": ("fires", "writes"), "heads": ("fires", "attends", "writes")}
MAX_TOKENS = 12000
MAX_REQUESTS, MAX_BYTES = 3000, 200_000_000  # per batch, under the API's 100,000 and 256 MB
SILENT = 0.05  # act_top under this: near-silent (§8, §13)
SHADOW = 1 / 4  # a pole on the negative lobe under this share of its partner's act_top is a faint shadow of it (§7)
HUB = 0.03  # a raw read's token in this share of the layer's top 64s is a hub (§7)
INPUTS, WRITES = ("read_S", "read_S_z", "read_R", "read_R_z"), ("write_R", "write_R_z", "write_S", "write_S_z")
SHOW = 24  # tokens a read shows: rank tiers 1-8 ‖ 9-24
SLICES = ("dclm", "zh", "code", "chat")
READ = {"read_S": "in·S", "read_S_z": "in·S z", "read_R": "in·R", "read_R_z": "in·R z",
        "write_S": "out·S", "write_S_z": "out·S z", "write_R": "out·R", "write_R_z": "out·R z"}
ID = {"neurons": re.compile(r"^L(\d+) n(\d+) ([+−-])(?:\s.*)?$"), "heads": re.compile(r"^L(\d+)[hH](\d+)(?:\s.*)?$")}
# The reading plan, per model: by depth, (last layer, the read a card shows, the prompt's line on it), for the input
# and the write side of a neuron, and of a head (its OV read); the layer from which a
# peaked write gets its note (§5); its depths are `arch.PLANS`. A card shows one read a side: the one raters found
# most coherent at that depth, blind to the lens (``checks/depth_reads.py``, two raters a model, 8-16 layers, findings
# §8); the write read is the R-lens's everywhere, which also predicts steering (§3, §19). Where the input read is the
# R-lens's but rarely names the contexts' tokens, the line says so: it can name a concept associated with what drives
# the pole rather than the tokens it fires on. The head plans follow each model's census of head reads (§10, §11, §19).
S_IN = "the literal tokens that drive the pole, at its position or a few before"
S_Z_IN = ("the tokens that drive the pole, at its position or a few before, scored against the layer's other neurons: what "
          "is specific to it, often a concept more than a token")
R_ASSOC = ("what the model is getting ready to say when the pole is driven; here that often differs from the tokens it fires on "
           "(an associated concept, a style or a script): take where it fires from the contexts, and say in the note what the "
           "read adds")
R_IN = "the concept around where the pole fires, and late often the word about to come next"
W_EARLY = "what its push makes the model say; this early, steering follows the read less closely than later"
W_MID = "what its push makes the model say"
W_LATE = "often the next word it predicts"
H_EARLY = "OV·S, the write's footprint (as if the attended token were there), is the better write read this early"
H_MID = "OV·R, what reading a token makes the head say, is the write read"
H_LATE = ("OV·R most often names what late heads copy, and 'seeks what it is about to say' tracks where they look: with a "
          "negative copying score, copy suppression")


def reading_plan(model_id: str) -> dict | None:
    """The model's reading plan as the prompts use it, from its depths (``arch.PLANS``): by depth, (last layer, the
    read a card shows, the prompt's line on it) for the input side, the write side and the heads (their OV read), and the
    layer from which a peaked write gets its note. None for a model without a plan."""
    p = arch.plan(model_id)
    if p is None:
        return None
    last = p.layers - 1
    s_line = S_Z_IN if p.s_read.endswith("_z") else S_IN
    ins = (([(p.s_last, p.s_read, s_line)] if p.s_last >= 0 else [])
           + ([(p.assoc_last, "read_R", R_ASSOC)] if p.assoc_last > p.s_last else []) + [(last, "read_R", R_IN)])
    outs = (([(p.write_edge - 1, "write_R", W_EARLY_OF.get(NAME, W_EARLY))] if p.write_edge > 0 else [])
            + [(p.late - 1, "write_R", W_MID), (last, "write_R", W_LATE)])
    heads = [(p.heads_early, p.heads_early_read, H_EARLY_OF.get(NAME, H_EARLY)), (p.heads_late - 1, "ov_R", H_MID),
             (last, "ov_R", H_LATE)]
    if NAME == "qwen3-8b":  # finer head lines, at its S-lens input edge and its next-word edge
        heads = [(p.s_last, "ov_S", H_EARLY + ", and the QK reads track real attention best at layer 0"),
                 (p.heads_early, "ov_S", "OV·S is still the better write read, though OV·R carries signal from about layer 4"),
                 (p.late - 1, "ov_R", H_MID + "; under the head's average query the QK reads say little about these heads"),
                 (p.heads_late - 1, "ov_R", H_MID + "; some heads already seek what they are about to say"),
                 (last, "ov_R", "late heads often attend to what they copy; " + H_LATE)]
    return {"in": ins, "out": outs, "heads": heads, "late": p.late}


W_EARLY_OF = {"gemma-4-12b": "what its push would make the model say; at these layers a push at its natural size barely "
                             "moves the output, and steering doesn't follow the read"}
H_EARLY_OF = {"gemma-4-12b": "early heads often attend by token identity or position: the pattern scores and the contexts "
                             "show it best"}
PLAN = reading_plan(MODEL)  # None for a model without a reading plan: its cards are refused
BANDS = tuple(arch.bands(MODEL)) if PLAN else ()  # the head check's bands (arch.depth_bands)
CHECK_PER_BAND = 20
CHECK_SHOWN = 4  # held-out firings per head in a firing check


def lenient(side: str) -> re.Pattern:
    """Recovers a label line whose JSON is malformed (an unescaped quote inside a label, say)."""
    return re.compile(r'"id"\s*:\s*"([^"]+)"' + "".join(
        rf'.*?"{k}"\s*:\s*\{{\s*"label"\s*:\s*"(.*?)"\s*,\s*"confidence"\s*:\s*([0-9.]+)' for k in FIELDS[side]))


def at_depth(side: str, layer: int):
    """The read (for heads, the OV reads) a card shows at this depth, and the prompt's line on it."""
    return next((reads, line) for last, reads, line in PLAN[side] if layer <= last)


def depth_table(side: str) -> str:
    """The prompt's guide to the reads by depth, from the plan: what the card shows at each depth and how far it goes."""
    def spans(plan):
        first = 0
        for last, reads, line in plan:
            yield (f"layer {first}" if first == last else f"layers {first}-{last}"), reads, line
            first = last + 1
    if side == "heads":
        return "\n".join(f"- {span}: {line}." for span, _, line in spans(PLAN["heads"]))
    return ("The input read on each card, by depth:\n" + "\n".join(f"- {span}, {READ[r]}: {line}." for span, r, line in spans(PLAN["in"]))
            + "\n\nThe write read on each card, by depth:\n"
            + "\n".join(f"- {span}, {READ[r]}: {line}." for span, r, line in spans(PLAN["out"])))


def system_prompt(side: str) -> str:
    """The shared prompt with the model's parts filled in: its own file (``PROMPT``) up to ``## Worked examples`` says
    what the model is, the rest is its worked examples; the depth guide comes from its plan."""
    if PLAN is None:
        raise SystemExit(f"no reading plan for {NAME} in arch.PLANS")
    about, mark, examples = PROMPT[side].read_text().partition("## Worked examples")
    return (TEMPLATE[side].read_text().replace("{about}", about.strip()).replace("{depths}", depth_table(side))
            .replace("{examples}", (mark + examples).strip()))


def stem(t: str) -> str:
    """Variant key: case, leading punctuation and common English suffixes removed (release, Released, -releases); every
    run of line breaks is one key (⏎, ⏎⏎⏎, ␍⏎)."""
    if t.strip() and set(t) <= set("⏎␍ \t"):
        return "⏎"
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
        tiers[0 if rank[k] < 8 else 1 if rank[k] < 24 else 2].append((k if k == "⏎" else t) + (f"×{count[k]}" if count[k] > 1 else ""))
    return " ‖ ".join(" | ".join(x) for x in tiers if x), set(count)


def lens(read: str) -> str:
    return "S" if "_S" in read else "R"


def firing_line(r: dict) -> str:
    """How often the pole fires, and the corpus slices where it fires at least twice as often or under half as often."""
    rate = r["rate"]
    odd = [f"{s} {r[f'rate_{s}'] / rate:.2g}×" for s in SLICES
           if rate > 0 and (r[f"rate_{s}"] >= 2 * rate or r[f"rate_{s}"] <= rate / 2)]
    return f"- fires at {rate:.1e} of positions" + (f"; {', '.join(odd)}" if odd else "")


def cards_layer(layer: int, flipped: bool = False) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    cache = {}

    def dec(i: int) -> str:
        if i not in cache:
            t = tok.decode([i], skip_special_tokens=True).replace("\n", "⏎").replace("\r", "␍")  # as the labelled cards
            cache[i] = t if t.strip() else repr(t)
        return cache[i]

    (main_in, _), (main_out, _) = at_depth("in", layer), at_depth("out", layer)
    other = {main_in: [b for b in INPUTS if lens(b) != lens(main_in)], main_out: [b for b in WRITES if lens(b) != lens(main_out)]}
    used = [main_in, main_out, *other[main_in], *other[main_out]]
    t = load_layer(layer)
    f = t.num_rows // 2  # rows: the + poles of neurons 0..f-1, then the − poles
    hub = {}
    for b in used:
        c = collections.Counter()
        for ids in t.column(b + "_ids").to_pylist():
            c.update(set(ids))
        hub[b] = set() if b.endswith("_z") else {k for k, v in c.items() if v / t.num_rows >= HUB}
    kurt = sorted(t.column("vocab_kurt").to_pylist())
    keep = ["vocab_skew", "vocab_kurt", "act_top", "rate", *(f"rate_{s}" for s in SLICES), "contexts"] + [b + "_ids" for b in used]
    rows = t.select(keep).to_pylist()
    lobe = t.column("neg_lobe").to_pylist()
    with open(OUT["neurons"] / "cards" / f"L{layer:02d}.jsonl", "w", encoding="utf-8") as fh:
        for n in range(f):
            if flipped and lobe[n] <= 0.5 and lobe[f + n] <= 0.5:
                continue
            lines = [f"## layer {layer}, neuron {n}"]
            for pole, r, partner, on_lobe in ((1, rows[n], rows[f + n], lobe[n]), (-1, rows[f + n], rows[n], lobe[f + n])):
                lines.append(f"\n### {'+' if pole > 0 else '−'}")
                shadow = on_lobe > 0.5 and partner["act_top"] >= SILENT and r["act_top"] < SHADOW * partner["act_top"]
                toks = {b: [dec(i) for i in r[b + "_ids"] if i not in hub[b]] for b in used}
                stems = {b: {stem(x) for x in toks[b]} - {""} for b in used}
                if shadow:
                    lines.append(f"This pole fires on the activation's negative lobe, a faint shadow of the {'−' if pole > 0 else '+'} pole: "
                                 "its input reads would name what that pole fires on, so the card leaves them out.")
                for b in ([main_out] if shadow else [main_in, main_out]):
                    lines.append(f"{READ[b]}: " + (grouped(toks[b][:SHOW])[0] or "(only the layer's hub tokens)"))
                for b in ([main_out] if shadow else [main_in, main_out]):
                    sh = stems[b] & set().union(*(stems[c] for c in other[b]))
                    if sh:
                        lines.append(f"{READ[other[b][0]]} agrees on: " + " | ".join(sorted(sh)[:16]))
                sh = set() if shadow else stems[main_in] & stems[main_out]
                if sh:
                    lines.append("shared by both sides: " + " | ".join(sorted(sh)[:16]))
                if layer >= PLAN["late"] and bisect.bisect_left(kurt, r["vocab_kurt"]) >= 0.9 * len(kurt):
                    lines.append("write shape: peaked, " + ("a next-word predictor" if r["vocab_skew"] > 0
                                                           else "its bottom tokens may be the next word it suppresses"))
                lines.append(firing_line(r))
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
SPAN = {"sliding_attention": "a sliding-window layer (1,024 tokens)", "full_attention": "a global layer"}


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

    def tokens(r: dict, name: str) -> str:
        return grouped([dec(i) for i in r[f"{name}_ids"]][:SHOW])[0]

    layout = arch.layout(MODEL)
    mixed = len({x.kind for x in layout}) > 1

    (OUT["heads"] / "cards").mkdir(parents=True, exist_ok=True)
    by_layer = collections.defaultdict(list)
    for r in rows:
        layer, h = r["layer"], r["head"]
        ov, _ = head_plan(layer)
        span = SPAN.get(layout[layer].kind) if mixed else None  # said only where a model has both kinds of layer
        lines = [f"## layer {layer}, attention head {h}"] + ([f"- {span}"] if span else []) + [
                 (f"- attention: sink {r['sink']:.2f}, previous token {r['prev']:.2f}, self {r['self']:.2f}, "
                  f"induction {r['induction']:.2f}, duplicate {r['duplicate']:.2f}"),
                 f"- fires at {100 * r['rate']:.2g}% of positions",
                 f"- {'OV·R' if ov == 'ov_R' else 'OV·S'}, copying score {r[f'{ov}_copy']:+.2f}",
                 f"  - copies most: {tokens(r, ov)}",
                 f"  - suppresses most: {tokens(r, f'{ov}_neg')}",
                 f"- QK·S: {tokens(r, 'qk_S')}",
                 f"- self-seeking, median z {r['qk_self_S']:+.2f}: {tokens(r, 'qk_self_S')}"]
        if layer:  # no R-lens input at L0
            lines.append(f"- seeks what it is about to say, median z {r['qk_self_R']:+.2f}: {tokens(r, 'qk_self_R')}")
        lines.append(f"- attends to what it copies: {r['qk_ov_rho']:+.2f}")
        lines.append(f"- drivers as the query: {tokens(r, 'drivers')}")
        lines.append(f"- drivers as the key: {tokens(r, 'keys')}")
        lines += [f"  - context: {c}" for c in str(r["contexts"]).split(" ‖ ")[:N_CONTEXTS] if c]
        by_layer[layer].append({"layer": layer, "unit": h, "card": "\n".join(lines)})
    for layer, cs in by_layer.items():
        with open(OUT["heads"] / "cards" / f"L{layer:02d}.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in cs)
    print(f"{time.strftime('%H:%M:%S')} {len(rows)} head cards -> {OUT['heads'] / 'cards'}")


def cards(args) -> None:
    system_prompt(args.side)  # refuses a model without a plan or a prompt
    if args.side == "heads":
        head_cards()
        return
    (OUT["neurons"] / "cards").mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        for layer in pool.imap_unordered(partial(cards_layer, flipped=args.flipped), range(LAYERS)):
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


def part_of(allc: dict, cid: str) -> list:
    """A request's cards: a chunk's (``L13_0494``), or one card of it (``L13_0494_3``, a retry card by card)."""
    if cid in allc:
        return allc[cid]
    chunk, k = cid.rsplit("_", 1)
    return [allc[chunk][int(k)]]


def request(side: str, cid: str, part: list, system: str, thinking: str) -> dict:
    extra = {} if thinking == "on" else {"thinking": {"type": "disabled"}}  # allowed at effort high or below
    n = len(part)
    what = (f"{n} neuron{'s' * (n > 1)}, {2 * n} poles" if side == "neurons" else f"{n} head{'s' * (n > 1)}")
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


def extent(allc: dict, ids) -> dict:
    """Requests, cards and card characters among ``ids``: what a run's cost scales with."""
    parts = [allc[i] for i in ids]
    return {"requests": len(parts), "cards": sum(len(p) for p in parts), "chars": sum(len(c) for p in parts for _, _, c in p)}


def projected(p: dict, whole: dict) -> float:
    """A full run's cost from a pilot: its input tokens per card character, output tokens per card and cached prompt
    per request, scaled to ``whole`` (``extent``), which may be another model's: one pilot prices every model."""
    u, n = p["usage"], p["size"]
    return cost({"input_tokens": u.get("input_tokens", 0) / n["chars"] * whole["chars"],
                 "output_tokens": u.get("output_tokens", 0) / n["cards"] * whole["cards"],
                 "cache_read_input_tokens": (u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0))
                                            / n["requests"] * whole["requests"]})


def latest_pilot(side: str, thinking: str) -> tuple[str, dict] | None:
    """This model's pilot at this thinking setting, else the newest of any model's."""
    own = load_state(side).get(f"pilot_{thinking}")
    if own and "size" in own:
        return NAME, own
    found = []
    for path in (DATA / "labels").glob(f"*/{side}/state.json"):
        p = json.loads(path.read_text()).get(f"pilot_{thinking}")
        if p and "size" in p:
            found.append((path.stat().st_mtime, path.parent.parent.name, p))
    return max(found)[1:] if found else None


def pilot(args) -> None:
    side = args.side
    c, system, allc = client(), system_prompt(side), chunks(side)
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
    p = {"batch": b.id, "requests": n, "parsed": ok, "usage": dict(use), "spent": cost(use), "size": extent(allc, pick)}
    s[f"pilot_{args.thinking}"] = p | {"projected_total": projected(p, extent(allc, allc)), "requests_total": len(allc)}
    save_state(side, s)
    print(json.dumps(s[f"pilot_{args.thinking}"], indent=1))


def submit(args) -> None:
    side = args.side
    s = load_state(side)
    c, system, allc = client(), system_prompt(side), chunks(side)
    found = latest_pilot(side, args.thinking)
    if not found:
        raise SystemExit(f"refusing: no pilot at --thinking {args.thinking} for {side}, of this model or another")
    source, p = found
    cost_here = projected(p, extent(allc, allc))
    print(f"pilot of {source}: parsed {p['parsed']}/{p['requests']}; this model's run projects ${cost_here:.2f}", flush=True)
    if cost_here > args.cap or p["parsed"] < 0.9 * p["requests"]:
        raise SystemExit(f"refusing: the projection is over ${args.cap} or the pilot parsed under 90%")
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
    fails, refused = set(s.get("failed", [])), set(s.get("refused", []))
    for i, info in s["batches"].items():
        if info["collected"] or c.messages.batches.retrieve(i).processing_status != "ended":
            continue
        out = collections.defaultdict(list)
        for r in c.messages.batches.results(i):
            if r.result.type != "succeeded":
                fails.add(r.custom_id)
                continue
            s["spent"] = s.get("spent", 0) + cost(usage_of(r.result.message))
            labs = parse(side, text_of(r.result.message), part_of(allc, r.custom_id))
            if labs is None:
                fails.add(r.custom_id)
                if r.result.message.stop_reason == "refusal":
                    refused.add(r.custom_id)
            else:
                fails.discard(r.custom_id)
                refused.discard(r.custom_id)
                out[labs[0]["layer"]] += labs
        for layer, labs in out.items():
            with open(OUT[side] / "labels" / f"L{layer:02d}.jsonl", "a", encoding="utf-8") as fh:
                fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in labs)
        info["collected"], s["failed"], s["refused"] = True, sorted(fails), sorted(refused)
        save_state(side, s)
        print(f"{time.strftime('%H:%M:%S')} collected {i}", flush=True)
    print(f"{len(fails)} requests failed or unparsed; spent ${s.get('spent', 0):.2f}")


def retry(args) -> None:
    side = args.side
    s, c, system, allc = load_state(side), client(), system_prompt(side), chunks(side)
    sent = collections.Counter(x for b in s["batches"].values() for x in b["ids"])
    # a chunk that failed twice goes card by card: a refusal never passes as it was, and one card's slip sinks ten
    ids = [y for x in s.get("failed", [])
           for y in ([f"{x}_{k}" for k in range(len(allc[x]))] if x in allc and sent[x] >= 2 else [x])]
    if ids:
        b = c.messages.batches.create(requests=[request(side, cid, part_of(allc, cid), system, args.thinking) for cid in ids])
        split = set(s["failed"]) - set(ids)
        s["batches"][b.id] = {"ids": ids, "collected": False, "thinking": args.thinking, "retry": True}
        s["failed"] = sorted(set(s["failed"]) - split)  # a split chunk's cards stand for it now
        s["refused"] = sorted(set(s.get("refused", [])) - split)
        save_state(side, s)
        print("retry batch", b.id, len(ids), "requests", f"({len(split)} chunks card by card)" if split else "")


def shards(args) -> None:
    """The labels table in catalog row order. A layer whose every unit has a card takes its labels from this run alone;
    one carded in part (``cards --flipped``) lands its new labels on the table already there. A carded pole or head the
    run left without a label gets empty labels, NaN confidences and a note saying why; one neither carded nor in the
    table is a KeyError."""
    side, fields = args.side, FIELDS[args.side]
    s, allc = load_state(side), chunks(side)
    unit = (lambda layer, u, p: (layer, u, p)) if side == "neurons" else (lambda layer, u, p: (layer, u))
    signs = (1, -1) if side == "neurons" else (None,)
    carded = {unit(layer, u, p) for part in allc.values() for layer, u, _ in part for p in signs}
    why = {}
    for cid in s.get("failed", []):
        for layer, u, _ in part_of(allc, cid):
            for p in signs:
                why[unit(layer, u, p)] = ("no label: the API refused this card" if cid in s.get("refused", [])
                                          else "no label: the labeller's replies to this card didn't parse")
    got = {}
    for path in sorted((OUT[side] / "labels").glob("L*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                o = json.loads(line)
                got[(o["layer"], o["neuron"], o["pole"]) if side == "neurons" else (o["layer"], o["head"])] = o

    def blank(key) -> dict:
        names = ("layer", "neuron", "pole") if side == "neurons" else ("layer", "head")
        return dict(zip(names, key)) | {k: "" for k in fields} | {f"{k}_conf": float("nan") for k in fields} | {
            "note": why.get(key, "no label: not collected")}
    missing = sorted(carded - set(got))
    got |= {k: blank(k) for k in missing}

    def columns(rows: list[dict]) -> dict:
        cols = {}
        for k in fields:
            cols[k] = pa.array([r[k] for r in rows])
            cols[f"{k}_conf"] = pa.array([r[f"{k}_conf"] for r in rows], pa.float32())
        cols["note"] = pa.array([r["note"] for r in rows])
        return cols
    if side == "neurons":
        for layer in sorted({k[0] for k in carded}):
            path = neuron_table("labels", layer)
            here = {k: v for k, v in got.items() if k[0] == layer and k in carded}
            have = {(r["layer"], r["neuron"], r["pole"]): r for r in pq.read_table(path).to_pylist()} if path.exists() else {}
            f = max(n for _, n, _ in (have.keys() | here.keys())) + 1
            if len(here) < 2 * f:  # carded in part: the rest stay as they were
                here = have | here
            rows = [here[(layer, n, 1)] for n in range(f)] + [here[(layer, n, -1)] for n in range(f)]
            write(neuron_keys(layer, f) | columns(rows), path)
    else:
        keys = pq.read_table(head_table("reads"), columns=["layer", "head", "kv_head"])
        L, H = max(keys.column("layer").to_pylist()) + 1, keys.num_rows // (max(keys.column("layer").to_pylist()) + 1)
        rows = [got[(layer, h)] for layer in range(L) for h in range(H)]
        write(head_keys(L, H, H // (max(keys.column("kv_head").to_pylist()) + 1)) | columns(rows), head_table("labels"))
    print(f"{time.strftime('%H:%M:%S')} {side}/labels written" + (f", {len(missing)} without a label" if missing else ""))


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
    if args.summary_only:  # the saved per-head scores, summarized again
        result = json.loads((SUMMARY / f"labels_heads_{NAME}.json").read_text())
        lines = summarize(result["heads"], result)
        (SUMMARY / f"labels_heads_{NAME}.json").write_text(json.dumps(result, indent=1))
        (SUMMARY / f"labels_heads_{NAME}.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
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
                held = list(range(N_CONTEXTS, min(HELD_OUT, len(src["top_doc"]))))
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
    """The card's section on the head labels, from the check, by depth band (``arch.depth_bands``). The judge picks one
    `writes` label of four, so its AUROC for the head's own label against the three others is 1 when it picks it and
    1/3 when it picks another (the own label loses to that one and ties the other two): chance is 0.5."""
    roles = arch.depth_bands(MODEL)
    band_of = lambda hid: next(f"{r} L{lo}-L{hi}" for r, (lo, hi) in roles.items() if lo <= int(hid[1:].split("h")[0]) <= hi)
    for hid, v in per.items():
        v["band"] = band_of(hid)
        if "writes" in v:
            v["writes_auroc"] = 1.0 if v["writes"] else 1 / 3

    def stat(key, band=None):
        xs = [v[key] for v in per.values() if key in v and (band is None or v["band"] == band)]
        return (sum(xs) / len(xs), *boot(xs)) if xs else (float("nan"),) * 3
    own, ctl, wr = stat("own"), stat("control"), stat("writes_auroc")
    result["summary"] = {"firing": own, "control": ctl, "writes_auroc": wr, "writes_picked": stat("writes"),
                         "bands": {b: {k: stat(k, b)[0] for k in ("own", "control", "writes_auroc")}
                                   for b in dict.fromkeys(band_of(hid) for hid in sorted(per, key=lambda h: int(h[1:].split("h")[0])))}}
    lines = ["Every head described by Claude Haiku 5.5 from its card (batch API, effort low, no thinking):",
             "`fires` (where its output is large), `attends` (what it attends to there) and `writes` (what its",
             "output promotes), each with a confidence 0-1, and `note`. The cards show the head's pattern scores,",
             "how often it fires, one OV read for its depth, its QK reads and its drivers as query and key (24 tokens",
             "each), and its first 5 top contexts; nothing about its other top firings.",
             f"The prompt is `scripts/prompts/heads.md` with `scripts/prompts/{NAME}/heads.md` on GitHub.", "",
             f"**Checked blind** on {len(per)} random heads, {result['per_band']} a band, by Claude Sonnet 5.5:", "",
             (f"- **Firing:** given a head's `fires` and `attends` labels and {2 * result['shown']} contexts, "
              f"{result['shown']} of its held-out top firings (ranks 6-24) and {result['shown']} of another head of its band, "
              f"the scorer separates them at AUROC {own[0]:.2f} [{own[1]:.2f}, {own[2]:.2f}]; a third head's labels on the "
              f"same contexts score {ctl[0]:.2f}."),
             (f"- **Writes:** given what the head's firings actually promote in context (`out_R_z`, not on the card), a "
              f"judge picks the head's `writes` label among four from its band: AUROC {wr[0]:.2f} [{wr[1]:.2f}, {wr[2]:.2f}] "
              f"for its own label against the other three (chance 0.5)."), "",
             "| band | firing AUROC | control | writes AUROC |", "|---|---|---|---|"]
    for b, v in result["summary"]["bands"].items():
        lines.append(f"| {b} | {v['own']:.2f} | {v['control']:.2f} | {v['writes_auroc']:.2f} |")
    return lines


def torch_load(path):
    import torch
    return torch.load(path, weights_only=False, map_location="cpu", mmap=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("cards", "pilot", "submit", "status", "collect", "retry", "shards", "check"))
    ap.add_argument("--summary-only", action="store_true", help="check: re-summarize the saved head check, no batch")
    ap.add_argument("--side", choices=("neurons", "heads"), default="neurons")
    ap.add_argument("--thinking", choices=("on", "off"), default="off")
    ap.add_argument("--n", type=int, default=60, help="pilot requests")
    ap.add_argument("--cap", type=float, default=90.0, help="most the projected full run may cost, dollars")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--flipped", action="store_true", help="cards: only the neurons with a pole on SiLU's negative lobe")
    args = ap.parse_args()
    if args.what == "check":
        args.side = "heads"
    OUT[args.side].mkdir(parents=True, exist_ok=True)
    {"cards": cards, "pilot": pilot, "submit": submit, "status": status, "collect": collect,
     "retry": retry, "shards": shards, "check": check}[args.what](args)


if __name__ == "__main__":
    main()
