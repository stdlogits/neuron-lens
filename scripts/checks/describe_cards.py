"""Evidence cards for the description pilot (findings §8), one markdown block per pole.

A card gathers what the catalog knows about a pole, from every side:

- **weights** (``reads``): ``‖W_down[:, n]‖``, ``cos(gate, up)``, and the eight lens reads of record,
  top 8 kept tokens each: the R-lens and S-lens reads of the write and of the read direction, plain
  and layer-contrastive (``write_R``, ``write_R_z``, ``write_S``, ``write_S_z``, ``read_R``, ...);
- **exact drivers** (``drivers``): the tokens that drive the pole most in real text, at their own
  position and summed over the 16 after;
- **the layer's legibility** (``legible``): the share of the layer's poles whose each read passes its
  test at their top firings (findings §7), a prior on which sides hold at this depth;
- **activations** (``acts``): the typical high activation (``act_top``, the median of the top
  documents' extremes), the maximum, the firing rate (past a quarter of ``act_top``) overall and per
  slice, the position it peaks at when a third of the shown documents share one, the first
  ``N_CONTEXTS`` top contexts (12 tokens before the peak in ``⟦ ⟧``, 3 after), and the first position
  (no context) when it fires there past a quarter of ``act_top``;
- **steering** (``causal_lens.py``'s ``data/causal/<model>-causal.pt``): the 8 kept tokens whose mean
  log-prob rises most when the pole's write is added at 1× and 4× its typical activation (``α`` from
  the activation statistics that file was measured on), and the largest rise;
- **outside labels:** ``neuron_lens.known`` (the persona experiment's steering-verified neurons and
  ``EXTERNAL``'s sources) and the Neuron Persona Library's persona rows for the pole
  (``paths.LIBRARY``), each with its source.

Nothing on a card summarizes the pole's top documents past the first ``N_CONTEXTS``, which
``describe_score.py`` holds out (AGENTS.md, "Description cards stay leak-free"): no peak-token counts
(``acts.peak_tokens`` counts all of them), none of the pole's own ``legible`` columns (its tests and
modes run on all its top firings: only the layer's shares are shown), and no ablation (``causal_lens``
ablated in the pole's own top documents).

The sample: ``--per-layer`` neurons of each layer, drawn (``random.Random(1)``, layers in order) from
the causal file's picks (``causal_lens.sample_neurons``: 32 random neurons a layer and the named
ones), both poles, and every named pole (``KNOWN``; ``EXTERNAL``, both poles where the source names
none). At the default, the recorded pilot's 197 poles (``data/summary/describe_labels_<model>.jsonl``):
144 drawn (72 neurons) and 53 named.

Explainers (Claude Code subagents, Sonnet, briefed with ``describe_prompt.md``) read
``cards/batch_XX.md`` and write ``cards/labels_XX.jsonl`` beside them; ``describe_score.py`` scores the
labels. Outputs, under ``data/checks/describe/<model>/``: ``cards/batch_XX.md`` (``--batch`` poles a
file, in pole order) and ``poles.json`` (``[layer, neuron, sign]`` per pole). Needs the catalog tables,
the causal file and the tokenizer; no model, about 10 seconds. Qwen3-8B only (the named poles, the causal
sample with them, and the library are Qwen3-8B's).

    python scripts/checks/describe_cards.py [--per-layer 2] [--batch 24]
"""
import argparse
import json
import random
from collections import Counter, defaultdict

import torch
from causal_lens import CAUSAL
from lens_blind import pole_id

from neuron_lens import arch
from neuron_lens.catalog import MODEL, N_CONTEXTS, kept, load_layer, vocab
from neuron_lens.known import EXTERNAL, KNOWN
from neuron_lens.paths import DATA, LIBRARY

NAME = arch.stem(MODEL)
OUT = DATA / "checks" / "describe" / NAME
TESTS = ("write_R", "write_S", "read_R", "read_S")  # legible's tests (findings §7)
FDR = 0.05
N_STEER = 8  # tokens per steering line
READS = (
    ("write_R", "writes (R-lens of record: what the write promotes downstream)"),
    ("write_R_z", "writes, layer-contrastive (R-lens z)"),
    ("write_S", "writes, as input tokens (S-lens of the write: tokens whose footprint it resembles)"),
    ("write_S_z", "writes, as input tokens, layer-contrastive (S-lens z)"),
    ("read_R", "reads (forward R-lens of the read direction, from the layer's MLP input)"),
    ("read_R_z", "reads, layer-contrastive (forward R-lens z)"),
    ("read_S", "reads, as input tokens (S-lens: tokens here or earlier that drive it)"),
    ("read_S_z", "reads, as input tokens, layer-contrastive (S-lens z)"),
    ("drivers", "exact drivers (tokens that drive it most in real text, at their own position)"),
    ("drivers_sum", "exact drivers, summed over their position and the 16 after"),
)


def named() -> set[tuple[int, int, int]]:
    """The named poles: ``KNOWN``'s, and ``EXTERNAL``'s (both poles where the source names none)."""
    out = {(l, n, s) for l, n, s in KNOWN.values()}
    return out | {(l, n, sign) for _, _, l, n, s in EXTERNAL for sign in ((s,) if s else (1, -1))}


def sample(picks: list[tuple[int, int]], per_layer: int) -> list[tuple[int, int, int]]:
    """``per_layer`` neurons of each layer's picks, both poles, with the named poles, sorted."""
    rng = random.Random(1)
    by_layer = defaultdict(list)
    for l, n in picks:
        by_layer[l].append(n)
    drawn = {(l, n, s) for l in sorted(by_layer) for n in rng.sample(by_layer[l], per_layer) for s in (1, -1)}
    return sorted(drawn | named())


def outside_labels() -> dict[tuple[int, int, int], list[str]]:
    """(layer, neuron, sign) -> each outside label of the pole, with its source."""
    labels = defaultdict(list)
    for name, (l, n, s) in KNOWN.items():
        labels[(l, n, s)].append(f"persona experiment (steering-verified persona): {name}")
    for source, label, l, n, s in EXTERNAL:
        for sign in ((s,) if s else (1, -1)):
            labels[(l, n, sign)].append(f"{source}: {label}")
    if not LIBRARY.exists():
        print(f"no Neuron Persona Library at {LIBRARY}: cards carry no library labels")
        return labels
    lib = json.loads(LIBRARY.read_text())
    persona = lib["categories"].index("persona")
    for r in lib["rows"]:
        row = dict(zip(lib["fields"], r))
        if row["cat"] == persona:
            labels[(row["L"], row["n"], row["s"])].append(
                f"Neuron Persona Library (steering on 'Who are you?'): {row['persona']} at strength {row['a']}")
    return labels


def cards(poles: list[tuple[int, int, int]], causal: dict) -> list[str]:
    """The card of each pole, in order."""
    v, keep = vocab(), kept()
    show = lambda ids: " | ".join(v[i] for i in ids)
    picks = {pick: i for i, pick in enumerate(causal["picks"])}
    labels = outside_labels()
    n_layers = arch.text_config(MODEL).num_hidden_layers
    tables, shares, out = {}, {}, []
    for l, n, s in poles:
        if l not in tables:
            tables[l] = load_layer(l)
            shares[l] = {t: (tables[l].column(f"{t}_q").to_numpy() <= FDR).mean() for t in TESTS}
        t = tables[l]
        e = t.slice(n if s > 0 else t.num_rows // 2 + n, 1).to_pylist()[0]
        lines = [f"### {pole_id(l, n, s)}",
                 f"- layer {l} of {n_layers}; ‖W_down[:, n]‖ {e['w_norm']:.3f}, cos(gate, up) {e['cos_gate_up']:+.2f}"]
        lines += [f"- {title}: {e[f'{key}_str']}" for key, title in READS if e[f"{key}_str"]]
        lines.append("- at this layer, the share of all poles whose read holds at their own top firings: "
                     + ", ".join(f"{k} {x:.2f}" for k, x in shares[l].items()))
        lines.append(f"- activation: typical high {e['act_top']:.2f}, max {e['act_max']:.2f}; fires (past a quarter "
                     f"of typical) at {e['rate']:.1e} of positions; by slice: "
                     + ", ".join(f"{k} {e[f'rate_{k}']:.1e}" for k in ("dclm", "zh", "code", "chat")))
        # Only the shown documents: the rest are held out for describe_score.py.
        fired = [p for p, x in zip(e["top_pos"][:N_CONTEXTS], e["top_val"][:N_CONTEXTS]) if x > 0]
        if fired:
            pos, k = Counter(fired).most_common(1)[0]
            if k >= len(fired) / 3:
                lines.append(f"- peaks at position {pos} in {k} of these {len(fired)} top documents")
        lines += [f"  - context: {c}" for c in str(e["contexts"]).split(" ‖ ")[:N_CONTEXTS] if c]
        if e["first_max"] > e["act_top"] / 4:
            lines.append(f"- at the first position (no context; skipped above): up to {e['first_max']:.3g} on "
                         f"{e['first_tokens']}; on '<|im_start|>' {e['first_chat']:.3g}")
        if (l, n) in picks:
            i, pi = picks[(l, n)], 0 if s > 0 else 1
            for di, dose in enumerate(causal["meta"]["doses"]):
                d = causal["steer"][i, pi, di].float()[keep]
                top = d.topk(N_STEER)
                lines.append(f"- steering at {dose:g}× typical activation (α {dose * causal['alpha'][i, pi]:.3g}) "
                             f"raises: {show(keep[top.indices].tolist())} (max Δlog-prob {top.values[0]:.2f})")
        lines += [f"- outside label: {x}" for x in labels.get((l, n, s), [])]
        out.append("\n".join(lines))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-layer", type=int, default=2)
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()
    if MODEL != arch.QWEN:
        raise SystemExit(f"the description pilot is Qwen3-8B's (its named poles and causal sample); not {MODEL}")
    causal = torch.load(CAUSAL / f"{NAME}-causal.pt", weights_only=False, mmap=True, map_location="cpu")
    poles = sample(causal["picks"], args.per_layer)
    body = cards(poles, causal)
    (OUT / "cards").mkdir(parents=True, exist_ok=True)
    (OUT / "poles.json").write_text(json.dumps(poles))
    for old in (OUT / "cards").glob("batch_*.md"):
        old.unlink()
    for b in range(0, len(body), args.batch):
        (OUT / "cards" / f"batch_{b // args.batch:02d}.md").write_text("\n\n".join(body[b:b + args.batch]) + "\n")
    print(f"{len(body)} cards ({len(named())} named poles) in {(len(body) + args.batch - 1) // args.batch} files "
          f"-> {OUT / 'cards'}")


if __name__ == "__main__":
    main()
