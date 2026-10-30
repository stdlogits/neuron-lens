"""The catalog: parquet tables under ``data/catalog/<model>/``, laid out as the release.

- ``neurons/<table>/L<layer>.parquet``: a row per (neuron, pole), neurons in order with pole ``+``
  first, then ``−``, with ``layer``, ``neuron`` and ``pole`` in front;
- ``heads/<table>.parquet``: a row per query head, with ``layer``, ``head`` and ``kv_head`` in front.

Tables, each from one pipeline stage: ``acts`` (``06_activations.py``), ``drivers``
(``07_drivers.py``), ``reads`` (``08_reads.py``), ``legible`` (``09_legible.py``) and ``labels``
(``10_labels.py``). Token reads cover the kept tokens (``kept``) and list 64 of them: ``*_ids``
(the model's token ids), ``*_val`` (their scores) and ``*_str`` (the top 8 decoded, `` | ``-joined).

The model is Qwen3-8B unless ``NEURON_LENS_MODEL`` names another (``google/gemma-4-12B``): every
stage reads it from here.

    python -m neuron_lens.catalog show 21 1180 -          # layer, neuron, pole
    python -m neuron_lens.catalog head 20 17              # layer, head
    python -m neuron_lens.catalog search 'Bible|圣经' [--column write_R_str] [--layers 8-20]
"""
import argparse
import functools
import itertools
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from neuron_lens.paths import DATA

MODEL = os.environ.get("NEURON_LENS_MODEL", "Qwen/Qwen3-8B")
TABLES = ("acts", "drivers", "reads", "legible", "labels")
TOP = 64  # tokens kept per read: signal holds to rank 64 (findings §7)
MIN_COUNT = 5  # tokens seen fewer times in the activation corpora are undertrained (findings §2)
CONTEXT_BEFORE, CONTEXT_AFTER = 12, 3
N_CONTEXTS = 5  # top contexts kept in the acts tables; the rest stay held out for scoring


def catalog_dir(model: str = MODEL):
    return DATA / "catalog" / model.split("/")[-1].lower()


def acts_path(model: str = MODEL):
    """``06_activations.py``'s collected state, with every document's tokens."""
    return DATA / "acts" / f"{model.split('/')[-1].lower()}.pt"


def doc_tokens(acts: dict) -> list[torch.Tensor]:
    """Each document's tokens from an activations file (stored flat, with each document's start)."""
    flat, start = acts["tokens"], acts["doc_start"].tolist()
    return [flat[a:b] for a, b in itertools.pairwise(start)]


def corpus_doc_tokens(model: str = MODEL, seq_len: int = 256) -> list[list[int]]:
    """Each document's tokens rebuilt from the activation slices, as ``06_activations.py`` cut them
    (``seq_len``), for where the activations file isn't at hand: ``docs.parquet`` gives each
    document's slice and line, and its token count, which every rebuilt row must match."""
    from neuron_lens.corpus import ACT_SLICES, _rows, act_file

    tok = tokenizer(model)
    rows = {name: list(_rows(act_file(name, model), tok, seq_len)) for name in ACT_SLICES}
    out = []
    for r in pq.read_table(catalog_dir(model) / "docs.parquet").to_pylist():
        ids = rows[r["slice"]][r["line"]]
        assert len(ids) == r["tokens"], f"document {r['doc']}: {len(ids)} tokens rebuilt, {r['tokens']} in the catalog"
        out.append(ids)
    return out


def neuron_table(table: str, layer: int, model: str = MODEL):
    return catalog_dir(model) / "neurons" / table / f"L{layer:02d}.parquet"


def head_table(table: str, model: str = MODEL):
    return catalog_dir(model) / "heads" / f"{table}.parquet"


@functools.cache
def tokenizer(model: str = MODEL):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model)


@functools.cache
def vocab(model: str = MODEL) -> list[str]:
    """Every token decoded, newlines as ``⏎`` and carriage returns as ``␍``."""
    tok = tokenizer(model)
    return [tok.decode([i]).replace("\n", "⏎").replace("\r", "␍") for i in range(len(tok))]


@functools.cache
def kept(model: str = MODEL) -> torch.Tensor:
    """The token ids every token read covers: seen ``MIN_COUNT``+ times in the activation corpora
    (``03_token_counts.py``), without byte-fallback pieces (tokens that decode to ``�``), which
    alone are out of distribution and top many early reads at once."""
    from neuron_lens.arch import counts_path
    counts = torch.load(counts_path(model), weights_only=False)["counts"]
    v = vocab(model)
    ids = (counts >= MIN_COUNT).nonzero().squeeze(1)
    return ids[torch.tensor(["�" not in v[i] for i in ids.tolist()])]


def kept_index(model: str = MODEL) -> torch.Tensor:
    """``[vocabulary]``: each token's row among ``kept``, −1 for the rest."""
    k = kept(model)
    idx = torch.full((len(tokenizer(model)),), -1)
    idx[k] = torch.arange(len(k))
    return idx


def fixed(t: torch.Tensor) -> pa.Array:
    """A ``[n, k]`` tensor as a fixed-size list column."""
    return pa.FixedSizeListArray.from_arrays(pa.array(t.flatten().numpy()), t.shape[1])


def joined(ids: torch.Tensor, k: int = 8, model: str = MODEL) -> pa.Array:
    v = vocab(model)
    return pa.array([" | ".join(v[i] for i in row[:k]) for row in ids.tolist()])


def put(cols: dict, name: str, ids: torch.Tensor, val: torch.Tensor) -> None:
    """One token list's columns: ``<name>_ids`` (token ids), ``<name>_val``, ``<name>_str``."""
    cols[f"{name}_ids"] = fixed(ids.to(torch.int32))
    cols[f"{name}_val"] = fixed(val.float())
    cols[f"{name}_str"] = joined(ids)


def neuron_keys(layer: int, f: int) -> dict:
    """The key columns of a neuron table: ``+`` poles of neurons ``0..f-1``, then ``−`` poles."""
    return {"layer": pa.array([layer] * 2 * f, pa.int16()), "neuron": pa.array(list(range(f)) * 2, pa.int16()),
            "pole": pa.array([1] * f + [-1] * f, pa.int8())}


def head_keys(n_layers: int, n_heads: int, group) -> dict:
    """The key columns of a head table, layer-major; ``kv_head`` is the key-value head a query head reads
    (``group`` query heads per key head: one number, or one per layer, as Gemma 4's)."""
    head = torch.arange(n_heads).repeat(n_layers)
    g = torch.tensor(group).repeat_interleave(n_heads) if isinstance(group, (list, tuple)) else group
    return {"layer": pa.array(torch.arange(n_layers).repeat_interleave(n_heads).numpy(), pa.int16()),
            "head": pa.array(head.numpy(), pa.int16()), "kv_head": pa.array((head // g).numpy(), pa.int16())}


def write(cols: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path)


def context(tok, ids, pos: int, src: int | None = None, after: int = CONTEXT_AFTER) -> str:
    """The text around ``pos`` with the token there in ``⟦ ⟧``, newlines as ``⏎``, ``after`` tokens
    after it. With ``src`` (a head's attended position), that token is in ``⟪ ⟫``; when it is more
    than ``CONTEXT_BEFORE`` tokens back, its own few tokens come first, then `` … ``."""
    ids = [int(i) for i in ids]
    dec = lambda a, b: tok.decode(ids[max(0, a):max(0, b)])
    start = max(0, pos - CONTEXT_BEFORE)
    tail = f"⟦{dec(pos, pos + 1)}⟧{dec(pos + 1, pos + 1 + after)}"
    if src is None or src == pos:
        text = dec(start, pos) + tail
    elif src >= start:
        text = f"{dec(start, src)}⟪{dec(src, src + 1)}⟫{dec(src + 1, pos)}" + tail
    else:
        text = f"{dec(src - 4, src)}⟪{dec(src, src + 1)}⟫{dec(src + 1, src + 3)} … {dec(pos - 8, pos)}" + tail
    return text.replace("\n", "⏎")


def load_layer(layer: int, model: str = MODEL) -> pa.Table:
    """A layer's neuron tables side by side (each holds the same rows in the same order)."""
    cols = {}
    for t in TABLES:
        path = neuron_table(t, layer, model)
        if path.exists():
            tab = pq.read_table(path)
            for name in tab.column_names:
                cols.setdefault(name, tab.column(name))
    return pa.table(cols)


def load_heads(model: str = MODEL) -> pa.Table:
    """The head tables side by side, one row per head."""
    cols = {}
    for t in TABLES:
        path = head_table(t, model)
        if path.exists():
            tab = pq.read_table(path)
            for name in tab.column_names:
                cols.setdefault(name, tab.column(name))
    return pa.table(cols)


def entry(layer: int, neuron: int, pole: int, model: str = MODEL) -> dict:
    t = load_layer(layer, model)
    row = neuron if pole > 0 else len(t) // 2 + neuron
    return {k: v[0] for k, v in t.slice(row, 1).to_pydict().items()}


def head_entry(layer: int, head: int, model: str = MODEL) -> dict:
    t = load_heads(model)
    n_heads = len(t) // (max(t.column("layer").to_pylist()) + 1)
    return {k: v[0] for k, v in t.slice(layer * n_heads + head, 1).to_pydict().items()}


def show(e: dict) -> None:
    for k, v in e.items():
        if k.endswith(("_ids", "_val")) or k.startswith(("top_", "src_pos")):
            continue
        if k == "contexts":
            for c in str(v).split(" ‖ "):
                print(f"  {'context':14s} {c}")
        else:
            print(f"  {k:14s} {v:.4g}" if isinstance(v, float) else f"  {k:14s} {v}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("show", "head", "search"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--layers", default=None, help="a-b or a,b,c")
    ap.add_argument("--column", default="write_R_str")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    if args.what == "show":
        layer, neuron = int(args.args[0]), int(args.args[1])
        for pole in [1, -1] if len(args.args) < 3 else [1 if args.args[2] == "+" else -1]:
            print(f"\nL{layer} n{neuron} {'+' if pole > 0 else '−'}")
            show(entry(layer, neuron, pole))
    elif args.what == "head":
        layer, head = int(args.args[0]), int(args.args[1])
        print(f"\nL{layer}h{head}")
        show(head_entry(layer, head))
    else:
        pattern = re.compile(args.args[0])
        if args.layers is None:
            layers = range(len(list((catalog_dir() / "neurons" / "reads").glob("L*.parquet"))))
        elif "-" in args.layers:
            a, b = args.layers.split("-")
            layers = range(int(a), int(b) + 1)
        else:
            layers = [int(x) for x in args.layers.split(",")]
        hits = 0
        for layer in layers:
            col = load_layer(layer).column(args.column).to_pylist()
            f = len(col) // 2
            for row, s in enumerate(col):
                if s and pattern.search(s):
                    print(f"L{layer} n{row % f} {'+' if row < f else '−'}  {s}")
                    hits += 1
                    if hits >= args.limit:
                        return


if __name__ == "__main__":
    main()
