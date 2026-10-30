"""Corpora: JSONL rows (``text`` fields), read as Drowse's ``lens fit --corpus`` reads them."""
import hashlib
import json
from pathlib import Path

from neuron_lens.paths import CORPUS_DIR

DOC_CHARS = 4000  # Drowse's per-document character cap
ACT_SLICES = ("dclm", "zh", "code", "chat")  # 05_act_corpus.py's slices, in the activation files' order


def _rows(path: Path, tok, seq_len: int):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                text = json.loads(line)["text"] if line.startswith("{") else line
                yield tok(text[:DOC_CHARS])["input_ids"][:seq_len]


def lens_docs(path: Path, tok, n: int, seq_len: int, min_len: int) -> list[list[int]]:
    """Token rows of the first ``n`` documents of ``path``, each cut to ``DOC_CHARS``
    characters and ``seq_len`` tokens; rows shorter than ``min_len`` tokens are dropped."""
    rows = []
    for r in _rows(path, tok, seq_len):
        rows.append(r)
        if len(rows) == n:
            break
    return [r for r in rows if len(r) >= min_len]


def act_file(name: str, model: str | None = None) -> Path:
    """An activation slice's JSONL (``05_act_corpus.py``). The chat slice is rendered for its model
    (in the chat template, or as plain turns), so models after Qwen3-8B have their own:
    ``act-chat-<model>.jsonl``. The other slices are shared."""
    from neuron_lens.catalog import MODEL
    model = model or MODEL
    if name == "chat" and model != "Qwen/Qwen3-8B":
        return CORPUS_DIR / f"act-chat-{model.split('/')[-1].lower()}.jsonl"
    return CORPUS_DIR / f"act-{name}.jsonl"


def act_docs(tok, seq_len: int, min_len: int, n: int | None = None) -> list[tuple[int, int, list[int]]]:
    """Every document of the activation slices (``act_file``), cut as
    ``lens_docs`` cuts them: ``(slice index, line in the slice's file, token row)``, the first ``n``
    lines of each slice, rows shorter than ``min_len`` dropped. The provenance records
    (``act-<slice>.provenance.json``) give each line's source dataset, revision and row id."""
    out = []
    for si, name in enumerate(ACT_SLICES):
        for line, r in enumerate(_rows(act_file(name), tok, seq_len)):
            if n is not None and line == n:
                break
            if len(r) >= min_len:
                out.append((si, line, r))
    return out


def rows_sha256(rows: list[list[int]]) -> str:
    """Drowse's ``token_ids_v1`` corpus hash."""
    return hashlib.sha256(repr([list(r) for r in rows]).encode("utf-8")).hexdigest()
