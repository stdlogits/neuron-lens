"""Export the activation corpora: held-out slices for activation statistics, one JSONL each.

- ``dclm``: DCLM-shuffled rows from 1,000 on, at the lens corpus's pinned revision, so no
  document the lenses were fitted on is reused (the lens corpus is rows [0, 1000)).
- ``zh``: FineWeb-2 ``cmn_Hani``, the first rows (Chinese web text; DCLM is English).
- ``code``: codeparrot-clean, the first rows (Python files), with leading comment and blank
  lines (license headers) dropped.
- ``chat``: UltraChat-200k ``test_sft``, rendered with the model's chat template, so the chat
  model is seen in the speaker frame it was trained for (a base model without one, as Gemma 4
  12B, sees plain ``User:`` / ``Assistant:`` turns).

Each slice goes to ``data/corpus/act-<slice>.jsonl`` (``text`` and ``id``; the chat slice of a
model other than Qwen3-8B to ``act-chat-<model>.jsonl``, ``corpus.act_file``) with a provenance record
at the dataset revision used. Only the DCLM slice is a uniform sample. ``NEURON_LENS_MODEL`` picks
the model whose chat template renders the chat slice.

    python scripts/05_act_corpus.py [--docs dclm=4000,zh=1000,code=1000,chat=1000]
"""
import argparse
import json
import os
import sys

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from neuron_lens.catalog import MODEL
from neuron_lens.corpus import act_file
from neuron_lens.paths import CORPUS_DIR

DCLM_SKIP = 1000  # the lens corpus, rows [0, 1000)
SOURCES = {
    "dclm": ("HuggingFaceFW/dclm_100BT-shuffled", None, "train"),
    "zh": ("HuggingFaceFW/fineweb-2", "cmn_Hani", "train"),
    "code": ("codeparrot/codeparrot-clean", None, "train"),
    "chat": ("HuggingFaceH4/ultrachat_200k", None, "test_sft"),
}


def strip_header(code: str) -> str:
    lines = code.splitlines(keepends=True)
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
        i += 1
    return "".join(lines[i:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="dclm=4000,zh=1000,code=1000,chat=1000")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for item in args.docs.split(","):
        name, n = item.split("=")
        n = int(n)
        repo, config, split = SOURCES[name]
        if name == "dclm":
            revision = json.loads((CORPUS_DIR / "dclm.provenance.json").read_text())["revision"]
        else:
            revision = HfApi().dataset_info(repo).sha
        stream = load_dataset(repo, config, split=split, streaming=True, revision=revision)
        skip = DCLM_SKIP if name == "dclm" else 0
        rows = []
        for i, row in enumerate(stream):
            if i < skip:
                continue
            if name == "code":
                text, rid = strip_header(row["content"]), f"{row['repo_name']}/{row['path']}"
            elif name == "chat":
                text = (tok.apply_chat_template(row["messages"], tokenize=False) if tok.chat_template else
                        "".join(f"{m['role'].capitalize()}: {m['content']}\n\n" for m in row["messages"]))
                rid = row["prompt_id"]
            else:
                text, rid = row["text"], row["id"]
            if text.strip():
                rows.append({"text": text, "id": rid})
            if len(rows) == n:
                break
        path = act_file(name)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        record = {"repo": repo, "config": config, "split": split, "revision": revision,
                  "file": path.name, "skip": skip, "n": len(rows), "ids": [r["id"] for r in rows]}
        path.with_name(path.name.replace(".jsonl", ".provenance.json")).write_text(json.dumps(record, indent=1) + "\n")
        print(f"{path}: {len(rows)} docs at {repo} {revision[:12]}", flush=True)


if __name__ == "__main__":
    main()
    # A streamed dataset leaves pyarrow threads that hang interpreter shutdown.
    sys.stdout.flush()
    os._exit(0)
