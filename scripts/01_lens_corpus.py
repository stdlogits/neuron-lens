"""Export the lens-fit corpus: the first documents of DCLM-shuffled, as one JSONL.

DCLM 100BT-shuffled is globally shuffled (seed 42), so its first rows are a uniform sample of
the whole set. Rows [0, n) go to ``data/corpus/dclm.jsonl`` (a ``text`` and an ``id`` per
line), at a pinned dataset revision, so a larger ``--docs`` later extends the same list and a
fit resumes onto it. The fit's own even and odd documents give the split-half check. Also
writes a provenance record with the revision and row ids.

    python scripts/01_lens_corpus.py [--docs 1000]
"""
import argparse
import json
import os
import sys

from datasets import load_dataset
from huggingface_hub import HfApi

from neuron_lens.paths import CORPUS_DIR

REPO = "HuggingFaceFW/dclm_100BT-shuffled"
PROVENANCE = CORPUS_DIR / "dclm.provenance.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=1000)
    ap.add_argument("--revision", default=None,
                    help="dataset revision (default: the one recorded in the provenance, else the latest)")
    args = ap.parse_args()
    revision = args.revision
    if revision is None and PROVENANCE.exists():
        revision = json.loads(PROVENANCE.read_text())["revision"]
    revision = revision or HfApi().dataset_info(REPO).sha
    stream = load_dataset(REPO, split="train", streaming=True, revision=revision)
    rows = []
    for row in stream:
        rows.append(row)
        if len(rows) == args.docs:
            break
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    path = CORPUS_DIR / "dclm.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"text": row["text"], "id": row["id"]}, ensure_ascii=False) + "\n")
    chars = [len(row["text"]) for row in rows]
    record = {"repo": REPO, "revision": revision, "file": path.name, "rows": [0, len(rows)],
              "ids": [row["id"] for row in rows], "chars": chars}
    PROVENANCE.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{path}: {len(rows)} docs at revision {revision[:12]}, median {sorted(chars)[len(rows) // 2]} chars")


if __name__ == "__main__":
    main()
    # A streamed dataset leaves pyarrow threads that hang interpreter shutdown.
    sys.stdout.flush()
    os._exit(0)
