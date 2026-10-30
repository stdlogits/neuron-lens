"""Token counts over the activation corpora, to mask undertrained tokens in token reads.

Every document of each ``data/corpus/act-<slice>.jsonl`` (``05_act_corpus.py``: held-out DCLM,
Chinese web text, Python and chat transcripts), whole, tokenized with the model's tokenizer
(``NEURON_LENS_MODEL``; Qwen3-8B's is shared by all Qwen3 sizes). Writes ``data/token_counts.pt``
(Qwen3-8B) or ``data/token_counts-<model>.pt`` ``{"counts": [vocab] int64 (all slices), "by_slice":
{slice: [vocab]}, "tokenizer"}``. The mask is ``counts >= 5``; the Chinese slice keeps common Chinese
tokens in it, which a DCLM-only count dropped.

    python scripts/03_token_counts.py [--slices dclm,zh,code,chat]
"""
import argparse
import json

import torch
from transformers import AutoTokenizer

from neuron_lens.arch import counts_path
from neuron_lens.catalog import MODEL
from neuron_lens.corpus import act_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="dclm,zh,code,chat")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    by_slice = {}
    for name in args.slices.split(","):
        counts = torch.zeros(len(tok), dtype=torch.long)
        with open(act_file(name), encoding="utf-8") as f:
            for line in f:
                ids = tok(json.loads(line)["text"])["input_ids"]
                counts += torch.bincount(torch.tensor(ids), minlength=len(tok))
        by_slice[name] = counts
        print(f"{name}: {counts.sum()} tokens, {(counts > 0).sum()} ids seen")
    total = sum(by_slice.values())
    torch.save({"counts": total, "by_slice": by_slice, "tokenizer": MODEL}, counts_path(MODEL))
    print(f"{total.sum()} tokens; {(total > 0).sum()} of {len(tok)} vocabulary ids seen, "
          f"{(total >= 5).sum()} seen 5+ times")


if __name__ == "__main__":
    main()
