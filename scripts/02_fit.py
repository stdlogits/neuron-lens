"""Fit an R-lens or an S-lens (``neuron_lens.transport``) over a lens corpus, or the residual's scale (``--lens scale``,
``neuron_lens.metric``: forward passes only, the per-coordinate σ of ``x / rms`` at every norm's input, which turns
a covector into a vector for the R-lens; text only, minutes per 1,000 documents).

Both lenses sum over offsets and take RelP derivatives. Documents are the first ``--docs`` rows
of a corpus JSONL in ``data/corpus/``, read as Drowse reads them. Document ``i`` goes to half
``i % 2``. The fit is saved every ``--save-minutes`` and at the end to
``data/fits/<lens>-<model>-L<layers>-<corpus>[-T<len>].pt`` (``{"meta", "state", "next_doc"}``; ``-T<len>``
for ``--full-length`` fits), and a rerun resumes from it, also with a larger ``--docs``. At the end it prints each layer's
split-half difference, how differently the two halves of the corpus read. The sums live on the
model's device unless ``--acc-device`` says otherwise (Qwen3-8B: 9.5 GB for the R-lens and 9.7 GB
for the S-lens, each with two targets; ``--targets`` fits one). Export R-lenses to Drowse with
``04_export_rlens.py``.

Several GPUs: one process per GPU with ``--shard i/n`` (documents ``i, i + n, ...``) and
``CUDA_VISIBLE_DEVICES=i``, each writing its own file; ``Accumulator.merge`` (or the
export) adds them up.

    python scripts/02_fit.py --lens S --model Qwen/Qwen3-0.6B --layers 16 --docs 12 --detach
    python scripts/02_fit.py --lens R --model Qwen/Qwen3-8B --docs 1000 --device cuda --detach
    python scripts/02_fit.py --lens R --model Qwen/Qwen3.5-35B-A3B --docs 1000 --seq-len 512 \
        --full-length --dim-batch 32 --device cuda --detach

Qwen3.5's DeltaNet layers hold a memory of about 128 keys a head, which 128-token documents
barely fill: ``--full-length`` fits documents of exactly ``--seq-len`` tokens (the first ones of
the corpus that long). On an H100, a dim batch of 32 with the sums on the GPU takes about 19 s per
512-token document for either lens (a larger batch is no faster). On CUDA in bf16, fp32 matrix
products run in TF32, as DeltaNet's kernels do (``neuron_lens.kernels``).

A model whose layers don't fit on the GPU beside the fit's working set keeps the first ``--offload``
layers there and streams the others from pinned host memory (``layers.offload``): Qwen3-32B on an
80 GB H100 holds 62 GB of layers against about 50 GB of sums, tangents and contracted MLPs.

    python scripts/02_fit.py --lens R --model Qwen/Qwen3-32B --docs 1000 --offload 20 --device cuda --detach
"""
import argparse
import os
import subprocess
import sys
import time
import zlib

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from neuron_lens import metric
from neuron_lens.corpus import lens_docs, rows_sha256
from neuron_lens.layers import offload
from neuron_lens.paths import CORPUS_DIR, FITS, LOGS, ROOT
from neuron_lens.transport import (
    DIM_BATCH,
    SKIP_FIRST,
    TARGETS,
    Accumulator,
    decoder_stack,
    lens_doc,
    r_lens_doc,
    s_lens_doc,
    target_layers,
)


def out_path(args):
    stem = "-".join([args.lens.lower(), args.model.split("/")[-1].lower(), f"L{args.n_layers}",
                     args.corpus.removesuffix(".jsonl")])
    if args.full_length:
        stem += f"-T{args.seq_len}"
    if args.n_shards > 1:
        stem += f"-s{args.shard}of{args.n_shards}"
    return FITS / f"{stem}.pt"


def load_model(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"auto": torch.float32 if device == "cpu" else torch.bfloat16,
             "fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    if device == "cuda" and dtype == torch.bfloat16:
        # fp32 products (DeltaNet's chunked ones) in TF32, which keeps more precision than the
        # model's bf16 products every tangent passes through
        torch.set_float32_matmul_precision("high")
    config = AutoConfig.from_pretrained(args.model)
    if args.layers is not None:  # only the first layers' weights load
        tc = config.get_text_config()
        tc.num_hidden_layers = args.layers
        if getattr(tc, "layer_types", None):
            tc.layer_types = tc.layer_types[:args.layers]
    lm = AutoModelForCausalLM.from_pretrained(args.model, config=config, dtype=dtype).eval().requires_grad_(False)
    if args.modality == "text":  # the decoder stack alone
        if args.offload is not None and args.lens != "scale":  # the scale's forward pass is the model's own
            return offload(decoder_stack(lm), device, args.offload), device, dtype
        return decoder_stack(lm).to(device), device, dtype
    args.lm = lm.to(device)  # image and audio documents need the whole model's embedders
    return decoder_stack(lm), device, dtype


def media_docs(args, proc) -> list[dict]:
    """The modality's documents (processor inputs), and ``args.rows``: per document its token
    ids and a CRC of its pixels or samples, for the resume check."""
    from neuron_lens.media import images, processed, sounds
    items = images(args.docs) if args.modality == "image" else \
        sounds(args.docs, "esc" if args.corpus == "esc50" else "mix")
    docs = [processed(proc, args.modality, x) for x in items]
    key = "pixel_values" if args.modality == "image" else "input_features"
    args.rows = [d["input_ids"][0].tolist() + [zlib.crc32(d[key].numpy().tobytes())] for d in docs]
    return docs


def media_doc(args, model, proc, doc: dict, acc: Accumulator, half: int, basis) -> int:
    """One image or audio document into ``acc``; returns its length. The rows before the first
    slot carry nothing (``first``); the S-lens also drops the text after the slots, which never
    reaches them (attention is causal outside an image's block)."""
    from neuron_lens.media import multimodal_embeds
    from neuron_lens.media import slot_token
    dev = model.embed_tokens.weight.device
    d = {k: v.to(dev, model.embed_tokens.weight.dtype) if v.is_floating_point() else v.to(dev) for k, v in doc.items()}
    ids = d["input_ids"][0]
    slots = ids == slot_token(proc, args.modality)
    where = slots.nonzero().squeeze(1)
    first, last = int(where[0]), int(where[-1])
    emb = multimodal_embeds(args.lm, d)
    block = slots if args.modality == "image" else None
    if args.lens == "S":
        T = last + 2  # through one past the last slot, which linearize drops
        rows = slots[first:T - 1]
        s_lens_doc(model, ids[:T], acc, half, dim_batch=args.dim_batch, basis=basis, source_rows=rows,
                   target_rows=rows, embeds=emb[:T], block=None if block is None else block[:T], first=first,
                   rules=args.rules)
    else:
        rows = slots[first:len(ids) - 1]
        r_lens_doc(model, ids, acc, half, dim_batch=args.dim_batch, seed_rows=torch.ones_like(rows),
                   read_rows=rows, embeds=emb, block=block, first=first, rules=args.rules)
    return len(ids)


def fit(args, model, tok, device: str, dtype) -> None:
    path = out_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    acc_device = device if args.acc_device == "auto" else args.acc_device
    d, n = model.config.hidden_size, len(model.layers)
    basis = None
    if args.modality == "text":
        rows = lens_docs(CORPUS_DIR / args.corpus, tok, args.docs, args.seq_len, SKIP_FIRST + 2, args.full_length)
        docs = None
    else:
        docs = media_docs(args, tok)  # tok is the processor here
        rows = args.rows
        if args.lens == "S":
            from neuron_lens.media import basis as media_basis
            basis = media_basis(args.lm, args.modality)
    meta = {"lens": args.lens, "model": args.model, "layers": n,
            "model_layers": args.model_layers, "corpus": args.corpus, "modality": args.modality,
            "basis": None if basis is None else f"{args.modality}:{len(basis)}", "rules": args.rules,
            "corpus_spec": f"file:{args.corpus}" if args.modality == "text" else f"hf:{args.corpus}",
            "rows_sha256": rows_sha256(rows),
            "docs": len(rows), "seq_len": args.seq_len, "full_length": args.full_length,
            "dim_batch": args.dim_batch,
            "targets": list(args.targets), "shard": [args.shard, args.n_shards],
            "dtype": str(dtype).removeprefix("torch.")}
    if path.exists():
        saved = torch.load(path, weights_only=False, map_location="cpu")
        loose = ("docs", "rows_sha256", "dim_batch")  # dim_batch changes speed, not the estimate
        older = {"modality": "text", "basis": None, "rules": "relp", "full_length": False}  # fits saved before these fields
        same = all(saved["meta"].get(k, older.get(k)) == v for k, v in meta.items() if k not in loose)
        if not same:
            raise SystemExit(f"{path} was fitted with {saved['meta']}, not {meta}")
        kind = metric.Sums if args.lens == "scale" else Accumulator
        acc, next_doc = kind.from_state(saved["state"], acc_device), saved["next_doc"]
    elif args.lens == "scale":
        acc, next_doc = metric.Sums(n, d, device=acc_device), 0
    else:
        layers = {t: target_layers(args.lens, t, n) for t in args.targets}
        acc, next_doc = Accumulator(layers, d, device=acc_device, d_in=None if basis is None else len(basis)), 0

    def save():
        tmp = path.with_suffix(".tmp")
        state = {"sums": {k: v.cpu() for k, v in acc.sums.items()}, "n_docs": acc.n_docs,
                 **({} if basis is None else {"basis": basis})}
        torch.save({"meta": meta, "state": state, "next_doc": next_doc}, tmp)
        os.replace(tmp, path)

    mine = [i for i in range(next_doc, len(rows)) if i % args.n_shards == args.shard]
    saved_at = time.monotonic()
    for count, i in enumerate(mine, 1):
        t0 = time.perf_counter()
        if args.lens == "scale":
            metric.scale_doc(model, torch.tensor(rows[i]), acc, i % 2, SKIP_FIRST)
            length = len(rows[i])
        elif docs is None:
            lens_doc(args.lens, model, torch.tensor(rows[i]), acc, i % 2, dim_batch=args.dim_batch)
            length = len(rows[i])
        else:
            length = media_doc(args, model, tok, docs[i], acc, i % 2, basis)
        if device == "cuda":
            torch.cuda.synchronize()
        next_doc = i + 1
        if count == len(mine) or time.monotonic() - saved_at > 60 * args.save_minutes:
            save()
            saved_at = time.monotonic()
        print(f"{time.strftime('%H:%M:%S')} {path.stem} doc {i + 1}/{len(rows)} "
              f"({length} tokens) in {time.perf_counter() - t0:.1f}s", flush=True)
    for target in args.targets:
        split = acc.split_half(target)
        print(f"split-half {target} by layer (‖even − odd‖ / ‖mean‖): "
              + " ".join(f"{x:.2f}" for x in split), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", choices=("R", "S", "scale"), required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--layers", type=int, default=None, help="first N decoder layers (default all)")
    ap.add_argument("--corpus", default=None,
                    help="text: a JSONL in data/corpus/ (default dclm.jsonl); image: coco; audio: esc50 or mix")
    ap.add_argument("--rules", default=None, choices=("relp", "true"),
                    help="derivative rules (default relp; true keeps every derivative exact)")
    ap.add_argument("--modality", default="text", choices=("text", "image", "audio"),
                    help="image or audio: documents of one input each (neuron_lens.media), seeds and reads at its "
                         "slots, the S-lens seeded in the subspace the embedder reaches")
    ap.add_argument("--docs", type=int, default=12)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--full-length", action="store_true",
                    help="only documents of at least --seq-len tokens (the first --docs of them)")
    ap.add_argument("--dim-batch", type=int, default=DIM_BATCH)
    ap.add_argument("--targets", default=None, help="targets (default: out,mid for R, mid,out for S)")
    ap.add_argument("--device", default="auto", help="auto (cuda, else cpu), cuda, mps, cpu")
    ap.add_argument("--dtype", default="auto", choices=("auto", "fp32", "bf16"))
    ap.add_argument("--acc-device", default="auto", help="where the sums live (auto: the model's)")
    ap.add_argument("--offload", type=int, default=None,
                    help="keep the first N decoder layers on the GPU and stream the rest from host memory")
    ap.add_argument("--shard", default="0/1", help="i/n: this process fits documents i, i+n, ...")
    ap.add_argument("--save-minutes", type=float, default=10)
    ap.add_argument("--detach", action="store_true")
    args = ap.parse_args()
    if args.offload is not None and args.modality != "text":
        raise SystemExit("--offload is for text fits")
    if args.lens == "scale" and args.modality != "text":
        raise SystemExit("the scale is the text lenses': a pole's read direction is one for every modality")
    args.targets = tuple(args.targets.split(",")) if args.targets else TARGETS.get(args.lens, ("in", "mid"))
    args.corpus = args.corpus or {"text": "dclm.jsonl", "image": "coco", "audio": "esc50"}[args.modality]
    # Held-out tests (checks/rule_diag.py, on clips without silent frames) favour RelP for every
    # modality: on audio, true derivatives read worse (0.64 against 0.81 at L0-L11) and their
    # per-document estimates are heavy-tailed (split-half near 2 by L11 at 16 documents).
    args.rules = args.rules or "relp"
    if args.rules != "relp" and args.modality == "text":
        raise SystemExit("text fits use the RelP rules")
    args.shard, args.n_shards = (int(x) for x in args.shard.split("/"))
    if args.detach:
        argv = [a for a in sys.argv[1:] if a != "--detach"]
        LOGS.mkdir(exist_ok=True)
        log = LOGS / f"fit_{args.lens.lower()}{'' if args.n_shards == 1 else f'_s{args.shard}'}.log"
        cmd = [sys.executable, __file__, *argv]
        if sys.platform == "darwin":
            cmd = ["caffeinate", "-i", *cmd]
        with open(log, "a") as out:  # the child keeps its own descriptor
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                                 start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        print(f"detached: pid {p.pid}, log {log}")
        return
    if args.modality == "text":
        tok = AutoTokenizer.from_pretrained(args.model)
    else:
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained(args.model)
    model, device, dtype = load_model(args)
    args.n_layers = len(model.layers)
    args.model_layers = AutoConfig.from_pretrained(args.model).get_text_config().num_hidden_layers
    fit(args, model, tok, device, dtype)


if __name__ == "__main__":
    main()
