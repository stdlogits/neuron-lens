"""Media exemplars for every neuron of Gemma 4 12B: which real and synthetic images and sounds
drive it, with a null and a held-out test.

Documents are one input each in its modality's prompt (``neuron_lens.media``), from the diverse
collections of ``neuron_lens.media_sets`` (labels ``source:label``; multi-label sources count an
item toward each of its labels; synthetic probes ``synth:``). Per document and neuron, three
statistics over the input's 64 tokens: the mean activation (the down projection's input), and
the max and min (read by the + and − poles; a neuron for one small object in a scene is not
averaged away).

Per label (``--min-docs`` documents or more) and neuron, the selectivity
``z = (mean over the label's documents − mean over all) / std over all``. Kept per neuron and
pole (+: highest z, −: lowest), for each statistic:
- the top ``--top-labels`` labels, with z and the label's mean;
- **null**: the same z with the labels shuffled across documents (one permutation): per layer,
  the 99th percentile of a neuron's top null z is the threshold a real top z must pass (``sig``);
- **split-half**: whether its top label is in the top 5 of the even documents' z and of the odd's;
- **held out**: the top label chosen on the even documents, its AUROC on the odd documents
  (the label's documents against the rest; mean statistic, both poles).
And each pole's top ``--top-docs`` documents (mean statistic). Written to
``data/catalog/<model>/media/<modality>_exemplars.pt`` (with every document's labels, source,
caption and ``doc_ref``: its dataset, revision and row identifiers, to link it rather than host
it), and the label means beside it.

    python scripts/13_media_activations.py --device cuda [--scale 1.0] [--modalities audio,image]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from neuron_lens.media import processed, slot_token
from neuron_lens.media_sets import audio_collection, image_collection
from neuron_lens.paths import DATA
from neuron_lens.transport import decoder_stack

MODEL = "google/gemma-4-12B"


def auroc_rows(acts, pos):
    """Per column: AUROC of ``acts`` ([docs, k]) for the rows ``pos`` marks ([docs, k] bool) against the rest."""
    ranks = acts.argsort(0).argsort(0).double() + 1
    n1 = pos.sum(0).double()
    n0 = pos.shape[0] - n1
    return ((ranks * pos).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0).clamp_min(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies every source's item count")
    ap.add_argument("--top-labels", type=int, default=32)
    ap.add_argument("--top-docs", type=int, default=16)
    ap.add_argument("--min-docs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--modalities", default="audio,image")
    ap.add_argument("--out", help="output folder (default data/catalog/<model>/media)")
    ap.add_argument("--wait-free-gb", type=float, default=0, help="before the forward passes, wait until the GPU has this much free")
    ap.add_argument("--chunk", type=int, default=8192, help="neurons per GPU chunk in the statistics")
    args = ap.parse_args()
    dev = args.device
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    model = decoder_stack(lm)
    proc = AutoProcessor.from_pretrained(MODEL)
    L, f = len(model.layers), model.layers[0].mlp.down_proj.weight.shape[1]
    F = L * f
    out_dir = Path(args.out) if args.out else DATA / "catalog" / MODEL.split("/")[-1].lower() / "media"
    out_dir.mkdir(parents=True, exist_ok=True)
    buf, mask = {}, {}

    def hook(i):
        def f_(_m, a):
            m = mask["m"]
            x = a[0].float()
            buf[i] = ((x * m[..., None]).sum(1) / m.sum(1, keepdim=True),
                      x.masked_fill(m[..., None] == 0, float("-inf")).amax(1),
                      x.masked_fill(m[..., None] == 0, float("inf")).amin(1))
        return model.layers[i].mlp.down_proj.register_forward_pre_hook(f_)
    hooks = [hook(i) for i in range(L)]
    for modality in args.modalities.split(","):
        t1 = time.perf_counter()
        docs_all = (image_collection if modality == "image" else audio_collection)(args.scale)
        items = [d[0] for d in docs_all]
        labels = [d[1] for d in docs_all]
        n = len(items)
        names = sorted({x for ls in labels for x in ls})
        idx = {x: j for j, x in enumerate(names)}
        member = torch.zeros(n, len(names), dtype=torch.bool)
        for d, ls in enumerate(labels):
            member[d, [idx[x] for x in ls]] = True
        perm = torch.from_numpy(np.random.default_rng(1).permutation(n))
        member_null = member[perm]  # document d gets document perm[d]'s labels
        print(f"{modality}: {n} documents, {len(names)} labels [{time.perf_counter() - t1:.0f}s load]", flush=True)
        stats = ("mean", "max", "min")
        acts = {s: torch.empty(n, F, dtype=torch.float16) for s in stats}  # every document's activations (RAM)
        sid = slot_token(proc, modality)
        while args.wait_free_gb and torch.cuda.mem_get_info(dev)[0] / 2 ** 30 < args.wait_free_gb:  # a shared GPU
            time.sleep(15)
        for start in range(0, n, args.batch):
            batch = [processed(proc, modality, x) for x in items[start:start + args.batch]]
            inp = {key: torch.cat([d[key] for d in batch]).to(dev) for key in batch[0]}
            inp = {key: v.to(torch.bfloat16) if v.is_floating_point() else v for key, v in inp.items()}
            mask["m"] = (inp["input_ids"] == sid).float()
            with torch.no_grad():
                lm.model(**inp, use_cache=False)
            B = inp["input_ids"].shape[0]
            for j, s in enumerate(stats):
                acts[s][start:start + B] = torch.cat([buf[i][j] for i in range(L)], 1).half().cpu()
            if start // args.batch % 100 == 0:
                print(f"  {start + B}/{n} [{time.perf_counter() - t1:.0f}s]", flush=True)
        ok = member.sum(0) >= args.min_docs
        even = torch.arange(n) % 2 == 0
        out = {"model": MODEL, "modality": modality, "labels": names, "label_docs": member.sum(0).tolist(),
               "doc_labels": labels, "doc_source": [d[3] for d in docs_all], "doc_caption": [d[2] for d in docs_all],
               "doc_ref": [d[4] if len(d) > 4 else None for d in docs_all],
               "min_docs": args.min_docs, "layers": L, "neurons": f}
        chunk = args.chunk  # neurons per GPU chunk
        for s in stats:
            poles = {"mean": (1, -1), "max": (1,), "min": (-1,)}[s]
            res = {p: {"labels": [], "z": [], "mean": [], "stable": [], "sig": [], "null_top": [], "auroc_heldout": []} for p in poles}
            for c0 in range(0, F, chunk):
                A = acts[s][:, c0:c0 + chunk].to(dev).float()  # [n, k]
                mu, sd = A.mean(0), A.std(0).clamp_min(1e-6)
                M = member.to(dev).float()
                z_of = lambda Mm, rows=None: (((Mm if rows is None else Mm[rows]).T @ (A if rows is None else A[rows]))
                                              / (Mm if rows is None else Mm[rows]).sum(0).clamp_min(1)[:, None] - mu) / sd
                z = z_of(M).masked_fill(~ok.to(dev)[:, None], float("nan"))  # [labels, k]
                zn = z_of(member_null.to(dev).float()).masked_fill(~ok.to(dev)[:, None], float("nan"))
                ze = z_of(M, even.to(dev)).masked_fill(~ok.to(dev)[:, None], float("nan"))
                zo = z_of(M, (~even).to(dev)).masked_fill(~ok.to(dev)[:, None], float("nan"))
                lmean = (M.T @ A) / M.sum(0).clamp_min(1)[:, None]
                for p in poles:
                    tz, tl = (p * z).nan_to_num(float("-inf")).topk(args.top_labels, dim=0)
                    e5 = (p * ze).nan_to_num(float("-inf")).topk(5, dim=0).indices
                    o5 = (p * zo).nan_to_num(float("-inf")).topk(5, dim=0).indices
                    r = res[p]
                    r["labels"].append(tl.T.short().cpu())
                    r["z"].append((p * tz).T.half().cpu())
                    r["mean"].append(lmean.gather(0, tl).T.half().cpu())
                    r["stable"].append(((e5 == tl[0]).any(0) & (o5 == tl[0]).any(0)).cpu())
                    r["null_top"].append((p * zn).nan_to_num(float("-inf")).amax(0).cpu())
                    r["sig"].append(tz[0].cpu())  # thresholded below, per layer
                    if s == "mean":  # held out: label chosen on even documents, scored on odd ones
                        top_e = (p * ze).nan_to_num(float("-inf")).argmax(0)  # [k]
                        odd = (~even).to(dev)
                        pos = member.to(dev)[odd][:, top_e]  # [n_odd, k]
                        r["auroc_heldout"].append(auroc_rows(p * A[odd], pos).float().cpu())
            for p, r in res.items():
                cat = {k: torch.cat(v, 0 if v[0].dim() == 1 else 0) for k, v in r.items() if v}
                top_z = cat["sig"].reshape(L, f)
                null_top = cat["null_top"].reshape(L, f)
                thresh = torch.nanquantile(null_top.float(), 0.99, dim=1)  # per layer
                sig = top_z > thresh[:, None]
                entry = {"labels": cat["labels"].reshape(L, f, -1), "z": cat["z"].reshape(L, f, -1),
                         "mean": cat["mean"].reshape(L, f, -1), "stable": cat["stable"].reshape(L, f),
                         "sig": sig, "null_threshold": thresh}
                line = (f"{modality} {s} {'+' if p > 0 else '-'}: top label above the shuffled-label 99th pct for "
                        f"{sig.float().mean():.0%}, replicates in both halves for {entry['stable'].float().mean():.0%}")
                if "auroc_heldout" in cat:
                    au = cat["auroc_heldout"].reshape(L, f)
                    entry["auroc_heldout"] = au.half()
                    line += f", held-out AUROC median {au.nanmedian():.2f}"
                print(line + "; by band " + " ".join(
                    f"L{a}-{b} {sig[a:b + 1].float().mean():.0%}/{entry['stable'][a:b + 1].float().mean():.0%}"
                    for a, b in ((0, 11), (12, 23), (24, 35), (36, 47)) if a < L), flush=True)
                out.setdefault(s, {})[p] = entry
        # Top documents per pole (mean statistic), in chunks.
        tops = {p: ([], []) for p in (1, -1)}
        for c0 in range(0, F, chunk):
            A = acts["mean"][:, c0:c0 + chunk].to(dev).float()
            for p in (1, -1):
                v, ix = (p * A).topk(args.top_docs, dim=0)
                tops[p][0].append((p * v).T.half().cpu())
                tops[p][1].append(ix.T.int().cpu())
        out["top_docs"] = {p: (torch.cat(v).reshape(L, f, -1), torch.cat(ix).reshape(L, f, -1)) for p, (v, ix) in tops.items()}
        torch.save(out, out_dir / f"{modality}_exemplars.pt")
        print(f"{modality}: wrote {out_dir / f'{modality}_exemplars.pt'} [{time.perf_counter() - t1:.0f}s]", flush=True)
        del acts
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
    # Streamed datasets leave pyarrow threads that hang interpreter shutdown (as 05_act_corpus.py).
    sys.stdout.flush()
    os._exit(0)
