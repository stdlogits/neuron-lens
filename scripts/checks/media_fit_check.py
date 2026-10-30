"""Check a media S-lens fit (02_fit.py --modality image|audio) against the pull-down.

For media_reads.py's panel poles in the fit's layers: the read ``u = (c · K_L Bᵀ) B`` from the
fit (``B`` its source basis) against ``input_reads_doc`` on the same documents with the same rows
and rules (the S-lens's transpose, no basis): they are the same quantity, so they must agree
to bf16. Also the fit's split-half by layer, and what it predicts at other document counts.

    python scripts/checks/media_fit_check.py --fit s-gemma-4-12b-L12-coco --modality image --layers 12 --docs 16
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from media_reads import MODEL  # noqa: E402

from neuron_lens.gemma4 import multimodal_embeds  # noqa: E402
from neuron_lens.media import audio_reads, images, pixel_reads, processed, slot_token, sounds  # noqa: E402
from neuron_lens.paths import DATA, FITS  # noqa: E402
from neuron_lens.transport import Accumulator, decoder_stack, input_reads_doc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--modality", required=True)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--docs", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    saved = torch.load(FITS / f"{args.fit}.pt", weights_only=False, map_location="cpu")
    meta, acc = saved["meta"], Accumulator.from_state(saved["state"])
    B = saved["state"]["basis"]
    rules = meta.get("rules", "relp")
    print(f"fit: {meta['lens']} {meta.get('modality')} basis {meta.get('basis')} rules {rules}, n_docs {acc.n_docs}", flush=True)
    config = AutoConfig.from_pretrained(MODEL)
    tc = config.text_config
    tc.num_hidden_layers, tc.layer_types = args.layers, tc.layer_types[:args.layers]
    lm = AutoModelForCausalLM.from_pretrained(MODEL, config=config, dtype=torch.bfloat16).to(args.device).eval().requires_grad_(False)
    model = decoder_stack(lm)
    proc = AutoProcessor.from_pretrained(MODEL)
    panel = torch.load(DATA / "summary" / "gemma4_media" / "reads.pt", weights_only=False)
    sel = [i for i, (L, _, _) in enumerate(panel["poles"]) if L < args.layers]
    poles, cov = [panel["poles"][i] for i in sel], panel["cov"][sel]
    layers = [L for L, _, _ in poles]
    K = acc.mean("mid")  # [layers, d, k]
    u_fit = torch.stack([(cov[i] @ K[L]) @ B for i, L in enumerate(layers)])
    items = images(sum(acc.n_docs)) if args.modality == "image" else sounds(sum(acc.n_docs), "esc")
    sid = slot_token(proc, args.modality)
    u_pull = torch.zeros_like(u_fit)
    for item in items:
        d = {k: v.to(args.device, torch.bfloat16) if v.is_floating_point() else v.to(args.device)
             for k, v in processed(proc, args.modality, item).items()}
        ids = d["input_ids"][0]
        slots = ids == sid
        where = slots.nonzero().squeeze(1)
        first, T = int(where[0]), int(where[-1]) + 2
        emb = multimodal_embeds(lm, d)
        rows = slots[first:T - 1]
        u_pull += input_reads_doc(model, ids[:T], cov, layers, seed_rows=rows, source_rows=rows, embeds=emb[:T],
                                  block=slots[:T] if args.modality == "image" else None, first=first, rules=rules).cpu()
    u_pull /= len(items)
    # The fit knows only the span of B (all the embedder reaches), so compare what the reads are
    # for: the pixel (audio) reads, where the rest of u vanishes; and u within the span.
    to_input = (lambda u: pixel_reads(lm, u).reshape(len(u), -1)) if args.modality == "image" else (lambda u: audio_reads(lm, u))
    for name, a, b in (("pixel/audio read", to_input(u_fit), to_input(u_pull)),
                       ("u within span(B)", u_fit, (u_pull @ B.T) @ B),
                       ("u, whole (not expected to match)", u_fit, u_pull)):
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
        rel = (a - b).norm(dim=1) / b.norm(dim=1)
        print(f"fit vs pull-down on the same {len(items)} documents, {name}: cosine min {cos.min():.5f}, "
              f"median {cos.median():.5f}; rel err median {rel.median():.2e}, max {rel.max():.2e}", flush=True)
    split = acc.split_half("mid")
    n = sum(acc.n_docs)
    print("split-half mid by layer: " + " ".join(f"{x:.2f}" for x in split), flush=True)
    med = sorted(split)[len(split) // 2]
    for N in (128, 256, 512, 1000):
        print(f"  predicted split-half median at {N} documents: {med * (n / N) ** 0.5:.3f}", flush=True)


if __name__ == "__main__":
    main()
