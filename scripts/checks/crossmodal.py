"""Do Gemma 4 12B neurons hold one concept across modalities? A neuron's reliable image or audio
label (13_media_activations.py: its top label by mean activation, above the shuffled-label null
and replicating in both halves; synthetic probes and source-wide labels left out) against its
text side, three ways, each as the rank of that label among all the modality's labels:

1. **text firing** (causal): the neuron's activation over a label's tokens when the model reads
   ``"Here is an example: {label}."``, one prompt per label;
2. **S-lens text read** (weights): ``u = c · K_L`` (the text S-lens, ``mid`` target) scored on
   each label's tokens (the mean of their centred embeddings);
3. **R-lens write** (weights): ``W_U diag(γ) J_L w`` (the text R-lens; the identity at the last
   layer) scored on the label's first token.

Rank 1 = the neuron's own label comes first among all labels; 0.5 = chance. The null scores a
random other label of the same modality for each neuron. Reported by layer band, with a list of
neurons passing all three (rank ≥ 0.95): one concept across image or audio and text. Then the
neuron poles that pass in both modalities, their image and audio labels side by side (one
concept in text, images and sound, or two that share a neuron), against the overlap the two
lists would have by chance. Every scored pole into ``data/catalog/gemma-4-12b/media/crossmodal_poles.pt``.

    python scripts/checks/crossmodal.py --device cuda
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_lens.paths import DATA, FITS
from neuron_lens.transport import Accumulator, decoder_stack

MODEL = "google/gemma-4-12B"
MEDIA = DATA / "catalog" / "gemma-4-12b" / "media"
GENERIC = {"photo scene", "speech", "music", "noise", "text", "texture", "food", "car", "aircraft", "flower", "painting",
           "doodle", "satellite image", "chest X-ray", "handwritten digit", "app screen", "captioned photo", "captioned sound",
           "document", "chart", "anime", "anime face", "cartoon", "pixel-art sprite", "logo", "emoji", "single-panel cartoon",
           "street scene", "dashcam", "read speech", "expressive speech", "parliament speech", "English speech",
           "Western instrument", "Chinese instrument", "male voice", "female voice", "other voice"}


def clean(label):
    src, _, name = label.partition(":")
    return src, name.replace("_", " ").strip()


def rank_of(scores, own):
    """Per column: the fraction of other rows scoring below row ``own[col]``. ``scores`` [labels, k]."""
    s_own = scores.gather(0, own[None]).squeeze(0)
    return ((scores < s_own).float().sum(0)) / (scores.shape[0] - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--s-fit", default="s-gemma-4-12b-L48-dclm")
    ap.add_argument("--r-fit", default="r-gemma-4-12b-L48-dclm")
    ap.add_argument("--min-z", type=float, default=0.0)
    ap.add_argument("--media", default=str(MEDIA), help="directory of the exemplar files")
    args = ap.parse_args()
    dev, t0 = args.device, time.perf_counter()
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    model = decoder_stack(lm)
    tok = AutoTokenizer.from_pretrained(MODEL)
    L, f = len(model.layers), model.layers[0].mlp.down_proj.weight.shape[1]
    E = model.embed_tokens.weight.float()
    Ec = E - E.mean(0)
    report, per_pole = {}, {}
    for modality in ("image", "audio"):
        ex = torch.load(Path(args.media) / f"{modality}_exemplars.pt", weights_only=False)
        names = ex["labels"]
        keep = [j for j, x in enumerate(names) if not x.startswith("synth:") and clean(x)[1] not in GENERIC]
        texts = [clean(names[j])[1] for j in keep]
        pos_of = {j: i for i, j in enumerate(keep)}
        print(f"{modality}: {len(keep)} concept labels of {len(names)} [{time.perf_counter() - t0:.0f}s]", flush=True)
        # Label token ids (with a leading space, as in running text).
        tids = [tok(" " + t, add_special_tokens=False)["input_ids"] for t in texts]
        # 1. Text firing: one prompt per label; mean activation over the label's tokens.
        fire = torch.zeros(len(texts), L * f, dtype=torch.float16)
        buf = {}
        hooks = [model.layers[i].mlp.down_proj.register_forward_pre_hook(
            lambda _m, a, i=i: buf.__setitem__(i, a[0])) for i in range(L)]
        for b0 in range(0, len(texts), 32):
            batch = texts[b0:b0 + 32]
            pre = [tok("Here is an example:", add_special_tokens=True)["input_ids"] for _ in batch]
            seqs = [p + tid + tok(".", add_special_tokens=False)["input_ids"] for p, tid in zip(pre, tids[b0:b0 + 32])]
            T = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), T), tok.pad_token_id)
            am = torch.zeros(len(seqs), T, dtype=torch.long)
            lab = torch.zeros(len(seqs), T)
            for r, (p, s) in enumerate(zip(pre, seqs)):
                ids[r, :len(s)] = torch.tensor(s)
                am[r, :len(s)] = 1
                lab[r, len(p):len(p) + len(tids[b0 + r])] = 1
            with torch.no_grad():
                model(input_ids=ids.to(dev), attention_mask=am.to(dev))
            m = (lab / lab.sum(1, keepdim=True)).to(dev)
            fire[b0:b0 + len(seqs)] = torch.cat([(buf[i].float() * m[..., None]).sum(1) for i in range(L)], 1).half().cpu()
        for h in hooks:
            h.remove()
        # Neurons with a reliable concept label.
        res = {"fire": [], "s_read": [], "r_write": [], "null_fire": [], "null_s": [], "null_r": [], "who": []}
        rng = np.random.default_rng(0)
        lab_emb = torch.stack([Ec[t].mean(0) for t in tids])  # [labels, d] centred embeddings
        first = torch.tensor([t[0] for t in tids], device=dev)
        if (FITS / f"{args.s_fit}.pt").exists():
            s_acc = Accumulator.from_state(torch.load(FITS / f"{args.s_fit}.pt", map_location="cpu", weights_only=False, mmap=True)["state"])
            r_acc = Accumulator.from_state(torch.load(FITS / f"{args.r_fit}.pt", map_location="cpu", weights_only=False, mmap=True)["state"])
            K_of = lambda l: (s_acc.sums[("mid", 0)][l] + s_acc.sums[("mid", 1)][l]).float() / sum(s_acc.n_docs)
            J_of = lambda l: (r_acc.sums[("out", 0)][l] + r_acc.sums[("out", 1)][l]).float() / sum(r_acc.n_docs)
        else:  # the released lenses (12_release.py): fp32 means
            from huggingface_hub import hf_hub_download
            Ks = torch.load(hf_hub_download("stdlogits/rs-lenses", "gemma-4-12b/s-lens/mid.pt"), weights_only=True, mmap=True)["K"]
            Js = torch.load(hf_hub_download("stdlogits/rs-lenses", "gemma-4-12b/r-lens/lens.pt"), weights_only=True, mmap=True)["J"]
            K_of, J_of = (lambda l: Ks[l].float()), (lambda l: Js[l].float())
        Uf = model.embed_tokens.weight.float()[first] * model.norm.weight.float()  # tied W_U rows of the label tokens, final γ
        for p in (1, -1):
            e = ex["mean"][p]
            for Lyr in range(L):
                top = e["labels"][Lyr, :, 0].long()
                good = e["sig"][Lyr] & e["stable"][Lyr] & (e["z"][Lyr, :, 0].float() * p > args.min_z)
                good &= torch.tensor([int(t) in pos_of for t in top])
                ns = good.nonzero().squeeze(1)
                if not len(ns):
                    continue
                own = torch.tensor([pos_of[int(top[n])] for n in ns], device=dev)
                null = torch.tensor([int(rng.integers(len(texts))) for _ in ns], device=dev)
                cols = (Lyr * f + ns).tolist()
                F_ = p * fire[:, cols].to(dev).float()  # [labels, k]
                layer = model.layers[Lyr]
                g = layer.pre_feedforward_layernorm.weight.float()
                a = layer.mlp.gate_proj.weight[ns].float() * g
                b = layer.mlp.up_proj.weight[ns].float() * g
                c = a / a.norm(dim=1, keepdim=True) + p * b / b.norm(dim=1, keepdim=True)  # [k, d]
                K = K_of(Lyr).to(dev)
                S_ = lab_emb.to(dev) @ (c @ K).T  # [labels, k]
                w = p * float(layer.layer_scalar) * (layer.post_feedforward_layernorm.weight.float()[:, None]
                                                     * layer.mlp.down_proj.weight[:, ns].float())  # [d, k]
                if Lyr < L - 1:
                    J = J_of(Lyr).to(dev)
                    w = J @ w
                R_ = Uf @ w  # [labels, k]
                for key, sc in (("fire", F_), ("s_read", S_), ("r_write", R_)):
                    res[key].append(rank_of(sc, own).cpu())
                    res["null_" + {"fire": "fire", "s_read": "s", "r_write": "r"}[key]].append(rank_of(sc, null).cpu())
                res["who"] += [(Lyr, int(n), p, texts[int(o)]) for n, o in zip(ns, own)]
            print(f"  pole {'+' if p > 0 else '-'} done [{time.perf_counter() - t0:.0f}s]", flush=True)
        cat = {k: torch.cat(v) for k, v in res.items() if k != "who" and v}
        layers = torch.tensor([w[0] for w in res["who"]])
        rep = {"neurons": len(res["who"])}
        print(f"{modality}: {len(res['who'])} neuron poles with a reliable concept label", flush=True)
        for key, nk in (("fire", "null_fire"), ("s_read", "null_s"), ("r_write", "null_r")):
            bands = {}
            for a_, b_ in ((0, 11), (12, 23), (24, 35), (36, 47)):
                sel = (layers >= a_) & (layers <= b_)
                if sel.any():
                    bands[f"L{a_}-{b_}"] = (round(float(cat[key][sel].median()), 3), round(float((cat[key][sel] >= 0.95).float().mean()), 3),
                                            int(sel.sum()))
            rep[key] = {"median": float(cat[key].median()), "share_ge_0.95": float((cat[key] >= 0.95).float().mean()),
                        "null_median": float(cat[nk].median()), "null_share_ge_0.95": float((cat[nk] >= 0.95).float().mean()),
                        "bands": bands}
            print(f"  {key:8s}: own label's rank median {cat[key].median():.2f}, ≥0.95 for {(cat[key] >= 0.95).float().mean():.0%} "
                  f"(null: median {cat[nk].median():.2f}, {(cat[nk] >= 0.95).float().mean():.0%}); by band "
                  + " ".join(f"{k} {v[0]:.2f}/{v[1]:.0%} (n={v[2]})" for k, v in bands.items()), flush=True)
        allthree = ((cat["fire"] >= 0.95) & (cat["s_read"] >= 0.95) & (cat["r_write"] >= 0.95)).nonzero().squeeze(1)
        rep["all_three"] = len(allthree)
        rep["examples"] = [res["who"][i] for i in allthree[:60].tolist()]
        print(f"  all three ≥0.95: {len(allthree)} neuron poles; e.g. "
              + "; ".join(f"L{w[0]} n{w[1]}{'+' if w[2] > 0 else '-'} {w[3]}" for w in rep["examples"][:12]), flush=True)
        report[modality] = rep
        per_pole[modality] = {w[:3]: {"label": w[3], **{k: float(cat[k][i]) for k in ("fire", "s_read", "r_write")}}
                              for i, w in enumerate(res["who"])}
    # Across all three: poles scored, and passing, in both modalities.
    if len(per_pole) == 2:
        im, au = per_pole["image"], per_pole["audio"]
        ok = lambda r, keys: all(r[k] >= 0.95 for k in keys)  # noqa: E731
        both = sorted(set(im) & set(au))
        tri = {}
        for name, keys in (("all_three", ("fire", "s_read", "r_write")), ("fire", ("fire",))):
            hit = [k for k in both if ok(im[k], keys) and ok(au[k], keys)]
            p_im = sum(ok(im[k], keys) for k in both) / max(1, len(both))
            p_au = sum(ok(au[k], keys) for k in both) / max(1, len(both))
            tri[name] = {"scored_in_both": len(both), "passing_both": len(hit), "expected_if_independent": p_im * p_au * len(both),
                         "poles": [(L_, n_, s_, im[(L_, n_, s_)]["label"], au[(L_, n_, s_)]["label"]) for L_, n_, s_ in hit]}
            print(f"text + image + audio ({name.replace('_', ' ')}): {len(hit)} of the {len(both)} poles scored in both pass "
                  f"in both (independent lists would share {p_im * p_au * len(both):.1f}); e.g. "
                  + "; ".join(f"L{a} n{b}{'+' if c > 0 else '-'} image {d!r} / audio {e!r}"
                              for a, b, c, d, e in tri[name]["poles"][:15]), flush=True)
        report["text_image_audio"] = {k: v | {"poles": v["poles"][:200]} for k, v in tri.items()}
    torch.save(per_pole, MEDIA / "crossmodal_poles.pt")
    out = DATA / "summary" / "gemma4_media" / "crossmodal.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"wrote {out} [{time.perf_counter() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
