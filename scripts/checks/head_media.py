"""Gemma 4 12B's attention heads in image and audio contexts: which heads read the media, how they
attend within it, which carry what it shows into the text, and which write its name.

Documents: ``N`` labelled images (mini-ImageNet, one per class as far as ``N`` allows, 384 × 384,
64 slots) and ``N`` labelled sounds (ESC-50, 2.56 s, 64 slots, the classes in turn), each in ``media.PROMPT`` ("...It
shows" / "...It sounds like"), followed by `` a <label>`` (``an`` before a vowel): its own label
(right) and another document's label of a different class (wrong). Per head, with eager attention:

- **reading the media:** the mean attention from the prompt's tokens after the media block to its
  slots (``read``), against the share the slots would get spread evenly over the keys
  (``read_ratio``); to position 0 (``sink``);
- **within the media:** from each slot, to itself (``self``), to the other slots (``within``), and
  to its grid neighbours (images: patches one step apart in ``image_position_ids``; sounds: the
  frame before, ``nbr``). Image slots attend to each other both ways in the sliding layers;
- **carrying it into the text** (the causal test): each head zeroed in turn (its slice of
  ``o_proj``'s input), the change in the label's loss (its tokens after the article, nats): right
  (``abl_right``), wrong (``abl_wrong``), and ``content = abl_right − abl_wrong``, what the head
  contributes to telling this document's label from another's, with its z over documents
  (``content_z``). The model's own right-over-wrong accuracy is in the summary;
- **writing the label:** each head's output at the article (the position predicting the label),
  as it reaches the residual (the post-attention norm's weight and scale), read through the
  released R-lens's ``mid`` target at its layer (``write_mid``) and the logit lens
  (``write_logit``): the mean z over the vocabulary of each class's label tokens, centred per class
  over the documents (labels some heads promote whatever they read drop out), and the right
  class's percentile among the classes, averaged over documents (0.5 by chance); ``top1``, the
  share of documents whose own class comes first. Its p-value is the
  share of ``PERMS`` label shuffles that score as high; legible at a 5% false discovery rate
  (Benjamini-Hochberg over all heads).

Per head into ``data/catalog/gemma-4-12b/heads/media.parquet`` (columns ``<modality>_<score>``);
the summary in ``data/summary/head_media_gemma-4-12b.{log,json}``, beside the text pattern classes
of ``head_patterns.py`` when present.

    python scripts/checks/head_media.py [--modality image,audio] [--n 64]
"""
import argparse
import json
import time

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoProcessor

from neuron_lens import arch
from neuron_lens.stats import bh, spearman
from neuron_lens.catalog import catalog_dir
from neuron_lens.media import labelled_images, processed, slot_token, sounds
from neuron_lens.paths import SUMMARY
from neuron_lens.readout import rms_norm_columns, transport

MODEL = "google/gemma-4-12B"
PERMS = 200
FDR = 0.05
TOP = 6


def documents(modality: str, n: int):
    if modality == "image":
        return labelled_images(n)
    clips, cats = sounds(8 * n, "esc", categories=True)  # ESC-50's rows run in runs of a class: take them in turn
    by = {}
    for c, k in zip(clips, cats):
        by.setdefault(k, []).append(c)
    order = [(k, v[i]) for i in range(max(map(len, by.values()))) for k, v in by.items() if i < len(v)][:n]
    return [c for _, c in order], [k.replace("_", " ") for k, _ in order]


def continuation(tok, label: str) -> tuple[list[int], int]:
    """`` a <label>``'s ids and how many of them are the article."""
    art = "an" if label[0].lower() in "aeiou" else "a"
    ids = tok(f" {art} {label}", add_special_tokens=False)["input_ids"]
    a = tok(f" {art}", add_special_tokens=False)["input_ids"]
    assert ids[:len(a)] == a, (label, ids, a)
    return ids, len(a)


def batch_inputs(base: list[dict], conts: list[list[int]], pad: int, dev: str) -> dict:
    """Each document's processed inputs (all ``T0`` long) with its continuation appended, padded
    at the end."""
    T0 = base[0]["input_ids"].shape[1]
    T = T0 + max(len(c) for c in conts)
    out = {}
    for key in base[0]:
        x = torch.cat([b[key] for b in base])
        if x.dim() == 2 and x.shape[1] == T0 and not x.is_floating_point():  # per-token: extend
            ext = torch.zeros(len(base), T - T0, dtype=x.dtype)
            if key == "input_ids":
                ext.fill_(pad)
            elif key == "attention_mask":
                for i, c in enumerate(conts):
                    ext[i, :len(c)] = 1
            x = torch.cat([x, ext], 1)
            if key == "input_ids":
                for i, c in enumerate(conts):
                    x[i, T0:T0 + len(c)] = torch.tensor(c)
        out[key] = x.to(dev)
        if out[key].is_floating_point():
            out[key] = out[key].to(torch.bfloat16)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="image,audio")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--layers", type=int, help="ablate only the first N layers (a smoke test)")
    args = ap.parse_args()
    dev = "cuda"
    lm, stack = arch.load_model(MODEL, dev)
    proc = AutoProcessor.from_pretrained(MODEL)
    tok = proc.tokenizer
    geo = arch.layout(MODEL)
    L, H = len(geo), geo[0].q_heads
    d = stack.config.hidden_size
    nm = arch.names(MODEL)
    from neuron_lens.weights import tensors
    w = tensors(MODEL, sorted(set(nm.values())))
    unembed = w[nm["unembed"]].to(dev, torch.float16)
    gamma = w[nm["norm"]].to(dev, torch.float32)
    del w
    J_mid = arch.lens_means(MODEL, parts=("J_mid",))["J_mid"]
    wo = [(b.post_attention_layernorm.weight.float()[:, None] * b.self_attn.o_proj.weight.float()) for b in stack.layers]
    cols = {"layer": torch.arange(L).repeat_interleave(H), "head": torch.arange(H).repeat(L)}
    report = {}
    for modality in args.modality.split(","):
        t0 = time.perf_counter()
        items, labels = documents(modality, args.n)
        n = len(items)
        classes = sorted(set(labels))
        cls_of = torch.tensor([classes.index(x) for x in labels])
        wrong = []
        for i in range(n):  # the next document of another class
            j = next(k for k in range(1, n + 1) if labels[(i + k) % n] != labels[i])
            wrong.append(labels[(i + j) % n])
        cont = {lab: continuation(tok, lab) for lab in set(labels) | set(wrong)}
        cls_ids = [torch.tensor(cont[c][0][cont[c][1]:]) for c in classes]  # each class's label tokens
        base = [processed(proc, modality, x) for x in items]
        T0 = base[0]["input_ids"].shape[1]
        ids0 = base[0]["input_ids"][0]
        slots = (ids0 == slot_token(proc, modality)).nonzero().squeeze(1)
        assert all(torch.equal(b["input_ids"][0], ids0) for b in base), "documents differ outside the media"
        after = torch.arange(int(slots[-1]) + 1, T0)  # the prompt's tokens after the media block
        S = len(slots)
        if modality == "image":
            p = base[0]["image_position_ids"][0]
            p = p[(p >= 0).all(-1)][:S].float()  # the real patches' (x, y), in slot order
            assert len(p) == S
            nbr = ((p[:, None] - p[None]).abs().sum(-1) == 1).float()  # [S, S]: patches one step apart
        else:
            nbr = torch.diag(torch.ones(S - 1), -1)  # the frame before
        nbr = nbr.to(dev)
        seqs = [cont[x] for x in labels] + [cont[x] for x in wrong]  # right, then wrong
        docs = base + base
        readout = torch.tensor([T0 + a - 1 for _, a in seqs[:n]])  # the article's last token, before the label

        stash, zeroed, ctx = {}, {}, {}

        def pre(l):
            hd = geo[l].head_dim

            def hook(m, x):
                if zeroed.get("layer") == l:
                    h = zeroed["head"]
                    x[0][..., h * hd:(h + 1) * hd] = 0
                if "rows" in ctx:
                    stash[l] = x[0][ctx["rows"], ctx["pos"]].float()  # [B, H·hd] at the readout
            return hook

        def post(l):
            def hook(m, a, out):
                if "rows" in ctx:
                    pa_ = stack.layers[l].post_attention_layernorm
                    rho = torch.rsqrt(out[0][ctx["rows"], ctx["pos"]].float().pow(2).mean(-1) + arch.norm_eps(pa_))
                    z = stash.pop(l).view(-1, H, geo[l].head_dim) * rho[:, None, None]
                    ctx["out"][l].append(torch.einsum("bhi,dhi->bhd", z, wo[l].view(d, H, -1)).half().cpu())
            return hook
        hooks = [b.self_attn.o_proj.register_forward_pre_hook(pre(l)) for l, b in enumerate(stack.layers)]
        hooks += [b.self_attn.register_forward_hook(post(l)) for l, b in enumerate(stack.layers)]

        def nll(attn: bool = False) -> torch.Tensor:
            """Each sequence's mean loss over its label tokens; with ``attn``, the right
            sequences' attention statistics and readout outputs too."""
            out = []
            for s in range(0, len(seqs), args.batch):
                part = seqs[s:s + args.batch]
                inp = batch_inputs(docs[s:s + args.batch], [c for c, _ in part], tok.pad_token_id, dev)
                rows = [i for i in range(len(part)) if s + i < n]
                if attn and rows:
                    ctx.update(rows=torch.tensor(rows, device=dev), pos=readout[s:s + len(rows)].to(dev))
                with torch.no_grad():
                    o = lm.model(**inp, use_cache=False, output_attentions=attn and bool(rows))
                ctx.pop("rows", None)
                hid = o.last_hidden_state
                for i, (c, a) in enumerate(part):
                    pos = torch.arange(T0 + a - 1, T0 + len(c) - 1, device=dev)
                    lp = torch.log_softmax(arch.logits(lm, hid[i, pos]), -1)
                    out.append(-lp.gather(1, torch.tensor(c[a:], device=dev)[:, None]).mean().item())
                if attn and rows:
                    for l, att in enumerate(o.attentions):  # [B, H, T, T]
                        a_ = att[rows].float()
                        q = a_[:, :, after]  # [B, H, |after|, T]
                        st["read"][l] += q[..., slots].sum(-1).mean(-1).sum(0).cpu()
                        st["sink"][l] += q[..., 0].mean(-1).sum(0).cpu()
                        m = a_[:, :, slots][..., slots]  # [B, H, S, S]
                        st["self"][l] += m.diagonal(dim1=-2, dim2=-1).mean(-1).sum(0).cpu()
                        st["within"][l] += (m.sum(-1) - m.diagonal(dim1=-2, dim2=-1)).mean(-1).sum(0).cpu()
                        st["nbr"][l] += (m * nbr).sum(-1).mean(-1).sum(0).cpu()
                del o
            return torch.tensor(out)

        st = {k: torch.zeros(L, H) for k in ("read", "sink", "self", "within", "nbr")}
        ctx["out"] = {l: [] for l in range(L)}
        base_nll = nll(attn=True)
        st = {k: v / n for k, v in st.items()}
        st["read_ratio"] = st["read"] / (S / (after.float() + 1)).mean().item()  # keys at query q: q + 1
        outs = {l: torch.cat(v) for l, v in ctx.pop("out").items()}  # [n, H, d] per layer
        acc = (base_nll[:n] < base_nll[n:]).float().mean().item()
        print(f"{modality}: {n} documents, {len(classes)} classes; label loss right {base_nll[:n].mean():.2f}, wrong "
              f"{base_nll[n:].mean():.2f} nats; right below wrong in {acc:.0%}; {time.perf_counter() - t0:.0f}s", flush=True)

        # Writing the label: each head's readout output through the R-lens mid and the logit lens.
        write, pval, zcls_all = {}, {}, {}
        g = torch.Generator().manual_seed(0)
        perms = torch.stack([torch.randperm(n, generator=g) for _ in range(PERMS)])
        for lens_name in ("mid", "logit"):
            sc = torch.zeros(L, H, n, len(classes))
            for l in range(L):
                O = outs[l].reshape(n * H, d).to(dev, torch.float32)
                t = rms_norm_columns(transport(O.T, J_mid[l] if lens_name == "mid" else None), gamma)
                lg = (unembed @ t.half()).float()  # [V, n·H]
                z = (lg - lg.mean(0)) / lg.std(0).clamp_min(1e-6)
                sc[l] = torch.stack([z[c.to(dev)].mean(0) for c in cls_ids], -1).view(n, H, -1).transpose(0, 1).cpu()
            sc = sc - sc.mean(2, keepdim=True)  # each class centred over the documents: no label is written everywhere
            pct = sc.argsort(-1).argsort(-1).float() / max(1, len(classes) - 1)  # each class's percentile per document
            own = pct.gather(-1, cls_of[None, None, :, None].expand(L, H, n, 1)).squeeze(-1).mean(-1)  # [L, H]
            null = torch.stack([pct.gather(-1, cls_of[p][None, None, :, None].expand(L, H, n, 1)).squeeze(-1).mean(-1)
                                for p in perms])  # [PERMS, L, H]
            write[lens_name] = own
            st[f"top1_{lens_name}"] = (sc.argmax(-1) == cls_of).float().mean(-1)
            pval[lens_name] = (1 + (null >= own).sum(0)) / (1 + PERMS)
            zcls_all[lens_name] = sc
        print(f"{modality}: write reads done, {time.perf_counter() - t0:.0f}s", flush=True)

        # Carrying it into the text: each head zeroed.
        abl = torch.full((L, H, 2 * n), torch.nan)
        for l in range(args.layers or L):
            for h in range(H):
                zeroed.update(layer=l, head=h)
                abl[l, h] = nll() - base_nll
            zeroed.clear()
            print(f"  {modality} L{l}: top content head h{int((abl[l, :, :n] - abl[l, :, n:]).mean(-1).argmax())}, "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
        for hk in hooks:
            hk.remove()
        diff = abl[..., :n] - abl[..., n:]
        st["abl_right"], st["abl_wrong"] = abl[..., :n].mean(-1), abl[..., n:].mean(-1)
        st["content"] = diff.mean(-1)
        st["content_z"] = diff.mean(-1) / (diff.std(-1) / n ** 0.5).clamp_min(1e-9)
        for k in ("mid", "logit"):
            st[f"write_{k}"] = write[k]
            st[f"write_{k}_q"] = bh(pval[k].flatten()).view(L, H)
        for k, v in st.items():
            cols[f"{modality}_{k}"] = v.flatten()
        report[modality] = {"n": n, "classes": len(classes), "accuracy": acc, "nll_right": base_nll[:n].mean().item(),
                            "nll_wrong": base_nll[n:].mean().item(), "slots": S, "st": st}
        # what the top writers write: per document, its label → the class the head ranks first
        top = zcls_all["mid"].argmax(-1)  # [L, H, n]
        report[modality]["top_classes"] = {(l, h): [f"{labels[i]} → {classes[int(top[l, h, i])]}" for i in range(min(TOP, n))]
                                           for l in range(L) for h in range(H)}
        torch.cuda.empty_cache()

    out_dir = catalog_dir(MODEL) / "heads"
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({k: pa.array(v.numpy()) for k, v in cols.items()}), out_dir / "media.parquet")

    pat_path = out_dir / "patterns.parquet"
    pat = pq.read_table(pat_path).to_pydict() if pat_path.exists() else None
    bands = arch.bands(L)
    hl = lambda i: f"L{i // H}h{i % H}"  # noqa: E731
    lines = [f"attention heads in image and audio contexts, {MODEL}: {L} × {H} heads; per modality N labelled documents in "
             "media.PROMPT, each followed by ' a <label>' (its own, and another class's)"]
    table = {}
    for modality, r in report.items():
        st = r["st"]
        lines += ["", f"== {modality}: {r['n']} documents, {r['classes']} classes, {r['slots']} slots; label loss right "
                  f"{r['nll_right']:.2f} / wrong {r['nll_wrong']:.2f} nats, right below wrong in {r['accuracy']:.0%}"]
        lines.append("band     read (×even)  sink   self   within  nbr    content>0 (z>3)  write_mid q≤.05  write_logit q≤.05")
        t = {}
        for lo, hi in bands:
            b = slice(lo, hi + 1)
            row = {"read": st["read"][b].mean().item(), "read_ratio": st["read_ratio"][b].mean().item(),
                   "sink": st["sink"][b].mean().item(), "self": st["self"][b].mean().item(),
                   "within": st["within"][b].mean().item(), "nbr": st["nbr"][b].mean().item(),
                   "content_heads": (st["content_z"][b] > 3).float().mean().item(),
                   "write_mid_heads": (st["write_mid_q"][b] <= FDR).float().mean().item(),
                   "write_logit_heads": (st["write_logit_q"][b] <= FDR).float().mean().item()}
            t[f"L{lo}-{hi}"] = row
            lines.append(f"L{lo:>2d}-{hi:<2d}  {row['read']:.3f} ({row['read_ratio']:.1f})  {row['sink']:.3f}  {row['self']:.3f}  "
                         f"{row['within']:.3f}   {row['nbr']:.3f}  {row['content_heads']:.2f}             "
                         f"{row['write_mid_heads']:.2f}             {row['write_logit_heads']:.2f}")
        table[modality] = {k: v for k, v in r.items() if k not in ("st", "top_classes")} | {"bands": t}
        for key, label in (("read_ratio", "media readers (attention from the text after the media, × an even spread)"),
                           ("nbr", "local heads (attention from a slot to its neighbours)"),
                           ("content", "content carriers (ablation raises the right label's loss more than a wrong one's, nats)"),
                           ("write_mid", "label writers (right class's mean percentile among classes, R-lens mid)")):
            v = st[key].flatten().nan_to_num(-1e9)
            top = v.topk(8).indices.tolist()
            lines.append(f"top {label}: " + ", ".join(
                f"{hl(i)} {st[key].flatten()[i]:.3f}" + (f" (z {st['content_z'].flatten()[i]:.1f})" if key == "content" else "")
                + (f" (q {st['write_mid_q'].flatten()[i]:.3f})" if key == "write_mid" else "") for i in top))
            table[modality][f"top_{key}"] = [[hl(i), st[key].flatten()[i].item()] for i in top]
        lines.append("  the top label writers, per document: its label → the class the head writes most (centred):")
        for i in st["write_mid"].flatten().topk(5).indices.tolist():
            lines.append(f"    {hl(i)} (top-1 {st['top1_mid'].flatten()[i]:.0%}): {' | '.join(r['top_classes'][(i // H, i % H)])}")
        if pat:
            ok = ~st["content"].flatten().isnan()
            pt = {k: torch.tensor(pat[k]) for k in ("sink", "prev", "induction", "duplicate", "ablate_copy")}
            lines.append("  Spearman over heads with the text patterns: " + ", ".join(
                f"{a} vs {k} {spearman(st[a].flatten()[ok], v[ok]):+.2f}" for a in ("read_ratio", "content") for k, v in pt.items()))
    if len(report) == 2:
        a, b = (report[m]["st"] for m in report)
        lines += ["", "image against audio, Spearman over heads: " + ", ".join(
            f"{k} {spearman(a[k].flatten().nan_to_num(0), b[k].flatten().nan_to_num(0)):+.2f}"
            for k in ("read_ratio", "sink", "self", "content", "write_mid"))]
        both = [i for i in range(L * H) if a["content_z"].flatten()[i] > 3 and b["content_z"].flatten()[i] > 3]
        lines.append(f"heads carrying content in both (z > 3 in each): {len(both)}: " + ", ".join(hl(i) for i in both[:20]))
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"head_media_{arch.stem(MODEL)}.log").write_text(text + "\n")
    (SUMMARY / f"head_media_{arch.stem(MODEL)}.json").write_text(json.dumps(table))


if __name__ == "__main__":
    main()
