"""In-context copying, and which attention heads it rests on (findings §11).

The heads' attention patterns (``sink``, ``prev``, ``self``, ``induction``, ``duplicate``) are
``06_activations.py``'s, in ``heads/acts``. This check measures what the pipeline does not: how well
the model copies the sequences those induction scores come from, and what removing each head costs.

Sequences: ``06_activations.py``'s repeated-token pass, rebuilt the same way: ``REP_SEQS`` (64)
sequences of ``REP_LEN`` (64) random kept tokens (``catalog.kept`` without special tokens,
``torch.randint`` seeded 0), each said twice (Gemma 4: after ``<bos>``). bf16, eager attention.

- **copying:** the mean next-token loss over each copy's tokens 2-64, each predicted from those
  before it: on the first copy the tokens are random (the loss is about the log of the pool), on
  the second they repeat the context (the second copy's first token is in neither: nothing
  predicts it). In batches of ``BATCH`` as the pipeline's pass, whose ``induction`` and
  ``duplicate`` scores (mean attention from each second-copy token with a successor to the token
  after its first occurrence, and to that occurrence) the same forward recomputes and checks
  against ``heads/acts``: the sequences are the pipeline's;
- **ablation** (``ablate_copy``): each head zeroed in turn (its slice of ``o_proj``'s input, at
  every position), the change in the mean loss over the second copy of the first ``N_ABLATE``
  sequences, in one batch, in nats: positive, the head helps copying. Zeroing a head of layer
  ``L`` leaves the layers below unchanged, so each ablated pass replays the decoder layers from
  ``L`` on, from layer ``L``'s input and arguments captured in one clean pass; the replay from
  layer 0 and from the middle reproduces the clean pass's loss (logged);
- **by pattern class** (``heads/acts``; induction and duplicate past 0.3, previous-token past 0.5,
  sink past 0.9): each class's mean, median and largest ablation effect, and its heads past
  ``HELPS``.

Outputs: ``data/heads/<model>-ablate-copy.pt`` (``ablate_copy`` ``[layers, heads]``, ``base``, the
clean loss, and ``meta``), read by ``head_circuits.py`` and ``head_media.py``; the summary in
``data/summary/head_patterns_<model>.{log,json}``. Qwen3-8B on an M5 Max (MPS): 0.05-0.09 s a
replayed layer, 1,152 heads in about 25 minutes (45 with the GPU shared); ``--heads`` takes a few
seconds a head.

    python scripts/checks/head_patterns.py [--heads L1h28,L19h20]
    NEURON_LENS_MODEL=google/gemma-4-12B python scripts/checks/head_patterns.py
"""
import argparse
import json
import time

import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import MODEL, head_table, kept, tokenizer
from neuron_lens.paths import DATA, SUMMARY

REP_SEQS, REP_LEN, BATCH = 64, 64, 16  # 06_activations.py's repeated-token pass
N_ABLATE = 32
CLASSES = {"induction": 0.3, "duplicate": 0.3, "prev": 0.5, "sink": 0.9}
HELPS, SUPPRESSES = 0.01, -0.005  # nats
STEM = arch.stem(MODEL)
OUT = DATA / "heads" / f"{STEM}-ablate-copy.pt"


def sequences(tok) -> torch.Tensor:
    """``[REP_SEQS, 2·REP_LEN]``: ``06_activations.py``'s random kept tokens, each sequence said twice."""
    special = set(tok.all_special_ids)
    pool = torch.tensor([i for i in kept().tolist() if i not in special])
    half = pool[torch.randint(len(pool), (REP_SEQS, REP_LEN), generator=torch.Generator().manual_seed(0))]
    return torch.cat([half, half], 1)


def zeroing(head_dim: int, state: dict):
    """An ``o_proj`` pre-hook zeroing head ``state["head"]``'s slice of the input."""
    def hook(m, a):
        h = state["head"]
        a[0][..., h * head_dim:(h + 1) * head_dim] = 0
    return hook


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", help="ablate only these heads, e.g. L1h28,L19h20 (a smoke test; nothing saved)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
    args = ap.parse_args()
    dev = args.device
    tok = tokenizer()
    lm, stack = arch.load_model(MODEL, dev)
    geo = arch.layout(MODEL)
    L, H = len(geo), geo[0].q_heads
    pre = arch.bos(tok)
    rep = sequences(tok)
    ids = torch.cat([torch.tensor(pre, dtype=rep.dtype).expand(len(rep), -1), rep], 1)
    o = len(pre)
    first = torch.arange(o, o + REP_LEN - 1, device=dev)  # positions predicting the first copy's tokens 2-64
    second = first + REP_LEN  # and the second copy's

    def nll(hidden: torch.Tensor, pos: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``[B, len(pos)]``: the loss at ``pos`` from the stack's last (normed) hidden state."""
        lp = torch.log_softmax(arch.logits(lm, hidden[:, pos]), -1)
        return -lp.gather(2, b[:, pos + 1, None]).squeeze(-1)

    # Copying, and the induction and duplicate scores of the same forward.
    att = {}
    hooks = [layer.self_attn.register_forward_hook(lambda m, x, out, l=l: att.__setitem__(l, out[1]))
             for l, layer in enumerate(stack.layers)]
    ind, dup, loss = torch.zeros(L, H), torch.zeros(L, H), []
    with torch.inference_mode():
        for s in range(0, REP_SEQS, BATCH):
            b = ids[s:s + BATCH].to(dev)
            hid = stack(input_ids=b, use_cache=False).last_hidden_state
            loss.append(torch.stack([nll(hid, first, b).mean(1), nll(hid, second, b).mean(1)], 1).cpu())
            for l in range(L):
                a = att.pop(l)
                ind[l] += a[:, :, second, second - REP_LEN + 1].float().mean(-1).sum(0).cpu()
                dup[l] += a[:, :, second, second - REP_LEN].float().mean(-1).sum(0).cpu()
    for hk in hooks:
        hk.remove()
    loss = torch.cat(loss).mean(0)
    ind, dup = ind / REP_SEQS, dup / REP_SEQS
    acts = pq.read_table(head_table("acts"), columns=["sink", "prev", "induction", "duplicate"]).to_pydict()
    pat = {k: torch.tensor(v) for k, v in acts.items()}
    d_ind = (ind.flatten() - pat["induction"]).abs().max().item()
    d_dup = (dup.flatten() - pat["duplicate"]).abs().max().item()
    print(f"loss on the first copy {loss[0]:.2f}, on the second {loss[1]:.3f} nats; induction and duplicate against "
          f"heads/acts: max |Δ| {d_ind:.4f}, {d_dup:.4f}", flush=True)

    # Ablation: replay the decoder layers from the zeroed head's layer, from a clean pass's inputs.
    abl = ids[:N_ABLATE].to(dev)
    seen = []
    cap = [layer.register_forward_pre_hook(lambda m, a, kw: seen.append((a[0].clone(), a[1:], kw)), with_kwargs=True)
           for layer in stack.layers]
    with torch.inference_mode():
        full = nll(stack(input_ids=abl, use_cache=False).last_hidden_state, second, abl).mean().item()
    for hk in cap:
        hk.remove()

    def replay(l: int) -> float:
        h = seen[l][0]
        for i in range(l, L):
            h = stack.layers[i](h, *seen[i][1], **seen[i][2])
        return nll(stack.norm(h), second, abl).mean().item()

    with torch.inference_mode():
        base, mid = replay(0), replay(L // 2)
    print(f"clean loss on the second copy of {N_ABLATE}: {full:.5f} (replayed from L0 {base:.5f}, from L{L // 2} {mid:.5f})",
          flush=True)
    todo = {l: list(range(H)) for l in range(L)}
    if args.heads:
        todo = {}
        for x in args.heads.split(","):
            l, h = map(int, x.strip()[1:].split("h"))
            todo.setdefault(l, []).append(h)
    effect = torch.full((L, H), torch.nan)
    state, t0 = {}, time.perf_counter()
    for l, heads in sorted(todo.items()):
        hook = stack.layers[l].self_attn.o_proj.register_forward_pre_hook(zeroing(geo[l].head_dim, state))
        with torch.inference_mode():
            for h in heads:
                state["head"] = h
                t1 = time.perf_counter()
                effect[l, h] = replay(l) - base
                if args.heads:
                    print(f"  L{l}h{h} {effect[l, h]:+.4f} nats ({time.perf_counter() - t1:.1f}s)", flush=True)
        hook.remove()
        e = effect[l].nan_to_num(-1)
        print(f"{time.strftime('%H:%M:%S')} L{l}: top h{int(e.argmax())} {e.max():+.3f} nats, {time.perf_counter() - t0:.0f}s",
              flush=True)
    if not args.heads:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ablate_copy": effect, "base": base,
                    "meta": {"model": MODEL, "sequences": N_ABLATE, "rep_len": REP_LEN, "seed": 0, "dtype": "bfloat16",
                             "attention": "eager"}}, OUT)

    done = ~effect.flatten().isnan()  # every head, but for a smoke test
    flat = effect.flatten().nan_to_num(-torch.inf)
    hl = lambda i: f"L{i // H}h{i % H}"  # noqa: E731
    cls = {k: (pat[k] > v) for k, v in CLASSES.items()}
    cls["other"] = ~torch.stack(list(cls.values())).any(0)
    cls["all"] = torch.ones_like(done)
    top = [i for i in flat.argsort(descending=True).tolist() if done[i]][:10]
    e = flat[done]
    table = {"model": MODEL, "sequences": REP_SEQS, "rep_len": REP_LEN, "loss_first": loss[0].item(),
             "loss_second": loss[1].item(), "induction_max_diff": d_ind, "duplicate_max_diff": d_dup,
             "ablation": {"sequences": N_ABLATE, "base": base, "full_forward": full, "max": e.max().item(),
                          "past_0.01": int((e > HELPS).sum()), "past_0.02": int((e > 0.02).sum()),
                          "below_-0.005": int((e < SUPPRESSES).sum()),
                          "top": [[hl(i), flat[i].item()] for i in top], "classes": {}}}
    lines = [(f"in-context copying and single-head ablation, {MODEL}: {L} layers × {H} query heads; {REP_SEQS} sequences "
              f"of {REP_LEN} random kept tokens said twice (06_activations.py's), bf16, eager attention"),
             (f"loss on the first copy {loss[0]:.2f}, on the second {loss[1]:.2f} nats; induction and duplicate recomputed "
              f"against heads/acts: max |Δ| {d_ind:.4f} / {d_dup:.4f}"),
             (f"ablation (each head zeroed, change in the second copy's loss over the first {N_ABLATE} sequences; clean "
              f"{base:.4f} nats, full forward {full:.4f}): largest {e.max():+.3f} nats; heads past {HELPS}: "
              f"{int((e > HELPS).sum())}, past 0.02: {int((e > 0.02).sum())}, below {SUPPRESSES}: {int((e < SUPPRESSES).sum())}"),
             "top: " + ", ".join(f"{hl(i)} {flat[i]:+.3f}" for i in top[:8]), "",
             "by pattern class (heads/acts: " + ", ".join(f"{k} > {v}" for k, v in CLASSES.items())
             + "): heads, mean / median / largest ablation effect, heads past " + str(HELPS)]
    for k, m in cls.items():
        v = flat[m & done]
        if not len(v):
            continue
        i = int(torch.where(m & done, flat, -torch.inf).argmax())
        row = {"heads": len(v), "mean": v.mean().item(), "median": v.median().item(), "max": [hl(i), flat[i].item()],
               "helps": int((v > HELPS).sum())}
        table["ablation"]["classes"][k] = row
        lines.append(f"  {k:9s} {row['heads']:4d}  {row['mean']:+.4f} / {row['median']:+.4f} / {flat[i]:+.3f} ({hl(i)})  "
                     f"{row['helps']}")
    lines += ["", "top heads by ablation effect, with their pattern scores (heads/acts)"]
    lines += [f"  {hl(i):7s} {flat[i]:+.3f}  " + "  ".join(f"{k} {pat[k][i]:.2f}" for k in ("induction", "duplicate", "prev", "sink"))
              for i in top]
    text = "\n".join(lines)
    print(text)
    if args.heads:
        return
    (SUMMARY / f"head_patterns_{STEM}.log").write_text(text + "\n")
    (SUMMARY / f"head_patterns_{STEM}.json").write_text(json.dumps(table))


if __name__ == "__main__":
    main()
