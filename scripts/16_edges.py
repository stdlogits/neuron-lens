"""Direct weight edges between neuron poles (pipeline stage 16; findings §17).

The residual stream is a sum: what a neuron of layer L writes reaches every later layer's MLP input
unchanged, so the direct path from a pole A to a pole B of a later layer M is one number from the
weights, the size of A's write along B's read direction. Per pole, ``neurons/edges/L<layer>.parquet``
lists the ``K`` poles of earlier layers whose writes push it most (``in_*``) and the ``K`` poles of
later layers it pushes most (``out_*``), each as ``_layer``, ``_neuron``, ``_pole``, ``_push`` and
``_cos``, strongest first (``−1`` and NaN where fewer clear the floor):

- ``cos``: the cosine of A's write ``±W_down[:, n]`` with B's read direction ``â ± b̂`` (gate and up
  rows times the MLP input norm's weight, normalized; ``neurons.directions``, as ``08_reads.py``);
- ``push``: ``cos · act_top_A · ‖w_A‖ / rms_M``, how far one of A's top firings moves B's normalized
  MLP input along B's read direction; ``rms_M`` is the median RMS of the residual entering layer M's
  MLP (``RMS_DOCS`` catalog documents, positions from 1). Positive: A's firing drives B's pole;
  negative: it pushes against it.

Edges rank by ``|push|`` among those with ``|cos| ≥ SIGMAS/√d``: random directions reach that about
once in 3.5 million, against about 430,000 candidates a pole at most, and without the floor neurons
with huge activations (undertrained tokens, findings §2) would top many lists at chance cosines.

An edge is the direct path at one position: A's write reaches B where A fired. What a head carries
between positions, and paths through other units, are not edges. ``push`` holds A at its typical top
firing; whether A is active when B fires is the context's, which ``checks/edges_in_context.py``
measures (§17). Per band into ``data/summary/edges_<model>.{log,json}``.

Qwen3-8B only: Gemma 4's post-feedforward norm and layer scalars rescale each write per position.

    python scripts/16_edges.py [--k 16] [--device mps]
"""
import argparse
import json
import time

import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import (
    MODEL,
    documents,
    fixed,
    neuron_keys,
    neuron_table,
    write,
)
from neuron_lens.neurons import directions
from neuron_lens.paths import SUMMARY

STEM = arch.stem(MODEL)
K = 16
SIGMAS = 5.0
RMS_DOCS = 64


@torch.inference_mode()
def residual_rms(dev) -> torch.Tensor:
    """``[L]``: the median RMS of the residual entering each layer's MLP, over ``RMS_DOCS`` documents
    spread through the catalog's (positions from 1)."""
    lm, stack = arch.load_model(MODEL, dev, eager=False)
    seen = [[] for _ in stack.layers]
    hooks = [lay.post_attention_layernorm.register_forward_pre_hook(
        lambda mod, a, i=i: seen[i].append(a[0][0, 1:].float().pow(2).mean(-1).sqrt().cpu()))
        for i, lay in enumerate(stack.layers)]
    docs = documents()
    for doc in docs[::len(docs) // RMS_DOCS][:RMS_DOCS]:
        stack(input_ids=doc[None].long().to(dev))
    for h in hooks:
        h.remove()
    del lm, stack
    return torch.stack([torch.cat(s).median() for s in seen])


def poles(layer: int, dev) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A layer's unit writes ``[d, f]``, each pole's write size at its typical top firing
    (``act_top · ‖w‖``, ``[2f]``, ``+`` poles first) and its read directions ``[2f, d]``."""
    w = directions(MODEL, layer)
    norm = w["write"].norm(dim=0)
    act = torch.from_numpy(pq.read_table(neuron_table("acts", layer), columns=["act_top"]).column(0).to_numpy().copy()).float()
    a = w["gate"] / w["gate"].norm(dim=1, keepdim=True)
    b = w["up"] / w["up"].norm(dim=1, keepdim=True)
    read = torch.cat([a + b, a - b])
    return (w["write"] / norm).to(dev), (act * torch.cat([norm, norm])).to(dev), (read / read.norm(dim=1, keepdim=True)).to(dev)


def merge(best: tuple, new: tuple, k: int) -> tuple:
    """Keep the ``k`` highest scores of two ``(score, id, push, cos)`` lists, row by row."""
    cat = [torch.cat([a, b], 1) for a, b in zip(best, new)]
    top = cat[0].topk(k, dim=1).indices
    return tuple(c.gather(1, top) for c in cat)


def compute(k: int, dev) -> tuple[list, list, torch.Tensor, torch.Tensor]:
    """Every layer's in- and out-lists, ``(score, id, push, cos)`` each ``[2f, k]``; ids are
    ``layer · 2f + row`` (rows in table order), scores ``|push|`` or −1 below the floor. And the null,
    ``[L, 2, 2f]``: each pole's largest |cos| with the layer before's writes, and with them rotated at random."""
    rms = residual_rms(dev)
    L = len(rms)
    units = [poles(layer, dev) for layer in range(L)]
    d, f = units[0][0].shape
    floor = SIGMAS / d ** 0.5
    empty = lambda: (torch.full((2 * f, k), -1.0, device=dev), torch.full((2 * f, k), -1, device=dev),
                     torch.full((2 * f, k), float("nan"), device=dev), torch.full((2 * f, k), float("nan"), device=dev))
    ins, outs = [empty() for _ in range(L)], [empty() for _ in range(L)]
    rot = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(0)))[0].to(dev)
    prev = torch.full((L, 2, 2 * f), float("nan"))
    t0 = time.perf_counter()
    for m in range(1, L):
        read = units[m][2]
        for l in range(m):
            w, size, _ = units[l]
            c = read @ w
            cos = torch.cat([c, -c], 1)  # [2f B poles, 2f A poles]
            del c
            if l == m - 1:
                prev[m, 0], prev[m, 1] = cos.abs().amax(1).cpu(), (read @ (rot @ w)).abs().amax(1).cpu()
            push = cos * size / rms[m]
            score = torch.where(cos.abs() >= floor, push.abs(), -1.0)
            ids = l * 2 * f + torch.arange(2 * f, device=dev)
            top = score.topk(k, dim=1).indices
            ins[m] = merge(ins[m], (score.gather(1, top), ids[top], push.gather(1, top), cos.gather(1, top)), k)
            top = score.topk(k, dim=0).indices.T  # [2f A poles, k]
            ids = m * 2 * f + top
            pick = lambda t: t.T.gather(1, top)
            outs[l] = merge(outs[l], (pick(score), ids, pick(push), pick(cos)), k)
            del cos, push, score
        print(f"{time.strftime('%H:%M:%S')} L{m}: in from L0-L{m - 1}, {time.perf_counter() - t0:.0f}s", flush=True)
    return [tuple(t.cpu() for t in x) for x in ins], [tuple(t.cpu() for t in x) for x in outs], rms, prev


def columns(cols: dict, side: str, lists: tuple, f: int) -> None:
    score, ids, push, cos = lists
    ok = score >= 0
    ids = torch.where(ok, ids, -1)
    layer, row = ids.div(2 * f, rounding_mode="floor"), ids % (2 * f)
    cols[f"{side}_layer"] = fixed(torch.where(ok, layer, -1).to(torch.int16))
    cols[f"{side}_neuron"] = fixed(torch.where(ok, row % f, -1).to(torch.int16))
    cols[f"{side}_pole"] = fixed(torch.where(ok, 1 - 2 * (row // f), 0).to(torch.int8))
    cols[f"{side}_push"] = fixed(torch.where(ok, push, float("nan")))
    cols[f"{side}_cos"] = fixed(torch.where(ok, cos, float("nan")))


def summary(ins: list, outs: list, rms: torch.Tensor, prev: torch.Tensor, k: int, d: int) -> None:
    L, f = len(ins), ins[0][0].shape[0] // 2
    lines = [f"direct weight edges, {MODEL}: top {k} in and out per pole by |push| among |cos| >= {SIGMAS}/sqrt(d) "
             f"= {SIGMAS / d ** 0.5:.3f}; residual RMS at the MLP input "
             + ", ".join(f"L{i} {r:.1f}" for i, r in enumerate(rms.tolist()) if i % 4 == 0 or i == L - 1), "",
             "band     prev  rotated  edged  top|cos|  top push  in-edges  gap1  gap  pos   hubs | out-edged  out-edges  out gap",
             "  (prev, rotated: median of each pole's largest |cos| with the layer before's writes, and with them rotated at random; edged: share of poles with an in-edge; top |cos|, top push: medians of the strongest in-edge's; "
             "in-edges: median count above the floor, of k; gap1: share of strongest in-edges from the layer before; "
             "gap: median layer gap; pos: share of strongest in-edges that drive the pole; hubs: share of the band's "
             "in-list entries held by the 1% most-listed sources; out-: the same for out-lists)"]
    table = {}
    for lo, hi in arch.bands(L) + [(0, L - 1)]:
        cat = lambda lists, i, a=lo, b=hi: torch.cat([lists[m][i] for m in range(max(a, 0), b + 1)])
        s_in, id_in, p_in, c_in = (cat(ins, i) for i in range(4))
        s_out, id_out = cat(outs, 0), cat(outs, 1)
        own = torch.cat([torch.full((2 * f,), m) for m in range(lo, hi + 1)])
        has, has_out = s_in[:, 0] >= 0, s_out[:, 0] >= 0
        listed = id_in[s_in >= 0]
        counts = torch.bincount(listed - listed.min()).sort(descending=True).values if len(listed) else torch.zeros(1)
        n_hub = max(1, int(0.01 * (counts > 0).sum()))
        p = prev[max(lo, 1):hi + 1]
        row = {"poles": len(has), "prev": p[:, 0].median().item(), "rotated": p[:, 1].median().item(), "edged": has.float().mean().item(),
               "top_cos": c_in[has, 0].abs().median().item() if has.any() else float("nan"),
               "top_push": p_in[has, 0].abs().median().item() if has.any() else float("nan"),
               "in_edges": (s_in >= 0).sum(1).float().median().item(),
               "gap1": (own[has] - id_in[has, 0] // (2 * f) == 1).float().mean().item() if has.any() else float("nan"),
               "gap": (own[has] - id_in[has, 0] // (2 * f)).float().median().item() if has.any() else float("nan"),
               "pos": (p_in[has, 0] > 0).float().mean().item() if has.any() else float("nan"),
               "hubs": (counts[:n_hub].sum() / counts.sum()).item() if len(listed) else float("nan"),
               "out_edged": has_out.float().mean().item(),
               "out_edges": (s_out >= 0).sum(1).float().median().item(),
               "out_gap": (id_out[has_out, 0] // (2 * f) - own[has_out]).float().median().item() if has_out.any() else float("nan")}
        table[f"L{lo}-{hi}"] = row
        lines.append(f"L{lo}-{hi}".ljust(8) + f" {row['prev']:4.3f}  {row['rotated']:7.3f}  {row['edged']:5.2f}  {row['top_cos']:8.3f}  {row['top_push']:8.3f}  "
                     f"{row['in_edges']:8.0f}  {row['gap1']:4.2f}  {row['gap']:3.0f}  {row['pos']:4.2f}  {row['hubs']:4.2f} | "
                     f"{row['out_edged']:9.2f}  {row['out_edges']:9.0f}  {row['out_gap']:7.0f}")
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"edges_{STEM}.log").write_text(text + "\n")
    (SUMMARY / f"edges_{STEM}.json").write_text(json.dumps({"model": MODEL, "k": k, "sigmas": SIGMAS,
                                                            "rms": rms.tolist(), "bands": table}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
    args = ap.parse_args()
    if arch.is_gemma4(MODEL):
        raise SystemExit("16_edges.py reads Qwen3's residual, where each write is fixed; Gemma 4's norms rescale it per position")
    ins, outs, rms, prev = compute(args.k, args.device)
    f = ins[0][0].shape[0] // 2
    for layer, (i, o) in enumerate(zip(ins, outs)):
        cols = neuron_keys(layer, f)
        columns(cols, "in", i, f)
        columns(cols, "out", o, f)
        write(cols, neuron_table("edges", layer))
    summary(ins, outs, rms, prev, args.k, directions(MODEL, 0)["write"].shape[0])


if __name__ == "__main__":
    main()
