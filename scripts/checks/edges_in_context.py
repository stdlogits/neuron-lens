"""Do the direct weight edges name what drives a pole when it fires? (findings §17)

**Poles.** §3's random neurons (``causal_lens.random_neurons``: 32 a layer, seed 0) of layers 1+,
both poles, at their first ``N_CONTEXTS`` top firings (the ``acts`` table's ``top_doc``, ``top_pos``:
the firings the cards show), where they fired (``top_val > 0``).

**The direct decomposition.** At a firing of pole B (layer M, position t), B's MLP input is the
residual ``x`` there, normalized, and ``x`` is a sum: the embedding, every head's write at layers ≤ M
and every neuron's at layers < M, all at t. Holding the norm's scale and SiLU's ratio ``s = silu(g)/g``
(RelP's rules), B's gate and up pre-activations ``g``, ``u`` split exactly over these sources, and so
does its activation by the half rule: source X contributes ``½ s (g u_X + u g_X)``, and the
contributions sum to ``a = silu(g) u``. A share is a contribution over ``a`` (positive: it drives the
pole). An upstream neuron is in the pole its activation's sign gives, and *fires* at ``act_top / 4``
or more there (the ``acts`` tables' rate threshold). The model runs in fp32 (bf16 rounds weak writes
away; §4), up to layer M's MLP.

**Per firing** (``K`` = the edges table's list length):

- ``emb``, ``heads``, ``neurons``: each kind of source's share (they sum to 1);
- ``top``: the share from the ``K`` neuron poles that contribute most there, a ceiling for any list;
  ``fire_top``: the share of the ``K`` largest |contributions| whose source fires;
- ``listed``: the share from the pole's ``K`` in-edges (``16_edges.py``), each counted where its
  source is in the listed pole; ``random``: from ``K`` random poles of layers < M;
- ``fire``: the share of in-edge sources that fire; ``fire_random``: of the random poles;
- ``recall``: the share of the ``K`` largest |contributions| that are in-edges; ``recall_ctrl``: that
  are in-edges of the next sampled pole of the layer;
- ``rho``: Spearman over every upstream neuron between its edge push (``s · cos · act_top · ‖w‖``,
  in the pole it is in) and its contribution; ``rho_cos``: with the cosine alone;
- ``gate_neg``: whether the gate pre-activation is negative there (SiLU's negative lobe, where the
  read direction ``â ± b̂`` points the wrong way).

A pole's value is its mean over its firings; bands give medians over poles (``gate_neg``: the share of
all their firings). ``err`` is the largest
``|Σ contributions − a| / |a|`` (the decomposition's check) and ``a_cat`` the median ratio of ``a``
to the catalog's ``top_val`` (bf16 activations).

``measure`` saves per-firing values to ``data/causal/<model>-edges.pt`` after every layer and
resumes there (Qwen3-8B on an M5 Max, MPS, fp32: 15 minutes); ``score`` writes
``data/summary/edges_in_context_<model>.{log,json}``; ``explain`` prints one pole's largest sources,
neurons and heads, by their mean share over its firings, beside its in-edges. The documents come from
``catalog.documents``.

    python scripts/checks/edges_in_context.py measure [--layers 1,14,30]
    python scripts/checks/edges_in_context.py score
    python scripts/checks/edges_in_context.py explain 14 1561 +
"""
import argparse
import json
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM

from causal_lens import CAUSAL, random_neurons
from neuron_lens import arch
from neuron_lens.catalog import (
    MODEL,
    N_CONTEXTS,
    column,
    documents,
    edge_list,
    entry,
    head_entry,
    neuron_table,
)
from neuron_lens.paths import SUMMARY
from neuron_lens.sources import split
from neuron_lens.stats import spearman
from neuron_lens.transport import decoder_stack

STEM = arch.stem(MODEL)
PER_LAYER = 32  # §3's random picks
BATCH = 16
FIRE = 1 / 4  # of act_top: the acts tables' rate threshold
PER_FIRING = ("emb", "heads", "neurons", "top", "fire_top", "listed", "random", "fire", "fire_random", "recall",
              "recall_ctrl", "rho", "rho_cos", "gate_neg", "err", "a_cat")
PATH = CAUSAL / f"{STEM}-edges.pt"


class Stop(Exception):
    pass


def firings(layer: int, neurons: list[int]) -> tuple[torch.Tensor, ...]:
    """``(doc, pos, val)``, each ``[n, 2, N_CONTEXTS]``: the neurons' first top firings per pole (``+``, ``−``)."""
    t = pq.read_table(neuron_table("acts", layer), columns=["top_doc", "top_pos", "top_val"])
    rows = torch.tensor(neurons + [t.num_rows // 2 + n for n in neurons])
    return tuple(column(t, k, N_CONTEXTS)[rows].view(2, len(neurons), -1).transpose(0, 1)
                 for k in ("top_doc", "top_pos", "top_val"))


def in_edges(layer: int, neurons: list[int]) -> torch.Tensor:
    """``[n, 2, K]``: each pole's in-edges as global pole ids ``l · 2f + row`` (−1 where none)."""
    t = pq.read_table(neuron_table("edges", layer), columns=["in_layer", "in_neuron", "in_pole"])
    f = t.num_rows // 2
    rows = torch.tensor(neurons + [f + n for n in neurons])
    l, n, p = (column(t, k)[rows].long() for k in ("in_layer", "in_neuron", "in_pole"))
    ids = l * 2 * f + n + f * (p < 0)
    return torch.where(l >= 0, ids, -1).view(2, len(neurons), -1).transpose(0, 1)


class Model:
    """Qwen3-8B in fp32 with what the decomposition needs: each pole's write size (``act_top · ‖w‖``)
    and activation scale (``act_top``), ``[L, 2f]``, and every layer's unit writes."""

    def __init__(self, dev):
        lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval().requires_grad_(False)
        self.stack, self.dev = decoder_stack(lm), dev
        self.L, self.f = len(self.stack.layers), self.stack.config.intermediate_size
        self.H = self.stack.config.num_attention_heads
        self.act_top = torch.stack([torch.from_numpy(pq.read_table(neuron_table("acts", l), columns=["act_top"])
                                                     .column(0).to_numpy().copy()).float() for l in range(self.L)]).to(dev)
        self.downs = [lay.mlp.down_proj.weight for lay in self.stack.layers]  # [d, f]
        self.unit = [w / w.norm(dim=0) for w in self.downs]
        self.size = self.act_top * torch.stack([w.norm(dim=0) for w in self.downs]).repeat(1, 2)

    @torch.inference_mode()
    def sources(self, M: int, neuron: torch.Tensor, pole: torch.Tensor, seqs: list[torch.Tensor]) -> dict:
        """Run each document (cut after its firing) to layer M's MLP and split the firing pole's
        activation there over its sources: ``emb`` ``[B]``, ``heads`` ``[B, (M + 1)·H]`` and
        ``neurons`` ``[B, M·f]`` shares; ``acts`` ``[B, M·f]``, the upstream activations; ``pid``,
        the global ids of the poles they are in; ``push``, ``cos``, their edge to the pole; ``g``, ``a``."""
        stack, dev, B, f = self.stack, self.dev, len(seqs), self.f
        t = torch.tensor([len(s) - 1 for s in seqs], device=dev)
        ids = torch.zeros(B, max(map(len, seqs)), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for r, s in enumerate(seqs):
            ids[r, :len(s)], mask[r, :len(s)] = s, 1
        at = lambda h: h[torch.arange(B, device=dev), t]
        got = {"z": [], "a": []}
        lay = stack.layers[M]
        norm = lay.post_attention_layernorm
        hooks = [stack.embed_tokens.register_forward_hook(lambda mod, a, o: got.__setitem__("emb", at(o)))]
        hooks += [stack.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mod, a: got["z"].append(at(a[0])))
                  for l in range(M + 1)]
        hooks += [stack.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, a: got["a"].append(at(a[0])))
                  for l in range(M)]
        hooks.append(norm.register_forward_pre_hook(lambda mod, a: got.__setitem__("x", at(a[0]))))

        def stop(mod, a):
            raise Stop
        hooks.append(lay.mlp.register_forward_pre_hook(stop))
        try:
            stack(input_ids=ids.to(dev), attention_mask=mask.to(dev))
        except Stop:
            pass
        finally:
            for h in hooks:
                h.remove()
        x = got["x"]
        inv = torch.rsqrt(x.pow(2).mean(-1) + norm.variance_epsilon)[:, None]  # [B, 1]: the norm's scale, held
        rg = lay.mlp.gate_proj.weight[neuron] * norm.weight  # [B, d]: read rows times the norm weight
        ru = lay.mlp.up_proj.weight[neuron] * norm.weight
        got_s = split(rg, ru, x, inv, got["emb"], got["z"], got["a"], self.downs[:M],
                      [stack.layers[l].self_attn.o_proj.weight for l in range(M + 1)], self.H)
        emb, heads, neurons, g, a = got_s["emb"], got_s["heads"], got_s["neurons"], got_s["g"], got_s["a"]
        acts = torch.cat(got["a"], 1)  # [B, M·f]
        sign = acts.sign()
        pid = (torch.arange(M, device=dev)[:, None] * 2 * f + torch.arange(f, device=dev)).flatten() + f * (sign < 0)
        a_hat, b_hat = (r / r.norm(dim=1, keepdim=True) for r in (rg, ru))
        c_b = a_hat + pole[:, None] * b_hat
        cos = torch.cat([c_b / c_b.norm(dim=1, keepdim=True) @ self.unit[l] for l in range(M)], 1) * sign
        return {"emb": emb, "heads": heads, "neurons": neurons, "acts": acts, "pid": pid,
                "cos": cos, "push": cos * self.size.flatten()[pid], "g": g, "a": a}

    def fires(self, src: dict, r: int, q: torch.Tensor) -> torch.Tensor:
        """Whether each global pole id ``q`` is the pole its neuron is in at row ``r``'s firing, and
        whether it fires there; ``q`` as an index into the row's neurons."""
        f = self.f
        n = (q // (2 * f)) * f + q % f
        in_pole = src["pid"][r, n] == q
        return in_pole, in_pole & (src["acts"][r, n].abs() >= FIRE * self.act_top.flatten()[q]), n


def batches(rows: list, doc, pos, docs):
    """The firings in batches of ``BATCH``, shortest documents together, each document cut after its firing."""
    rows = sorted(rows, key=lambda r: int(pos[r]))
    for b in range(0, len(rows), BATCH):
        batch = rows[b:b + BATCH]
        yield batch, [docs[int(doc[r])][:int(pos[r]) + 1].long() for r in batch]


def measure(args) -> None:
    model = Model(args.device)
    dev, L, f = model.dev, model.L, model.f
    docs = documents()
    picks = random_neurons(L, f, PER_LAYER)
    K = pq.read_schema(neuron_table("edges", 0)).field("in_layer").type.list_size
    meta = {"model": MODEL, "per_layer": PER_LAYER, "contexts": N_CONTEXTS, "k": K, "dtype": "fp32"}
    if PATH.exists():
        out = torch.load(PATH, weights_only=False)
        if out["meta"] != meta:
            raise SystemExit(f"{PATH} was measured with {out['meta']}, not {meta}: move it to start afresh")
    else:
        out = {"meta": meta, "picks": picks, "done": []} | {k: torch.full((len(picks), 2, N_CONTEXTS), float("nan"))
                                                              for k in PER_FIRING}
    gen = torch.Generator().manual_seed(0)
    layers = range(1, L) if args.layers is None else [int(x) for x in args.layers.split(",")]
    t0 = time.perf_counter()
    for M in layers:
        if M in out["done"]:
            continue
        idx = [i for i, (l, _) in enumerate(picks) if l == M]
        neurons = [picks[i][1] for i in idx]
        doc, pos, val = firings(M, neurons)
        listed = in_edges(M, neurons).to(dev)
        rows = [(j, p, k) for j in range(len(idx)) for p in (0, 1) for k in range(N_CONTEXTS) if val[j, p, k] > 0]
        for batch, seqs in batches(rows, doc, pos, docs):
            pole = torch.tensor([1.0 - 2 * p for _, p, _ in batch], device=dev)
            src = model.sources(M, torch.tensor([neurons[j] for j, _, _ in batch], device=dev), pole, seqs)
            rand = torch.randint(0, M * 2 * f, (len(batch), K), generator=gen).to(dev)
            for r, (j, p, k) in enumerate(batch):
                lst = listed[j, p][listed[j, p] >= 0]
                ctrl = listed[(j + 1) % len(idx), p]
                c = src["neurons"][r]
                in_l, fire_l, n_l = model.fires(src, r, lst)
                in_r, fire_r, n_r = model.fires(src, r, rand[r])
                big = c.abs().topk(K).indices
                _, fire_big, _ = model.fires(src, r, src["pid"][r, big])
                v = {"emb": src["emb"][r], "heads": src["heads"][r].sum(), "neurons": c.sum(), "top": c.topk(K).values.sum(),
                     "fire_top": fire_big.float().mean(), "listed": (c[n_l] * in_l).sum(), "random": (c[n_r] * in_r).sum(),
                     "fire": fire_l.float().mean(), "fire_random": fire_r.float().mean(),
                     "recall": torch.isin(src["pid"][r, big], lst).float().sum() / K,
                     "recall_ctrl": torch.isin(src["pid"][r, big], ctrl[ctrl >= 0]).float().sum() / K,
                     "rho": spearman(src["push"][r], c), "rho_cos": spearman(src["cos"][r], c), "gate_neg": src["g"][r] < 0,
                     "err": (src["emb"][r] + src["heads"][r].sum() + c.sum() - 1).abs(),
                     "a_cat": pole[r] * src["a"][r] / val[j, p, k]}
                for name, value in v.items():
                    out[name][idx[j], p, k] = float(value)
        out["done"].append(M)
        torch.save(out, PATH)
        print(f"{time.strftime('%H:%M:%S')} L{M}: {len(rows)} firings, {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"-> {PATH}", flush=True)


def score(args) -> None:
    saved = torch.load(PATH, weights_only=False)
    picks, done = saved["picks"], sorted(saved["done"])
    L = arch.text_config(MODEL).num_hidden_layers
    pole = {k: saved[k].nanmean(-1) for k in PER_FIRING}  # [P, 2]
    pole["err"] = saved["err"].nan_to_num(0).amax(-1)
    pole["a_cat"] = saved["a_cat"].nanmedian(-1).values
    scored = ~saved["neurons"].isnan().all(-1)
    in_layers = lambda lo, hi: torch.tensor([lo <= l <= hi for l, _ in picks])[:, None] & scored
    cols = [k for k in PER_FIRING if k != "err"]

    def row(sel):
        r = {k: pole[k][sel].nanmedian().item() for k in cols} | {"poles": int(sel.sum())}
        r["gate_neg"] = saved["gate_neg"][sel].nanmean().item()
        return r | {"err": pole["err"][sel].max().item() if sel.any() else float("nan")}
    bands = [(max(lo, 1), hi) for lo, hi in arch.bands(L) + [(0, L - 1)] if any(lo <= l <= hi for l in done)]
    table = {f"L{lo}-{hi}": row(in_layers(lo, hi)) for lo, hi in bands}
    layers = {layer: row(in_layers(layer, layer)) for layer in done}
    lines = [f"direct edges in context, {MODEL}: {saved['meta']}, layers {done}",
             "medians over poles of each pole's mean over its firings. emb, heads, neurons: each source's share of the "
             "activation (sum 1); top: the K largest neuron contributions (fire_top: the share of them whose source fires, "
             "at act_top/4 or more); listed: the pole's K in-edges, random: K random upstream poles, each where its source "
             "is in the listed pole; fire: in-edge sources that fire; recall: the share of the K largest |contributions| "
             "that are in-edges (ctrl: another pole's); rho: Spearman of edge push (rho_cos: cosine) against contribution "
             "over all upstream neurons; gate_neg: the share of firings with a negative gate; err: the largest decomposition error; a_cat: "
             "activation over the catalog's top_val",
             ""]
    head = "band     poles  " + "  ".join(f"{k:>11s}" for k in cols)
    lines.append(head + "          err")
    for name, r in table.items():
        lines.append(f"{name:8s} {r['poles']:5d}  " + "  ".join(f"{r[k]:11.3f}" for k in cols) + f"  {r['err']:11.1e}")
    lines += ["", "per layer", head] + [f"L{l:<7d} {r['poles']:5d}  " + "  ".join(f"{r[k]:11.3f}" for k in cols)
                                        for l, r in layers.items()]
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"edges_in_context_{STEM}.log").write_text(text + "\n")
    (SUMMARY / f"edges_in_context_{STEM}.json").write_text(json.dumps({"meta": saved["meta"], "bands": table,
                                                                       "layers": layers}))


def explain(args) -> None:
    M, n, p = int(args.pole[0]), int(args.pole[1]), 1 if args.pole[2] == "+" else -1
    model = Model(args.device)
    dev, f, H = model.dev, model.f, model.H
    doc, pos, val = (t[0, 0 if p > 0 else 1] for t in firings(M, [n]))
    rows = [k for k in range(N_CONTEXTS) if val[k] > 0]
    docs = documents()
    src = model.sources(M, torch.tensor([n] * len(rows), device=dev), torch.full((len(rows),), float(p), device=dev),
                        [docs[int(doc[k])][:int(pos[k]) + 1].long() for k in rows])
    e = entry(M, n, p)
    print(f"L{M} n{n} {'+' if p > 0 else '−'}: {e['fires']}\n  in-edges: {edge_list(e, 'in', 8)}\n"
          f"  mean shares over {len(rows)} firings: embedding {src['emb'].mean():.3f}, heads "
          f"{src['heads'].sum(1).mean():.3f}, neurons {src['neurons'].sum(1).mean():.3f}")
    neuron_share = torch.zeros(M * 2 * f, device=dev).index_add_(0, src["pid"].flatten(), src["neurons"].flatten()) / len(rows)
    head_share = src["heads"].mean(0)
    for q in neuron_share.abs().topk(args.k).indices.tolist():
        l, r = divmod(q, 2 * f)
        pl = 1 if r < f else -1
        print(f"  {neuron_share[q]:+.3f}  L{l} n{r % f} {'+' if pl > 0 else '−'}  {entry(l, r % f, pl)['fires']}")
    for q in head_share.abs().topk(args.k // 2).indices.tolist():
        l, h = divmod(q, H)
        he = head_entry(l, h)
        print(f"  {head_share[q]:+.3f}  L{l}h{h}  attends: {he['attends']} | writes: {he['writes']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("measure", "score", "explain"))
    ap.add_argument("pole", nargs="*", help="explain: layer neuron +|-")
    ap.add_argument("--layers", default=None, help="measure only these layers (a,b,c)")
    ap.add_argument("--k", type=int, default=12, help="explain: neuron sources to list (and half as many heads)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    args = ap.parse_args()
    if arch.is_gemma4(MODEL):
        raise SystemExit("the edges are Qwen3's (16_edges.py)")
    {"measure": measure, "score": score, "explain": explain}[args.what](args)


if __name__ == "__main__":
    main()
