"""Qwen3-8B's attention sinks and the dims that carry them (findings §9), on one fixed document set.

- **Masked dims:** the residual dims where the final norm's weight is near zero, and their share of
  the final residual's squared norm (they set the final RMS, so the output's temperature).
- **Secondary sinks:** mid-document positions where L20's residual is massive on dim 2276, their
  tokens, and how much later tokens attend to them against position 0, by layer.
- **Formation:** each early layer's MLP output zeroed at position 0 (and a control position):
  L6's sink neurons there, dim 2276 after L6, attention on position 0 later on, the loss on the
  token predicted at position 0, at position 1, and after.
- **L0's trigger:** L0's MLP output at position 0 replaced by what the same token gets at position 1
  (after a newline); and L0's neurons ranked by attribution (gradient × activation) to dim 2276
  after L6, the top k ablated at position 0, against random ones.

The documents are the first full-length ones of the DCLM, code and chat activation slices
(128 tokens). Eager attention, bf16.

    python scripts/checks/sinks.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_lens.corpus import act_file, lens_docs
from neuron_lens.paths import SUMMARY

MODEL = "Qwen/Qwen3-8B"
SLICES = {"dclm": 64, "code": 32, "chat": 32}
SEQ = 128
SINK_DIM = 2276
SINK_NEURONS = (5723, 8518, 422)  # L6, He's dims
LAYERS = (6, 9, 15, 21, 27, 33)
BATCH = 32


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = []
    for name, k in SLICES.items():
        rows += lens_docs(act_file(name), tok, 4 * k, SEQ, SEQ)[:k]
    ids = torch.tensor(rows)
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.layers
    lines = [f"attention sinks, {MODEL}: {len(ids)} documents of {SEQ} tokens "
             f"({', '.join(f'{k} {v}' for k, v in SLICES.items())}), {len(set(ids[:, 0].tolist()))} distinct first tokens"]

    cap = {}
    layers[6].mlp.down_proj.register_forward_pre_hook(
        lambda m, x: cap.setdefault("a6", []).append(x[0][:, 0, list(SINK_NEURONS)].float().cpu()))
    layers[6].register_forward_hook(
        lambda m, x, o: cap.setdefault("h6", []).append((o[0] if isinstance(o, tuple) else o)[:, 0, SINK_DIM].float().cpu()))
    layers[20].register_forward_hook(
        lambda m, x, o: cap.setdefault("h20", []).append((o[0] if isinstance(o, tuple) else o)[..., SINK_DIM].float().cpu()))
    model.model.norm.register_forward_pre_hook(lambda m, x: cap.setdefault("hf", []).append(x[0].float().cpu()))

    def run(pre=None, post=None, at: list[tuple[int, int]] | None = None) -> dict:
        """One pass with optional hooks on L0-L5; attention on position 0 (and on ``at`` positions)."""
        hooks = [layers[l].mlp.down_proj.register_forward_pre_hook(f) for l, f in (pre or [])]
        hooks += [layers[l].mlp.register_forward_hook(f) for l, f in (post or [])]
        cap.clear()
        att0 = {l: 0.0 for l in range(len(layers))}
        atts = {l: [] for l in range(len(layers))}
        nll = []
        try:
            with torch.no_grad():
                for i in range(0, len(ids), BATCH):
                    b = ids[i:i + BATCH]
                    out = model(b.to(dev), output_attentions=True)
                    lp = torch.log_softmax(out.logits[:, :-1].float(), -1)
                    nll.append(-lp.gather(2, b[:, 1:, None].to(dev)).squeeze(-1).cpu())
                    for l, a in enumerate(out.attentions):
                        att0[l] += a[:, :, 1:, 0].float().mean((1, 2)).sum().item()
                        for d, t in (at or []):
                            if i <= d < i + BATCH:
                                atts[l].append((a[d - i, :, t + 1:, t].float().mean().item(), a[d - i, :, t + 1:, 0].float().mean().item()))
        finally:
            for h in hooks:
                h.remove()
        h6 = torch.cat(cap["h6"])
        return {"a6": torch.cat(cap["a6"]).median(0).values, "h6": h6.median().item(), "sink": (h6.abs() > 2000).float().mean().item(),
                "att0": {l: v / len(ids) for l, v in att0.items()}, "atts": atts, "nll": torch.cat(nll),
                "h20": torch.cat(cap["h20"]), "hf": torch.cat(cap["hf"])}

    clean = run()
    # Masked dims.
    g = model.model.norm.weight.float().cpu()
    masked = (g.abs() < 0.01).nonzero().squeeze(1)
    hf = clean["hf"][:, 1:-1].reshape(-1, g.numel())
    share = hf[:, masked].pow(2).sum(1) / hf.pow(2).sum(1)
    vis = hf.clone()
    vis[:, masked] = 0
    lines += ["", f"masked dims (final norm |γ| < 0.01): {masked.tolist()}, γ {[f'{v:.1e}' for v in g[masked].tolist()]} "
              f"(median |γ| {g.abs().median():.2f})",
              f"  share of the final residual's |h|² in them, positions 1+: q10 {share.quantile(0.1):.2f}, median "
              f"{share.median():.2f}, q90 {share.quantile(0.9):.2f}; values median "
              + ", ".join(f"dim {d} {hf[:, d].median():.0f}" for d in masked.tolist()),
              f"  final RMS median {hf.pow(2).mean(1).sqrt().median():.1f}, without them {vis.pow(2).mean(1).sqrt().median():.1f}"]
    # Secondary sinks.
    d20 = clean["h20"].clone()
    d20[:, 0] = 0
    where = (d20.abs() > 5000).nonzero().tolist()
    lines += ["", f"secondary sinks: {len(where)} mid-document positions with |L20 dim {SINK_DIM}| > 5000, in "
              f"{len({d for d, _ in where})} documents: "
              + ", ".join(f"{tok.decode([ids[d, t].item()])!r}@{t}" for d, t in where)]
    if where:
        sec = run(at=[tuple(x) for x in where])
        lines.append("  attention from later tokens (mean over heads): on the secondary sink | on position 0 | even spread")
        even = sum((1 / torch.arange(t + 2, SEQ + 1).float()).mean().item() for _, t in where) / len(where)
        for l in (3, 6, 9, 12, 18, 24, 30, 33, 35):
            v = sec["atts"][l]
            lines.append(f"  L{l:2d}: {sum(x for x, _ in v) / len(v):.3f} | {sum(y for _, y in v) / len(v):.3f} | {even:.3f}")

    def row(name: str, r: dict) -> str:
        return (f"  {name:34s} {r['a6'][0]:7.0f} {r['a6'][1]:7.0f} {r['a6'][2]:6.0f}  {r['h6']:8.0f}  {r['sink']:4.0%}  "
                + " / ".join(f"{r['att0'][l]:.2f}" for l in LAYERS)
                + f"  {r['nll'][:, 0].mean():.2f} / {r['nll'][:, 1].mean():.2f} / {r['nll'][:, 2:].mean():.3f}")

    def zero_pos(p):  # an MLP's whole output at position p
        return lambda m, x, o: o.index_fill_(1, torch.tensor([p], device=o.device), 0)

    def zero_neurons(idx):  # these neurons at position 0
        def f(m, x):
            x[0][:, 0].index_fill_(1, idx, 0)
        return f
    lines += ["", f"formation: per condition, L6 n{'/n'.join(map(str, SINK_NEURONS))} at position 0 (median), dim {SINK_DIM} after "
              f"L6 (median), share of documents with a sink (|dim| > 2000), attention on position 0 at "
              f"L{'/'.join(map(str, LAYERS))}, NLL of token 1 (predicted at position 0) / token 2 / tokens 3+",
              row("clean", clean)]
    for name, ls, p in ([(f"L{l} MLP zeroed at 0", (l,), 0) for l in range(6)]
                        + [("L2+L3 MLP zeroed at 0", (2, 3), 0), ("L0-L5 MLP zeroed at 0", tuple(range(6)), 0),
                           ("L2+L3 MLP zeroed at 5 (control)", (2, 3), 5), ("L0 MLP zeroed at 5 (control)", (0,), 5)]):
        lines.append(row(name, run(post=[(l, zero_pos(p)) for l in ls])))
    # L0's trigger: the same token's L0 MLP output at position 1.
    pair = torch.stack([torch.full((len(ids),), tok("\n")["input_ids"][0]), ids[:, 0]], 1)
    got = {}
    hk = layers[0].mlp.register_forward_hook(lambda m, x, o: got.__setitem__("m", o[:, 1].clone()))
    with torch.no_grad():
        model.model(pair.to(dev))
    hk.remove()
    state = {"i": 0}

    def patch(m, x, o):
        o[:, 0] = got["m"][state["i"]:state["i"] + len(o)]
        state["i"] += len(o)
        return o
    lines.append(row("L0 MLP at 0 <- same token at 1", run(post=[(0, patch)])))
    # Attribution of dim 2276 after L6 to L0's neurons at position 0.
    class Stop(Exception):
        pass

    attr = []
    for i in range(0, len(ids), BATCH):
        leaf = {}

        def grab(m, x):
            leaf["a"] = x[0].detach().clone().requires_grad_(True)
            return (leaf["a"],)

        def stop(m, x, o):
            leaf["h"] = (o[0] if isinstance(o, tuple) else o)[:, 0, SINK_DIM]
            raise Stop
        hk = layers[0].mlp.down_proj.register_forward_pre_hook(grab)
        hh = layers[6].register_forward_hook(stop)
        try:
            with torch.enable_grad():
                model.model(ids[i:i + BATCH].to(dev))
        except Stop:
            pass
        finally:
            hk.remove()
            hh.remove()
        with torch.enable_grad():
            leaf["h"].float().sum().backward()
        a = leaf["a"]
        attr.append((a.grad[:, 0].float() * a[:, 0].detach().float()).cpu())
    attr = torch.cat(attr)
    med = attr.median(0).values
    order = med.argsort(descending=True)
    cum = med[order].cumsum(0) / med.sum()
    lines.append(f"  L0 attribution to dim {SINK_DIM} after L6: top 10 {cum[9]:.2f}, 100 {cum[99]:.2f}, 300 {cum[299]:.2f} of the "
                 f"median total; top neurons {order[:5].tolist()}")
    for k in (30, 100, 300):
        lines.append(row(f"L0 top {k} by attribution zeroed at 0", run(pre=[(0, zero_neurons(order[:k].to(dev)))])))
    rand = torch.randperm(len(med), generator=torch.Generator().manual_seed(0))[:300].to(dev)
    lines.append(row("L0 300 random zeroed at 0", run(pre=[(0, zero_neurons(rand))])))
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"sinks_{MODEL.split('/')[-1].lower()}.log").write_text(text + "\n")


if __name__ == "__main__":
    main()
