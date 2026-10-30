"""Outside labels of Qwen3-8B neurons (``neuron_lens.known.EXTERNAL``) against our catalog.

For each labelled neuron, both poles unless the source names one: our write read (lens of record,
plain and layer-contrastive), the read side, the activation scale, rates and peak tokens, two
top contexts, and the Neuron Persona Library's label for each sign (its local copy, if present).
With ``--sink``, also a direct check of the attention-sink claim (``he``), which the activation
statistics cannot see because they skip position 0: each layer's MLP energy at token 0 on a few
documents, the share of the largest neurons in it, and where the three named dims rank.

    python scripts/checks/external_crosscheck.py [--lens R] [--sink]
"""
import argparse
import json
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_lens.catalog import entry
from neuron_lens.corpus import lens_docs
from neuron_lens.known import EXTERNAL, SOURCES
from neuron_lens.paths import CORPUS_DIR, LIBRARY, SUMMARY

MODEL = "Qwen/Qwen3-8B"


def sink_check(lines: list[str], n_docs: int = 16) -> None:
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).model.to(dev).eval()
    ids = torch.tensor(lens_docs(CORPUS_DIR / "act-dclm.jsonl", tok, 4 * n_docs, 64, 64)[:n_docs], device=dev)
    energy0, energy = {}, {}
    norms, resid = {}, {}

    def hook(l):
        def f(mod, a):
            x = a[0].float()
            energy0[l] = x[:, 0].pow(2).mean(0).cpu()
            energy[l] = x[:, 1:].pow(2).mean((0, 1)).cpu()
        return f
    # Each layer's output residual, before any norm (hidden_states' last entry is after the final norm).
    hs = [layer.mlp.down_proj.register_forward_pre_hook(hook(l)) for l, layer in enumerate(model.layers)]
    hs += [layer.register_forward_hook(lambda m, a, o, l=l: resid.__setitem__(l, o[0] if isinstance(o, tuple) else o))
           for l, layer in enumerate(model.layers)]
    with torch.inference_mode():
        model(input_ids=ids)
    for h in hs:
        h.remove()
    for l, h in resid.items():
        norms[l] = (h[:, 0].float().norm(dim=-1).median().item(), h[:, 1:].float().norm(dim=-1).median().item())
    lines.append(f"\nattention sink: MLP activation energy (squared activations) at token 0 ({n_docs} documents), by "
                 "layer: total vs a later position, top-3 neurons' share, residual norm after the layer (before the "
                 "final norm) at token 0 vs later")
    for l in range(len(model.layers)):
        e0 = energy0[l]
        top = e0.topk(3)
        share = (top.values / e0.sum()).tolist()
        lines.append(f"  L{l:2d}  {e0.sum():10.1f} vs {energy[l].sum():8.1f}  top "
                     + ", ".join(f"n{i} {s:.2f}" for i, s in zip(top.indices.tolist(), share))
                     + f"  |h| {norms[l][0]:8.1f} vs {norms[l][1]:6.1f}")
    e0 = energy0[6]
    order = e0.argsort(descending=True)
    for n in (8518, 422, 5723):
        rank = int((order == n).nonzero())
        lines.append(f"  L6 n{n}: rank {rank} at token 0, share {e0[n] / e0.sum():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="R", help="the catalog's lens-of-record column name")
    ap.add_argument("--sink", action="store_true")
    args = ap.parse_args()
    library = defaultdict(dict)
    if LIBRARY.exists():
        lib = json.loads(LIBRARY.read_text())
        for r in lib["rows"]:
            row = dict(zip(lib["fields"], r))
            library[(row["L"], row["n"])][row["s"]] = (row["persona"], lib["categories"][row["cat"]], row["a"])
    lines = ["Outside labels against the catalog. Sources and their methods:"]
    lines += [f"  {k}: {v}" for k, v in SOURCES.items()]
    for source, label, layer, neuron, sign in EXTERNAL:
        lines.append(f"\n[{source}] {label}: L{layer} n{neuron} {'+' if sign > 0 else '−' if sign < 0 else '±'}")
        lib = library.get((layer, neuron), {})
        for s in (1, -1):
            if sign and s != sign:
                continue
            e = entry(layer, neuron, s)
            p = "+" if s > 0 else "−"
            got = lib.get(s)
            lines.append(f"  {p} library: {got[0]} ({got[1]}, strength {got[2]})" if got else f"  {p} library: —")
            lines.append(f"  {p} writes:  {e.get(f'write_{args.lens}_str')}")
            lines.append(f"  {p} z:       {e.get(f'write_{args.lens}_z_str')}")
            lines.append(f"  {p} reads:   {e.get(f'read_{args.lens}_str')}")
            if "act_top" in e:
                rates = " ".join(f"{k.removeprefix('rate_')} {v:.1e}" for k, v in e.items() if k.startswith("rate_"))
                lines.append(f"  {p} act:     top {e['act_top']:.2f}, max {e['act_max']:.2f}, rate {e['rate']:.1e}; {rates}")
                lines.append(f"  {p} peaks:   {e['peak_tokens']}")
                for c in str(e["contexts"]).split(" ‖ ")[:3]:
                    lines.append(f"  {p} context: {c}")
    if args.sink:
        sink_check(lines)
    text = "\n".join(lines)
    print(text)
    (SUMMARY / "external_crosscheck_qwen3-8b.log").write_text(text + "\n")


if __name__ == "__main__":
    main()
