"""Which read direction do the poles on SiLU's negative lobe have (findings §17)?

For the active poles with ``neg_lobe`` past a half, three weight-only directions through the S-lens: ``â ± b̂``, its
negative, and the up row alone, ``∓b̂``. Two scores against the tokens a pole actually fires on (``acts``' ``peak_tokens``):
the percentile of its most frequent one among the kept tokens (50 is chance), and whether any of them is in the
read's top 64, raw and z-scored across the layer's poles (every other pole as in ``neurons.read_rows``).

    python scripts/checks/lobe_reads.py --layers 0,1,2,3,4 [--device mps]
"""
import argparse
import ast
import re

import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import MODEL, TOP, kept, neuron_table, tokenizer
from neuron_lens.neurons import directions
from neuron_lens.readout import s_tops
from neuron_lens.weights import tensors

SILENT = 0.05  # act_top under this: near-silent, as 10_labels.py
SAMPLE = 1500  # poles per layer for the percentile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True, help="comma-separated")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
    args = ap.parse_args()
    dev = args.device
    keep = kept()
    nm = arch.names(MODEL)
    emb = tensors(MODEL, [nm["embed"]])[nm["embed"]][keep].float() * arch.embed_scale(MODEL)
    emb = (emb - emb.mean(0)).to(dev, torch.float16)
    K_mid = arch.lens(MODEL, "S_mid", arch.fits(MODEL)[1])
    tok = tokenizer()
    index = {}
    for i, t in enumerate(keep.tolist()):
        index.setdefault(tok.decode([t], skip_special_tokens=True).strip().lower(), []).append(i)

    def peaks(s: str) -> list[list[int]]:
        out = []
        for p in str(s).split(" | "):
            m = re.match(r"^(.*)×(\d+)$", p.strip())
            if m:
                try:
                    out.append(index.get(ast.literal_eval(m[1]).strip().lower(), []))
                except (SyntaxError, ValueError):
                    pass
        return [x for x in out if x]

    for l in (int(x) for x in args.layers.split(",")):
        t = pq.read_table(neuron_table("acts", l), columns=["neg_lobe", "act_top", "peak_tokens"])
        lobe = torch.tensor(t["neg_lobe"].to_pylist()) > 0.5
        top = torch.tensor(t["act_top"].to_pylist())
        P = [peaks(p) for p in t["peak_tokens"].to_pylist()]
        flip = [i for i in lobe.nonzero()[:, 0].tolist() if top[i] >= SILENT and P[i]]
        w = directions(MODEL, l)
        a, b = w["gate"].float().to(dev), w["up"].float().to(dev)
        a, b = a / a.norm(dim=1, keepdim=True), b / b.norm(dim=1, keepdim=True)
        main = torch.cat([a + b, a - b])
        K = K_mid[l].float().to(dev)
        sample = torch.tensor(flip)[torch.randperm(len(flip), generator=torch.Generator().manual_seed(l))[:SAMPLE]]
        first = torch.tensor([P[i][0][0] for i in sample.tolist()], device=dev)
        line = f"L{l:<3d} {len(flip):6d} poles"
        for name, neg in (("â±b̂", main), ("−(â±b̂)", -main), ("up", torch.cat([-b, b]))):
            cs = torch.where(lobe.to(dev)[:, None], neg, main) @ K
            cs = cs / cs.norm(dim=1, keepdim=True)
            st = s_tops(cs, emb, (1,), TOP)
            hit = lambda kind: sum(bool({j for x in P[i] for j in x} & set(st[kind, 1][0][i].tolist())) for i in flip) / len(flip)
            sc = neg[sample.to(dev)] @ K
            sc = (sc / sc.norm(dim=1, keepdim=True)).float() @ emb.float().T
            pct = (sc < sc.gather(1, first[:, None])).float().mean(1).mean().item()
            line += f" | {name}: {100 * pct:.0f}, top {TOP} {hit('raw'):.2f} raw, {hit('z'):.2f} z"
        print(line, flush=True)


if __name__ == "__main__":
    main()
