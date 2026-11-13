"""Does splitting a pole's write by what it reads split its causal meaning? (findings §18)

``measure`` (Qwen3-8B, MPS, fp32): §3's random neurons (``causal_lens.random_neurons``), 16 a layer at even layers L4-L34, both
poles. Steering by §3's protocol at 1× (the vector at positions 16+ of 16 documents, ``alpha`` the pole's ``act_top``) with the whole
write, the part along the read direction ``â ± b̂``, the part orthogonal to it, and the part orthogonal to the whole gate-up plane
(``neurons.orthogonal_write``). And removing the firing's whole contribution or either first two parts at up to 8 top firings (clean − removed
log-probs, summed over the firing and the next 8 positions). Appends a JSON line per pole to
``data/summary/write_parts/<model>.jsonl``; resumes by skipping poles written.

``score``: by depth band, the effect sizes and correlations, and the blind ratings of ``data/summary/write_parts/``: for each
comparison a shuffled pair file (``<name>_pairs.jsonl``, what the rater saw), its key (``key_<name>.json``) and the rater's letters
(``<name>_ratings.json``: S same, R related, D different, U unclear). Writes ``data/summary/write_parts_<model>.log``.

    python scripts/checks/write_parts.py measure
    python scripts/checks/write_parts.py score
"""
import argparse
import collections
import json
import statistics as st
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from causal_lens import random_neurons  # noqa: E402

from neuron_lens import arch  # noqa: E402
from neuron_lens.catalog import MODEL, column, documents, kept, neuron_table, vocab  # noqa: E402
from neuron_lens import metric  # noqa: E402
from neuron_lens.neurons import directions, orthogonal_write  # noqa: E402
from neuron_lens.paths import DATA, SUMMARY  # noqa: E402
from neuron_lens.readout import rms_norm_columns  # noqa: E402
from neuron_lens.stats import pearson  # noqa: E402

STEM = arch.stem(MODEL)
OUT = SUMMARY / "write_parts"
ORDER = [20, 12, 28, 8, 16, 24, 32, 4, 6, 10, 14, 18, 22, 26, 30, 34]
PER_LAYER, FIRST, WINDOW, FIRINGS, TOP_TOKENS = 16, 16, 8, 8, 24
BANDS = [("L4-8", 4, 8), ("L10-12", 10, 12), ("L14-20", 14, 20), ("L22-26", 22, 26), ("L28-34", 28, 34)]


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def measure(args) -> None:
    dev = torch.device(args.device)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    layers = lm.model.layers
    fit = arch.fits(MODEL)[0]
    J_out, J_mid = arch.lens(MODEL, "R_out", fit).to(dev, torch.float32), arch.lens(MODEL, "R_mid", fit).to(dev, torch.float32)
    gamma = lm.model.norm.weight.detach().float()
    keep = kept().to(dev)
    U = lm.lm_head.weight.detach().float()[keep]
    V = vocab()
    docs = documents()
    saved = torch.load(DATA / "causal" / f"{STEM}-causal.pt", weights_only=False)
    picks = saved["picks"]
    rand = set(random_neurons(36, 12288, saved["meta"]["per_layer"]))
    ids = saved["contexts"].to(dev)
    state = {"L": None, "fn": None}
    cap = {}
    for L in ORDER:
        layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, i, L=L: cap.__setitem__("a", i[0].detach()) if state["L"] == L else None)
        layers[L].mlp.register_forward_hook(lambda m, i, o, L=L: o + state["fn"](o) if state["L"] == L and state["fn"] is not None else o)
    lens = lambda v, J: (U @ rms_norm_columns(J @ v.T, gamma)).T
    r = lambda a, b: pearson(a, b).item()
    strs = lambda t, k=TOP_TOKENS: [V[keep[j]].strip() or "␣" for j in t.topk(k).indices.tolist()]

    @torch.no_grad()
    def steered(fn):
        state["fn"] = fn
        out = torch.log_softmax(lm(input_ids=ids, use_cache=False).logits[:, FIRST:].float(), -1).mean((0, 1))[keep]
        state["fn"] = None
        return out

    def add(v, a):
        def f(o):
            d = torch.zeros_like(o)
            d[:, FIRST:] = a * v
            return d
        return f

    OUT.mkdir(exist_ok=True)
    path = OUT / f"{STEM}.jsonl"
    done = {tuple(json.loads(line)[k] for k in ("layer", "neuron", "pole")) for line in path.read_text().splitlines()} if path.exists() else set()
    out, rng, t0 = path.open("a"), __import__("random").Random(1), time.time()
    with torch.no_grad():
        base = steered(None)
    for L in ORDER:
        idx = sorted(rng.sample([i for i, p in enumerate(picks) if p[0] == L and p in rand], PER_LAYER))
        d = directions(MODEL, L, dev)
        s2 = metric.s2(MODEL, "mid", L, dev)  # as neurons.poles
        gn, un = (d[k] / metric.cov_norm(d[k], s2) for k in ("gate", "up"))
        W = d["write"]
        ns = torch.tensor([picks[i][1] for i in idx], device=dev)
        plane = orthogonal_write(W, gn, un, s2)
        state["L"] = L
        rd = {pi: lens(unit(metric.vector(gn[ns] + s * un[ns], s2)), J_mid[L]) for pi, s in enumerate((1.0, -1.0))}
        t = pq.read_table(neuron_table("reads", L), columns=["read_R_ids"])
        for j, i in enumerate(idx):
            n = picks[i][1]
            for pi, s in enumerate((1.0, -1.0)):
                pole = 1 if s > 0 else -1
                alpha = saved["alpha"][i, pi].item()
                if (L, n, pole) in done or alpha <= 0:
                    continue
                w, c = W[:, n], unit(gn[n] + s * un[n])
                par = c * (w @ c)
                perp = w - par
                row = {"layer": L, "neuron": n, "pole": pole, "alpha": alpha, "cos": (s * unit(w) @ c).item(),
                       "read_R": [V[k].strip() or "␣" for k in column(t, "read_R_ids", 12)[pi * 12288 + n].tolist()]}
                with torch.no_grad():
                    E = {name: steered(add(s * v, alpha)) - base for name, v in (("whole", w), ("along", par), ("orth", perp), ("plane", plane[:, n]))}
                    Rd, Rc = rd[pi][j] - rd[pi].mean(0), rd[pi][(j + 1) % len(idx)] - rd[pi].mean(0)
                    row["steer"] = {"r_additive": r(E["whole"], E["along"] + E["orth"]), "r_whole_along": r(E["whole"], E["along"]),
                                    "r_whole_orth": r(E["whole"], E["orth"]), "r_along_orth": r(E["along"], E["orth"]),
                                    "size_along": (E["along"].norm() / E["whole"].norm()).item(), "size_orth": (E["orth"].norm() / E["whole"].norm()).item(),
                                    "read_along": r(E["along"], Rd), "read_orth": r(E["orth"], Rd), "read_along_ctrl": r(E["along"], Rc), "read_orth_ctrl": r(E["orth"], Rc),
                                    "top_whole": strs(E["whole"]), "top_along": strs(E["along"]), "top_orth": strs(E["orth"]), "top_plane": strs(E["plane"])}
                    tot, used = [0, 0, 0], 0
                    for f in range(FIRINGS):
                        if saved["fired"][i, pi, f].item() <= 0:
                            continue
                        dc, p = saved["top_doc"][i, pi, f].item(), saved["top_pos"][i, pi, f].item()
                        x = docs[dc][:p + WINDOW + 1][None].to(dev).repeat(4, 1)
                        def rm(o, n=n, p=p):
                            a = cap["a"][0, p, n].float()
                            z = torch.zeros_like(o)
                            if (a * s).item() > 0:
                                z[1, p], z[2, p], z[3, p] = -a * w, -a * par, -a * perp
                            return z
                        state["fn"] = rm
                        lp = torch.log_softmax(lm(input_ids=x, use_cache=False).logits[:, p:].float(), -1)
                        state["fn"] = None
                        if (cap["a"][0, p, n].float() * s).item() <= 0:
                            continue
                        tot = [a + b for a, b in zip(tot, (lp[0] - lp[1:]).sum(1)[:, keep])]
                        used += 1
                    row["firings"] = used
                    if used:
                        F = {"whole": tot[0] / used, "along": tot[1] / used, "orth": tot[2] / used}
                        row["fire"] = {"size_along": (F["along"].norm() / F["whole"].norm()).item(), "size_orth": (F["orth"].norm() / F["whole"].norm()).item(),
                                       "r_whole_orth": r(F["whole"], F["orth"]), "top_along": strs(F["along"]), "top_orth": strs(F["orth"])}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
        print(f"layer {L} done {time.time() - t0:.0f}s", flush=True)


def score(_args) -> None:
    rows = [json.loads(line) for line in (OUT / f"{STEM}.jsonl").read_text().splitlines()]
    lines = [f"write parts, {MODEL}: {len(rows)} poles"]
    med = st.median
    for name, lo, hi in BANDS:
        sub = [r for r in rows if lo <= r["layer"] <= hi]
        g = lambda k: med([r["steer"][k] for r in sub])
        lines.append(f"{name:7s} {len(sub):3d} poles: cos {med([r['cos'] for r in sub]):+.2f}, |along|/|whole| {g('size_along'):.2f}, |orth|/|whole| {g('size_orth'):.2f}, "
                     f"r(whole, orth) {g('r_whole_orth'):+.2f}, r(along, orth) {g('r_along_orth'):+.2f}")
    for job, title in (("C_read", "read vs orthogonal to read direction (steering, rater 1)"), ("A_steer", "along vs orthogonal (steering)"), ("B_fire", "along vs orthogonal (firing)")):
        key, rat = json.loads((OUT / f"key_{job}.json").read_text()), json.loads((OUT / f"{job}_ratings.json").read_text())
        lines += ["", title]
        for tag, f in (("L4-20", lambda k: key[k]["layer"] <= 20), ("L22-34", lambda k: key[k]["layer"] >= 22)):
            for kind in ("real", "shuffled"):
                c = collections.Counter(rat[k] for k in key if key[k]["kind"] == kind and f(k))
                lines.append(f"  {tag:6s} {kind:8s} S {c['S']:3d} R {c['R']:3d} D {c['D']:3d} U {c['U']:3d}")
    key, rat = json.loads((OUT / "key_D.json").read_text()), json.loads((OUT / "D_ratings.json").read_text())
    lines += ["", "read vs orthogonal to read direction, and to the gate-up plane (one rater over both, rater 2)"]
    for kind in ("read_orth", "plane", "shuffled"):
        for tag, f in (("L4-20", lambda k: key[k]["layer"] <= 20), ("L22-34", lambda k: key[k]["layer"] >= 22)):
            c = collections.Counter(rat[k] for k in key if key[k]["kind"] == kind and f(k))
            lines.append(f"  {kind:9s} {tag:6s} S {c['S']:3d} R {c['R']:3d} D {c['D']:3d} U {c['U']:3d}")
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"write_parts_{STEM}.log").write_text(text + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("measure", "score"))
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    {"measure": measure, "score": score}[args.what](args)


if __name__ == "__main__":
    main()
