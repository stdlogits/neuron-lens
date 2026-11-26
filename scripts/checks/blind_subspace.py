"""What the output can't read and attention can: the readout's weakest directions in Qwen3-8B (findings §26).

N is the bottom share of the right-singular directions of ``W_U diag(γ_f)`` (the readout, as ``08_reads.py``'s
``null_share``), less the two massive dimensions every norm zeroes (§9). Random subspaces of the same size are the
control throughout.

Parts:
- ``reads``: weights only. The share of each head's query, key and value read rows (after the input norm's scale), and
  of each MLP's gate and up rows, in the bottom 1%; the heads whose choices ``task_circuits.py decide`` traced.
- ``legibility``: the units that build a mover's key at the answer's token (``decide``) against the units that carry the
  answer's identity at the same tokens (the T traces' graphs), each measure as a z-score within its layer.
- ``swap``: at ``task_circuits.py order``'s 64 prompts, the component in N (5% and 10%) of the residual at the answer's
  and the other end's mentions swaps, entering one layer (once: a swap at every later layer would swap back). T, how
  often the answer still beats both other names, and the attention of the six heads from L24 on that attend most to the
  answer unswapped.
- ``identity``: "Repeat this name exactly: X." The component in N at the name's token, entering one layer, is patched
  in from the same prompt with another name Y; how often the model then says Y. Positive control: the whole residual.

Reads ``task_circuits.py``'s outputs (``data/exploration/task_circuits/``); prints its results.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).parent))
import task_circuits as tc  # noqa: E402
from neuron_lens.catalog import neuron_table  # noqa: E402

MASSIVE = [1838, 2276]  # §9


def blind(tr, share: float) -> torch.Tensor:
    """N: the readout's weakest ``share`` of directions, orthonormal, the massive dimensions left out."""
    Wu = tr.lm.lm_head.weight.float() * tr.stack.norm.weight.float()
    _, V = torch.linalg.eigh((Wu.T @ Wu).cpu().double())
    return _basis(V[:, :round(share * tr.d)], tr)


def random_subspace(tr, n: int, seed: int) -> torch.Tensor:
    return _basis(torch.randn(tr.d, n, generator=torch.Generator().manual_seed(seed)), tr)


def _basis(M: torch.Tensor, tr) -> torch.Tensor:
    mass = torch.zeros(tr.d)
    mass[MASSIVE] = 1
    M = M.float() - mass[:, None] * M.float()
    return torch.linalg.qr(M)[0].to(tr.dev)


def chosen_heads() -> list[tuple[int, int]]:
    dec = json.loads((tc.OUT / "decide.json").read_text())
    return sorted({tuple(int(x) for x in v["head"][1:].split(".")[:2]) for v in dec.values()})


# --------------------------------------------------------------------------------------------- reads

def reads(args) -> None:
    tr = tc.Tracer(args.device, labels=False)
    N = blind(tr, 0.01)
    share = lambda W: ((W @ N).pow(2).sum(-1) / W.pow(2).sum(-1))  # noqa: E731  (rows of W)
    print(f"N: {N.shape[1]} directions, a random subspace of that size holds {N.shape[1] / tr.d:.3f}")
    print("layer   query (median, max)   key (median, max)   value (median)   MLP gate and up rows (median)")
    for l in range(0, tr.L, 2):
        lay = tr.stack.layers[l]
        at, g1, g2 = lay.self_attn, lay.input_layernorm.weight, lay.post_attention_layernorm.weight
        per = lambda W, n: torch.stack([((W[h] @ N).pow(2).sum() / W[h].pow(2).sum()) for h in range(n)]).cpu().numpy()  # noqa: E731
        q = per((at.q_proj.weight * g1).view(tr.H, tr.hd, tr.d), tr.H)
        k = per((at.k_proj.weight * g1).view(tr.kv, tr.hd, tr.d), tr.kv)
        v = per((at.v_proj.weight * g1).view(tr.kv, tr.hd, tr.d), tr.kv)
        m = torch.cat([share(lay.mlp.gate_proj.weight * g2), share(lay.mlp.up_proj.weight * g2)]).cpu().numpy()
        print(f"L{l:<4d}  {np.median(q):.3f} {q.max():.3f}          {np.median(k):.3f} {k.max():.3f}         {np.median(v):.3f}            {np.median(m):.3f}")
    print("the heads whose choices were traced:")
    for l, h in chosen_heads():
        at, g1 = tr.stack.layers[l].self_attn, tr.stack.layers[l].input_layernorm.weight
        Wq = (at.q_proj.weight * g1).view(tr.H, tr.hd, tr.d)[h]
        Wk = (at.k_proj.weight * g1).view(tr.kv, tr.hd, tr.d)[h // (tr.H // tr.kv)]
        print(f"  L{l}h{h}: query {((Wq @ N).pow(2).sum() / Wq.pow(2).sum()).item():.3f}  key {((Wk @ N).pow(2).sum() / Wk.pow(2).sum()).item():.3f}")


# --------------------------------------------------------------------------------------------- legibility

def legibility(args) -> None:
    L, f = 36, 12288
    meas = {}
    for l in range(L):
        lab = pq.read_table(neuron_table("labels", l), columns=["fires_conf", "writes_conf"]).to_pydict()
        rd = pq.read_table(neuron_table("reads", l), columns=["null_share", "vocab_kurt", "write_R_z_val"])
        meas[l] = {"fires_conf": np.array(lab["fires_conf"], float), "writes_conf": np.array(lab["writes_conf"], float),
                   "null_share": rd.column("null_share").to_numpy().astype(float),
                   "kurtosis": np.log1p(np.maximum(0, rd.column("vocab_kurt").to_numpy().astype(float))),
                   "top_z": np.array([v[0] if v else np.nan for v in rd.column("write_R_z_val").to_pylist()], float)}
    names = list(meas[0])

    def z(l, n, s):  # within the layer, both poles
        r = n if s > 0 else f + n
        return [(meas[l][k][r] - np.nanmean(meas[l][k])) / np.nanstd(meas[l][k]) for k in names]
    dec = json.loads((tc.OUT / "decide.json").read_text())
    content, choice = set(), set()
    for k, d in dec.items():
        g = json.loads((tc.OUT / "graphs" / f"{k.split('@')[0]}.json").read_text())
        pa = d["pos_a"]
        content |= {(n["layer"], n["neuron"], n["pole"]) for n in g["nodes"] if n["kind"] == "neuron" and n["pos"] == pa}
        for u in d["units"]:
            if u["pos"] == pa and " n" in u["unit"]:
                l, n, s = u["unit"].split(" ")
                choice.add((int(l[1:]), int(n[1:]), 1 if s == "+" else -1))
    A, B = np.array([z(*u) for u in sorted(content)]), np.array([z(*u) for u in sorted(choice)])
    rng = np.random.default_rng(0)
    print(f"at the answer's token: {len(content)} units carrying it (T traces), {len(choice)} building a mover's key on it "
          f"(choice traces), {len(content & choice)} in both; z within layer (null_share: higher is darker)")
    for j, k in enumerate(names):
        a, b = A[:, j][~np.isnan(A[:, j])], B[:, j][~np.isnan(B[:, j])]
        pool, d0 = np.concatenate([a, b]), b.mean() - a.mean()
        p = np.mean([abs((x := rng.permutation(pool))[len(a):].mean() - x[:len(a)].mean()) >= abs(d0) for _ in range(4000)])
        print(f"  {k:11s} identity {a.mean():+.2f}   key {b.mean():+.2f}   difference {d0:+.2f}  (permutation p {p:.3f})")


# --------------------------------------------------------------------------------------------- swap, identity

def _forward(tr, ids, edits: dict, att_layers=()) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The real model, with ``edits`` {layer: fn(x) -> x} applied to the residual entering those layers, and the attention
    of ``att_layers``' heads from the last position."""
    cap = {"A": [None] * tr.L, "z": [None] * tr.L}
    hooks = []
    for l, lay in enumerate(tr.stack.layers):
        lay.self_attn._cap = cap if l in att_layers else None
        if l in edits:
            def hook(m, a, k, fn=edits[l]):
                x = fn((a[0] if a else k["hidden_states"]).clone())
                if a:
                    return (x, *a[1:]), k
                k["hidden_states"] = x
                return a, k
            hooks.append(lay.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        with torch.no_grad():
            lg = tr.lm(input_ids=ids[None].to(tr.dev)).logits[0, -1].float()
    finally:
        for h in hooks:
            h.remove()
        for lay in tr.stack.layers:
            lay.self_attn._cap = None
    A = torch.stack([cap["A"][l][:, -1] for l in att_layers]) if att_layers else None
    return lg, A


def swap(args) -> None:
    tr = tc.Tracer(args.device, labels=False)
    S = {"N205": blind(tr, 0.05), "N410": blind(tr, 0.10)}
    for n in (205, 410):
        for seed in (1, 2):
            S[f"R{n}.{seed}"] = random_subspace(tr, n, seed)
    conds = [("unswapped", None, None)] + [(f"{s} at L{l}", s, l) for s in S for l in (12, 16, 20, 24, 28)]
    late = list(range(24, tr.L))
    res = {c[0]: {"T": [], "win": [], "on_answer": [], "on_other": []} for c in conds}
    for it in json.loads((tc.OUT / "order" / "items.json").read_text()):
        ids, toks, roles = tc.encode(tr.tok, it)
        at = lambda w: [q for q in range(len(toks) - 1) if toks[q].strip() == w.strip() and roles[q] == "q"][0]  # noqa: E731
        pa, pb = at(it["answer"]), at(it["distractor"])
        names = torch.tensor([it["answer_tok"], it["middle_tok"], it["distractor_tok"]])
        movers = None
        for name, sub, l in conds:
            def fn(x, B=S.get(sub)):
                a, b = x[0, pa], x[0, pb]
                ca, cb = B @ (B.T @ a), B @ (B.T @ b)
                x[0, pa], x[0, pb] = a - ca + cb, b - cb + ca
                return x
            lg, A = _forward(tr, ids, {l: fn} if sub else {}, late)
            if movers is None:
                movers = A[:, :, pa].flatten().topk(6).indices
            Af = A.reshape(-1, A.shape[-1])[movers]
            r = res[name]
            r["T"].append((lg[it["answer_tok"]] - lg[it["distractor_tok"]]).item())
            r["win"].append(int(lg[names].argmax()) == 0)
            r["on_answer"].append(Af[:, pa].mean().item())
            r["on_other"].append(Af[:, pb].mean().item())
    print(f"{'swapped':16s} {'T':>7s} {'answer wins':>12s} {'movers on answer':>17s} {'on the other end':>17s}")
    for name, *_ in conds:
        r = res[name]
        print(f"{name:16s} {np.mean(r['T']):+7.2f} {np.mean(r['win']):12.2f} {np.mean(r['on_answer']):17.3f} {np.mean(r['on_other']):17.3f}")


def identity(args) -> None:
    tr = tc.Tracer(args.device, labels=False)
    S = {"N205": blind(tr, 0.05), "N410": blind(tr, 0.10), "R205": random_subspace(tr, 205, 1),
         "R410": random_subspace(tr, 410, 1), "whole residual": None}
    names = [n for trio in tc.ORDER_NAMES for n in trio]
    layers = [12, 16, 20, 24]
    enc = {}
    for n in names:
        it = {"q": f"Repeat this name exactly: {n}.", "instr": " Answer with one word.", "prefill": "Answer:"}
        ids, toks, roles = tc.encode(tr.tok, it)
        enc[n] = (ids, [q for q in range(len(toks)) if toks[q].strip() == n and roles[q] == "q"][0], tc.first_token(tr.tok, " " + n))
    assert len({len(v[0]) for v in enc.values()}) == 1, "the names must tokenize alike"
    cache = {}
    for n in names:  # each name's residual at its token, entering each layer
        ids, pos, _ = enc[n]
        grab = {l: (lambda x, l=l: (cache.__setitem__((n, l), x[0, pos].clone()), x)[1]) for l in layers}
        _forward(tr, ids, grab)
    pairs = list(itertools.permutations(names, 2))[::3]
    print(f"{len(pairs)} pairs: the model asked to repeat X, with Y's component patched in at X's token")
    for sub, B in S.items():
        for l in layers:
            says, T = [], []
            for x, y in pairs:
                ids, pos, tx = enc[x]
                ty, src = enc[y][2], cache[(y, l)]

                def fn(h, B=B, src=src):
                    h[0, pos] = src if B is None else h[0, pos] - B @ (B.T @ h[0, pos]) + B @ (B.T @ src)
                    return h
                lg, _ = _forward(tr, ids, {l: fn})
                T.append((lg[ty] - lg[tx]).item())
                says.append(int(lg.argmax()) == ty)
            print(f"  {sub:15s} at L{l}: Y over X {np.mean(T):+7.2f}, says Y {np.mean(says):.2f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("reads", "legibility", "swap", "identity"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    args = ap.parse_args()
    globals()[args.what](args)


if __name__ == "__main__":
    main()
