"""One mixed unit traced (findings §23): Qwen3-32B's L55 n11703, a Colorado neuron that also fires on code using the
Tornado web framework, through its parent in Qwen3-14B. Qwen3-14B (the catalog's model for this check), bf16 on CUDA
or MPS for the activation parts, the released R-lens and S-lens; Qwen3-32B's weights for ``lineage``.

The hand-written texts are in ``colorado_tornado_texts.json``. A unit's activation is its coordinate of ``down_proj``'s
input, signed by its pole and given as a share of its ``act_top``; a text's value is its peak over positions from 1.

Parts (each writes ``data/summary/colorado_tornado_<part>.log``):

- ``lineage`` (weights only): the 32B neuron's gate row, up row and down column against every 14B neuron of every
  layer; for the copied layers, the same-index cosines of a sample of the layer's neurons against a shuffled-index null;
  the 32B's neurons past the 14B's width against their best 14B match.
- ``probe``: the chain's units on the texts (Colorado, highland, plateau and steppe prose; Tornado code and prose; other
  web libraries; MongoDB code; storm text; Chinese-commented code).
- ``files``: the same units on the code slice's files that import Tornado, against files of other web libraries and random
  code (first 512 tokens), every catalog pole whose reads or drivers name Colorado on them by layer, and how often each
  library's files hold Chinese text.
- ``split``: the exact split of an activation at one firing over the embedding, each head and each earlier neuron
  (half rule, the norm's rms held; Qwen3 is pre-norm, so the MLP input is their sum), and the R-lens read of the MLP input
  there, for the Colorado neuron and the L14 Colorado-context unit on Tornado code and Colorado prose.
- ``wiring``: weight-only reads of the chain. Ancestry through the ``edges`` tables (each pole spreads unit weight over its
  in-edges by |push|, multiplied along paths of up to four hops); full-vocabulary ranks of Tornado tokens in the S-lens
  read, z-scored against 3,000 poles of the layer, and in a read pulled back through the pole's own in-edges; the same
  reads with the layer's top 8-512 read directions (PCs) removed, or the read's own top 8-32 R-lens tokens removed one at
  a time (matching pursuit); and each of the bridge's top positive in-edges checked by running the source's own five top
  contexts and reading the bridge's peak there, against random poles' contexts.
- ``mongo``: what ties the Tornado units to the bridge. The output tokens their writes and the bridge's read share
  (R-lens, z-scored, ``zA·zB`` where both are positive); token directions at the bridge's MLP input,
  ``u_t = J_midᵀ (γ ⊙ W_U[t])``, and their cosine with its read; the edge cosine with each source's write projected off
  the span of a token group's directions; the bridge's activation with its MLP input projected off a group's span; the
  edges from MongoDB-detecting units against random poles of their layer; and where MongoDB tokens rank in the bridge's
  read and other readers', against 4,096 random poles, and after removing its weight inputs' writes from the read.

    python scripts/checks/colorado_tornado.py lineage|probe|files|split|wiring|mongo
"""
import argparse
import collections
import json
import os
import random
import re
from pathlib import Path

os.environ.setdefault("NEURON_LENS_MODEL", "Qwen/Qwen3-14B")
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors import safe_open  # noqa: E402

from neuron_lens import arch, metric  # noqa: E402
from neuron_lens.catalog import MODEL, documents, entry, head_entry, kept, lobe_signs, neuron_table, tokenizer  # noqa: E402
from neuron_lens.lenses import load_lens  # noqa: E402
from neuron_lens.neurons import poles  # noqa: E402
from neuron_lens.paths import DATA, SUMMARY  # noqa: E402
from neuron_lens.readout import read  # noqa: E402
from neuron_lens.weights import tensors  # noqa: E402

assert MODEL == "Qwen/Qwen3-14B", "this check is about Qwen3-14B's units"
DEV = "cuda" if torch.cuda.is_available() else "mps"
TEXTS = json.loads(Path(__file__).with_name("colorado_tornado_texts.json").read_text())
COLORADO, BRIDGE = (31, 11703, 1), (11, 6575, -1)
TORNADO = [(6, 7200, -1), (7, 12137, -1)]
CHAIN = [(13, 11715, -1), (14, 489, -1), (17, 4541, -1), (21, 15023, 1), (23, 2209, -1), (29, 10886, 1)]
CODE = DATA / "corpus" / "act-code.jsonl"
TORNADO_FILE = re.compile(r"import tornado|from tornado")
WEB_FILE = re.compile(r"(import|from) (aiohttp|twisted|flask|django|requests)")
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def name(u):
    return f"L{u[0]} n{u[1]} {'+' if u[2] > 0 else '−'}"


def model():
    from transformers import AutoModelForCausalLM
    torch.set_grad_enabled(False)
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEV).eval().requires_grad_(False)


class Probe:
    """Pole-signed peaks of units over texts, as shares of their act_top."""

    def __init__(self, lm, units):
        self.lm, self.units, self.cap = lm, units, {}
        self.top = {u: entry(*u)["act_top"] for u in units}
        for l in sorted({u[0] for u in units}):
            lm.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: self.cap.__setitem__(l, a[0][0].float().cpu()))

    def __call__(self, ids):
        self.lm(ids[None].to(DEV))
        out = []
        for l, n, p in self.units:
            a = self.cap[l][1:, n] * p
            out.append((float(a.max()) / self.top[(l, n, p)], int(a.argmax()) + 1))
        return out


def lineage(args):
    def opener(repo):
        idx = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))["weight_map"]
        return lambda k: safe_open(hf_hub_download(repo, idx[k]), "pt").get_tensor(k).float()
    g32, g14 = opener("Qwen/Qwen3-32B"), opener(MODEL)
    gn = lambda x: x / x.norm(dim=-1, keepdim=True)
    def mats(get, L):
        p = f"model.layers.{L}.mlp."
        return gn(get(p + "gate_proj.weight")), gn(get(p + "up_proj.weight")), gn(get(p + "down_proj.weight").T)
    f14 = arch.text_config(MODEL).intermediate_size
    G, U, D = mats(g32, 55)
    say("32B L55 n11703 against every 14B layer: the best gate-row match, and the same index")
    for L in range(arch.text_config(MODEL).num_hidden_layers):
        g, u, d = mats(g14, L)
        cg = g @ G[11703]
        i = int(cg.abs().argmax())
        say(f"  14B L{L:2d}: best n{i} gate {cg[i]:+.3f} up {float(u[i] @ U[11703]):+.3f} down {float(d[i] @ D[11703]):+.3f} | "
            f"n11703 gate {float(g[11703] @ G[11703]):+.3f} up {float(u[11703] @ U[11703]):+.3f} down {float(d[11703] @ D[11703]):+.3f}")
    say("\ncopied layers: same-index cosines (a sample of every 37th neuron) against a shuffled-index null")
    S = torch.arange(0, f14, 37)
    perm = S[torch.randperm(len(S), generator=torch.Generator().manual_seed(0))]
    for L32, L14, n in [(11, 11, 6575), (35, 11, 6575), (31, 31, 11703), (55, 31, 11703)]:
        G2, U2, D2 = mats(g32, L32)
        g, u, d = mats(g14, L14)
        same = lambda A, B: (A[S] * B[S]).sum(-1).median()
        say(f"  32B L{L32} vs 14B L{L14}: n{n} gate {float(G2[n] @ g[n]):+.3f} up {float(U2[n] @ u[n]):+.3f} down {float(D2[n] @ d[n]):+.3f}"
            f" | layer medians gate {same(G2, g):.3f} up {same(U2, u):.3f} down {same(D2, d):.3f} | null |gate| {(G2[S] * g[perm]).sum(-1).abs().median():.3f}")
    G2, _, _ = mats(g32, 55)
    g, _, _ = mats(g14, 31)
    best = (G2[f14::40] @ g.T).abs().max(1).values
    say(f"\n32B L55 neurons past {f14}: best |cos| to any 14B L31 gate row, median {best.median():.3f}, max {best.max():.3f} (n={len(best)})")


def probe(args):
    lm = model()
    tok = tokenizer()
    units = TORNADO + [BRIDGE] + CHAIN[:2] + [CHAIN[4], COLORADO, (6, 7831, 1), (7, 5551, 1), (5, 3686, -1), (2, 62, 1), (3, 3032, 1), (5, 6381, -1)]
    P = Probe(lm, units)
    for u in units:
        e = entry(*u)
        say(f"{name(u):14s} act_top {e['act_top']:6.1f}  peaks {str(e['peak_tokens'])[:70]} | write_R {str(e['write_R_str'])[:50]}")
    say("\n" + f"{'':22s}" + "".join(f"{name(u):>14s}" for u in units))
    for k, t in TEXTS.items():
        r = P(tok(t, return_tensors="pt").input_ids[0])
        say(f"{k:22s}" + "".join(f"{v:14.2f}" for v, _ in r))


def files(args):
    lm = model()
    tok = tokenizer()
    docs = [json.loads(l)["text"] for l in open(CODE)]
    rng = random.Random(0)
    tor = [d for d in docs if TORNADO_FILE.search(d)]
    web = [d for d in docs if WEB_FILE.search(d) and not TORNADO_FILE.search(d)][:len(tor)]
    rnd = rng.sample([d for d in docs if not TORNADO_FILE.search(d)], len(tor))
    col = []  # catalog poles naming Colorado on two of their read_R, write_R and drivers tops
    for L in range(arch.text_config(MODEL).num_hidden_layers):
        r = pq.read_table(neuron_table("reads", L), columns=["neuron", "pole", "read_R_str", "write_R_str"]).to_pylist()
        d = pq.read_table(neuron_table("drivers", L), columns=["drivers_str"]).to_pylist()
        a = pq.read_table(neuron_table("acts", L), columns=["act_top"]).to_pylist()
        for x, y, z in zip(r, d, a):
            tops = [[s.strip() for s in (c or "").split(" | ")[:3]] for c in (x["read_R_str"], x["write_R_str"], y["drivers_str"])]
            if z["act_top"] >= 1 and sum(any(t in ("Colorado", "Denver", "Boulder") for t in top) for top in tops) >= 2:
                col.append((L, x["neuron"], x["pole"]))
    units = sorted(set(TORNADO + [BRIDGE] + CHAIN + [COLORADO] + col))
    P = Probe(lm, units)
    def run(group):
        return torch.tensor([[v for v, _ in P(tok(d, return_tensors="pt").input_ids[0, :512])] for d in group])
    T, C, R = run(tor), run(web), run(rnd)
    K = torch.tensor([[v for v, _ in P(tok(TEXTS[k], return_tensors="pt").input_ids[0])] for k in ("colorado_news", "colorado_hike")])
    say(f"{len(tor)} files import Tornado; {len(web)} files of other web libraries and {len(rnd)} random code files as controls")
    say(f"{'unit':16s} {'tornado med':>11s} {'max':>5s} {'≥0.3':>5s} {'web ctrl':>8s} {'random':>7s} {'colorado':>8s}  peaks | write_R")
    for i, u in enumerate(units):
        e = entry(*u)
        say(f"{name(u):16s} {T[:, i].median():11.2f} {T[:, i].max():5.2f} {(T[:, i] >= 0.3).float().mean():5.2f} {C[:, i].median():8.2f} "
            f"{R[:, i].median():7.2f} {K[:, i].mean():8.2f}  {str(e['peak_tokens'])[:50]} | {str(e['write_R_str'])[:40]}")
    cjk = re.compile(r"[一-鿿]")
    say("\nfiles holding Chinese text (≥ 5 CJK characters)")
    for lib, pat in [("tornado", r"(import|from) tornado"), ("flask", r"(import|from) flask"), ("django", r"(import|from) django"),
                     ("twisted", r"(import|from) twisted"), ("requests", r"^\s*(import|from) requests"), ("MongoDB", r"pymongo|motor\.motor|mongoose|mongodb://|MongoClient"),
                     ("any python", r"^\s*(import|from) \w+")]:
        rx = re.compile(pat, re.M)
        hit = [d for d in docs if rx.search(d)]
        say(f"  {lib:10s} {len(hit):5d} files, {sum(len(cjk.findall(d)) >= 5 for d in hit):3d} with Chinese")


def split(args):
    lm = model()
    tok = tokenizer()
    layers, H, hd = lm.model.layers, lm.config.num_attention_heads, lm.config.head_dim
    W_U, gfin = lm.lm_head.weight, lm.model.norm.weight.float()
    for (M, N, P), names in [(COLORADO, ["tornado_httpclient", "tornado_web", "colorado_news", "colorado_hike"]),
                             (CHAIN[1], ["tornado_httpclient", "tornado_web", "colorado_news"])]:
        J = load_lens("hub@mid", MODEL)[M]
        lay, cap = layers[M], {}
        def pre(key):
            def h(m, a):
                cap[key] = a[0][0].float()
            return h
        hooks = [layers[l].mlp.down_proj.register_forward_pre_hook(pre(("a", l))) for l in range(M + 1)]
        hooks += [layers[l].self_attn.o_proj.register_forward_pre_hook(pre(("z", l))) for l in range(M + 1)]
        hooks += [lay.post_attention_layernorm.register_forward_pre_hook(pre("x")),
                  lm.model.embed_tokens.register_forward_hook(lambda m, a, o: cap.__setitem__("e", o[0].float()))]
        class Stop(Exception):
            pass
        def stop(m, a, o):
            raise Stop
        hooks.append(lay.mlp.register_forward_hook(stop))
        gam = lay.post_attention_layernorm.weight.float()
        wg, wu = lay.mlp.gate_proj.weight[N].float(), lay.mlp.up_proj.weight[N].float()
        for nm_ in names:
            ids = tok(TEXTS[nm_], return_tensors="pt").input_ids[0]
            try:
                lm(ids[None].to(DEV))
            except Stop:
                pass
            a_all = cap[("a", M)][:, N] * P
            a_all[0] = -1e9
            t = int(a_all.argmax())
            s = [tok.decode([i]) for i in ids.tolist()]
            x = cap["x"][t]
            rms = x.pow(2).mean().add(1e-6).rsqrt()
            rg, ru = gam * wg * rms, gam * wu * rms
            g, u = rg @ x, ru @ x
            sc = torch.nn.functional.silu(g) / g
            a = sc * g * u
            contrib = lambda v: 0.5 * sc * (g * (v @ ru) + u * (v @ rg)) / a
            say(f"\n=== {name((M, N, P))} on {nm_}: a = {a:.1f} at {s[t]!r} (…{''.join(s[max(0, t - 10):t + 1])!r})")
            pr, ix = read(x[None].T, W_U, gfin, J, k=10)
            say("  R-lens read of the MLP input: " + " | ".join(f"{tok.decode([j])!r} {q:.3f}" for q, j in zip(pr[0].tolist(), ix[0].tolist())))
            neur = torch.stack([cap[("a", l)][t] * contrib(layers[l].mlp.down_proj.weight.float().T) for l in range(M)])
            head = torch.stack([contrib(torch.einsum("dhk,hk->hd", layers[l].self_attn.o_proj.weight.float().view(-1, H, hd), cap[("z", l)][t].view(H, hd)))
                                for l in range(M + 1)])
            say(f"  shares: embedding {contrib(cap['e'][t]):.3f}, heads {head.sum():.3f}, neurons {neur.sum():.3f}")
            flat = neur.flatten()
            for q in flat.abs().topk(10).indices.tolist():
                l, n = divmod(q, neur.shape[1])
                act = float(cap[("a", l)][t, n])
                e = entry(l, n, 1 if act > 0 else -1)
                say(f"   {flat[q]:+.3f}  {name((l, n, 1 if act > 0 else -1))} act {act:+.1f}/{e['act_top']:.1f}  peaks {str(e['peak_tokens'])[:60]} | write_R {str(e['write_R_str'])[:40]}")
            hf = head.flatten()
            for q in hf.abs().topk(5).indices.tolist():
                l, h = divmod(q, H)
                say(f"   {hf[q]:+.3f}  L{l}h{h}  ov_R {str(head_entry(l, h).get('ov_R_str', ''))[:70]}")
        for h in hooks:
            h.remove()


def _ins(cache, l, n, p):
    if l not in cache:
        t = pq.read_table(neuron_table("edges", l), columns=["neuron", "pole", "in_layer", "in_neuron", "in_pole", "in_push"]).to_pydict()
        cache[l] = {(nn, pp): list(zip(a, b, c, d)) for nn, pp, a, b, c, d in zip(t["neuron"], t["pole"], t["in_layer"], t["in_neuron"], t["in_pole"], t["in_push"])}
    rows = [(a, b, c, d) for a, b, c, d in cache[l][(n, p)] if a >= 0 and d == d]
    tot = sum(abs(d) for *_, d in rows) or 1
    return [((a, b, c), d / tot) for a, b, c, d in rows]


def wiring(args):
    tok = tokenizer()
    keep = kept()
    strs = [tok.decode([int(i)]) for i in keep]
    say("1. weight-edge ancestry: rank of the Tornado-library units among each unit's ancestors (four hops, beam 400)")
    cache = {}
    for target in (BRIDGE, CHAIN[1], COLORADO):
        front, anc, hop = {target: 1.0}, collections.defaultdict(float), {}
        for h in range(1, 5):
            nxt = collections.defaultdict(float)
            for u, w in front.items():
                if u[0] > 0:
                    for v, s in _ins(cache, *u):
                        nxt[v] += w * s
            for v, w in nxt.items():
                anc[v] += w
                hop.setdefault(v, h)
            front = dict(sorted(nxt.items(), key=lambda kv: -abs(kv[1]))[:400])
        early = [u for u in sorted(anc, key=lambda v: -abs(anc[v])) if u[0] <= 10]
        say(f"  {name(target)}: {len(anc)} ancestors; " + "; ".join(
            f"{name(u)} #{early.index(u) + 1 if u in early else '—'} of L0-L10 (hop {hop.get(u, '—')})" for u in TORNADO + [(6, 7831, 1)]))
        say("     top early ancestors: " + " | ".join(f"{name(u)} [{str(entry(*u)['drivers_str'])[:30]}]" for u in early[:6]))
    nm = arch.names(MODEL)
    hd = tensors(MODEL, sorted({nm["unembed"], nm["norm"], nm["embed"]}))
    emb = hd[nm["embed"]][keep].float()
    emb = (emb - emb.mean(0)).to(DEV)
    W_U, gam = hd[nm["unembed"]][keep].to(DEV, torch.float32), hd[nm["norm"]].to(DEV, torch.float32)
    K = arch.lens_means(MODEL, parts=("K_mid",))["K_mid"]
    Jm = load_lens("hub@mid", MODEL)
    watch = [i for i, s in enumerate(strs) if re.search(r"ornado|ioloop|IOLoop|RequestHandler|httpclient|HTTPClient|coroutine|websocket|_secure|\.settings|\.cookies|_FEED|StringField|xsrf", s)]
    rread = lambda v, J: W_U @ ((lambda t: t / t.norm())(J.to(DEV).float() @ v.to(DEV).float()) * gam)
    sread = lambda c, Kl: emb @ (lambda s: s / s.norm())(c.to(DEV).float() @ Kl.to(DEV).float())
    def show(label, sc):
        order = sc.argsort(descending=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(len(order), device=order.device)
        best = sorted((int(rank[i]) + 1, strs[i]) for i in watch)[:3]
        say(f"     {label:26s} best Tornado-token ranks {best} | top: {' '.join(repr(strs[i]) for i in order[:8].tolist())}")
    say("\n2. reads of the chain's read rows, raw and deflated (ranks among 124k kept tokens)")
    for L, n, p in [TORNADO[1], BRIDGE, COLORADO]:
        P = poles(MODEL, L, lobe_signs(L))
        f = P["write"].shape[1]
        c, s2 = P["cov"][n if p > 0 else f + n].clone(), metric.s2(MODEL, "mid", L)
        X = P["cov"][torch.randint(0, 2 * f, (4096,), generator=torch.Generator().manual_seed(0))].float()
        X = X / X.norm(dim=1, keepdim=True)
        Uc = torch.linalg.svd(X - X.mean(0), full_matrices=False)[2]
        say(f"  {name((L, n, p))}")
        show("R", rread(metric.vector(c[None], s2)[0], Jm[L]))
        show("S", sread(c, K[L]))
        for k in (8, 64, 512):
            d = c - (c @ Uc[:k].T) @ Uc[:k]
            show(f"R, top {k} PCs removed", rread(metric.vector(d[None], s2)[0], Jm[L]))
            show(f"S, top {k} PCs removed", sread(d, K[L]))
        v = metric.vector(c[None], s2)[0].to(DEV).float()
        J = Jm[L].to(DEV).float()
        for it in range(32):
            tt = int(rread(v, J).argmax())
            uu = J.T @ (gam * W_U[tt])
            uu = uu / uu.norm()
            v = v - (v @ uu) * uu
            if it + 1 in (8, 32):
                show(f"R, top {it + 1} tokens removed", rread(v, J))
                show("   its S read", sread(metric.covector(v[None].cpu(), s2)[0], K[L]))
        del P
    say("\n3. the bridge's top positive in-edges, each checked on its own five top contexts (the bridge's peak there, "
        "share of act_top)")
    lm = model()
    docs = documents()
    rng = random.Random(0)
    def ctx(e, k):
        out = []
        for d, t, v in zip(e["top_doc"], e["top_pos"], e["top_val"]):
            if d >= 0 and v > 0:
                out.append(docs[int(d)][max(1, int(t) - 96):int(t) + 9].long())
            if len(out) == k:
                break
        return out
    for target in (BRIDGE, CHAIN[1]):
        Pr = Probe(lm, [target])
        e = entry(*target)
        own = [Pr(s)[0][0] for s in ctx(e, 5)]
        say(f"  {name(target)} (own contexts {sum(own) / len(own):.2f})")
        for a, b, c, d in [(a, b, c, d) for a, b, c, d in zip(e["in_layer"], e["in_neuron"], e["in_pole"], e["in_push"]) if a >= 0 and d > 0][:12]:
            v = [Pr(s)[0][0] for s in ctx(entry(a, b, c), 5)]
            say(f"     push {d:5.2f}  {name((a, b, c)):16s} mean {sum(v) / len(v):.2f} max {max(v):.2f}   source drivers: {str(entry(a, b, c)['drivers_str'])[:50]}")
        ctl = []
        for _ in range(6):
            ctl += [Pr(s)[0][0] for s in ctx(entry(rng.randint(0, target[0] - 1), rng.randrange(17408), rng.choice((1, -1))), 3)]
        say(f"     random poles' contexts (n={len(ctl)}): mean {sum(ctl) / len(ctl):.2f}, max {max(ctl):.2f}")


def mongo(args):
    tok = tokenizer()
    nm = arch.names(MODEL)
    hd = tensors(MODEL, sorted({nm["unembed"], nm["norm"]}))
    W_U, gam = hd[nm["unembed"]].float(), hd[nm["norm"]].float()
    keep = kept()
    Jm, Jo = load_lens("hub@mid", MODEL), load_lens("hub", MODEL)
    M, N = BRIDGE[0], BRIDGE[1]
    PB = poles(MODEL, M, lobe_signs(M))
    f = PB["write"].shape[1]
    cB, vB, s2 = PB["cov"][f + N].float(), PB["vec"][f + N], metric.s2(MODEL, "mid", M)
    Wk = W_U[keep].to(DEV, torch.float16)
    strs = [tok.decode([int(i)]) for i in keep]
    def R(vecs, J):
        t = vecs.to(DEV).float() @ J.to(DEV).float().T
        t = t / t.norm(dim=-1, keepdim=True)
        return ((t * gam.to(DEV)).half() @ Wk.T).float()
    g0 = torch.Generator().manual_seed(0)
    zs = lambda x: (x.mean(0), x.std(0).clamp_min(1e-4))
    stB = zs(R(PB["vec"][torch.randint(0, 2 * f, (1024,), generator=g0)], Jm[M]))
    zB = (R(vB[None], Jm[M])[0] - stB[0]) / stB[1]
    ecos = lambda w: float((cB * (w / metric.vec_norm(w[None], s2)[0, 0])).sum() / metric.cov_norm(cB[None], s2)[0, 0])
    sources = TORNADO + [(6, 7831, 1), (7, 5551, 1), (5, 3686, -1), (2, 62, 1)]
    say("1. output tokens each source's write (R-lens out) shares with the bridge's read (R-lens mid), z-scored")
    writes = {}
    for L, n, p in sources:
        PA = poles(MODEL, L, lobe_signs(L))
        w = p * PA["write"][:, n]
        writes[(L, n, p)] = w
        stA = zs(R(PA["write"][:, torch.randint(0, PA["write"].shape[1], (1024,), generator=g0)].T, Jo[L]))
        zA = (R(w[None], Jo[L])[0] - stA[0]) / stA[1]
        top = (zA.clamp_min(0) * zB.clamp_min(0)).argsort(descending=True)[:12].tolist()
        say(f"  {name((L, n, p)):16s} edge cos {ecos(w):+.3f}: " + " ".join(repr(strs[i]) for i in top))
        del PA
    ids = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws if len(tok.encode(w, add_special_tokens=False)) == 1})
    groups = {"mongo": ids([" mongo", "mongo", " pymongo", " mongodb", " MongoDB", " bson", " BSON", " Mongo", "Mongo"]),
              "Mongolia": ids([" Mongolia", " Mongolian", "蒙古", "内蒙古", " Ulaanbaatar"]),
              "highland": ids([" Bolivia", " Nepal", " Tibet", " Peru", " Wyoming", " Utah", "青海", "西藏"]),
              "Colorado": ids([" Colorado", " Denver", " Boulder", " Rockies"]),
              "random": random.Random(0).sample(keep.tolist(), 9)}
    J = Jm[M].float()
    D = {k: (lambda U: U / U.norm(dim=1, keepdim=True))((J.T @ (gam[:, None] * W_U[v].T)).T) for k, v in groups.items()}
    cvh = metric.vector(cB[None], s2)[0]
    cvh = cvh / cvh.norm()
    say("\n2. token directions at the bridge's MLP input: mean cosine with its read vector; mongo against the others")
    for k, Ud in D.items():
        say(f"  {k:9s} {float((Ud @ cvh).mean()):+.3f} (max {float((Ud @ cvh).max()):+.3f})")
    say(f"  mongo·Mongolia {float((D['mongo'] @ D['Mongolia'].T).mean()):+.3f}, mongo·highland {float((D['mongo'] @ D['highland'].T).mean()):+.3f}, "
        f"mongo·random {float((D['mongo'] @ D['random'].T).mean()):+.3f}")
    basis = lambda U: torch.linalg.qr(U.T)[0]
    say("\n3. edge cosine with each source's write projected off a group's span")
    for u, w in writes.items():
        say(f"  {name(u):16s} raw {ecos(w):+.3f}  " + "  ".join(f"−{k} {ecos(w - basis(D[k]) @ (basis(D[k]).T @ w)):+.3f}" for k in ("mongo", "Mongolia", "highland", "random")))
    say("\n4. edges into the bridge from MongoDB-detecting units, against 4,000 random poles of each layer")
    for (L, n, p), what in [((3, 3032, 1), "'Mong-' (mongo and Mongolia)"), ((4, 16256, -1), "pymongo"), ((5, 6381, -1), "MongoDB (pymongo, bson)"),
                            ((5, 17189, 1), "'ongo' / mongo"), ((8, 5938, 1), "pymongo"), ((0, 13634, 1), "Kafka / mongo")] + [(u, "Tornado library") for u in TORNADO]:
        Wl = poles(MODEL, L, lobe_signs(L))["write"]
        idx = torch.randint(0, Wl.shape[1], (4000,), generator=g0)
        sg = torch.where(torch.rand(4000, generator=g0) < 0.5, 1.0, -1.0)
        null = torch.tensor([abs(ecos(sg[i] * Wl[:, idx[i]])) for i in range(4000)])
        c = ecos(p * Wl[:, n])
        say(f"  {name((L, n, p)):16s} {what:30s} cos {c:+.3f} (null |cos| median {null.median():.3f}, q99 {null.quantile(0.99):.3f})")
    watch = {s: i for i, s in enumerate(strs) if s in (" bson", " mongo", " mongodb", " pymongo", " MongoDB", " Mongolia", " Colorado", " Bolivia", " tornado", " flask", " redis")}
    mids = [i for s, i in watch.items() if s in (" bson", " mongo", " mongodb")]
    def ranks(z):
        order = z.argsort(descending=True)
        rk = torch.empty_like(order)
        rk[order] = torch.arange(len(order), device=order.device)
        return rk
    say("\n5. token ranks (of 124k) in readers' z-scored R-lens reads; a MongoDB token's best rank in 4,096 random poles' reads")
    for L, n, p, what in [(11, 6575, -1, "bridge"), (7, 5551, 1, "Hebei / Inner Mongolia reader"), (13, 11715, -1, "Colorado context"), (31, 11703, 1, "Colorado neuron")]:
        Pl = poles(MODEL, L, lobe_signs(L))
        fl = Pl["write"].shape[1]
        st = zs(R(Pl["vec"][torch.randint(0, 2 * fl, (1024,), generator=g0)], Jm[L]))
        rk = ranks((R(Pl["vec"][n if p > 0 else fl + n][None], Jm[L])[0] - st[0]) / st[1])
        best = []
        for chunk in torch.randint(0, 2 * fl, (4096,), generator=g0).split(256):
            z = (R(Pl["vec"][chunk], Jm[L]) - st[0]) / st[1]
            best.append(((z[:, None, :] > z[:, mids][:, :, None]).sum(-1).min(1).values + 1).cpu())
        best = torch.cat(best)
        say(f"  {name((L, n, p)):16s} ({what:29s}) " + "  ".join(f"{s.strip()} #{int(rk[i]) + 1}" for s, i in watch.items())
            + f" | random poles with one in the top 20: {(best <= 20).float().mean():.4f}, top 50: {(best <= 50).float().mean():.4f}")
        del Pl
    e = next(r for r in pq.read_table(neuron_table("edges", M), columns=["neuron", "pole", "in_layer", "in_neuron", "in_pole"]).to_pylist()
             if r["neuron"] == N and r["pole"] == BRIDGE[2])
    ins = [(l, n, p) for l, n, p in zip(e["in_layer"], e["in_neuron"], e["in_pole"]) if l >= 0][:64]
    cov_of = lambda u: metric.covector((u[2] * poles(MODEL, u[0], lobe_signs(u[0]))["write"][:, u[1]])[None], s2)[0]
    allw = torch.stack([cov_of(u) for u in ins])
    off = lambda c, W: c - (lambda Q: Q @ (Q.T @ c))(torch.linalg.qr(W.T.float())[0])
    zr = lambda c: ranks((R(metric.vector(c[None], s2), Jm[M])[0] - stB[0]) / stB[1])
    say("\n6. the bridge's read with weight inputs' writes removed (ranks)")
    for what, c in [("raw", cB), ("off its two Tornado inputs", off(cB, torch.stack([cov_of(u) for u in TORNADO]))),
                    (f"off its top {len(ins)} weight inputs", off(cB, allw))]:
        rk = zr(c)
        say(f"  {what:28s} " + "  ".join(f"{s.strip()} #{int(rk[i]) + 1}" for s, i in watch.items()))
    say("\n7. causal: the bridge's peak with its MLP input projected off a group's span, as a share of the unablated peak")
    lm = model()
    lay, cap, state = lm.model.layers[M], {}, {"Q": None}
    def pre_norm(m, a):
        if state["Q"] is not None:
            x = a[0].float()
            Q = state["Q"].to(x.device)
            return ((x - (x @ Q) @ Q.T).to(a[0].dtype),)
    lay.post_attention_layernorm.register_forward_pre_hook(pre_norm)
    lay.mlp.down_proj.register_forward_pre_hook(lambda m, a: cap.__setitem__("a", (BRIDGE[2] * a[0][0, 1:, N]).float()))
    def peak(k):
        lm(tok(TEXTS[k], return_tensors="pt").input_ids.to(DEV))
        return float(cap["a"].max())
    texts = ["tornado_web", "tornado_httpclient", "bolivia", "inner_mongolia_zh", "wyoming", "colorado_news"]
    base = {k: peak(k) for k in texts}
    say("  " + "".join(f"{k:>20s}" for k in texts))
    for k_ in ("mongo", "Mongolia", "highland", "Colorado", "random"):
        state["Q"] = basis(D[k_])
        say(f"  −{k_:9s}" + "".join(f"{peak(k) / base[k]:20.2f}" for k in texts))
        state["Q"] = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("lineage", "probe", "files", "split", "wiring", "mongo"))
    args = ap.parse_args()
    globals()[args.part](args)
    (SUMMARY / f"colorado_tornado_{args.part}.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
