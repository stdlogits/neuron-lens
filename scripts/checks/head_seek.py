"""Heads that look up what the residual already says (findings §11): QK's cross-lens diagonal, against attention in
context and what the head writes there.

``08_reads.py``'s ``qk_self_S`` is QK's diagonal with the S-lens footprint on both sides: query ``v`` (the position
holds ``v``) against key ``v``, which finds duplicate-token heads. Here the query is instead the R-lens direction at
layer ``L − 1``'s output that promotes ``v`` (``W_U diag(γ_f) J^out_{L−1}``, the position says ``v``), the key still
its S-lens footprint: ``RS``, where key ``v`` ranks among the keys a query about to say ``v`` favours, the QK half
of copy suppression (McDougall et al. 2023). Beside it ``SS`` (``qk_self_S``, recomputed) and ``RR`` (the R-lens
direction on both sides). Each is ``neuron_lens.heads.own_token_z`` over the kept tokens, at unit length, with the
q/k norms' true derivative and RoPE averaged over the head's own offsets, as ``08_reads.py``; a head's score is
the median over tokens. ``rho_ov``: Spearman over tokens between a head's ``RS`` and its ``ov_R`` per-token z
(negative: it seeks what it suppresses).

In context (a bf16 pass with eager attention over every ``EVERY``-th activation document): at each query (16+), a
target token, and the attention to earlier keys holding it (past the sink, renormalized without it), per matching
key relative to uniform over the query's keys. Targets: ``lens``, the R-lens top-1 at ``L − 1``'s output over the
kept tokens (what the residual says where the head reads it); ``final``, the model's top-1 (which a copying head
can cause rather than follow); ``ctrl``, the token at a random earlier position. Only keys two or more positions
back count (early, the R-lens top-1 is often a recent token, which previous-token attention would match); keys whose
previous token is the query's (induction matches) and targets equal to the query's own token (duplicate matches) are
left out. A head's
``seek`` is its lift on a target minus its lift on ``ctrl``; its ``effect`` the direct effect of its output on the
target's logit (``W_U diag(γ_f)`` row through the head's write as the residual sees it: ``W_O^h``, for Gemma 4 times
``γ_post`` and the post-attention norm's scale at that position; before the final norm's positive scale) minus that
on ``ctrl``'s, at the same queries. The late groups are the last two bands (Qwen3-8B L21-L35, Gemma 4 L24-L47). The
pass is cached in ``data/heads/<model>-seek-context.pt``. Summary in ``data/summary/head_seek_<model>.{log,json}``.

    python scripts/checks/head_seek.py
    NEURON_LENS_MODEL=google/gemma-4-12B python scripts/checks/head_seek.py
"""
import json
import time

import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import MODEL, acts_path, doc_tokens, head_table, kept
from neuron_lens.heads import norm_derivative, own_token_z, rope_mean
from neuron_lens.paths import DATA, SUMMARY
from neuron_lens.stats import auroc, centered_ranks, spearman
from neuron_lens.transport import SKIP_FIRST
from neuron_lens.weights import tensors

STEM = arch.stem(MODEL)
TOKEN_Z = DATA / "heads" / f"{STEM}-token-z.pt"
CACHE = DATA / "heads" / f"{STEM}-seek-context.pt"
EVERY = 32
BATCH = 16
TARGETS = ("lens", "final", "ctrl")
READS = ("RS", "SS", "RR")
EXAMPLES = {arch.QWEN: ((32, 9), (33, 21), (27, 7), (31, 30), (28, 23), (29, 11), (23, 4), (22, 23), (1, 28))}


def unembed(dev, rows=None) -> torch.Tensor:
    """``W_U diag(γ_f)``, over ``rows`` (all tokens if None)."""
    nm = arch.names(MODEL)
    head = tensors(MODEL, [nm["unembed"], nm["norm"]])
    W = head[nm["unembed"]] if rows is None else head[nm["unembed"]][rows]
    return W.float() * head[nm["norm"]].float()


def diagonals(dev, tot: dict, extra: dict) -> dict:
    """``[L, H, V]`` per-token z of the three diagonals (NaN at L0, where there is no R-lens input)."""
    geo = arch.layout(MODEL)
    L, H = len(geo), geo[0].q_heads
    keep = kept()
    V = len(keep)
    nm = arch.names(MODEL)
    E = tensors(MODEL, [nm["embed"]])[nm["embed"]][keep].float() * arch.embed_scale(MODEL)
    E = (E - E.mean(0)).to(dev)
    Wu = unembed(dev, keep).to(dev)
    R_FIT, S_FIT = arch.fits(MODEL)
    K_out, J_out = arch.lens(MODEL, "S_out", S_FIT), arch.lens(MODEL, "R_out", R_FIT)
    z = {k: torch.full((L, H, V), float("nan"), dtype=torch.float16) for k in READS}
    for layer in range(1, L):
        lay = geo[layer]
        kvh, hd = lay.kv_heads, lay.head_dim
        qk = arch.qk(MODEL, lay, dev)
        Aq = norm_derivative(qk["q_norm"], tot["q_r"][layer].float().to(dev), tot["q_m"][layer, :, :hd, :hd].float().to(dev),
                             float(tot["n_q"]))
        Ak = norm_derivative(qk["k_norm"], tot["k_r"][layer, :kvh].float().to(dev),
                             tot["k_m"][layer, :kvh, :hd, :hd].float().to(dev), float(tot["n_k"]))
        off = tot["offset"][layer].float().to(dev)
        Rbar = rope_mean(off / off.sum(-1, keepdim=True), arch.layer_inv_freq(extra["inv_freq"], MODEL, layer).to(dev))
        unit = lambda X: X / X.norm(dim=1, keepdim=True)
        side = {"S": unit(E @ K_out[layer - 1].to(dev).T), "R": unit(Wu @ J_out[layer - 1].to(dev))}

        def keys(U):
            Kv = torch.einsum("gij,vgj->vgi", Ak, (U @ qk["Wk"].T).view(V, kvh, hd)).repeat_interleave(H // kvh, dim=1)
            return torch.einsum("hij,vhj->vhi", Rbar, Kv)  # R̄ K

        for k_side in ("S", "R"):
            K = keys(side[k_side])
            for name in READS:
                if name[1] == k_side:
                    Q = torch.einsum("hij,vhj->vhi", Aq, (side[name[0]] @ qk["Wq"].T).view(V, H, hd))
                    z[name][layer] = own_token_z(K, Q).T.half().cpu()
                    del Q
            del K
        del side
        print(f"{time.strftime('%H:%M:%S')} diagonals L{layer}", flush=True)
    return z


def context(dev) -> dict:
    """Per head and target: attention per matching key relative to uniform (``lift``) and the direct effect on the
    target's logit (``effect``), each a mean over the queries with an earlier copy of the target; ``n`` per layer."""
    saved = torch.load(acts_path(), weights_only=False, map_location="cpu", mmap=True)
    seqs = doc_tokens(saved)
    docs = list(range(0, len(seqs), EVERY))
    lm, stack = arch.load_model(MODEL, dev)
    L, H = len(arch.layout(MODEL)), arch.layout(MODEL)[0].q_heads
    keep = kept().to(dev)
    Wu_full = unembed(dev).to(dev, torch.bfloat16)
    Wu_keep = Wu_full[keep]
    R_FIT, _ = arch.fits(MODEL)
    J_out = arch.lens(MODEL, "R_out", R_FIT)
    J = [None] + [J_out[l - 1].to(dev, torch.bfloat16) for l in range(1, L)]
    acc = {t: torch.zeros(L, H, dtype=torch.float64) for t in TARGETS}
    dla = {t: torch.zeros(L, H, dtype=torch.float64) for t in TARGETS}
    n = {t: torch.zeros(L, dtype=torch.float64) for t in TARGETS}
    ctx = {"on": False}
    gen = torch.Generator().manual_seed(0)

    def masks(y, ok):
        M = ctx["base"] & (ctx["ids"][:, None, :] == y[:, :, None]) & ok[:, :, None]  # [B, query, key]
        c = M.sum(-1)
        hit = c > 0
        w = torch.where(hit, ctx["ar"].float()[None] / c.clamp_min(1), torch.zeros((), device=dev))  # keys 1..t over matches
        return M.float(), w, hit, y

    def pre_layer(l):
        def f(m, args, kwargs):
            if not ctx["on"]:
                return
            if l == 0:
                ctx[l] = None
                return
            x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]  # layer l − 1's output
            B, T, d = x.shape
            lens = keep[((x.reshape(-1, d).to(torch.bfloat16) @ J[l].T) @ Wu_keep.T).argmax(-1)].view(B, T)
            ctx[l] = {"lens": masks(lens, ctx["q_ok"] & (lens != ctx["ids"])), "final": ctx["final"], "ctrl": ctx["ctrl"]}
        return f

    def attn(l):
        def f(m, args, out):
            if not ctx["on"] or ctx[l] is None:
                return
            a = out[1].float()  # [B, H, T, T]
            a = a / a[..., 1:].sum(-1, keepdim=True).clamp_min(1e-30)  # without the sink
            for t in TARGETS:
                M, w, hit, _ = ctx[l][t]
                acc[t][l] += (torch.einsum("bhtj,btj->bht", a, M) * w[:, None]).sum((0, 2)).cpu().double()
                n[t][l] += float(hit.sum())
        return f

    def o_proj(l, layer, W_O):
        def f(m, args, out):
            if not ctx["on"] or ctx[l] is None:
                return
            z = args[0]  # [B, T, H·hd]
            hd = z.shape[-1] // H
            scale = arch.attn_scale(layer, out)  # [B, T]: the post-attention norm's scale (Gemma 4), else None
            for t in TARGETS:
                _, _, hit, y = ctx[l][t]
                e = ((Wu_full[y[hit]] @ W_O).view(-1, H, hd) * z[hit].view(-1, H, hd)).sum(-1).float()  # [queries, H]
                if scale is not None:
                    e = e * scale[hit][:, None]
                dla[t][l] += e.sum(0).cpu().double()
        return f

    hooks = []
    for l, layer in enumerate(stack.layers):
        hooks.append(layer.register_forward_pre_hook(pre_layer(l), with_kwargs=True))
        hooks.append(layer.self_attn.register_forward_hook(attn(l)))
        W_O = arch.head_write(layer).to(dev, torch.bfloat16)  # × γ_post for Gemma 4
        hooks.append(layer.self_attn.o_proj.register_forward_hook(o_proj(l, layer, W_O)))
    t0 = time.perf_counter()
    for s in range(0, len(docs), BATCH):
        part = [seqs[i] for i in docs[s:s + BATCH]]
        Tm = max(len(q) for q in part)
        ids = torch.zeros(len(part), Tm, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for r, q in enumerate(part):
            ids[r, :len(q)], mask[r, :len(q)] = torch.as_tensor(q).long(), 1
        ids, mask = ids.to(dev), mask.to(dev)
        real = mask.bool()
        ar = torch.arange(Tm, device=dev)
        ctx["on"] = False
        with torch.inference_mode():
            hid = stack(input_ids=ids, attention_mask=mask).last_hidden_state
            final = torch.stack([arch.logits(lm, hid[b]).argmax(-1) for b in range(len(part))])
        q_ok = real & (ar >= SKIP_FIRST)
        causal = (ar[None, :] <= ar[:, None] - 2) & (ar[None, :] >= 1)  # keys two or more back, past the sink
        prev = torch.cat([torch.full_like(ids[:, :1], -1), ids[:, :-1]], 1)  # each key's previous token
        ctx.update(ids=ids, ar=ar, q_ok=q_ok, base=causal[None] & real[:, None, :] & (prev[:, None, :] != ids[:, :, None]))
        jr = (torch.rand(ids.shape, generator=gen).to(dev) * ar.clamp_min(1)).long().clamp(1, Tm - 1)
        ctrl = ids.gather(1, jr)
        ctx["final"] = masks(final, q_ok & (final != ids))
        ctx["ctrl"] = masks(ctrl, q_ok & (ctrl != ids) & (ctrl != final))
        ctx["on"] = True
        with torch.inference_mode():
            stack(input_ids=ids, attention_mask=mask)
        if s // BATCH % 10 == 0:
            print(f"{time.strftime('%H:%M:%S')} {s + len(part)}/{len(docs)} documents, {time.perf_counter() - t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()
    return {"lift": {t: (acc[t] / n[t].clamp_min(1)[:, None]).float() for t in TARGETS},
            "effect": {t: (dla[t] / n[t].clamp_min(1)[:, None]).float() for t in TARGETS},
            "n": {t: n[t] for t in TARGETS}, "docs": docs}


def main() -> None:
    dev = "mps" if torch.backends.mps.is_available() else "cuda"
    saved = torch.load(acts_path(), weights_only=False, map_location="cpu", mmap=True)
    tot, extra = saved["state"]["total"], saved["state"]["extra"]
    z = diagonals(dev, tot, extra)
    tz = torch.load(TOKEN_Z, weights_only=False, mmap=True)
    diff = (z["SS"][1:].float() - tz["qk_self_S"][1:].float()).abs().max().item()
    L, H, _ = z["RS"].shape
    med = {k: z[k].float().median(-1).values for k in READS}
    rho_ov = torch.full((L, H), float("nan"))
    for l in range(1, L):
        a, b = (centered_ranks(x[l].float()) for x in (z["RS"], tz["ov_R"]))
        rho_ov[l] = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))
    del z, tz

    docs = list(range(0, saved["meta"]["docs"], EVERY))
    cx = torch.load(CACHE, weights_only=False) if CACHE.exists() else None
    if cx is None or cx["docs"] != docs:
        cx = context(dev)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cx, CACHE)
    lift, eff = cx["lift"], cx["effect"]
    seek = {t: lift[t] - lift["ctrl"] for t in ("lens", "final")}
    effect = {t: eff[t] - eff["ctrl"] for t in ("lens", "final")}
    copy = torch.tensor(pq.read_table(head_table("reads"), columns=["ov_R_copy"]).column("ov_R_copy").to_numpy()).view(L, H).float()
    other = (torch.arange(H)[None, :] + torch.randint(1, H, (L, H), generator=torch.Generator().manual_seed(0))) % H
    sp = lambda a, b: spearman(a.flatten(), b.flatten())

    bands = [(max(lo, 1), hi) for lo, hi in arch.bands(L)]
    lines = [(f"QK's cross-lens diagonal, {MODEL}: {L - 1} layers × {H} heads from L1; RS (R-lens query, S-lens key), "
              f"SS (08_reads.py's qk_self_S, recomputed: max |Δz| {diff:.3f}), RR; in context over every {EVERY}th activation "
              f"document ({len(docs):,}), queries with an earlier copy of the target per layer (mean): "
              + ", ".join(f"{t} {cx['n'][t][1:].mean():,.0f}" for t in TARGETS)), "",
             "by band: median over heads of each head's median z, RS / SS / RR; median rho(RS, ov_R) over tokens; attention per "
             "matching key relative to uniform, mean over heads, lens / final / ctrl"]
    table = {"bands": {}, "heads": {}, "qk_self_max_diff": diff}
    for lo, hi in bands + [(1, L - 1)]:
        s = slice(lo, hi + 1)
        row = {"median": [med[k][s].median().item() for k in READS], "rho_ov": rho_ov[s].nanmedian().item(),
               "lift": [lift[t][s].mean().item() for t in TARGETS],
               "rho_seek": {k: sp(med[k][s], seek["lens"][s]) for k in READS},
               "rho_seek_control": sp(med["RS"][s].gather(1, other[s]), seek["lens"][s]),
               "rho_seek_final": sp(med["RS"][s], seek["final"][s]),
               "rho_seek_effect": sp(seek["lens"][s], effect["lens"][s]), "rho_RS_effect": sp(med["RS"][s], effect["lens"][s])}
        table["bands"][f"L{lo}-{hi}"] = row
        lines.append(f"  L{lo}-{hi}: {' / '.join(f'{x:+.2f}' for x in row['median'])}   rho_ov {row['rho_ov']:+.2f}   "
                     f"lift {' / '.join(f'{x:.2f}' for x in row['lift'])}")
    lines += ["", ("Spearman across heads with seek (lens target): RS own / RS control head / SS / RR; RS with seek (final "
                   "target); seek with effect; RS with effect")]
    for b, row in table["bands"].items():
        r = row["rho_seek"]
        lines.append(f"  {b}: {r['RS']:+.2f} / {row['rho_seek_control']:+.2f} / {r['SS']:+.2f} / {r['RR']:+.2f}   final "
                     f"{row['rho_seek_final']:+.2f}   seek–effect {row['rho_seek_effect']:+.2f}   RS–effect {row['rho_RS_effect']:+.2f}")
    lo_late = bands[-2][0]
    late = slice(lo_late, L)
    top = seek["lens"][late] > seek["lens"][late].quantile(0.8)
    groups = {"seek and suppress": top & (effect["lens"][late] < 0), "seek and boost": top & (effect["lens"][late] > 0)}
    lines += ["", f"L{lo_late}-{L - 1}, the top 20% of heads by seek (lens), split by the sign of their effect: AUROC of each score"]
    table["late"] = {}
    for name, m in groups.items():
        sc = {"RS": auroc(med["RS"][late].flatten(), m.flatten()), "SS": auroc(med["SS"][late].flatten(), m.flatten()),
              "RR": auroc(med["RR"][late].flatten(), m.flatten()),
              "ov_R_copy": auroc((copy[late] if "boost" in name else -copy[late]).flatten(), m.flatten())}
        table["late"][name] = {"heads": int(m.sum()), **sc}
        lines.append(f"  {name} ({int(m.sum())} heads): RS {sc['RS']:.2f}, SS {sc['SS']:.2f}, RR {sc['RR']:.2f}, "
                     f"{'' if 'boost' in name else '−'}ov_R_copy {sc['ov_R_copy']:.2f}")
        if "suppress" in name:
            heads = [(lo_late + i // H, i % H) for i in m.flatten().nonzero().flatten().tolist()]
            table["late"][name]["list"] = [f"L{l}h{h}" for l, h in heads]
            lines.append("    " + ", ".join(f"L{l}h{h}" for l, h in heads))
    for name, m in (("copiers (ov_R_copy > 0.5)", copy[late] > 0.5), ("suppressors (ov_R_copy < −0.5)", copy[late] < -0.5)):
        lines.append(f"  {name}: {int(m.sum())} heads, median seek final {seek['final'][late][m].median():+.2f}, "
                     f"lens {seek['lens'][late][m].median():+.2f}; median effect (lens) {effect['lens'][late][m].median():+.3f}")
    f = lambda l, h: (f"L{l}h{h}: RS {med['RS'][l, h]:+.2f} SS {med['SS'][l, h]:+.2f} RR {med['RR'][l, h]:+.2f} rho_ov "
                      f"{rho_ov[l, h]:+.2f} ov_R_copy {copy[l, h]:+.2f}; seek lens {seek['lens'][l, h]:+.2f} final "
                      f"{seek['final'][l, h]:+.2f}, effect {effect['lens'][l, h]:+.3f}")
    lines += ["", f"the top 12 heads by RS from L{lo_late}:"]
    lines += [f"  {f(lo_late + i // H, i % H)}" for i in med["RS"][late].flatten().topk(12).indices.tolist()]
    lines += ["", "the top 12 heads by seek (lens):"]
    lines += [f"  {f(1 + i // H, i % H)}" for i in seek["lens"][1:].flatten().topk(12).indices.tolist()]
    if MODEL in EXAMPLES:
        lines += ["", "named:"] + [f"  {f(l, h)}" for l, h in EXAMPLES[MODEL]]
    for l in range(1, L):
        for h in range(H):
            table["heads"][f"L{l}h{h}"] = {**{k: med[k][l, h].item() for k in READS}, "rho_ov": rho_ov[l, h].item(),
                                           "seek_lens": seek["lens"][l, h].item(), "seek_final": seek["final"][l, h].item(),
                                           "effect_lens": effect["lens"][l, h].item()}
    text = "\n".join(lines)
    print(text)
    (SUMMARY / f"head_seek_{STEM}.log").write_text(text + "\n")
    (SUMMARY / f"head_seek_{STEM}.json").write_text(json.dumps(table))


if __name__ == "__main__":
    main()
