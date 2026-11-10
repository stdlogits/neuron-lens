"""What drives a neuron pole when it fires: an exact split of its activation over its sources (findings §17).

At a firing of a neuron of layer ``M`` at position ``t``, its MLP input is the residual ``x`` there,
normalized, and ``x`` is a sum: the embedding, every head's write at layers ≤ ``M`` and every neuron's at
layers < ``M``, all at ``t``. Holding the norm's scale and the activation's ratio ``s = φ(g)/g`` (RelP's rules),
the gate and up pre-activations ``g``, ``u`` split exactly over these sources, and so does the activation
by the half rule: source X contributes ``½ s (g u_X + u g_X)``, and the contributions sum to
``a = φ(g) u``. A share is a contribution over ``a``.

What a model does to a write on its way to ``x`` is ``arch``'s: a norm after the attention or the MLP divides
it by that output's RMS at the position, and a layer scalar multiplies everything the layer passes on. At one
firing these are numbers, so the sum stays exact (``Stream``).

A head has three inputs where a neuron has one read direction, and at one firing each becomes a read of the
residual entering its layer, at a position or several (``HeadSources``). The sources of each are the embedding, and
the heads and neurons of earlier layers, at those positions.

- **Value** (``v``, what it copies): with the attention pattern, the norms' scales and the head's own output ``o`` held,
  ``o`` is linear in the residual at every position it attends to. A source's share is its part of ``o`` along ``o``,
  and the shares sum to 1.
- **Query** and **key** (``q``, ``k``, why it looks there): the score ``q_t · k_s`` is bilinear. The number split is the
  most attended key's margin (past the sink) over the mean of the keys (past the sink), so ``q`` splits it over the
  sources at the query's position with the keys held, and ``k`` over the sources at the keys' positions with the query
  held. Each sums to 1, exactly for the score and to first order for the pattern it sets.
"""
from collections.abc import Callable

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from neuron_lens import arch, layers
from neuron_lens.catalog import MODEL, N_CONTEXTS, TOP, column, fixed, head_keys, head_table, neuron_keys, neuron_table
from neuron_lens.firings import firing_plan
from neuron_lens.rules import norm_eps

RANK = 256  # candidates kept per firing for driven-by
RANK_WITH = 1024  # ... and for fires-with, where a source shared by several firings needs to survive each
FIRE = 1 / 4  # of act_top: the acts tables' rate threshold
CHANNELS = ("v", "q", "k")  # a head's inputs: value, query, key


class Stream:
    """A decoder stack's residual as a sum of writes, for any model ``arch``'s geometry covers."""

    def __init__(self, stack):
        self.layers, self.embed = stack.layers, stack.embed_tokens
        self.n_heads = stack.config.num_attention_heads
        heads = [arch.head_write(lay) for lay in self.layers]  # [d, H·hd], hd may differ by layer (Gemma 4's full-attention layers)
        self.all_heads = torch.cat(heads, 1)  # a layer's sources are a prefix of these
        self.all_neurons = torch.cat([arch.mlp_write(lay) for lay in self.layers], 1)  # [d, L·f]
        self.neurons = self.all_neurons.split(stack.config.intermediate_size, 1)  # per layer, views of it
        dev = self.all_neurons.device
        self.head_end = torch.tensor([w.shape[1] for w in heads]).cumsum(0).tolist()  # layer l's heads end at this column
        self.head_of = torch.cat([l * self.n_heads + torch.arange(w.shape[1], device=dev) // (w.shape[1] // self.n_heads)
                                  for l, w in enumerate(heads)])  # the head each column of all_heads belongs to
        self.scalars = torch.tensor([arch.layer_scale(lay) for lay in self.layers], device=dev)

    def hooks(self, cap: dict, at: Callable, upto: int | None = None) -> list:
        """Forward hooks filling ``cap`` during a pass, each value at the positions ``at`` picks from a
        ``[B, T, ...]`` tensor: ``"emb"``; per layer, ``("z", l)`` the heads' pre-``o_proj`` outputs, ``("x", l)`` the
        residual entering the MLP's norm, ``("a", l)`` the activations and, where a norm follows,
        ``("attn", l)`` and ``("mlp", l)`` its scale. ``upto``: stop at that layer's MLP input."""
        def store(key):
            return lambda mod, a: cap.__setitem__(key, at(a[0]).float())

        def scale(key, of):
            def hook(mod, a, o):
                v = of(o[0] if isinstance(o, tuple) else o)
                if v is not None:
                    cap[key] = at(v)
            return hook
        out = [self.embed.register_forward_hook(lambda mod, a, o: cap.__setitem__("emb", at(o).float()))]
        for l, lay in enumerate(self.layers[:None if upto is None else upto + 1]):
            out.append(lay.self_attn.o_proj.register_forward_pre_hook(store(("z", l))))
            out.append(lay.self_attn.register_forward_hook(scale(("attn", l), lambda o, lay=lay: arch.attn_scale(lay, o))))
            out.append(arch.mid_norm(lay).register_forward_pre_hook(store(("x", l))))
            if l != upto:
                out.append(lay.mlp.down_proj.register_forward_pre_hook(store(("a", l))))
                out.append(lay.mlp.register_forward_hook(scale(("mlp", l), lambda o, lay=lay: arch.mlp_scale(lay, o))))
        return out

    def sources(self, cap: dict, M: int, neuron: torch.Tensor, u: torch.Tensor) -> dict:
        """``split`` for the firings ``u`` (indices into ``cap``'s positions) of ``neuron`` ``[N]`` in layer ``M``."""
        lay, norm = self.layers[M], arch.mid_norm(self.layers[M])
        x = cap["x", M][u]
        inv = torch.rsqrt(x.pow(2).mean(-1) + norm_eps(norm))[:, None]
        rg, ru = lay.mlp.gate_proj.weight[neuron] * norm.weight, lay.mlp.up_proj.weight[neuron] * norm.weight

        def reaching(key: str, scale: str, l: int) -> torch.Tensor:
            v = cap[key, l][u] * self.scalars[l:M].prod()
            return v * cap[scale, l][u][:, None] if (scale, l) in cap else v
        return split(rg, ru, x, inv, cap["emb"][u] * self.scalars[:M].prod(),
                     torch.cat([reaching("z", "attn", l) for l in range(M + 1)], 1),
                     torch.cat([reaching("a", "mlp", l) for l in range(M)], 1) if M else x[:, :0],
                     self.all_neurons[:, :M * self.neurons[0].shape[1]], self.all_heads[:, :self.head_end[M]],
                     self.head_of[:self.head_end[M]], (M + 1) * self.n_heads, lay.mlp.act_fn)


def split(rg: torch.Tensor, ru: torch.Tensor, x: torch.Tensor, inv: torch.Tensor, emb: torch.Tensor,
          z: torch.Tensor, acts: torch.Tensor, downs: torch.Tensor, wos: torch.Tensor, head_of: torch.Tensor, n_heads: int,
          act: Callable) -> dict:
    """Per firing (rows): ``rg``, ``ru`` ``[N, d]``, the neuron's gate and up rows times the norm's weight;
    ``x`` ``[N, d]``, the residual entering the norm; ``inv`` ``[N, 1]``, the norm's scale; ``emb`` the
    embedding, ``z`` ``[N, C]`` the heads' pre-``o_proj`` outputs at layers ≤ ``M``, ``acts`` ``[N, M·f]``
    the activations at layers < ``M``, these three scaled as they reach ``x``; ``downs`` ``[d, M·f]``, ``wos``
    ``[d, C]`` the writes as the residual sees them; ``head_of`` ``[C]`` the head each of ``C`` columns belongs to, of
    ``n_heads``; ``act`` the MLP's activation. Returns ``g``, ``a`` ``[N]`` and the shares ``emb`` ``[N]``, ``heads``
    ``[N, n_heads]``, ``neurons`` ``[N, M·f]``, which sum to 1 over the three."""
    g, u = (rg * x).sum(-1) * inv[:, 0], (ru * x).sum(-1) * inv[:, 0]
    a = act(g) * u
    share = lambda gx, ux: 0.5 * (gx / g[:, None] + ux / u[:, None])  # ½ s (g u_X + u g_X) / (s g u)
    e = share((rg * emb).sum(-1, keepdim=True) * inv, (ru * emb).sum(-1, keepdim=True) * inv)[:, 0]
    per = lambda r: torch.zeros(len(rg), n_heads, device=z.device).index_add_(1, head_of, (r @ wos) * z) * inv
    return {"g": g, "a": a, "emb": e, "heads": share(per(rg), per(ru)), "neurons": share((rg @ downs) * acts * inv, (ru @ downs) * acts * inv)}


def merged(rows: torch.Tensor, ids: torch.Tensor, vals: torch.Tensor, n_ids: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Entries ``(row, id, value)``, flat: the unique ``(row, id)`` with their values summed."""
    key = rows.long() * n_ids + ids.long()
    uniq, inv = key.unique(return_inverse=True)
    total = torch.zeros(len(uniq), dtype=vals.dtype).index_add_(0, inv, vals)
    return uniq // n_ids, uniq % n_ids, total


def top_by_row(rows: torch.Tensor, ids: torch.Tensor, vals: torch.Tensor, n_rows: int, k: int,
               score: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``([n_rows, k] ids, vals)``: each row's ``k`` entries by largest ``score`` (default ``|vals|``),
    ``−1`` and NaN where it has fewer."""
    score = vals.abs() if score is None else score
    first = score.argsort(descending=True, stable=True)
    order = first[rows[first].argsort(stable=True)]
    r = rows[order]
    counts = torch.bincount(r, minlength=n_rows)
    rank = torch.arange(len(r)) - (counts.cumsum(0)[r] - counts[r])
    keep = rank < k
    out_i, out_v = torch.full((n_rows, k), -1, dtype=ids.dtype), torch.full((n_rows, k), float("nan"), dtype=vals.dtype)
    out_i[r[keep], rank[keep]], out_v[r[keep], rank[keep]] = ids[order][keep], vals[order][keep]
    return out_i, out_v


def fires_with(r: torch.Tensor, ids: torch.Tensor, n: torch.Tensor, rate: torch.Tensor, floor: float, n_ids: int,
               n_rows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(ids, lift, co)``, ``[n_rows, TOP]``, from the (row, pole) pairs that fired together; ``n`` ``[n_rows]`` the firings a row
    has. A pole must fire at two of a row's firings, or at its one."""
    rows, uids, total = merged(r, ids, torch.ones(len(ids)), n_ids)
    enough = total >= torch.minimum(n[rows], torch.tensor(2.0))
    rows, uids, total = rows[enough], uids[enough], total[enough]
    co = total / n[rows]
    lift = co / rate[uids].clamp_min(floor)
    wi, wl = top_by_row(rows, uids, lift, n_rows, TOP)
    return wi, wl, top_by_row(rows, uids, co, n_rows, TOP, score=lift)[1]


def head_list(cols: dict, name: str, mean: torch.Tensor, H: int, silent: torch.Tensor) -> None:
    """``<name>_head_layer``, ``<name>_head``, ``<name>_head_share``: the ``TOP`` largest |mean share| of ``mean`` ``[rows, heads]``."""
    v, i = mean.abs().topk(min(TOP, mean.shape[1]), dim=1)
    pad = TOP - v.shape[1]
    ids = torch.nn.functional.pad(i, (0, pad), value=-1)
    share = torch.nn.functional.pad(mean.gather(1, i), (0, pad), value=float("nan"))
    ids[silent], share[silent] = -1, float("nan")
    cols[f"{name}_head_layer"] = fixed(torch.where(ids >= 0, ids // H, -1).to(torch.int16))
    cols[f"{name}_head"] = fixed(torch.where(ids >= 0, ids % H, -1).to(torch.int16))
    cols[f"{name}_head_share"] = fixed(share)


def pole_list(cols: dict, name: str, ids: torch.Tensor, value: torch.Tensor, value_name: str, f: int) -> None:
    """``<name>_layer``, ``<name>_neuron``, ``<name>_pole``, ``<name>_<value_name>`` from pole ids ``layer · 2f + row``, ``−1`` for none."""
    ok = ids >= 0
    layer, row = ids.div(2 * f, rounding_mode="floor"), ids % (2 * f)
    cols[f"{name}_layer"] = fixed(torch.where(ok, layer, -1).to(torch.int16))
    cols[f"{name}_neuron"] = fixed(torch.where(ok, row % f, -1).to(torch.int16))
    cols[f"{name}_pole"] = fixed(torch.where(ok, 1 - 2 * (row // f), 0).to(torch.int8))
    cols[f"{name}_{value_name}"] = fixed(value.float())


def acts_column(layer: int, name: str) -> torch.Tensor:
    return torch.from_numpy(pq.read_table(neuron_table("acts", layer), columns=[name]).column(0).to_numpy().copy()).float()


class Sources:
    """The ``sources`` tables, filled over a forward pass: ``locate`` a batch's firings, run the stack, ``split`` them."""

    def __init__(self, stack, layers: list[int], dev, positions: int):
        self.stack, self.layers, self.dev, self.flow = stack, layers, dev, Stream(stack)
        self.rate_floor = 1 / positions  # a pole seen firing at all fires at least once
        self.L, self.f, self.H = len(stack.layers), stack.config.intermediate_size, stack.config.num_attention_heads
        self.act_top = torch.stack([acts_column(l, "act_top") for l in range(self.L)]).to(dev).flatten()
        self.rate = torch.stack([acts_column(l, "rate") for l in range(self.L)]).to(dev).flatten()
        self.plan = {}
        for l in layers:
            t = pq.read_table(neuron_table("acts", l), columns=["top_doc", "top_pos", "top_val"])
            doc, pos, row = firing_plan(column(t, "top_doc", N_CONTEXTS).long(), column(t, "top_pos", N_CONTEXTS).long(), column(t, "top_val", N_CONTEXTS))
            self.plan[l] = (doc.to(dev), pos.to(dev), row.to(dev))
        self.cap = {}
        self.sparse = {l: {"driven": [], "with": []} for l in layers}  # per batch, on the CPU: (row, ids, vals); fires-with (row, id), flat
        self.dense = {l: {"heads": torch.zeros(2 * self.f, (l + 1) * self.H), "emb": torch.zeros(2 * self.f), "neurons": torch.zeros(2 * self.f),
                          "n": torch.zeros(2 * self.f), "err": []} for l in layers}
        self.at, self.pending = None, {}

    def hooks(self) -> list:
        return self.flow.hooks(self.cap, lambda h: h[self.at])

    def locate(self, a: int, b: int, T: int) -> None:
        """Documents ``a..b`` (rows ``0..``, ``T`` tokens): the positions some firing sits at, for the hooks to keep."""
        slices = {}
        for l in self.layers:
            doc, pos, row = self.plan[l]
            lo, hi = torch.searchsorted(doc, a), torch.searchsorted(doc, b)
            slices[l] = (doc[lo:hi] - a, pos[lo:hi], row[lo:hi])
        keys = torch.cat([s[0].long() * T + s[1].long() for s in slices.values()]) if slices else torch.zeros(0, dtype=torch.long, device=self.dev)
        uniq, inv = keys.unique(return_inverse=True)
        self.at = (uniq // T, uniq % T)
        self.pending, start = {}, 0
        for l, s in slices.items():
            self.pending[l] = (s[2], inv[start:start + len(s[0])])
            start += len(s[0])

    @torch.inference_mode()
    def split(self) -> None:
        for M, (row, u) in self.pending.items():
            if len(row):
                self.layer(M, row, u)
        self.cap.clear()

    def layer(self, M: int, row: torch.Tensor, u: torch.Tensor) -> None:
        cap, f = self.cap, self.f
        s = self.flow.sources(cap, M, row.long() % f, u)
        d = self.dense[M]
        r = row.long().cpu()
        d["heads"].index_add_(0, r, s["heads"].cpu())
        d["emb"].index_add_(0, r, s["emb"].cpu())
        d["neurons"].index_add_(0, r, s["neurons"].sum(1).cpu())
        d["n"].index_add_(0, r, torch.ones(len(r)))
        d["err"].append((s["emb"] + s["heads"].sum(1) + s["neurons"].sum(1) - 1).abs().amax().item())
        if M == 0:
            return
        acts = torch.cat([cap["a", l][u] for l in range(M)], 1)  # [N, M·f]
        pid = (torch.arange(M, device=self.dev)[:, None] * 2 * f + torch.arange(f, device=self.dev)).flatten() + f * (acts < 0)  # [N, M·f]
        top = s["neurons"].abs().topk(min(RANK, s["neurons"].shape[1]), dim=1).indices
        self.sparse[M]["driven"].append((r, pid.gather(1, top).int().cpu(), s["neurons"].gather(1, top).cpu()))
        fires = (acts.abs() >= FIRE * self.act_top[pid]) & (self.act_top[pid] > 0)  # a silent pole doesn't fire at zero
        lift = fires / self.rate[pid].clamp_min(self.rate_floor)
        top = lift.topk(min(RANK_WITH, lift.shape[1]), dim=1).indices
        n, k = fires.gather(1, top).nonzero(as_tuple=True)
        self.sparse[M]["with"].append((r[n.cpu()].int(), pid.gather(1, top)[n, k].int().cpu()))

    def state(self) -> dict:
        return {"dense": self.dense, "sparse": self.sparse}

    def load(self, state: dict) -> None:
        self.dense, self.sparse = state["dense"], state["sparse"]

    def table(self, M: int) -> dict:
        f, L, H = self.f, self.L, self.H
        d = self.dense[M]
        n = d["n"].clamp_min(1)
        cols = neuron_keys(M, f) | {"n_contexts": pa.array(d["n"].numpy(), pa.int16())}
        for k in ("emb", "neurons"):
            cols[f"{k}_share"] = pa.array((d[k] / n).numpy())
        mean_heads = d["heads"] / n[:, None]
        cols["heads_share"] = pa.array(mean_heads.sum(1).numpy())
        head_list(cols, "driven", mean_heads, H, d["n"] == 0)  # no firing: no sources
        driven = self.sparse[M]["driven"]
        if driven:
            r = torch.cat([x[0][:, None].expand_as(x[1]) for x in driven]).flatten()
            ids, vals = torch.cat([x[1] for x in driven]).flatten(), torch.cat([x[2] for x in driven]).flatten()
            rows, uids, total = merged(r, ids, vals, M * 2 * f)
            mean = total / n[rows]
            di, dv = top_by_row(rows, uids, mean, 2 * f, TOP)
            wit = self.sparse[M]["with"]
            r, ids = torch.cat([x[0] for x in wit]), torch.cat([x[1] for x in wit])
            wi, wl, wc = fires_with(r, ids, n, self.rate.cpu(), self.rate_floor, M * 2 * f, 2 * f)
        else:
            di, dv, wi, wl, wc = (torch.full((2 * f, TOP), -1), torch.full((2 * f, TOP), float("nan")), torch.full((2 * f, TOP), -1),
                                  torch.full((2 * f, TOP), float("nan")), torch.full((2 * f, TOP), float("nan")))
        pole_list(cols, "driven", di, dv, "share", f)
        pole_list(cols, "with", wi, wl, "lift", f)
        cols["with_co"] = fixed(wc.float())
        return cols


class HeadSources:
    """The ``heads/sources`` table: what drives each head at its first ``N_CONTEXTS`` top firings, per channel (module docstring).
    Every document with such a firing goes through the stack alone; ``run``, then ``table`` and ``summary``."""

    def __init__(self, stack, dev, positions: int):
        self.stack, self.dev, self.flow = stack, dev, Stream(stack)
        self.L, self.f, self.H = len(stack.layers), stack.config.intermediate_size, stack.config.num_attention_heads
        self.rate_floor = 1 / positions
        self.act_top = torch.stack([acts_column(l, "act_top") for l in range(self.L)]).to(dev).flatten()
        self.rate = torch.stack([acts_column(l, "rate") for l in range(self.L)]).to(dev).flatten()
        t = pq.read_table(head_table("acts"), columns=["top_doc", "top_pos", "top_val"])
        self.doc, self.pos, self.row = firing_plan(column(t, "top_doc", N_CONTEXTS).long(), column(t, "top_pos", N_CONTEXTS).long(), column(t, "top_val", N_CONTEXTS))
        n = self.L * self.H
        self.count = torch.zeros(n)
        self.dense = {c: {"emb": torch.zeros(n), "neurons": torch.zeros(n), "heads": torch.zeros(n, n)} for c in CHANNELS}
        self.sparse = {**{c: [] for c in CHANNELS}, "with": []}  # per firing: (row, ids, values); fires-with (row, ids)
        self.err = {c: torch.zeros(self.L) for c in CHANNELS}  # per layer, the largest departure from summing to 1

    def reads(self, m: layers.Attention, M: int, h: int, t: int, inv: torch.Tensor) -> dict:
        """Per channel, ``(R [n, d], pos [n])``: the reads of the residual entering layer ``M``, each at its position, whose
        sum over the sources is 1. ``m``: the layer's ``attention_map``, ``inv`` ``[T, 1]`` the input norm's scale."""
        lay = self.flow.layers[M]
        at, gamma = lay.self_attn, lay.input_layernorm.weight.float()
        hd, g, kv, dev = at.head_dim, at.num_key_value_groups, h // at.num_key_value_groups, self.dev
        query = h % g * inv.shape[0] + t
        q, p, k, v = m.qr[kv, query].float(), m.p[kv, query, :t + 1].float(), m.kr[kv, :t + 1].float(), m.vn[kv, :t + 1].float()
        back = lambda w, y: (y @ w.float()) * gamma  # a read y of a projection's output, as a read of its input
        start = self.flow.head_end[M - 1] if M else 0
        w = self.flow.all_heads[:, start + h * hd:start + (h + 1) * hd]
        o = w @ (p @ v)
        c = w.T @ o / o.pow(2).sum()
        rows = lambda W, i: W[i * hd:(i + 1) * hd]
        scaled = lambda r, *i: 1.0 if r is None else r[*i, 0]
        V = (p * scaled(m.rv, torch.arange(t + 1, device=dev), kv) * inv[:t + 1, 0])[:, None] * back(rows(m.wk if m.wv is None else m.wv, kv), c)[None]
        s = p[1:].argmax() + 1  # the most attended key past the sink
        margin = (k[s] - k[1:].mean(0)) * m.scale
        size = (q * margin).sum()
        Q = back(rows(m.wq, h), layers._rope(margin, m.cos[t, 0], -m.sin[t, 0]) * (1.0 if m.wqn is None else m.wqn))[None] * scaled(m.rq, t, h) * inv[t, 0] / size
        keys = torch.arange(1, t + 1, device=dev)
        weight = torch.full((t,), -1.0 / t, device=dev)
        weight[s - 1] += 1
        K = back(rows(m.wk, kv), layers._rope(q.expand(t, -1), m.cos[keys, 0], -m.sin[keys, 0]) * (1.0 if m.wkn is None else m.wkn))
        K = K * (weight * scaled(m.rk, keys, kv) * inv[keys, 0] * m.scale / size)[:, None]
        return {"v": (V, torch.arange(t + 1, device=dev)), "q": (Q, torch.tensor([t], device=dev)), "k": (K, keys)}

    def split(self, cap: dict, M: int, R: torch.Tensor, pos: torch.Tensor) -> tuple:
        """``(embedding, heads [M·H], neurons [2M·f])``: each source's part of a read ``R`` at ``pos``, the neurons in the pole
        their activation's sign gives."""
        s = self.flow.scalars
        e = (R * cap["emb"][pos]).sum() * s[:M].prod()
        if not M:
            return e, torch.zeros(0, device=self.dev), torch.zeros(0, device=self.dev)
        reach = lambda key, scale, l: cap[key, l][pos] * s[l:M].prod() * (cap[scale, l][pos][:, None] if (scale, l) in cap else 1)
        z = torch.cat([reach("z", "attn", l) for l in range(M)], 1)
        a = torch.cat([reach("a", "mlp", l) for l in range(M)], 1)
        C = self.flow.head_end[M - 1]
        heads = torch.zeros(M * self.H, device=self.dev).index_add_(0, self.flow.head_of[:C], ((R @ self.flow.all_heads[:, :C]) * z).sum(0))
        each = (R @ self.flow.all_neurons[:, :M * self.f]) * a
        return e, heads, torch.cat([(each * (a > 0)).sum(0).view(M, self.f), (each * (a < 0)).sum(0).view(M, self.f)], 1).flatten()

    def firing(self, cap: dict, m: layers.Attention, M: int, r: int, t: int, inv: torch.Tensor) -> None:
        self.count[r] += 1
        for c, (R, pos) in self.reads(m, M, r % self.H, t, inv).items():
            e, heads, neurons = self.split(cap, M, R, pos)
            d = self.dense[c]
            d["emb"][r] += e.cpu()
            d["neurons"][r] += neurons.sum().cpu()
            d["heads"][r, :M * self.H] += heads.cpu()
            self.err[c][M] = max(self.err[c][M].item(), (e + heads.sum() + neurons.sum() - 1).abs().item())
            if M:
                top = neurons.abs().topk(min(RANK, len(neurons))).indices
                self.sparse[c].append((r, top.cpu(), neurons[top].cpu()))
        if M:
            a = torch.cat([cap["a", l][t] for l in range(M)])
            pid = (torch.arange(M, device=self.dev)[:, None] * 2 * self.f + torch.arange(self.f, device=self.dev)).flatten() + self.f * (a < 0)
            fires = (a.abs() >= FIRE * self.act_top[pid]) & (self.act_top[pid] > 0)
            top = (fires / self.rate[pid].clamp_min(self.rate_floor)).topk(min(RANK_WITH, len(a))).indices
            self.sparse["with"].append((r, pid[top[fires[top]]].cpu()))

    @torch.inference_mode()
    def run(self, docs: list[torch.Tensor]) -> None:
        cap, flow = {}, self.flow
        hooks = flow.hooks(cap, lambda h: h[0])
        for l, lay in enumerate(flow.layers):
            hooks.append(lay.input_layernorm.register_forward_pre_hook(lambda mod, a, l=l: cap.__setitem__(("y", l), a[0][0].float())))
        keep = self.doc < len(docs)
        doc, pos, row = self.doc[keep], self.pos[keep].tolist(), self.row[keep].tolist()
        uniq, counts = doc.unique_consecutive(return_counts=True)
        at = 0
        for d, k in zip(uniq.tolist(), counts.tolist()):
            cap.clear()
            self.stack(input_ids=docs[d].long().to(self.dev)[None])
            maps = {}
            for t, r in zip(pos[at:at + k], row[at:at + k]):
                M = r // self.H
                if M not in maps:
                    y, lay = cap["y", M], flow.layers[M]
                    inv = torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + norm_eps(lay.input_layernorm))
                    maps[M] = (layers.attention_map(self.stack, M, lay.input_layernorm.weight.float() * y * inv), inv)
                self.firing(cap, *maps[M][:1], M, r, t, maps[M][1])
            at += k
        for x in hooks:
            x.remove()

    def table(self) -> dict:
        L, f, H = self.L, self.f, self.H
        n, silent = self.count.clamp_min(1), self.count == 0
        cols = head_keys(L, H, arch.groups(MODEL)) | {"n_contexts": pa.array(self.count.numpy(), pa.int16())}
        for c in CHANNELS:
            d = self.dense[c]
            cols[f"{c}_emb_share"], cols[f"{c}_neurons_share"] = pa.array((d["emb"] / n).numpy()), pa.array((d["neurons"] / n).numpy())
            mean = d["heads"] / n[:, None]
            cols[f"{c}_heads_share"] = pa.array(mean.sum(1).numpy())
            head_list(cols, f"{c}_driven", mean, H, silent)
            ids, values = torch.full((L * H, TOP), -1), torch.full((L * H, TOP), float("nan"))
            if self.sparse[c]:
                r = torch.cat([torch.full_like(i, x) for x, i, _ in self.sparse[c]])
                rows, uids, total = merged(r, torch.cat([i for _, i, _ in self.sparse[c]]), torch.cat([v for _, _, v in self.sparse[c]]), L * 2 * f)
                ids, values = top_by_row(rows, uids, total / n[rows], L * H, TOP)
            pole_list(cols, f"{c}_driven", ids, values, "share", f)
        ids, lift, co = torch.full((L * H, TOP), -1), torch.full((L * H, TOP), float("nan")), torch.full((L * H, TOP), float("nan"))
        if self.sparse["with"]:
            r = torch.cat([torch.full_like(i, x) for x, i in self.sparse["with"]])
            ids, lift, co = fires_with(r, torch.cat([i for _, i in self.sparse["with"]]), n, self.rate.cpu(), self.rate_floor, L * 2 * f, L * H)
        pole_list(cols, "with", ids, lift, "lift", f)
        cols["with_co"] = fixed(co.float())
        return cols

    def summary(self) -> dict:
        """Per layer and channel: the firings, the largest error and each kind of source's mean share."""
        out = {}
        for M in range(self.L):
            rows = range(M * self.H, (M + 1) * self.H)
            n = self.count[rows].sum().clamp_min(1)
            out[M] = {"firings": int(self.count[rows].sum())}
            for c in CHANNELS:
                d = self.dense[c]
                out[M][c] = {"err": self.err[c][M].item(), "emb": float(d["emb"][rows].sum() / n), "neurons": float(d["neurons"][rows].sum() / n), "heads": float(d["heads"][rows].sum() / n)}
        return out
