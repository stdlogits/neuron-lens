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
"""
from collections.abc import Callable

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from neuron_lens import arch
from neuron_lens.catalog import N_CONTEXTS, TOP, column, fixed, neuron_keys, neuron_table
from neuron_lens.firings import firing_plan
from neuron_lens.rules import norm_eps

RANK = 256  # candidates kept per firing for driven-by
RANK_WITH = 1024  # ... and for fires-with, where a source shared by several firings needs to survive each
FIRE = 1 / 4  # of act_top: the acts tables' rate threshold


class Stream:
    """A decoder stack's residual as a sum of writes, for any model ``arch``'s geometry covers."""

    def __init__(self, stack):
        self.layers, self.embed = stack.layers, stack.embed_tokens
        self.n_heads = stack.config.num_attention_heads
        self.heads = [arch.head_write(lay) for lay in self.layers]  # [d, H·hd]
        self.neurons = [arch.mlp_write(lay) for lay in self.layers]  # [d, f]
        self.scalars = torch.tensor([arch.layer_scale(lay) for lay in self.layers], device=self.neurons[0].device)

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
                     [reaching("z", "attn", l) for l in range(M + 1)], [reaching("a", "mlp", l) for l in range(M)],
                     self.neurons[:M], self.heads[:M + 1], self.n_heads, lay.mlp.act_fn)


def split(rg: torch.Tensor, ru: torch.Tensor, x: torch.Tensor, inv: torch.Tensor, emb: torch.Tensor,
          z: list[torch.Tensor], acts: list[torch.Tensor], downs: list[torch.Tensor], wos: list[torch.Tensor],
          n_heads: int, act: Callable) -> dict:
    """Per firing (rows): ``rg``, ``ru`` ``[N, d]``, the neuron's gate and up rows times the norm's weight;
    ``x`` ``[N, d]``, the residual entering the norm; ``inv`` ``[N, 1]``, the norm's scale; ``emb`` the
    embedding, ``z[l]`` ``[N, H·hd]`` the heads' pre-``o_proj`` outputs at layers ≤ ``M``, ``acts[l]`` ``[N, f]``
    the activations at layers < ``M``, these three scaled as they reach ``x``; ``downs``, ``wos`` the writes as
    the residual sees them; ``act`` the MLP's activation. Returns ``g``, ``a`` ``[N]`` and the shares ``emb`` ``[N]``,
    ``heads`` ``[N, (M + 1)·H]``, ``neurons`` ``[N, M·f]``, which sum to 1 over the three."""
    g, u = (rg * x).sum(-1) * inv[:, 0], (ru * x).sum(-1) * inv[:, 0]
    a = act(g) * u
    share = lambda gx, ux: 0.5 * (gx / g[:, None] + ux / u[:, None])  # ½ s (g u_X + u g_X) / (s g u)
    e = share((rg * emb).sum(-1, keepdim=True) * inv, (ru * emb).sum(-1, keepdim=True) * inv)[:, 0]
    heads = []
    for zl, wo in zip(z, wos):
        per = lambda r: ((r @ wo) * zl).view(len(rg), n_heads, -1).sum(-1) * inv
        heads.append(share(per(rg), per(ru)))
    neurons = [share((rg @ w) * al * inv, (ru @ w) * al * inv) for al, w in zip(acts, downs)]
    return {"g": g, "a": a, "emb": e, "heads": torch.cat(heads, 1),
            "neurons": torch.cat(neurons, 1) if neurons else torch.zeros(len(rg), 0, device=rg.device)}


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
        v, i = mean_heads.abs().topk(min(TOP, mean_heads.shape[1]), dim=1)
        pad = TOP - v.shape[1]
        i_full = torch.nn.functional.pad(i, (0, pad), value=-1)
        s_full = torch.nn.functional.pad(mean_heads.gather(1, i), (0, pad), value=float("nan"))
        silent = d["n"] == 0  # no firing: no sources
        i_full[silent], s_full[silent] = -1, float("nan")
        cols["driven_head_layer"] = fixed(torch.where(i_full >= 0, i_full // H, -1).to(torch.int16))
        cols["driven_head"] = fixed(torch.where(i_full >= 0, i_full % H, -1).to(torch.int16))
        cols["driven_head_share"] = fixed(s_full)
        driven = self.sparse[M]["driven"]
        if driven:
            r = torch.cat([x[0][:, None].expand_as(x[1]) for x in driven]).flatten()
            ids, vals = torch.cat([x[1] for x in driven]).flatten(), torch.cat([x[2] for x in driven]).flatten()
            rows, uids, total = merged(r, ids, vals, M * 2 * f)
            mean = total / n[rows]
            di, dv = top_by_row(rows, uids, mean, 2 * f, TOP)
            wit = self.sparse[M]["with"]
            r, ids = torch.cat([x[0] for x in wit]), torch.cat([x[1] for x in wit])
            rows, uids, total = merged(r, ids, torch.ones(len(ids)), M * 2 * f)
            enough = total >= torch.minimum(n[rows], torch.tensor(2.0))  # active at two of the pole's firings, or its one
            rows, uids, total = rows[enough], uids[enough], total[enough]
            co = total / n[rows]
            lift = co / self.rate.cpu()[uids].clamp_min(self.rate_floor)
            wi, wl = top_by_row(rows, uids, lift, 2 * f, TOP)
            _, wc = top_by_row(rows, uids, co, 2 * f, TOP, score=lift)
        else:
            di, dv, wi, wl, wc = (torch.full((2 * f, TOP), -1), torch.full((2 * f, TOP), float("nan")), torch.full((2 * f, TOP), -1),
                                  torch.full((2 * f, TOP), float("nan")), torch.full((2 * f, TOP), float("nan")))
        for name, ids_, value, value_name in (("driven", di, dv, "share"), ("with", wi, wl, "lift")):
            ok = ids_ >= 0
            layer, row = ids_.div(2 * f, rounding_mode="floor"), ids_ % (2 * f)
            cols[f"{name}_layer"] = fixed(torch.where(ok, layer, -1).to(torch.int16))
            cols[f"{name}_neuron"] = fixed(torch.where(ok, row % f, -1).to(torch.int16))
            cols[f"{name}_pole"] = fixed(torch.where(ok, 1 - 2 * (row // f), 0).to(torch.int8))
            cols[f"{name}_{value_name}"] = fixed(value.float())
        cols["with_co"] = fixed(wc.float())
        return cols
