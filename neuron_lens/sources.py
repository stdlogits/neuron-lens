"""What drives a neuron pole when it fires: an exact split of its activation over its sources (findings §12).

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

- **Value** (``v``, what it copies): with the attention pattern, the norms' scales, an output gate and the head's own
  output ``o`` held, ``o`` is linear in the residual at every position it attends to. A source's share is its part of
  ``o`` along ``o``, and the shares sum to 1. A Gated DeltaNet's head mixes its values through ``A`` (``o = A v``,
  ``layers.delta_pattern``), each value a SiLU of a causal convolution over the four positions up to it: with ``A``,
  SiLU's ratio ``σ(c)`` and the output's gate (its norm's scale and ``silu(z)``) held, ``o`` is linear in the residual
  at every position up to the query, and splits the same way. Its query, key, write strength and decay set ``A``
  together, a split this table leaves for later: a DeltaNet head's ``q`` and ``k`` columns are empty.
- **Query** and **key** (``q``, ``k``, why it looks there): the score ``q_t · k_s`` is bilinear. The number split is the
  most attended key's margin (past the sink) over the mean of the keys (past the sink), so ``q`` splits it over the
  sources at the query's position with the keys held, and ``k`` over the sources at the keys' positions with the query
  held. Each sums to 1, exactly for the score and to first order for the pattern it sets.

A model with biases (GPT-NeoX) adds a constant source, ``const``: the residual's bias writes, read as the rest are, and
the read's own constant (an ungated neuron's ``W_in β + b_in``, a head's ``W β + b`` for its value, query and key, ``β``
the LayerNorm's bias). With it the shares still sum to 1. An ungated activation ``φ(g) g`` splits as ``g`` does (the
identity rule; no half rule), and in a parallel block (GPT-NeoX's) a layer's MLP reads its input, so its heads' sources
are the earlier layers'. Where the norms are LayerNorms, the reads are centred and the scale is ``1 / std``.
"""
from collections.abc import Callable

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from neuron_lens import anatomy, arch, experts, layers
from neuron_lens.catalog import (
    MODEL,
    N_CONTEXTS,
    TOP,
    column,
    fixed,
    head_keys,
    head_rows,
    head_table,
    neuron_dtype,
    neuron_keys,
    neuron_table,
)
from neuron_lens.firings import firing_plan
from neuron_lens.rules import attention, is_centred, mlp_act, norm_eps, norm_weight

RANK = 256  # candidates kept per firing for driven-by
RANK_WITH = 1024  # ... and for fires-with, where a source shared by several firings needs to survive each
FIRE = 1 / 4  # of act_top: the acts tables' rate threshold
PIECE = 2**26  # entries in one [firings, M·f] table at a time: Qwen3-14B's late layers made 2 GB ones a batch
CHANNELS = ("v", "q", "k")  # a head's inputs: value, query, key


def _down(lay):
    return anatomy.module(lay, anatomy.parts(lay).down)


def _input_norm(lay):
    """The heads' input norm, or ``None`` where they read the residual itself (OLMo 2 and 3)."""
    return anatomy.module(lay, anatomy.parts(lay).n1)


def _per_head(w: torch.Tensor | None, r: torch.Tensor | None, head: int, pos, hd: int):
    """A q/k norm's scale (``[hd]``) and rsqrt (at ``pos``) for one head: per head (``r`` ``[T, H, 1]``), or OLMo's over the
    whole projection (``w`` ``[H·hd]``, ``r`` ``[T, 1]``, one rsqrt a position for every head)."""
    scale = 1.0 if w is None else (w if w.numel() == hd else w.view(-1, hd)[head])
    rs = 1.0 if r is None else (r[pos, 0] if r.dim() == 2 else r[pos, head, 0])
    return scale, rs


def _bias(mod, d: int, dev) -> torch.Tensor:
    return torch.zeros(d, device=dev) if mod is None or mod.bias is None else mod.bias.float()  # (a sparse MLP: none)


def joined(parts: list[torch.Tensor]) -> torch.Tensor:
    """``parts`` side by side, ``[d, Σ cols]``. A part that is the model's own weight (fp32, no norm folded in: Qwen3's
    ``o_proj`` and ``down_proj``) becomes a view of the result, so the stack isn't held twice: Qwen3-14B's fp32
    weights (59 GB) and a second copy of its writes (18.5 GB) don't fit an 80 GB GPU beside a pass."""
    whole = torch.empty(parts[0].shape[0], sum(p.shape[1] for p in parts), dtype=parts[0].dtype, device=parts[0].device)
    start = 0
    for p in parts:
        end = start + p.shape[1]
        whole[:, start:end] = p
        if isinstance(p, torch.nn.Parameter):
            p.data = whole[:, start:end]
        start = end
    return whole


class Stream:
    """A decoder stack's residual as a sum of writes, for any model ``arch``'s geometry covers."""

    def __init__(self, stack):
        self.layers, self.embed = stack.layers, stack.get_input_embeddings()
        geo = arch.layout(MODEL)
        self.n_heads = arch.max_heads(MODEL)  # heads on the grid: head h of layer l is l · n_heads + h
        first = stack.layers[0]
        self.parallel = anatomy.parts(first).parallel
        self.centred = any(is_centred(anatomy.module(first, n)) for n in anatomy.parts(first).norms)
        heads = [arch.head_write(lay) for lay in self.layers]  # [d, H·hd], hd may differ by layer (Gemma 4's full-attention layers)
        self.all_heads = joined(heads)  # a layer's sources are a prefix of these
        self.f = experts.stack_width(stack)  # a sparse layer's: its experts' neurons side by side, then its shared expert's
        # a sparse stack's writes stay its experts' own weights (Qwen3.5-35B-A3B's side by side would be 43 GB): its reads
        # are made a layer at a time (neuron_reads)
        self.sparse = any(experts.sparse(lay) for lay in self.layers)
        self.all_neurons = None if self.sparse else joined([arch.mlp_write(lay) for lay in self.layers])  # [d, L·f]
        self.neurons = None if self.sparse else self.all_neurons.split(self.f, 1)  # per layer, views of it
        self.rows = {}  # a sparse layer's gate and up rows (experts.layer_rows), by layer
        dev = self.all_heads.device
        self.head_end = torch.tensor([w.shape[1] for w in heads]).cumsum(0).tolist()  # layer l's heads end at this column
        self.head_of = torch.cat([l * self.n_heads + torch.arange(w.shape[1], device=dev) // geo[l].head_dim
                                  for l, w in enumerate(heads)])  # the head each column of all_heads belongs to
        self.scalars = torch.tensor([arch.layer_scale(lay) for lay in self.layers], device=dev)
        # the residual's constant (its writes' biases) entering each layer, and each MLP's input
        d = self.all_heads.shape[0]
        ab = [_bias(anatomy.module(lay, anatomy.parts(lay).out), d, dev) for lay in self.layers]
        cin = [torch.zeros(d, device=dev)]
        for l, lay in enumerate(self.layers):
            cin.append((cin[-1] + ab[l] + _bias(_down(lay), d, dev)) * self.scalars[l])
        self.bias_in = torch.stack(cin)  # [L + 1, d]
        self.bias_mid = torch.stack([cin[l] + (0 if self.parallel else ab[l]) for l in range(len(self.layers))])

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
            out.append(anatomy.module(lay, anatomy.parts(lay).out).register_forward_pre_hook(store(("z", l))))
            out.append(anatomy.module(lay, anatomy.parts(lay).mixer).register_forward_hook(  # (attention or a DeltaNet)
                scale(("attn", l), lambda o, lay=lay: arch.attn_scale(lay, o))))
            out.append(arch.mid_point(lay).register_forward_pre_hook(store(("x", l))))
            if l != upto and experts.sparse(lay):  # the activations at the picked positions alone (experts.Picked)
                out += self._sparse_hook(cap, at, l, lay)
                out.append(lay.mlp.register_forward_hook(scale(("mlp", l), lambda o, lay=lay: arch.mlp_scale(lay, o))))
            elif l != upto:
                out.append(_down(lay).register_forward_pre_hook(store(("a", l))))
                out.append(lay.mlp.register_forward_hook(scale(("mlp", l), lambda o, lay=lay: arch.mlp_scale(lay, o))))
        return out

    def _sparse_hook(self, cap: dict, at: Callable, l: int, lay) -> list:
        picked = {}

        def rows(shape):  # the rows of the positions ``at`` picks, in its order
            idx = at(torch.arange(shape.numel(), device=self.all_heads.device).view(shape)).flatten()
            picked["rows"] = idx
            mask = torch.zeros(shape.numel(), dtype=torch.bool, device=idx.device)
            mask[idx] = True
            return mask
        return experts.Tap(lay.mlp, lambda r: cap.__setitem__(("a", l), experts.Picked(r, picked["rows"])), rows).hooks

    def layer_reads(self, r: torch.Tensor, l: int) -> torch.Tensor:
        """``[n, f]``: read rows ``r`` ``[n, d]`` against layer ``l``'s neurons' writes."""
        return experts.layer_write_reads(self.layers[l], r) if self.sparse else times(r, self.neurons[l])

    def neuron_reads(self, r: torch.Tensor, M: int) -> torch.Tensor:
        """``[n, M·f]``: read rows ``r`` against the writes of every neuron of layers ``< M``."""
        if not self.sparse:
            return times(r, self.all_neurons[:, :M * self.f])
        return torch.cat([self.layer_reads(r, l) for l in range(M)], 1) if M else r[:, :0]

    def read_rows(self, lay, l: int, neuron: torch.Tensor, gamma) -> tuple:
        """The gate and up rows of ``neuron`` times the MLP input norm's scale ``gamma``, and the gate's bias."""
        p = anatomy.parts(lay)
        if p.sparse:
            if l not in self.rows:
                self.rows[l] = experts.layer_rows(lay)
            return self.rows[l]["gate"][neuron] * gamma, self.rows[l]["up"][neuron] * gamma, None
        gp, up = anatomy.module(lay, p.gate), anatomy.module(lay, p.up)
        return gp.weight[neuron] * gamma, None if up is None else up.weight[neuron] * gamma, gp

    def sources(self, cap: dict, M: int, neuron: torch.Tensor, u: torch.Tensor,
                acts_raw: torch.Tensor | None = None) -> dict:
        """``split`` for the firings ``u`` (indices into ``cap``'s positions) of ``neuron`` ``[N]`` in layer ``M``.
        Pass ``acts_raw`` (``[N, M·f]`` unscaled activations) to avoid concatenating them twice when no post-MLP norm
        or layer scalar rescales them."""
        lay, norm = self.layers[M], arch.mid_norm(self.layers[M])  # no norm: the MLP reads the residual itself (OLMo)
        x = cap["x", M][u]
        xc = x - x.mean(-1, keepdim=True) if self.centred else x
        inv = torch.ones_like(x[:, :1]) if norm is None else torch.rsqrt(xc.pow(2).mean(-1) + norm_eps(norm))[:, None]
        mlp, p = lay.mlp, anatomy.parts(lay)
        gamma = 1.0 if norm is None else norm_weight(norm)
        rg, ru, gp = self.read_rows(lay, M, neuron, gamma)  # (a routed neuron's routing weight cancels in its shares)
        b0 = torch.zeros(len(neuron), device=x.device) if gp is None or gp.bias is None else gp.bias[neuron].float()
        if getattr(norm, "bias", None) is not None:  # W_in β
            b0 = b0 + gp.weight[neuron].float() @ norm.bias.float()
        if self.centred:
            rg = rg - rg.mean(-1, keepdim=True)
            ru = None if ru is None else ru - ru.mean(-1, keepdim=True)
        nh = M + (not self.parallel)  # the layers whose heads write the MLP's input
        end = self.head_end[nh - 1] if nh else 0

        def reaching(key: str, scale: str, l: int) -> torch.Tensor:
            v = cap[key, l][u]
            sc = self.scalars[l:M].prod()
            if sc != 1.0:
                v = v * sc
            return v * cap[scale, l][u][:, None] if (scale, l) in cap else v
        if not M:
            acts = x[:, :0]
        elif acts_raw is not None and all(("mlp", l) not in cap for l in range(M)) and (self.scalars[:M] == 1.0).all():
            acts = acts_raw
        else:
            acts = torch.cat([reaching("a", "mlp", l) for l in range(M)], 1)
        return split(rg, ru, x, inv, cap["emb"][u] * self.scalars[:M].prod(),
                     torch.cat([reaching("z", "attn", l) for l in range(nh)], 1) if nh else x[:, :0],
                     acts,
                     lambda r: self.neuron_reads(r, M), self.all_heads[:, :end],
                     self.head_of[:end], nh * self.n_heads, mlp.experts.act_fn if p.sparse else mlp_act(mlp),
                     self.bias_mid[M], b0)


def times(r: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``r @ w``; on MPS, whose tensors stop at 2³¹ elements (Gemma 4's writes from 37 layers up pass it), in column
    blocks under it."""
    if w.device.type != "mps" or w.untyped_storage().nbytes() < 2**31 * w.element_size():  # a view counts its base
        return r @ w
    step = 2**30 // w.shape[0]
    return torch.cat([r @ w[:, i:i + step].contiguous() for i in range(0, w.shape[1], step)], 1)


def split(rg: torch.Tensor, ru: torch.Tensor | None, x: torch.Tensor, inv: torch.Tensor, emb: torch.Tensor,
          z: torch.Tensor, acts: torch.Tensor, downs: Callable, wos: torch.Tensor, head_of: torch.Tensor, n_heads: int,
          act: Callable, bias: torch.Tensor, b0: torch.Tensor) -> dict:
    """Per firing (rows): ``rg``, ``ru`` ``[N, d]``, the neuron's gate and up rows times the norm's weight;
    ``x`` ``[N, d]``, the residual entering the norm; ``inv`` ``[N, 1]``, the norm's scale; ``emb`` the
    embedding, ``z`` ``[N, C]`` the heads' pre-``o_proj`` outputs at layers ≤ ``M``, ``acts`` ``[N, M·f]``
    the activations at layers < ``M``, these three scaled as they reach ``x``; ``downs(r)`` ``[N, M·f]``, read rows
    against the neurons' writes as the residual sees them (``Stream.neuron_reads``), ``wos``
    ``[d, C]`` the heads'; ``head_of`` ``[C]`` the head each of ``C`` columns belongs to, of
    ``n_heads``; ``act`` the MLP's activation; ``bias`` ``[d]`` the residual's constant, ``b0`` ``[N]`` the pre-activation's
    own. Returns ``g``, ``a`` ``[N]`` and the shares ``emb``, ``const`` ``[N]``, ``heads`` ``[N, n_heads]``, ``neurons``
    ``[N, M·f]``, which sum to 1 over the four. Ungated (``ru`` None): ``a = φ(g) g`` splits as ``g`` (rows centred
    where the norm centres)."""
    if ru is None:
        g = (rg * x).sum(-1) * inv[:, 0] + b0
        r_eff = (inv / g[:, None]) * rg
        heads = torch.zeros(len(rg), n_heads, device=z.device).index_add_(1, head_of, (r_eff @ wos) * z)
        return {"g": g, "a": act(g), "emb": (r_eff * emb).sum(-1),
                "const": r_eff @ bias + b0 / g, "heads": heads,
                "neurons": downs(r_eff) * acts}
    g, u = (rg * x).sum(-1) * inv[:, 0], (ru * x).sum(-1) * inv[:, 0]
    a = act(g) * u
    # By linearity, ½ (gx / g + ux / u) collapses into one effective read row r_eff = ½ inv (rg / g + ru / u),
    # halving the large [N, d] @ [d, M·f] and [N, d] @ [d, C] GEMMs exactly.
    r_eff = (0.5 * inv / g[:, None]) * rg + (0.5 * inv / u[:, None]) * ru
    heads = torch.zeros(len(rg), n_heads, device=z.device).index_add_(1, head_of, (r_eff @ wos) * z)
    return {"g": g, "a": a, "emb": (r_eff * emb).sum(-1), "const": r_eff @ bias, "heads": heads,
            "neurons": downs(r_eff) * acts}


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
    none = torch.nn.functional.pad(v == 0, (0, pad), value=True)  # fewer heads before it than TOP: the rest are no source
    ids[silent], share[silent] = -1, float("nan")
    ids[none], share[none] = -1, float("nan")
    cols[f"{name}_head_layer"] = fixed(torch.where(ids >= 0, ids // H, -1).to(torch.int16))
    cols[f"{name}_head"] = fixed(torch.where(ids >= 0, ids % H, -1).to(torch.int16))
    cols[f"{name}_head_share"] = fixed(share)


def pole_list(cols: dict, name: str, ids: torch.Tensor, value: torch.Tensor, value_name: str, f: int) -> None:
    """``<name>_layer``, ``<name>_neuron``, ``<name>_pole``, ``<name>_<value_name>`` from pole ids ``layer · P·f + row``, ``−1``
    for none (``P`` the poles a neuron has, ``arch.n_poles``)."""
    ok = ids >= 0
    P = arch.n_poles(MODEL)
    layer, row = ids.div(P * f, rounding_mode="floor"), ids % (P * f)
    cols[f"{name}_layer"] = fixed(torch.where(ok, layer, -1).to(torch.int16))
    cols[f"{name}_neuron"] = fixed(torch.where(ok, row % f, -1).to(neuron_dtype(f)))
    cols[f"{name}_pole"] = fixed(torch.where(ok, 1 - 2 * (row // f), 0).to(torch.int8))
    cols[f"{name}_{value_name}"] = fixed(value.float())


def acts_column(layer: int, name: str) -> torch.Tensor:
    return torch.from_numpy(pq.read_table(neuron_table("acts", layer), columns=[name]).column(0).to_numpy().copy()).float()


class Sources:
    """The ``sources`` tables, filled over a forward pass: ``locate`` a batch's firings, run the stack, ``split`` them."""

    def __init__(self, stack, layers: list[int], dev, positions: int):
        self.stack, self.layers, self.dev, self.flow = stack, layers, dev, Stream(stack)
        self.rate_floor = 1 / positions  # a pole seen firing at all fires at least once
        self.L, self.f, self.H = len(stack.layers), self.flow.f, self.flow.n_heads  # f: a sparse layer's experts' and shared
        # expert's neurons (the config's is a dense MLP's); heads on the grid
        self.P = arch.n_poles(MODEL)
        self.act_top = torch.stack([acts_column(l, "act_top") for l in range(self.L)]).to(dev)  # [L, P·f]
        self.rate = torch.stack([acts_column(l, "rate") for l in range(self.L)]).to(dev)  # [L, P·f]
        # Pre-split per-pole thresholds and inverse rates by sign (avoids materializing [N, M·f] int64 pole ids).
        at_p = self.act_top.view(self.L, self.P, self.f)
        rt_p = self.rate.view(self.L, self.P, self.f).clamp_min(self.rate_floor)
        self.thresh_pos = (FIRE * at_p[:, 0]).reshape(-1)
        self.inv_rate_pos = (1.0 / rt_p[:, 0]).reshape(-1)
        if self.P == 2:
            self.thresh_neg = (FIRE * at_p[:, 1]).reshape(-1)
            self.inv_rate_neg = (1.0 / rt_p[:, 1]).reshape(-1)
        self.act_top = self.act_top.flatten()
        self.rate = self.rate.flatten()
        self.plan = {}
        for l in layers:
            t = pq.read_table(neuron_table("acts", l), columns=["top_doc", "top_pos", "top_val"])
            doc, pos, row = firing_plan(column(t, "top_doc", N_CONTEXTS).long(), column(t, "top_pos", N_CONTEXTS).long(), column(t, "top_val", N_CONTEXTS))
            self.plan[l] = (doc.to(dev), pos.to(dev), row.to(dev))
        self.cap = {}
        self.sparse = {l: {"driven": [], "with": []} for l in layers}  # per batch, on the CPU: (row, ids, vals); fires-with (row, id), flat
        rows, nh = self.P * self.f, lambda l: l + (not self.flow.parallel)  # a layer's poles; the layers whose heads reach its MLP
        # Keep small dense accumulators on device; keep `heads` on device when total footprint fits comfortably.
        heads_bytes = sum(rows * nh(l) * self.H * 4 for l in layers)
        heads_dev = dev if (not self.flow.sparse and heads_bytes <= 4 * 2**30) else "cpu"
        self.heads_dev = heads_dev
        self.dense = {l: {"heads": torch.zeros(rows, nh(l) * self.H, device=heads_dev),
                          "emb": torch.zeros(rows, device=dev), "neurons": torch.zeros(rows, device=dev),
                          "const": torch.zeros(rows, device=dev), "n": torch.zeros(rows, device=dev),
                          "err": torch.zeros((), device=dev)} for l in layers}
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
            step = max(1, PIECE // max(1, M * self.f))  # firings at a time (each one's split is its own)
            for i in range(0, len(row), step):
                self.layer(M, row[i:i + step], u[i:i + step])
        self.cap.clear()

    def _col_to_pid(self, col: torch.Tensor, neg: torch.Tensor | None) -> torch.Tensor:
        f = self.f
        pid = (col // f) * (self.P * f) + (col % f)
        return pid + f * neg.long() if neg is not None else pid

    def layer(self, M: int, row: torch.Tensor, u: torch.Tensor) -> None:
        cap, f = self.cap, self.f
        acts = torch.cat([cap["a", l][u] for l in range(M)], 1) if M else None  # [N, M·f], reused by flow.sources
        s = self.flow.sources(cap, M, row.long() % f, u, acts_raw=acts)
        d = self.dense[M]
        r_gpu = row.long()
        d["heads"].index_add_(0, r_gpu if self.heads_dev == self.dev else r_gpu.cpu(),
                              s["heads"] if self.heads_dev == self.dev else s["heads"].cpu())
        nsum = s["neurons"].sum(1)
        d["emb"].index_add_(0, r_gpu, s["emb"])
        d["neurons"].index_add_(0, r_gpu, nsum)
        d["const"].index_add_(0, r_gpu, s["const"])
        d["n"].index_add_(0, r_gpu, torch.ones(len(r_gpu), device=self.dev))
        d["err"] = torch.maximum(d["err"], (s["emb"] + s["const"] + s["heads"].sum(1) + nsum - 1).abs().amax())
        if M == 0:
            return
        Mf = M * f
        top = s["neurons"].abs().topk(min(RANK, Mf), dim=1).indices
        if self.P == 2:
            neg = acts < 0
            top_pid = self._col_to_pid(top, neg.gather(1, top))
            thresh = torch.where(neg, self.thresh_neg[:Mf], self.thresh_pos[:Mf])
            fires = (acts.abs() >= thresh) & (thresh > 0)
            inv_rate = torch.where(neg, self.inv_rate_neg[:Mf], self.inv_rate_pos[:Mf])
        else:
            neg = None
            top_pid = top
            thresh = self.thresh_pos[:Mf]
            fires = (acts >= thresh) & (thresh > 0)
            inv_rate = self.inv_rate_pos[:Mf]
        r_cpu = r_gpu.cpu()
        self.sparse[M]["driven"].append((r_cpu, top_pid.int().cpu(), s["neurons"].gather(1, top).cpu()))
        # Sparse fires-with: when few columns fire per row (typical), filter via nonzero instead of dense topk over M·f.
        lift = torch.where(fires, inv_rate, 0.0)
        top_w = lift.topk(min(RANK_WITH, Mf), dim=1).indices
        n, k = fires.gather(1, top_w).nonzero(as_tuple=True)
        col_w = top_w[n, k]
        pid_w = self._col_to_pid(col_w, neg[n, col_w] if neg is not None else None)
        self.sparse[M]["with"].append((r_gpu[n].int().cpu(), pid_w.int().cpu()))

    def state(self) -> dict:
        dense = {l: {k: v.cpu() for k, v in d.items()} for l, d in self.dense.items()}
        return {"dense": dense, "sparse": self.sparse}

    def load(self, state: dict) -> None:
        self.dense = {}
        for l, d in state["dense"].items():
            err = d["err"]
            err = torch.tensor(max(err) if err else 0.0, device=self.dev) if isinstance(err, list) else err.to(self.dev)
            self.dense[l] = {
                "heads": d["heads"].to(self.heads_dev),
                "emb": d["emb"].to(self.dev), "neurons": d["neurons"].to(self.dev),
                "const": d["const"].to(self.dev), "n": d["n"].to(self.dev), "err": err,
            }
        self.sparse = state["sparse"]

    def table(self, M: int) -> dict:
        f, L, H = self.f, self.L, self.H
        Pf = self.P * f
        d = {k: v.cpu() for k, v in self.dense[M].items()}
        n = d["n"].clamp_min(1)
        cols = neuron_keys(M, f) | {"n_contexts": pa.array(d["n"].numpy(), pa.int16())}
        for k in ("emb", "neurons", "const"):
            cols[f"{k}_share"] = pa.array((d[k] / n).numpy())
        mean_heads = d["heads"] / n[:, None]
        cols["heads_share"] = pa.array(mean_heads.sum(1).numpy())
        head_list(cols, "driven", mean_heads, H, d["n"] == 0)  # no firing: no sources
        driven = self.sparse[M]["driven"]
        if driven:
            r = torch.cat([x[0][:, None].expand_as(x[1]) for x in driven]).flatten()
            ids, vals = torch.cat([x[1] for x in driven]).flatten(), torch.cat([x[2] for x in driven]).flatten()
            rows, uids, total = merged(r, ids, vals, M * Pf)
            mean = total / n[rows]
            di, dv = top_by_row(rows, uids, mean, Pf, TOP)
            wit = self.sparse[M]["with"]
            r, ids = torch.cat([x[0] for x in wit]), torch.cat([x[1] for x in wit])
            wi, wl, wc = fires_with(r, ids, n, self.rate.cpu(), self.rate_floor, M * Pf, Pf)
        else:
            di, dv, wi, wl, wc = (torch.full((Pf, TOP), -1), torch.full((Pf, TOP), float("nan")), torch.full((Pf, TOP), -1),
                                  torch.full((Pf, TOP), float("nan")), torch.full((Pf, TOP), float("nan")))
        pole_list(cols, "driven", di, dv, "share", f)
        pole_list(cols, "with", wi, wl, "lift", f)
        cols["with_co"] = fixed(wc.float())
        return cols


class HeadSources:
    """The ``heads/sources`` table: what drives each head at its first ``N_CONTEXTS`` top firings, per channel (module docstring).
    Every document with such a firing goes through the stack alone; ``run``, then ``table`` and ``summary``."""

    def __init__(self, stack, dev, positions: int):
        self.stack, self.dev, self.flow = stack, dev, Stream(stack)
        self.L, self.f, self.H = len(stack.layers), self.flow.f, self.flow.n_heads  # f: a sparse layer's experts' and shared
        # expert's neurons (the config's is a dense MLP's); heads on the grid
        self.P = arch.n_poles(MODEL)
        self.rate_floor = 1 / positions
        self.act_top = torch.stack([acts_column(l, "act_top") for l in range(self.L)]).to(dev).flatten()
        self.rate = torch.stack([acts_column(l, "rate") for l in range(self.L)]).to(dev).flatten()
        t = pq.read_table(head_table("acts"), columns=["top_doc", "top_pos", "top_val"])
        self.doc, self.pos, row = firing_plan(column(t, "top_doc", N_CONTEXTS).long(), column(t, "top_pos", N_CONTEXTS).long(), column(t, "top_val", N_CONTEXTS))
        self.row = head_rows()[row.long()]  # the table's rows on the grid
        n = self.L * self.H
        self.count = torch.zeros(n)
        self.dense = {c: {"emb": torch.zeros(n), "neurons": torch.zeros(n), "const": torch.zeros(n), "heads": torch.zeros(n, n)}
                      for c in CHANNELS}
        self.sparse = {**{c: [] for c in CHANNELS}, "with": []}  # per firing: (row, ids, values); fires-with (row, ids)
        self.err = {c: torch.zeros(self.L) for c in CHANNELS}  # per layer, the largest departure from summing to 1

    def reads(self, m: layers.Attention, M: int, h: int, t: int, inv: torch.Tensor) -> dict:
        """Per channel, ``(R [n, d], pos [n], const)``: the reads of the residual entering layer ``M``, each at its position,
        and the read's own constant (its projection's and the norm's biases), whose sum over the sources is 1. ``m``: the
        layer's ``attention_map``, ``inv`` ``[T, 1]`` the input norm's scale."""
        lay = self.flow.layers[M]
        at, ln = attention(lay), _input_norm(lay)
        gamma = 1.0 if ln is None else norm_weight(ln).float()
        hd = at.head_dim if hasattr(at, "head_dim") else at.head_size
        g = getattr(at, "num_key_value_groups", 1)
        kv, dev = h // g, self.dev
        query = h % g * inv.shape[0] + t
        q, p, k, v = m.qr[kv, query].float(), m.p[kv, query, :t + 1].float(), m.kr[kv, :t + 1].float(), m.vn[kv, :t + 1].float()

        def back(w, y):  # a read y of a projection's output, as a read of its input (centred, where the norm centres)
            r = (y @ w.float()) * gamma
            return r - r.mean(-1, keepdim=True) if self.flow.centred else r
        start = self.flow.head_end[M - 1] if M else 0
        w = self.flow.all_heads[:, start + h * hd:start + (h + 1) * hd]
        if m.sg is not None:  # gated attention: the gate at the query, held
            w = w * m.sg[t, h * hd:(h + 1) * hd]
        o = w @ (p @ v)
        c = w.T @ o / o.pow(2).sum()
        if m.wqkv is not None:  # GPT-NeoX's fused projection: per head the query, key, value rows
            W, B = m.wqkv.view(-1, 3, hd, m.wqkv.shape[-1]), at.query_key_value.bias
            Wq, Wk, Wv = W[h, 0], W[kv, 1], W[kv, 2]
            bq, bk, bv = (None, None, None) if B is None else B.float().view(-1, 3, hd)[[h, kv, kv], [0, 1, 2]]
        else:
            rows = lambda W, i: W[i * hd:(i + 1) * hd]
            Wq = m.wq[2 * h * hd:(2 * h + 1) * hd] if m.sg is not None else rows(m.wq, h)  # gated: the query rows, then the gate's
            Wk, Wv = rows(m.wk, kv), rows(m.wk if m.wv is None else m.wv, kv)
            bias = lambda mod, i: None if mod is None or mod.bias is None else rows(mod.bias.float(), i)
            bq, bk, bv = bias(at.q_proj, h), bias(at.k_proj, kv), bias(getattr(at, "v_proj", None) or at.k_proj, kv)
        beta = getattr(ln, "bias", None) if ln is not None else None

        def own(W, b):  # the projection's output for a zero residual: W β + b
            out = torch.zeros(hd, device=dev) if b is None else b
            return out if beta is None else out + W.float() @ beta.float()
        scaled = lambda r, *i: 1.0 if r is None else r[*i, 0]
        V = (p * scaled(m.rv, torch.arange(t + 1, device=dev), kv) * inv[:t + 1, 0])[:, None] * back(Wv, c)[None]
        s = p[1:].argmax() + 1  # the most attended key past the sink
        margin = (k[s] - k[1:].mean(0)) * m.scale
        size = (q * margin).sum()
        mq = layers._rope(margin, m.cos[t, 0], -m.sin[t, 0])  # the margin as a read of the query, before RoPE
        wq, rq = _per_head(m.wqn, m.rq, h, t, hd)
        Q = back(Wq, mq * wq)[None] * rq * inv[t, 0] / size
        keys = torch.arange(1, t + 1, device=dev)
        weight = torch.full((t,), -1.0 / t, device=dev)
        weight[s - 1] += 1
        qk = layers._rope(q.expand(t, -1), m.cos[keys, 0], -m.sin[keys, 0])  # the query as a read of each key, before RoPE
        wk, rk = _per_head(m.wkn, m.rk, kv, keys, hd)
        K = back(Wk, qk * wk)
        K = K * (weight * rk * inv[keys, 0] * m.scale / size)[:, None]
        consts = ((c * own(Wv, bv)).sum(), (mq * own(Wq, bq)).sum() / size,  # (no biases, no LayerNorm: zero)
                  (weight * (qk @ own(Wk, bk))).sum() * m.scale / size)
        return {"v": (V, torch.arange(t + 1, device=dev), consts[0]), "q": (Q, torch.tensor([t], device=dev), consts[1]),
                "k": (K, keys, consts[2])}

    def delta_reads(self, d: layers.DeltaPattern, M: int, h: int, t: int, inv: torch.Tensor) -> dict:
        """A DeltaNet head's value channel (module docstring), as ``reads``'s: the reads of the residual entering layer
        ``M`` at every position up to ``t``. ``d``: the document's ``delta_pattern``, ``inv`` ``[T, 1]`` the input norm's
        scale."""
        lay = self.flow.layers[M]
        la, ln, g = anatomy.module(lay, anatomy.parts(lay).mixer), _input_norm(lay), arch.layout(MODEL)[M]
        gamma = 1.0 if ln is None else norm_weight(ln).float()
        dv, kd2 = g.head_dim, 2 * la.key_dim
        start = self.flow.head_end[M - 1] if M else 0
        w = self.flow.all_heads[:, start + h * dv:start + (h + 1) * dv] * d.gate[0, t, h]  # the gate at the query, held
        cols = slice(kd2 + h * dv, kd2 + (h + 1) * dv)  # the head's value channels of the convolution
        sv = torch.sigmoid(d.c[0, :t + 1, cols])  # SiLU's ratio, held
        v = sv * d.c[0, :t + 1, cols]
        o = w @ (d.A[0, h, t, :t + 1] @ v)
        c = w.T @ o / o.pow(2).sum()
        coef = layers._conv_t((d.A[0, h, t, :t + 1, None] * sv)[None], la.conv1d.weight.squeeze(1)[cols].float())[0]
        R = (coef * c) @ layers.delta_parts(la).qkv[cols].float() * gamma * inv[:t + 1]
        return {"v": (R, torch.arange(t + 1, device=self.dev), torch.zeros((), device=self.dev))}

    def split(self, cap: dict, M: int, R: torch.Tensor, pos: torch.Tensor) -> tuple:
        """``(embedding, heads [M·H], neurons [M·P·f], const)``: each source's part of a read ``R`` at ``pos``, the neurons in
        the pole their activation's sign gives, and the residual's constant (its writes' biases)."""
        s = self.flow.scalars
        e = (R * cap["emb"][pos]).sum() * s[:M].prod()
        b = (R * self.flow.bias_in[M]).sum()
        if not M:
            return e, torch.zeros(0, device=self.dev), torch.zeros(0, device=self.dev), b
        reach = lambda key, scale, l: cap[key, l][pos] * s[l:M].prod() * (cap[scale, l][pos][:, None] if (scale, l) in cap else 1)
        z = torch.cat([reach("z", "attn", l) for l in range(M)], 1)
        C = self.flow.head_end[M - 1]
        heads = torch.zeros(M * self.H, device=self.dev).index_add_(0, self.flow.head_of[:C], ((R @ self.flow.all_heads[:, :C]) * z).sum(0))
        if self.flow.sparse:  # a layer at a time: a sparse stack's [positions, M·f] tables would be gigabytes
            parts = []
            for l in range(M):
                a = reach("a", "mlp", l)
                each = self.flow.layer_reads(R, l) * a
                parts.append(each.sum(0) if self.P == 1 else torch.cat([(each * (a > 0)).sum(0), (each * (a < 0)).sum(0)]))
            return e, heads, torch.cat(parts), b
        a = torch.cat([reach("a", "mlp", l) for l in range(M)], 1)
        each = (R @ self.flow.all_neurons[:, :M * self.f]) * a
        if self.P == 1:
            return e, heads, each.sum(0), b
        return e, heads, torch.cat([(each * (a > 0)).sum(0).view(M, self.f), (each * (a < 0)).sum(0).view(M, self.f)], 1).flatten(), b

    def firing(self, cap: dict, m: layers.Attention | layers.DeltaPattern, M: int, r: int, t: int, inv: torch.Tensor) -> None:
        self.count[r] += 1
        reads = self.delta_reads if isinstance(m, layers.DeltaPattern) else self.reads
        for c, (R, pos, own) in reads(m, M, r % self.H, t, inv).items():
            e, heads, neurons, b = self.split(cap, M, R, pos)
            d = self.dense[c]
            d["emb"][r] += e.cpu()
            d["neurons"][r] += neurons.sum().cpu()
            d["const"][r] += (b + own).cpu()
            d["heads"][r, :M * self.H] += heads.cpu()
            self.err[c][M] = max(self.err[c][M].item(), (e + b + own + heads.sum() + neurons.sum() - 1).abs().item())
            if M:
                top = neurons.abs().topk(min(RANK, len(neurons))).indices
                self.sparse[c].append((r, top.cpu(), neurons[top].cpu()))
        if M:
            a = torch.cat([cap["a", l][t] for l in range(M)])
            pid = (torch.arange(M, device=self.dev)[:, None] * self.P * self.f + torch.arange(self.f, device=self.dev)).flatten()
            pid, size = (pid + self.f * (a < 0), a.abs()) if self.P == 2 else (pid, a)
            fires = (size >= FIRE * self.act_top[pid]) & (self.act_top[pid] > 0)
            top = (fires / self.rate[pid].clamp_min(self.rate_floor)).topk(min(RANK_WITH, len(a))).indices
            self.sparse["with"].append((r, pid[top[fires[top]]].cpu()))

    @torch.inference_mode()
    def run(self, docs: list[torch.Tensor]) -> None:
        cap, flow = {}, self.flow
        hooks = flow.hooks(cap, lambda h: h[0])
        for l, lay in enumerate(flow.layers):  # the residual the heads read: their input norm's input, or the layer's
            ln = _input_norm(lay)
            hooks.append(ln.register_forward_pre_hook(lambda mod, a, l=l: cap.__setitem__(("y", l), a[0][0].float())) if ln is not None
                         else lay.register_forward_pre_hook(lambda mod, a, kw, l=l: cap.__setitem__(
                             ("y", l), (a[0] if a else kw["hidden_states"])[0].float()), with_kwargs=True))
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
                    y, ln = cap["y", M], _input_norm(flow.layers[M])
                    yc = y - y.mean(-1, keepdim=True) if flow.centred else y
                    inv = torch.ones_like(y[:, :1]) if ln is None else torch.rsqrt(yc.pow(2).mean(-1, keepdim=True) + norm_eps(ln))
                    lay, xn = flow.layers[M], layers._normed(ln, y)[0]
                    mix = (layers.delta_pattern(anatomy.module(lay, anatomy.parts(lay).mixer), xn[None]) if anatomy.parts(lay).deltanet
                           else layers.attention_map(self.stack, M, xn))
                    maps[M] = (mix, inv)
                self.firing(cap, *maps[M][:1], M, r, t, maps[M][1])
            at += k
        for x in hooks:
            x.remove()

    def table(self) -> dict:
        L, f, H = self.L, self.f, self.H
        n, silent = self.count.clamp_min(1), self.count == 0
        delta = torch.tensor([g.deltanet for g in arch.layout(MODEL)]).repeat_interleave(H)  # [L·H]: q, k not split
        cols = head_keys() | {"n_contexts": pa.array(self.count.numpy(), pa.int16())}
        for c in CHANNELS:
            d = self.dense[c]
            none = delta & (c != "v")
            share = lambda x, none=none: pa.array(torch.where(none, float("nan"), x / n).numpy())
            cols[f"{c}_emb_share"], cols[f"{c}_neurons_share"] = share(d["emb"]), share(d["neurons"])
            cols[f"{c}_const_share"] = share(d["const"])
            mean = d["heads"] / n[:, None]
            cols[f"{c}_heads_share"] = share(d["heads"].sum(1))
            head_list(cols, f"{c}_driven", mean, H, silent | none)
            ids, values = torch.full((L * H, TOP), -1), torch.full((L * H, TOP), float("nan"))
            if self.sparse[c]:
                r = torch.cat([torch.full_like(i, x) for x, i, _ in self.sparse[c]])
                rows, uids, total = merged(r, torch.cat([i for _, i, _ in self.sparse[c]]), torch.cat([v for _, _, v in self.sparse[c]]), L * self.P * f)
                ids, values = top_by_row(rows, uids, total / n[rows], L * H, TOP)
            pole_list(cols, f"{c}_driven", ids, values, "share", f)
        ids, lift, co = torch.full((L * H, TOP), -1), torch.full((L * H, TOP), float("nan")), torch.full((L * H, TOP), float("nan"))
        if self.sparse["with"]:
            r = torch.cat([torch.full_like(i, x) for x, i in self.sparse["with"]])
            ids, lift, co = fires_with(r, torch.cat([i for _, i in self.sparse["with"]]), n, self.rate.cpu(), self.rate_floor, L * self.P * f, L * H)
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
                out[M][c] = {"err": self.err[c][M].item(), "emb": float(d["emb"][rows].sum() / n), "neurons": float(d["neurons"][rows].sum() / n),
                             "heads": float(d["heads"][rows].sum() / n), "const": float(d["const"][rows].sum() / n)}
        return out
