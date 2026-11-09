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

import torch

from neuron_lens import arch
from neuron_lens.rules import norm_eps


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
