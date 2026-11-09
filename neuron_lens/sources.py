"""What drives a neuron pole when it fires: an exact split of its activation over its sources (findings §17).

At a firing of a neuron of layer ``M`` at position ``t``, its MLP input is the residual ``x`` there,
normalized, and ``x`` is a sum: the embedding, every head's write at layers ≤ ``M`` and every neuron's at
layers < ``M``, all at ``t``. Holding the norm's scale and SiLU's ratio ``s = silu(g)/g`` (RelP's rules),
the gate and up pre-activations ``g``, ``u`` split exactly over these sources, and so does the activation
by the half rule: source X contributes ``½ s (g u_X + u g_X)``, and the contributions sum to
``a = silu(g) u``. A share is a contribution over ``a``. Qwen3 only: Gemma 4's norms rescale each write
per position.
"""
import torch
from torch.nn.functional import silu


def split(rg: torch.Tensor, ru: torch.Tensor, x: torch.Tensor, inv: torch.Tensor, emb: torch.Tensor,
          z: list[torch.Tensor], acts: list[torch.Tensor], downs: list[torch.Tensor], wos: list[torch.Tensor],
          n_heads: int) -> dict:
    """Per firing (rows): ``rg``, ``ru`` ``[N, d]``, the neuron's gate and up rows times the norm's weight;
    ``x`` ``[N, d]``, the residual entering the norm; ``inv`` ``[N, 1]``, the norm's scale; ``emb`` the
    embedding, ``z[l]`` ``[N, H·hd]`` the heads' pre-``o_proj`` outputs at layers ≤ ``M``, ``acts[l]`` ``[N, f]``
    the activations at layers < ``M``. Returns ``g``, ``a`` ``[N]`` and the shares ``emb`` ``[N]``,
    ``heads`` ``[N, (M + 1)·H]``, ``neurons`` ``[N, M·f]``, which sum to 1 over the three."""
    g, u = (rg * x).sum(-1) * inv[:, 0], (ru * x).sum(-1) * inv[:, 0]
    s = silu(g) / g
    a = s * g * u
    share = lambda gx, ux: 0.5 * s[:, None] * (g[:, None] * ux + u[:, None] * gx) / a[:, None]
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
