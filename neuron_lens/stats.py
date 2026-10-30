"""Small statistics shared by the pipeline and the checks."""
import torch


def bh(p: torch.Tensor) -> torch.Tensor:
    """Benjamini-Hochberg q-values of the p-values ``p``."""
    order = p.argsort()
    ranked = p[order] * len(p) / torch.arange(1, len(p) + 1)
    q = torch.empty_like(p)
    q[order] = ranked.flip(0).cummin(0).values.flip(0).clamp_max(1)
    return q


def null_p(stat: torch.Tensor, null: torch.Tensor) -> torch.Tensor:
    """Each statistic's p-value: the share of the null (mismatched pairs) scoring as high, with
    the add-one correction; NaN statistics get 1."""
    null = null[~null.isnan()].sort().values
    ge = len(null) - torch.searchsorted(null, torch.nan_to_num(stat, nan=-float("inf")).contiguous())
    return (1 + ge) / (1 + len(null))


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = (x.argsort().argsort().float() for x in (a, b))
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


def auroc(score: torch.Tensor, pos: torch.Tensor) -> float:
    r = score.argsort().argsort().float() + 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return ((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item() if n1 and n0 else float("nan")


def pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per row, over columns."""
    x = x - x.mean(-1, keepdim=True)
    y = y - y.mean(-1, keepdim=True)
    return (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1)).clamp_min(1e-12)


def centered_ranks(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Ranks along ``dim``, centered: their correlations are Spearman's."""
    r = x.argsort(dim).argsort(dim).float()
    return r - r.mean(dim, keepdim=True)


def modal_share(tok: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    """Per row, the share of usable items (``ok``) that carry the row's most common token."""
    same = (tok[..., :, None] == tok[..., None, :]) & ok[..., :, None] & ok[..., None, :]
    return same.sum(-1).amax(-1) / ok.sum(-1).clamp_min(1)
