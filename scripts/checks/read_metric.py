"""How a neuron's read covector becomes a vector for the R-lens, and what that does to ``read_R`` (findings §13).

A read row ``c`` (the gate or up row times the MLP input norm's weight, what the activation applies to the residual)
is a covector; the R-lens transports vectors. Rescaling a residual coordinate with every norm's weight compensating is
an approximate symmetry, and in the frame a norm reads, ``y = γ ⊙ x / rms``, the coordinates don't move with it, so a
read vector built there and mapped back by ``1/γ`` is free of the units: ``v = (1/γ) ⊙ f(Σ_y) w``, with ``w`` the raw row
and ``Σ_y = diag((γσ)²)`` (``neuron_lens.metric``). Arms, each ``v = s² ⊙ c`` with the catalog's pole construction done
in the metric (``neurons.poles``):

- ``euclidean``: ``s² = 1``, the read before the metric (in the norm's frame, ``γ² w``: a weight by the units alone);
- ``w/γ``: ``f = I``, the row itself, ``s² = 1/γ̃²`` (``γ̃``: ``|γ|`` floored at half its median, since ``w/γ`` blows up where
  ``γ ≈ 0``);
- ``w/γ reg``: the same, with the coordinates the norm switches off reading nothing: ``s² = (1/γ²) · t² / (t² + ε²)``,
  ``t = γσ / median(γσ)`` the post-norm scale, ``ε = 0.1``;
- ``σ⊙w``: ``f = Σ_y^½``, ``s² = σ/|γ|``, the catalog's (``metric.s2``);
- ``σ²γ⊙w``: ``f = Σ_y``, ``s² = σ²``, the diagonal Haufe pattern;
- ``1/‖J‖``: the lens's own metric, ``s = 1/‖J_i‖`` (a column norm moves with the units as ``γ`` does).

Scores per layer over its live poles (``act_top`` 0.05+): ``net``, the exact drivers' mean z under the read's R-lens logits
(``07_drivers.py``) as ``reads_at_firings.py`` scores ``read_R`` (the share past the mismatched pairs' 99th percentile, minus
theirs); ``firing``, the share of the read's layer-contrastive top 64 in ``before_R_z``'s (``17_firings.py``: what the R-lens
reads from the residual at the top firings); ``hub``, the share of top-10 tokens among the layer's 100 most listed. And per
band, the catalog's ``read_R_z`` against ``before_R_z`` (another pole of the layer and sign in brackets). And ``write_S``
(``write_arms``): the write against the S-lens's footprints, Euclidean (the release before the metric) or through the
lens's polar factor (``metric.polar``, the catalog's), scored on the peak tokens. σ is the scale
fit's (``02_fit.py --lens scale``). Writes ``data/summary/read_metric_<model>.{log,json}``.

    python scripts/checks/read_metric.py --layers 9,12,16,20,24,28,31
    NEURON_LENS_MODEL=google/gemma-4-12B python scripts/checks/read_metric.py --layers 6,12,18,24,30,36,42
"""
import argparse
import json

import pyarrow.parquet as pq
import torch

from neuron_lens import arch, metric
from neuron_lens.catalog import MODEL, TOP, column, firing_tokens, flat_tokens, kept, kept_index, lobe_signs, neuron_table
from neuron_lens.neurons import directions, read_rows, softcap
from neuron_lens.paths import SUMMARY
from neuron_lens.readout import layer_reads, rms_norm_columns, transport
from neuron_lens.weights import tensors

STEM = arch.stem(MODEL)
CFG = arch.text_config(MODEL)
L, F = CFG.num_hidden_layers, CFG.intermediate_size
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
CAP = softcap(MODEL)
CHUNK = 2048
SILENT = 0.05
ARMS = ("euclidean", "w/γ", "w/γ reg", "σ⊙w", "σ²γ⊙w", "1/‖J‖")
EPS = 0.1  # ``w/γ reg``: coordinates whose post-norm scale γσ is under about EPS of the median read nothing


def shared(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per row, the share of ``a``'s tokens in ``b`` (−1 is padding)."""
    out = torch.zeros(len(a))
    for s in range(0, len(a), CHUNK):
        x, y = a[s:s + CHUNK], b[s:s + CHUNK]
        out[s:s + CHUNK] = (((x[:, :, None] == y[:, None, :]).any(-1)) & (x >= 0)).float().mean(1)
    return out


def pair(g: torch.Generator) -> torch.Tensor:
    return torch.cat([torch.randperm(F, generator=g), F + torch.randperm(F, generator=g)])


def mean_z(vecs: torch.Tensor, table: torch.Tensor, tok: torch.Tensor, perm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's mean z (over the layer's rows) of the scores of its tokens ``tok`` (``−1`` padding), and the same
    with another row's vector (``perm``)."""
    n, dev = len(vecs), vecs.device
    s1, s2 = torch.zeros(len(table), device=dev), torch.zeros(len(table), device=dev)
    for s in range(0, n, CHUNK):
        sc = (vecs[s:s + CHUNK] @ table.T).float()
        s1 += sc.sum(0)
        s2 += sc.pow(2).sum(0)
    mean = s1 / n
    std = (s2 / n - mean.pow(2)).clamp_min(1e-12).sqrt()
    ok, t = tok >= 0, tok.clamp_min(0)
    out = []
    for v in (vecs, vecs[perm]):
        z = torch.empty(tok.shape)
        for s in range(0, n, CHUNK):
            tt = t[s:s + CHUNK].to(dev)
            z[s:s + CHUNK] = ((torch.einsum("ckd,cd->ck", table[tt], v[s:s + CHUNK]).float() - mean[tt]) / std[tt]).cpu()
        out.append(torch.where(ok.sum(1) >= 3, (z * ok).sum(1) / ok.sum(1).clamp_min(1), torch.nan))
    return out[0], out[1]


def net(m: torch.Tensor, n: torch.Tensor, live: torch.Tensor) -> float:
    thr = n[~n.isnan()].quantile(0.99).item()
    return (((m > thr) & live).float().mean() - ((n > thr) & live).float().mean()).item()


def hub(ids: torch.Tensor) -> float:
    v, c = torch.unique(ids.flatten(), return_counts=True)
    return torch.isin(ids, v[c.topk(100).indices]).float().mean().item()


def bands() -> tuple[list[str], dict]:
    g = torch.Generator().manual_seed(0)
    per = {}
    for layer in range(L):
        read = column(pq.read_table(neuron_table("reads", layer), columns=["read_R_z_ids"]), "read_R_z_ids").long()
        fr = pq.read_table(neuron_table("firings", layer), columns=["before_R_z_ids", "n_firings"])
        before, ok = column(fr, "before_R_z_ids").long(), torch.from_numpy(fr["n_firings"].to_numpy().copy()) > 0
        per[layer] = (shared(read, before)[ok].mean().item(), shared(read, before[pair(g)])[ok].mean().item())
    table, lines = {}, [f"the catalog's read_R_z top 64 in before_R_z's [another pole], {MODEL}, mean over layers"]
    for lo, hi in tuple(arch.bands(L)) + ((0, L - 1),):
        rows = [per[l] for l in range(lo, hi + 1)]
        table[f"L{lo}-{hi}"] = {"agree": sum(r[0] for r in rows) / len(rows), "control": sum(r[1] for r in rows) / len(rows)}
        lines.append(f"L{lo}-{hi}".ljust(9) + f"{table[f'L{lo}-{hi}']['agree']:.3f} [{table[f'L{lo}-{hi}']['control']:.3f}]")
    return lines, table


def arms(layers: list[int]) -> tuple[list[str], dict]:
    keep, kidx = kept(), kept_index()
    nm = arch.names(MODEL)
    head = tensors(MODEL, sorted({nm["unembed"], nm["norm"]}))
    unembed, gamma = head[nm["unembed"]][keep].to(DEV, torch.float16), head[nm["norm"]].to(DEV, torch.float32)
    J_mid = arch.lens(MODEL, "R_mid", arch.fits(MODEL)[0])
    g = torch.Generator().manual_seed(0)
    lines = [f"\nread_R by covector-to-vector map, {MODEL}: net (exact drivers) / firing (top 64 in before_R_z's) / hub"]
    table = {}
    for layer in layers:
        w = directions(MODEL, layer, DEV)
        ca, cb = w["gate"], w["up"]
        gam = metric.norm_weight(MODEL, "mid", layer).abs().clamp_min(metric.TINY).to(DEV)
        sigma = metric.scales(MODEL)["mid"][layer].to(DEV)
        J = J_mid[layer].to(DEV).float()
        lobe = lobe_signs(layer)
        acts = torch.from_numpy(pq.read_table(neuron_table("acts", layer), columns=["act_top"]).column(0).to_numpy().copy())
        drv = column(pq.read_table(neuron_table("drivers", layer), columns=["drivers_ids"]), "drivers_ids").long()
        k_drv = torch.where(drv >= 0, kidx[drv.clamp_min(0)], -1)
        fr = pq.read_table(neuron_table("firings", layer), columns=["before_R_z_ids", "n_firings"])
        before, fired = column(fr, "before_R_z_ids").long(), torch.from_numpy(fr["n_firings"].to_numpy().copy()) > 0
        live, perm = (acts >= SILENT) & fired, pair(g)
        t = gam * sigma / (gam * sigma).median()  # the post-norm scale, relative
        s2s = {"euclidean": torch.ones_like(gam), "w/γ": 1 / gam.clamp_min(0.5 * gam.median()).pow(2),
               "w/γ reg": t.pow(2) / (t.pow(2) + EPS ** 2) / gam.pow(2),
               "σ⊙w": sigma / gam, "σ²γ⊙w": sigma.pow(2), "1/‖J‖": 1 / J.norm(dim=0).pow(2)}
        table[layer] = {}
        for name in ARMS:
            s2 = s2s[name]
            a, b = (x / metric.cov_norm(x, s2) for x in (ca, cb))
            v = metric.vector(read_rows(a, b, lobe), s2)
            v = v / v.norm(dim=1, keepdim=True)  # the lens is scale-free, up to rms_norm_columns' eps
            rd = [layer_reads(v[i * F:(i + 1) * F].T, unembed, gamma, J, signs=(1,), k=TOP, kz=TOP, softcap=CAP) for i in (0, 1)]
            ids, zids = (keep[torch.cat([r[k][1] for r in rd])].long() for k in ("ids", "zids"))
            del rd
            t = rms_norm_columns(transport(v.T, J), gamma).T
            m, n = mean_z(t.half(), unembed, k_drv, perm)
            del t
            e = {"net": net(m, n, live), "firing": shared(zids, before)[live].mean().item(),
                 "firing_control": shared(zids, before[perm])[live].mean().item(), "hub": hub(ids[:, :10])}
            table[layer][name] = e
            lines.append(f"  L{layer} {name:9s} net {e['net']:+.3f}  firing {e['firing']:.3f} [{e['firing_control']:.3f}]  hub {e['hub']:.2f}")
            torch.mps.empty_cache() if DEV.type == "mps" else None
    return lines, table


WRITE_ARMS = ("euclidean", "polar")


def write_arms(layers: list[int]) -> tuple[list[str], dict]:
    """``write_S``, scored as ``reads_at_firings.py`` scores it: the peak tokens' mean z under the write's read against
    the S-lens's footprints at layer ``L``'s output, against another pole's: ``euclidean``, ``w · K ē``; ``polar``,
    ``w · (K Kᵀ)^-½ K ē``, the footprints' rotation (the catalog's)."""
    keep, kidx = kept(), kept_index()
    nm = arch.names(MODEL)
    emb = tensors(MODEL, [nm["embed"]])[nm["embed"]][keep].float() * arch.embed_scale(MODEL)
    emb = (emb - emb.mean(0)).to(DEV, torch.float16)
    K_out = arch.lens(MODEL, "S_out", arch.fits(MODEL)[1])
    flat, start = flat_tokens()
    g = torch.Generator().manual_seed(1)
    lines = [f"\nwrite_S, Euclidean against the footprints or through the S-lens's polar factor, {MODEL}: net (peak tokens)"]
    table = {}
    for layer in layers:
        wd = directions(MODEL, layer, DEV)["write"]
        K = K_out[layer].to(DEV).float()
        t = pq.read_table(neuron_table("acts", layer), columns=["act_top", "top_doc", "top_pos", "top_val"])
        doc, pos = column(t, "top_doc").long().clamp_min(0), column(t, "top_pos").long().clamp_min(0)
        cur = firing_tokens(flat, start, doc, pos).long()
        k_cur = torch.where((cur >= 0) & (column(t, "top_val") > 0), kidx[cur.clamp_min(0)], -1)
        live = torch.from_numpy(t["act_top"].to_numpy().copy()) >= SILENT
        perm = pair(g)
        cov = {"euclidean": wd.T, "polar": wd.T @ metric.polar(K)}
        table[layer] = {}
        for name in WRITE_ARMS:
            sw = cov[name] @ K
            sw = sw / sw.norm(dim=1, keepdim=True)
            m, n = mean_z(torch.cat([sw, -sw]).half(), emb, k_cur, perm)
            table[layer][name] = {"net": net(m, n, live)}
            lines.append(f"  L{layer} {name:9s} net {table[layer][name]['net']:+.3f}")
    return lines, table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="16,30")
    ap.add_argument("--write-layers", default="0,4,8", help="layers for the write_S arms")
    args = ap.parse_args()
    lines, by_band = bands()
    more, by_layer = arms([int(x) for x in args.layers.split(",")])
    wl, by_write = write_arms([int(x) for x in args.write_layers.split(",")])
    text = "\n".join(lines + more + wl)
    print(text)
    (SUMMARY / f"read_metric_{STEM}.log").write_text(text + "\n")
    (SUMMARY / f"read_metric_{STEM}.json").write_text(json.dumps({"bands": by_band, "layers": by_layer, "write_S": by_write}))


if __name__ == "__main__":
    main()
