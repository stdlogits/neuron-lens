"""Is a neuron's change (its part perpendicular to the base write, Δ⊥) related to what the neuron already wrote?
Both read through the post-trained R-lens, z-scored against the layer's writes. Two scores per live neuron, each
against a null pairing Δ⊥ with another neuron's write in the same layer:
- logit corr: correlation over the kept vocabulary of z(Δ⊥) and z(w_base);
- neighbours: cosine between the centroid of Δ⊥'s top-16 tokens (either end) and of w_base's top-16, in the
  (centred, unit) unembedding: does the change point at tokens near the neuron's own?"""
import sys, torch
from neuron_lens import arch, metric
from neuron_lens.catalog import MODEL, kept, lobe_signs
from neuron_lens.neurons import poles
from neuron_lens.readout import layer_logit_stats, rms_norm_columns, transport
from neuron_lens.weights import tensors
POST, BASE = MODEL, MODEL + "-Base"; dev = "mps"; K = 16; CH = 1024
S = torch.load(sys.argv[1]); L, F = S["cos_w"].shape
mx = lambda t: torch.maximum(t[:, :F], t[:, F:])
live = (mx(S["top_base"]) >= 0.5) & (mx(S["top_post"]) >= 0.5)
KEEP = kept(); nm = arch.names(POST); head = tensors(POST, [nm["unembed"], nm["norm"]])
U = head[nm["unembed"]][KEEP].to(dev, torch.float16); G = head[nm["norm"]].to(dev, torch.float32)
Uc = U.float() - U.float().mean(0); Uc = Uc / Uc.norm(dim=1, keepdim=True)
J_out = arch.lens(POST, "R_out", arch.fits(POST)[0])
g = torch.Generator().manual_seed(0)
res = {k: torch.full((L, F), float("nan")) for k in ("corr", "corr_null", "nb_pos", "nb_neg", "nb_null_pos", "nb_null_neg", "nb_self_w")}
def zlog(v, Jo, mean, std):
    t = rms_norm_columns(transport(v, Jo), G)
    return ((U @ t.half()).float() - mean[:, None]) / std[:, None]
for l in range(L):
    lobe = lobe_signs(l)
    wp, wb = poles(POST, l, lobe, dev)["write"], poles(BASE, l, lobe, dev)["write"]
    so = metric.s2(POST, "out", l, dev)
    D = wp - wb
    D = D - ((D * wb / so[:, None]).sum(0) / (wb * wb / so[:, None]).sum(0)) * wb  # Δ⊥ in the metric
    Jo = J_out[l].to(dev) if l < len(J_out) else None
    mean, std = layer_logit_stats(torch.cat([wb, -wb], 1), U, G, Jo)
    idx = live[l].nonzero().flatten()
    perm = idx[torch.randperm(len(idx), generator=g)]
    cw, cdp, cdn = (torch.empty(len(idx), Uc.shape[1], device=dev) for _ in range(3))
    for s in range(0, len(idx), CH):
        i = idx[s:s + CH].to(dev); j = perm[s:s + CH].to(dev)
        zw, zd, zn = zlog(wb[:, i], Jo, mean, std), zlog(D[:, i], Jo, mean, std), zlog(wb[:, j], Jo, mean, std)
        c = lambda a, b: ((a - a.mean(0)) * (b - b.mean(0))).mean(0) / (a.std(0) * b.std(0))
        res["corr"][l, idx[s:s + CH]] = c(zw, zd).cpu(); res["corr_null"][l, idx[s:s + CH]] = c(zn, zd).cpu()
        cw[s:s + CH] = Uc[zw.topk(K, dim=0).indices].mean(0)
        cdp[s:s + CH] = Uc[zd.topk(K, dim=0).indices].mean(0); cdn[s:s + CH] = Uc[(-zd).topk(K, dim=0).indices].mean(0)
    unit = lambda x: x / x.norm(dim=1, keepdim=True)
    cw, cdp, cdn = unit(cw), unit(cdp), unit(cdn)
    pi = torch.argsort(torch.argsort(perm))  # position of each neuron in idx order for the permuted partner
    order = {n: k for k, n in enumerate(idx.tolist())}
    pj = torch.tensor([order[n] for n in perm.tolist()], device=dev)
    for key, cd in (("pos", cdp), ("neg", cdn)):
        res["nb_" + key][l, idx] = (cw * cd).sum(1).cpu()
        res["nb_null_" + key][l, idx] = (cw[pj] * cd).sum(1).cpu()
    res["nb_self_w"][l, idx] = (cw * cw[pj]).sum(1).cpu()  # two random writes' token neighbourhoods, for scale
    a = res
    print(f"L{l:2d} corr {a['corr'][l, idx].median():+.3f} (null {a['corr_null'][l, idx].median():+.3f})  "
          f"nb +end {a['nb_pos'][l, idx].median():.3f} −end {a['nb_neg'][l, idx].median():.3f} (null {a['nb_null_pos'][l, idx].median():.3f}/{a['nb_null_neg'][l, idx].median():.3f}; w vs w {a['nb_self_w'][l, idx].median():.3f})", flush=True)
torch.save(res, sys.argv[2])
