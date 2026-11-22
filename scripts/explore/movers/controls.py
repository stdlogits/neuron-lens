"""Controls: Δ reads of random live neurons at their layer's median rotation (motor band), and weight change against
activity change over all live neurons."""
import sys, torch, pyarrow.parquet as pq
from neuron_lens import arch
from neuron_lens.catalog import MODEL, kept, lobe_signs, tokenizer, neuron_table
from neuron_lens.neurons import poles
from neuron_lens.readout import layer_logit_stats, read_z
from neuron_lens.weights import tensors
POST, BASE = MODEL, MODEL + "-Base"; dev = "mps"
S = torch.load(sys.argv[1]); L, F = S["cos_w"].shape
KEEP, TOK = kept(), tokenizer(); words = lambda ids: [TOK.decode([int(KEEP[i])]) for i in ids]
mx = lambda t: torch.maximum(t[:, :F], t[:, F:])
live = (mx(S["top_base"]) >= 0.5) & (mx(S["top_post"]) >= 0.5)
a = 1 - S["cos_w"]
c = lambda x, y: (((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std())).item()
# activity vs weight change, live neurons, the more active pole
rb = torch.maximum(S["rate_base"][:, :F], S["rate_base"][:, F:]); rp = torch.maximum(S["rate_post"][:, :F], S["rate_post"][:, F:])
ok = live & (rb > 1e-5) & (rp > 1e-5)
dr = (rp / rb).log().abs(); dt = (mx(S["top_post"]) / mx(S["top_base"])).log().abs()
print("live neurons with rates:", ok.sum().item())
print("corr 1-cos(write) vs |log rate ratio|:", round(c(a[ok], dr[ok]), 3), "  vs |log act_top ratio|:", round(c(a[ok], dt[ok]), 3))
for b, (lo, hi) in arch.depth_bands(POST).items():
    m = ok.clone(); m[:lo] = False; m[hi + 1:] = False
    print(f"  {b}: corr rate {c(a[m], dr[m]):.3f}, act_top {c(a[m], dt[m]):.3f};  median |log rate ratio| {dr[m].median():.3f}, q99 {torch.quantile(dr[m], .99):.2f}")
q = torch.quantile(a[ok], torch.tensor([.5, .99]))
hi_ = ok & (a > torch.quantile(a[ok], .99)); lo_ = ok & (a < q[0])
print("median |log rate ratio|: top 1% movers", round(dr[hi_].median().item(), 3), " below-median movers", round(dr[lo_].median().item(), 3))
# random controls: 10 live motor-band neurons near their layer's median rotation
nm = arch.names(POST); head = tensors(POST, [nm["unembed"], nm["norm"]])
U = head[nm["unembed"]][KEEP].to(dev, torch.float16); G = head[nm["norm"]].to(dev, torch.float32)
J_out = arch.lens(POST, "R_out", arch.fits(POST)[0])
g = torch.Generator().manual_seed(0)
for layer in (23, 29, 32, 34, 35):
    lobe = lobe_signs(layer)
    wp, wb = poles(POST, layer, lobe, dev)["write"], poles(BASE, layer, lobe, dev)["write"]
    Jo = J_out[layer].to(dev) if layer < len(J_out) else None
    mean_w, std_w = layer_logit_stats(torch.cat([wp, -wp], 1), U, G, Jo)
    med = a[layer][live[layer]].median()
    cand = (live[layer] & ((a[layer] - med).abs() < 0.0003)).nonzero().flatten()
    for n in cand[torch.randperm(len(cand), generator=g)[:2]].tolist():
        d = (wp - wb)[:, n:n + 1]
        z = read_z(torch.cat([d, -d], 1), U, G, Jo, mean_w, std_w, k=8)
        w = pq.read_table(neuron_table("reads", layer), columns=["write_R_str"]).column(0)[n].as_py()
        print(f"L{layer}/{n} cos {S['cos_w'][layer, n]:.4f}  write_R+: {' | '.join(w.split(' | ')[:5])}\n   Δ: {' '.join(map(repr, words(z[0])))}  ||  {' '.join(map(repr, words(z[1])))}")
