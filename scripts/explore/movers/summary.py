"""Band summary of related.py's scores: how often a live neuron's Δ⊥ lands nearer its own write's token
neighbourhood than a permuted same-layer partner's (the nearer end of Δ⊥ either way), for all live neurons, the
top 1% by write rotation and the below-median half; then the last four layers."""
import sys, torch
from neuron_lens import arch
from neuron_lens.catalog import MODEL
S, R = torch.load(sys.argv[1]), torch.load(sys.argv[2])
a = 1 - S["cos_w"]
own_all = torch.maximum(R["nb_pos"], R["nb_neg"]); nul_all = torch.maximum(R["nb_null_pos"], R["nb_null_neg"])
def row(name, own, nul, aa):
    t, lo = aa > torch.quantile(aa, .99), aa < aa.median()
    w = lambda m: (own[m] > nul[m]).float().mean().item()
    print(f"{name:12s} win {w(slice(None)):.3f}  top-1% movers {w(t):.3f}  below-median {w(lo):.3f}  "
          f"median own−null {(own - nul).median():+.4f} (top-1% {(own[t] - nul[t]).median():+.4f})")
for b, (lo, hi) in arch.depth_bands(MODEL).items():
    m = ~R["nb_pos"][lo:hi + 1].isnan()
    row(b, own_all[lo:hi + 1][m], nul_all[lo:hi + 1][m], a[lo:hi + 1][m])
for l in range(len(a) - 4, len(a)):
    m = ~R["nb_pos"][l].isnan()
    row(f"L{l}", own_all[l][m], nul_all[l][m], a[l][m])
m = ~R["nb_pos"][-1].isnan()
print(f"L{len(a) - 1} medians: +end {R['nb_pos'][-1][m].median():.3f}  −end {R['nb_neg'][-1][m].median():.3f}  "
      f"null {R['nb_null_pos'][-1][m].median():.3f}/{R['nb_null_neg'][-1][m].median():.3f}  "
      f"logit corr {R['corr'][-1][m].median():+.3f} (null {R['corr_null'][-1][m].median():+.3f})")
