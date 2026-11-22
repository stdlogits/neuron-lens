"""The most changed live neurons of a Qwen3 against its base: before (base catalog), after (post catalog), and the
change itself read through the post-trained lenses. Prints a report and saves the rows as JSON."""
import json, sys
import torch, pyarrow.parquet as pq
from neuron_lens import arch, metric
from neuron_lens.catalog import MODEL, kept, lobe_signs, neuron_table, tokenizer
from neuron_lens.neurons import directions, poles
from neuron_lens.readout import layer_logit_stats, read_z
from neuron_lens.weights import tensors

STATS, OUT = sys.argv[1], sys.argv[2]
POST, BASE = MODEL, MODEL + "-Base"
dev = "mps"
S = torch.load(STATS)
L, F = S["cos_w"].shape
KEEP, TOK = kept(), tokenizer()
words = lambda ids: [TOK.decode([int(KEEP[i])]) for i in ids]
BANDS = arch.depth_bands(POST)
band_of = lambda l: next(b for b, (a, z) in BANDS.items() if a <= l <= z)
mx = lambda t: torch.maximum(t[:, :F], t[:, F:])
live = (mx(S["top_base"]) >= 0.5) & (mx(S["top_post"]) >= 0.5)
aw = torch.where(live, 1 - S["cos_w"], torch.tensor(-1.0))
med = torch.stack([(1 - S["cos_w"][l])[live[l]].median() for l in range(L)])
ratio = aw / med[:, None]
lr = torch.where(live, (S["nw_post"] / S["nw_base"]).log(), torch.tensor(float("nan")))

pick = []  # (list name, layer, neuron)
seen = set()
def add(name, flat):
    for i in flat.tolist():
        l, n = divmod(i, F)
        if (l, n) not in seen:
            seen.add((l, n)); pick.append((name, l, n))
add("top rotation, whole model", aw.flatten().topk(10).indices)
for b, (a, z) in BANDS.items():
    r = torch.full_like(ratio, -1.0); r[a:z + 1] = ratio[a:z + 1]
    add(f"top rotation in the {b} band (vs its layer's median)", r.flatten().topk(5 if b != "motor" else 4).indices)
add("most amplified write", lr.nan_to_num(-9).flatten().topk(5).indices)
add("most shrunk write", (-lr).nan_to_num(-9).flatten().topk(3).indices)

nm = arch.names(POST)
head = tensors(POST, [nm["unembed"], nm["norm"]])
U = head[nm["unembed"]][KEEP].to(dev, torch.float16); G = head[nm["norm"]].to(dev, torch.float32)
R_fit = arch.fits(POST)[0]
J_out, J_mid = arch.lens(POST, "R_out", R_fit), arch.lens(POST, "R_mid", R_fit)
plan_in = {l: ("read_S" if l <= BANDS["sensory"][1] else "read_R") for l in range(L)}

def cat(model, table, layer, cols, rows):
    t = pq.read_table(neuron_table(table, layer, model), columns=cols).to_pylist()
    return [t[r] for r in rows]

rows = []
for layer in sorted({l for _, l, _ in pick}):
    ns = [n for _, l, n in pick if l == layer]
    lobe = lobe_signs(layer)
    pp, pb = poles(POST, layer, lobe, dev), poles(BASE, layer, lobe, dev)
    Jo = J_out[layer].to(dev) if layer < len(J_out) else None
    Jm = J_mid[layer].to(dev)
    wp, wb = pp["write"], pb["write"]
    so = metric.s2(POST, "out", layer, dev)
    mean_w, std_w = layer_logit_stats(torch.cat([wp, -wp], 1), U, G, Jo)
    D = wp - wb
    mean_d, std_d = layer_logit_stats(torch.cat([D, -D], 1), U, G, Jo)
    lam, V = torch.linalg.eigh((D @ D.T).cpu().double()); V = V.flip(1)[:, :4].float().to(dev)
    mean_r, std_r = layer_logit_stats(pp["vec"].T, U, G, Jm)
    for n in ns:
        # the pole to show: the more active one after post-training
        a_post = (S["top_post"][layer, n].item(), S["top_post"][layer, F + n].item())
        k = 0 if a_post[0] >= a_post[1] else 1
        sign, pole = (1, k) if k == 0 else (-1, k)
        r = pole * F + n
        d = sign * D[:, n:n + 1]
        par = ((d[:, 0] * wb[:, n] * sign / so).sum() / (wb[:, n].pow(2) / so).sum().sqrt() / (d[:, 0].pow(2) / so).sum().sqrt()).item()
        top4 = (V.T @ D[:, n]).pow(2).sum().item() / D[:, n].pow(2).sum().item()
        dz_w = read_z(torch.cat([d, -d], 1), U, G, Jo, mean_w, std_w, k=10)
        dz_d = read_z(torch.cat([d, -d], 1), U, G, Jo, mean_d, std_d, k=10)
        dv = (pp["vec"][r] - pb["vec"][r])[:, None]
        dz_r = read_z(torch.cat([dv, -dv], 1), U, G, Jm, mean_r, std_r, k=10)
        vp, vb = pp["vec"][r], pb["vec"][r]; ip = lambda x, y: (x * y / pp["s2"]).sum()
        rcos = (ip(vp, vb) / (ip(vp, vp) * ip(vb, vb)).sqrt()).item()  # vectors, so the inverse metric
        row = {"list": next(x for x, l, m in pick if (l, m) == (layer, n)), "layer": layer, "band": band_of(layer), "neuron": n,
               "pole": "+-"[pole], "cos_write": S["cos_w"][layer, n].item(), "layer_median_1mcos": med[layer].item(),
               "write_norm_ratio": (S["nw_post"][layer, n] / S["nw_base"][layer, n]).item(), "cos_read_row": rcos,
               "cos_gate": S["cos_g"][layer, n].item(), "cos_up": S["cos_u"][layer, n].item(),
               "delta_parallel_cos": par, "delta_in_layer_top4": top4,
               "delta_write_z_vs_writes": words(dz_w[0]), "delta_write_z_vs_writes_other_end": words(dz_w[1]),
               "delta_write_z_vs_deltas": words(dz_d[0]), "delta_read_R_z": words(dz_r[0]), "delta_read_R_z_other_end": words(dz_r[1])}
        for m, tag in ((BASE, "base"), (POST, "post")):
            a = cat(m, "acts", layer, ["act_top", "rate", "rate_chat", "peak_tokens", "next_tokens", "contexts"], [r])[0]
            rd = cat(m, "reads", layer, ["write_R_str", "write_R_z_str", plan_in[layer] + "_str", "read_R_str"], [r])[0]
            dr = cat(m, "drivers", layer, ["drivers_str"], [r])[0]
            row[tag] = a | rd | dr
        row["post"]["label"] = cat(POST, "labels", layer, ["fires", "fires_conf", "writes", "writes_conf"], [r])[0]
        row["input_read"] = plan_in[layer] + "_str"
        rows.append(row)
    print(f"L{layer}: {len(ns)}", flush=True)
json.dump(rows, open(OUT, "w"), ensure_ascii=False, indent=1)
