"""Can a lens predict what pushing two or three neurons together makes Qwen3-8B claim to be?

The Neuron Persona Library (jfonsecarivera on X, inspired by Sauers' steering; neuronpersonas.com)
pushed pairs and triples of 167
reproducible single-neuron personas together on "Who are you?" and kept the replies whose
persona is new, one that no parent makes alone (``combos.json``, ``combos3.json``, fetched
2026-10-29 from neuronpersonas.com/data/qwen/, kept beside the library copy in the ignored
``data/external/qwen_neuron_library/``; sha256 b0a3638a… and c926f7a8…). Each parent is pushed as
``h_L += ± a · k · r_L · ŵ`` at its own layer: ``ŵ`` the unit write vector, ``a`` its solo
strength, ``k`` a shared multiplier, ``r_L`` the median residual norm after layer ``L`` on the
prompt (the library's method).

A lens reads the combination as ``W_U norm(Σ_i J_{L_i} v_i)`` with ``v_i = ± a_i r_{L_i} ŵ_i``
(``k`` drops out: the read is scale-free), and each parent as ``W_U norm(J_{L_i} v_i)``. The new
persona's name is scored as in ``library_crosscheck.py``: the best vocabulary rank of the first
token of its words (with and without a leading space and capital), over kept tokens (seen 5+
times). Words of the name not in any parent's name are the novel words, the test of a concept
neither part has. Per lens and combination size:
- the share of names in the top 10 / 100 / 1000 of the combined read, and of the best parent's
  read (the best of 2 or 3 reads, so it has more chances);
- control for each: the same reads scored against another combination's name; the lift is
  matched minus control;
- the share where the combined read ranks the name better than every parent's read does.

    python scripts/checks/combo_crosscheck.py [--lenses logit,J,R:dclm]
"""
import argparse
import json

import torch
from library_crosscheck import LIBRARY, best_rank, name_tokens
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_lens.lenses import load_lens
from neuron_lens.paths import DATA, SUMMARY
from neuron_lens.readout import rms_norm_columns, transport
from neuron_lens.weights import down_proj, readout_names, tensors

MODEL = "Qwen/Qwen3-8B"
SOURCE = LIBRARY.parent
KS = (10, 100, 1000)


def residual_norms(tok, dev) -> torch.Tensor:
    """``r_L``: the median norm of the residual after each layer over "Who are you?" in the chat
    template (thinking off), the first token excluded."""
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    text = tok.apply_chat_template([{"role": "user", "content": "Who are you?"}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    norms = []
    hooks = [layer.register_forward_hook(lambda m, x, o: norms.append(
        (o[0] if isinstance(o, tuple) else o)[0, 1:].float().norm(dim=-1).median().item()))
        for layer in model.model.layers]
    with torch.no_grad():
        model.model(ids)
    for h in hooks:
        h.remove()
    del model
    return torch.tensor(norms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lenses", default="logit,J,R:dclm")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    r = residual_norms(tok, dev)
    keep = (torch.load(DATA / "token_counts.pt", weights_only=False)["counts"] >= 5).nonzero().squeeze(1)
    pos = torch.full((len(tok),), -1, dtype=torch.long)
    pos[keep] = torch.arange(len(keep))
    head = tensors(MODEL, readout_names())
    unembed = head["lm_head.weight"][keep].to(dev, torch.float16)
    gamma = head["model.norm.weight"].to(dev, torch.float32)

    def kept(ids: list[int]) -> list[int]:
        return [int(pos[i]) for i in ids if pos[i] >= 0]
    sets = {}
    for size, f in ((2, "combos.json"), (3, "combos3.json")):
        j = json.loads((SOURCE / f).read_text())
        pool = [dict(zip(j["pool_fields"], p)) for p in j["pool"]]
        rows = [dict(zip(j["fields"], x)) for x in j["rows"]]
        sets[size] = (pool, rows)
    pool = sets[2][0]
    assert [p["id"] for p in sets[3][0]] == [p["id"] for p in pool], "the two files' pools differ"
    # Each parent's push, v = s · a · r_L · ŵ, at its layer.
    pushes = []
    for p in pool:
        wd = tensors(MODEL, [down_proj(p["L"])])[down_proj(p["L"])][:, p["n"]].float()
        pushes.append(p["s"] * p["a"] * r[p["L"]].item() * wd / wd.norm())
    parent_words = [set(w.lower() for w in p["name"].split()) for p in pool]
    lines = [f"Neuron Persona Library combinations against lens reads of the summed pushes, {MODEL} "
             f"(r_L from 'Who are you?': {r[0]:.0f} at L0 to {r[-1]:.0f} at L35). Shares of names in the top "
             f"{' / '.join(map(str, KS))} of the read: combined (control: another combination's name) | best "
             "parent (control); and 'beats': combined ranks the name better than every parent (control)"]
    table = {}
    for spec in args.lenses.split(","):
        lens = load_lens(spec, MODEL)

        def moved(i: int) -> torch.Tensor:
            jac = None if lens is None else lens.get(pool[i]["L"])
            return transport(pushes[i][:, None].to(dev), jac)[:, 0]
        moved_all = torch.stack([moved(i) for i in range(len(pool))], 1)  # [d, parents]
        parent_logits = (unembed @ rms_norm_columns(moved_all, gamma).half()).float().cpu()  # [V, parents]
        for size, (_, rows) in sets.items():
            cols = torch.stack([moved_all[:, row["p"]].sum(1) for row in rows], 1)
            logits = torch.cat([(unembed @ rms_norm_columns(cols[:, s:s + 512], gamma).half()).float().cpu()
                                for s in range(0, cols.shape[1], 512)], 1)
            for words in ("all", "novel"):
                cands = []
                for row in rows:
                    parents = set().union(*(parent_words[i] for i in row["p"]))
                    name = row["persona"] if words == "all" else " ".join(
                        w for w in row["persona"].split() if w.lower() not in parents)
                    cands.append(kept(name_tokens(tok, name)))
                has = torch.tensor([bool(c) for c in cands])
                swapped = cands[1:] + cands[:1]
                rank = best_rank(logits, [c or [0] for c in cands])
                ctrl = best_rank(logits, [c or [0] for c in swapped])
                par, par_ctrl = (torch.stack([best_rank(parent_logits[:, row["p"]], [c or [0]] * len(row["p"])).min()
                                              for row, c in zip(rows, cs)]) for cs in (cands, swapped))
                share = {name: {k: (v[has] < k).float().mean().item() for k in KS}
                         for name, v in (("combined", rank), ("combined_control", ctrl), ("parent", par),
                                         ("parent_control", par_ctrl))}
                beats = ((rank < par)[has].float().mean().item(), (ctrl < par_ctrl)[has].float().mean().item())
                cells = "  ".join(f"@{k} {share['combined'][k]:.2f} ({share['combined_control'][k]:.2f}) | "
                                  f"{share['parent'][k]:.2f} ({share['parent_control'][k]:.2f})" for k in KS)
                lines.append(f"{spec:8s} {size} neurons, {words:5s} words, n={int(has.sum()):5d}: {cells}   "
                             f"beats {beats[0]:.2f} ({beats[1]:.2f})")
                table.setdefault(spec, {}).setdefault(str(size), {})[words] = {
                    "n": int(has.sum()), "share": share, "beats": beats[0], "beats_control": beats[1]}
                print(lines[-1], flush=True)
    text = "\n".join(lines)
    (SUMMARY / f"combo_crosscheck_{MODEL.split('/')[-1].lower()}.log").write_text(text + "\n")
    (SUMMARY / f"combo_crosscheck_{MODEL.split('/')[-1].lower()}.json").write_text(json.dumps(table))


if __name__ == "__main__":
    main()
