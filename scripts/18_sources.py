"""What drives each neuron pole when it fires, and what fires with it (pipeline stage 18; findings §17).

``neurons/sources/L<layer>.parquet``, a row per pole, from one fp32 forward pass over the documents holding
its first ``N_CONTEXTS`` top firings (the firings the cards show, so these can sit on them):

- **driven by** (``driven_*``): the exact split of each firing's activation over the embedding, the heads
  and earlier neurons (``neuron_lens.sources``), each upstream neuron in the pole its activation's sign gives.
  ``driven_layer``, ``driven_neuron``, ``driven_pole``, ``driven_share``: the ``TOP`` largest |mean share over
  the firings|; ``driven_head_layer``, ``driven_head``, ``driven_head_share``: the same of the heads;
  ``emb_share``, ``heads_share``, ``neurons_share``: each kind of source's mean share (they sum to 1);
- **fires with** (``with_*``): the earlier poles active at ``act_top / 4`` or more at the pole's firings,
  by ``with_lift``, the share of the firings where they are active over their overall ``rate``; ``with_co``, that share.

``n_firings``: the firings used. Each list is built from ``RANK`` candidates a firing (``RANK_WITH`` for
fires-with), a source outside them counting 0 at that firing; ``with_*`` needs a source active at two of the pole's firings (one, if the pole has only one),
and its ``rate`` is at least one position in the corpus. The pass checkpoints every ``--save-minutes`` and drops the checkpoint with its tables.
Per layer in ``data/summary/sources_<model>.json``: each kind's share and the decomposition's largest
error. ``checks/relations.py`` compares the three relations (driven by, wired to, fires with).

    python scripts/18_sources.py [--detach] [--docs N] [--layers a-b]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM

from neuron_lens import arch
from neuron_lens.catalog import MODEL, N_CONTEXTS, TOP, column, documents, fixed, neuron_keys, neuron_table, write
from neuron_lens.firings import firing_plan
from neuron_lens.paths import DATA, LOGS, ROOT, SUMMARY
from neuron_lens.sources import Stream, merged, top_by_row
from neuron_lens.transport import decoder_stack

STEM = arch.stem(MODEL)
STATE = DATA / "acts" / f"{STEM}-sources.pt"
RANK = 256  # candidates kept per firing for driven-by
RANK_WITH = 1024  # ... and for fires-with, where a source shared by several firings needs to survive each
FIRE = 1 / 4  # of act_top: the acts tables' rate threshold


def acts_column(layer: int, name: str) -> torch.Tensor:
    return torch.from_numpy(pq.read_table(neuron_table("acts", layer), columns=[name]).column(0).to_numpy().copy()).float()


class Pass:
    def __init__(self, stack, layers: list[int], dev, positions: int):
        self.stack, self.layers, self.dev, self.flow = stack, layers, dev, Stream(stack)
        self.rate_floor = 1 / positions  # a pole seen firing at all fires at least once
        self.L, self.f, self.H = len(stack.layers), stack.config.intermediate_size, stack.config.num_attention_heads
        self.act_top = torch.stack([acts_column(l, "act_top") for l in range(self.L)]).to(dev).flatten()
        self.rate = torch.stack([acts_column(l, "rate") for l in range(self.L)]).to(dev).flatten()
        self.plan = {}
        for l in layers:
            t = pq.read_table(neuron_table("acts", l), columns=["top_doc", "top_pos", "top_val"])
            doc, pos, row = firing_plan(column(t, "top_doc", N_CONTEXTS).long(), column(t, "top_pos", N_CONTEXTS).long(), column(t, "top_val", N_CONTEXTS))
            self.plan[l] = (doc.to(dev), pos.to(dev), row.to(dev))
        self.cap = {}
        self.sparse = {l: {"driven": [], "with": []} for l in layers}  # (row, ids, vals) per batch, on the CPU
        self.dense = {l: {"heads": torch.zeros(2 * self.f, (l + 1) * self.H), "emb": torch.zeros(2 * self.f), "neurons": torch.zeros(2 * self.f),
                          "n": torch.zeros(2 * self.f), "err": []} for l in layers}
        self.a = self.b = 0
        self.at = None

    def hooks(self) -> list:
        return self.flow.hooks(self.cap, lambda h: h[self.at])

    def batch(self, ids: torch.Tensor, mask: torch.Tensor, a: int) -> None:
        """One forward pass, keeping only the positions some firing sits at, then the split per layer."""
        b = a + len(ids)
        slices = {}
        for l in self.layers:
            doc, pos, row = self.plan[l]
            lo, hi = torch.searchsorted(doc, a), torch.searchsorted(doc, b)
            slices[l] = (doc[lo:hi] - a, pos[lo:hi], row[lo:hi])
        T = ids.shape[1]
        keys = torch.cat([s[0].long() * T + s[1].long() for s in slices.values()]) if slices else torch.zeros(0, dtype=torch.long)
        if not len(keys):
            return
        uniq, inv = keys.unique(return_inverse=True)
        self.at = (uniq // T, uniq % T)
        where, start = {}, 0
        for l, s in slices.items():
            where[l] = inv[start:start + len(s[0])]
            start += len(s[0])
        with torch.inference_mode():
            self.stack(input_ids=ids.to(self.dev), attention_mask=mask.to(self.dev))
            for M, (_, _, row) in slices.items():
                if len(row):
                    self.layer(M, row, where[M])
        self.cap.clear()

    @torch.inference_mode()
    def layer(self, M: int, row: torch.Tensor, u: torch.Tensor) -> None:
        cap, f = self.cap, self.f
        s = self.flow.sources(cap, M, row.long() % f, u)
        d = self.dense[M]
        r = row.long().cpu()
        d["heads"].index_add_(0, r, s["heads"].cpu())
        d["emb"].index_add_(0, r, s["emb"].cpu())
        d["neurons"].index_add_(0, r, s["neurons"].sum(1).cpu())
        d["n"].index_add_(0, r, torch.ones(len(r)))
        d["err"].append((s["emb"] + s["heads"].sum(1) + s["neurons"].sum(1) - 1).abs().amax().item())
        if M == 0:
            return
        acts = torch.cat([cap["a", l][u] for l in range(M)], 1)  # [N, M·f]
        pid = (torch.arange(M, device=self.dev)[:, None] * 2 * f + torch.arange(f, device=self.dev)).flatten() + f * (acts < 0)  # [N, M·f]
        top = s["neurons"].abs().topk(min(RANK, s["neurons"].shape[1]), dim=1).indices
        self.sparse[M]["driven"].append((r, pid.gather(1, top).int().cpu(), s["neurons"].gather(1, top).cpu()))
        fires = (acts.abs() >= FIRE * self.act_top[pid]) & (self.act_top[pid] > 0)  # a silent pole doesn't fire at zero
        lift = fires / self.rate[pid].clamp_min(self.rate_floor)
        top = lift.topk(min(RANK_WITH, lift.shape[1]), dim=1).indices
        chosen = fires.gather(1, top)
        self.sparse[M]["with"].append((r, pid.gather(1, top).int().cpu(), chosen.float().cpu()))

    def state(self) -> dict:
        return {"dense": self.dense, "sparse": self.sparse}

    def load(self, state: dict) -> None:
        self.dense, self.sparse = state["dense"], state["sparse"]

    def table(self, M: int) -> dict:
        f, L, H = self.f, self.L, self.H
        d = self.dense[M]
        n = d["n"].clamp_min(1)
        cols = neuron_keys(M, f) | {"n_firings": pa.array(d["n"].numpy(), pa.int16())}
        for k in ("emb", "neurons"):
            cols[f"{k}_share"] = pa.array((d[k] / n).numpy())
        mean_heads = d["heads"] / n[:, None]
        cols["heads_share"] = pa.array(mean_heads.sum(1).numpy())
        v, i = mean_heads.abs().topk(min(TOP, mean_heads.shape[1]), dim=1)
        pad = TOP - v.shape[1]
        i_full = torch.nn.functional.pad(i, (0, pad), value=-1)
        s_full = torch.nn.functional.pad(mean_heads.gather(1, i), (0, pad), value=float("nan"))
        silent = d["n"] == 0  # no firing: no sources
        i_full[silent], s_full[silent] = -1, float("nan")
        cols["driven_head_layer"] = fixed(torch.where(i_full >= 0, i_full // H, -1).to(torch.int16))
        cols["driven_head"] = fixed(torch.where(i_full >= 0, i_full % H, -1).to(torch.int16))
        cols["driven_head_share"] = fixed(s_full)
        driven = self.sparse[M]["driven"]
        if driven:
            r = torch.cat([x[0][:, None].expand_as(x[1]) for x in driven]).flatten()
            ids, vals = torch.cat([x[1] for x in driven]).flatten(), torch.cat([x[2] for x in driven]).flatten()
            rows, uids, total = merged(r, ids, vals, M * 2 * f)
            mean = total / n[rows]
            di, dv = top_by_row(rows, uids, mean, 2 * f, TOP)
            wit = self.sparse[M]["with"]
            r = torch.cat([x[0][:, None].expand_as(x[1]) for x in wit]).flatten()
            ids, vals = torch.cat([x[1] for x in wit]).flatten(), torch.cat([x[2] for x in wit]).flatten()
            rows, uids, total = merged(r, ids, vals, M * 2 * f)
            enough = total >= torch.minimum(n[rows], torch.tensor(2.0))  # active at two of the pole's firings, or its one
            rows, uids, total = rows[enough], uids[enough], total[enough]
            co = total / n[rows]
            lift = co / self.rate.cpu()[uids].clamp_min(self.rate_floor)
            wi, wl = top_by_row(rows, uids, lift, 2 * f, TOP)
            _, wc = top_by_row(rows, uids, co, 2 * f, TOP, score=lift)
        else:
            di, dv, wi, wl, wc = (torch.full((2 * f, TOP), -1), torch.full((2 * f, TOP), float("nan")), torch.full((2 * f, TOP), -1),
                                  torch.full((2 * f, TOP), float("nan")), torch.full((2 * f, TOP), float("nan")))
        for name, ids_, value, value_name in (("driven", di, dv, "share"), ("with", wi, wl, "lift")):
            ok = ids_ >= 0
            layer, row = ids_.div(2 * f, rounding_mode="floor"), ids_ % (2 * f)
            cols[f"{name}_layer"] = fixed(torch.where(ok, layer, -1).to(torch.int16))
            cols[f"{name}_neuron"] = fixed(torch.where(ok, row % f, -1).to(torch.int16))
            cols[f"{name}_pole"] = fixed(torch.where(ok, 1 - 2 * (row // f), 0).to(torch.int8))
            cols[f"{name}_{value_name}"] = fixed(value.float())
        cols["with_co"] = fixed(wc.float())
        return cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=None, help="a-b (default all from L1)")
    ap.add_argument("--docs", type=int, default=None, help="a smoke test over the first N documents, writing no table")
    ap.add_argument("--batch-docs", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    ap.add_argument("--save-minutes", type=float, default=30)
    ap.add_argument("--detach", action="store_true")
    args = ap.parse_args()
    if args.detach:
        LOGS.mkdir(exist_ok=True)
        log = LOGS / f"sources_{STEM}.log"
        cmd = [sys.executable, __file__, *[a for a in sys.argv[1:] if a != "--detach"]]
        if sys.platform == "darwin":
            cmd = ["caffeinate", "-i", *cmd]
        with open(log, "a") as out:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        print(f"detached: pid {p.pid}, log {log}")
        return
    n = arch.text_config(MODEL).num_hidden_layers
    lo, hi = (1, n - 1) if args.layers is None else map(int, args.layers.split("-"))
    layers = list(range(lo, hi + 1))
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(args.device).eval().requires_grad_(False)
    stack = decoder_stack(lm)
    docs = documents()
    if args.docs is not None:
        docs = docs[:args.docs]
    p = Pass(stack, layers, args.device, sum(len(d) for d in docs))
    hooks = p.hooks()
    meta, next_doc = {"model": MODEL, "layers": layers, "contexts": N_CONTEXTS, "top": TOP, "docs": len(docs)}, 0
    if STATE.exists() and args.docs is None:
        saved = torch.load(STATE, weights_only=False, map_location="cpu")
        if saved["meta"] != meta:
            raise SystemExit(f"{STATE} was collected with {saved['meta']}, not {meta}: move it to start afresh")
        p.load(saved["state"])
        next_doc = saved["next_doc"]
    saved_at, t0 = time.monotonic(), time.perf_counter()
    for i, a in enumerate(range(next_doc, len(docs), args.batch_docs)):
        b = min(a + args.batch_docs, len(docs))
        T = max(len(d) for d in docs[a:b])
        ids, mask = torch.zeros(b - a, T, dtype=torch.long), torch.zeros(b - a, T, dtype=torch.long)
        for j, d in enumerate(docs[a:b]):
            ids[j, :len(d)], mask[j, :len(d)] = d.long(), 1
        p.batch(ids, mask, a)
        if args.docs is None and (b == len(docs) or time.monotonic() - saved_at > 60 * args.save_minutes):
            STATE.parent.mkdir(parents=True, exist_ok=True)
            tmp = STATE.with_suffix(".tmp")
            torch.save({"meta": meta, "state": p.state(), "next_doc": b}, tmp)
            os.replace(tmp, STATE)
            saved_at = time.monotonic()
        if i % 20 == 0 or b == len(docs):
            print(f"{time.strftime('%H:%M:%S')} {b}/{len(docs)} documents, {time.perf_counter() - t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()
    summary = {}
    for M in layers:
        d = p.dense[M]
        err = max(d["err"]) if d["err"] else float("nan")
        summary[M] = {"firings": int(d["n"].sum()), "err": err, "emb": float((d["emb"].sum() / d["n"].sum().clamp_min(1))), "neurons": float(d["neurons"].sum() / d["n"].sum().clamp_min(1)), "heads": float(d["heads"].sum() / d["n"].sum().clamp_min(1))}
        print(f"L{M}: {summary[M]}", flush=True)
        if args.docs is None:
            write(p.table(M), neuron_table("sources", M))
    if args.docs is None:
        (SUMMARY / f"sources_{STEM}.json").write_text(json.dumps(summary))
        STATE.unlink(missing_ok=True)
    else:
        print("smoke test done: no table written", flush=True)
        p.table(layers[-1])  # the merge, on the partial data


if __name__ == "__main__":
    main()
