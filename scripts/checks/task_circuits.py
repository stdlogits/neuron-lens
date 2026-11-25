"""What Qwen3-8B computes on prompts of different kinds: relevance graphs over its neurons and heads (exploratory).

A prompt (``task_circuits_prompts.json``: ten kinds of question, each with a one-token answer and a distractor) is
traced from one number, ``T``: the answer's logit minus the distractor's at the last position. Under RelP's rules
(``rules.relp_rules``: norms hold their scale, SiLU its ratio, the gate passes half of each factor's change) and with
every attention pattern held at its value, the model is a linear map of each layer's input at the prompt, so a
backward pass from ``T`` gives each unit an exact share of it:

- a neuron at a position: its activation times ``∂T/∂a`` (the pole is the activation's sign);
- a head at a position: its output (before ``o_proj``) dotted with ``∂T/∂z``;
- the embedding at a position, whose shares sum to ``T`` (``trace`` checks it).

A unit's relevance counts every path from it to ``T``, through later units included, so the relevances of a layer's
units are the part of ``T`` that passes through that layer, and units of different layers overlap. Its *direct*
relevance counts the path straight to the readout (units at the last position only); those sum to ``T`` too.

Relevance moves between units along *flows*: the part of a unit's relevance that one source unit puts in directly (the
source's write, read by the target's read at the target's input, times ``∂T`` at the target). Into a neuron, the
sources are the units at its position; into a head, the units at every position it attends to, weighted by its
pattern. A unit's inflows sum to its relevance (``graph`` checks it), and its outflows plus its direct relevance sum
to it too.

Attention patterns are held, so relevance never flows through queries and keys: a unit whose job is to make a head
look somewhere gets none. ``graph`` lists, for the heads it keeps, where they look and which units set their query.

Parts: ``check`` (each prompt's top tokens and ``T``), ``trace`` (every prompt's relevances), ``compare`` (the kinds
of prompt against each other), ``graph <task> <i>`` (one prompt's graph with flows), ``ablate`` (the graphs' nodes
removed from the real model, against random nodes). Outputs go to ``data/exploration/task_circuits/``.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3 as mq

from neuron_lens.catalog import MODEL, neuron_table, head_table
from neuron_lens.rules import relp_rules
from neuron_lens.transport import decoder_stack

HERE = Path(__file__).parent
PROMPTS = HERE / "task_circuits_prompts.json"
OUT = Path("data/exploration/task_circuits")
FREEZE = True  # hold the attention patterns (``_frozen``)


# --------------------------------------------------------------------------------------------- prompts

def items() -> list[dict]:
    """Every prompt: its question, instruction, the assistant's prefill, and the answer's and distractor's tokens' text.

    A row is ``[question, answer, distractor, instruction?, prefill?]``. The instruction defaults to the file's
    suffix (``""``: none). The reply starts with the file's prefill (``Answer:``) and the answer after a space; a
    digit answer follows ``Answer: `` and the row's prefill (the digits before it); a row whose answer starts with a
    space continues its own prefill (the IOI sentences)."""
    spec = json.loads(PROMPTS.read_text())
    out = []
    for task, rows in spec["tasks"].items():
        for i, row in enumerate(rows):
            q, ans, dis = row[:3]
            instr = spec["suffix"] if len(row) < 4 else (" " + row[3] if row[3] else "")
            extra = row[4] if len(row) > 4 else ""
            if ans.startswith(" "):
                prefill, a, d = extra, ans, dis
            elif ans.isdigit():
                prefill, a, d = spec["prefill"] + " " + extra, ans, dis
            else:
                prefill, a, d = spec["prefill"], " " + ans, " " + dis
            out.append({"task": task, "i": i, "q": q, "instr": instr, "prefill": prefill, "answer": a, "distractor": d})
    return out


def encode(tok, it: dict) -> tuple[torch.Tensor, list[str], list[str]]:
    """The prompt's token ids, each token's text and its role: ``pre`` (the chat template before the question), ``q``,
    ``instr``, ``post`` (the template after it, the empty think block), ``prefill``; the last token is ``last``."""
    head = tok.apply_chat_template([{"role": "user", "content": it["q"] + it["instr"]}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    text = head + it["prefill"]
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    q0 = text.index(it["q"])
    spans = [("pre", 0), ("q", q0), ("instr", q0 + len(it["q"])), ("post", q0 + len(it["q"]) + len(it["instr"])),
             ("prefill", len(head))]
    roles = []
    for s, _ in enc["offset_mapping"]:
        roles.append([r for r, start in spans if s >= start][-1])
    roles[-1] = "last"
    ids = torch.tensor(enc["input_ids"])
    return ids, [tok.decode([i]) for i in ids.tolist()], roles


def first_token(tok, s: str) -> int:
    return tok(s, add_special_tokens=False)["input_ids"][0]


# --------------------------------------------------------------------------------------------- the model

def _frozen(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Eager attention that keeps its pattern and per-head output (``module._cap``) and, under ``FREEZE``, holds the
    pattern: the output stays the model's, the derivative passes through the values only."""
    k = mq.repeat_kv(key, module.num_key_value_groups)
    v = mq.repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask[..., :k.shape[-2]]
    A = torch.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    cap, l = module._cap, module._l
    if cap is not None:
        cap["A"][l] = A.detach()[0]
    o = torch.matmul(A.detach() if FREEZE else A, v).transpose(1, 2).contiguous()  # [B, T, H, hd]
    if cap is not None:
        if o.requires_grad:
            o.retain_grad()
        cap["z"][l] = o
    return o, A


class Tracer:
    """Qwen3-8B in fp32 with what a trace needs: each layer's weights and the catalog's labels and ``act_top``."""

    def __init__(self, dev: str, labels: bool = True):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager")
        self.lm = lm.to(dev).eval().requires_grad_(False)
        self.stack, self.dev = decoder_stack(lm), dev
        cfg = lm.config
        self.L, self.H, self.hd = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.head_dim
        self.kv, self.f, self.d = cfg.num_key_value_heads, cfg.intermediate_size, cfg.hidden_size
        self.eps = cfg.rms_norm_eps
        for l, lay in enumerate(self.stack.layers):
            lay.self_attn._cap, lay.self_attn._l = None, l
        mq.eager_attention_forward = _frozen
        if labels:
            self.labels = [pq.read_table(neuron_table("labels", l)).to_pydict() for l in range(self.L)]
            self.act_top = torch.stack([torch.from_numpy(pq.read_table(neuron_table("acts", l), columns=["act_top"])
                                                         .column(0).to_numpy().copy()).float() for l in range(self.L)])
            self.head_labels = pq.read_table(head_table("labels")).to_pydict()

    # ----------------------------------------------------------------------------- one prompt
    def prompt(self, it: dict) -> dict:
        ids, toks, roles = encode(self.tok, it)
        return {"ids": ids, "toks": toks, "roles": roles, "a": first_token(self.tok, it["answer"]),
                "d": first_token(self.tok, it["distractor"])}

    @torch.no_grad()
    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm(input_ids=ids[None].to(self.dev)).logits[0, -1].float()

    def run(self, ids: torch.Tensor, a: int, d: int) -> dict:
        """One forward pass under RelP with the patterns held, and the backward pass from ``T``: everything a trace
        reads (activations, per-head outputs, patterns, the residual at each norm's input) and their gradients."""
        L, cap = self.L, {k: [None] * self.L for k in ("A", "z", "a", "g", "u", "x1", "x2", "x1g", "x2g")}
        hooks = []

        def emb(mod, inp, out):
            out = out.detach().requires_grad_(True)
            cap["emb"] = out
            return out
        hooks.append(self.stack.embed_tokens.register_forward_hook(emb))
        for l, lay in enumerate(self.stack.layers):
            lay.self_attn._cap = cap
            def resid(key, l):
                def hook(m, x):
                    if not x[0].requires_grad:  # relp_rules' own check of the norm
                        return
                    x[0].retain_grad()
                    cap[key + "g"][l] = x[0]
                    cap[key][l] = x[0].detach()[0]
                return hook
            hooks.append(lay.input_layernorm.register_forward_pre_hook(resid("x1", l)))
            hooks.append(lay.post_attention_layernorm.register_forward_pre_hook(resid("x2", l)))

            def down(m, x, l=l):
                x[0].retain_grad()
                cap["a"][l] = x[0]
            hooks.append(lay.mlp.down_proj.register_forward_pre_hook(down))
            hooks.append(lay.mlp.gate_proj.register_forward_hook(lambda m, x, o, l=l: cap["g"].__setitem__(l, o.detach()[0])))
            hooks.append(lay.mlp.up_proj.register_forward_hook(lambda m, x, o, l=l: cap["u"].__setitem__(l, o.detach()[0])))
        def final(m, x):
            x[0].retain_grad()
            cap["xf"] = x[0]
        hooks.append(self.stack.norm.register_forward_pre_hook(final))
        try:
            with relp_rules(self.stack), torch.enable_grad():
                self.stack(input_ids=ids[None].to(self.dev))
                xf = cap["xf"][0]
                r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps).detach()
                dvec = self.lm.lm_head.weight[a] - self.lm.lm_head.weight[d]
                cap["cov_T"] = (dvec * self.stack.norm.weight * r[-1]).detach()  # the readout's covector at the last position
                T = (self.stack.norm.weight * xf[-1] * r[-1]) @ dvec
                T.backward()
        finally:
            for h in hooks:
                h.remove()
            for lay in self.stack.layers:
                lay.self_attn._cap = None
        cap["T"] = T.item()
        return cap

    # ----------------------------------------------------------------------------- relevances
    def relevances(self, cap: dict) -> dict:
        """Every unit's relevance at every position: ``emb`` ``[P]``, ``heads`` ``[L, P, H]``, ``neurons`` ``[L, P, f]``;
        the activations ``acts`` ``[L, P, f]``; direct relevances at the last position (``direct_*``)."""
        emb = (cap["emb"][0] * cap["emb"].grad[0]).sum(-1)
        heads = torch.stack([(z[0] * z.grad[0]).sum(-1) for z in cap["z"]])
        acts = torch.stack([a[0].detach() for a in cap["a"]])
        neurons = torch.stack([a[0].detach() * a.grad[0] for a in cap["a"]])
        c = cap["cov_T"]
        d_neurons = torch.stack([(c @ lay.mlp.down_proj.weight) * cap["a"][l][0, -1].detach()
                                 for l, lay in enumerate(self.stack.layers)])
        d_heads = torch.stack([((c @ lay.self_attn.o_proj.weight).view(self.H, self.hd) * cap["z"][l][0, -1].detach()).sum(-1)
                               for l, lay in enumerate(self.stack.layers)])
        d_emb = c @ cap["emb"][0, -1].detach()
        # the residual's relevance at each position, entering each layer's attention and MLP and after the last layer:
        # at every depth these sum to T over positions
        res = torch.stack([(x[0] * x.grad[0]).sum(-1) for x in cap["x1g"]] + [(cap["xf"][0] * cap["xf"].grad[0]).sum(-1)])
        mid = torch.stack([(x[0] * x.grad[0]).sum(-1) for x in cap["x2g"]])
        return {"res": res.detach(), "mid": mid.detach(), "emb": emb.detach(), "heads": heads.detach(), "neurons": neurons, "acts": acts, "direct_neurons": d_neurons,
                "direct_heads": d_heads.detach(), "direct_emb": d_emb.detach()}

    # ----------------------------------------------------------------------------- flows
    def sources(self, cap: dict, cov: torch.Tensor, pos: torch.Tensor, n_heads: int, n_neurons: int) -> dict:
        """What every unit at positions ``pos`` puts into a covector ``cov`` ``[K, d]`` read there (directly, through the
        residual): ``emb`` ``[K]``, ``heads`` ``[n_heads, K, H]`` (layers below ``n_heads``), ``neurons``
        ``[n_neurons, K, f]``."""
        out = {"emb": (cov * cap["emb"][0, pos].detach()).sum(-1)}
        out["heads"] = torch.stack([((cov @ self.stack.layers[l].self_attn.o_proj.weight).view(len(pos), self.H, self.hd)
                                     * cap["z"][l][0, pos].detach()).sum(-1) for l in range(n_heads)]) if n_heads else None
        out["neurons"] = torch.stack([(cov @ self.stack.layers[l].mlp.down_proj.weight) * cap["a"][l][0, pos].detach()
                                      for l in range(n_neurons)]) if n_neurons else None
        return out

    def neuron_cov(self, cap: dict, M: int, n: int, p: int) -> torch.Tensor:
        """A neuron's read at its MLP input, under the LN and half rules, times ``∂T/∂a``: ``[1, d]``."""
        lay = self.stack.layers[M]
        g, u = cap["g"][M][p, n], cap["u"][M][p, n]
        x = cap["x2"][M][p]
        r = torch.rsqrt(x.pow(2).mean() + self.eps)
        read = 0.5 * (torch.sigmoid(g) * u * lay.mlp.gate_proj.weight[n] + torch.nn.functional.silu(g) * lay.mlp.up_proj.weight[n])
        return (cap["a"][M].grad[0, p, n] * r * lay.post_attention_layernorm.weight * read)[None]

    def head_cov(self, cap: dict, M: int, h: int, p: int) -> tuple[torch.Tensor, torch.Tensor]:
        """A head's value read at every position it attends to, weighted by its pattern, times ``∂T/∂z``: the
        positions ``[K]`` and covectors ``[K, d]``."""
        lay = self.stack.layers[M]
        A = cap["A"][M][h, p, :p + 1]
        pos = torch.arange(p + 1, device=A.device)
        kvh = h // (self.H // self.kv)
        Wv = lay.self_attn.v_proj.weight.view(self.kv, self.hd, self.d)[kvh]
        back = cap["z"][M].grad[0, p, h] @ Wv  # [d]
        x = cap["x1"][M][pos]
        r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return pos, A[:, None] * r * lay.input_layernorm.weight * back


    def _qk(self, M: int, h: int, x: torch.Tensor, pos: int, which: str) -> torch.Tensor:
        """Head ``h``'s query or key (``which``) at ``pos`` from the residual ``x`` entering layer ``M`` there, with its
        norms' scales held at their values (the input norm's and the q/k norm's), rotated for ``pos``."""
        at, lay = self.stack.layers[M].self_attn, self.stack.layers[M]
        r = torch.rsqrt(x.detach().pow(2).mean() + self.eps)
        xn = lay.input_layernorm.weight * x * r
        if which == "q":
            W, norm, head = at.q_proj.weight.view(self.H, self.hd, self.d), at.q_norm, h
        else:
            W, norm, head = at.k_proj.weight.view(self.kv, self.hd, self.d), at.k_norm, h // (self.H // self.kv)
        raw = W[head] @ xn
        v = norm.weight * raw * torch.rsqrt(raw.detach().pow(2).mean() + norm.variance_epsilon)
        cos, sin = self.stack.rotary_emb(v[None, None], position_ids=torch.tensor([[pos]], device=v.device))
        return v * cos[0, 0] + mq.rotate_half(v[None])[0] * sin[0, 0]

    def qk_split(self, cap: dict, M: int, h: int, p: int, a: int, b: int) -> dict:
        """Why head ``h`` of layer ``M`` at ``p`` looks at ``a`` rather than ``b``: its score margin ``s(p, a) − s(p, b)``
        (before the softmax) split exactly over the units at ``p`` that build its query (keys held), and over the units
        at ``a`` and at ``b`` that build the two keys (query held)."""
        x1 = cap["x1"][M]
        xq = x1[p].clone().requires_grad_(True)
        xa, xb = x1[a].clone().requires_grad_(True), x1[b].clone().requires_grad_(True)
        at = self.stack.layers[M].self_attn
        with torch.enable_grad():
            q = self._qk(M, h, xq, p, "q")
            ka, kb = self._qk(M, h, xa, a, "k"), self._qk(M, h, xb, b, "k")
            margin = q.detach() @ (ka.detach() - kb.detach()) * at.scaling
            (q @ (ka.detach() - kb.detach()) * at.scaling).backward(retain_graph=True)
            (q.detach() @ ka * at.scaling).backward(retain_graph=True)
            (-(q.detach() @ kb) * at.scaling).backward()
        out = {"margin": margin.item(), "A_a": cap["A"][M][h, p, a].item(), "A_b": cap["A"][M][h, p, b].item()}
        for side, x, where in (("q", xq, p), ("ka", xa, a), ("kb", xb, b)):
            with torch.no_grad():
                out[side] = self.sources(cap, x.grad[None], torch.tensor([where], device=self.dev), M, M)
        return out


# --------------------------------------------------------------------------------------------- parts

def cased(s: str, how: str) -> str:
    """``s`` with its first letter as ``how`` says (``as``, ``lower``, ``upper``), its leading space kept."""
    lead = s[:len(s) - len(s.lstrip())]
    w = s.lstrip()
    if how == "lower":
        w = w[:1].lower() + w[1:]
    elif how == "upper":
        w = w[:1].upper() + w[1:]
    return lead + w


def check(args) -> None:
    """Each prompt's top tokens and ``T``. The answer takes the case the model prefers (its first token's logit), and the
    distractor the same case: ``answer_tok`` and ``distractor_tok`` in ``check.json`` are what the traces use."""
    tr = Tracer(args.device, labels=False)
    rows = []
    for it in items():
        pr = tr.prompt(it)
        lg = tr.logits(pr["ids"])
        how = max(("as", "lower", "upper"), key=lambda h: lg[first_token(tr.tok, cased(it["answer"], h))].item())
        a, d = first_token(tr.tok, cased(it["answer"], how)), first_token(tr.tok, cased(it["distractor"], how))
        p = lg.softmax(-1)
        top = p.topk(5)
        rows.append({**{k: it[k] for k in ("task", "i", "q", "answer", "distractor")}, "case": how, "answer_tok": a,
                     "distractor_tok": d, "T": (lg[a] - lg[d]).item(), "p_answer": p[a].item(), "p_distractor": p[d].item(),
                     "rank": int((lg > lg[a]).sum()),
                     "top": [(tr.tok.decode([i]), round(v, 3)) for i, v in zip(top.indices.tolist(), top.values.tolist())]})
        r = rows[-1]
        print(f"{r['task']:15s} {r['i']:2d} rank {r['rank']} p {r['p_answer']:.2f} T {r['T']:+6.2f} {tr.tok.decode([a])!r:14s} "
              f"vs {tr.tok.decode([d])!r:14s} top {r['top'][:3]}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


def checked() -> list[dict]:
    """``items()`` with the tokens and ``T`` that ``check`` chose."""
    got = {(r["task"], r["i"]): r for r in json.loads((OUT / "check.json").read_text())}
    return [{**it, **{k: got[it["task"], it["i"]][k] for k in ("answer_tok", "distractor_tok", "T", "p_answer", "rank", "top")}}
            for it in items()]


TOPK = 4000  # (unit, position) pairs kept per prompt, by |relevance|


def trace(args) -> None:
    """Every prompt's relevances: per layer and position, the heads' (all) and the neurons' sums (signed, absolute), each
    pole's relevance summed over positions (all poles), the ``TOPK`` largest (neuron, position) relevances, the direct
    relevances at the last position, and the checks (the embedding's relevances and the direct ones sum to ``T``)."""
    tr = Tracer(args.device, labels=False)
    (OUT / "traces").mkdir(parents=True, exist_ok=True)
    L, f = tr.L, tr.f
    for it in checked():
        path = OUT / "traces" / f"{it['task']}_{it['i']:02d}.pt"
        if path.exists() and not args.force:
            continue
        ids, toks, roles = encode(tr.tok, it)
        cap = tr.run(ids, it["answer_tok"], it["distractor_tok"])
        rel = tr.relevances(cap)
        T = cap["T"]
        R, acts = rel["neurons"], rel["acts"]  # [L, P, f]
        P = R.shape[1]
        pole = torch.zeros(L, 2, f, device=R.device)
        pole[:, 0] = (R * (acts > 0)).sum(1)
        pole[:, 1] = (R * (acts < 0)).sum(1)
        flat = R.flatten()
        top = flat.abs().topk(min(TOPK, flat.numel())).indices
        l, rest = top // (P * f), top % (P * f)
        p, n = rest // f, rest % f
        dn = rel["direct_neurons"].flatten()
        dtop = dn.abs().topk(2000).indices
        out = {
            "item": it, "toks": toks, "roles": roles, "T": T,
            "check_emb": rel["emb"].sum().item(), "check_direct": (rel["direct_neurons"].sum() + rel["direct_heads"].sum()
                                                                     + rel["direct_emb"]).item(),
            "res": rel["res"].cpu(), "mid": rel["mid"].cpu(), "emb": rel["emb"].cpu(), "heads": rel["heads"].cpu(), "direct_heads": rel["direct_heads"].cpu(),
            "direct_emb": rel["direct_emb"].item(),
            "neuron_sum": R.sum(-1).cpu(), "neuron_abs": R.abs().sum(-1).cpu(), "direct_neuron_sum": rel["direct_neurons"].sum(-1).cpu(),
            "pole": pole.cpu(),
            "top_l": l.cpu().short(), "top_p": p.cpu().short(), "top_n": n.cpu().short(), "top_R": flat[top].cpu(),
            "top_act": acts[l, p, n].cpu(),
            "dtop_l": (dtop // f).cpu().short(), "dtop_n": (dtop % f).cpu().short(), "dtop_R": dn[dtop].cpu(),
        }
        torch.save(out, path)
        spread = (rel["res"].sum(-1) - T).abs().max().item()
        print(f"{it['task']:15s} {it['i']:2d} P {P:3d} T {T:+7.3f} emb {out['check_emb']:+7.3f} direct {out['check_direct']:+7.3f} "
              f"residual cuts within {spread:.1e} "
              f"heads {rel['heads'].sum().item() / T:+.2f}·T neurons {R.sum().item() / T:+.2f}·T", flush=True)
        del cap, rel, R, acts
        if args.device == "mps":
            torch.mps.empty_cache()


BANDS = [("sensory", 0, 4), ("association", 4, 9), ("workspace", 9, 21), ("motor", 21, 36)]  # arch.depth_bands, Qwen3-8B
ROLES = ("pre", "q", "instr", "post", "prefill", "last")
PARTS = ("start", "content", "instr", "template", "last")  # the residual flow's parts


def parts(roles: list[str], task: str) -> list[str]:
    """Each position's part of the prompt: the template's first tokens (``start``, the sink among them), the question
    (``content``; for IOI, its sentence, which the reply's prefill holds), the instruction, the rest of the template
    and the prefill (``template``), and the last position."""
    out = []
    for r in roles:
        if r == "pre":
            out.append("start")
        elif r == "q" or (r == "prefill" and task == "ioi"):
            out.append("content")
        elif r == "instr":
            out.append("instr")
        elif r == "last":
            out.append("last")
        else:
            out.append("template")
    return out


def loaded() -> list[dict]:
    rows = []
    for path in sorted((OUT / "traces").glob("*.pt")):
        rows.append(torch.load(path, weights_only=False))
    order = list(json.loads(PROMPTS.read_text())["tasks"])
    return sorted(rows, key=lambda r: (order.index(r["item"]["task"]), r["item"]["i"]))


def unit_vectors(rows: list[dict]) -> tuple[torch.Tensor, int, int, int]:
    """Each prompt's |relevance| per unit over T: the poles (``[L, 2, f]`` flattened) then the heads (``[L, H]``),
    summed over positions."""
    L, _, f = rows[0]["pole"].shape
    H = rows[0]["heads"].shape[-1]
    V = torch.stack([torch.cat([r["pole"].abs().flatten(), r["heads"].sum(1).abs().flatten()]) / abs(r["T"]) for r in rows])
    return V, L, f, H


def unit_name(u: int, L: int, f: int, H: int) -> tuple:
    if u < L * 2 * f:
        l, rest = divmod(u, 2 * f)
        return ("n", l, rest % f, 1 if rest < f else -1)
    l, h = divmod(u - L * 2 * f, H)
    return ("h", l, h)


def compare(args) -> None:
    """The kinds of prompt against each other: where relevance passes (layers, heads or MLPs, the prompt's parts), where
    the answer is written, how concentrated each prompt's units are, which units the kinds share, and a PCA of the
    prompts by their units. Writes ``compare.json`` (the page's data) and prints the summary."""
    rows = loaded()
    tasks = list(dict.fromkeys(r["item"]["task"] for r in rows))
    ok = [r["item"]["T"] > 0 for r in rows]
    L, P_ = rows[0]["heads"].shape[0], None
    res = {"tasks": tasks, "prompts": [], "bands": BANDS}
    for r, good in zip(rows, ok):
        T = r["T"]
        mlp = r["neuron_sum"].sum(1) / T  # [L] the part of T passing through each layer's MLP
        attn = r["heads"].sum((1, 2)) / T
        direct = (r["direct_neuron_sum"] + r["direct_heads"].sum(-1)) / T
        by_role = {}
        for role in ROLES:
            m = torch.tensor([x == role for x in r["roles"]])
            if role == "prefill" and r["item"]["task"] == "ioi":
                pass
            by_role[role] = {"mlp": (r["neuron_sum"][:, m].sum(1) / T).tolist(), "attn": (r["heads"][:, m].sum((1, 2)) / T).tolist()}
        a = torch.cat([r["pole"].abs().flatten(), r["heads"].sum(1).abs().flatten()])
        srt = a.sort(descending=True).values
        cum = srt.cumsum(0) / srt.sum()
        res["prompts"].append({
            "task": r["item"]["task"], "i": r["item"]["i"], "q": r["item"]["q"], "answer": r["item"]["answer"],
            "distractor": r["item"]["distractor"], "T": T, "p": r["item"]["p_answer"], "rank": r["item"]["rank"], "ok": good,
            "toks": r["toks"], "roles": r["roles"], "mlp": mlp.tolist(), "attn": attn.tolist(), "direct": direct.tolist(),
            "by_role": by_role, "emb": (r["emb"] / T).tolist(),
            "flow": {part: (r["res"][:, torch.tensor([x == part for x in parts(r["roles"], r["item"]["task"])])].sum(1) / T).tolist()
                     for part in PARTS},
            "res": (r["res"] / T).tolist(),
            "n50": int((cum < 0.5).sum()) + 1, "n80": int((cum < 0.8).sum()) + 1,
            "pr": float(a.sum() ** 2 / (a ** 2).sum()),
        })
    V, L, f, H = unit_vectors(rows)
    V = V / V.sum(1, keepdim=True)  # each prompt's units as shares of its total
    band_of = torch.cat([torch.arange(L).repeat_interleave(2 * f), torch.arange(L).repeat_interleave(H)])
    sims = {}
    for name, lo, hi in BANDS + [("all", 0, L)]:
        m = (band_of >= lo) & (band_of < hi)
        X = V[:, m] - V[:, m].mean(0)  # centred: what a prompt uses more or less than the average prompt
        X = X / X.norm(dim=1, keepdim=True)
        S = X @ X.T
        sims[name] = S.tolist()
    res["sims"] = sims
    # units by kind: each kind's mean share, over its correct prompts
    idx = {t: [k for k, r in enumerate(rows) if r["item"]["task"] == t and ok[k]] for t in tasks}
    M = torch.stack([V[idx[t]].mean(0) for t in tasks])  # [tasks, units]
    total = M.sum(0)
    spec = M / total.clamp_min(1e-12)
    general = (M.min(0).values * len(tasks)).topk(40).indices
    res["general"] = [{"unit": unit_name(int(u), L, f, H), "mean": total[u].item() / len(tasks),
                       "by_task": M[:, u].tolist()} for u in general]
    res["specific"] = {}
    for k, t in enumerate(tasks):
        score = M[k] * spec[k]  # large in this kind, and mostly this kind's
        top = score.topk(25).indices
        res["specific"][t] = [{"unit": unit_name(int(u), L, f, H), "share": M[k, u].item(), "spec": spec[k, u].item()} for u in top]
    # PCA of the prompts on log shares of the 20,000 units with the most variance
    Y = torch.log(V + 1e-7)
    keep = V.var(0).topk(20000).indices
    Y = Y[:, keep] - Y[:, keep].mean(0)
    U, S_, Vt = torch.linalg.svd(Y, full_matrices=False)
    res["pca"] = (U[:, :3] * S_[:3]).tolist()
    res["pca_var"] = (S_[:3] ** 2 / (S_ ** 2).sum()).tolist()
    (OUT / "compare.json").write_text(json.dumps(res, ensure_ascii=False))
    # ------------------------------------------------------------------ summary
    def agg(ps, f):
        """A kind's value: the sum over its prompts of ``f(p)·T`` over the sum of their ``T``, so prompts with a small
        ``T`` don't dominate the shares."""
        return sum(f(p) * p["T"] for p in ps) / sum(p["T"] for p in ps)

    print(f"{len(rows)} prompts, {sum(ok)} with T > 0")
    print("throughput by band (mean over a kind's correct prompts, as a share of T): MLP / attention")
    print(f"{'kind':15s}" + "".join(f"{b:>22s}" for b, _, _ in BANDS) + "      n50      n80     PR")
    for t in tasks:
        ps = [res["prompts"][k] for k in idx[t]]
        line = f"{t:15s}"
        for _, lo, hi in BANDS:
            line += f"   {agg(ps, lambda p: sum(p['mlp'][lo:hi])):6.2f} / {agg(ps, lambda p: sum(p['attn'][lo:hi])):6.2f}  "
        line += f"{np.median([p['n50'] for p in ps]):9.0f}{np.median([p['n80'] for p in ps]):9.0f}{np.median([p['pr'] for p in ps]):7.0f}"
        print(line)
    print("where the answer is written directly (share of T by band: embedding+units at the last position)")
    for t in tasks:
        ps = [res["prompts"][k] for k in idx[t]]
        print(f"{t:15s}" + "".join(f"{agg(ps, lambda p: sum(p['direct'][lo:hi])):9.2f}" for _, lo, hi in BANDS))
    depths = (0, 4, 9, 13, 17, 21, 25, 29, 33, 36)
    for part in PARTS:
        print(f"share of T the residual holds at the prompt's {part} tokens, entering layer (36: after the last)")
        print(f"{'kind':15s}" + "".join(f"{'L' + str(l):>7s}" for l in depths))
        for t in tasks:
            ps = [res["prompts"][k] for k in idx[t]]
            print(f"{t:15s}" + "".join(f"{agg(ps, lambda p: p['flow'][part][l]):7.2f}" for l in depths))
    for name in ("workspace", "motor"):
        S = torch.tensor(sims[name])
        print(f"cosine between prompts' units, {name} band (rows: kind; diagonal: within, off-diagonal pairs left out)")
        print(" " * 15 + "".join(f"{t[:7]:>8s}" for t in tasks))
        for a_ in tasks:
            line = f"{a_:15s}"
            for b_ in tasks:
                blk = S[idx[a_]][:, idx[b_]]
                if a_ == b_:
                    n = len(idx[a_])
                    v = (blk.sum() - blk.diagonal().sum()) / (n * n - n)
                else:
                    v = blk.mean()
                line += f"{v.item():8.2f}"
            print(line)
    print("PCA variance", [round(v, 3) for v in res["pca_var"]])
    yn = [k for k, r in enumerate(rows) if ok[k] and r["item"]["answer"].strip() in ("Yes", "No")]
    yn_tasks = [t for t in tasks if sum(rows[k]["item"]["task"] == t for k in yn) >= 2]
    print(f"Yes/No prompts only ({len(yn)}: " + ", ".join(f"{t} {sum(rows[k]['item']['task'] == t for k in yn)}" for t in yn_tasks)
          + "), centred over them; within a kind / between kinds, and within the same answer / opposite answers")
    for name, lo, hi in BANDS:
        m = (band_of >= lo) & (band_of < hi)
        X = V[yn][:, m]
        X = X - X.mean(0)
        X = X / X.norm(dim=1, keepdim=True)
        S = X @ X.T
        task = [rows[k]["item"]["task"] for k in yn]
        ans = [rows[k]["item"]["answer"].strip() for k in yn]
        n = len(yn)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        def mean(cond):
            v = [S[i, j].item() for i, j in pairs if cond(i, j)]
            return np.mean(v) if v else float("nan")
        print(f"  {name:12s} same kind {mean(lambda i, j: task[i] == task[j]):+.2f}   other kind {mean(lambda i, j: task[i] != task[j]):+.2f}"
              f"   | same answer {mean(lambda i, j: ans[i] == ans[j]):+.2f}   opposite {mean(lambda i, j: ans[i] != ans[j]):+.2f}"
              f"   | same kind, opposite answer {mean(lambda i, j: task[i] == task[j] and ans[i] != ans[j]):+.2f}"
              f"   other kind, same answer {mean(lambda i, j: task[i] != task[j] and ans[i] == ans[j]):+.2f}")


GRAPHS = [("recall", 0), ("multihop", 0), ("arithmetic", 7), ("logic", 0), ("ioi", 0), ("theory_of_mind", 8),
          ("self_knowledge", 0), ("metacognition", 0), ("metacognition", 8), ("metacognition", 3), ("translation", 12),
          ("sentiment", 6)]
NODE_MIN = 0.01  # of T: a graph keeps the units at least this relevant (outside the template's first tokens)
MAX_NEURONS, MAX_HEADS = 160, 60
FLOW_MIN = 0.004  # of T: flows kept for drawing


def graph(args) -> None:
    """One prompt's graph (``rest``: ``task i`` pairs; default ``GRAPHS``): the units at least ``NODE_MIN`` of ``T``
    relevant outside the template's first tokens (the sink's, which every head reads by default: one node, ``start``),
    the embedding's, and the flows between them, each target's inflow split into its kept sources, ``start`` and the rest
    (which sum to its relevance: the check). Writes ``graphs/<task>_<i>.json`` and prints the nodes with their labels."""
    todo = [(args.rest[k], int(args.rest[k + 1])) for k in range(0, len(args.rest), 2)] or GRAPHS
    tr = Tracer(args.device)
    its = {(it["task"], it["i"]): it for it in checked()}
    hl = {(l, h): k for k, (l, h) in enumerate(zip(tr.head_labels["layer"], tr.head_labels["head"]))}
    (OUT / "graphs").mkdir(parents=True, exist_ok=True)
    L, H, f = tr.L, tr.H, tr.f
    for task, i in todo:
        it = its[task, i]
        ids, toks, roles = encode(tr.tok, it)
        cap = tr.run(ids, it["answer_tok"], it["distractor_tok"])
        rel = tr.relevances(cap)
        T, P = cap["T"], len(toks)
        start = torch.tensor([r == "pre" for r in roles], device=tr.dev)
        R, acts = rel["neurons"], rel["acts"]
        Rm = R.abs() * (~start)[None, :, None]
        cand = torch.nonzero(Rm >= NODE_MIN * abs(T))
        cand = cand[Rm[cand[:, 0], cand[:, 1], cand[:, 2]].argsort(descending=True)][:MAX_NEURONS]
        Hm = rel["heads"].abs() * (~start)[None, :, None]
        hc = torch.nonzero(Hm >= NODE_MIN * abs(T))
        hc = hc[Hm[hc[:, 0], hc[:, 1], hc[:, 2]].argsort(descending=True)][:MAX_HEADS]
        nodes, key = [], {}

        def add(node):
            key[node["id"]] = len(nodes)
            nodes.append(node)
        add({"id": "logit", "kind": "logit", "layer": L, "pos": P - 1, "R": T})
        add({"id": "start", "kind": "start", "layer": -1, "pos": 0, "R": None})
        for p in range(P):
            if not start[p] and abs(rel["emb"][p].item()) >= NODE_MIN * abs(T):
                add({"id": f"e{p}", "kind": "emb", "layer": -1, "pos": p, "R": rel["emb"][p].item()})
        for l, p, n in cand.tolist():
            a = acts[l, p, n].item()
            pole = 1 if a > 0 else -1
            row = n if pole > 0 else f + n
            lab = tr.labels[l]
            add({"id": f"n{l}.{n}.{p}", "kind": "neuron", "layer": l, "neuron": n, "pole": pole, "pos": p, "R": R[l, p, n].item(),
                 "act": a, "act_top": tr.act_top[l][row].item(), "fires": lab["fires"][row], "writes": lab["writes"][row],
                 "fires_conf": lab["fires_conf"][row], "writes_conf": lab["writes_conf"][row], "note": lab["note"][row],
                 "direct": rel["direct_neurons"][l, n].item() if p == P - 1 else 0.0})
        for l, p, h in hc.tolist():
            A = cap["A"][l][h, p, :p + 1]
            top = A.topk(min(3, p + 1))
            k = hl.get((l, h))
            lab = {c: tr.head_labels[c][k] for c in ("fires", "attends", "writes", "note")} if k is not None else {}
            add({"id": f"h{l}.{h}.{p}", "kind": "head", "layer": l, "head": h, "pos": p, "R": rel["heads"][l, p, h].item(),
                 "attn": [(int(q), float(v)) for q, v in zip(top.indices.tolist(), top.values.tolist())], **lab,
                 "direct": rel["direct_heads"][l, h].item() if p == P - 1 else 0.0})
        kept_n = {(nd["layer"], nd["pos"], nd["neuron"]): nd["id"] for nd in nodes if nd["kind"] == "neuron"}
        kept_h = {(nd["layer"], nd["pos"], nd["head"]): nd["id"] for nd in nodes if nd["kind"] == "head"}
        kept_e = {nd["pos"]: nd["id"] for nd in nodes if nd["kind"] == "emb"}
        edges, worst = [], 0.0
        for nd in nodes:
            if nd["kind"] == "logit":
                pos, cov, nh, nn = torch.tensor([P - 1], device=tr.dev), cap["cov_T"][None], L, L
            elif nd["kind"] == "neuron":
                pos, cov = torch.tensor([nd["pos"]], device=tr.dev), tr.neuron_cov(cap, nd["layer"], nd["neuron"], nd["pos"])
                nh, nn = nd["layer"] + 1, nd["layer"]
            elif nd["kind"] == "head":
                pos, cov = tr.head_cov(cap, nd["layer"], nd["head"], nd["pos"])
                nh, nn = nd["layer"], nd["layer"]
            else:
                continue
            with torch.no_grad():
                src = tr.sources(cap, cov, pos, nh, nn)
            total = src["emb"].sum() + (src["heads"].sum() if nh else 0) + (src["neurons"].sum() if nn else 0)
            worst = max(worst, abs(total.item() - nd["R"]) / abs(T))
            st = start[pos]
            from_start = src["emb"][st].sum() + (src["heads"][:, st].sum() if nh else 0) + (src["neurons"][:, st].sum() if nn else 0)
            kept = 0.0
            pl = pos.tolist()
            ins = []
            for k, q in enumerate(pl):
                if q in kept_e:
                    ins.append((kept_e[q], src["emb"][k].item()))
            for (l, q, n), name in kept_n.items():
                if l < nn and q in pl:
                    ins.append((name, src["neurons"][l, pl.index(q), n].item()))
            for (l, q, h), name in kept_h.items():
                if l < nh and q in pl:
                    ins.append((name, src["heads"][l, pl.index(q), h].item()))
            kept = sum(v for _, v in ins)
            nd["in_kept"], nd["in_start"] = kept, from_start.item()
            nd["in_rest"] = total.item() - kept - from_start.item()
            for name, v in ins:
                if abs(v) >= FLOW_MIN * abs(T):
                    edges.append({"src": name, "dst": nd["id"], "flow": v})
            if abs(from_start.item()) >= FLOW_MIN * abs(T):
                edges.append({"src": "start", "dst": nd["id"], "flow": from_start.item()})
        # QK: where the answer and the distractor (or their lower-case forms) stand in the prompt, and why the most
        # relevant heads at the last position look at one rather than the other
        def last_at(word):
            w = word.strip().lower()
            hits = [q for q in range(P - 1) if toks[q].strip().lower() == w]
            return hits[-1] if hits else None
        pa, pb = last_at(tr.tok.decode([it["answer_tok"]])), last_at(tr.tok.decode([it["distractor_tok"]]))
        qk = []
        if pa is not None and pb is not None:
            heads_last = [nd for nd in nodes if nd["kind"] == "head" and nd["pos"] == P - 1]
            heads_last.sort(key=lambda nd: -abs(nd["R"]))
            for nd in heads_last[:6]:
                sp = tr.qk_split(cap, nd["layer"], nd["head"], P - 1, pa, pb)
                entry = {"head": nd["id"], "margin": sp["margin"], "A_a": sp["A_a"], "A_b": sp["A_b"], "pos_a": pa, "pos_b": pb}
                for side, where in (("q", P - 1), ("ka", pa), ("kb", pb)):
                    src = sp[side]
                    flat = [("e", -1, -1, src["emb"][0].item())]
                    if src["heads"] is not None:
                        hv = src["heads"][:, 0]
                        for k in hv.flatten().abs().topk(min(8, hv.numel())).indices.tolist():
                            l, hh = divmod(k, H)
                            flat.append(("h", l, hh, hv[l, hh].item()))
                    if src["neurons"] is not None:
                        nv = src["neurons"][:, 0]
                        for k in nv.flatten().abs().topk(min(12, nv.numel())).indices.tolist():
                            l, n = divmod(k, f)
                            flat.append(("n", l, n, nv[l, n].item()))
                    flat.sort(key=lambda x: -abs(x[3]))
                    lst = []
                    for kind, l, u, v in flat[:10]:
                        if kind == "n":
                            a_ = acts[l, where, u].item()
                            row = u if a_ > 0 else f + u
                            lst.append({"unit": f"L{l} n{u} {'+' if a_ > 0 else '−'}", "v": v, "fires": tr.labels[l]["fires"][row],
                                        "writes": tr.labels[l]["writes"][row]})
                        elif kind == "h":
                            k_ = hl.get((l, u))
                            lst.append({"unit": f"L{l}h{u}", "v": v, "writes": tr.head_labels["writes"][k_] if k_ is not None else ""})
                        else:
                            lst.append({"unit": "embedding", "v": v})
                    entry[side] = lst
                    entry[side + "_total"] = (src["emb"].sum() + (src["heads"].sum() if src["heads"] is not None else 0)
                                              + (src["neurons"].sum() if src["neurons"] is not None else 0)).item()
                qk.append(entry)
                print(f"  QK {nd['id']}: looks at {toks[pa]!r}@{pa} {sp['A_a']:.2f} vs {toks[pb]!r}@{pb} {sp['A_b']:.2f}, margin "
                      f"{sp['margin']:+.2f} (query side {entry['q_total']:+.2f}; key side {entry['ka_total']:+.2f} at the answer, "
                      f"{entry['kb_total']:+.2f} at the distractor); "
                      f"query from " + ", ".join(f"{x['unit']} {x['v']:+.2f}" for x in entry["q"][:4]))
        res = {"qk": qk, "item": {k: it[k] for k in ("task", "i", "q", "instr", "prefill", "answer", "distractor", "p_answer", "rank", "top")},
               "T": T, "toks": toks, "roles": roles, "nodes": nodes, "edges": edges,
               "res": (rel["res"] / T).tolist(), "check": worst}
        (OUT / "graphs" / f"{task}_{i:02d}.json").write_text(json.dumps(res, ensure_ascii=False))
        nn_, nh_ = sum(nd["kind"] == "neuron" for nd in nodes), sum(nd["kind"] == "head" for nd in nodes)
        cover = nodes[0]["in_kept"] / T
        print(f"\n=== {task} {i}: {it['q']!r} → {it['answer']!r} vs {it['distractor']!r}  T {T:+.2f}  "
              f"{nn_} neurons, {nh_} heads, {len(edges)} flows; logit's inflow from kept nodes {cover:.2f}; worst inflow check {worst:.1e}")
        for nd in sorted(nodes[2:], key=lambda x: -abs(x["R"]))[:45]:
            where = f"@{nd['pos']:2d} {toks[nd['pos']]!r:12s}"
            if nd["kind"] == "neuron":
                print(f"  {nd['R'] / T:+.3f}  L{nd['layer']:2d} n{nd['neuron']:5d} {'+' if nd['pole'] > 0 else '−'} {where} "
                      f"a/top {nd['act'] / nd['act_top']:+.2f}  fires: {nd['fires'][:60]} | writes: {nd['writes'][:60]}")
            elif nd["kind"] == "head":
                att = ", ".join(f"{toks[q]!r}{v:.2f}" for q, v in nd["attn"])
                print(f"  {nd['R'] / T:+.3f}  L{nd['layer']:2d}h{nd['head']:2d}       {where} attn {att} | writes: {nd.get('writes', '')[:60]}")
            else:
                print(f"  {nd['R'] / T:+.3f}  emb          {where}")
        del cap, rel
        if args.device == "mps":
            torch.mps.empty_cache()


class Ablate:
    """Zero chosen neurons (their activation) and heads (their output before ``o_proj``) at chosen positions in the real
    model: no RelP, patterns free."""

    def __init__(self, tr: Tracer):
        self.tr, self.cut = tr, []

    def __enter__(self):
        tr, hooks, cut = self.tr, [], self.cut
        for l, lay in enumerate(tr.stack.layers):
            ns = [(p, n) for kind, ll, p, n in cut if kind == "n" and ll == l]
            hs = [(p, h) for kind, ll, p, h in cut if kind == "h" and ll == l]
            if ns:
                def down(m, x, ns=ns):
                    y = x[0].clone()
                    for p, n in ns:
                        y[0, p, n] = 0
                    return (y,)
                hooks.append(lay.mlp.down_proj.register_forward_pre_hook(down))
            if hs:
                def oproj(m, x, hs=hs):
                    y = x[0].clone()
                    for p, h in hs:
                        y[0, p, h * tr.hd:(h + 1) * tr.hd] = 0
                    return (y,)
                hooks.append(lay.self_attn.o_proj.register_forward_pre_hook(oproj))
        self.hooks = hooks
        return self

    def __exit__(self, *exc):
        for h in self.hooks:
            h.remove()


def ablate(args) -> None:
    """The graphs' nodes removed from the real model. Alone: each of the 16 most relevant units, its change in ``T``
    against its linear prediction, ``−R``. Together: the ``k`` most positively relevant units, against ``k`` random
    units at the same layers and positions (5 draws), and what the model then says. Writes ``ablate.json``."""
    tr = Tracer(args.device, labels=False)
    rng = np.random.default_rng(0)
    out = {}
    for path in sorted((OUT / "graphs").glob("*.json")):
        g = json.loads(path.read_text())
        it = next(x for x in checked() if x["task"] == g["item"]["task"] and x["i"] == g["item"]["i"])
        ids, _, _ = encode(tr.tok, it)
        a, d = it["answer_tok"], it["distractor_tok"]

        def measure(cut):
            ab = Ablate(tr)
            ab.cut = cut
            with ab:
                lg = tr.logits(ids)
            return (lg[a] - lg[d]).item(), tr.tok.decode([int(lg.argmax())]), lg.softmax(-1)[a].item()
        T0, top0, p0 = measure([])
        units = [nd for nd in g["nodes"] if nd["kind"] in ("neuron", "head")]
        units.sort(key=lambda nd: -abs(nd["R"]))

        def unit(nd):
            return ("n", nd["layer"], nd["pos"], nd["neuron"]) if nd["kind"] == "neuron" else ("h", nd["layer"], nd["pos"], nd["head"])

        def rand(nd):
            if nd["kind"] == "neuron":
                return ("n", nd["layer"], nd["pos"], int(rng.integers(tr.f)))
            return ("h", nd["layer"], nd["pos"], int(rng.integers(tr.H)))
        alone = []
        for nd in units[:16]:
            dT, _, _ = measure([unit(nd)])
            alone.append({"id": nd["id"], "R": nd["R"], "dT": dT - T0})
        groups = []
        pos_units = [nd for nd in units if nd["R"] > 0]
        for k in (5, 10, 20, 40):
            sel = pos_units[:k]
            Tk, topk, pk = measure([unit(nd) for nd in sel])
            ctrl = [measure([rand(nd) for nd in sel])[0] for _ in range(5)]
            groups.append({"k": len(sel), "T": Tk, "top": topk, "p": pk, "R_sum": sum(nd["R"] for nd in sel), "random_T": ctrl})
        neg = [nd for nd in units if nd["R"] < 0][:10]
        Tn, topn, pn = measure([unit(nd) for nd in neg])
        r = np.corrcoef([x["R"] for x in alone], [-x["dT"] for x in alone])[0, 1]
        out[path.stem] = {"T": T0, "top": top0, "p": p0, "alone": alone, "groups": groups,
                          "negative": {"k": len(neg), "T": Tn, "top": topn, "p": pn}, "r_alone": r}
        print(f"{path.stem:18s} T {T0:+6.2f} ({top0!r}) | alone: r(R, −ΔT) {r:+.2f} | top-k positive removed: " +
              "  ".join(f"k{gr['k']} {gr['T']:+6.2f} ({gr['top']!r}) rand {np.mean(gr['random_T']):+6.2f}" for gr in groups) +
              f" | 10 negative removed {Tn:+6.2f}", flush=True)
    (OUT / "ablate.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


NAMES = {"recall": "Recall", "arithmetic": "Arithmetic", "multihop": "Two-hop recall", "logic": "Logic",
         "theory_of_mind": "Theory of mind", "self_knowledge": "Self-knowledge", "metacognition": "Knowing what it knows",
         "ioi": "Who gets the drink (IOI)", "translation": "Translation", "sentiment": "Sentiment"}


def page(args) -> None:
    """The page's data (``rest``: the folder to write to): ``overview.json`` (each kind's residual flow and throughput,
    the similarity of kinds by band, the Yes/No control, the prompts) and ``graphs.json`` (the traced prompts, slimmed,
    with their QK splits and ablations)."""
    dest = Path(args.rest[0]) if args.rest else OUT / "page"
    dest.mkdir(parents=True, exist_ok=True)
    cmp_ = json.loads((OUT / "compare.json").read_text())
    tasks = cmp_["tasks"]
    ps = cmp_["prompts"]

    def agg(sel, f):
        return (sum(np.array(f(p)) * p["T"] for p in sel) / sum(p["T"] for p in sel)).round(4).tolist()
    by = {t: [p for p in ps if p["task"] == t and p["ok"]] for t in tasks}
    ov = {"tasks": [{"key": t, "name": NAMES[t], "n": len([p for p in ps if p["task"] == t]), "ok": len(by[t])} for t in tasks],
          "flow": {t: {part: agg(by[t], lambda p: p["flow"][part]) for part in PARTS} for t in tasks},
          "mlp": {t: agg(by[t], lambda p: p["mlp"]) for t in tasks}, "attn": {t: agg(by[t], lambda p: p["attn"]) for t in tasks},
          "prompts": [{"task": p["task"], "i": p["i"], "q": p["q"], "answer": p["answer"], "distractor": p["distractor"],
                       "T": round(p["T"], 2), "p": round(p["p"], 3), "ok": p["ok"], "pc": [round(v, 2) for v in pc[:2]]}
                      for p, pc in zip(ps, cmp_["pca"])], "pca_var": cmp_["pca_var"][:2]}
    idx = {t: [k for k, p in enumerate(ps) if p["task"] == t and p["ok"]] for t in tasks}
    ov["sims"] = {}
    for band in ("workspace", "motor"):
        S = np.array(cmp_["sims"][band])
        M = np.zeros((len(tasks), len(tasks)))
        for a, ta in enumerate(tasks):
            for b, tb in enumerate(tasks):
                blk = S[np.ix_(idx[ta], idx[tb])]
                M[a, b] = (blk.sum() - np.trace(blk)) / (blk.size - len(idx[ta])) if a == b else blk.mean()
        ov["sims"][band] = M.round(3).tolist()
    yn = {}
    for line in (OUT / "compare.log").read_text().splitlines():
        if "same kind" in line and "|" in line:
            w = line.split()
            yn[w[0]] = {"same_kind": float(w[3]), "other_kind": float(w[6]), "same_answer": float(w[10]), "opposite": float(w[12])}
    ov["yesno"] = yn
    (dest / "overview.json").write_text(json.dumps(ov, ensure_ascii=False))
    abl = json.loads((OUT / "ablate.json").read_text()) if (OUT / "ablate.json").exists() else {}
    graphs = []
    for task, i in GRAPHS:
        g = json.loads((OUT / "graphs" / f"{task}_{i:02d}.json").read_text())
        T = g["T"]
        nodes = []
        for nd in g["nodes"]:
            x = {"id": nd["id"], "k": nd["kind"][0], "l": nd["layer"], "p": nd["pos"], "R": None if nd["R"] is None else round(nd["R"] / T, 4)}
            if nd["kind"] == "neuron":
                x.update(n=nd["neuron"], s=nd["pole"], a=round(nd["act"] / nd["act_top"], 2), fires=nd["fires"], writes=nd["writes"],
                         d=round(nd["direct"] / T, 4))
            elif nd["kind"] == "head":
                x.update(h=nd["head"], attn=[[q, round(v, 2)] for q, v in nd["attn"]], attends=nd.get("attends", ""),
                         writes=nd.get("writes", ""), d=round(nd["direct"] / T, 4))
            if "in_kept" in nd:
                x.update(ik=round(nd["in_kept"] / T, 4), ist=round(nd["in_start"] / T, 4), ir=round(nd["in_rest"] / T, 4))
            nodes.append(x)
        edges = [[e["src"], e["dst"], round(e["flow"] / T, 4)] for e in g["edges"]]
        qk = []
        for q in g.get("qk", []):
            qk.append({k: q[k] for k in ("head", "margin", "A_a", "A_b", "pos_a", "pos_b", "q_total", "ka_total", "kb_total")} |
                      {side: [{**u, "v": round(u["v"], 3)} for u in q[side][:8]] for side in ("q", "ka", "kb")})
        ab = abl.get(f"{task}_{i:02d}", {})
        graphs.append({"key": f"{task}_{i:02d}", "task": task, "name": NAMES[task], "i": i, "q": g["item"]["q"], "instr": g["item"]["instr"],
                       "prefill": g["item"]["prefill"], "answer": g["item"]["answer"], "distractor": g["item"]["distractor"],
                       "T": round(T, 2), "p": round(g["item"]["p_answer"], 3), "top": g["item"]["top"][:3], "toks": g["toks"],
                       "roles": g["roles"], "nodes": nodes, "edges": edges, "qk": qk,
                       "res": [[round(v, 3) for v in row] for row in g["res"]],
                       "ablate": {"groups": [{k: gr[k] for k in ("k", "T", "top", "p")} | {"random": round(float(np.mean(gr["random_T"])), 2)}
                                             for gr in ab.get("groups", [])], "r": ab.get("r_alone"),
                                  "negative": ab.get("negative"), "alone": ab.get("alone", [])} if ab else None})
    # labels quote byte-fallback pieces, whose U+FFFD the page's host refuses
    (dest / "graphs.json").write_text(json.dumps(graphs, ensure_ascii=False).replace("�", ""))
    print(f"wrote {dest}: overview {(dest / 'overview.json').stat().st_size / 1e3:.0f} kB, graphs {(dest / 'graphs.json').stat().st_size / 1e3:.0f} kB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("check", "trace", "compare", "graph", "ablate", "page"))
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    ap.add_argument("--force", action="store_true", help="trace: redo prompts already traced")
    args = ap.parse_args()
    globals()[args.what](args)


if __name__ == "__main__":
    main()
