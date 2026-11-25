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
    if "gen" in it:  # a free reply's prefix: the model's own tokens, exactly as it generated them
        enc = tok(head, return_offsets_mapping=True, add_special_tokens=False)
        q0 = head.index(it["q"])
        spans = [("pre", 0), ("q", q0), ("post", q0 + len(it["q"]))]
        roles = [[r for r, start in spans if s >= start][-1] for s, _ in enc["offset_mapping"]] + ["prefill"] * len(it["gen"])
        roles[-1] = "last"
        ids = torch.tensor(enc["input_ids"] + it["gen"])
        return ids, [tok.decode([i]) for i in ids.tolist()], roles
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
        if "q" in cap:  # a choice being traced: this layer's queries and keys, rotated, one per head
            cap["q"][l], cap["k"][l] = query, k
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

    def run(self, ids: torch.Tensor, a: int, d: int, choice: tuple | None = None) -> dict:
        """One forward pass under RelP with the patterns held, and the backward pass from ``T``: everything a trace
        reads (activations, per-head outputs, patterns, the residual at each norm's input) and their gradients.

        With ``choice`` (layer M, head h, query position p, positions a and b), the target is instead that head's score
        margin ``s(p, a) − s(p, b)``: why it looks at a rather than b. Its query and keys read the residual entering
        layer M through the input norm and the q/k norms, all held (the LN rule), and the score is bilinear in them, so
        the half rule splits it: the relevances still sum to the margin at every cut below layer M."""
        keys = ("A", "z", "a", "g", "u", "x1", "x2", "x1g", "x2g") + (("q", "k") if choice else ())
        L, cap = self.L, {k: [None] * self.L for k in keys}
        patched = []
        if choice:  # the q/k norms of the chosen layer hold their scale, as the residual's norms do
            from neuron_lens.rules import _norm
            at = self.stack.layers[choice[0]].self_attn
            for norm in (at.q_norm, at.k_norm):
                norm._rules_orig_forward = norm.forward
                norm.forward = _norm.__get__(norm)
                patched.append(norm)
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
                if choice:
                    M, h, p, pa, pb = choice
                    q, k = cap["q"][M][0, h, p], cap["k"][M][0, h, pa] - cap["k"][M][0, h, pb]
                    scale = self.stack.layers[M].self_attn.scaling
                    T = (q @ k).detach() * scale + 0.5 * scale * ((q - q.detach()) @ k.detach() + q.detach() @ (k - k.detach()))
                T.backward()
        finally:
            for norm in patched:
                del norm.forward
                del norm._rules_orig_forward
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


def key(it: dict) -> str:
    return it.get("key") or f"{it['task']}_{it['i']:02d}"


def all_items() -> dict:
    """Every traced prompt by its key: the 120 (``checked``), the order control and the free replies, where run."""
    out = {key(it): it for it in checked()}
    for name in ("order", "free"):
        path = OUT / name / "items.json"
        if path.exists():
            out.update({key(it): it for it in json.loads(path.read_text())})
    return out


def item_set(name: str) -> list[dict]:
    return checked() if name == "main" else json.loads((OUT / name / "items.json").read_text())


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
    dest = OUT / "traces" if args.set == "main" else OUT / args.set / "traces"
    dest.mkdir(parents=True, exist_ok=True)
    L, f = tr.L, tr.f
    for it in item_set(args.set):
        if it.get("skip"):
            continue
        path = dest / f"{key(it)}.pt"
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


GRAPHS = ["recall_00", "multihop_00", "arithmetic_07", "logic_00", "ioi_00", "theory_of_mind_08", "self_knowledge_00",
          "metacognition_00", "metacognition_08", "metacognition_03", "translation_12", "sentiment_06"]
NODE_MIN = 0.01  # of T: a graph keeps the units at least this relevant (outside the template's first tokens)
MAX_NEURONS, MAX_HEADS = 160, 60
FLOW_MIN = 0.004  # of T: flows kept for drawing


def graph(args) -> None:
    """One prompt's graph (``rest``: ``task i`` pairs; default ``GRAPHS``): the units at least ``NODE_MIN`` of ``T``
    relevant outside the template's first tokens (the sink's, which every head reads by default: one node, ``start``),
    the embedding's, and the flows between them, each target's inflow split into its kept sources, ``start`` and the rest
    (which sum to its relevance: the check). Writes ``graphs/<task>_<i>.json`` and prints the nodes with their labels."""
    todo = args.rest or GRAPHS
    tr = Tracer(args.device)
    its = all_items()
    hl = {(l, h): k for k, (l, h) in enumerate(zip(tr.head_labels["layer"], tr.head_labels["head"]))}
    (OUT / "graphs").mkdir(parents=True, exist_ok=True)
    L, H, f = tr.L, tr.H, tr.f
    for gk in todo:
        it = its[gk]
        task, i = it["task"], it["i"]
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
        res = {"qk": qk, "key": gk, "item": {k: it.get(k) for k in ("task", "i", "q", "instr", "prefill", "answer", "distractor", "p_answer", "rank", "top", "set", "reply")},
               "T": T, "toks": toks, "roles": roles, "nodes": nodes, "edges": edges,
               "res": (rel["res"] / T).tolist(), "check": worst}
        (OUT / "graphs" / f"{gk}.json").write_text(json.dumps(res, ensure_ascii=False))
        nn_, nh_ = sum(nd["kind"] == "neuron" for nd in nodes), sum(nd["kind"] == "head" for nd in nodes)
        cover = nodes[0]["in_kept"] / T
        print(f"\n=== {gk}: {it['q']!r} → {it['answer']!r} vs {it['distractor']!r}  T {T:+.2f}  "
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
    out = json.loads((OUT / "ablate.json").read_text()) if (OUT / "ablate.json").exists() and not args.force else {}
    items = all_items()
    for path in sorted((OUT / "graphs").glob("*.json")):
        if path.stem in out:
            continue
        g = json.loads(path.read_text())
        it = items[path.stem]
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
         "ioi": "Who gets the drink (IOI)", "translation": "Translation", "sentiment": "Sentiment", "order": "Tallest or last"}
ORDER_GRAPHS = ["order_00", "order_08", "order_12"]  # Tom, Sam, Ben, taller: told and asked the top; swapped, the top; swapped, the bottom
FREE_GRAPHS = ["free_recall_00", "free_multihop_02", "free_arithmetic_07", "free_logic_00", "free_theory_of_mind_08", "free_self_knowledge_00"]


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
    abl = json.loads((OUT / "ablate.json").read_text()) if (OUT / "ablate.json").exists() else {}
    # the order control and the free replies, summarised
    if (OUT / "order" / "items.json").exists():
        rows = json.loads((OUT / "order" / "items.json").read_text())
        hs = json.loads((OUT / "order" / "heads.json").read_text())
        conds = []
        for o in ("told", "swapped"):
            for e in ("top", "bottom"):
                sel = [r for r in rows if r["order"] == o and r["end"] == e]
                hsel = [h for h in hs if h["order"] == o and h["end"] == e]
                conds.append({"order": o, "end": e, "n": len(sel), "right": sum(r["pick"] == "answer" for r in sel),
                              "T": float(np.mean([r["T"] for r in sel])), "T_middle": float(np.mean([r["T_middle"] for r in sel])),
                              "att": {k: float(np.mean([h["att"][k] for h in hsel])) for k in ("answer", "other end", "middle")},
                              "example": next(r["q"] for r in sel if r["names"][0] == "Tom" and r["rel"] == "taller")})
        ov["order"] = {"conds": conds, "n": len(rows)}
    if (OUT / "free" / "compare.json").exists():
        fc = json.loads((OUT / "free" / "compare.json").read_text())
        ov["free"] = {"bands": fc["bands"], "prompts": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items() if k != "flow"} |
                      ({"gen": round(e["flow"]["generated"][29], 3), "gen21": round(e["flow"]["generated"][21], 3),
                        "flow": {k: [round(x, 3) for x in v] for k, v in e["flow"].items()}} if "flow" in e else {}) for e in fc["prompts"]]}
    if (OUT / "decide.json").exists():
        ov["decide"] = {k: {**{x: v[x] for x in ("head", "pos_a", "pos_b", "margin", "A_a", "A_b", "toks", "roles")},
                            "res": [[round(x, 3) for x in row] for row in v["res"]],
                            "units": [{**u, "R": round(u["R"], 4)} for u in v["units"][:24]]}
                        for k, v in json.loads((OUT / "decide.json").read_text()).items()}
    (dest / "overview.json").write_text(json.dumps(ov, ensure_ascii=False).replace("\ufffd", ""))
    graphs = []
    for gk in GRAPHS + ORDER_GRAPHS + FREE_GRAPHS:
        if not (OUT / "graphs" / f"{gk}.json").exists():
            continue
        g = json.loads((OUT / "graphs" / f"{gk}.json").read_text())
        task, i, kind = g["item"]["task"], g["item"]["i"], g["item"].get("set") or "constrained"
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
        ab = abl.get(gk, {})
        graphs.append({"key": gk, "task": task, "set": kind, "name": NAMES[task], "i": i, "q": g["item"]["q"], "instr": g["item"]["instr"],
                       "reply": g["item"].get("reply"),
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


# --------------------------------------------------------------------------------------------- the order control

ORDER_NAMES = [("Tom", "Sam", "Ben"), ("Anna", "Lisa", "Mark"), ("Jack", "Kate", "Emma"), ("Paul", "Mary", "John")]
ORDER_ATTRS = {"height": ("taller", "shorter", "tallest", "shortest"), "age": ("older", "younger", "oldest", "youngest")}


def order_items() -> list[dict]:
    """The tallest-or-last control: three names in a chain (A R B, B R C), told in order or with the two statements
    swapped, asked for either end: 2 attributes × 2 relations × 2 orders × 2 ends × 4 name sets. Told in order, the
    chain's top is named first and its bottom last; swapped (B R C, A R B), the top is the third name told, the bottom
    the second, and the last name told is always the middle one, never the answer."""
    suffix = json.loads(PROMPTS.read_text())["suffix"]
    out = []
    for attr, (up, down, most, least) in ORDER_ATTRS.items():
        for word, top_w, bot_w in ((up, most, least), (down, least, most)):
            for order in ("told", "swapped"):
                for end in ("top", "bottom"):
                    for A, B, C in ORDER_NAMES:
                        s1, s2 = f"{A} is {word} than {B}.", f"{B} is {word} than {C}."
                        text = f"{s1} {s2}" if order == "told" else f"{s2} {s1}"
                        ans, dis = (A, C) if end == "top" else (C, A)
                        k = len(out)
                        out.append({"task": "order", "i": k, "key": f"order_{k:02d}", "set": "order",
                                    "q": f"{text} Who is the {top_w if end == 'top' else bot_w}?", "instr": suffix,
                                    "prefill": "Answer:", "answer": " " + ans, "distractor": " " + dis, "middle": " " + B,
                                    "attr": attr, "rel": word, "order": order, "end": end, "names": [A, B, C]})
    return out


def order(args) -> None:
    """The order control: each prompt's answer against the other end and the middle name, and for the six heads at the
    last position most relevant to T, where they look (each name's mentions) and why they look at the answer rather
    than the other end (the QK split, summed over prompts by the unit that builds the query). Writes ``order/items.json``
    and ``order/heads.json``; ``trace --set order`` then traces them."""
    tr = Tracer(args.device)
    hl = {(l, h): k for k, (l, h) in enumerate(zip(tr.head_labels["layer"], tr.head_labels["head"]))}
    (OUT / "order").mkdir(parents=True, exist_ok=True)
    rows, heads = [], []
    for it in order_items():
        ids, toks, roles = encode(tr.tok, it)
        a, d, m = (first_token(tr.tok, it[k]) for k in ("answer", "distractor", "middle"))
        lg = tr.logits(ids)
        names = torch.tensor([a, m, d])
        it.update(answer_tok=a, distractor_tok=d, middle_tok=m, T=(lg[a] - lg[d]).item(), T_middle=(lg[a] - lg[m]).item(),
                  p_answer=lg.softmax(-1)[a].item(), rank=int((lg > lg[a]).sum()), pick=["answer", "middle", "other end"][int(lg[names].argmax())],
                  top=[(tr.tok.decode([i]), round(v, 3)) for v, i in zip(*[x.tolist() for x in lg.softmax(-1).topk(3)])])
        rows.append(it)
        # where each name is mentioned
        P = len(toks)
        def at(word):
            return [q for q in range(P - 1) if toks[q].strip() == word.strip() and roles[q] == "q"]
        pos = {"answer": at(it["answer"]), "other end": at(it["distractor"]), "middle": at(it["middle"])}
        cap = tr.run(ids, a, d)
        R = torch.stack([(z[0, -1] * z.grad[0, -1]).sum(-1) for z in cap["z"]])  # [L, H] at the last position
        top = R.flatten().abs().topk(6).indices.tolist()
        for k in top:
            l, h = divmod(k, tr.H)
            A_ = cap["A"][l][h, -1]
            att = {name: float(A_[ps].sum()) for name, ps in pos.items()}
            sp = tr.qk_split(cap, l, h, P - 1, pos["answer"][0], pos["other end"][0])
            qn = sp["q"]["neurons"][:, 0]
            qh = sp["q"]["heads"][:, 0]
            heads.append({"key": it["key"], "order": it["order"], "end": it["end"], "head": f"L{l}h{h}", "R": R[l, h].item() / it["T"] if it["T"] else 0.0,
                          "att": att, "margin": sp["margin"], "q_total": sp["q"]["emb"].sum().item() + qn.sum().item() + qh.sum().item(),
                          "ka": sum(v.sum().item() for v in sp["ka"].values() if v is not None),
                          "kb": sum(v.sum().item() for v in sp["kb"].values() if v is not None),
                          "q_heads": {f"L{l2}h{h2}": qh[l2, h2].item() for l2, h2 in [divmod(j, tr.H) for j in qh.flatten().abs().topk(6).indices.tolist()]},
                          "q_neurons": {f"L{l2} n{n2} {'+' if cap['a'][l2][0, -1, n2] > 0 else '−'}": qn[l2, n2].item()
                                        for l2, n2 in [divmod(j, tr.f) for j in qn.flatten().abs().topk(6).indices.tolist()]},
                          "writes": tr.head_labels["writes"][hl[(l, h)]] if (l, h) in hl else ""})
        del cap
        if args.device == "mps":
            torch.mps.empty_cache()
        print(f"{it['key']} {it['order']:7s} {it['end']:6s} {it['q'][:70]!r:72s} picks {it['pick']:9s} T {it['T']:+6.2f} "
              f"vs middle {it['T_middle']:+6.2f}", flush=True)
    (OUT / "order" / "items.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    (OUT / "order" / "heads.json").write_text(json.dumps(heads, ensure_ascii=False))
    order_report(rows, heads)


def order_report(rows=None, heads=None) -> None:
    rows = rows or json.loads((OUT / "order" / "items.json").read_text())
    heads = heads or json.loads((OUT / "order" / "heads.json").read_text())
    print("\ncondition (answer's place among the names told)      right  middle  other end   mean T   T vs middle")
    where = {("told", "top"): "first", ("told", "bottom"): "last", ("swapped", "top"): "third", ("swapped", "bottom"): "second"}
    for o in ("told", "swapped"):
        for e in ("top", "bottom"):
            sel = [r for r in rows if r["order"] == o and r["end"] == e]
            c = {k: sum(r["pick"] == k for r in sel) for k in ("answer", "middle", "other end")}
            print(f"  {o:8s} {e:7s} (answer told {where[o, e]:6s})                  {c['answer']:3d}/{len(sel)}  {c['middle']:5d}  {c['other end']:8d}"
                  f"   {np.mean([r['T'] for r in sel]):+6.2f}   {np.mean([r['T_middle'] for r in sel]):+6.2f}")
    for attr in ORDER_ATTRS:
        for rel in sorted({r["rel"] for r in rows if r["attr"] == attr}):
            sel = [r for r in rows if r["rel"] == rel]
            print(f"  {rel:8s} right {sum(r['pick'] == 'answer' for r in sel):2d}/{len(sel)}")
    print("\nthe heads at the last position most relevant to T: attention to each name's mentions, by condition")
    for o in ("told", "swapped"):
        for e in ("top", "bottom"):
            sel = [h for h in heads if h["order"] == o and h["end"] == e]
            print(f"  {o:8s} {e:7s} answer {np.mean([h['att']['answer'] for h in sel]):.2f}  other end {np.mean([h['att']['other end'] for h in sel]):.2f}"
                  f"  middle {np.mean([h['att']['middle'] for h in sel]):.2f}   QK margin answer − other end {np.mean([h['margin'] for h in sel]):+.2f}")
    from collections import Counter
    cnt = Counter(h["head"] for h in heads)
    print("  most frequent:", ", ".join(f"{k} ×{v}" for k, v in cnt.most_common(10)))
    print("\nwhat builds their queries' preference for the answer over the other end (summed over prompts and heads)")
    for o in ("told", "swapped"):
        for e in ("top", "bottom"):
            sel = [h for h in heads if h["order"] == o and h["end"] == e]
            acc = Counter()
            for h in sel:
                for u, v in {**h["q_heads"], **h["q_neurons"]}.items():
                    acc[u] += v / len(sel)
            print(f"  {o:8s} {e:7s} " + ", ".join(f"{u} {v:+.2f}" for u, v in sorted(acc.items(), key=lambda x: -abs(x[1]))[:8]))



# --------------------------------------------------------------------------------------------- free replies

FREE_TOKENS = 96


def free(args) -> None:
    """Free replies: each question as a user would ask it (no format instruction, no prefill, "Answer yes or no."
    dropped), answered greedily with thinking off, up to ``FREE_TOKENS`` tokens. The answer's first appearance in the
    reply is found (a number's last digit, a word's first token), and the reply up to it becomes the prompt to trace,
    with the constrained trace's contrast: the answer's token against the distractor's in the same form (case, leading
    space). IOI is left out: its prompt is already a free continuation. Writes ``free/items.json``; ``trace --set free``
    then traces them."""
    import re
    tr = Tracer(args.device, labels=False)
    (OUT / "free").mkdir(parents=True, exist_ok=True)
    path = OUT / "free" / "items.json"
    done = {key(x): x for x in json.loads(path.read_text())} if path.exists() and not args.force else {}
    stop = {tr.tok.convert_tokens_to_ids("<|im_end|>"), tr.tok.eos_token_id}
    rows = []
    for it in checked():
        if it["task"] == "ioi":
            continue
        k = f"free_{it['task']}_{it['i']:02d}"
        if k in done:
            rows.append(done[k])
            continue
        q = it["q"].replace(" Answer yes or no.", "")
        head = tr.tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        hid = tr.tok(head, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = tr.lm.generate(torch.tensor([hid], device=tr.dev), attention_mask=torch.ones(1, len(hid), dtype=torch.long, device=tr.dev),
                                 max_new_tokens=FREE_TOKENS, do_sample=False, pad_token_id=tr.tok.eos_token_id)
        gen = out[0, len(hid):].tolist()
        cut = [j for j, t in enumerate(gen) if t in stop]
        gen = gen[:cut[0]] if cut else gen
        pieces = [tr.tok.decode([t]) for t in gen]
        reply = "".join(pieces)
        starts = np.cumsum([0] + [len(x) for x in pieces])
        ans = it["answer"].strip()
        row = {"task": it["task"], "i": it["i"], "key": k, "set": "free", "q": q, "instr": "", "reply": reply, "constrained": key(it)}
        if ans.isdigit():
            extra = it["prefill"][len("Answer:"):].strip()
            m = re.search(rf"(?<!\d){extra}{ans}(?!\d)", reply)
            char = m.start() + len(extra) if m else None
        else:
            m = re.search(rf"\b{re.escape(ans)}\b", reply, flags=re.I)
            char = m.start() if m else None
        if char is None:
            row.update(skip=True, reason="the answer is not in the reply")
        else:
            j = int(np.searchsorted(starts, char, side="right") - 1)
            piece = pieces[j]
            lead = piece[:len(piece) - len(piece.lstrip())]
            if starts[j] + len(lead) != char:
                row.update(skip=True, reason=f"the answer starts inside the token {piece!r}")
            else:
                word = reply[char:char + len(ans)]
                dis = it["distractor"].strip()
                if not ans.isdigit():
                    dis = (dis[0].upper() if word[0].isupper() else dis[0].lower()) + dis[1:]
                a_tok, d_tok = gen[j], first_token(tr.tok, lead + dis)
                lg = tr.logits(torch.tensor(hid + gen[:j]))
                p = lg.softmax(-1)
                row.update(gen=gen[:j], at=j, answer_tok=a_tok, distractor_tok=d_tok, answer=tr.tok.decode([a_tok]),
                           distractor=tr.tok.decode([d_tok]), T=(lg[a_tok] - lg[d_tok]).item(), p_answer=p[a_tok].item(),
                           rank=int((lg > lg[a_tok]).sum()), prefill="", aligned=first_token(tr.tok, lead + word) == a_tok,
                           top=[(tr.tok.decode([i]), round(v, 3)) for v, i in zip(*[x.tolist() for x in p.topk(3)])])
        rows.append(row)
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        shown = reply.replace("\n", " ⏎ ")
        print(f"{k:26s} {'skip: ' + row['reason'] if row.get('skip') else 'at token %3d  T %+6.2f' % (row['at'], row['T'])}  {shown[:90]!r}", flush=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))


# By hand, after reading every reply: which of the answer's appearances answers (0-based), where the first that isn't a
# restated sentence is a restated chain ("Sam > Ben"), working ("a red box and a blue box"), or a name ("National
# Basketball Association"); and the replies left out.
FREE_AT = {"free_logic_00": 3, "free_logic_01": 3, "free_logic_02": 2, "free_logic_03": 2, "free_logic_08": 1, "free_multihop_11": 1}
FREE_SKIP = {"free_logic_06": "its constrained target was a number's first digit", "free_logic_07": "its constrained target was a number's first digit",
             "free_logic_10": "the reply stops before it answers", "free_theory_of_mind_00": "the reply stops before it answers"}


def _plain(text: str) -> str:
    import re
    text = re.sub(r"[*_`$]", "", text.lower())
    text = re.sub(r"^\s*(\d+\.|[-•])\s*", "", text)
    return re.sub(r"\s+", " ", text).strip(" .,:;!?\"'")


def free_locate(args) -> None:
    """Where each free reply answers: the answer's appearances in the reply, less those inside a sentence that only
    restates the question (its plain text a part of the question's), or the one ``FREE_AT`` names. The reply is
    tokenized again, and the prefix up to that appearance kept only if the model's own greedy pick there is the answer's
    token, so the prefix is the one it generated. Rewrites ``free/items.json`` and prints each answer in context."""
    import re
    tr = Tracer(args.device, labels=False)
    path = OUT / "free" / "items.json"
    rows = json.loads(path.read_text())
    main = {key(it): it for it in checked()}
    for row in rows:
        it = main[row["constrained"]]
        reply, q = row["reply"], row["q"]
        ans = it["answer"].strip()
        if ans.isdigit():
            extra = it["prefill"][len("Answer:"):].strip()
            occ = [m.start() + len(extra) for m in re.finditer(rf"(?<!\d){extra}{ans}(?!\d)", reply)]
        else:
            occ = [m.start() for m in re.finditer(rf"\b{re.escape(ans)}\b", reply, flags=re.I)]
            if ans in ("Yes", "No"):  # a yes or a no answers only where it opens a sentence ("No, I don't…"), not "there is no…"
                occ = [c for c in occ if (c == 0 or re.search(r"[.!?\n]\s*$", reply[:c])) and re.match(r"(yes|no)\b[,.!]", reply[c:], flags=re.I)]
        plainq = _plain(q)
        spans = [(m.start(), m.end()) for m in re.finditer(r"[^\n.!?]+[.!?]?", reply)]
        def restated(c):
            for a, b in spans:
                if a <= c < b:
                    seg = _plain(reply[a:b])
                    return len(seg) >= 10 and seg in plainq
            return False
        keep = [c for c in occ if not restated(c)]
        row.pop("skip", None); row.pop("reason", None)
        if row["key"] in FREE_AT:
            keep = occ[FREE_AT[row["key"]]:FREE_AT[row["key"]] + 1]
        if row["key"] in FREE_SKIP:
            row.update(skip=True, reason=FREE_SKIP[row["key"]])
            print(f"{row['key']:28s} skip: {row['reason']}")
            continue
        if not keep:
            row.update(skip=True, reason="the answer is not in the reply" if not occ else "the answer only restates the question")
            print(f"{row['key']:28s} skip: {row['reason']}")
            continue
        char = keep[0]
        head = tr.tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        hid = tr.tok(head, add_special_tokens=False)["input_ids"]
        enc = tr.tok(reply, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        j = next((k for k, (a, b) in enumerate(offs) if a <= char < b), None)
        piece = reply[offs[j][0]:offs[j][1]] if j is not None else ""
        lead = piece[:len(piece) - len(piece.lstrip())]
        if j is None or offs[j][0] + len(lead) != char:
            row.update(skip=True, reason=f"the answer starts inside the token {piece!r}")
            print(f"{row['key']:28s} skip: {row['reason']}")
            continue
        word = reply[char:char + len(ans)]
        dis = it["distractor"].strip()
        if not ans.isdigit():
            dis = (dis[0].upper() if word[0].isupper() else dis[0].lower()) + dis[1:]
        a_tok, d_tok = ids[j], first_token(tr.tok, lead + dis)
        lg = tr.logits(torch.tensor(hid + ids[:j]))
        p = lg.softmax(-1)
        greedy = int(lg.argmax()) == a_tok
        row.update(gen=ids[:j], at=j, answer_tok=a_tok, distractor_tok=d_tok, answer=tr.tok.decode([a_tok]), distractor=tr.tok.decode([d_tok]),
                   T=(lg[a_tok] - lg[d_tok]).item(), p_answer=p[a_tok].item(), rank=int((lg > lg[a_tok]).sum()), prefill="", greedy=greedy,
                   occurrence=occ.index(char), n_occ=len(occ), char=char,
                   top=[(tr.tok.decode([i]), round(v, 3)) for v, i in zip(*[x.tolist() for x in p.topk(3)])])
        if not greedy:
            row.update(skip=True, reason="tokenized again, the reply's prefix isn't what the model generated")
        ctx = (reply[max(0, char - 60):char] + "⟦" + reply[char:char + len(ans)] + "⟧" + reply[char + len(ans):char + len(ans) + 25]).replace("\n", " ⏎ ")
        print(f"{row['key']:28s} {occ.index(char)}/{len(occ)} tok {j:3d} T {row['T']:+6.2f}{'' if greedy else ' NOT GREEDY'}  …{ctx}", flush=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))


FREE_PARTS = ("start", "content", "template", "generated", "last")


def free_compare(args) -> None:
    """Free replies against their constrained traces: where in the reply the answer comes, how much of T the model's
    own words carry by depth, and how alike the two traces' units are (centred cosine by band, against other prompts of
    the same kind traced constrained). Writes ``free/compare.json``."""
    frees = [x for x in json.loads((OUT / "free" / "items.json").read_text())]
    main = {r["item"]["task"] + f"_{r['item']['i']:02d}": r for r in loaded()}
    tasks = list(dict.fromkeys(r["item"]["task"] for r in main.values()))
    fr = {}
    for x in frees:
        path = OUT / "free" / "traces" / f"{x['key']}.pt"
        if not x.get("skip") and path.exists():
            fr[x["key"]] = torch.load(path, weights_only=False)
    keys_m = list(main)
    Vm, L, f, H = unit_vectors([main[k] for k in keys_m])
    Vm = Vm / Vm.sum(1, keepdim=True)
    keys_f = list(fr)
    Vf, *_ = unit_vectors([fr[k] for k in keys_f])
    Vf = Vf / Vf.sum(1, keepdim=True)
    mean = torch.cat([Vm, Vf]).mean(0)
    band_of = torch.cat([torch.arange(L).repeat_interleave(2 * f), torch.arange(L).repeat_interleave(H)])
    out = {"prompts": [], "bands": {}}
    for x in frees:
        e = {k: x.get(k) for k in ("key", "task", "i", "q", "reply", "at", "T", "p_answer", "skip", "reason", "constrained", "answer", "distractor",
                                   "char", "occurrence", "n_occ")}
        if x["key"] in fr:
            r = fr[x["key"]]
            roles = r["roles"]
            part = ["start" if q == "pre" else "content" if q == "q" else "generated" if q == "prefill" else "last" if q == "last" else "template" for q in roles]
            e["flow"] = {pt: (r["res"][:, torch.tensor([y == pt for y in part])].sum(1) / r["T"]).tolist() for pt in FREE_PARTS}
            e["T_constrained"] = main[x["constrained"]]["T"]
        out["prompts"].append(e)
    for name, lo, hi in BANDS:
        m = (band_of >= lo) & (band_of < hi)
        A = Vm[:, m] - mean[m]
        B = Vf[:, m] - mean[m]
        A = A / A.norm(dim=1, keepdim=True)
        B = B / B.norm(dim=1, keepdim=True)
        S = B @ A.T  # free × constrained
        same, kind, other = [], [], []
        for a, kf in enumerate(keys_f):
            c = fr[kf]["item"]["constrained"]
            t = fr[kf]["item"]["task"]
            for b, km in enumerate(keys_m):
                (same if km == c else kind if main[km]["item"]["task"] == t else other).append(S[a, b].item())
        out["bands"][name] = {"same": float(np.mean(same)), "kind": float(np.mean(kind)), "other": float(np.mean(other))}
        per = {}
        for a, kf in enumerate(keys_f):
            t = fr[kf]["item"]["task"]
            b = keys_m.index(fr[kf]["item"]["constrained"])
            per.setdefault(t, []).append((S[a, b].item(), np.mean([S[a, j].item() for j, km in enumerate(keys_m) if main[km]["item"]["task"] == t and km != keys_m[b]])))
        out["bands"][name]["by_task"] = {t: [float(np.mean([v[0] for v in vs])), float(np.mean([v[1] for v in vs]))] for t, vs in per.items()}
    (OUT / "free" / "compare.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"{len(frees)} free replies, {len(fr)} traced")
    print(f"{'kind':15s} found  right  answer at token (median)  T free / constrained   on its own words L21 / L29   last L29")
    for t in tasks:
        ps = [e for e in out["prompts"] if e["task"] == t]
        if not ps:
            continue
        tr_ = [e for e in ps if "flow" in e]
        if not tr_:
            print(f"{t:15s} {0:5d}/{len(ps)}")
            continue
        print(f"{t:15s} {len(tr_):5d}/{len(ps)} {sum(e['T'] > 0 for e in tr_):5d}  {np.median([e['at'] for e in tr_]):10.0f}"
              f"            {np.mean([e['T'] for e in tr_]):6.1f} / {np.mean([e['T_constrained'] for e in tr_]):5.1f}"
              f"              {np.mean([e['flow']['generated'][21] for e in tr_]):5.2f} / {np.mean([e['flow']['generated'][29] for e in tr_]):5.2f}"
              f"     {np.mean([e['flow']['last'][29] for e in tr_]):5.2f}")
    print("units, free against constrained (centred cosine): the same question / other questions of its kind / other kinds")
    for name, *_ in BANDS:
        b = out["bands"][name]
        print(f"  {name:12s} {b['same']:+.2f} / {b['kind']:+.2f} / {b['other']:+.2f}   by kind (same, kind): " +
              ", ".join(f"{t[:6]} {v[0]:+.2f} {v[1]:+.2f}" for t, v in b["by_task"].items()))


# --------------------------------------------------------------------------------------------- where a choice is made

DECIDE = ["order_00", "order_08", "order_12", "ioi_00", "logic_00", "self_knowledge_00"]


def decide(args) -> None:
    """Where a head's choice is made: for each graph (``rest``, default ``DECIDE``), the head at the last position that
    most prefers the answer's position over the distractor's (attention, from the graph's QK splits) has its score
    margin traced instead of T. The residual's relevance to the margin by depth and position (it sums to the margin
    at every cut below the head's layer) and the most relevant units. Writes ``decide.json``."""
    tr = Tracer(args.device)
    items = all_items()
    path = OUT / "decide.json"
    out = json.loads(path.read_text()) if path.exists() and not args.force else {}
    hl = {(l, h): k for k, (l, h) in enumerate(zip(tr.head_labels["layer"], tr.head_labels["head"]))}
    for name in args.rest or DECIDE:
        gk, _, chosen = name.partition("@")  # "order_08@h24.29.35": that head instead of the strongest mover
        g = json.loads((OUT / "graphs" / f"{gk}.json").read_text())
        if not g.get("qk"):
            continue
        best = dict(max(g["qk"], key=lambda q: q["A_a"] - q["A_b"]))
        if chosen:
            best["head"] = chosen
        M, h, p = (int(x) for x in best["head"][1:].split("."))
        it = items[gk]
        ids, toks, roles = encode(tr.tok, it)
        cap = tr.run(ids, it["answer_tok"], it["distractor_tok"], choice=(M, h, p, best["pos_a"], best["pos_b"]))
        margin = cap["T"]
        best["A_a"], best["A_b"] = cap["A"][M][h, p, best["pos_a"]].item(), cap["A"][M][h, p, best["pos_b"]].item()
        res = torch.stack([(x[0] * x.grad[0]).sum(-1) for x in cap["x1g"][:M + 1]])  # [M + 1, P]
        Rn = torch.stack([a[0].detach() * a.grad[0] for a in cap["a"][:M]])  # [M, P, f]
        Rh = torch.stack([(z[0] * z.grad[0]).sum(-1) for z in cap["z"][:M]])  # [M, P, H]
        acts = torch.stack([a[0].detach() for a in cap["a"][:M]])
        units = []
        for k in Rn.flatten().abs().topk(30).indices.tolist():
            l, rest = divmod(k, Rn.shape[1] * tr.f)
            q, n = divmod(rest, tr.f)
            a_ = acts[l, q, n].item()
            row = n if a_ > 0 else tr.f + n
            units.append({"unit": f"L{l} n{n} {'+' if a_ > 0 else '−'}", "id": f"n{l}.{n}.{q}", "pos": q, "R": Rn[l, q, n].item() / margin,
                          "fires": tr.labels[l]["fires"][row], "writes": tr.labels[l]["writes"][row]})
        for k in Rh.flatten().abs().topk(15).indices.tolist():
            l, rest = divmod(k, Rh.shape[1] * tr.H)
            q, hh = divmod(rest, tr.H)
            units.append({"unit": f"L{l}h{hh}", "id": f"h{l}.{hh}.{q}", "pos": q, "R": Rh[l, q, hh].item() / margin,
                          "writes": tr.head_labels["writes"][hl[(l, hh)]] if (l, hh) in hl else "",
                          "attn": [[int(i), round(float(v), 2)] for v, i in zip(*[x.tolist() for x in cap["A"][l][hh, q].topk(3)])]})
        units.sort(key=lambda u: -abs(u["R"]))
        emb = (cap["emb"][0] * cap["emb"].grad[0]).sum(-1)
        out[name] = {"head": best["head"], "pos_a": best["pos_a"], "pos_b": best["pos_b"], "margin": margin, "A_a": best["A_a"], "A_b": best["A_b"],
                   "res": (res / margin).tolist(), "emb": (emb / margin).tolist(), "units": units[:36], "toks": toks, "roles": roles,
                   "check": (emb.sum().item() - margin) / abs(margin)}
        print(f"\n=== {name}: why {best['head']} looks at {toks[best['pos_a']]!r} rather than {toks[best['pos_b']]!r}: margin {margin:+.2f}, "
              f"embedding sum {emb.sum().item():+.2f}")
        for L_ in sorted({0, M // 3, M // 2, 2 * M // 3, M}):
            r = res[L_] / margin
            top = r.abs().topk(5).indices.tolist()
            print(f"   entering L{L_:2d}: " + ", ".join(f"{toks[q]!r}@{q} {r[q]:+.2f}" for q in top))
        for u in units[:14]:
            print(f"   {u['R']:+.3f}  {u['unit']:15s} @{u['pos']:2d} {toks[u['pos']]!r:12s} {(u.get('fires') or u.get('writes') or '')[:70]}")
        del cap
        if args.device == "mps":
            torch.mps.empty_cache()
        path.write_text(json.dumps(out, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("check", "trace", "compare", "graph", "ablate", "page", "order", "free", "free_compare", "decide", "free_locate"))
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    ap.add_argument("--force", action="store_true", help="trace, ablate: redo prompts already done")
    ap.add_argument("--set", default="main", choices=("main", "order", "free"), help="trace: which prompts")
    args = ap.parse_args()
    globals()[args.what](args)


if __name__ == "__main__":
    main()
