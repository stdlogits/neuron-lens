"""A second mixed unit traced (findings §23): Qwen3-14B's L33 n4648, whose poles fire on baseball and on blood glucose,
and the baseball unit L29 n16174+, which fires as hard on SSH. Qwen3-14B, bf16 on CUDA or MPS, the released R-lens;
``scan`` runs on any catalog model (Qwen3-8B for the comparison), and ``lineage`` reads Qwen3-14B-Base's and Qwen3-32B's
weights.

The hand-written texts are in ``baseball_glucose_texts.json``; code files are the code slice's (their first 512 tokens).
A unit's value on a text is its pole-signed peak over positions from 1, as a share of its ``act_top``.

Parts (each writes ``data/summary/baseball_glucose_<part>.log``):

- ``lineage`` (weights only): the neuron's gate row, up row and down column against the same index in Qwen3-14B-Base's
  L33 and Qwen3-32B's L33 and L55-L58, with the layers' same-index medians.
- ``probe``: the chain's units on the texts (baseball and other sports; diabetes, glucose chemistry and recipe sugar;
  other diseases; SSH, shell and password text; the senses a read suggests as routes: batter, pitcher, control, numbers,
  dots, host, brewing, yeast, banking, Jenkins, bash, ...).
- ``scan``: every pole on baseball, glucose, SSH, password and control texts (SSH, password and control partly code
  files' first 256 tokens); for each pair of topics, the poles at a median of 0.3 or more on both while the controls'
  90th percentile stays under 0.15.
- ``files``: the chain's units on the code slice's files that use SSH, hold passwords, run shell ``pwd``, mention
  baseball, or none of these.
- ``routes``: the target's peak with its MLP input projected off a token group's directions there,
  ``u_t = J_midᵀ (γ ⊙ W_U[t])`` (SSH, credentials, git and FTP, baseball, glucose, random tokens).
- ``reads``: the best rank of token families (SSH, credentials, devops, brewing, yeast, banking, Harry Potter, Cuba, ...)
  in the chain's z-scored R-lens reads, against 4,096 random poles of each layer; and the target's gate read off its 64
  weight inputs' writes and off its two topics' token directions.
- ``sweep``: the target on random activation documents; those that reach 0.3 of a pole's ``act_top``, by topic.

    python scripts/checks/baseball_glucose.py lineage|probe|files|routes|reads|sweep
    NEURON_LENS_MODEL=Qwen/Qwen3-8B python scripts/checks/baseball_glucose.py scan
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

os.environ.setdefault("NEURON_LENS_MODEL", "Qwen/Qwen3-14B")
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors import safe_open  # noqa: E402

from neuron_lens import arch, metric  # noqa: E402
from neuron_lens.catalog import MODEL, context, documents, entry, kept, lobe_signs, neuron_table, tokenizer  # noqa: E402
from neuron_lens.lenses import load_lens  # noqa: E402
from neuron_lens.neurons import poles  # noqa: E402
from neuron_lens.paths import DATA, SUMMARY  # noqa: E402
from neuron_lens.weights import tensors  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "mps"
TEXTS = json.loads(Path(__file__).with_name("baseball_glucose_texts.json").read_text())
TARGET, BRIDGE = (33, 4648), (29, 16174, 1)
BASEBALL = [(31, 6703, -1), (32, 2063, 1)]
GLUCOSE = [(32, 1921, -1), (32, 15513, 1)]  # diabetes and insulin; sugar
CHAIN = [(*TARGET, 1), (*TARGET, -1), BRIDGE] + BASEBALL + GLUCOSE
CODE = DATA / "corpus" / "act-code.jsonl"
SSH_FILE = re.compile(r"import paramiko|from paramiko|twisted\.conch|asyncssh|StrictHostKeyChecking|known_hosts|ssh_config|sshd|\bssh -|ssh_key|SSHClient", re.M)
PW_FILE = re.compile(r"\bpwd\b|passwd|password", re.M)
SHELL_FILE = re.compile(r"^\s*\$?\s*pwd\s*$|os\.getcwd|\$PWD|\bPWD\b", re.M)
BB_FILE = re.compile(r"baseball|inning|pitcher|MLB|batting", re.M | re.I)
FAMILIES = {
    "baseball": r"^ ?(baseball|Baseball|MLB|pitcher|pitchers|pitching|inning|innings|bullpen|outfield|batting)$",
    "glucose": r"^ ?(diabetes|Diabetes|diabetic|insulin|glucose|血糖|糖尿病|glyc)$",
    "sugar": r"^ ?(sugar|Sugar|sugars|糖|蔗|sweetness)$",
    "ssh": r"^[\s(_./=,\t]*(ssh|SSH|sftp|scp|paramiko|openssh|OpenSSH)$",
    "credentials": r"^[\s(_./=,\t]*(pwd|PWD|Pwd|passwd|password|Password|PASSWORD|username|USERNAME|bcrypt|userid|UserName)$",
    "devops": r"^[\s(_./-]*(git|Git|repo|Repo|github|FTP|ftp|scp|bash|Bash|nfs|NFS|ldap|docker|Jenkins|ansible|rpm|yum|krb)$",
    "gl": r"^[\s(_./=,\t]*(gl|GL|glGet|GLenum|GLint|GLuint|glBind)$",
    "brewing": r"^ ?(brewers|brewing|breweries|brewery|Brewery|brewed|brew|beer|beers|酿酒|啤酒)$",
    "yeast": r"^ ?(yeast|fungal|fungus|fungi|ferment|fermentation|酵|inoc)$",
    "banking": r"^ ?(bank|banking|banks|Banking|mortgage|mortgages|bankers|银行|銀行|商业银行)$",
    "potter": r"^ ?(Harry|Hermione|Dumbledore|Rowling|Ginny)$",
    "cuba": r"^ ?(Havana|Castro|Guantanamo|cigar|cigars|Cuba|Cuban)$",
}
TOPIC_TOKENS = (  # the target's two topics as its read spells them, removed from its gate in ``reads``
    r"^ ?(baseball|Baseball|MLB|pitcher|pitchers|pitching|pitched|pitch|Pitch|inning|innings|bullpen|outfield|infield|batting|fastball|"
    r"ballpark|RBI|Rookie|softball|baseman|strikeouts|Marlins|Dodgers|Padres|Braves|Twins|Orioles|Mets|Nationals|Tampa|Phillies|Astros|"
    r"Sox|Cubs|Yankees|垒|职|職|_pitch|ITCH|Thrown|ittings|pitches|preseason)$",
    r"^ ?(diabetes|Diabetes|diabetic|insulin|glucose|血糖|糖尿病|糖尿|glyc|gly|Gly|ucose|胰|糖|Sugar|sugar|sugars|白糖|糖果|蔗|甜|sweetness|"
    r"carbs|carb|carbohydrate|carbohydrates|GLUT|glu|gluc|苷|醛|neurop|fasting|syrup|Sug|sug|mell|abetic|abetes|pancre|pancreatic|酮|甘|葡|"
    r"葡萄|Zucker|IDF|ulin|甜蜜|Sweet|sweet|cane|uctose|bohydr|suc|ADA)$")
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def name(u):
    return f"L{u[0]} n{u[1]}{'+' if u[2] > 0 else '−'}"


def need_14b():
    assert MODEL == "Qwen/Qwen3-14B", "this part is about Qwen3-14B's units"


def model():
    from transformers import AutoModelForCausalLM
    torch.set_grad_enabled(False)
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEV).eval().requires_grad_(False)


class Probe:
    """Pole-signed peaks of units over a text, as shares of their act_top."""

    def __init__(self, lm, units):
        self.lm, self.units, self.cap = lm, units, {}
        self.top = {u: entry(*u)["act_top"] for u in units}
        for l in sorted({u[0] for u in units}):
            lm.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: self.cap.__setitem__(l, a[0][0].float().cpu()))

    def __call__(self, ids):
        self.lm(ids[None].to(DEV))
        return [float((self.cap[l][1:, n] * p).max()) / self.top[(l, n, p)] for l, n, p in self.units]


def code_files():
    docs = [json.loads(l)["text"] for l in open(CODE)]
    rng = random.Random(0)
    ssh = [d for d in docs if SSH_FILE.search(d)]
    return {"ssh": ssh,
            "password": rng.sample([d for d in docs if PW_FILE.search(d) and not SSH_FILE.search(d)], len(ssh)),
            "shell pwd": rng.sample([d for d in docs if SHELL_FILE.search(d) and not SSH_FILE.search(d)], len(ssh)),
            "baseball code": rng.sample([d for d in docs if BB_FILE.search(d)], len(ssh)),
            "random": rng.sample([d for d in docs if not any(r.search(d) for r in (SSH_FILE, PW_FILE, SHELL_FILE, BB_FILE))], len(ssh))}


def lineage(args):
    need_14b()
    def opener(repo):
        idx = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))["weight_map"]
        return lambda k: safe_open(hf_hub_download(repo, idx[k]), "pt").get_tensor(k).float()
    gn = lambda x: x / x.norm(dim=-1, keepdim=True)
    def mats(get, L):
        p = f"model.layers.{L}.mlp."
        return gn(get(p + "gate_proj.weight")), gn(get(p + "up_proj.weight")), gn(get(p + "down_proj.weight").T)
    L, N = TARGET
    G, U, D = mats(opener(MODEL), L)
    S = torch.arange(0, G.shape[0], 37)
    med = lambda A, B: float((A[S] * B[S]).sum(-1).median())
    for repo, layers in (("Qwen/Qwen3-14B-Base", [33]), ("Qwen/Qwen3-32B", [33, 55, 56, 57, 58])):
        get = opener(repo)
        for l in layers:
            g, u, d = mats(get, l)
            cg = g[:G.shape[0]] @ G[N]
            i = int(cg.abs().argmax())
            say(f"{repo} L{l}: n{N} gate {float(g[N] @ G[N]):+.3f} up {float(u[N] @ U[N]):+.3f} down {float(d[N] @ D[N]):+.3f}"
                f" | layer medians {med(g, G):.3f} {med(u, U):.3f} {med(d, D):.3f} | best gate match n{i} {float(cg[i]):+.3f}")


def probe(args):
    need_14b()
    lm, tok = model(), tokenizer()
    P = Probe(lm, CHAIN)
    for u in CHAIN:
        e = entry(*u)
        say(f"{name(u):14s} act_top {e['act_top']:6.1f} | {e['fires']} | drivers {str(e['drivers_str'])[:60]}")
    say("\n" + f"{'':22s}" + "".join(f"{name(u):>13s}" for u in CHAIN))
    for k, t in TEXTS.items():
        say(f"{k:22s}" + "".join(f"{v:13.2f}" for v in P(tok(t, return_tensors="pt").input_ids[0])))


def scan(args):
    docs = [json.loads(l)["text"] for l in open(CODE)]
    rng = random.Random(1)
    groups = {"baseball": [TEXTS[k] for k in ("baseball_recap", "baseball_stats", "baseball_trade", "baseball_zh", "lhp_rhp", "batting_avg_numbers")],
              "ssh": [TEXTS["ssh_config"]] + rng.sample([d for d in docs if SSH_FILE.search(d)], 6),
              "password": [TEXTS["password_code"]] + rng.sample([d for d in docs if PW_FILE.search(d) and not SSH_FILE.search(d)], 6),
              "glucose": [TEXTS[k] for k in ("diabetes_prose", "diabetes_t1_blog", "glucose_biochem", "diabetes_zh", "metformin", "a1c")],
              "control": [TEXTS[k] for k in ("basketball", "football", "hockey", "cricket", "cancer", "hypertension", "sugar_recipe", "news_random", "golf")]
              + rng.sample([d for d in docs if not (SSH_FILE.search(d) or PW_FILE.search(d))], 4)}
    lm, tok = model(), tokenizer()
    nL = lm.config.num_hidden_layers
    top = [torch.tensor(pq.read_table(neuron_table("acts", L), columns=["act_top"]).column("act_top").to_numpy()).float() for L in range(nL)]
    cap = {}
    for L in range(nL):
        lm.model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, a, L=L: cap.__setitem__(L, a[0][0, 1:].float()))
    res = {}
    for g, texts in groups.items():
        R = []
        for t in texts:
            lm(tok(t, return_tensors="pt").input_ids[:, :256].to(DEV))
            R.append(torch.stack([torch.cat([cap[L].max(0).values, (-cap[L]).max(0).values]).cpu() / top[L].clamp_min(1e-3) for L in range(nL)]))
        res[g] = torch.stack(R)
    med = {g: r.median(0).values for g, r in res.items()}
    ctl = res["control"].quantile(0.9, dim=0)
    f = med["baseball"].shape[1] // 2
    say(f"{MODEL}: poles at a median ≥ 0.3 on both topics, controls' q90 < 0.15")
    for a, b in (("baseball", "glucose"), ("baseball", "ssh"), ("glucose", "password"), ("glucose", "ssh"), ("baseball", "password")):
        both = (med[a] >= 0.3) & (med[b] >= 0.3) & (ctl < 0.15)
        say(f"\n  {a} + {b}: {int(both.sum())} poles; {a} alone (any control) {int(((med[a] >= 0.3) & (ctl < 0.3)).sum())}, "
            f"{b} alone {int(((med[b] >= 0.3) & (ctl < 0.3)).sum())}")
        for L, i in both.nonzero().tolist():
            n, p = (i, 1) if i < f else (i - f, -1)
            lab = pq.read_table(neuron_table("labels", L), columns=["fires"]).column("fires")[i].as_py()
            say(f"    {name((L, n, p)):14s} {a} {med[a][L, i]:.2f}  {b} {med[b][L, i]:.2f}  controls q90 {ctl[L, i]:.2f} | {lab}")


def files(args):
    need_14b()
    lm, tok = model(), tokenizer()
    P = Probe(lm, CHAIN)
    say("median / q90 / share ≥ 0.3 of each unit on each kind of code file (first 512 tokens)")
    say(f"{'':18s}" + "".join(f"{name(u):>20s}" for u in CHAIN))
    for g, ds in code_files().items():
        R = torch.tensor([P(tok(d, return_tensors="pt").input_ids[0, :512]) for d in ds])
        say(f"{g + f' ({len(ds)})':18s}" + "".join(f"{R[:, i].median():8.2f}{R[:, i].quantile(0.9):6.2f}{(R[:, i] >= 0.3).float().mean():6.2f}" for i in range(len(CHAIN))))


def token_ids(tok, words):
    return sorted({tok.encode(w, add_special_tokens=False)[0] for w in words if len(tok.encode(w, add_special_tokens=False)) == 1})


def routes(args):
    need_14b()
    lm, tok = model(), tokenizer()
    L, N = TARGET
    nm = arch.names(MODEL)
    hd = tensors(MODEL, sorted({nm["unembed"], nm["norm"]}))
    W_U, gam = hd[nm["unembed"]].float(), hd[nm["norm"]].float()
    J = load_lens("hub@mid", MODEL)[L].float()
    groups = {"ssh": [" ssh", "ssh", " SSH", "SSH", " sftp", " scp", " paramiko", "_ssh", ".ssh", " OpenSSH"],
              "credentials": [" pwd", "pwd", "PWD", " PWD", "(pwd", "_pwd", " password", "password", " Password", " passwd", "_password", " PASSWORD"],
              "git/ftp": [" git", " Git", " Repo", " repo", " FTP", " ftp", " github"],
              "baseball": [" baseball", " Baseball", " MLB", " pitcher", " pitchers", " inning", " innings", " bullpen", " outfield", " batting"],
              "glucose": [" diabetes", " Diabetes", " diabetic", " insulin", " glucose", "血糖", "糖尿病", " glyc"]}
    ids = {k: token_ids(tok, v) for k, v in groups.items()}
    ids["random"] = random.Random(0).sample(range(1000, 100000), 10)
    span = {k: torch.linalg.qr(J.T @ (gam[:, None] * W_U[v].T))[0] for k, v in ids.items()}
    cap, state = {}, {"Q": None}
    lay = lm.model.layers[L]
    def pre(m, a):
        if state["Q"] is not None:
            x, Q = a[0].float(), state["Q"].to(a[0].device)
            return ((x - (x @ Q) @ Q.T).to(a[0].dtype),)
    lay.post_attention_layernorm.register_forward_pre_hook(pre)
    lay.mlp.down_proj.register_forward_pre_hook(lambda m, a: cap.__setitem__("a", a[0][0, 1:, N].float().cpu()))
    texts = ["baseball_recap", "batting_avg_numbers", "pitch_control", "diabetes_t1_blog", "glycemic_control", "metformin", "ssh_config", "shell_pwd"]
    def peak(k, p):
        lm(tok(TEXTS[k], return_tensors="pt").input_ids.to(DEV))
        return float((cap["a"] * p).max())
    for p in (1, -1):
        top = entry(L, N, p)["act_top"]
        base = {k: peak(k, p) for k in texts}
        say(f"\n{name((L, N, p))}: peak with the MLP input projected off a group's span, as a share of the unablated peak")
        say(f"{'':14s}" + "".join(f"{k[:13]:>14s}" for k in texts))
        say(f"{'unablated':14s}" + "".join(f"{base[k] / top:14.2f}" for k in texts) + "   (share of act_top)")
        for g, Q in span.items():
            state["Q"] = Q
            say(f"{'−' + g:14s}" + "".join(f"{peak(k, p) / base[k]:14.2f}" for k in texts))
            state["Q"] = None


def reads(args):
    need_14b()
    tok, keep = tokenizer(), kept()
    strs = [tok.decode([int(i)]) for i in keep]
    fam = {k: torch.tensor([i for i, s in enumerate(strs) if re.match(rx, s)]) for k, rx in FAMILIES.items()}
    nm = arch.names(MODEL)
    hd = tensors(MODEL, sorted({nm["unembed"], nm["norm"]}))
    W_U, gam = hd[nm["unembed"]].float(), hd[nm["norm"]].float().to(DEV)
    Wk = W_U[keep].to(DEV, torch.float16)
    Jm = load_lens("hub@mid", MODEL)
    g0 = torch.Generator().manual_seed(0)
    def reader(L):
        J = Jm[L].to(DEV).float()
        def R(v):
            t = v.to(DEV).float() @ J.T
            t = t / t.norm(dim=-1, keepdim=True)
            return ((t * gam).half() @ Wk.T).float()
        return R
    def ranks(z):
        order = z.argsort(descending=True)
        rk = torch.empty_like(order)
        rk[order] = torch.arange(len(order), device=order.device)
        return rk
    say("1. best rank (of 124k kept tokens) of each family in the chain's z-scored R-lens reads; in brackets, the share of 4,096 random poles of the layer with one at that rank or better")
    for L in sorted({u[0] for u in CHAIN}):
        R, P = reader(L), poles(MODEL, L, lobe_signs(L))
        f = P["write"].shape[1]
        st = R(P["vec"][torch.randint(0, 2 * f, (1024,), generator=g0)])
        mu, sd = st.mean(0), st.std(0).clamp_min(1e-4)
        null = {k: [] for k in fam}
        for ch in torch.randint(0, 2 * f, (4096,), generator=g0).split(256):
            z = (R(P["vec"][ch]) - mu) / sd
            for k, ix in fam.items():
                null[k].append(((z > z[:, ix.to(DEV)].max(1).values[:, None]).sum(1) + 1).cpu())
        null = {k: torch.cat(v) for k, v in null.items()}
        for u in [u for u in CHAIN if u[0] == L]:
            rk = ranks((R(P["vec"][u[1] if u[2] > 0 else f + u[1]][None])[0] - mu) / sd)
            parts = []
            for k, ix in fam.items():
                r = int(rk[ix.to(DEV)].min()) + 1
                parts.append(f"{k} #{r} {strs[int(ix[rk[ix.to(DEV)].argmin().cpu()])]!r} [{(null[k] <= r).float().mean():.4f}]")
            say(f"  {name(u)}: " + " | ".join(parts))
        del P
    L, N = TARGET
    R, P = reader(L), poles(MODEL, L, lobe_signs(L))
    f, s2 = P["write"].shape[1], P["s2"]
    st = R(P["vec"][torch.randint(0, 2 * f, (1024,), generator=g0)])
    mu, sd = st.mean(0), st.std(0).clamp_min(1e-4)
    topics = {i for i, s in enumerate(strs) if any(re.match(rx, s) for rx in TOPIC_TOKENS)}
    def show(label, cov, k=40):
        v = metric.vector(cov[None], s2)[0]
        order = ((R((v / v.norm())[None])[0] - mu) / sd).argsort(descending=True).tolist()
        say(f"\n  {label}\n    top 16: " + " | ".join(repr(strs[i]) for i in order[:16]))
        say(f"    top {k} outside the two topics' families: " + " | ".join(repr(strs[i]) for i in [i for i in order if i not in topics][:k]))
    say("\n2. the target's gate read, raw and with parts removed")
    gate = P["a"][N].float()
    show("raw", gate)
    e = next(r for r in pq.read_table(neuron_table("edges", L), columns=["neuron", "pole", "in_layer", "in_neuron", "in_pole"]).to_pylist()
             if r["neuron"] == N and r["pole"] == 1)
    ins = [(l, n, p) for l, n, p in zip(e["in_layer"], e["in_neuron"], e["in_pole"]) if l >= 0][:64]
    writes = {}
    def wcov(u):
        if u[0] not in writes:
            writes[u[0]] = poles(MODEL, u[0], lobe_signs(u[0]))["write"]
        return metric.covector((u[2] * writes[u[0]][:, u[1]])[None], s2)[0]
    Q = torch.linalg.qr(torch.stack([wcov(u) for u in ins]).float().T)[0]
    say(f"\n  share of the gate's squared norm in its {len(ins)} weight inputs' span: {float((Q.T @ gate).norm() ** 2 / gate.norm() ** 2):.3f}")
    show(f"off its {len(ins)} weight inputs' writes", gate - Q @ (Q.T @ gate))
    J = Jm[L].float()
    tix = sorted(topics)
    U = (J.T @ (gam.cpu()[:, None] * W_U[keep[tix]].T)).T
    Qt = torch.linalg.qr(metric.covector(U, s2).T.float())[0]
    say(f"  share of the gate's squared norm in its topics' {len(tix)} token directions: {float((Qt.T @ gate).norm() ** 2 / gate.norm() ** 2):.3f}")
    show("off its two topics' token directions", gate - Qt @ (Qt.T @ gate))


def sweep(args):
    need_14b()
    L, N = TARGET
    docs, tok = documents(), tokenizer()
    pick = sorted(random.Random(1).sample(range(len(docs)), args.docs))
    lm = model()
    cap = {}

    class Stop(Exception):
        pass

    def hook(m, a):
        cap["a"] = a[0][:, :, N].float().cpu()
        raise Stop
    lm.model.layers[L].mlp.down_proj.register_forward_pre_hook(hook)
    tp, tm = entry(L, N, 1)["act_top"], entry(L, N, -1)["act_top"]
    out = []
    for b in range(0, len(pick), 16):
        seqs = [docs[i].long() for i in pick[b:b + 16]]
        T = max(len(s) for s in seqs)
        ids, mask = torch.zeros(len(seqs), T, dtype=torch.long), torch.zeros(len(seqs), T, dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)], mask[j, :len(s)] = s, 1
        try:
            lm(ids.to(DEV), attention_mask=mask.to(DEV))
        except Stop:
            pass
        for j, i in enumerate(pick[b:b + 16]):
            x = cap["a"][j, 1:len(seqs[j])]
            out.append((i, float(x.max()) / tp, int(x.argmax()) + 1, float(-x.min()) / tm, int(x.argmin()) + 1))
    BB = re.compile(r"baseball|inning|pitcher|pitching|MLB|batting|home run|homer|bullpen|outfield|infield|shortstop|RBI|\bERA\b|strikeout|Yankees|Red Sox|Dodgers|Cubs|Mets|Phillies|Cardinals|Orioles|Braves|Astros|Mariners|Padres|Pirates|Brewers|softball|棒球|投手|打者|全垒打|全壘打|安打|大联盟|大聯盟|職棒|职棒", re.I)
    GL = re.compile(r"diabet|glucose|insulin|blood sugar|glyc|carbohydrate|monosacchar|disacchar|fructose|sucrose|lactose|血糖|糖尿|胰岛|胰島|葡萄糖|A1C", re.I)
    for p, (vi, ti) in (("+", (1, 2)), ("−", (3, 4))):
        hits = sorted(((r[vi], r[0], r[ti]) for r in out if r[vi] >= 0.3), reverse=True)
        cls = {"baseball": [], "glucose": [], "both": [], "neither": []}
        for v, i, t in hits:
            text = tok.decode(docs[i].tolist())
            b, g = bool(BB.search(text)), bool(GL.search(text))
            cls["both" if b and g else "baseball" if b else "glucose" if g else "neither"].append((v, i, t))
        say(f"\npole {p}: {len(hits)} of {len(out)} documents reach 0.3: " + ", ".join(f"{k} {len(v)}" for k, v in cls.items()))
        for k in ("both", "neither"):
            for v, i, t in cls[k]:
                say(f"  [{k}] {v:.2f} d{i} | {context(tok, docs[i], t, after=4)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("lineage", "probe", "scan", "files", "routes", "reads", "sweep"))
    ap.add_argument("--docs", type=int, default=8000, help="sweep: random activation documents")
    args = ap.parse_args()
    globals()[args.part](args)
    stem = "" if MODEL == "Qwen/Qwen3-14B" else "_" + MODEL.split("/")[-1].lower()
    (SUMMARY / f"baseball_glucose_{args.part}{stem}.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
