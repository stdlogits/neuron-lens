"""OLMo 3 7B's stage-1 training batches, rebuilt from OLMo-core's recipe, and a check that a checkpoint saw them.

OLMo-core's official 7B scripts (``src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-{1,2}.py``) train on a
``NumpyFSLDataset`` over ``DataMix.OLMo_mix_0625_official``: its 1,016 token files (raw ``uint32`` arrays of the dolma3
tokenizer's ids, no header) cut in manifest order into windows of 8,192 tokens, each file's remainder dropped. The
loader shuffles the window indices once, ``np.random.Generator(PCG64(34521 + epoch))`` with epoch 1, and takes 512
windows a step (about 4M tokens). Steps count from one, so step N trains on the shuffled indices [512(N − 1), 512N) and
the checkpoint ``stage1-stepN`` has seen [0, 512N). Part 2 resumes at step 596,047 with the trainer's state and runs
to the epoch's end, so one order covers all of stage 1. The token files are served over HTTP with ranges: a step costs
its 512 windows (16 MB), not the files.

Parts (data in ``data/olmo3/``):

- ``index``: each file's size (a HEAD request), the windows' offsets and the shuffled order. The epoch should hold
  exactly the stage-1 steps the checkpoint list records (``LAST_STEP``), a check on the manifest and the sizes.
- ``batch <step>...``: each step's windows (token ids, ``[512, 8192]``) with each window's file (``array``) and position, saved;
  prints the first windows' sources and text.
- ``check --ckpts a,b --steps s,...``: the loss of each checkpoint on the first ``--windows`` windows of each step's
  batch, and per window the difference between the checkpoints. A batch the later checkpoint trained on in its last
  steps should drop by more than the batches neither has seen; a wrong order shows no difference. The same windows
  under both checkpoints cancel what the batches hold.

    python scripts/checks/olmo3_batches.py index
    python scripts/checks/olmo3_batches.py batch 100000
    python scripts/checks/olmo3_batches.py check --ckpts 100000,99000 --steps 100000,99999,99500,100001,101000
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from neuron_lens.paths import DATA

MODEL = "allenai/Olmo-3-1025-7B"
OLMO_CORE = "29ab5b92b94322dbd6e28e7d39af5068b89a23dc"  # the manifest is unchanged since its first commit (600d2fe)
MANIFEST = (f"https://raw.githubusercontent.com/allenai/OLMo-core/{OLMO_CORE}/"
            "src/olmo_core/data/mixes/OLMo-mix-0625-official.txt")
BASE, TOKENIZER = "https://olmo-data.org/", "allenai/dolma3-tokenizer"
AGENT = {"User-Agent": "neuron-lens"}  # the file server refuses urllib's own
SEQ, BATCH, SEED, EPOCH = 8192, 512, 34521, 1
ITEM = np.dtype(np.uint32).itemsize
LAST_STEP = 1_413_814  # the last stage-1 checkpoint, after 5,929,965,715,456 tokens (OLMo-3-1025-7B.csv)
OUT = DATA / "olmo3"


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _open(req: urllib.request.Request, tries: int = 6):
    for k in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=120)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def fetch(url: str, start: int | None = None, n: int | None = None) -> bytes:
    """A URL's body, or ``n`` bytes from ``start`` (refusing a server that ignores the range: these files run to 236 GB)."""
    headers = AGENT | ({"Range": f"bytes={start}-{start + n - 1}"} if start is not None else {})
    with _open(urllib.request.Request(url, headers=headers)) as r:
        if start is not None and r.status != 206:
            raise IOError(f"{url}: status {r.status} to a range request")
        data = r.read()
    if n is not None and len(data) != n:
        raise IOError(f"{url}: {len(data)} bytes of {n}")
    return data


def size(url: str) -> int:
    with _open(urllib.request.Request(url, method="HEAD", headers=AGENT)) as r:
        return int(r.headers["Content-Length"])


def files() -> dict:
    f = json.loads((OUT / "files.json").read_text())
    w = np.array(f["sizes"], dtype=np.int64) // (ITEM * SEQ)
    f["ends"] = np.cumsum(w)
    f["starts"] = f["ends"] - w
    return f


def index() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels, paths = zip(*(line.split(",", 1) for line in fetch(MANIFEST).decode().split()))
    urls = [BASE + p.replace("{TOKENIZER}", TOKENIZER) for p in paths]
    with ThreadPoolExecutor(32) as ex:
        sizes = list(ex.map(size, urls))
    assert all(s % ITEM == 0 for s in sizes), "a file's size isn't a whole number of tokens"
    (OUT / "files.json").write_text(json.dumps({"labels": labels, "urls": urls, "sizes": sizes}))
    n = int(files()["ends"][-1])
    say(f"{len(urls)} files, {sum(sizes) // ITEM:,} tokens, {n:,} windows: {n // BATCH:,} steps of {BATCH} "
        f"(stage 1 ends at step {LAST_STEP:,}: {'matches' if n // BATCH == LAST_STEP else 'MISMATCH'})")
    by = {}
    for lab, s in zip(labels, sizes):
        by[lab] = by.get(lab, 0) + s // ITEM
    for lab, t in sorted(by.items(), key=lambda x: -x[1])[:12]:
        say(f"  {lab:48s} {t / 1e9:8.1f}B tokens")
    t0 = time.time()
    order = np.arange(n, dtype=np.uint32)
    np.random.Generator(np.random.PCG64(seed=SEED + EPOCH)).shuffle(order)
    np.save(OUT / "order.npy", order)
    say(f"order shuffled in {time.time() - t0:.0f} s (numpy {np.__version__}); first indices {order[:4].tolist()}")


def batch(step: int) -> dict:
    """Step ``step``'s windows (one-based steps), fetched once and kept."""
    path = OUT / f"batch-{step}.npz"
    if path.exists():
        return dict(np.load(path))
    f = files()
    idx = np.load(OUT / "order.npy", mmap_mode="r")[BATCH * (step - 1): BATCH * step].astype(np.int64)
    array = np.searchsorted(f["ends"], idx, side="right")
    local = idx - f["starts"][array]
    with ThreadPoolExecutor(32) as ex:
        bufs = list(ex.map(lambda a: fetch(f["urls"][a[0]], int(a[1]) * SEQ * ITEM, SEQ * ITEM), zip(array, local)))
    b = {"index": idx, "array": array, "local": local, "tokens": np.stack([np.frombuffer(x, dtype=np.uint32) for x in bufs])}
    np.savez(path, **b)
    return b


def show(step: int, k: int = 3) -> None:
    from transformers import AutoTokenizer
    b, f = batch(step), files()
    tok = AutoTokenizer.from_pretrained(MODEL)
    say(f"step {step}: {len(b['index'])} windows, ids up to {int(b['tokens'].max()):,} (vocab {len(tok):,})")
    labels = [f["labels"][i] for i in b["array"]]
    top = sorted({lab: labels.count(lab) for lab in labels}.items(), key=lambda x: -x[1])[:8]
    say("  sources: " + ", ".join(f"{lab} {c}" for lab, c in top))
    for i in range(k):
        text = tok.decode(b["tokens"][i][:120]).replace("\n", " ")
        say(f"  [{labels[i]} window {int(b['local'][i])}] {text[:300]}")


@torch.no_grad()
def window_loss(model, x: torch.Tensor) -> torch.Tensor:
    """The mean next-token loss of each window in ``x`` (``[b, SEQ]``)."""
    h = model.model(input_ids=x).last_hidden_state
    out = []
    for i in range(len(x)):
        logits = model.lm_head(h[i, :-1]).float()
        out.append(torch.nn.functional.cross_entropy(logits, x[i, 1:]).cpu())
    return torch.stack(out)


def check(ckpts: list[int], steps: list[int], n: int, bs: int) -> None:
    from transformers import AutoModelForCausalLM
    data = {s: torch.from_numpy(batch(s)["tokens"][:n].astype(np.int64)) for s in steps}
    loss = {}
    for c in ckpts:
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=f"stage1-step{c}", dtype=torch.bfloat16,
                                                     attn_implementation="sdpa").cuda().eval()
        for s in steps:
            loss[c, s] = torch.cat([window_loss(model, data[s][i:i + bs].cuda()) for i in range(0, n, bs)])
            say(f"ckpt {c} batch {s}: loss {loss[c, s].mean():.4f}")
        del model
        torch.cuda.empty_cache()
        say(f"ckpt {c} in {time.time() - t0:.0f} s")
    print(f"\nmean loss over {n} windows per batch (rows: checkpoints; a batch at or before a checkpoint's step it has seen)")
    print("ckpt      " + "".join(f"{s:>12d}" for s in steps))
    for c in ckpts:
        print(f"{c:<10d}" + "".join(f"{loss[c, s].mean():>11.4f}{'*' if s <= c else ' '}" for s in steps))
    if len(ckpts) == 2:
        a, b = ckpts
        print(f"\nper window, ckpt {a} − ckpt {b}: mean ± s.e.")
        for s in steps:
            d = loss[a, s] - loss[b, s]
            seen = [c for c in ckpts if s <= c]
            print(f"  batch {s:>9d}: {d.mean():+.4f} ± {d.std() / len(d) ** 0.5:.4f}   "
                  f"seen by {', '.join(map(str, seen)) or 'neither'}"
                  + (f" ({max(ckpts) - s} steps before)" if seen else ""))
    torch.save({"ckpts": ckpts, "steps": steps, "loss": loss}, OUT / f"check-{'-'.join(map(str, ckpts))}.pt")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("part", choices=("index", "batch", "check"))
    ap.add_argument("step", nargs="*", type=int)
    ap.add_argument("--ckpts", default="100000,99000")
    ap.add_argument("--steps", default="100000,99999,99500,100001,101000")
    ap.add_argument("--windows", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()
    if args.part == "index":
        index()
    elif args.part == "batch":
        for s in args.step:
            show(s)
    else:
        check([int(c) for c in args.ckpts.split(",")], [int(s) for s in args.steps.split(",")], args.windows,
              args.batch_size)


if __name__ == "__main__":
    main()
