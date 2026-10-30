"""Export R-lens fits (``02_fit.py --lens R``) to a Drowse home,
so Drowse and ``lens_compare.py`` (``R:<name>``) can load them.
Several fits (shards, corpora) are summed into one lens. ``--half`` exports only the even (0) or
odd (1) documents, so ``lens_compare.py`` can compare the halves neuron by neuron.

    python scripts/04_export_rlens.py data/fits/r-qwen3-8b-L36-dclm.pt --name dclm
    python scripts/04_export_rlens.py data/fits/r-qwen3-8b-L36-dclm.pt --name dclm-even --half 0
"""
import argparse
from pathlib import Path

from neuron_lens.lenses import load_fits, save_drowse_lens
from neuron_lens.paths import LENS_HOMES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fits", nargs="+", type=Path)
    ap.add_argument("--name", required=True, help="Drowse home data/lenses/<name>/")
    ap.add_argument("--half", type=int, choices=(0, 1), default=None, help="even (0) or odd (1) documents only")
    args = ap.parse_args()
    metas, acc = load_fits(args.fits, "R")
    path = save_drowse_lens(metas, acc, LENS_HOMES / args.name, half=args.half)
    n = sum(acc.n_docs) if args.half is None else acc.n_docs[args.half]
    print(f"{n} documents -> {path}")
    print("split-half by layer (‖even − odd‖ / ‖mean‖): "
          + " ".join(f"{x:.2f}" for x in acc.split_half("out")))


if __name__ == "__main__":
    main()
