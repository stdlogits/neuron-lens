"""Figures of the catalog's units, lenses and checks, from ``data/summary``, the released lenses'
provenance and the catalog, into ``docs/figures/`` (PNG, light), with ``docs/figures/README.md``: each
figure's caption. Figures 1-17 are Qwen3-8B's (findings §1-§12; 1 and 2 are schematics, no data); 18
sets Qwen3-8B and Gemma 4 12B side by side, and 19-25 are Gemma 4 12B's: steering, the census, its
image and sound reads, the cross-modal check and its heads in media contexts (findings §13-§16), from
``data/summary`` and Gemma's catalog tables (``data/catalog/gemma-4-12b``: ``neurons/image_optimised``,
``neurons/audio_optimised``, ``heads/media`` and the ``read_png`` column of the image tables the gallery
draws). No image or sound from a dataset is drawn: only lens reads and optimised patterns.

    python scripts/11_figures.py              # every figure and the README
    python scripts/11_figures.py --only 01 03 # just these figures (the README is left as it is)
"""
import argparse
import io
import json
import math
import re

import matplotlib
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path

from neuron_lens.catalog import catalog_dir, head_table
from neuron_lens.paths import DATA, SUMMARY

MODEL = "Qwen/Qwen3-8B"
STEM = "qwen3-8b"
OUT = DATA.parent / "docs" / "figures"
RELEASE = DATA / "release" / STEM
GEMMA, GEMMA_STEM, GEMMA_LAYERS = "google/gemma-4-12B", "gemma-4-12b", 48  # figures 18-25
# The data-viz reference palette (validated, light mode): categorical slots in order, chrome and ink.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
R_COLOR, S_COLOR = SLOT[0], SLOT[2]  # the R-lens and the S-lens, in every figure
HEAD_COLOR, NEURON_COLOR = SLOT[6], SLOT[1]  # the anatomy schematic's two units
BANDS = [(0, 8), (9, 12), (13, 20), (21, 27), (28, 35)]  # the catalog's depth bands
MATH = {"mathtext.fontset": "custom", "mathtext.rm": "Helvetica Neue", "mathtext.it": "Helvetica Neue:italic",
        "mathtext.bf": "Helvetica Neue:bold"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Arial Unicode MS", "DejaVu Sans"],  # a list, so glyphs fall back
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.labelsize": 9, "text.color": INK,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.major.size": 0, "ytick.major.size": 0,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 8,
    "legend.labelcolor": INK2, "axes.axisbelow": True,
})
CAPTIONS = []


def summary(name: str):
    return json.loads((SUMMARY / name).read_text())


def figure(ncols: int = 1, width: float = 8.0, height: float | None = None, top: float | None = None,
           bottom: float | None = None):
    """One row of panels. Two-panel figures carry their legends below the panels."""
    height = height or (4.4 if ncols == 1 else 5.0)
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), dpi=100, squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.86 if ncols == 1 else 0.92, top=top or (0.78 if ncols == 1 else 0.75),
                        bottom=bottom or (0.17 if ncols == 1 else 0.31), wspace=0.32)
    return fig, list(axes[0])


def legend_above(ax, ncols: int) -> None:
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncols=ncols, borderaxespad=0, columnspacing=1.6)


def legend_below(ax, ncols: int = 2) -> None:
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncols=ncols, borderaxespad=0, columnspacing=1.2)


def log_ticks(ax, percent: bool = False) -> None:
    """Plain tick labels on a log axis (0.1%, 1%, 10% or 0.01, 0.1, 1), not powers of ten."""
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        (lambda v, _: f"{v * 100:g}%") if percent else (lambda v, _: f"{v:g}")))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def finish(fig, name: str, title: str, subtitle: str, source: str, caption: str) -> None:
    """Title and subtitle at the top left, the source at the bottom left; save and record the caption."""
    fig.text(0.01, 0.955, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.885, subtitle, fontsize=9, color=INK2, va="top", wrap=True)
    fig.text(0.01, 0.015, source, fontsize=7.5, color=MUTED, va="bottom")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    CAPTIONS.append((name, title, caption))
    print(f"{name}.png", flush=True)


def end_labels(ax, items: list[tuple[float, float, str]], gap_px: float = 11) -> None:
    """Direct labels at line ends, ``(x, y, text)``, nudged apart vertically; ink, not series color."""
    fig = ax.figure
    fig.canvas.draw()
    pts = [(ax.transData.transform((x, y))[1], x, y, t) for x, y, t in items]
    pts.sort()
    placed = []
    for py, x, y, t in pts:
        placed.append(max(py, placed[-1] + gap_px) if placed else py)
    shift = (sum(p for p, *_ in pts) - sum(placed)) / len(placed)  # keep the group centred
    for (py, x, y, t), p in zip(pts, placed):
        dy = (p + shift - py) * 72 / fig.dpi
        ax.annotate(t, (x, y), xytext=(7, dy), textcoords="offset points", va="center", ha="left",
                    fontsize=8, color=INK2, annotation_clip=False)


def series(ax, x, y, slot: int | None, label: str, end: bool = True, **kw):
    """One line: a categorical slot, or the muted reference (``slot=None``), with an end dot."""
    color = MUTED if slot is None else SLOT[slot]
    xs = [a for a, b in zip(x, y) if not math.isnan(b)]
    ys = [b for b in y if not math.isnan(b)]
    ax.plot(xs, ys, color=color, label=label, lw=1.2 if slot is None else 1.6, zorder=2 if slot is None else 3, **kw)
    if end:
        ax.plot(xs[-1:], ys[-1:], "o", color=color, ms=4.5, mec=SURFACE, mew=1.2, zorder=4)
    return xs[-1], ys[-1]


def layer_axis(ax, label: str = "layer", n: int = 36) -> None:
    """Layers 0 to ``n - 1``: ticks every 4 for Qwen3-8B's 36, every 6 for Gemma 4 12B's 48."""
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_xticks(range(0, n, 4 if n <= 36 else 6))
    ax.set_xlabel(label)


def pct(ax) -> None:
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def tint(color: str, t: float) -> tuple:
    """``color`` mixed into the surface: 0 is the surface, 1 the full color."""
    return tuple(s + (c - s) * t for c, s in zip(to_rgb(color), to_rgb(SURFACE)))


def ink_on(rgb) -> str:
    """Text color for a cell: ink on light fills, the surface on dark ones."""
    lum = sum(w * (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) for w, c in zip((0.2126, 0.7152, 0.0722), rgb))
    return SURFACE if lum < 0.3 else INK


def band_name(lo: int, hi: int) -> str:
    return f"L{lo}-L{hi}"


# --------------------------------------------------------------------------------------------- schematics


def schematic(w_in: float, h_in: float, top: float):
    """A blank canvas for a schematic: square data units, 100 across."""
    fig = plt.figure(figsize=(w_in, h_in), dpi=100)
    ax = fig.add_axes((0, 0.05, 1, top - 0.05))
    H = 100 * (top - 0.05) * h_in / w_in
    ax.set_xlim(0, 100)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.grid(False)
    return fig, ax


def box(ax, x, y, w, h, text="", size=9.0, ec=BASE, lw=0.9, weight="normal", fc=SURFACE, color=INK, z=3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, fontweight=weight, zorder=z + 1,
                linespacing=1.15)


def fig_anatomy() -> None:
    """A schematic, no data: one layer acting on the residual stream, an attention head and an MLP neuron."""
    with plt.rc_context(MATH):
        _anatomy()


def _anatomy() -> None:
    fig, ax = schematic(9.0, 7.6, 0.80)
    words = ["she", "heard", "the", "owl", "and"]
    xs = [11.0, 26.0, 41.0, 56.0, 71.0]
    tok_y, att, mlp = 28.8, (33.6, 46.4), (48.4, 57.8)  # token row; the attention and MLP bands (bottom, top)
    top_y = mlp[1] + 3.4

    # The layer: two sublayers in one box; the residual stream runs up through both at every position.
    box(ax, 50.0, (att[0] + mlp[1]) / 2, 97.0, mlp[1] - att[0] + 2.0, ec=BASE, lw=0.9, z=0)
    for (lo, hi), name in ((att, "attention"), (mlp, "MLP")):
        ax.add_patch(FancyBboxPatch((2.5, lo), 95.0, hi - lo, boxstyle="round,pad=0,rounding_size=0.7",
                                    fc="#f4f3ee", ec="none", zorder=0))
        ax.text(4.4, (lo + hi) / 2, name, rotation=90, ha="center", va="center", fontsize=8.5, color=INK2,
                fontweight="bold")
    ax.text(1.5, mlp[1] + 2.0, "one layer", ha="left", va="baseline", fontsize=9, color=INK, fontweight="bold")
    for x, w in zip(xs, words):
        ax.plot([x, x], [tok_y + 2.3, top_y], color=BASE, lw=3.2, zorder=1, solid_capstyle="butt")
        ax.add_patch(FancyArrowPatch((x, top_y - 0.2), (x, top_y + 1.4), arrowstyle="-|>", mutation_scale=11,
                                     color=BASE, lw=0, zorder=1))
        box(ax, x, tok_y, 8.6, 4.6, w, 9.5, z=3)
    ax.text(xs[-1] + 5.6, tok_y, "input tokens; above each,\nits residual stream", ha="left", va="center",
            fontsize=8, color=MUTED, style="italic", linespacing=1.2)
    ax.text(xs[-1] + 1.8, top_y + 0.5, "to the next layer", ha="left", va="center", fontsize=8, color=MUTED,
            style="italic")

    # The attention head at "and": QK sets how much it reads from each position (arc widths), OV what it moves.
    base, q = att[0] + 2.6, xs[-1]
    hy = base + 3.4
    for x, wt in zip(xs[:-1], [0.70, 0.04, 0.04, 0.10]):
        h = 0.13 * (q - x)
        path = Path([(x, base), (x, base + h * 1.33), (q - 6, hy + 0.2 * h), (q - 3.4, hy)],
                    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        strong = wt > 0.5
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>" if strong else "-", mutation_scale=13,
                                     color=HEAD_COLOR, lw=0.7 + 4.0 * wt, alpha=1.0 if strong else 0.45, zorder=2))
        ax.add_patch(Circle((x, base), 0.55 + 1.1 * wt ** 0.5, fc=HEAD_COLOR, ec=SURFACE, lw=0.8,
                            alpha=1.0 if strong else 0.55, zorder=3))
    box(ax, q, hy, 6.6, 3.4, "head", 8.5, ec=HEAD_COLOR, lw=1.4, color=HEAD_COLOR, weight="bold", z=4)
    ax.text((xs[0] + q) / 2 - 3, base + 7.6, "QK: which positions to read from (here, mostly “she”)",
            ha="center", va="bottom", fontsize=8.5, color=HEAD_COLOR, zorder=4,
            bbox={"boxstyle": "square,pad=0.1", "fc": "#f4f3ee", "ec": "none"})
    ax.text(q + 4.6, hy, "OV: what to copy from there\ninto this position (“she”)", ha="left", va="center",
            fontsize=8.5, color=HEAD_COLOR, linespacing=1.25)

    # The MLP neuron: the same neuron at every position, firing only at "owl".
    ny, nr = (mlp[0] + mlp[1]) / 2 + 0.9, 2.0
    for x, w in zip(xs, words):
        fire = w == "owl"
        cx = x + 5.2
        ax.add_patch(Circle((cx, ny), nr, fc=SURFACE, ec=NEURON_COLOR if fire else BASE, lw=1.6 if fire else 0.9,
                            zorder=3))
        ax.text(cx, ny, "n", ha="center", va="center", fontsize=9.5, style="italic", zorder=4,
                color=NEURON_COLOR if fire else MUTED, fontweight="bold" if fire else "normal")
        ax.text(cx, ny - nr - 0.8, "fires: +" if fire else "≈ 0", ha="center", va="top", fontsize=8 if fire else 7.5,
                color=NEURON_COLOR if fire else MUTED, fontweight="bold" if fire else "normal")
        if fire:
            ax.add_patch(FancyArrowPatch((x + 0.5, ny - 2.6), (cx - nr - 0.1, ny - 0.7), arrowstyle="-|>",
                                         mutation_scale=9, color=NEURON_COLOR, lw=1.2, zorder=2))
            ax.add_patch(FancyArrowPatch((cx - nr - 0.1, ny + 0.7), (x + 0.5, ny + 2.6), arrowstyle="-|>",
                                         mutation_scale=9, color=NEURON_COLOR, lw=2.2, zorder=2))
            ax.text(x - 0.9, ny + 2.4, "writes", ha="right", va="center", fontsize=8, color=NEURON_COLOR)
            ax.text(x - 0.9, ny - 2.4, "reads", ha="right", va="center", fontsize=8, color=NEURON_COLOR)

    # Below: each unit's mechanism, an example, and what the catalog records about it.
    ty, lh = 22.4, 2.3
    for x0, color, head, lines, unit, rows in (
            (1.5, HEAD_COLOR, "An attention head moves information between positions", [
                "Its QK circuit decides which earlier positions to read from; its OV circuit",
                "decides what to copy from there into the current position. The same head",
                "runs at every position.",
                "e.g. L20h17 copies she (她, herself) and pushes down he (himself, his)."],
             "head", (("fires", "top firings, activation statistics, attention patterns"),
                      ("attends to", "QK read: which tokens it looks at"),
                      ("copies", "OV read: what it writes for each token it reads"))),
            (51.5, NEURON_COLOR, "An MLP neuron works within one position", [
                r"It reads the stream along one direction; its activation, $\mathrm{silu}(gate) \cdot up$,",
                r"can be positive or negative (the + and − poles). It adds its write $W_{\rm down}[:, n]$,",
                "scaled by that activation, back to the stream.",
                "e.g. L8 n344 + fires on owl and writes owl, noct(urnal), bedtime, night."],
             "pole", (("fires", "top contexts and activation statistics over 15M positions"),
                      ("reads", "what drives it: lens reads of its read direction; exact drivers"),
                      ("writes", "what it pushes toward: lens reads of its write")))):
        ax.text(x0, ty, head, ha="left", va="baseline", fontsize=9.5, color=color, fontweight="bold")
        for i, line in enumerate(lines):
            ax.text(x0, ty - 2.9 - i * lh, line, ha="left", va="baseline", fontsize=8.5, color=INK2)
        cy = ty - 2.9 - len(lines) * lh - 1.6
        ax.text(x0, cy, f"The catalog records, for each {unit}:", ha="left", va="baseline", fontsize=8.5, color=INK,
                fontweight="bold")
        for i, (side, how) in enumerate(rows):
            y = cy - (i + 1) * lh
            ax.text(x0, y, side, ha="left", va="baseline", fontsize=8.5, color=INK)
            ax.text(x0 + 8.0, y, how, ha="left", va="baseline", fontsize=8.5, color=INK2)

    finish(fig, "01_anatomy", "What attention heads and MLP neurons do",
           "A transformer keeps one vector per token, the residual stream. Each of Qwen3-8B's 36 layers reads it and "
           "adds to it twice: its 32 attention heads move information between positions, then its 12,288 MLP neurons "
           "work on each position on its own.",
           "schematic; the examples are catalog entries (findings §6, §11)",
           "One layer acting on five positions of \"she heard the owl and\". At \"and\", an attention head's QK "
           "circuit weighs the earlier positions (arc widths) and its OV circuit copies what it reads from \"she\" "
           "into the current position. In the MLP, the same neuron sits at every position; it reads the stream "
           "along one direction, its activation `silu(gate)·up` takes either sign (the `+` and `−` poles), and it "
           "adds its write `W_down[:, n]`, scaled by that activation, back to the stream. Here it fires only at "
           "\"owl\" (L8 n344 +, which writes owl, nocturnal and bedtime words). Below each unit: what the catalog "
           "records for every head and every neuron pole.")


def fig_lenses() -> None:
    """A schematic, no data: the residual stream, one neuron at layer L, and each lens as a path."""
    with plt.rc_context(MATH):
        _lenses()


def _lenses() -> None:
    fig, ax = schematic(9.0, 6.95, 0.80)
    R_, J_, S_, LOGIT, K_ = R_COLOR, SLOT[1], S_COLOR, SLOT[3], SLOT[4]

    def route(points, color, lw=1.8, ls="-", head=True, radius=1.6):
        """A polyline with rounded corners and an arrowhead at its end."""
        def toward(a, b):
            d = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
            return a[0] + radius * (b[0] - a[0]) / d, a[1] + radius * (b[1] - a[1]) / d
        verts, codes = [points[0]], [Path.MOVETO]
        for p, v, n in zip(points, points[1:], points[2:]):
            verts += [toward(v, p), v, toward(v, n)]
            codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
        verts.append(points[-1])
        codes.append(Path.LINETO)
        ax.add_patch(FancyArrowPatch(path=Path(verts, codes), arrowstyle="-|>" if head else "-", mutation_scale=11,
                                     color=color, lw=lw, ls=ls, zorder=2, capstyle="round", joinstyle="round"))

    def label(x, y, name, gloss, ha="left"):
        """A path's name (bold) over a short gloss, the gloss's baseline at ``y``."""
        ax.text(x, y, gloss, ha=ha, va="baseline", fontsize=8.5, color=INK2)
        ax.text(x, y + 2.1, name, ha=ha, va="baseline", fontsize=9.5, color=INK, fontweight="bold")

    # The residual stream, left to right.
    S = 31.45
    xs = {"tok": 5.0, "WE": 14.5, "L0": 22.0, "Lm": 32.0, "L": 40.0, "Lp": 48.0, "L35": 60.0, "norm": 69.0,
          "WU": 77.5, "out": 89.5}
    ax.plot([9, 84], [S, S], color=BASE, lw=3.2, zorder=1, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch((82, S), (84.6, S), arrowstyle="-|>", mutation_scale=12, color=BASE, lw=0, zorder=1))
    box(ax, xs["tok"], S, 8.0, 8.0, "input\ntokens", 8.5)
    box(ax, xs["WE"], S, 5.5, 8.0, "$W_E$", 10)
    for k, t in (("L0", "L0"), ("Lm", "L−1"), ("Lp", "L+1"), ("L35", "L35")):
        box(ax, xs[k], S, 5.0, 8.0, t, 8.5)
    for x in ((xs["L0"] + xs["Lm"]) / 2, (xs["Lp"] + xs["L35"]) / 2):
        ax.text(x, S, "···", ha="center", va="center", fontsize=10, color=INK2, zorder=4,
                bbox={"boxstyle": "square,pad=0.15", "fc": SURFACE, "ec": "none"})
    box(ax, xs["L"], S, 8.0, 10.0, "layer L", 9, ec=INK, lw=1.3, weight="bold")
    box(ax, xs["norm"], S, 6.0, 8.0, "final\nnorm", 8)
    box(ax, xs["WU"], S, 5.5, 8.0, "$W_U$", 10)
    box(ax, xs["out"], S, 11.0, 8.0, "next-token\nlogits", 8.5)
    ax.text(9.2, S - 5.6, "residual stream", ha="left", va="top", fontsize=8, color=MUTED, style="italic")

    # The lenses above the stream: K_L summarizes the layers before L, J_L the layers after it.
    lane, top_y, bh = 43.45, 53.45, 5.0
    route([(xs["WE"] - 0.65, S + 4.2), (xs["WE"] - 0.65, lane + 0.7), (38.3, lane + 0.7), (38.3, S + 5.2)], S_, lw=1.6)
    route([(xs["WE"] + 0.65, S + 4.2), (xs["WE"] + 0.65, lane - 0.7), (37.0, lane - 0.7), (37.0, S + 5.2)], K_, lw=1.6)
    route([(40.0, S + 5.2), (40.0, top_y), (71.0, top_y), (71.0, S + 4.2)], LOGIT, ls=(0, (4, 2.2)))
    route([(42.0, S + 5.2), (42.0, lane + 0.7), (68.6, lane + 0.7), (68.6, S + 4.2)], R_, lw=1.6)
    route([(43.3, S + 5.2), (43.3, lane - 0.7), (67.3, lane - 0.7), (67.3, S + 4.2)], J_, lw=1.6)
    box(ax, 27.0, lane, 15.0, bh, "$K_L$", 11)
    box(ax, 54.0, lane, 17.0, bh, "$J_L$", 11)
    ax.text(27.0, lane - bh / 2 - 0.9, "mean Jacobian: embeddings → L", ha="center", va="top", fontsize=8,
            color=INK2)
    ax.text(54.0, lane - bh / 2 - 0.9, "mean Jacobian: L → final residual", ha="center", va="top", fontsize=8,
            color=INK2)
    label(15.8, lane + bh / 2 + 1.6, "K-lens, S-lens", "which input tokens put it there")
    label(45.5, lane + bh / 2 + 1.6, "J-lens, R-lens", "what it makes the model say")
    label(45.5, top_y + 1.6, "logit lens", "as if the later layers did nothing")

    # One neuron at layer L, below the stream: reads the MLP input, writes back.
    ny, nr = 19.45, 2.3
    ax.add_patch(Circle((xs["L"], ny), nr, fc=SURFACE, ec=INK, lw=1.3, zorder=3))
    ax.text(xs["L"], ny, "n", ha="center", va="center", fontsize=10, color=INK, style="italic", zorder=4)
    ax.add_patch(FancyArrowPatch((37.6, S - 5.1), (38.6, ny + 1.9), arrowstyle="-|>", mutation_scale=10,
                                 color=INK2, lw=1.3, zorder=2))
    ax.add_patch(FancyArrowPatch((41.4, ny + 1.9), (42.4, S - 5.1), arrowstyle="-|>", mutation_scale=10,
                                 color=INK2, lw=1.3, zorder=2))
    ax.text(36.8, ny + 2.6, r"reads the MLP input along $\hat{a} \pm \hat{b}$", ha="right", va="baseline", fontsize=8.5,
            color=INK)
    ax.text(36.8, ny + 0.4, "gate and up rows; + and − poles", ha="right", va="baseline", fontsize=8, color=INK2)
    ax.text(43.2, ny + 2.6, r"writes $W_{\rm down}[:, n]$", ha="left", va="baseline", fontsize=8.5, color=INK)
    ax.text(43.2, ny + 0.4, "back into the stream", ha="left", va="baseline", fontsize=8, color=INK2)

    # Two keys along the bottom: the lenses' formulas, and the neuron's three sides.
    ky, dy = 12.65, 2.45
    ax.text(1.0, ky + 1.0, "Lens: read of a direction h at layer L", ha="left", va="baseline", fontsize=8.5,
            color=INK, fontweight="bold")
    for i, (color, name, formula, note, ls) in enumerate((
            (LOGIT, "logit", r"$W_U\, \mathrm{norm}(h)$", r"no Jacobian ($J_L = I$)", (0, (4, 2.2))),
            (J_, "J-lens", r"$W_U\, \mathrm{norm}(J_L\, h)$", "true gradients (Gurnee et al. 2026)", "-"),
            (R_, "R-lens", r"$W_U\, \mathrm{norm}(J_L\, h)$", "RelP rules", "-"),
            (K_, "K-lens", r"$W_E\, K_L^{\top} h$", "true gradients, from the input up (Vogel 2026)", "-"),
            (S_, "S-lens", r"$W_E\, K_L^{\top} h$", "RelP rules, from the input up", "-"))):
        y = ky - (i + 1) * dy
        ax.plot([1.0, 4.0], [y + 0.55] * 2, color=color, lw=1.8, ls=ls, solid_capstyle="round")
        ax.text(5.0, y, name, ha="left", va="baseline", fontsize=8.5, color=INK)
        ax.text(10.5, y, formula, ha="left", va="baseline", fontsize=8.5, color=INK)
        ax.text(22.5, y, note, ha="left", va="baseline", fontsize=8.5, color=INK2)
    ax.text(56.0, ky + 1.0, "One pole of a neuron, three sides", ha="left", va="baseline", fontsize=8.5,
            color=INK, fontweight="bold")
    for i, (side, how) in enumerate((("writes", r"R-lens and S-lens of $W_{\rm down}[:, n]$"),
                                     ("reads", r"R-lens (from the MLP input) and S-lens of $\hat{a} \pm \hat{b}$"),
                                     ("fires", "activations over 15M positions; exact token drivers"))):
        y = ky - (i + 1) * dy
        ax.text(56.0, y, side, ha="left", va="baseline", fontsize=8.5, color=INK)
        ax.text(62.5, y, how, ha="left", va="baseline", fontsize=8.5, color=INK2)

    finish(fig, "02_lenses", "The lenses: reading a neuron forward to the output and back to the input",
           "Each lens maps a direction h at layer L (a neuron's write or read) linearly onto tokens. The R-lens and "
           "S-lens, fitted here, average their Jacobians under RelP rules (norm scales frozen, SwiGLU gate linearised, "
           "half rule on the product) instead of true gradients.",
           "schematic; docs/design.md (Estimators); findings §1, §3, §6",
           "The residual stream runs from the input tokens through the embedding `W_E`, layers L0-L35, the final "
           "norm and the unembedding `W_U` to next-token logits. A neuron at layer L reads the MLP input along "
           "`â ± b̂` (one direction per pole) and writes `W_down[:, n]` back. The logit lens reads a direction `h` at "
           "layer L as `W_U · norm(h)`; the J-lens and R-lens first carry it through `J_L`, the corpus-mean Jacobian "
           "of the final residual with respect to layer L (true gradients and RelP rules respectively); the K-lens and "
           "S-lens take the Jacobian from the input embeddings up to layer L instead (`K_L`, true gradients and RelP "
           "rules) and read with `W_E`.")


# --------------------------------------------------------------------------------------------- data


def fig_depth_map() -> None:
    """Every legibility share by layer, one row per side, as an annotated heatmap."""
    rows = summary(f"legible_{STEM}.json")["neurons"]["layers"]
    steer = [r["steer_predicted"] for r in summary(f"legibility_census_{STEM}.json")["layers"]]
    col = lambda k: [r[k] for r in rows]
    groups = [
        ("What makes a pole fire (its read direction)", [
            ("read_S", "its S-lens read names the firing token", col("read_S"), S_COLOR),
            ("read_R", "its R-lens read names the firing token", col("read_R"), R_COLOR),
            ("detector", "a third of its top firings are on one token", col("token_detector"), SLOT[3])]),
        ("What a pole does (its write)", [
            ("write_S", "its S-lens read names the firing token", col("write_S"), S_COLOR),
            ("write_R", "its R-lens read predicts the next token, in context", col("write_R"), R_COLOR),
            ("steering", "its R-lens read predicts its effect, amplified", steer, R_COLOR)]),
        ("Neither", [
            ("dark", "fires, but no test or detector passes", col("dark"), INK2),
            ("silent", "barely fires (act_top under 0.05)", col("silent"), MUTED)]),
    ]
    n_rows = sum(len(g) for _, g in groups)
    fig = plt.figure(figsize=(10.0, 6.6), dpi=100)
    ax = fig.add_axes((0.265, 0.1, 0.725, 0.69))
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    gap, cell = 0.95, 1.0
    y, ticks = 0.0, []
    for gi, (gname, items) in enumerate(groups):
        if gi:
            y += gap
        ax.text(-0.35, y + 0.1, gname, transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=9,
                fontweight="bold", color=INK)
        y += 0.22
        for key, gloss, vals, color in items:
            for layer, v in enumerate(vals):
                t = min(1.0, max(v, 0.0)) ** 0.5
                fill = tint(color, 0.06 + 0.94 * t)
                ax.add_patch(Rectangle((layer - 0.5, y), 1, cell, fc=fill, ec=SURFACE, lw=1.0))
                if v >= 0.005:
                    ax.text(layer, y + cell / 2, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.6,
                            color=ink_on(fill))
            ax.text(-0.35, y + cell * 0.36, key, transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=8.5, color=INK, fontweight="bold")
            ax.text(-0.35, y + cell * 0.36 + 0.42, gloss, transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=7.5, color=INK2)
            ticks.append(y + cell / 2)
            y += cell
    ax.set_ylim(y + 0.1, -0.75)
    ax.set_xlim(-0.6, 35.6)
    ax.set_yticks([])
    ax.set_xticks(range(36), [str(i) if i % 2 == 0 else "" for i in range(36)])
    ax.tick_params(axis="x", labelsize=7.5, pad=2)
    ax.set_xlabel("layer")
    for lo, hi in BANDS:  # the depth bands, along the top, with a rule between them
        ax.text((lo + hi) / 2, -0.38, band_name(lo, hi), ha="center", va="bottom", fontsize=8, color=INK2)
        if lo:
            ax.plot([lo - 0.5] * 2, [-0.6, y + 0.05], color=INK2, lw=0.9, zorder=5)
    assert n_rows == len(ticks)
    finish(fig, "03_depth_map", "Each side of a neuron reads at its own depth: inputs early, outputs late, a dark middle",
           "Share of each layer's poles (%) whose weight-only read passes its test at the pole's top firings (5% false "
           "discovery rate), except steering: of 64 random poles a layer, the share whose write predicts the effect of "
           "adding it at every position (amplified, not one firing in context). Blank: under 0.5%.",
           "findings §3, §4, §6, §7; data/summary/legible_qwen3-8b.json, legibility_census_qwen3-8b.json (steering)",
           "The input side reads early: the S-lens names the firing token for 78% of L0's poles and 14-58% at L1-L8, "
           "and almost none at L10-L20. The write side reads late: the R-lens write predicts the next token after a "
           "firing for at most 2% of poles before L23, rising to 35-37% at L28-L30. In between, 77-86% of L13-L20's "
           "poles are dark. Amplified, the write side reads nearly everywhere: from L5, 92-100% of random poles' "
           "writes predict the effect of steering them.")


def fig_lens_reliability() -> None:
    fig, (ax,) = figure(top=0.68)
    for i, (f, label) in enumerate((("r-lens/lens.pt", "R-lens from each layer's output: reads writes"),
                                    ("r-lens/mid.pt", "R-lens from each MLP input: reads read directions"),
                                    ("s-lens/mid.pt", "S-lens to each MLP input: reads read directions"),
                                    ("s-lens/out.pt", "S-lens to each layer's output: reads writes"))):
        split = torch.load(RELEASE / f, weights_only=True, mmap=True)["provenance"]["split_half"]
        series(ax, range(len(split)), split, i, label)
    ax.set_ylim(0, 0.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    layer_axis(ax)
    ax.set_ylabel("difference between the two halves' lenses")
    legend_above(ax, 2)
    finish(fig, "04_lens_reliability", "The lenses are stable: fits on two halves of the corpus differ by under 6%",
           "Relative difference between the lens fitted on the even and the odd half of the corpus (500 DCLM documents "
           "each), by layer, for each of the four lens targets. Lower is more reliable.",
           "findings §1, §6; the split-half provenance of the released lenses (stdlogits/rs-lenses)",
           "Split-half difference of each lens target by layer: the R-lens from each layer's output (reads MLP writes) "
           "and from each MLP input (reads read directions), and the S-lens to each MLP input and to each layer's "
           "output. All stay under 6%; the R-lens is least stable around L7, the S-lens from L16-L17.")


def fig_chat_lens() -> None:
    j = summary(f"causal_lens_chat-vs-dclm_{STEM}.json")["targets"]["steer 1x"]
    fig, (ax,) = figure()
    for spec, label, slot in (("R:dclm-even", "web-text (DCLM) lens, even half", 0),
                              ("R:dclm-odd", "web-text (DCLM) lens, odd half", 2),
                              ("R:chat", "chat lens (500 conversations)", 1)):
        series(ax, j[spec]["layer"], j[spec]["r_med"], slot, label)
    layer_axis(ax)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("median r, lens read vs steering effect")
    ax.legend(loc="lower right")
    finish(fig, "05_chat_lens", "An R-lens fitted on chat predicts steering as well as the web-text lens",
           "How well each R-lens's read of a neuron's write predicts the measured effect of steering it (as in figure "
           "7), for a lens fitted on 500 chat conversations and the two halves of the DCLM lens of record (which "
           "overlap almost exactly).", "findings §1; data/summary/causal_lens_chat-vs-dclm_qwen3-8b.json",
           "Median r with 1× steering for the chat-fitted lens and the DCLM lens's two halves: within 0.04 at every "
           "layer (the chat lens is lowest against them at L12-L18), though its top tokens differ (findings §1).")


def fig_activations() -> None:
    rows = []
    for line in (SUMMARY / f"act_summary_{STEM}.log").read_text().splitlines():
        m = re.match(r"^L(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            rows.append([float(v) for v in m.groups()])
    x = [int(r[0]) for r in rows]
    fig, (a, b) = figure(2, height=4.6, bottom=0.27)
    series(a, x, [r[1] for r in rows], 0, "median act_top")
    a.set_yscale("log")
    log_ticks(a)
    a.set_title("Typical top activation (act_top)")
    a.set_ylabel("median over the layer's poles")
    series(b, x, [r[5] for r in rows], 1, "quiet: act_top under 0.05")
    series(b, x, [r[6] for r in rows], 6, "fires harder at the first position than anywhere else")
    b.set_title("Early layers' odd populations")
    b.set_ylim(0, 1)
    b.set_ylabel("share of the layer's poles")
    pct(b)
    for ax in (a, b):
        layer_axis(ax)
    legend_below(b, 1)
    finish(fig, "06_activations", "Early poles are quiet or work at the first token; from L5 every pole fires",
           "act_top: the median of a pole's 24 top-document maxima over 15M positions of web, Chinese, code and chat "
           "text. The first position is measured separately, every common token alone.",
           "findings §2; data/summary/act_summary_qwen3-8b.log",
           "Left: median act_top by layer (log scale), flat at 3-5 from L5 to L20 and rising tenfold to 51 at L35. "
           "Right: 56-61% of L1-L2's poles are quiet (42% at L3), and 57-70% of L2-L3's fire harder at the first "
           "position than anywhere in the corpus; from L5 nearly every pole fires.")


def steering_panels(stem: str, names: dict, n: int = 36):
    """Median r of each lens's write read against the measured effect of steering (left) and of ablation
    where the pole fires (right), by layer; the first of ``names`` is the lens of record, with its
    interquartile band."""
    j = summary(f"causal_lens_{stem}.json")["targets"]
    record = next(iter(names))
    fig, (a, b) = figure(2)
    for ax, target, title in ((a, "steer 1x", "Steering: the write added at 1× its scale"),
                              (b, "ablate win", "Ablation where it fires (bf16)")):
        t = j[target][record]
        ax.fill_between(t["layer"], t["r_q25"], t["r_q75"], color=R_COLOR, alpha=0.12, lw=0, zorder=1)
        for spec, (label, slot) in names.items():
            series(ax, j[target][spec]["layer"], j[target][spec]["r_med"], slot, label)
        series(ax, t["layer"], t["control_med"], None, "control: another neuron's read", end=False)
        layer_axis(ax, n=n)
        ax.set_ylim(-0.05, 1.0)
        ax.set_title(title)
    a.set_ylabel("median r, lens read vs measured effect")
    legend_below(a)
    return fig, j


def fig_steering() -> None:
    fig, _ = steering_panels(STEM, {"R:dclm": ("R-lens (of record)", 0), "J": ("J-lens (Neuronpedia)", 1),
                                    "logit": ("logit lens", 3)})
    finish(fig, "07_steering", "A neuron's write, read through the R-lens, predicts what steering it does",
           "Correlation over the vocabulary between a lens's read of a pole's write and the measured change in "
           "log-probabilities; 32 random neurons a layer, both poles. Band: the R-lens's interquartile range.",
           "findings §3, §4; data/summary/causal_lens_qwen3-8b.json",
           "Median r between each lens's write read and the measured effect, by layer. Left, steering at 1× the "
           "pole's typical activation: the R-lens reaches 0.46-0.51 at L4-L8 and 0.9 or more from L22, leads the "
           "J-lens by up to 0.12 from L4 to L23, and the logit lens trails until about L20 (L8: 0.51 against 0.12). "
           "Right, ablating the pole where it fires: far lower before L21, partly because this ran in bf16 and partly "
           "because a firing's effect depends on its context (figure 9).")


def fig_amplified() -> None:
    steer = summary(f"legibility_census_{STEM}.json")["layers"]
    rows = summary(f"legible_{STEM}.json")["neurons"]["layers"]
    fig, (ax,) = figure()
    x = [r["layer"] for r in rows]
    items = [series(ax, x, [r["steer_predicted"] for r in steer], 0, "write predicts its steering effect (amplified)")
             + ("amplified",),
             series(ax, x, [r["write_R_net"] for r in rows], 1, "write predicts the next token (when it fires)")
             + ("in context",)]
    series(ax, x, [r["steer_null"] for r in steer], None, "chance for steering (mismatched pairs)", end=False)
    layer_axis(ax)
    ax.set_ylim(-0.02, 1.02)
    end_labels(ax, items)
    pct(ax)
    ax.set_ylabel("share of poles")
    ax.legend(loc="center right")
    finish(fig, "08_amplified_vs_in_context", "Amplified, nearly every write does what the lens says; in context, few do",
           "Blue: of 64 random poles a layer, the share whose R-lens write read predicts their own 1× steering effect. "
           "Orange: of all poles, the share whose write promotes the token after their top firings, net of chance.",
           "findings §4; data/summary/legibility_census_qwen3-8b.json (steering), legible_qwen3-8b.json (in context)",
           "From L5, 92-100% of random poles' writes predict their steering effect; before L22, at most 4% of poles "
           "predict what follows their own firings, rising to 31-32% at L28-L29. The gap is the difference between a "
           "write's average push and a firing's effect in its context (figure 9).")


def fig_context_linearity() -> None:
    j = summary(f"context_linearity_{STEM}.json")["layers"]
    x = [int(k) for k in j]
    col = lambda k: [j[str(l)][k] for l in x]
    fig, (a, b) = figure(2)
    for k, label, slot in (("nonlin win", "ablation vs its linear estimate", 0),
                           ("split lin", "12 firings vs the other 12", 2),
                           ("ctx LOO win", "one firing vs the other 23", 1)):
        series(a, x, col(k), slot, label)
    a.set_title("What one firing does")
    for k, label, slot in (("steer~lens", "steering vs lens", 0), ("lens/mean win", "lens vs mean of 24 firings", 2),
                           ("steer~lin", "steering vs mean of 24 firings", 3), ("lens/ctx win", "lens vs one firing", 1)):
        series(b, x, col(k), slot, label)
    b.set_title("What the lens and steering capture")
    for ax in (a, b):
        layer_axis(ax)
        ax.set_ylim(-0.05, 1.05)
        legend_below(ax)
    a.set_ylabel("median r over the vocabulary")
    finish(fig, "09_context_linearity", "A firing's effect is linear but depends on context; the lens reads its average",
           "64 random poles a layer (figure 7's sample) at their 24 top firings, in fp32: the effect of ablating each "
           "firing, and its linear estimate (central difference), summed over 9 positions; medians by layer.",
           "findings §4; data/summary/context_linearity_qwen3-8b.json (H100, fp32 with TF32)",
           "Left: ablation matches the linearized effect (r 0.78-1.00 by layer) except at L1-L2, whose poles barely "
           "fire and sit at the precision floor; before L20 one firing's effect barely predicts the others' (r "
           "0.02-0.14). Right: the lens predicts steering (both average over contexts) far better than either "
           "predicts a single firing.")


def fig_side_agreement() -> None:
    r = summary(f"side_agreement_{STEM}.json")["reads"]
    fig, (a, b) = figure(2)
    for k, label, slot in (("write_R_dclm", "write promotes the next token", 0),
                           ("suppress_R_dclm", "write suppresses it", 1)):
        series(a, r[k]["layer"], r[k]["mean"], slot, label)
    series(a, r["write_R_dclm"]["layer"], r["write_R_dclm"]["control_mean"], None, "control: a random pole's read",
           end=False)
    a.set_title("Write side: the token after each top firing")
    for k, label, slot in (("read_S", "S-lens read", 2), ("read_S_z", "S-lens, contrasted within the layer", 3),
                           ("read_R_dclm", "R-lens read (from the MLP input)", 0)):
        series(b, r[k]["layer"], [max(v, 1e-3) for v in r[k]["mean"]], slot, label)
    series(b, r["read_S"]["layer"], [max(v, 1e-3) for v in r["read_S"]["control_mean"]], None, "control", end=False)
    b.set_yscale("log")
    b.set_ylim(1e-3, 1)
    b.set_title("Read side: the token each top firing is on")
    for ax in (a, b):
        layer_axis(ax)
        pct(ax)
        legend_below(ax)
    log_ticks(b, percent=True)
    a.set_ylabel("share of top firings in the read's top 10")
    finish(fig, "10_side_agreement", "Writes predict the next token late; reads name the firing token early",
           "Mean share of each pole's 24 top firings whose token is in its read's top 10, all 884,736 poles; "
           "control: a random pole's read. Right panel on a log scale, floored at 0.1%.",
           "findings §5, §6; data/summary/side_agreement_qwen3-8b.json",
           "Left: prediction rises from L20 and peaks at 7.3% at L29-L30; suppression overtakes it at L35 (early "
           "layers' writes match the control: hub tokens). Right: the S-lens names the firing token at L0-L8 (40% at "
           "L0), with a spike at L8; the R-lens read from the MLP input takes over from L9.")


def fig_drivers() -> None:
    r = summary(f"side_agreement_{STEM}.json")["reads"]
    fig, (ax,) = figure(top=0.72)
    for k, label, slot in (("drivers_S", "S-lens", 2), ("drivers_S_z", "S-lens, contrasted within the layer", 3),
                           ("drivers_R_dclm", "R-lens from the MLP input", 0)):
        series(ax, r[k]["layer"], [max(v, 1e-3) for v in r[k]["mean"]], slot, label)
    series(ax, r["drivers_S"]["layer"], [max(v, 1e-3) for v in r["drivers_S"]["control_mean"]], None,
           "control: a random pole's read", end=False)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 0.6)
    for x0, x1 in ((-0.8, 8.5), (12.5, 20.5)):
        ax.axvspan(x0, x1, color=GRID, alpha=0.35, lw=0, zorder=0)
    ax.text(3.85, 0.45, "S-lens works", ha="center", fontsize=8, color=INK2)
    ax.text(16.5, 0.45, "every read at its weakest", ha="center", fontsize=8, color=INK2)
    layer_axis(ax)
    log_ticks(ax, percent=True)
    ax.set_ylabel("share of the read's top 10 that are exact drivers")
    legend_above(ax, 4)
    finish(fig, "11_drivers", "The S-lens names a neuron's real input tokens early, the R-lens from L8",
           "Every common token placed in 8 real documents; a pole's exact drivers are the 20 that raise it most. "
           "Share of each weight-only read's top 10 among them; log scale, floored at 0.1%.",
           "findings §6; data/summary/side_agreement_qwen3-8b.json; neurons/drivers",
           "At L0, 31% of the S-lens read's top 10 are exact drivers, and 4-11% at L1-L4; the R-lens read of the input "
           "direction leads from L8 (5-8% at L8-L12 and L22-L29). L13-L20 is the weakest stretch for every "
           "weight-only input read (R-lens 2-4%, S-lens under 1%).")


def fig_census() -> None:
    rows = summary(f"legible_{STEM}.json")["neurons"]["layers"]
    fig, (ax,) = figure()
    x = [r["layer"] for r in rows]
    weights = [r["any_test"] for r in rows]
    silent = [r["silent"] for r in rows]
    dark = [r["dark"] for r in rows]
    detector = [max(0.0, 1 - w - s - d) for w, s, d in zip(weights, silent, dark)]
    bottom = [0.0] * len(x)
    for vals, label, color in ((weights, "a weight-only read passes", R_COLOR),
                               (detector, "only a token detector:\na third of firings on one token", SLOT[3]),
                               (dark, "dark: fires, nothing passes", BASE), (silent, "silent", GRID)):
        ax.bar(x, vals, bottom=bottom, width=0.86, color=color, label=label, edgecolor=SURFACE, linewidth=0.8)
        bottom = [b + v for b, v in zip(bottom, vals)]
    layer_axis(ax)
    ax.set_ylim(0, 1)
    pct(ax)
    ax.set_ylabel("share of the layer's poles")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), labelspacing=0.9)
    ax.grid(False)
    band = summary(f"legible_{STEM}.json")["neurons"]["bands"]
    finish(fig, "12_census", "Most mid-network poles are dark: they fire, but no token-level test reads them",
           "Every pole's weight-only reads (write: next token or firing token; read: firing token; R-lens and S-lens) "
           "tested at its top firings against other poles of the layer, at a 5% false discovery rate.",
           "findings §7; data/summary/legible_qwen3-8b.json; neurons/legible",
           f"Shares of each layer's poles: some weight-only read passes, only a token detector, dark (fires but "
           f"nothing passes), or silent. Over all layers {100 * band['L0-35']['any_test']:.0f}% of poles have a "
           f"passing read and {100 * band['L0-35']['dark']:.0f}% are dark; dark averages "
           f"{100 * band['L13-20']['dark']:.0f}% of L13-L20's poles.")


def fig_rank_depth() -> None:
    text = (SUMMARY / f"rank_depth_{STEM}.log").read_text().splitlines()
    rows = {}
    for line in text[1:]:
        m = re.match(r"^(L\d+ .+?)\s+(r1-8 .*)$", line)
        if m:
            rows[m.group(1).strip()] = [(int(a), int(b), float(c), float(d)) for a, b, c, d in
                                        re.findall(r"r(\d+)-(\d+) ([\d.]+)/([\d.]+)", m.group(2))]
    pick = [("L28 write R  -> next", "L28 write, R-lens → next token", 0),
            ("L16 write Rz -> next", "L16 write, R-lens contrasted within the layer → next token", 1),
            ("L12 read fwd -> peak", "L12 read, R-lens → firing token", 2),
            ("L4 read S   -> peak", "L4 read, S-lens → firing token", 3)]
    fig, (ax,) = figure(top=0.68)
    items = []
    for key, label, slot in pick:
        key = next(k for k in rows if k.replace(" ", "") == key.replace(" ", ""))
        mid = [(a + b) / 2 for a, b, *_ in rows[key]]
        ratio = [m / max(c, 1e-5) for *_, m, c in rows[key]]
        items.append(series(ax, mid, ratio, slot, label, marker="o", ms=3.5, mec=SURFACE) + (label.split(",")[0],))
    ax.axhline(1, color=MUTED, lw=1.0, zorder=1)
    ax.annotate("chance (a mismatched pole)", (64, 1), xytext=(-4, 4), textcoords="offset points", ha="right",
                fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_xlim(0, 66)
    ax.set_xticks([4.5, 12.5, 20.5, 28.5, 40.5, 56.5], ["1-8", "9-16", "17-24", "25-32", "33-48", "49-64"])
    ax.set_xlabel("rank in the read")
    ax.set_ylabel("hit rate / mismatched pole's hit rate")
    log_ticks(ax)
    legend_above(ax, 2)
    end_labels(ax, items)
    finish(fig, "13_rank_depth", "A read's signal holds down to rank 64, not just its top tokens",
           "How often the token at each rank of a pole's read shows up in its behaviour (the next token for writes, "
           "the firing token for reads), divided by the same for a mismatched pole. Log scale.",
           "findings §7; data/summary/rank_depth_qwen3-8b.log",
           "Signal fades with rank but ranks 49-64 still run several times above a mismatched pole (L28's write read "
           "30×, L12's read 6×), which is why the catalog keeps the top 64 of every read. L4's S-lens ratio is noisy: "
           "its mismatched rate is near zero.")


def fig_lens_blind() -> None:
    d = summary(f"lens_blind_{STEM}.json")
    bands = [band_name(lo, hi) for lo, hi in BANDS]
    x = list(range(len(bands)))
    fig, axes = figure(2, width=9.0)
    arms = [a for a in (("lens", "lens reads only", 0), ("contexts", "top contexts only", 1),
                        ("both", "both (Sonnet)", 2), ("opus", "both (Opus)", 3)) if f"all/{a[0]}" in d]
    for ax, key, ref, ref_label, ylabel, lo, title in (
            (axes[0], "fires_all", None, "control: another pole's label", "detection AUROC on held-out firings", 0.5,
             "Where it fires"),
            (axes[1], "writes_top1", 0.25, "chance", "own label picked from steered text", 0.0,
             "What it promotes, when steered")):
        ax.set_title(title)
        if ref is None:  # the derangement control, both arms pooled
            ctrl = [(d[f"{b}/lens"]["fires_all_control"] + d[f"{b}/contexts"]["fires_all_control"]) / 2 for b in bands]
            series(ax, x, ctrl, None, ref_label, ls="--")
        else:
            series(ax, x, [ref] * len(x), None, ref_label, end=False, ls="--")
        for k, (arm, label, slot) in enumerate(arms):
            off = (k - (len(arms) - 1) / 2) * 0.1
            ys = [d[f"{b}/{arm}"][key] for b in bands]
            series(ax, [i + off for i in x], ys, slot, label)
            for i, b in enumerate(bands):
                a, z = d[f"{b}/{arm}"][f"{key}_ci"]
                ax.plot([i + off] * 2, [a, z], color=SLOT[slot], lw=1.0, alpha=0.55, zorder=2)
        ax.set_xticks(x, bands)
        ax.set_xlim(-0.4, len(bands) - 0.6)
        ax.set_ylim(lo, 1.0)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("layer band")
        if key == "writes_top1":
            pct(ax)
            ax.annotate("chance", (len(bands) - 1, 0.25), xytext=(0, -10), textcoords="offset points", ha="center",
                        fontsize=8, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[1:] + handles[:1], labels[1:] + labels[:1], loc="upper left", bbox_to_anchor=(0, -0.2),
                   ncols=5, borderaxespad=0, columnspacing=1.2)
    finish(fig, "14_lens_blind", "Labels written blind: lens reads predict what a pole promotes, contexts where it fires",
           "160 random poles, 32 a band. Explainers saw lens reads, top contexts or both (Sonnet; Opus for both), "
           "never steering; blind Sonnet scorers tested their labels on held-out firings and on steered text.",
           "findings §8; data/summary/lens_blind_qwen3-8b.json",
           "Left: mean detection AUROC of each arm's \"fires\" label on 6 held-out top contexts against 6 quiet "
           "positions (half from the same documents): lens reads alone 0.80, contexts 0.87, both 0.89; dashed, "
           "another pole's label (0.58). Right: how often a judge picks the pole's own \"promotes\" label out of four "
           "from its band, from greedy text steered at 0.5-2 nats: lens reads 65% (95% at L28-L35), contexts 54%; "
           "dashed, chance. Bars: 95% bootstrap intervals over poles.")


def fig_library() -> None:
    j = summary(f"library_judge_{STEM}.json")
    sides = [(k, label, color) for k, label, color in (("write", "write side", R_COLOR),
                                                         ("input", "input side", S_COLOR),
                                                         ("both", "both sides together", SLOT[1])) if k in j]
    groups = [g for g in j["write"]["all names"] if g.startswith("L") and g[1].isdigit()]
    fig, (ax,) = figure(top=0.72)
    bars = sides + [("control", "control: swapped names, both sides", MUTED)]
    w = 0.8 / len(bars)
    for i, (k, label, color) in enumerate(bars):
        src, idx = (j[k]["all names"], 0) if k != "control" else (j["both" if "both" in j else "write"]["all names"], 1)
        xs = [g + (i - (len(bars) - 1) / 2) * w for g in range(len(groups))]
        ax.bar(xs, [src[g]["clear"][idx] for g in groups], width=w - 0.03, color=color, label=label)
    ax.set_xticks(range(len(groups)), [re.sub(r"L(\d+)-(\d+)", r"L\1-L\2", g) if g != "L0-35" else "all (equal weight)"
                                       for g in groups])
    ax.set_xlabel("layers")
    ax.set_ylim(0, 1)
    pct(ax)
    ax.set_ylabel("share of persona rows judged a clear match")
    legend_above(ax, len(bars))
    finish(fig, "15_library_judge", "One persona label in three matches the pole's lens reads; swapped names rarely do",
           "The Neuron Persona Library's persona rows (from steering on \"Who are you?\"; jfonsecarivera, after "
           "Sauers): 60 per layer group, judged blind and generously by Claude against the pole's write reads, its "
           "input reads, or both, beside swapped names.",
           "findings §10; data/summary/library_judge_qwen3-8b.json",
           "Share of persona rows whose reads clearly match the name (the concept, a translation, or a strongly "
           "associated thing): 34% from the write side, 27% from the input side and 40% with both shown together, "
           "against 2-5% for swapped names. Matches are most common at L24-L35; the input side is weakest at "
           "L18-L23 (10%).")


def fig_head_lens_sides() -> None:
    d = summary(f"head_lens_sides_{STEM}.json")
    keys = ["L1-8", "L9-12", "L13-20", "L21-27", "L28-35"]
    names = ["L1-L8", "L9-L12", "L13-L20", "L21-L27", "L28-L35"]
    brackets = [("S→S", "S-lens in, S-lens out (ov_S)", S_COLOR, 1.0), ("R→S", "R-lens in, S-lens out", S_COLOR, 0.32),
                ("S→R", "S-lens in, R-lens out (ov_R)", R_COLOR, 1.0), ("R→R", "R-lens in, R-lens out", R_COLOR, 0.32)]
    fig, axes = figure(2, width=9.0, height=4.9, bottom=0.27)
    w = 0.19
    for ax, test, title in ((axes[0], "copy", "Copy test: how strongly it copies what it reads"),
                            (axes[1], "column", "Column test: its whole output")):
        for i, (k, label, color, t) in enumerate(brackets):
            vals = [d[test][k][b][0] - d[test][k][b][1] for b in keys]
            xs = [g + (i - 1.5) * w + (0.04 if i >= 2 else -0.04) for g in range(len(keys))]
            ax.bar(xs, vals, width=w - 0.02, color=tint(color, t), label=label, edgecolor=color if t < 1 else "none",
                   linewidth=0.8)
        ax.axhline(0, color=BASE, lw=0.8, zorder=1)
        ax.set_xticks(range(len(keys)), names)
        ax.set_xlim(-0.55, len(keys) - 0.45)
        ax.set_xlabel("layer band")
        ax.set_title(title)
        ax.set_ylim(-0.05, 0.9 if test == "copy" else 0.3)
    axes[0].set_ylabel("Spearman, own head minus control head")
    legend_below(axes[0], 4)
    finish(fig, "16_head_lens_sides", "Read a head's input with the S-lens; its output with the S-lens early, the R-lens late",
           "Each head's weight-only OV read, checked against what it actually writes at its top firings (23,112 "
           "firings from L1, all documents): with the S-lens or the R-lens on the attended token (input) and on the "
           "head's write (output). Bars: own head minus another head of the layer.",
           "findings §11; data/summary/head_lens_sides_qwen3-8b.json",
           "Margins over a control head, by band. Solid bars (S-lens input) beat pale ones (R-lens input) with the same "
           "output lens in every band, on both tests, except the copy test at L28-L35 with an S-lens output, which "
           "the catalog doesn't use there. On the output side the S-lens leads at L1-L8 (copy +0.44 against +0.39) and "
           "roughly ties at L13-L20; the R-lens leads at L9-L12 and by far from L21 (+0.65 and +0.80).")


def fig_head_census() -> None:
    t = pq.read_table(head_table("legible", MODEL)).to_pandas()
    tests = ("ov_R", "ov_copy", "ov_S", "qk_S")
    t["passes"] = (t[[f"{k}_q" for k in tests]] <= 0.05).any(axis=1)
    t["mode_only"] = ((t.cur_mode >= 1 / 3) | (t.src_mode >= 1 / 3) | t.pattern) & ~t["passes"]
    by = t.groupby("layer")
    x = list(by.groups)
    fig, (ax,) = figure()
    bottom = [0.0] * len(x)
    for k, label, color in (("passes", "an OV read passes (ov_R, ov_copy;\nov_S and qk_S pass for none)", R_COLOR),
                            ("mode_only", ("only a token mode or pattern:\na third of firings on one token,\n"
                                           "or induction, duplicate-token,\nprevious-token, self"), SLOT[3]),
                            ("dark", "dark: none of these", BASE)):
        vals = by[k].mean().tolist()
        ax.bar(x, vals, bottom=bottom, width=0.86, color=color, label=label, edgecolor=SURFACE, linewidth=0.8)
        bottom = [b + v for b, v in zip(bottom, vals)]
    assert max(abs(b - 1) for b in bottom) < 1e-6
    layer_axis(ax)
    ax.set_ylim(0, 1)
    pct(ax)
    ax.set_ylabel("share of the layer's 32 heads")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), labelspacing=0.9)
    ax.grid(False)
    band = summary(f"legible_{STEM}.json")["heads"]["bands"]
    late = t[t.layer >= 28]["passes"].mean()
    assert abs(late - band["L28-35"]["any_test"]) < 1e-6
    finish(fig, "17_head_census", "Late heads' OV reads predict the next token; mid-network heads are mostly dark",
           "Each head's weight-only OV and QK reads tested at its 24 strongest firings against the layer's other heads "
           "(5% false discovery rate), with the tokens it fires on and attends to, and its attention pattern.",
           "findings §12; data/summary/legible_qwen3-8b.json; heads/legible",
           f"Shares of each layer's heads: an OV read passes, only a token mode or attention pattern, or dark. `ov_R` "
           f"(the attended token's column predicts the next token) passes for {100 * band['L28-35']['ov_R']:.0f}% of "
           f"L28-L35's heads, half at L33; the S-side reads (`ov_S`, `qk_S`) never separate a head from its layer. "
           f"Dark is {100 * band['L0-35']['dark']:.0f}% of heads, {100 * band['L9-12']['dark']:.0f}% at L9-L12 and "
           f"{100 * band['L13-20']['dark']:.0f}% at L13-L20.")


# --------------------------------------------------------------------------------------------- Gemma 4 12B


def pole_name(layer: int, neuron: int, pole: int) -> str:
    return f"L{layer} n{neuron} {'+' if pole > 0 else '−'}"


def minus(v: float, digits: int) -> str:
    """A number for a caption, with a true minus sign."""
    return f"{v:.{digits}f}".replace("-", "−")


def band_label(band: str) -> str:
    """A summary's band key as the figures write bands: ``L0-11`` → ``L0-L11``."""
    return re.sub(r"^L(\d+)-(\d+)$", r"L\1-L\2", band)


def fig_two_models() -> None:
    """Qwen3-8B and Gemma 4 12B on one axis of relative depth: steering against the R-lens, and the census."""
    fig, (a, b) = figure(2)
    at = {}
    for name, stem, record, slot in (("Qwen3-8B", STEM, "R:dclm", 6), ("Gemma 4 12B", GEMMA_STEM, "hub", 1)):
        steer = summary(f"causal_lens_{stem}.json")["targets"]["steer 1x"]
        rows = summary(f"legible_{stem}.json")["neurons"]["layers"]
        last = len(rows) - 1
        depth = lambda layers, last=last: [layer / last for layer in layers]
        series(a, depth(steer[record]["layer"]), steer[record]["r_med"], slot, f"{name}: R-lens")
        a.plot(depth(steer["logit"]["layer"]), steer["logit"]["r_med"], color=SLOT[slot], lw=0.9, ls=(0, (3, 2)),
               zorder=2, label=f"{name}: logit lens")
        x = depth([r["layer"] for r in rows])
        series(b, x, [r["any_test"] for r in rows], slot, f"{name}: a weight-only read passes")
        series(b, x, [r["dark"] for r in rows], slot, f"{name}: dark", ls=(0, (4, 2)))
        at[name] = (steer[record]["r_med"], rows)
    for ax in (a, b):
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
        ax.set_xlabel("relative depth: layer / last layer")
        legend_below(ax)
    a.set_ylim(-0.05, 1.0)
    a.set_title("Steering: the write added at 1× its scale")
    a.set_ylabel("median r, lens read vs steering effect")
    b.set_ylim(0, 1)
    pct(b)
    b.set_title("Which reads hold at the top firings")
    b.set_ylabel("share of the layer's poles")
    (qr, qrows), (gr, grows) = at["Qwen3-8B"], at["Gemma 4 12B"]
    first = lambda r: next(i for i, v in enumerate(r) if v >= 0.9)
    peak = lambda rows, k: max(rows, key=lambda r: r[k])
    qd, gd = peak(qrows, "dark"), peak(grows, "dark")
    qw, gw = peak(qrows[len(qrows) // 2:], "any_test"), peak(grows[len(grows) // 2:], "any_test")
    finish(fig, "18_two_models", "Qwen3-8B and Gemma 4 12B read alike: inputs early, writes late, a dark middle",
           "Both models on one depth axis, layer / (layers − 1): 36 layers for Qwen3-8B, 48 for Gemma 4 12B. Left: "
           "median r between a pole's R-lens write read and its 1× steering effect (dashed: the logit lens). Right: "
           "shares of each layer's poles at a 5% false discovery rate.",
           "findings §13; data/summary/causal_lens_qwen3-8b.json, causal_lens_gemma-4-12b.json, "
           "legible_qwen3-8b.json, legible_gemma-4-12b.json",
           f"Both models against relative depth (layer / last layer). Left: median r between the R-lens read of a "
           f"pole's write and its 1× steering effect. In Qwen3-8B it predicts steering from L4 ({min(qr[4:9]):.2f}-"
           f"{max(qr[4:9]):.2f} at L4-L8); in Gemma 4 12B it stays at {max(gr[:11]):.2f} or less through L10 (relative "
           f"depth 0.21), jumps to {gr[11]:.2f} at L11 and reaches 0.9 at L{first(gr)} (Qwen3-8B: L{first(qr)}). The "
           f"logit lens (dashed) trails the R-lens in both until late. Right: a weight-only read passes for "
           f"{100 * qrows[0]['any_test']:.0f}% of Qwen3-8B's L0 poles and {100 * min(r['any_test'] for r in grows[:2]):.0f}-"
           f"{100 * max(r['any_test'] for r in grows[:2]):.0f}% of Gemma's L0-L1, for at most "
           f"{100 * max(r['any_test'] for r in qrows[13:21]):.0f}% of Qwen3-8B's L13-L20 and "
           f"{100 * max(r['any_test'] for r in grows[15:31]):.0f}% of Gemma's L15-L30, and again late, mostly through "
           f"the R-lens write ({100 * qw['any_test']:.0f}% at Qwen3-8B's L{qw['layer']}, {100 * gw['any_test']:.0f}% at "
           f"Gemma's L{gw['layer']}). Dark peaks "
           f"mid-network: {100 * qd['dark']:.0f}% at Qwen3-8B's L{qd['layer']} and {100 * gd['dark']:.0f}% at Gemma's "
           f"L{gd['layer']} (relative depth {qd['layer'] / 35:.2f} and {gd['layer'] / 47:.2f}).")


def fig_gemma_steering() -> None:
    fig, j = steering_panels(GEMMA_STEM, {"hub": ("R-lens (of record)", 0), "logit": ("logit lens", 3)}, GEMMA_LAYERS)
    s1, lg, aw = j["steer 1x"]["hub"], j["steer 1x"]["logit"], j["ablate win"]["hub"]
    at = lambda t, layer: t["r_med"][t["layer"].index(layer)]
    lead = [at(s1, layer) - at(lg, layer) for layer in range(11, 31)]
    ctrl = max(abs(v) for t in (s1, aw) for v in t["control_med"])
    finish(fig, "19_gemma_steering", "Gemma 4 12B: a neuron's write, read through the R-lens, predicts steering from L11",
           "Correlation over the vocabulary between a lens's read of a pole's write (as it reaches the residual, after "
           "the post-feedforward norm) and the measured change in log-probabilities; 24 random neurons a layer, both "
           "poles. Band: the R-lens's interquartile range.",
           "findings §13; data/summary/causal_lens_gemma-4-12b.json",
           f"Median r between each lens's write read and the measured effect, by layer, as figure 7 for Qwen3-8B "
           f"(Gemma 4 12B has no J-lens). Left, steering at 1× the pole's typical activation: the R-lens stays at "
           f"{max(s1['r_med'][:11]):.2f} or less through L10, jumps to {at(s1, 11):.2f} at L11, and reaches "
           f"{at(s1, 29):.2f} at L29 and {at(s1, 40):.2f} at L40; the logit lens trails it by {min(lead):.2f}-"
           f"{max(lead):.2f} from L11 to L30 and closes the gap only in the last layers. Right, ablating the pole where "
           f"it fires: under 0.1 before L27, then rising to {at(aw, 41):.2f} at L41 and {at(aw, 46):.2f} at L46. "
           f"Controls stay within ±{ctrl:.2f}.")


def fig_gemma_census() -> None:
    rows = summary(f"legible_{GEMMA_STEM}.json")["neurons"]["layers"]
    x = [r["layer"] for r in rows]
    fig, (ax,) = figure()
    dash = (0, (4, 2))
    for k, label, slot, ls in (("read_S", "read_S: its S-lens read\nnames the firing token", 2, "-"),
                               ("write_S", "write_S: its S-lens write\nnames the firing token", 2, dash),
                               ("write_R", "write_R: its R-lens write\npredicts the next token", 0, "-"),
                               ("read_R", "read_R: its R-lens read\nnames the firing token", 0, dash),
                               ("dark", "dark: fires, but no test\nor detector passes", None, "-")):
        series(ax, x, [r[k] for r in rows], slot, label, ls=ls)
    layer_axis(ax, n=GEMMA_LAYERS)
    ax.set_ylim(-0.02, 1.0)
    pct(ax)
    ax.set_ylabel("share of the layer's poles")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), labelspacing=0.9)
    band = summary(f"legible_{GEMMA_STEM}.json")["neurons"]["bands"]
    span = lambda k, lo, hi: (100 * min(r[k] for r in rows[lo:hi + 1]), 100 * max(r[k] for r in rows[lo:hi + 1]))
    finish(fig, "20_gemma_census", "Gemma 4 12B: inputs read early, writes late, and most of the middle is dark",
           "Share of each layer's 30,720 poles whose weight-only read passes its test at the pole's top firings, "
           "against other poles of the layer, at a 5% false discovery rate (as figures 3 and 12 for Qwen3-8B).",
           "findings §13; data/summary/legible_gemma-4-12b.json; neurons/legible",
           "The input side reads early: the S-lens read names the firing token for {:.0f}-{:.0f}% of L0-L1's poles, "
           "falls to {:.1f}% at L9, and rises again at L10-L13 ({:.0f}-{:.0f}%), where the S-lens write also holds for "
           "{:.0f}-{:.0f}% at L11-L13. The write side reads late: the R-lens write predicts the next token after a "
           "firing from L32, for {:.0f}-{:.0f}% of poles at L33-L40, falling to {:.0f}-{:.0f}% at L44-L47. In between, "
           "{:.0f}-{:.0f}% of L16-L31's poles are dark. Over all layers {:.0f}% of poles pass some test and {:.0f}% are "
           "dark (Qwen3-8B: 22% and 57%).".format(
               *span("read_S", 0, 1), 100 * rows[9]["read_S"], *span("read_S", 10, 13), *span("write_S", 11, 13),
               *span("write_R", 33, 40), *span("write_R", 44, 47), *span("dark", 16, 31),
               100 * band["L0-47"]["any_test"], 100 * band["L0-47"]["dark"]))


def fig_media_reads() -> None:
    j = summary("gemma4_media/media_lens.json")
    bars = (("matched_proj", "S-lens read: its read projection", S_COLOR, 1.0),
            ("matched_act", "S-lens read: its activation", S_COLOR, 0.32),
            ("optimised_act", "optimised read: its activation", SLOT[1], 1.0))
    fig, axes = figure(2)
    w = 0.26
    for ax, key, title in ((axes[0], "image", "Images (COCO)"), (axes[1], "audio_esc", "Sounds (ESC-50)")):
        B = j[key]["B"]
        bands = list(B["matched_proj"]["bands"])
        for i, (k, label, color, t) in enumerate(bars):
            ax.bar([g + (i - 1) * w for g in range(len(bands))], [B[k]["bands"][b] for b in bands], w * 0.9,
                   color=tint(color, t), edgecolor=color if t < 1 else "none", linewidth=0.8, label=label)
        ax.axhline(0.5, color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.set_xticks(range(len(bands)), [band_label(b) for b in bands])
        ax.set_xlim(-0.55, len(bands) - 0.45)
        ax.set_xlabel("layer band")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
    axes[0].set_ylabel("rank of its own change (median)")
    legend_below(axes[0], 3)
    im, au = j["image"]["B"], j["audio_esc"]["B"]

    def span(vals) -> str:  # "0.86-1.00", or one number when they agree
        lo, hi = f"{min(vals):.2f}", f"{max(vals):.2f}"
        return lo if lo == hi else f"{lo}-{hi}"

    early = ("L0-5", "L6-11", "L12-23")
    finish(fig, "21_media_reads", "The S-lens's image read is a projection; an optimised read makes the neuron fire",
           "A pole's read added to 32 held-out images or sounds; its change ranked among its changes under 191 other "
           "poles' reads (1 the most; dashed: 0.5, chance), over a panel of 192 poles. Optimised: gradient ascent from the read.",
           "findings §14; data/summary/gemma4_media/media_lens.json",
           "Median rank of a pole's own change by band. Images: the S-lens read moves the pole's read projection "
           "({:.2f}, {:.2f} and {:.2f} at L0-L5, L6-L11 and L12-L23) but raises its activation only at L0-L5 ({:.2f}; "
           "{:.2f} at L6-L11). Sounds: the S-lens read is near chance on both ({} by band). The read optimised from "
           "it (one pattern tiled over the 64 slots, at 8% of the pixel range or 0.3 of a frame's RMS) raises its pole "
           "through L23 (images {}, sounds {}) and is near chance from L24.".format(
               *(im["matched_proj"]["bands"][b] for b in early), im["matched_act"]["bands"]["L0-5"],
               im["matched_act"]["bands"]["L6-11"],
               span([v for k in ("matched_proj", "matched_act") for v in au[k]["bands"].values()]),
               span([im["optimised_act"]["bands"][b] for b in early]),
               span([au["optimised_act"]["bands"][b] for b in early])))


def optimised_picks(modality: str):
    """The gallery's poles: in each six-layer band from L0 to L23, the three whose exemplar labels are strongest
    (one pole per label) among those whose optimised read raises them on held-out documents."""
    opt = pq.read_table(catalog_dir(GEMMA) / "neurons" / f"{modality}_optimised.parquet").to_pandas()
    picks = []
    for lo in range(0, 24, 6):
        d = opt[(opt.layer >= lo) & (opt.layer < lo + 6) & (opt.gain > 0)]
        picks += list(d.sort_values("label_z", ascending=False, kind="stable").drop_duplicates("label").head(3).itertuples())
    return picks, opt


def fig_media_gallery() -> None:
    picks, _ = optimised_picks("image")
    png = lambda cell: np.asarray(Image.open(io.BytesIO(cell["bytes"])))
    reads = {}
    for layer in sorted({p.layer for p in picks}):
        t = pq.read_table(catalog_dir(GEMMA) / "neurons" / "image" / f"L{layer:02d}.parquet",
                          columns=["neuron", "pole", "read_png"],
                          filters=[("neuron", "in", [p.neuron for p in picks if p.layer == layer])]).to_pylist()
        reads.update({(layer, r["neuron"], r["pole"]): r["read_png"] for r in t})
    n = len(picks)
    fig, axes = plt.subplots(2, n, figsize=(1.0 * n + 0.6, 3.9), dpi=100)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.69, bottom=0.12, wspace=0.08, hspace=0.22)
    for i, p in enumerate(picks):
        for row, img in ((0, png(reads[(p.layer, p.neuron, p.pole)])), (1, png(p.opt_png))):
            ax = axes[row, i]
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0, i].set_title(pole_name(p.layer, p.neuron, p.pole), fontsize=7.5, fontweight="normal")
        axes[1, i].set_xlabel(p.label if len(p.label) <= 15 else p.label[:14] + "…", fontsize=7, color=INK2)
    axes[0, 0].set_ylabel("S-lens read", fontsize=8)
    axes[1, 0].set_ylabel("optimised", fontsize=8)
    finish(fig, "22_media_gallery", "The patch each neuron reads, and the patch that drives it",
           "Top: the pole's read direction through the image S-lens, back to pixels (a 48 × 48 patch, ±3 sd around "
           "grey). Bottom: the patch optimised from it to raise the pole. Below: the pole's top exemplar label.",
           "findings §14; neurons/image (read_png), neurons/image_optimised (opt_png)",
           "Three poles from each six-layer band from L0 to L23, those with the strongest reliable exemplar labels (one "
           "pole per label) among the poles whose optimised patch raises them on held-out images. The S-lens reads share "
           "one pattern of column streaks from pole to pole, as text reads share hub tokens; the optimised patches, one "
           "pattern tiled over all 64 slots and added at 8% of the pixel range, differ by pole. Early labels are "
           "primitives (a checkerboard, a grating, scripts by writing system), later ones mostly objects.")


def fig_crossmodal() -> None:
    j = summary("gemma4_media/crossmodal.json")
    tests = (("fire", "its words fire it", 1), ("s_read", "the text S-lens reads it", 2),
             ("r_write", "the text R-lens writes it", 0))
    fig, (a, b) = figure(2, height=5.2)
    w = 0.26
    for ax, mod, title in ((a, "image", "Image labels"), (b, "audio", "Audio labels")):
        bands = list(j[mod]["fire"]["bands"])
        for i, (k, label, slot) in enumerate(tests):
            ax.bar([g + (i - 1) * w for g in range(len(bands))], [j[mod][k]["bands"][x][1] for x in bands], w * 0.9,
                   color=SLOT[slot], label=label)
        ax.axhline(0.05, color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.set_xticks(range(len(bands)), [band_label(x) for x in bands])
        ax.set_xlim(-0.55, len(bands) - 0.45)
        ax.set_xlabel("layer band")
        pct(ax)
        ax.set_title(f"{title}: {j[mod]['all_three']:,} poles pass all three")
    a.set_ylabel("share of labelled poles, own label ranked ≥ 0.95")
    legend_below(a, 3)
    im, au, tri = j["image"], j["audio"], j["text_image_audio"]["all_three"]
    share = lambda m, k, band: 100 * m[k]["bands"][band][1]
    finish(fig, "23_crossmodal", "Image labels hold on the text side from L12; sound labels only a little",
           "For each pole with a reliable image or audio exemplar label, the rank of that label among the modality's "
           "labels on the text side, three ways; bars: the share of poles at 0.95 or more (dashed: 5%, chance).",
           "findings §15; data/summary/gemma4_media/crossmodal.json",
           f"Of {sum(v[2] for v in im['fire']['bands'].values()):,} poles with an image label, the label's words fire "
           f"the pole for {share(im, 'fire', 'L0-11'):.0f}% at L0-L11 and {share(im, 'fire', 'L12-23'):.0f}% at "
           f"L12-L23 (5% by chance), the text R-lens writes it for up to {share(im, 'r_write', 'L12-23'):.0f}% (L12-L23), the text "
           f"S-lens reads it for {min(v[1] for v in im['s_read']['bands'].values()) * 100:.0f}-"
           f"{max(v[1] for v in im['s_read']['bands'].values()) * 100:.0f}%. Audio labels pass "
           f"{min(min(v[1] for v in au[k]['bands'].values()) for k in ('fire', 's_read', 'r_write')) * 100:.0f}-"
           f"{max(max(v[1] for v in au[k]['bands'].values()) for k in ('fire', 's_read', 'r_write')) * 100:.0f}% "
           f"by band and test. {tri['passing_both']} of {tri['scored_in_both']:,} poles with both labels pass all three "
           f"text tests for both (independent lists would share {tri['expected_if_independent']:.1f}): places and "
           f"peoples with their language, such as mosques and Arabic, pagodas and Mandarin, tacos and Spanish.")


def fig_heads_media() -> None:
    t = pq.read_table(catalog_dir(GEMMA) / "heads" / "media.parquet").to_pandas()
    fig, (a, b) = figure(2)
    colors = {"image": 6, "audio": 1}
    for mod, slot in colors.items():
        g = t.groupby("layer")[f"{mod}_read_ratio"]
        layers = list(g.max().index)
        series(a, layers, list(g.max()), slot, f"{mod}: the layer's strongest reader")
        a.plot(layers, list(g.median()), color=tint(SLOT[slot], 0.5), lw=1.0, zorder=2,
               label=f"{mod}: the layer's median head")
    a.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=1)
    a.annotate("even spread", (GEMMA_LAYERS - 0.5, 1.0), xytext=(0, 3), textcoords="offset points", ha="right",
               fontsize=8, color=INK2)
    layer_axis(a, n=GEMMA_LAYERS)
    a.set_ylim(0, 1.5)
    a.set_title("Attention from the text to the media")
    a.set_ylabel("× an even spread over the keys")
    legend_below(a)
    z = t.groupby("layer")["image_content_z"].max()
    b.bar(z.index, z.values, color=SLOT[colors["image"]], width=0.8)
    b.axhline(3, color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=3)
    layer_axis(b, n=GEMMA_LAYERS)
    b.set_title("Images: the layer's strongest content carrier")
    b.set_ylabel("z of the right label's extra loss")
    top = t.sort_values("image_content_z", ascending=False).head(4)
    mid = t[(t.layer >= 24) & (t.layer <= 35)]
    aud = next(t.sort_values("audio_read_ratio", ascending=False).itertuples())
    acc = summary(f"head_media_{GEMMA_STEM}.json")["audio"]["accuracy"]
    finish(fig, "24_heads_media", "Single heads carry what an image shows into the text; for sound, none does",
           "64 labelled images and 64 sounds (mini-ImageNet, ESC-50), each followed by ' a <label>': attention from the "
           "text after the media to its slots, and each head zeroed in turn (right; dashed: z = 3).",
           "findings §16; data/summary/head_media_gemma-4-12b.json; heads/media",
           "Left: attention from the text after the image or sound to its 64 slots, as a multiple of an even spread over "
           "the keys, for the layer's strongest head and its median head. The text reads images most at L23-L31 and L44 "
           f"(up to {t.image_read_ratio.max():.2f} of an even spread) and sounds most at L0-L10 (L{aud.layer}h{aud.head}: "
           f"{aud.audio_read_ratio:.2f} times). Right: zeroing single heads raises the right image label's loss more than a "
           "wrong one's, most for " + ", ".join(f"L{r.layer}h{r.head} (z {r.image_content_z:.1f})"
                                                 for r in top.itertuples()) +
           f"; {100 * (mid.image_content_z > 3).mean():.0f}% of L24-L35's heads exceed z = 3. For sound none does "
           f"(the largest z is {t.audio_content_z.max():.1f}), and the model ranks a sound's own label above another's "
           f"only {100 * acc:.0f}% of the time, too weakly to find its carriers.")


FIGURES = {"01": fig_anatomy, "02": fig_lenses, "03": fig_depth_map, "04": fig_lens_reliability, "05": fig_chat_lens,
           "06": fig_activations, "07": fig_steering, "08": fig_amplified, "09": fig_context_linearity,
           "10": fig_side_agreement, "11": fig_drivers, "12": fig_census, "13": fig_rank_depth, "14": fig_lens_blind,
           "15": fig_library, "16": fig_head_lens_sides, "17": fig_head_census, "18": fig_two_models,
           "19": fig_gemma_steering, "20": fig_gemma_census, "21": fig_media_reads, "22": fig_media_gallery,
           "23": fig_crossmodal, "24": fig_heads_media}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(FIGURES), metavar="NN",
                    help="draw only these figures, by number; the README is written only when all are drawn")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for k, f in FIGURES.items():
        if not args.only or k in args.only:
            f()
    if args.only:
        return
    names = {f"{name}.png" for name, *_ in CAPTIONS}
    for stale in OUT.glob("*.png"):  # figures renumbered or retired
        if stale.name not in names:
            stale.unlink()
    intro = (f"Drawn by `scripts/11_figures.py`. Figures 1-17 are {MODEL}'s: 1 and 2 are schematics, the others draw "
             "`data/summary/`, the released lenses' provenance and the catalog's `heads/legible` table. Figure 18 sets "
             f"it beside {GEMMA}, and 19-24 are Gemma's, from `data/summary/` and its catalog's media and head tables; "
             "no image or sound from a dataset is drawn, only lens reads and optimised patterns. Each figure's footer "
             "names its findings section and source.")
    md = ["# Figures", "", intro, ""]
    for name, title, caption in CAPTIONS:
        md += [f"## {int(name[:2])}. {title}", "", f"![{title}]({name}.png)", "", caption, ""]
    (OUT / "README.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
