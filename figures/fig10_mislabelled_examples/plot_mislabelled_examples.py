"""
figures/fig10_mislabelled_examples/plot_mislabelled_examples.py
===============================================================
Figure 10 — Five images with large prediction errors that are plausibly due
to label noise, displayed in a single row.

Each panel shows the image tile annotated with:
    - Visual score (human rater, 1–9 ordinal scale)
    - Model score  (EVA-02-B predicted value)
    - Signed error (model − visual, colored by direction)
    - An optional short note explaining the suspected noise source

Configuration
-------------
Edit the EXAMPLES list below to specify the five image files, scores, and
notes before running.

Image directory
---------------
UAV imagery is stored outside the repository.  Set IMAGE_DIR below or
override with the UAV4SLB_IMAGE_DIR environment variable:

    export UAV4SLB_IMAGE_DIR=/path/to/final_sliced
    python figures/fig10_mislabelled_examples/plot_mislabelled_examples.py

Default HPC path (sunny):
    /mnt/research-projects/j/jlgage/RawUAVData01/uavforslb/
        uav_for_slb/final_image/final_sliced

Output (written to this script's directory)
------------------------------------------
    fig10_mislabelled_examples.{pdf,png}

Usage
-----
    python figures/fig10_mislabelled_examples/plot_mislabelled_examples.py
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import apply_style, save_fig

apply_style()

# ---------------------------------------------------------------------------
# Image directory  (overridable via environment variable)
# ---------------------------------------------------------------------------
_DEFAULT_IMAGE_DIR = (
    "/mnt/research-projects/j/jlgage/RawUAVData01/uavforslb/final_image/final_sliced"
)
IMAGE_DIR  = Path(os.environ.get("UAV4SLB_IMAGE_DIR", _DEFAULT_IMAGE_DIR))
OUTPUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# EXAMPLES  ← edit this block before running
# ---------------------------------------------------------------------------
# image_path : filename relative to IMAGE_DIR (or an absolute path)
# actual     : human visual score (1–9 ordinal scale)
# predicted  : EVA-02-B model score
# note       : short explanation of the suspected noise source ('' to omit)
# ---------------------------------------------------------------------------

@dataclass
class Example:
    image_path: str
    actual:     float
    predicted:  float
    note:       str = ""


EXAMPLES: list[Example] = [
    Example(
        image_path="20250710_C7B_01725.jpg",
        actual=7.5, predicted=4.31,
        note="Necrosis visible aerially",
    ),
    Example(
        image_path="20240726_C6B_00130.jpg",
        actual=3, predicted=7.94,
        note="Dense canopy obscures lesions",
    ),
    Example(
        image_path="20240718_C6B_01911.jpg",
        actual=7, predicted=2.89,
        note="Possible data-entry error",
    ),
    Example(
        image_path="20250718_C10_06687.jpg",
        actual=3.0, predicted=6.47,
        note="Processing artifacts distort image",
    ),
    Example(
        image_path="20230727_G3_03053.jpg",
        actual=4, predicted=6.25,
        note="Shadows confound score",
    ),
]

# ---------------------------------------------------------------------------
# Error color mapping
# ---------------------------------------------------------------------------
# Over-prediction (positive error) → warm orange; under-prediction → dark teal.
# These match COVARIATE_COLORS and are drawn from the batlow palette.
COLOR_OVER  = "#D29343"   # batlow-171 — model over-predicts
COLOR_UNDER = "#1C5A62"   # batlow-58  — model under-predicts
COLOR_EXACT = "#3C6D56"   # batlow-86  — zero error

# ---------------------------------------------------------------------------
# Figure geometry
# ---------------------------------------------------------------------------
DPI            = 300
IMAGE_HEIGHT   = 4.0   # inches per image row (aspect ratio preserved)
LABEL_PAD      = 0.55  # inches below image reserved for score labels
H_GAP          = 0.15  # inches between panels
LEFT_MARGIN    = 0.15
RIGHT_MARGIN   = 0.15
TOP_MARGIN     = 0.70  # room for optional suptitle
BOTTOM_MARGIN  = 0.10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else IMAGE_DIR / p


def _load_image(path: Path) -> Image.Image | None:
    """Load image from disk; return None with a warning if not found."""
    if not path.exists():
        warnings.warn(f"Image not found: {path}", stacklevel=2)
        return None
    return Image.open(path).convert("RGB")


def _error_color(actual: float, predicted: float) -> str:
    """Map the sign of (predicted - actual) to a display color."""
    err = predicted - actual
    if err > 0:
        return COLOR_OVER
    elif err < 0:
        return COLOR_UNDER
    return COLOR_EXACT


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(examples: list[Example]) -> plt.Figure:
    """
    Assemble the single-row image panel figure.

    Panel width is derived from the pixel aspect ratio of the first loadable
    image so that images are displayed without distortion.

    Layout uses figure-level fractional coordinates via fig.add_axes() so that
    the score labels sit precisely below each image regardless of font metrics.

    Parameters
    ----------
    examples : list of exactly 5 Example instances

    Returns
    -------
    matplotlib Figure
    """
    n = len(examples)

    # ── determine panel width from image aspect ratio ─────────────────────
    aspect = 1.0   # square fallback if no image loads
    for ex in examples:
        img = _load_image(_resolve_path(ex.image_path))
        if img is not None:
            w_px, h_px = img.size
            aspect = w_px / h_px
            break

    panel_w_in = IMAGE_HEIGHT * aspect   # inches

    total_w = LEFT_MARGIN + n * panel_w_in + (n - 1) * H_GAP + RIGHT_MARGIN
    total_h = TOP_MARGIN + IMAGE_HEIGHT + LABEL_PAD + BOTTOM_MARGIN

    fig = plt.figure(figsize=(total_w, total_h))

    # ── add one Axes per panel ────────────────────────────────────────────
    axes: list[plt.Axes] = []
    for i in range(n):
        x0 = (LEFT_MARGIN + i * (panel_w_in + H_GAP)) / total_w
        y0 = (BOTTOM_MARGIN + LABEL_PAD) / total_h
        w  = panel_w_in / total_w
        h  = IMAGE_HEIGHT / total_h
        axes.append(fig.add_axes([x0, y0, w, h]))

    # ── draw each panel ───────────────────────────────────────────────────
    for i, (ax, ex) in enumerate(zip(axes, examples)):
        img = _load_image(_resolve_path(ex.image_path))
        if img is not None:
            ax.imshow(img)
        else:
            # Placeholder when image file is missing
            ax.set_facecolor("#cccccc")
            ax.text(0.5, 0.5, "Image not found",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=8, color="#666666")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_edgecolor("#444444")

        # ── score labels beneath each image (figure-level coordinates) ────
        err       = ex.predicted - ex.actual
        sign      = "+" if err >= 0 else ""
        err_color = _error_color(ex.actual, ex.predicted)

        # Horizontal centre of this panel in figure coordinates
        cx = (LEFT_MARGIN + i * (panel_w_in + H_GAP) + panel_w_in / 2) / total_w

        # Three lines of text, bottom-aligned just below the image
        line_gap = (LABEL_PAD * 0.28) / total_h
        y_base   = (BOTTOM_MARGIN + LABEL_PAD * 0.90) / total_h

        lines = [
            (f"Visual score: {ex.actual}",        "#222222"),
            (f"Model score:  {ex.predicted:.1f}", "#222222"),
            (f"Error: {sign}{err:.1f}",            err_color),
        ]
        for j, (text, color) in enumerate(reversed(lines)):
            fig.text(
                cx, y_base - j * line_gap,
                text,
                ha="center", va="top",
                fontsize=11, color=color,
                fontweight="bold" if "Error" in text else "normal",
                fontfamily="sans-serif",
            )

        # ── optional noise-source note above the image ─────────────────────
        if ex.note:
            y_top = (BOTTOM_MARGIN + LABEL_PAD + IMAGE_HEIGHT) / total_h
            fig.text(
                cx, y_top + 0.005,
                ex.note,
                ha="center", va="bottom",
                fontsize=9, color="#555555",
                style="italic", fontfamily="sans-serif",
            )

    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(EXAMPLES) != 5:
        raise ValueError(f"EXAMPLES must contain exactly 5 entries; got {len(EXAMPLES)}.")

    print(f"IMAGE_DIR  : {IMAGE_DIR}")
    print(f"Output dir : {OUTPUT_DIR}\n")

    fig = build_figure(EXAMPLES)
    save_fig(fig, OUTPUT_DIR, "fig10_mislabelled_examples", dpi=DPI)
    print("Done.")


if __name__ == "__main__":
    main()
