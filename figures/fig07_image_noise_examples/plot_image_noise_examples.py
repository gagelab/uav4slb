"""
figures/fig07_image_noise_examples/plot_image_noise_examples.py
===============================================================
Figure 7 — Publication-quality 2 × 4 image panel figure.

    Row 1  (Worst Predictions) — four images, each labeled with a noise source
           (Weeds, Shadows, Brightness, Contrast) and annotated with the visual
           score, model score, and signed error.
    Row 2  (Best Predictions)  — four images annotated with scores and error
           but without noise-source badges.

Configuration
-------------
Edit the IMAGE_CONFIG block below to set the eight image paths, visual scores,
and model scores before running.  The script will warn (not crash) if an image
file is not found and will substitute a placeholder.

Image directory
---------------
UAV imagery is stored outside the repository due to file size.  Set
IMAGE_DIR to the path on your system, or override it with the
UAV4SLB_IMAGE_DIR environment variable:

    export UAV4SLB_IMAGE_DIR=/path/to/final_sliced
    python figures/fig07_image_noise_examples/plot_image_noise_examples.py

Default path used on the NCSU HPC (sunny):
    /mnt/research-projects/j/jlgage/RawUAVData01/uavforslb/
        uav_for_slb/final_image/final_sliced

Output
------
    fig07_image_noise_examples.{pdf,png}  written to this script's directory.

Usage
-----
    python figures/fig07_image_noise_examples/plot_image_noise_examples.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import apply_style, save_fig, NOISE_LABEL_COLORS, NOISE_LABEL_TEXT_COLORS

apply_style()

# ---------------------------------------------------------------------------
# Image directory  (overridable via environment variable)
# ---------------------------------------------------------------------------
_DEFAULT_IMAGE_DIR = Path(
    "/mnt/research-projects/j/jlgage/RawUAVData01/uavforslb/final_image/final_sliced"
)
IMAGE_DIR = Path(os.environ.get("UAV4SLB_IMAGE_DIR", _DEFAULT_IMAGE_DIR))

# ---------------------------------------------------------------------------
# IMAGE CONFIGURATION  ← edit this block before running
# ---------------------------------------------------------------------------
# Each dict requires:
#   path      : str   path relative to IMAGE_DIR (or an absolute path)
#   actual    : float human visual score (1–9 ordinal scale)
#   predicted : float EVA-02-B model score
#
# WORST_IMAGES additionally require:
#   label     : str   one of "Weeds" | "Shadows" | "Brightness" | "Contrast"

WORST_IMAGES: list[dict] = [
    {
        "path":      "20240718_C6B_02952.jpg",
        "actual":    3.0,
        "predicted": 6.8,
        "label":     "Weeds",
    },
    {
        "path":      "20230720_I3B_10866.jpg",
        "actual":    5.0,
        "predicted": 1.6,
        "label":     "Shadows",
    },
    {
        "path":      "20250704_C10_05016.jpg",
        "actual":    4.0,
        "predicted": 7.9,
        "label":     "Brightness",
    },
    {
        "path":      "20250714_C10_06820.jpg",
        "actual":    6.0,
        "predicted": 2.3,
        "label":     "Contrast",
    },
]

BEST_IMAGES: list[dict] = [
    {
        "path":      "20250701_C7B_00407.jpg",
        "actual":    7.0,
        "predicted": 7.1,
    },
    {
        "path":      "20250711_C10_04975.jpg",
        "actual":    3.0,
        "predicted": 3.2,
    },
    {
        "path":      "20250711_C10_05578.jpg",
        "actual":    5.0,
        "predicted": 5.0,
    },
    {
        "path":      "20250704_C10_04676.jpg",
        "actual":    8.0,
        "predicted": 7.8,
    },
]

OUTPUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _resolve_path(rel_or_abs: str) -> Path:
    """Return an absolute path: join with IMAGE_DIR if not already absolute."""
    p = Path(rel_or_abs)
    return p if p.is_absolute() else IMAGE_DIR / p


def _load_image(path: Path) -> np.ndarray:
    """
    Load an RGB image from disk.

    If the file does not exist, emit a warning and return a synthetic
    placeholder (a grey grid) so the layout can still be inspected.
    """
    if path.is_file():
        return np.array(Image.open(path).convert("RGB"))
    warnings.warn(f"Image not found: {path}  →  using placeholder.", stacklevel=2)
    rng   = np.random.default_rng(abs(hash(str(path))) % (2 ** 31))
    placeholder = rng.integers(60, 180, (256, 256, 3), dtype=np.uint8)
    placeholder[::32, :] = 40    # horizontal grid lines
    placeholder[:, ::32] = 40    # vertical grid lines
    return placeholder


# ---------------------------------------------------------------------------
# Per-axes drawing helpers
# ---------------------------------------------------------------------------

def _noise_badge(ax: plt.Axes, label: str) -> None:
    """
    Draw a filled rounded-rectangle badge in the top-left corner of *ax*.

    The badge color and text color are looked up from the shared palette so
    that Figures 7 and 8 share a consistent visual code.
    """
    color      = NOISE_LABEL_COLORS.get(label, "#333333")
    text_color = NOISE_LABEL_TEXT_COLORS.get(label, "black")
    ax.text(
        0.04, 0.97, label,
        transform=ax.transAxes,
        fontsize=11, fontweight="bold",
        color=text_color,
        va="top", ha="left",
        zorder=5,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor=color,
            edgecolor="none",
            alpha=0.9,
        ),
    )


def _score_caption(ax: plt.Axes, actual: float, predicted: float) -> None:
    """
    Place a three-line caption below *ax* showing visual score, model score,
    and signed error (positive = over-prediction, negative = under-prediction).
    """
    error = predicted - actual
    sign  = "+" if error >= 0 else ""
    ax.set_xlabel(
        f"Visual score: {actual:.1f}\n"
        f"Model score:  {predicted:.1f}\n"
        f"Error: {sign}{error:.1f}",
        fontsize=9,
        labelpad=4,
        linespacing=1.55,
    )


# ---------------------------------------------------------------------------
# Aspect ratio helper
# ---------------------------------------------------------------------------

def _image_aspect_ratio(entries: list[dict]) -> float:
    """
    Return height/width pixel ratio of the first loadable image.

    Falls back to 1.0 (square) if no image can be opened.
    """
    for e in entries:
        p = _resolve_path(e["path"])
        if p.is_file():
            with Image.open(p) as im:
                w, h = im.size
            return h / w
    return 1.0


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def make_figure(worst: list[dict], best: list[dict]) -> plt.Figure:
    """
    Assemble the 2 × 4 image panel figure.

    Layout (using figure-level fractional coordinates):
      - Each row begins with a centred bold row title.
      - Four image axes follow, with a score caption below each.
      - A small horizontal gap separates adjacent images.

    Parameters
    ----------
    worst : list of 4 dicts (must include "label" key)
    best  : list of 4 dicts

    Returns
    -------
    matplotlib Figure
    """
    n_cols, n_rows = 4, 2

    # ── figure geometry ─────────────────────────────────────────────────────
    # Full journal column width (≈ 7 in); row height calculated from the
    # actual image aspect ratio so pixels are not distorted.
    fig_w      = 7.0
    cell_w_in  = fig_w / n_cols
    aspect     = _image_aspect_ratio(worst + best)
    img_h_in   = cell_w_in * aspect      # image height in inches
    cap_h_in   = 0.55                    # caption area below each image
    title_h_in = 0.28                    # row title height
    v_gap_in   = 0.08                    # gap between title and image row
    block_h_in = title_h_in + v_gap_in + img_h_in + cap_h_in

    fig_h = block_h_in * n_rows + 0.15   # tiny bottom margin

    # Convert all geometry to figure fractions (required by add_axes)
    col_w_f    = 1.0 / n_cols
    img_h_f    = img_h_in  / fig_h
    cap_h_f    = cap_h_in  / fig_h
    title_h_f  = title_h_in / fig_h
    vgap_f     = v_gap_in  / fig_h
    h_pad_f    = 0.018                   # horizontal gap between images

    fig = plt.figure(figsize=(fig_w, fig_h))

    row_configs = [
        {"data": worst, "title": "Worst Predictions", "badges": True},
        {"data": best,  "title": "Best Predictions",  "badges": False},
    ]

    # matplotlib y=0 is at the bottom; row 0 (worst) occupies the top
    row_bottoms = [
        1.0 - block_h_in / fig_h,       # top row
        1.0 - 2 * block_h_in / fig_h,   # bottom row
    ]

    for cfg, row_bot in zip(row_configs, row_bottoms):
        # ── row title ────────────────────────────────────────────────────────
        title_bot = row_bot + cap_h_f + img_h_f + vgap_f
        ax_title  = fig.add_axes([0.0, title_bot, 1.0, title_h_f])
        ax_title.set_axis_off()
        ax_title.text(
            0.5, 0.5, cfg["title"],
            transform=ax_title.transAxes,
            fontsize=16, fontweight="bold",
            va="center", ha="center",
        )

        # ── image axes ────────────────────────────────────────────────────────
        img_bot = row_bot + cap_h_f
        for col_idx, entry in enumerate(cfg["data"]):
            left = col_idx * col_w_f + h_pad_f / 2
            w    = col_w_f - h_pad_f

            ax = fig.add_axes([left, img_bot, w, img_h_f])
            ax.imshow(_load_image(_resolve_path(entry["path"])), aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#aaaaaa")

            _score_caption(ax, entry["actual"], entry["predicted"])

            # Noise-source badge (worst row only)
            if cfg["badges"]:
                _noise_badge(ax, entry["label"])

    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    assert len(WORST_IMAGES) == 4, "WORST_IMAGES must contain exactly 4 entries."
    assert len(BEST_IMAGES)  == 4, "BEST_IMAGES must contain exactly 4 entries."
    for e in WORST_IMAGES:
        assert "label" in e, f"Missing 'label' in WORST_IMAGES entry: {e}"
        assert e["label"] in NOISE_LABEL_COLORS, (
            f"label {e['label']!r} not in NOISE_LABEL_COLORS. "
            f"Valid: {list(NOISE_LABEL_COLORS)}"
        )

    print(f"IMAGE_DIR  : {IMAGE_DIR}")
    print(f"Output dir : {OUTPUT_DIR}\n")

    fig = make_figure(WORST_IMAGES, BEST_IMAGES)
    save_fig(fig, OUTPUT_DIR, "fig07_image_noise_examples")
    print("Done.")


if __name__ == "__main__":
    main()
