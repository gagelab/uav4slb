"""
figures/style.py
================
Shared matplotlib style settings, color palettes, and plotting helpers used by
every manuscript figure script in uav4slb.

Usage
-----
Each figure script imports from this module via an explicit sys.path insert so
the repo does not require installation:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from style import apply_style, despine, save_fig, BATLOW_10, FAMILY_COLORS

Color palette
-------------
All colors are drawn from the perceptually-uniform, colorblind-safe batlow
colormap (Crameri 2018), sampled at 10 evenly-spaced stops from dark to light.
Named groupings (FAMILY_COLORS, FOLD_COLORS, etc.) assign specific stops to
semantic categories so that visual encoding is consistent across all figures.

Reference
---------
Crameri, F. (2018). Scientific colour maps. Zenodo.
https://doi.org/10.5281/zenodo.1243862
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------------------------------------------------------------------
# Batlow 10-swatch sequence  (dark → light)
# ---------------------------------------------------------------------------
# Stops are indexed 1 through 256; we sample 10 at equal intervals.
# The hex values below correspond to the swatches in colors.txt.
BATLOW_10: list[str] = [
    "#011959",   # stop   1 — very dark blue   (coldest)
    "#103F60",   # stop  29 — dark blue
    "#1C5A62",   # stop  58 — dark teal
    "#3C6D56",   # stop  86 — mid green
    "#687B3E",   # stop 114 — olive
    "#9D892B",   # stop 143 — amber-brown
    "#D29343",   # stop 171 — warm orange
    "#F8A17B",   # stop 199 — light orange / salmon
    "#FDB7BC",   # stop 228 — pink
    "#FACCFA",   # stop 256 — pale purple      (warmest)
]

# ---------------------------------------------------------------------------
# Architecture family colors  (Figures 3, S3)
# ---------------------------------------------------------------------------
# Three families are separated across the batlow range: CNN anchors the warm
# end, ViT the green mid-range, Hybrid the dark-blue cold end.  This gives
# maximum perceptual distance while staying within the palette.
FAMILY_COLORS: dict[str, str] = {
    "CNN":    "#3C6D56",   # batlow-171 — warm orange
    "ViT":    "#D29343",   # batlow-86  — mid green
    "Hybrid": "#103F60",   # batlow-29  — dark blue
}

# ---------------------------------------------------------------------------
# CV0 fold colors — one per held-out test year  (Figures 4, 9)
# ---------------------------------------------------------------------------
# Three folds span from the darkest (2023) to the warm-orange mid-range (2025),
# leaving the mid-green for the 2024 fold, which is visually intermediate.
FOLD_COLORS: dict[str, str] = {
    "fold_23": "#011959",   # batlow-1   — 2023 test year
    "fold_24": "#3C6D56",   # batlow-86  — 2024 test year
    "fold_25": "#D29343",   # batlow-171 — 2025 test year
}

# Convenience mapping: integer year → fold name (used in some figure scripts)
YEAR_TO_FOLD: dict[int, str] = {2023: "fold_23", 2024: "fold_24", 2025: "fold_25"}

# Convenience mapping: integer year → batlow color (mirrors FOLD_COLORS)
YEAR_COLORS: dict[int, str] = {
    2023: FOLD_COLORS["fold_23"],   # batlow-1   — very dark blue
    2024: FOLD_COLORS["fold_24"],   # batlow-86  — mid green
    2025: FOLD_COLORS["fold_25"],   # batlow-171 — warm orange
}

# ---------------------------------------------------------------------------
# Field colors — one per field  (EDA / supplemental figures)
# ---------------------------------------------------------------------------
# Six fields are mapped to six distinct batlow stops, roughly sorted so that
# 2023 fields are cooler and 2025 fields are warmer, matching FOLD_COLORS.
FIELD_COLORS: dict[str, str] = {
    "G3":  "#011959",   # 2023 — batlow-1
    "I3B": "#1C5A62",   # 2023 — batlow-58
    "B7A": "#3C6D56",   # 2024 — batlow-86
    "C6B": "#687B3E",   # 2024 — batlow-114
    "C10": "#D29343",   # 2025 — batlow-171
    "C7B": "#9D892B",   # 2025 — batlow-143
}

# ---------------------------------------------------------------------------
# Image noise source label colors  (Figure 7)
# ---------------------------------------------------------------------------
# Each noise category maps to a batlow stop that is visually distinct on a
# bright image background.  Text is always white for contrast.
NOISE_LABEL_COLORS: dict[str, str] = {
    "Weeds":      "#011959",   # batlow-1   — very dark blue
    "Shadows":    "#1C5A62",   # batlow-58  — dark teal
    "Brightness": "#687B3E",   # batlow-114 — olive
    "Contrast":   "#D29343",   # batlow-171 — warm orange
}

NOISE_LABEL_TEXT_COLORS: dict[str, str] = {key: "white" for key in NOISE_LABEL_COLORS}

# ---------------------------------------------------------------------------
# Image-level covariate scatter colors  (Figure 8)
# ---------------------------------------------------------------------------
# Intentionally matched to NOISE_LABEL_COLORS so that Figures 7 and 8 share
# a consistent visual code: "Weeds → #011959", "Shadows → #1C5A62", etc.
COVARIATE_COLORS: dict[str, str] = {
    "frac_weed":       "#011959",   # batlow-1   (Weeds)
    "sf_illuminorm":   "#1C5A62",   # batlow-58  (Shadows)
    "mean_brightness": "#687B3E",   # batlow-114 (Brightness)
    "contrast_rms":    "#D29343",   # batlow-171 (Contrast)
}

# Human-readable axis labels for the four image covariates
COVARIATE_LABELS: dict[str, str] = {
    "frac_weed":       "Weed fraction",
    "sf_illuminorm":   "Shadow fraction",
    "mean_brightness": "Mean brightness",
    "contrast_rms":    "Contrast RMS",
}

# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """
    Apply publication-quality matplotlib rcParams.

    Call once at the top of each figure script, before creating any axes.

    Key settings
    ------------
    - pdf.fonttype = 42  →  editable text in Illustrator / Inkscape
    - font: Arial with DejaVu fallback
    - 8 pt base size (individual scripts scale up per element as needed)
    - top and right spines hidden by default
    """
    rcParams["pdf.fonttype"]       = 42
    rcParams["ps.fonttype"]        = 42
    rcParams["font.family"]        = "sans-serif"
    rcParams["font.sans-serif"]    = ["Arial", "Helvetica", "DejaVu Sans"]
    rcParams["font.size"]          = 8
    rcParams["axes.linewidth"]     = 1.0
    rcParams["axes.spines.top"]    = False   # remove top spine globally
    rcParams["axes.spines.right"]  = False   # remove right spine globally
    rcParams["xtick.major.width"]  = 1.0
    rcParams["ytick.major.width"]  = 1.0
    rcParams["xtick.direction"]    = "out"
    rcParams["ytick.direction"]    = "out"
    rcParams["legend.framealpha"]  = 0.9
    rcParams["legend.edgecolor"]   = "#cccccc"


# ---------------------------------------------------------------------------
# Axes helpers
# ---------------------------------------------------------------------------

def despine(ax: plt.Axes, top: bool = True, right: bool = True) -> None:
    """
    Remove the top and/or right spines from an axes.

    Parameters
    ----------
    ax    : matplotlib Axes to modify
    top   : remove the top spine (default True)
    right : remove the right spine (default True)
    """
    if top:
        ax.spines["top"].set_visible(False)
    if right:
        ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Figure output helper
# ---------------------------------------------------------------------------

def save_fig(
    fig: plt.Figure,
    out_dir: str | Path,
    stem: str,
    dpi: int = 300,
    close: bool = True,
) -> None:
    """
    Save a figure as both PDF (vector) and PNG (raster) in *out_dir*.

    Parameters
    ----------
    fig     : matplotlib Figure to save
    out_dir : target directory; created automatically if it does not exist
    stem    : filename stem without extension, e.g. "model_comparison"
    dpi     : raster resolution for PNG output (default 300)
    close   : if True, close the figure after saving (default True)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved → {path}")
    if close:
        plt.close(fig)
