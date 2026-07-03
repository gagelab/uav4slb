#!/usr/bin/env python3
"""
scripts/compile_cv0_predictions.py
===================================
Compile the three per-fold CV0 test-prediction CSVs into a single combined
CSV enriched with per-image metadata, for consumption by the figure scripts
under figures/ and figures_defense/ (fig04, fig05, fig06, fig08, fig09, ...).

Why this script exists
-----------------------
train_cv0.py and predict_cv0.py each write one prediction CSV per fold
(image_filename, actual, predicted, signed_error, abs_error). Several figure
scripts need all three folds pooled into one file, additionally tagged with
which model/fold produced each row and joined with the per-image
date/field/plot metadata already present in the CV0 split CSVs. This script
performs that join + concat once, rather than duplicating it in every figure
script.

Combining step
--------------
For each fold, the per-fold prediction CSV
    <experiment_dir>/predictions/<fold>_test_predictions.csv
is joined on image_filename with that fold's test split CSV
    <splits_dir>/cv0/<fold>/test.csv
        (plot, field, year, date, score, flight_date, image_filename)
to recover the plot/field/flight-date metadata (the "date" column below is
the UAV flight date, not the in-field visual rating date), tagged with
model and fold, and concatenated across all three folds.

This can run against predictions written by either train_cv0.py or
predict_cv0.py — it only needs the standard <fold>_test_predictions.csv
files to already exist in <experiment_dir>/predictions/.

Usage
-----
    python scripts/compile_cv0_predictions.py --config configs/eva02_base_cv0.yaml

Output
------
    <cfg.cv.output_dir>/predictions/cv0_combined_predictions.csv
    Columns: model, fold, test_year, image_filename, date, field, plot,
             actual, predicted, signed_error, abs_error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CV0_FOLDS = ["fold_2023", "fold_2024", "fold_2025"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compile per-fold CV0 test predictions into one combined CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", required=True,
        help="Path to model config YAML (e.g. configs/eva02_base_cv0.yaml).",
    )
    p.add_argument(
        "--experiment-dir", default=None,
        help="Override cfg.cv.output_dir (directory containing predictions/).",
    )
    p.add_argument(
        "--splits-dir", default=None,
        help="Override cfg.cv.splits_dir (directory containing cv0/<fold>/test.csv).",
    )
    p.add_argument(
        "--output", default=None,
        help=(
            "Output CSV path. Default: "
            "<experiment_dir>/predictions/cv0_combined_predictions.csv"
        ),
    )
    return p.parse_args()


def compile_predictions(
    experiment_dir: Path,
    splits_dir:     Path,
    model_name:     str,
) -> pd.DataFrame:
    """Join each fold's predictions with its test-split metadata and concat."""
    preds_dir = experiment_dir / "predictions"
    fold_frames = []

    for fold_name in CV0_FOLDS:
        pred_path = preds_dir / f"{fold_name}_test_predictions.csv"
        test_path = splits_dir / "cv0" / fold_name / "test.csv"
        for path in (pred_path, test_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}.\n"
                    "Run train_cv0.py (or predict_cv0.py) for predictions, and "
                    "create_cv_splits.py for splits, first."
                )

        preds = pd.read_csv(pred_path)

        # plot is zero-padded to 5 digits in image_filename (e.g. "_09565.jpg")
        # but read back as a bare int by pandas, so re-pad it here.
        meta = pd.read_csv(test_path, dtype={"plot": str})
        meta["plot"] = meta["plot"].str.zfill(5)
        # "date" in the combined output is the UAV flight date (matches the
        # date encoded in image_filename), not the in-field visual-rating date.
        meta["date"] = pd.to_datetime(
            meta["flight_date"], format="%m/%d/%y"
        ).dt.strftime("%Y-%m-%d")
        meta = meta[["image_filename", "date", "field", "plot"]]

        merged = preds.merge(meta, on="image_filename", how="left", validate="one_to_one")
        missing = merged["date"].isna()
        if missing.any():
            raise ValueError(
                f"{fold_name}: {int(missing.sum())} predicted image(s) have no "
                f"match in {test_path} "
                f"(e.g. {merged.loc[missing, 'image_filename'].iloc[0]!r})."
            )

        merged.insert(0, "test_year", int(fold_name.split("_")[1]))
        merged.insert(0, "fold", fold_name)
        merged.insert(0, "model", model_name)
        fold_frames.append(merged)

    combined = pd.concat(fold_frames, ignore_index=True)
    return combined[[
        "model", "fold", "test_year", "image_filename", "date", "field", "plot",
        "actual", "predicted", "signed_error", "abs_error",
    ]]


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(Path(args.config))

    model_name     = str(cfg.model.name)
    experiment_dir = Path(args.experiment_dir) if args.experiment_dir else Path(cfg.cv.output_dir)
    splits_dir     = Path(args.splits_dir)     if args.splits_dir     else Path(cfg.cv.splits_dir)
    output_path    = (
        Path(args.output) if args.output
        else experiment_dir / "predictions" / "cv0_combined_predictions.csv"
    )

    combined = compile_predictions(experiment_dir, splits_dir, model_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(
        f"Saved {len(combined)} rows across {combined['fold'].nunique()} folds "
        f"→ {output_path}"
    )


if __name__ == "__main__":
    main()
