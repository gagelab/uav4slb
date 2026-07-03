#!/usr/bin/env python
"""
scripts/predict_cv0.py
======================
Generate predictions from trained CV0 model checkpoints.

Two operating modes
-------------------
1. **Reproduce paper results** (default)
   Loads the best-checkpoint for each CV0 fold (fold_2023, fold_2024, fold_2025)
   and runs inference on the *same test split CSVs* used during training.
   Outputs are written to:
       results/<model_name>_cv0/predictions/<fold>_test_predictions.csv

   Combined predictions across all folds are written to:
       results/<model_name>_cv0/predictions/cv0_all_folds_predictions.csv

2. **Inference on new images** (--image-dir / --labels-csv flags)
   If ``--image-dir`` is provided, the test split CSVs are replaced by
   ``--labels-csv`` (or the built-in full_dataset.csv) and predictions are
   generated for every image in that CSV that can be matched to a file in
   ``--image-dir``.  In this mode ``--fold`` must specify which fold's
   checkpoint to use (the model trained on that fold's train/val split).

Usage examples
--------------
# Reproduce paper results for EVA-02-B (all three folds):
python scripts/predict_cv0.py \
    --config configs/eva02_base_cv0.yaml \
    --weights-dir results/eva02_base_cv0/checkpoints

# Single fold only:
python scripts/predict_cv0.py \
    --config configs/eva02_base_cv0.yaml \
    --weights-dir results/eva02_base_cv0/checkpoints \
    --fold fold_2025

# Inference on new images using fold_2025 checkpoint:
python scripts/predict_cv0.py \
    --config configs/eva02_base_cv0.yaml \
    --weights-dir results/eva02_base_cv0/checkpoints \
    --fold fold_2025 \
    --image-dir /path/to/new/plot_images \
    --labels-csv /path/to/new_labels.csv \
    --output predictions_new_images.csv

Checkpoint convention
---------------------
This script expects one checkpoint file per fold, named:
    <weights-dir>/fold_2023/checkpoint_best.pt
    <weights-dir>/fold_2024/checkpoint_best.pt
    <weights-dir>/fold_2025/checkpoint_best.pt

These are written automatically by train_cv0.py.

Output columns
--------------
image_filename, actual (NaN for unlabelled inference), predicted,
signed_error (NaN for unlabelled), abs_error (NaN for unlabelled),
fold, model
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure the project root is on PYTHONPATH when called from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import UAVDataset
from src.data.augmentation import build_transforms
from src.models import MODEL_REGISTRY
from src.utils.config import load_config
from src.utils.reproducibility import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CV0 fold definitions (test year, fold directory name)
# ---------------------------------------------------------------------------
CV0_FOLDS = [
    ("fold_2023", 2023),
    ("fold_2024", 2024),
    ("fold_2025", 2025),
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate predictions from CV0 best checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", required=True,
        help="Path to architecture YAML config (e.g. configs/eva02_base_cv0.yaml).",
    )
    p.add_argument(
        "--weights-dir", required=True,
        help=(
            "Directory containing fold subdirectories with checkpoint_best.pt.  "
            "Typically results/<model>_cv0/checkpoints."
        ),
    )
    p.add_argument(
        "--fold", default=None,
        choices=["fold_2023", "fold_2024", "fold_2025"],
        help="Run a single fold only.  Omit to run all three folds.",
    )
    p.add_argument(
        "--splits-dir", default=None,
        help=(
            "Override cfg.cv.splits_dir.  Defaults to the value in the YAML config.  "
            "Set to 'data/cv_splits' if running from the repo root."
        ),
    )
    # ---- New-image inference mode ----
    p.add_argument(
        "--image-dir", default=None,
        help=(
            "Path to directory containing plot-level image tiles.  "
            "If provided, run inference on images matched to --labels-csv "
            "instead of the published test split."
        ),
    )
    p.add_argument(
        "--labels-csv", default=None,
        help=(
            "CSV with at least an 'image_filename' column.  "
            "Required when --image-dir is set.  "
            "A 'score' column is used as ground truth if present."
        ),
    )
    p.add_argument(
        "--output", default=None,
        help=(
            "Path for the output predictions CSV (new-image mode only).  "
            "Defaults to <weights-dir>/predictions_new_images.csv."
        ),
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Override inference batch size from config.",
    )
    p.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker processes.",
    )
    p.add_argument(
        "--device", default=None,
        help="Torch device string (e.g. 'cuda:0', 'cpu').  Auto-detected if omitted.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a forward pass over all batches in *loader* and collect predictions.

    Parameters
    ----------
    model   : Model already loaded with best-checkpoint weights.
    loader  : DataLoader yielding (images, targets) batches.
    device  : Torch device.

    Returns
    -------
    preds     : float32 array of shape (N,) — model predictions
    targets   : float32 array of shape (N,) — ground-truth labels (NaN if absent)
    """
    model.eval()
    all_preds, all_targets = [], []

    for images, targets in tqdm(loader, desc="  inference", leave=False):
        images = images.to(device, non_blocking=True)
        preds  = model(images).float().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.numpy().tolist())

    return (
        np.array(all_preds,   dtype=np.float32),
        np.array(all_targets, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Per-fold prediction
# ---------------------------------------------------------------------------

def predict_fold(
    fold_name: str,
    cfg,
    weights_dir: Path,
    splits_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
    image_dir_override: Optional[Path] = None,
    labels_csv_override: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load the best checkpoint for *fold_name* and generate predictions.

    In **reproduce mode** (image_dir_override is None), the test split CSV
    from data/cv_splits/cv0/<fold_name>/test.csv is used.

    In **inference mode**, labels_csv_override replaces the test split and
    image_dir_override replaces cfg.data.image_dir.

    Parameters
    ----------
    fold_name           : e.g. 'fold_2025'
    cfg                 : OmegaConf config node
    weights_dir         : Root checkpoint directory; checkpoint at
                          <weights_dir>/<fold_name>/checkpoint_best.pt
    splits_dir          : Root of cv_splits (parent of cv0/)
    device              : Torch device
    args                : Parsed CLI arguments
    image_dir_override  : Optional override for image directory (new images)
    labels_csv_override : Optional override for labels CSV (new images)

    Returns
    -------
    pd.DataFrame with columns:
        image_filename, actual, predicted, signed_error, abs_error, fold, model
    """
    ckpt_path = weights_dir / fold_name / "checkpoint_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run train_cv0.py first, or verify --weights-dir points to the "
            f"directory that contains fold_2023/, fold_2024/, fold_2025/ subdirectories."
        )

    logger.info(f"[{fold_name}] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # ---- Build model from registry ------------------------------------------
    model_name = cfg.model.name
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'.  "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    model = MODEL_REGISTRY[model_name](
        pretrained=False,          # Weights come from checkpoint, not timm hub
        dropout_rate=cfg.model.dropout_rate,
        freeze_backbone=False,
    ).to(device)

    # Load weights; strip DataParallel prefix if present
    state = ckpt["model_state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)

    logger.info(
        f"[{fold_name}] Checkpoint epoch={ckpt.get('epoch', '?')}  "
        f"val_r2={ckpt.get('val_r2', float('nan')):.4f}"
    )

    # ---- Build dataset / dataloader -----------------------------------------
    image_size = tuple(cfg.data.image_size)          # (H, W)
    mean       = cfg.data.normalize.mean
    std        = cfg.data.normalize.std

    # Inference-time transforms: resize + normalize only (no augmentation)
    transforms = build_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        augment=False,
    )

    _tmp_csv: Optional[Path] = None

    if image_dir_override is not None:
        # New-image inference mode
        image_dir = image_dir_override
        labels_df = pd.read_csv(labels_csv_override)
        has_labels = cfg.data.target_column in labels_df.columns
        if not has_labels:
            # UAVDataset requires the target column to exist; inject a NaN placeholder
            labels_df = labels_df.copy()
            labels_df[cfg.data.target_column] = float("nan")
            _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
            labels_df.to_csv(_tmp.name, index=False)
            _tmp.close()
            _tmp_csv = Path(_tmp.name)
            labels_csv_path = _tmp_csv
        else:
            labels_csv_path = labels_csv_override
    else:
        # Reproduce mode: use published test split
        image_dir = Path(cfg.data.image_dir)
        split_csv = splits_dir / "cv0" / fold_name / "test.csv"
        if not split_csv.exists():
            raise FileNotFoundError(
                f"Test split CSV not found: {split_csv}\n"
                f"Run scripts/create_cv_splits.py first."
            )
        labels_csv_path = split_csv
        has_labels = True

    # Determine batch size: CLI override > config val batch size
    batch_size = args.batch_size or cfg.data.dataloader.batch_size_val

    dataset = UAVDataset(
        data_dir=str(image_dir),
        labels_csv=str(labels_csv_path),
        image_filename_column=cfg.data.image_filename_column,
        target_column=cfg.data.target_column,
        transform=transforms,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,           # Preserve order for CSV alignment
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    logger.info(
        f"[{fold_name}] Running inference on {len(dataset)} images "
        f"(batch_size={batch_size})"
    )

    preds, targets = run_inference(model, loader, device)
    filenames = dataset.df[cfg.data.image_filename_column].tolist()

    if _tmp_csv is not None:
        _tmp_csv.unlink(missing_ok=True)

    # ---- Assemble output DataFrame ------------------------------------------
    signed_error = (preds - targets) if has_labels else np.full_like(preds, np.nan)
    abs_errors   = np.abs(signed_error) if has_labels else np.full_like(preds, np.nan)
    actual_col   = targets              if has_labels else np.full_like(preds, np.nan)

    out_df = pd.DataFrame({
        "image_filename": filenames,
        "actual":         actual_col,
        "predicted":      preds,
        "signed_error":   signed_error,   # predicted − actual
        "abs_error":      abs_errors,
        "fold":           fold_name,
        "model":          cfg.model.name,
    })

    return out_df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    # ---- Load config ---------------------------------------------------------
    cfg = load_config(args.config)
    logger.info(f"Config: {args.config}  |  model: {cfg.model.name}")

    # ---- Device selection ----------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU found — running on CPU (will be slow).")

    # ---- Resolve directories -------------------------------------------------
    weights_dir = Path(args.weights_dir)
    if not weights_dir.exists():
        raise FileNotFoundError(f"--weights-dir not found: {weights_dir}")

    splits_dir = Path(args.splits_dir) if args.splits_dir else Path(cfg.cv.splits_dir)

    # Default output root: inside the results directory.
    # cfg.cv.output_dir is set to "results/<model>_cv0" in each YAML config.
    exp_dir  = Path(cfg.cv.output_dir)
    pred_dir = exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # ---- Validate new-image inference mode arguments -------------------------
    image_dir_override  = Path(args.image_dir)  if args.image_dir  else None
    labels_csv_override = Path(args.labels_csv) if args.labels_csv else None

    if image_dir_override is not None:
        if labels_csv_override is None:
            raise ValueError(
                "--labels-csv is required when --image-dir is provided."
            )
        if args.fold is None:
            raise ValueError(
                "--fold is required in new-image inference mode "
                "(which fold's checkpoint should be used?)."
            )
        if not image_dir_override.exists():
            raise FileNotFoundError(f"--image-dir not found: {image_dir_override}")
        if not labels_csv_override.exists():
            raise FileNotFoundError(f"--labels-csv not found: {labels_csv_override}")

    # ---- Determine which folds to run ----------------------------------------
    if args.fold:
        folds_to_run = [(args.fold, int(args.fold.split("_")[1]))]
    else:
        folds_to_run = CV0_FOLDS

    # ---- Run prediction per fold ---------------------------------------------
    all_fold_dfs = []

    for fold_name, _ in folds_to_run:
        logger.info(f"\n{'='*60}\nFold: {fold_name}\n{'='*60}")
        fold_df = predict_fold(
            fold_name           = fold_name,
            cfg                 = cfg,
            weights_dir         = weights_dir,
            splits_dir          = splits_dir,
            device              = device,
            args                = args,
            image_dir_override  = image_dir_override,
            labels_csv_override = labels_csv_override,
        )

        # Save per-fold CSV (reproduce mode only; new-image mode gets single output)
        if image_dir_override is None:
            fold_out = pred_dir / f"{fold_name}_test_predictions.csv"
            fold_df.to_csv(fold_out, index=False)
            logger.info(f"Saved {len(fold_df)} predictions → {fold_out}")

        all_fold_dfs.append(fold_df)

    # ---- Combine and save combined CSV --------------------------------------
    combined_df = pd.concat(all_fold_dfs, ignore_index=True)

    if image_dir_override is not None:
        # New-image mode: single output file at user-specified path or default
        out_path = (
            Path(args.output) if args.output
            else weights_dir / "predictions_new_images.csv"
        )
        combined_df.to_csv(out_path, index=False)
        logger.info(f"\nSaved {len(combined_df)} predictions → {out_path}")
    else:
        # Reproduce mode: combined across all folds
        combined_out = pred_dir / "cv0_all_folds_predictions.csv"
        combined_df.to_csv(combined_out, index=False)
        logger.info(f"\nCombined predictions ({len(combined_df)} rows) → {combined_out}")

        # Print aggregate metrics if labels are present
        if combined_df["actual"].notna().all():
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            r2   = r2_score(combined_df["actual"], combined_df["predicted"])
            rmse = mean_squared_error(combined_df["actual"], combined_df["predicted"]) ** 0.5
            mae  = mean_absolute_error(combined_df["actual"], combined_df["predicted"])
            logger.info(
                f"\nAggregate CV0 metrics (all folds combined)\n"
                f"  R²   = {r2:.4f}\n"
                f"  RMSE = {rmse:.4f}\n"
                f"  MAE  = {mae:.4f}\n"
            )


if __name__ == "__main__":
    main()
