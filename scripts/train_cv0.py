#!/usr/bin/env python3
"""
train_cv0.py
============
Unified CV0 (leave-one-year-out) training script for all nine architectures.

This script iterates over the three CV0 fold directories produced by
create_cv_splits.py (one held-out year each: 2023, 2024, 2025), trains a
model on each fold, and evaluates on the held-out year's test set.  All nine
architectures are dispatched through a central MODEL_REGISTRY so that any
model can be trained by changing only the --config argument.

Architectures
-------------
  CNN:    EfficientNet V2-S, ConvNeXt V2-B, ConvNeXt V2-L
  ViT:    DINOv2 ViT-S/14, DINOv2 ViT-B/14, EVA-02-Base
  Hybrid: MaxViT-S, SwinV2-B, CoAtNet-2

Checkpoint modes
----------------
  Default
    Fresh training from pretrained ImageNet weights.

  --resume <path>
    Fully restore an interrupted training run from a checkpoint saved by this
    script.  Loads model weights, optimizer state, scheduler state, AMP
    GradScaler state, epoch counter, and patience counter.  Intended for use
    when a job is interrupted mid-fold (e.g. cluster preemption).  The split
    CSV must be identical to the one used when the checkpoint was saved; the
    split hash stored in the checkpoint is verified before resuming to prevent
    silent data leakage.

  --warm-start <path>
    Load model weights only; the optimizer, scheduler, and scaler are freshly
    initialised from the config.  Intended for incremental retraining on an
    expanded dataset (new field season appended to the labels CSV).  Because
    the data distribution has changed, inheriting old optimizer momentum
    buffers would misdirect early gradient updates.  Lower learning rates
    relative to the original run are recommended; see eva02_base_retrain.yaml.

  Only --resume or --warm-start may be specified at once, not both.

AMP dtype
---------
  Default: bfloat16.  bfloat16 has the same exponent range as float32 and
  does not require loss scaling, so GradScaler is a no-op and is disabled.
  Use --amp-dtype float16 on pre-Ampere GPUs that lack bfloat16 hardware
  support; float16 requires GradScaler to prevent underflow.

Optimizer / param groups
------------------------
  All factories expose model.get_optimizer_param_groups(backbone_lr, head_lr,
  weight_decay).  DINOv2 variants return 2 groups (backbone, head); all other
  models return 4 groups (backbone_decay, backbone_nodecay, head_decay,
  head_nodecay).  The LR logging helper introspects param group count at
  runtime so the correct head-group index is used automatically.

Image-size mod-32 constraint
-----------------------------
  CoAtNet-2, MaxViT-S, and SwinV2-B have five stride-2 downsampling stages,
  requiring image_size to be a multiple of 32.  The script enforces this with
  an assertion at startup when --image-size is overridden via the CLI.

Signed error convention
-----------------------
  signed_error = predicted − actual throughout (consistent with evaluate_cv0.py
  and the error analysis pipeline).

Outputs (written to <cfg.cv.output_dir>/)
-----------------------------------------
  checkpoints/<fold_name>/checkpoint_best.pt
  checkpoints/<fold_name>/checkpoint_latest.pt
  predictions/<fold_name>_test_predictions.csv
  history/<fold_name>_history.csv
  logs/<fold_name>.log
  tensorboard/<fold_name>/
  cv_results.csv          (fold-level summary table)
  summary.json            (aggregate mean ± std across folds)
  config_resolved.yaml    (full resolved config for reproducibility)

Usage
-----
  # Train any architecture from scratch
  python scripts/train_cv0.py --config configs/eva02_base_cv0.yaml

  # Resume an interrupted run (same data, same splits)
  python scripts/train_cv0.py --config configs/eva02_base_cv0.yaml \\
      --fold fold_2025 \\
      --resume experiments/eva02_base_cv0/checkpoints/fold_2025/checkpoint_latest.pt

  # Warm-start on expanded dataset (new season, lower LRs)
  python scripts/train_cv0.py --config configs/eva02_base_retrain.yaml \\
      --warm-start experiments/eva02_base_cv0/checkpoints/fold_2025/checkpoint_best.pt

  # Smoke-test: one fold, two epochs
  python scripts/train_cv0.py --config configs/eva02_base_cv0.yaml \\
      --max-folds 1 --max-epochs 2
"""

import argparse
import hashlib
import importlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import mean_absolute_error, r2_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Project path bootstrap — allows running from any working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import create_train_augmentation, create_val_augmentation
from src.data.dataset import UAVDataset
from src.utils.reproducibility import create_generator, seed_worker, set_seed

# Root logger: INFO to console with timestamps.  Per-fold file handlers are
# added and removed inside train_fold() so each fold also gets its own log.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_cv0")


# ===========================================================================
# Model registry
# ===========================================================================

@dataclass(frozen=True)
class ModelEntry:
    """Static metadata describing one registered architecture.

    Attributes
    ----------
    module          Dotted import path within src/models/
    factory         Name of the factory function in that module
    requires_mod32  When True, image_size must be divisible by 32.
                    CoAtNet-2, MaxViT-S, and SwinV2-B each have exactly five
                    stride-2 stages, giving a 32× spatial reduction from
                    input to feature map.  Passing an image size not divisible
                    by 32 produces asymmetric feature maps and misaligned
                    positional embeddings.
    default_config  Path (relative to project root) to the default YAML config
    """
    module:         str
    factory:        str
    requires_mod32: bool
    default_config: str


# Maps cfg.model.name → ModelEntry.
# Add a row here when introducing a new architecture; everything else
# (optimizer groups, scheduler, AMP, logging) is handled generically.
MODEL_REGISTRY: Dict[str, ModelEntry] = {
    "coatnet2": ModelEntry(
        module="src.models.coatnet2",
        factory="create_coatnet2",
        requires_mod32=True,
        default_config="configs/coatnet2_cv0.yaml",
    ),
    "convnextv2_base": ModelEntry(
        module="src.models.convnextv2",
        factory="create_convnextv2_base",
        requires_mod32=False,
        default_config="configs/convnextv2_cv0.yaml",
    ),
    "convnextv2_large": ModelEntry(
        module="src.models.convnextv2_large",
        factory="create_convnextv2_large",
        requires_mod32=False,
        default_config="configs/convnextv2_large_cv0.yaml",
    ),
    "dinov2_vitb14": ModelEntry(
        module="src.models.dinov2_vitb14",
        factory="create_dinov2_vitb14",
        requires_mod32=False,
        default_config="configs/dinov2_cv0.yaml",
    ),
    "dinov2_vits14": ModelEntry(
        module="src.models.dinov2_vits14",
        factory="create_dinov2_vits14",
        requires_mod32=False,
        default_config="configs/dinov2_vits14_cv0.yaml",
    ),
    "efficientnetv2_s": ModelEntry(
        module="src.models.efficientnetv2s",
        factory="create_efficientnetv2s",
        requires_mod32=False,
        default_config="configs/efficientnetv2_s_cv0.yaml",
    ),
    "eva02_base": ModelEntry(
        module="src.models.eva02_base",
        factory="create_eva02_base",
        requires_mod32=False,
        default_config="configs/eva02_base_cv0.yaml",
    ),
    "maxvit_small": ModelEntry(
        module="src.models.maxvit_small",
        factory="create_maxvit_small",
        requires_mod32=True,
        default_config="configs/maxvit_small_cv0.yaml",
    ),
    "swinv2_base": ModelEntry(
        module="src.models.swinv2_base",
        factory="create_swinv2_base",
        requires_mod32=True,
        default_config="configs/swinv2_base_cv0.yaml",
    ),
}


def load_model(model_name: str, cfg) -> nn.Module:
    """Dynamically import and instantiate a model from the registry.

    All registered factory functions must accept these keyword arguments:
      pretrained      : bool
      dropout_rate    : float
      freeze_backbone : bool

    Using dynamic import (importlib) rather than static imports keeps this
    script independent of architecture-specific dependencies; models not
    installed in the environment simply will not be in the registry.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'.  "
            f"Registered names: {sorted(MODEL_REGISTRY)}"
        )
    entry   = MODEL_REGISTRY[model_name]
    module  = importlib.import_module(entry.module)
    factory = getattr(module, entry.factory)
    return factory(
        pretrained=cfg.model.pretrained,
        dropout_rate=cfg.model.dropout_rate,
        freeze_backbone=cfg.model.freeze_backbone,
    )


# ===========================================================================
# GPU selection
# ===========================================================================

def select_gpu(gpu_id: Optional[int] = None) -> torch.device:
    """Return the target device.

    When gpu_id is None the function scans all visible GPUs and selects the
    one with the most free VRAM.  This avoids manually tracking which GPU is
    in use when running multiple sequential experiments.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        return torch.device("cpu")

    if gpu_id is not None:
        n = torch.cuda.device_count()
        assert gpu_id < n, f"GPU {gpu_id} requested but only {n} device(s) visible."
        logger.info(f"Using specified GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        return torch.device(f"cuda:{gpu_id}")

    # Auto-select: iterate all GPUs and pick the one with the most free bytes
    best_id, best_free = 0, 0
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        # memory_reserved includes the allocator cache; subtracting it from
        # total gives a conservative estimate of actually available memory.
        free = props.total_memory - torch.cuda.memory_reserved(i)
        logger.info(
            f"  GPU {i}: {props.name} — "
            f"{free / 1e9:.1f} GB free / {props.total_memory / 1e9:.1f} GB total"
        )
        if free > best_free:
            best_id, best_free = i, free

    logger.info(f"Auto-selected GPU {best_id}: {torch.cuda.get_device_name(best_id)}")
    return torch.device(f"cuda:{best_id}")


# ===========================================================================
# Checkpoint utilities
# ===========================================================================

def _hash_dataframe(df: pd.DataFrame) -> str:
    """Compute an MD5 fingerprint of a DataFrame for split integrity checking.

    Rows are sorted by image_filename before hashing so that row-order
    differences between regenerated splits do not produce false mismatches.
    The hash is stored inside every checkpoint and verified on --resume to
    guard against accidentally resuming a run on a different (e.g. expanded)
    dataset, which would silently introduce train/test leakage.

    This check is intentionally skipped for --warm-start runs because the
    whole point of warm-start is that the data has changed.

    Parameters
    ----------
    df : pd.DataFrame
        The fold DataFrame (train, val, or test) to fingerprint.

    Returns
    -------
    str
        Lowercase hex MD5 digest.
    """
    sort_col = "image_filename" if "image_filename" in df.columns else df.columns[0]
    canonical_csv = df.sort_values(sort_col).reset_index(drop=True).to_csv(index=False)
    return hashlib.md5(canonical_csv.encode()).hexdigest()


def save_checkpoint(
    path:           Path,
    epoch:          int,
    model:          nn.Module,
    optimizer:      torch.optim.Optimizer,
    scheduler,
    scaler:         GradScaler,
    best_val_r2:    float,
    best_val_loss:  float,
    patience_ctr:   int,
    fold_name:      str,
    cfg,
    train_df:       pd.DataFrame,
) -> None:
    """Persist a full training state checkpoint.

    Stores everything required to resume an interrupted run exactly where it
    left off, including optimizer and scheduler state, AMP scaler state, early
    stopping counters, and a data-integrity hash of the training split.

    Parameters
    ----------
    path            Destination .pt file path.
    epoch           Last *completed* epoch (1-indexed, matching the training loop).
    model           The model being trained.
    optimizer       AdamW optimizer with per-layer param groups.
    scheduler       SequentialLR (warmup + cosine) instance.
    scaler          GradScaler (may be disabled for bfloat16 runs).
    best_val_r2     Best validation R² seen so far (for checkpoint selection).
    best_val_loss   Corresponding validation MSE loss.
    patience_ctr    Early stopping patience counter at time of save.
    fold_name       Name of the current fold (e.g. "fold_2025").
    cfg             OmegaConf config node (serialised for reproducibility).
    train_df        Training split DataFrame — hashed for resume integrity.
    """
    torch.save(
        {
            # ── Epoch bookkeeping ──────────────────────────────────────────
            # epoch is 1-indexed (last completed); resume starts from epoch+1.
            "epoch":            epoch,

            # ── Model weights ─────────────────────────────────────────────
            "model_state_dict": model.state_dict(),

            # ── Optimizer, scheduler, AMP scaler ─────────────────────────
            # All three must be restored together for --resume to produce
            # identical numerical behaviour.  The scaler is a no-op for
            # bfloat16 but is saved anyway for consistency.
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "scaler_state":     scaler.state_dict(),

            # ── Early stopping state ───────────────────────────────────────
            "best_val_r2":      best_val_r2,
            "best_val_loss":    best_val_loss,
            "patience_ctr":     patience_ctr,

            # ── Fold and config metadata ───────────────────────────────────
            "fold":             fold_name,
            "config":           OmegaConf.to_container(cfg, resolve=True),

            # ── Data integrity fingerprint ────────────────────────────────
            # MD5 of the training split DataFrame (sorted by image_filename).
            # Verified on --resume to catch accidental data substitution.
            # Not checked on --warm-start (data is intentionally different).
            "train_split_hash": _hash_dataframe(train_df),
        },
        path,
    )


def load_checkpoint_for_resume(
    ckpt_path:  Path,
    model:      nn.Module,
    optimizer:  torch.optim.Optimizer,
    scheduler,
    scaler:     GradScaler,
    train_df:   pd.DataFrame,
    device:     torch.device,
) -> Tuple[int, float, float, int]:
    """Restore full training state from a checkpoint for --resume.

    Loads model weights, optimizer state, scheduler state, AMP scaler state,
    and early stopping counters.  Verifies the stored split hash against the
    current training DataFrame to guard against accidental data substitution.

    Parameters
    ----------
    ckpt_path   Path to the checkpoint .pt file saved by save_checkpoint().
    model       Model instance (must match the checkpoint architecture exactly).
    optimizer   AdamW optimizer — state will be overwritten in-place.
    scheduler   SequentialLR scheduler — state will be overwritten in-place.
    scaler      GradScaler — state will be overwritten in-place.
    train_df    Current training split DataFrame for hash verification.
    device      Target device (checkpoint is first loaded to CPU to avoid
                memory duplication on the GPU before transfer).

    Returns
    -------
    Tuple of (start_epoch, best_val_r2, best_val_loss, patience_ctr) where
    start_epoch is the next epoch to run (last completed epoch + 1).

    Raises
    ------
    RuntimeError
        If the stored split hash does not match the current training DataFrame.
        This indicates the data has changed since the checkpoint was saved,
        and resuming would produce results that are not comparable to the
        original run.  Use --warm-start instead when adding new data.
    """
    logger.info(f"[--resume] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # ── Data integrity check ──────────────────────────────────────────────────
    # Compute the MD5 of the current training split and compare to the hash
    # stored in the checkpoint.  A mismatch means the labels CSV or split
    # generation script has changed, which would silently break comparability.
    stored_hash  = ckpt.get("train_split_hash", None)
    current_hash = _hash_dataframe(train_df)
    if stored_hash is not None and stored_hash != current_hash:
        raise RuntimeError(
            "\n\n[--resume] Split hash mismatch — the training data has changed "
            "since this checkpoint was saved.\n"
            f"  Stored hash  : {stored_hash}\n"
            f"  Current hash : {current_hash}\n\n"
            "This means the labels CSV or CV split files were regenerated after "
            "the checkpoint was written.  Resuming on different data would "
            "produce results that are not comparable to the original run and "
            "may silently introduce train/test leakage.\n\n"
            "Options:\n"
            "  1. Restore the original split CSV and retry --resume.\n"
            "  2. If you have added new data, use --warm-start instead of "
            "--resume (see the module docstring for details)."
        )
    if stored_hash is None:
        logger.warning(
            "[--resume] Checkpoint does not contain a split hash "
            "(saved by an older version of this script).  "
            "Data integrity cannot be verified — proceeding anyway."
        )

    # ── Restore model, optimizer, scheduler, scaler ───────────────────────────
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    optimizer.load_state_dict(ckpt["optimizer_state"])
    # Move optimizer tensors (momentum buffers etc.) to the correct device.
    # PyTorch does not do this automatically when loading from CPU.
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])

    # ── Restore training progress ─────────────────────────────────────────────
    last_epoch   = int(ckpt["epoch"])          # last completed epoch (1-indexed)
    start_epoch  = last_epoch + 1              # next epoch to run
    best_val_r2  = float(ckpt["best_val_r2"])
    best_val_loss = float(ckpt["best_val_loss"])
    patience_ctr = int(ckpt["patience_ctr"])

    logger.info(
        f"[--resume] Restored from epoch {last_epoch}  |  "
        f"best_val_r2={best_val_r2:.4f}  patience_ctr={patience_ctr}  "
        f"fold={ckpt.get('fold', 'unknown')}"
    )
    return start_epoch, best_val_r2, best_val_loss, patience_ctr


def load_checkpoint_for_warm_start(
    ckpt_path: Path,
    model:     nn.Module,
    device:    torch.device,
) -> None:
    """Load model weights only from a checkpoint for --warm-start retraining.

    This is the correct loading mode when retraining on an expanded dataset
    (e.g. a new field season appended to the labels CSV).  The optimizer,
    scheduler, and scaler are NOT loaded — they remain freshly initialised
    from the config so that their internal state reflects the new data
    distribution rather than the old gradient landscape.

    Why not load optimizer state?
    ─────────────────────────────
    AdamW maintains per-parameter first- and second-moment estimates (m_t, v_t)
    that encode the gradient history of the previous training run.  When the
    data distribution shifts (new images, new year), these moments point toward
    a loss landscape that no longer exists.  Loading them would misdirect early
    gradient updates in the new run and typically produces worse convergence
    than a fresh optimizer.

    Why initialise from model weights at all?
    ─────────────────────────────────────────
    The backbone is already adapted to the UAV + SLB domain after the original
    training run.  Starting from these weights rather than ImageNet weights
    places the model in a region of the loss surface that is much closer to the
    new optimum, substantially reducing the number of epochs needed to converge.
    This is often called a "warm start" or "fine-tuning from a domain-adapted
    checkpoint."

    Parameters
    ----------
    ckpt_path   Path to the checkpoint .pt file to load weights from.
    model       Model instance — weights will be overwritten in-place.
    device      Target device.

    Notes
    -----
    The checkpoint must have been saved by save_checkpoint() (keys
    "model_state_dict", "optimizer_state", ...).  If the checkpoint was saved
    by an older version that stored only "model_state_dict", it will still work
    because only that key is accessed here.
    """
    logger.info(f"[--warm-start] Loading model weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Log provenance metadata from the source checkpoint for traceability.
    source_epoch = ckpt.get("epoch", "unknown")
    source_fold  = ckpt.get("fold",  "unknown")
    source_r2    = ckpt.get("val_r2", ckpt.get("best_val_r2", float("nan")))
    logger.info(
        f"[--warm-start] Weights sourced from: "
        f"fold={source_fold}  epoch={source_epoch}  val_r2={source_r2:.4f}\n"
        f"  Optimizer, scheduler, and AMP scaler are freshly initialised "
        f"from the config (not loaded from checkpoint).\n"
        f"  This is correct for expanded-dataset retraining — see module "
        f"docstring for rationale."
    )


# ===========================================================================
# DataLoader construction
# ===========================================================================

# Temporary directory for fold CSV files.  UAVDataset requires a CSV path,
# so fold DataFrames are written here before being passed to the dataset class.
_TMP_FOLD_DIR = Path("/tmp/slb_cv0_folds")


def _make_transform(cfg, split: str):
    """Build an Albumentations transform pipeline from the config.

    Training splits receive the full augmentation pipeline; validation and
    test splits receive only the resize + normalise pipeline (no augmentation)
    to ensure deterministic evaluation.
    """
    image_size = tuple(cfg.data.image_size)
    mean       = list(cfg.data.normalize.mean)
    std        = list(cfg.data.normalize.std)

    # The augmentation config is passed as an OmegaConf node; convert to a
    # plain dict so the augmentation factory can consume it directly.
    aug_cfg = OmegaConf.create({
        "preprocessing": {
            "image_size": list(image_size),
            "normalize":  {"mean": mean, "std": std},
        },
        "augmentation": OmegaConf.to_container(cfg.augmentation, resolve=True),
    })

    if split == "train":
        return create_train_augmentation(aug_cfg, image_size, mean, std)
    return create_val_augmentation(image_size, mean, std)


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    cfg,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Materialise fold DataFrames as CSV files and wrap them in DataLoaders.

    UAVDataset expects a labels CSV path rather than an in-memory DataFrame,
    so fold CSVs are written to a temporary directory and cleaned up between
    runs.  The test DataLoader deliberately uses the val transform (no
    augmentation) to ensure deterministic test-set evaluation.

    The training DataLoader uses a seeded Generator and seed_worker to produce
    fully reproducible data order across restarts.
    """
    _TMP_FOLD_DIR.mkdir(parents=True, exist_ok=True)

    def _write_tmp(df: pd.DataFrame, tag: str) -> str:
        """Write a fold DataFrame to a temporary CSV and return its path."""
        path = _TMP_FOLD_DIR / f"{tag}_{seed}.csv"
        df.to_csv(path, index=False)
        return str(path)

    image_dir  = cfg.data.image_dir
    img_col    = cfg.data.image_filename_column
    target_col = cfg.data.target_column
    img_ext    = cfg.data.image_extension
    image_size = tuple(cfg.data.image_size)
    dl_cfg     = cfg.data.dataloader

    def _make_dataset(csv_path: str, split: str) -> UAVDataset:
        return UAVDataset(
            data_dir=image_dir,
            labels_csv=csv_path,
            image_filename_column=img_col,
            target_column=target_col,
            task_type="regression",
            image_size=image_size,
            transform=_make_transform(cfg, split),
            image_extension=img_ext,
        )

    train_ds = _make_dataset(_write_tmp(train_df, "train"), "train")
    val_ds   = _make_dataset(_write_tmp(val_df,   "val"),   "val")
    test_ds  = _make_dataset(_write_tmp(test_df,  "test"),  "val")  # no augment for test

    # Seeded generator ensures reproducible sample order on the training loader
    g       = create_generator(seed)
    val_bsz = dl_cfg.get("batch_size_val", dl_cfg.batch_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=dl_cfg.batch_size,
        shuffle=True,
        num_workers=dl_cfg.num_workers,
        pin_memory=dl_cfg.pin_memory,
        worker_init_fn=seed_worker,   # seeds each worker deterministically
        generator=g,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_bsz, shuffle=False,
        num_workers=dl_cfg.num_workers, pin_memory=dl_cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=val_bsz, shuffle=False,
        num_workers=dl_cfg.num_workers, pin_memory=dl_cfg.pin_memory,
    )
    return train_loader, val_loader, test_loader


# ===========================================================================
# Optimizer and learning-rate scheduler
# ===========================================================================

def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    """Construct an AdamW optimizer with per-layer learning rates.

    All model classes expose get_optimizer_param_groups(backbone_lr, head_lr,
    weight_decay) which returns either:
      2 groups (DINOv2 variants): backbone, head
      4 groups (all others):      backbone_decay, backbone_nodecay,
                                  head_decay, head_nodecay

    The head receives a higher LR than the backbone (typically 10×) to allow
    the newly-initialised regression head to fit quickly while the pre-trained
    backbone adapts more slowly.  No-decay groups exclude biases and
    LayerNorm / BatchNorm scale parameters from weight decay.
    """
    opt_cfg      = cfg.training.optimizer
    param_groups = model.get_optimizer_param_groups(
        backbone_lr=opt_cfg.backbone_lr,
        head_lr=opt_cfg.head_lr,
        weight_decay=opt_cfg.weight_decay,
    )
    return torch.optim.AdamW(
        param_groups,
        betas=tuple(opt_cfg.betas),
        eps=opt_cfg.eps,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg):
    """Construct a linear warm-up → cosine annealing learning-rate schedule.

    During the warm-up phase (first `warmup_epochs` epochs) the LR ramps
    linearly from near-zero to the configured base LR.  After warm-up,
    cosine annealing decays the LR to `min_lr` over the remaining epochs.

    SequentialLR splices these two schedulers at the warm-up milestone so
    that a single scheduler.step() call advances both phases correctly.
    """
    sched_cfg     = cfg.training.scheduler
    warmup_epochs = int(sched_cfg.get("warmup_epochs", 5))
    # T_max is the cosine annealing half-period; we use the remaining epochs
    # after warm-up so the cosine decay covers the full post-warmup training.
    t_max         = int(sched_cfg.get("T_max", cfg.training.max_epochs - warmup_epochs))
    min_lr        = float(sched_cfg.get("min_lr", 1e-7))

    def lr_lambda(epoch: int) -> float:
        """Linear ramp from 1/warmup_epochs → 1.0 over warmup_epochs steps."""
        return float(epoch + 1) / float(warmup_epochs) if epoch < warmup_epochs else 1.0

    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=min_lr
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


# ===========================================================================
# LR logging helper
# ===========================================================================

def _lr_backbone_head(optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    """Extract (backbone_lr, head_lr) regardless of the number of param groups.

    2-group layout (DINOv2 variants):
        param_groups[0] → backbone
        param_groups[1] → head

    4-group layout (all other architectures):
        param_groups[0] → backbone_decay
        param_groups[1] → backbone_nodecay
        param_groups[2] → head_decay       ← head LR lives here
        param_groups[3] → head_nodecay
    """
    n           = len(optimizer.param_groups)
    backbone_lr = optimizer.param_groups[0]["lr"]
    head_lr     = optimizer.param_groups[1 if n == 2 else 2]["lr"]
    return backbone_lr, head_lr


# ===========================================================================
# Evaluation
# ===========================================================================

@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    use_amp:   bool        = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    """Evaluate the model on a DataLoader and return a metrics dict.

    Loss is accumulated as a sample-weighted sum (loss × batch_size) and
    then divided by the total number of samples.  This matches the per-sample
    MSE convention and avoids any bias from a final batch that is smaller
    than the configured batch size.

    AMP is forwarded even during evaluation so inference numerics match the
    training forward pass exactly.

    Returns
    -------
    dict with keys: loss, rmse, r2, mae, preds (np.ndarray), targets (np.ndarray)
    """
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0

    for batch in loader:
        # Support both dict-style batches and tuple-style (image, target) batches
        if isinstance(batch, dict):
            images  = batch["image"].to(device)
            targets = batch["target"].to(device)
        else:
            images, targets = batch
            images  = images.to(device)
            targets = targets.to(device)

        with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            preds = model(images)
            loss  = criterion(preds, targets.float())

        # Accumulate sample-weighted loss for unbiased per-sample averaging
        total_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().float().numpy())
        all_targets.extend(targets.cpu().float().numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    n           = len(all_targets)
    avg_loss    = total_loss / n

    return {
        "loss":    avg_loss,
        "rmse":    float(np.sqrt(avg_loss)),        # RMSE from per-sample MSE
        "r2":      float(r2_score(all_targets, all_preds)),
        "mae":     float(mean_absolute_error(all_targets, all_preds)),
        "preds":   all_preds,
        "targets": all_targets,
    }


# ===========================================================================
# Single-fold training
# ===========================================================================

def train_fold(
    fold_num:    int,
    fold_name:   str,
    train_df:    pd.DataFrame,
    val_df:      pd.DataFrame,
    test_df:     pd.DataFrame,
    cfg,
    device:      torch.device,
    output_dir:  Path,
    resume_ckpt: Optional[Path] = None,
    warm_start_ckpt: Optional[Path] = None,
) -> Dict:
    """Train one CV0 fold end-to-end and return a metrics summary dict.

    This function handles the complete lifecycle for a single fold:
      1. Construct output directories
      2. Set up per-fold log file handler
      3. Set random seed for reproducibility
      4. Resolve AMP dtype and configure GradScaler
      5. Build DataLoaders, model, optimizer, scheduler
      6. Optionally restore state (--resume) or weights (--warm-start)
      7. Training loop with TensorBoard logging
      8. Best-checkpoint tracking via validation R²
      9. Early stopping (if enabled in config)
      10. Test evaluation with best checkpoint
      11. Save prediction CSV, training history CSV

    Parameters
    ----------
    fold_num         1-based fold index (for display only)
    fold_name        Fold directory name (e.g. "fold_2025")
    train_df / val_df / test_df  DataFrames from the splits CSVs
    cfg              OmegaConf config node
    device           Target torch.device
    output_dir       Root experiment output directory
    resume_ckpt      Path to checkpoint for full state restore (--resume).
                     Mutually exclusive with warm_start_ckpt.
    warm_start_ckpt  Path to checkpoint for weights-only init (--warm-start).
                     Mutually exclusive with resume_ckpt.

    Returns
    -------
    dict containing fold-level metrics and metadata for the CV summary
    """

    # ---- Output subdirectories ----------------------------------------------
    ckpt_dir    = output_dir / "checkpoints" / fold_name
    history_dir = output_dir / "history"
    preds_dir   = output_dir / "predictions"
    log_dir     = output_dir / "logs"
    tb_dir      = output_dir / "tensorboard" / fold_name
    for d in [ckpt_dir, history_dir, preds_dir, log_dir, tb_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Per-fold log file handler ------------------------------------------
    fold_log_path = log_dir / f"cv0_{fold_name}.log"
    fh = logging.FileHandler(fold_log_path)
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)

    # ---- Random seed --------------------------------------------------------
    seed = int(cfg.training.seed)
    set_seed(seed)

    # ---- AMP configuration --------------------------------------------------
    amp_dtype_str = str(cfg.training.get("amp_dtype", "bfloat16"))
    amp_dtype     = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16
    use_amp       = bool(cfg.training.use_amp)

    # GradScaler is only meaningful with float16.
    scaler = GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    logger.info(f"\n{'='*70}")
    logger.info(
        f"FOLD {fold_num}: {fold_name}  |  "
        f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}"
    )
    logger.info(f"{'='*70}")

    # ---- Data ---------------------------------------------------------------
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df, cfg, seed=seed
    )

    # ---- Model --------------------------------------------------------------
    model_name = str(cfg.model.name)
    model      = load_model(model_name, cfg).to(device)
    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model : {model_name}  ({n_params:,} params, {n_trainable:,} trainable)")

    # ---- Loss / optimizer / scheduler / gradient scaler --------------------
    criterion = nn.MSELoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    clip_val  = float(cfg.training.gradient_clip)

    # ---- Checkpoint loading (--resume or --warm-start) ----------------------
    # Default training state for a fresh run.
    start_epoch  = 1
    best_val_r2  = -float("inf")
    best_val_loss = float("inf")
    patience_ctr = 0

    if resume_ckpt is not None:
        # Full restore: model weights + optimizer + scheduler + scaler +
        # early stopping counters.  Split hash is verified inside this call.
        start_epoch, best_val_r2, best_val_loss, patience_ctr = (
            load_checkpoint_for_resume(
                ckpt_path=resume_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                train_df=train_df,
                device=device,
            )
        )
        logger.info(
            f"[--resume] Continuing from epoch {start_epoch}  |  "
            f"best_val_r2={best_val_r2:.4f}  patience_ctr={patience_ctr}"
        )

    elif warm_start_ckpt is not None:
        # Weights-only restore: model initialised from domain-adapted weights,
        # optimizer / scheduler / scaler are fresh from the config.
        load_checkpoint_for_warm_start(
            ckpt_path=warm_start_ckpt,
            model=model,
            device=device,
        )
        # Log the reduced learning rates for transparency in the run log.
        bb_lr, hd_lr = _lr_backbone_head(optimizer)
        logger.info(
            f"[--warm-start] Fresh optimizer initialised  |  "
            f"backbone_lr={bb_lr:.2e}  head_lr={hd_lr:.2e}  "
            f"(reduced from original run — see config warm_start block)"
        )

    # ---- TensorBoard writer -------------------------------------------------
    writer = SummaryWriter(log_dir=str(tb_dir))

    # ---- Training loop ------------------------------------------------------
    max_epochs   = int(cfg.training.max_epochs)
    log_interval = int(cfg.logging.log_interval)
    es_cfg       = cfg.training.early_stopping
    best_ckpt    = ckpt_dir / "checkpoint_best.pt"
    latest_ckpt  = ckpt_dir / "checkpoint_latest.pt"
    history      = []

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        all_train_preds, all_train_targets = [], []
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Support both dict-style and tuple-style batches from UAVDataset
            if isinstance(batch, dict):
                images  = batch["image"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
            else:
                images, targets = batch
                images  = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                preds = model(images)
                loss  = criterion(preds, targets.float())

            if scaler.is_enabled():
                # float16 path: scale the loss to prevent underflow
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                scaler.step(optimizer)
                scaler.update()
            else:
                # bfloat16 / no-AMP path: standard backward without scaling
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1
            all_train_preds.extend(preds.detach().cpu().float().numpy())
            all_train_targets.extend(targets.cpu().float().numpy())

            if (batch_idx + 1) % log_interval == 0:
                logger.debug(
                    f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}]  "
                    f"batch_loss={loss.item():.4f}"
                )

        # Advance the LR schedule one step
        scheduler.step()

        # ---- End-of-epoch metrics -------------------------------------------
        train_loss  = epoch_loss / n_batches
        train_r2    = float(r2_score(np.array(all_train_targets),
                                     np.array(all_train_preds)))
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp, amp_dtype)
        elapsed     = time.time() - t0

        lr_backbone, lr_head = _lr_backbone_head(optimizer)
        logger.info(
            f"Epoch {epoch:3d}/{max_epochs}  "
            f"train_loss={train_loss:.4f}  train_r2={train_r2:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_r2={val_metrics['r2']:.4f}  "
            f"val_rmse={val_metrics['rmse']:.4f}  "
            f"lr_backbone={lr_backbone:.2e}  lr_head={lr_head:.2e}  "
            f"time={elapsed:.1f}s"
        )

        # Log scalars to TensorBoard for live monitoring
        writer.add_scalar("Loss/train",       train_loss,           epoch)
        writer.add_scalar("Loss/val",         val_metrics["loss"],  epoch)
        writer.add_scalar("Metrics/train_r2", train_r2,             epoch)
        writer.add_scalar("Metrics/val_r2",   val_metrics["r2"],    epoch)
        writer.add_scalar("Metrics/val_rmse", val_metrics["rmse"],  epoch)
        writer.add_scalar("LR/backbone",      lr_backbone,          epoch)
        writer.add_scalar("LR/head",          lr_head,              epoch)

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_r2":   train_r2,
            "val_loss":   val_metrics["loss"],
            "val_r2":     val_metrics["r2"],
            "val_rmse":   val_metrics["rmse"],
            "val_mae":    val_metrics["mae"],
        })

        # ---- Best-checkpoint logic ------------------------------------------
        if val_metrics["r2"] > best_val_r2 + es_cfg.min_delta:
            best_val_r2   = val_metrics["r2"]
            best_val_loss = val_metrics["loss"]
            patience_ctr  = 0
            # Save full state so this checkpoint can be used for --resume
            save_checkpoint(
                path=best_ckpt,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_r2=best_val_r2,
                best_val_loss=best_val_loss,
                patience_ctr=patience_ctr,
                fold_name=fold_name,
                cfg=cfg,
                train_df=train_df,
            )
            logger.info(f"  ✓ Saved best checkpoint  (val_r2={best_val_r2:.4f})")
        else:
            patience_ctr += 1
            if es_cfg.enabled and patience_ctr >= es_cfg.patience:
                logger.info(
                    f"  Early stopping triggered at epoch {epoch} "
                    f"(no improvement for {es_cfg.patience} epochs)."
                )
                break

        # ---- Save latest checkpoint every epoch ----------------------------
        # The latest checkpoint enables --resume after cluster preemption even
        # if no val R² improvement has occurred since the last best checkpoint.
        save_checkpoint(
            path=latest_ckpt,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val_r2=best_val_r2,
            best_val_loss=best_val_loss,
            patience_ctr=patience_ctr,
            fold_name=fold_name,
            cfg=cfg,
            train_df=train_df,
        )

    # ---- Test evaluation (always use the best checkpoint) -------------------
    logger.info(f"\nLoading best checkpoint for test evaluation: {best_ckpt}")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device, use_amp, amp_dtype)

    logger.info(
        f"\n{'='*70}\n"
        f"TEST RESULTS — {fold_name}\n"
        f"  R²   = {test_metrics['r2']:.4f}\n"
        f"  RMSE = {test_metrics['rmse']:.4f}\n"
        f"  MAE  = {test_metrics['mae']:.4f}\n"
        f"{'='*70}"
    )

    # ---- Save per-fold prediction CSV ---------------------------------------
    img_col  = cfg.data.image_filename_column
    preds_df = pd.DataFrame({
        "image_filename": test_df[img_col].values,
        "actual":         test_metrics["targets"],
        "predicted":      test_metrics["preds"],
        "signed_error":   test_metrics["preds"] - test_metrics["targets"],
        "abs_error":      np.abs(test_metrics["targets"] - test_metrics["preds"]),
    })
    pred_path = preds_dir / f"{fold_name}_test_predictions.csv"
    preds_df.to_csv(pred_path, index=False)
    logger.info(f"Saved test predictions   → {pred_path}")

    # ---- Save training history CSV ------------------------------------------
    history_df = pd.DataFrame(history)
    hist_path  = history_dir / f"{fold_name}_history.csv"
    history_df.to_csv(hist_path, index=False)
    logger.info(f"Saved training history   → {hist_path}")

    writer.close()

    # Remove the per-fold FileHandler to prevent log duplication in later folds
    logging.getLogger().removeHandler(fh)
    fh.close()

    return {
        "fold":             fold_num,
        "fold_name":        fold_name,
        "train_samples":    len(train_df),
        "val_samples":      len(val_df),
        "test_samples":     len(test_df),
        "best_val_loss":    best_val_loss,
        "best_val_r2":      best_val_r2,
        "final_train_loss": history_df["train_loss"].iloc[-1],
        "final_train_r2":   history_df["train_r2"].iloc[-1],
        "final_val_loss":   history_df["val_loss"].iloc[-1],
        "final_val_r2":     history_df["val_r2"].iloc[-1],
        "test_loss":        test_metrics["loss"],
        "test_r2":          test_metrics["r2"],
        "test_rmse":        test_metrics["rmse"],
        "test_mae":         test_metrics["mae"],
        "epochs_trained":   len(history_df),
    }


# ===========================================================================
# CV0 orchestrator
# ===========================================================================

def run_cv0(
    cfg,
    device:          torch.device,
    output_dir:      Path,
    max_folds:       Optional[int]  = None,
    resume_ckpt:     Optional[Path] = None,
    warm_start_ckpt: Optional[Path] = None,
    target_fold:     Optional[str]  = None,
) -> list:
    """Iterate over all CV0 fold directories and run train_fold on each.

    Fold directories are discovered by listing the cv0/ subdirectory inside
    cfg.cv.splits_dir.  When --resume or --warm-start is active and --fold is
    specified, only that fold is processed so that the checkpoint can be
    applied to the correct held-out year.

    Parameters
    ----------
    cfg              OmegaConf config node
    device           Target torch.device
    output_dir       Root directory for all experiment outputs
    max_folds        If set, stop after this many folds (smoke-testing only)
    resume_ckpt      Path to checkpoint for --resume (full state restore)
    warm_start_ckpt  Path to checkpoint for --warm-start (weights only)
    target_fold      If set, only run the fold whose directory name matches
                     this string (e.g. "fold_2025").  Required when --resume
                     is active so the correct split hash is verified.
    """
    splits_dir = Path(cfg.cv.splits_dir) / "cv0"
    assert splits_dir.exists(), (
        f"CV0 splits directory not found: {splits_dir}\n"
        "Run scripts/create_cv_splits.py --cv-strategies cv0 first."
    )

    fold_dirs = sorted(splits_dir.iterdir())

    # When a specific fold is targeted (--resume or single-fold re-run),
    # filter to only that fold.
    if target_fold is not None:
        fold_dirs = [d for d in fold_dirs if d.name == target_fold]
        if not fold_dirs:
            raise ValueError(
                f"--fold '{target_fold}' not found in {splits_dir}.  "
                f"Available folds: {[d.name for d in sorted(splits_dir.iterdir())]}"
            )

    if max_folds:
        fold_dirs = fold_dirs[:max_folds]

    logger.info(f"Found {len(fold_dirs)} CV0 fold(s): {[d.name for d in fold_dirs]}")

    all_results = []
    for fold_num, fold_dir in enumerate(fold_dirs, 1):
        train_df = pd.read_csv(fold_dir / "train.csv")
        val_df   = pd.read_csv(fold_dir / "val.csv")
        test_df  = pd.read_csv(fold_dir / "test.csv")

        result = train_fold(
            fold_num=fold_num,
            fold_name=fold_dir.name,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            cfg=cfg,
            device=device,
            output_dir=output_dir,
            resume_ckpt=resume_ckpt,
            warm_start_ckpt=warm_start_ckpt,
        )
        all_results.append(result)

    # ---- Cross-fold summary --------------------------------------------------
    model_name = str(cfg.model.name)
    summary_df = pd.DataFrame(all_results)

    logger.info(f"\n{'='*70}")
    logger.info(f"CV0 SUMMARY — {model_name}")
    logger.info(f"{'='*70}")
    display_cols = ["fold_name", "val_samples", "test_samples",
                    "best_val_r2", "test_r2", "test_rmse", "test_mae"]
    print(summary_df[display_cols].to_string(index=False))

    mean_val_r2    = summary_df["best_val_r2"].mean()
    std_val_r2     = summary_df["best_val_r2"].std()
    mean_test_r2   = summary_df["test_r2"].mean()
    std_test_r2    = summary_df["test_r2"].std()
    mean_test_rmse = summary_df["test_rmse"].mean()
    std_test_rmse  = summary_df["test_rmse"].std()
    mean_test_mae  = summary_df["test_mae"].mean()

    logger.info(f"\n  Mean Val  R²   = {mean_val_r2:.4f} ± {std_val_r2:.4f}")
    logger.info(f"  Mean Test R²   = {mean_test_r2:.4f} ± {std_test_r2:.4f}")
    logger.info(f"  Mean Test RMSE = {mean_test_rmse:.4f} ± {std_test_rmse:.4f}")
    logger.info(f"  Mean Test MAE  = {mean_test_mae:.4f}")

    summary_df.to_csv(output_dir / "cv_results.csv", index=False)

    aggregate = {
        "model":          model_name,
        "cv_type":        "cv0",
        "n_folds":        len(all_results),
        "mean_val_r2":    float(mean_val_r2),
        "std_val_r2":     float(std_val_r2),
        "mean_val_loss":  float(summary_df["best_val_loss"].mean()),
        "std_val_loss":   float(summary_df["best_val_loss"].std()),
        "mean_test_r2":   float(mean_test_r2),
        "std_test_r2":    float(std_test_r2),
        "mean_test_rmse": float(mean_test_rmse),
        "std_test_rmse":  float(std_test_rmse),
        "mean_test_mae":  float(mean_test_mae),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    logger.info(f"\n✓ Results saved → {output_dir}")
    return all_results


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CV0 leave-one-year-out training for all SLB architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to model config YAML (cfg.model.name must be in MODEL_REGISTRY).",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU index to use (default: auto-select GPU with most free VRAM).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override cfg.cv.output_dir.  Default reads from the config file.",
    )
    parser.add_argument(
        "--max-folds", type=int, default=None,
        help="Stop after N folds (default: run all folds).  Useful for smoke-testing.",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help="Override cfg.training.max_epochs.",
    )
    parser.add_argument(
        "--image-size", type=int, default=None,
        help=(
            "Override cfg.data.image_size (applied as a square crop). "
            "Must be a multiple of 32 for coatnet2, maxvit_small, swinv2_base."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override cfg.data.dataloader.batch_size.",
    )
    parser.add_argument(
        "--splits-dir", type=str, default=None,
        help="Override cfg.cv.splits_dir.",
    )
    parser.add_argument(
        "--amp-dtype", type=str, default=None,
        choices=["bfloat16", "float16"],
        help=(
            "AMP precision dtype (default: value in config, typically bfloat16). "
            "Use float16 on pre-Ampere GPUs that do not support bfloat16."
        ),
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable mixed-precision entirely.",
    )

    # ── Checkpoint arguments ──────────────────────────────────────────────────
    ckpt_group = parser.add_mutually_exclusive_group()
    ckpt_group.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CKPT_PATH",
        help=(
            "Path to a checkpoint (.pt) to fully resume an interrupted training "
            "run.  Restores model weights, optimizer state, scheduler state, AMP "
            "scaler, epoch counter, and early stopping patience counter.  The "
            "training split hash is verified against the stored hash — if the "
            "data has changed since the checkpoint was saved, the script will "
            "abort with an informative error.  Use --warm-start instead when "
            "retraining on an expanded dataset.  Requires --fold to identify "
            "which CV0 fold the checkpoint belongs to."
        ),
    )
    ckpt_group.add_argument(
        "--warm-start",
        type=str,
        default=None,
        metavar="CKPT_PATH",
        help=(
            "Path to a checkpoint (.pt) whose model weights are used to "
            "initialise training.  The optimizer, scheduler, and AMP scaler "
            "are freshly initialised from the config — only the weights are "
            "carried over.  Use this mode when retraining EVA-02-B on an "
            "expanded dataset (e.g. a new field season appended to the labels "
            "CSV).  The recommended config for this mode is "
            "configs/eva02_base_retrain.yaml, which uses lower learning rates "
            "than the original run (backbone_lr=1e-5, head_lr=1e-4) and a "
            "shorter warmup.  Split hash verification is NOT performed — it is "
            "expected and correct that the data is different from the source run."
        ),
    )
    parser.add_argument(
        "--fold",
        type=str,
        default=None,
        metavar="FOLD_NAME",
        help=(
            "Target a single CV0 fold by name (e.g. 'fold_2025').  Required "
            "with --resume so the correct split hash is verified.  Optional "
            "with --warm-start (defaults to running all folds if omitted)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Validate checkpoint argument combinations --------------------------
    if args.resume and args.fold is None:
        raise SystemExit(
            "--resume requires --fold to identify which CV0 fold the checkpoint "
            "belongs to (needed for split hash verification).\n"
            "Example:  --resume experiments/.../checkpoint_latest.pt --fold fold_2025"
        )

    # ---- Load and validate config -------------------------------------------
    cfg_path = Path(args.config)
    assert cfg_path.exists(), f"Config not found: {cfg_path}"
    cfg = OmegaConf.load(cfg_path)

    model_name = str(cfg.model.name)
    if model_name not in MODEL_REGISTRY:
        raise SystemExit(
            f"cfg.model.name='{model_name}' is not in MODEL_REGISTRY.\n"
            f"Valid names: {sorted(MODEL_REGISTRY)}"
        )
    entry = MODEL_REGISTRY[model_name]

    # ---- Apply CLI overrides to config values --------------------------------
    if args.max_epochs  is not None: cfg.training.max_epochs        = args.max_epochs
    if args.batch_size  is not None: cfg.data.dataloader.batch_size = args.batch_size
    if args.splits_dir  is not None: cfg.cv.splits_dir              = args.splits_dir
    if args.output_dir  is not None: cfg.cv.output_dir              = args.output_dir
    if args.amp_dtype   is not None: cfg.training.amp_dtype         = args.amp_dtype
    if args.no_amp:                  cfg.training.use_amp           = False

    if args.image_size is not None:
        if entry.requires_mod32:
            assert args.image_size % 32 == 0, (
                f"{model_name} requires image_size divisible by 32 "
                f"(five stride-2 downsampling stages).  Got {args.image_size}."
            )
        cfg.data.image_size = [args.image_size, args.image_size]

    # ---- Resolve checkpoint paths -------------------------------------------
    resume_ckpt     = Path(args.resume)     if args.resume     else None
    warm_start_ckpt = Path(args.warm_start) if args.warm_start else None

    if resume_ckpt and not resume_ckpt.exists():
        raise SystemExit(f"--resume checkpoint not found: {resume_ckpt}")
    if warm_start_ckpt and not warm_start_ckpt.exists():
        raise SystemExit(f"--warm-start checkpoint not found: {warm_start_ckpt}")

    # ---- Output directory ---------------------------------------------------
    output_dir = Path(cfg.cv.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, output_dir / "config_resolved.yaml")

    # ---- Device selection ---------------------------------------------------
    device = select_gpu(args.gpu)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # ---- Attach global log file handler -------------------------------------
    fh = logging.FileHandler(output_dir / "train.log")
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)

    amp_dtype_str = str(cfg.training.get("amp_dtype", "bfloat16"))
    logger.info("=" * 70)
    logger.info(f"{model_name} — CV0 Leave-One-Year-Out Training")
    logger.info("=" * 70)
    logger.info(f"Config      : {cfg_path}")
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"Device      : {device}")
    logger.info(f"Image size  : {cfg.data.image_size}")
    logger.info(f"Batch size  : {cfg.data.dataloader.batch_size}")
    logger.info(f"Max epochs  : {cfg.training.max_epochs}")
    logger.info(f"AMP         : {cfg.training.use_amp}  dtype={amp_dtype_str}")

    # Log checkpoint mode for audit trail
    if resume_ckpt:
        logger.info(f"Mode        : RESUME from {resume_ckpt}  (fold={args.fold})")
    elif warm_start_ckpt:
        logger.info(f"Mode        : WARM-START from {warm_start_ckpt}")
    else:
        logger.info("Mode        : fresh training from pretrained weights")

    # ---- Run CV0 ------------------------------------------------------------
    run_cv0(
        cfg=cfg,
        device=device,
        output_dir=output_dir,
        max_folds=args.max_folds,
        resume_ckpt=resume_ckpt,
        warm_start_ckpt=warm_start_ckpt,
        target_fold=args.fold,
    )


if __name__ == "__main__":
    main()
