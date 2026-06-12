"""
src/models/swinv2_base.py
-------------------------
SwinV2-B regression wrapper for UAV-based SLB disease severity prediction.

Architecture
------------
  Backbone : swinv2_base_window12to16_192to256.ms_in22k_ft_in1k (timm)
             Shifted-window hierarchical ViT with 4 stages.  Feature dim = 1024.
             ~88 M params.  Patch size 4; window size 16 at 256px input
             (12 at 192px pretraining, adapted to 16 at 256px fine-tuning via
             the bicubic log-spaced relative position bias interpolation
             introduced in SwinV2).
             Pretrained on ImageNet-22k (multi-scale), fine-tuned on ImageNet-1k
             at 256×256.
  Head     : Dropout → Linear(1024 → 256) → GELU → Dropout → Linear(256 → 1)

Key SwinV2 improvements over SwinV1
--------------------------------------
  1. Log-spaced continuous relative position bias — generalises cleanly across
     window sizes and resolutions without ad-hoc interpolation.
  2. Post-normalization (LayerNorm after attention/FFN rather than before) —
     stabilises deep training and large-scale pretraining.
  3. Cosine attention (replaces dot-product softmax) — prevents attention
     entropy collapse at scale.
  These make SwinV2-B a more robust baseline than SwinV1-B and an interesting
  comparison point against ConvNeXt V2-B (~88 M, pure CNN) and MaxViT-S (~69 M,
  hybrid) at similar parameter counts.

Relationship to other models in this study
-------------------------------------------
  SwinV2-B sits at the same ~88 M parameter tier as ConvNeXt V2-B and EVA-02-B/
  DINOv2 ViT-B/14 (both ~86 M), enabling comparisons across:
    CNN (ConvNeXt V2-B) vs. sliding-window ViT (SwinV2-B) at matched scale
    Supervised pretraining (SwinV2) vs. self-supervised (ConvNeXt V2 FCMAE,
    DINOv2) at matched scale
    Local-window attention (SwinV2) vs. global attention (DINOv2 ViT-B)

  Full architecture comparison
    EfficientNet-V2-S  : ~21 M,  384×384, supervised
    DINOv2 ViT-S/14   : ~22 M,  448×448, self-supervised
    MaxViT-S           : ~69 M,  224×224, supervised (IN-1k)
    EVA-02-B           : ~86 M,  448×448, CLIP-MIM
    DINOv2 ViT-B/14   : ~86 M,  518×518, self-supervised
    ConvNeXt V2-B     : ~88 M,  224×224, FCMAE self-supervised
    SwinV2-B           : ~88 M,  256×256, supervised (IN-22k → IN-1k) ← this module
    ConvNeXt V2-L     : ~198 M, 224×224, FCMAE self-supervised

Differential learning rates
-----------------------------
  get_optimizer_param_groups() separates:
    - backbone (all Swin stages + patch embed)  → lower LR (default: 2e-5)
    - regression head                           → higher LR (default: 2e-4)

  Mirrors the pattern used in convnextv2.py and dinov2_vitb14.py for a fair
  architectural comparison.

Usage
-----
  from src.models.swinv2_base import create_swinv2_base

  model = create_swinv2_base(
      pretrained=True,
      dropout_rate=0.3,
      freeze_backbone=False,
  )
  param_groups = model.get_optimizer_param_groups(
      backbone_lr=2e-5, head_lr=2e-4, weight_decay=5e-2
  )
"""

from __future__ import annotations

import logging
from typing import List, Dict

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for SwinV2.  Install with:  pip install timm"
    ) from e

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SWINV2_BASE_FEATURE_DIM = 1024
SWINV2_BASE_TIMM_NAME   = "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k"


# ===========================================================================
# Regression head
# ===========================================================================

class RegressionHead(nn.Module):
    """Two-layer MLP regression head with dropout and GELU activation.

    Structurally identical to the head used across all other models in this
    study — the only variable in architectural comparisons is the backbone.
    in_features is 1024 for SwinV2-B, matching ConvNeXt V2-B.
    """

    def __init__(self, in_features: int, hidden_dim: int = 256, dropout_rate: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


# ===========================================================================
# Full model
# ===========================================================================

class SwinV2BaseRegressor(nn.Module):
    """
    SwinV2-B backbone + regression head for disease severity scoring.

    SwinV2's hierarchical shifted-window attention processes images in
    non-overlapping local windows at each stage, gradually merging spatial
    tokens via patch merging.  This gives it strong inductive biases for
    local texture (relevant for individual SLB lesions) while the shifted-
    window mechanism introduces cross-window connectivity at every other layer.
    The log-spaced relative position bias allows the 192px-pretrained model to
    generalise to the 256px fine-tuning resolution without performance loss —
    the same mechanism that makes SwinV2 more resolution-transferable than SwinV1.

    Parameters
    ----------
    pretrained : bool
        Load timm's default pretrained weights (IN-22k MS → IN-1k 256px FT).
    dropout_rate : float
        Dropout applied in the regression head (default 0.3).
    freeze_backbone : bool
        If True, freeze all backbone parameters (linear-probe mode).
    hidden_dim : int
        Hidden dimension of the regression head MLP (default 256).
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # ---- Backbone (feature extractor, no classifier) --------------------
        self.backbone = timm.create_model(
            SWINV2_BASE_TIMM_NAME,
            pretrained=pretrained,
            num_classes=0,       # Remove classifier; returns (B, 1024) after GAP
            global_pool="avg",   # Global average pooling over the final stage grid
        )
        feature_dim = self.backbone.num_features  # 1024 for swinv2_base
        logger.info(
            f"SwinV2-B backbone loaded  |  pretrained={pretrained}  "
            f"feature_dim={feature_dim}  "
            f"params={sum(p.numel() for p in self.backbone.parameters()) / 1e6:.1f}M"
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            logger.info("Backbone frozen (linear-probe mode).")

        # ---- Regression head ------------------------------------------------
        self.head = RegressionHead(
            in_features=feature_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, C, H, W)
            Input must be divisible by the patch-embed stride × window size.
            For this checkpoint: multiples of 32 work; 256 is the native size.

        Returns
        -------
        torch.Tensor  shape (B,)  — predicted disease severity scores
        """
        features = self.backbone(x)   # (B, 1024)
        return self.head(features)    # (B,)

    def get_optimizer_param_groups(
        self,
        backbone_lr: float = 2e-5,
        head_lr: float = 2e-4,
        weight_decay: float = 5e-2,
    ) -> List[Dict]:
        """
        Return parameter groups for differential learning rates:
          - backbone  : lower LR to preserve pretrained representations
          - head      : higher LR for task-specific regression learning

        Bias and LayerNorm scale terms are excluded from weight decay in each
        group.  SwinV2 uses post-LayerNorm; both bias and LayerNorm parameters
        should be no-decay, consistent with the original SwinV2 training recipe.
        """
        backbone_params = list(self.backbone.parameters())
        head_params     = list(self.head.parameters())

        def _split_wd(params):
            decay, no_decay = [], []
            for p in params:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1:      # bias, LayerNorm scale/shift
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        bb_decay, bb_no_decay = _split_wd(backbone_params)
        hd_decay, hd_no_decay = _split_wd(head_params)

        param_groups = [
            {"params": bb_decay,    "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone_decay"},
            {"params": bb_no_decay, "lr": backbone_lr, "weight_decay": 0.0,          "name": "backbone_no_decay"},
            {"params": hd_decay,    "lr": head_lr,     "weight_decay": weight_decay, "name": "head_decay"},
            {"params": hd_no_decay, "lr": head_lr,     "weight_decay": 0.0,          "name": "head_no_decay"},
        ]

        total_params = sum(p.numel() for g in param_groups for p in g["params"])
        logger.info(
            f"Optimizer param groups  |  backbone_lr={backbone_lr:.1e}  "
            f"head_lr={head_lr:.1e}  wd={weight_decay:.2e}  "
            f"total_params={total_params / 1e6:.1f}M"
        )
        return param_groups


# ===========================================================================
# Factory function
# ===========================================================================

def create_swinv2_base(
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
    hidden_dim: int = 256,
) -> SwinV2BaseRegressor:
    """
    Instantiate a SwinV2-B regressor.

    Parameters
    ----------
    pretrained     : Load IN-22k MS → IN-1k 256px pretrained weights via timm.
    dropout_rate   : Dropout probability in the regression head.
    freeze_backbone: Freeze backbone for linear-probe evaluation.
    hidden_dim     : Hidden units in the two-layer MLP head.

    Returns
    -------
    SwinV2BaseRegressor
    """
    model = SwinV2BaseRegressor(
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        hidden_dim=hidden_dim,
    )
    n_total     = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"SwinV2-B Regressor  |  total={n_total:.1f}M  trainable={n_trainable:.1f}M")
    return model
