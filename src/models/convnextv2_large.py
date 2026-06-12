"""
src/models/convnextv2_large.py
------------------------------
ConvNeXt V2-L regression wrapper for UAV-based SLB disease severity prediction.

Architecture
------------
  Backbone : convnextv2_large (timm)
             Hierarchical CNN with 4 stages, global average pooling head.
             Feature dim = 1536.  ~198 M params.
             Pretrained with FCMAE self-supervised learning on ImageNet-22k —
             the large-scale variant of the ConvNeXt V2 family.
  Head     : Dropout → Linear(1536 → 256) → GELU → Dropout → Linear(256 → 1)

Relationship to other models in this study
-------------------------------------------
  ConvNeXt V2-L is the scaling-up counterpart to ConvNeXt V2-B (~88 M).
  Both share the same FCMAE pretraining strategy and hierarchical CNN design.
  Comparing the two isolates the effect of model scale within the CNN family,
  directly analogous to comparing DINOv2 ViT-S/14 vs. ViT-B/14 on the ViT side.

  Full architecture comparison
    EfficientNet-V2-S  : ~21 M,  384×384, supervised
    DINOv2 ViT-S/14   : ~22 M,  448×448, self-supervised
    MaxViT-S           : ~69 M,  224×224, supervised (IN-1k)
    EVA-02-B           : ~86 M,  448×448, CLIP-MIM
    DINOv2 ViT-B/14   : ~86 M,  518×518, self-supervised
    ConvNeXt V2-B     : ~88 M,  224×224, FCMAE self-supervised
    ConvNeXt V2-L     : ~198 M, 224×224, FCMAE self-supervised  ← this module

Differential learning rates
-----------------------------
  get_optimizer_param_groups() separates:
    - backbone (stages 0–3)  → lower LR (default: 1e-5)
    - regression head        → higher LR (default: 1e-4)

  backbone_lr is halved relative to ConvNeXt V2-B (2e-5 → 1e-5) because the
  larger model has a deeper and richer feature hierarchy worth preserving more
  conservatively.  The 10× head/backbone ratio is maintained.

Usage
-----
  from src.models.convnextv2_large import create_convnextv2_large

  model = create_convnextv2_large(
      pretrained=True,
      dropout_rate=0.3,
      freeze_backbone=False,
  )
  param_groups = model.get_optimizer_param_groups(
      backbone_lr=1e-5, head_lr=1e-4, weight_decay=5e-2
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
        "timm is required for ConvNeXt V2.  Install with:  pip install timm"
    ) from e

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONVNEXTV2_LARGE_FEATURE_DIM = 1536   # num_features for convnextv2_large in timm
CONVNEXTV2_LARGE_TIMM_NAME   = "convnextv2_large"


# ===========================================================================
# Regression head
# ===========================================================================

class RegressionHead(nn.Module):
    """Two-layer MLP regression head with dropout and GELU activation.

    Structurally identical to the head used across all other models in this
    study — the only variable in architectural comparisons is the backbone.
    in_features is 1536 for ConvNeXt V2-L (vs. 1024 for ConvNeXt V2-B).
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

class ConvNeXtV2LargeRegressor(nn.Module):
    """
    ConvNeXt V2-L backbone + regression head for disease severity scoring.

    ConvNeXt V2-L doubles the channel widths of ConvNeXt V2-B at each stage
    (96/192/384/768 → 192/384/768/1536), yielding ~198 M parameters and a
    richer 1536-dimensional feature space.  The same FCMAE pretraining on
    ImageNet-22k is used, making this a direct scale-up of ConvNeXt V2-B
    under otherwise identical training conditions.

    Parameters
    ----------
    pretrained : bool
        Load timm's default pretrained weights (ImageNet-22k FCMAE → IN-1k FT).
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
            CONVNEXTV2_LARGE_TIMM_NAME,
            pretrained=pretrained,
            num_classes=0,       # Remove classifier head; returns (B, 1536)
            global_pool="avg",   # Global average pooling over spatial dims
        )
        feature_dim = self.backbone.num_features  # 1536 for convnextv2_large
        logger.info(
            f"ConvNeXt V2-L backbone loaded  |  pretrained={pretrained}  "
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

        Returns
        -------
        torch.Tensor  shape (B,)  — predicted disease severity scores
        """
        features = self.backbone(x)   # (B, 1536)
        return self.head(features)    # (B,)

    def get_optimizer_param_groups(
        self,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 5e-2,
    ) -> List[Dict]:
        """
        Return parameter groups for differential learning rates:
          - backbone  : lower LR to preserve pretrained representations
          - head      : higher LR for task-specific regression learning

        backbone_lr defaults to 1e-5 (half of ConvNeXt V2-B's 2e-5) to
        account for the larger and richer feature hierarchy.  The 10× ratio
        between backbone and head is maintained.

        Bias and LayerNorm scale terms are excluded from weight decay in each
        group, matching the ConvNeXt V2-B and all other baselines.
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

def create_convnextv2_large(
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
    hidden_dim: int = 256,
) -> ConvNeXtV2LargeRegressor:
    """
    Instantiate a ConvNeXt V2-L regressor.

    Parameters
    ----------
    pretrained     : Load ImageNet-22k FCMAE pretrained weights via timm.
    dropout_rate   : Dropout probability in the regression head.
    freeze_backbone: Freeze backbone for linear-probe evaluation.
    hidden_dim     : Hidden units in the two-layer MLP head.

    Returns
    -------
    ConvNeXtV2LargeRegressor
    """
    model = ConvNeXtV2LargeRegressor(
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        hidden_dim=hidden_dim,
    )
    n_total     = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"ConvNeXt V2-L Regressor  |  total={n_total:.1f}M  trainable={n_trainable:.1f}M")
    return model
