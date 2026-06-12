"""
src/models/convnextv2.py
-----------------------
ConvNeXt V2-B regression wrapper for UAV-based SLB disease severity prediction.

Architecture
------------
  Backbone : convnextv2_base (timm)
             Hierarchical CNN with 4 stages, global average pooling head.
             Feature dim = 1024.  Pretrained with FCMAE self-supervised learning
             on ImageNet-22k — CNN counterpart to DINOv2 ViT-B/14.
  Head     : Dropout → Linear(1024 → 256) → GELU → Dropout → Linear(256 → 1)

Differential learning rates
-----------------------------
  get_optimizer_param_groups() separates:
    - backbone (stages 0-3)      → lower LR (default: 2e-5)
    - regression head            → higher LR (default: 2e-4)

  This mirrors the pattern used in dinov2_vitb14.py and dinov2_vits14.py for a
  clean, fair architectural comparison.

Usage
-----
  from src.models.convnextv2 import create_convnextv2_base

  model = create_convnextv2_base(
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
from typing import List, Dict, Optional

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
CONVNEXTV2_BASE_FEATURE_DIM = 1024   # num_features for convnextv2_base in timm


# ===========================================================================
# Regression head
# ===========================================================================

class RegressionHead(nn.Module):
    """Two-layer MLP regression head with dropout and GELU activation."""

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

class ConvNeXtV2Regressor(nn.Module):
    """
    ConvNeXt V2-B backbone + regression head for disease severity scoring.

    Parameters
    ----------
    pretrained : bool
        Load timm's default pretrained weights (ImageNet-22k FCMAE).
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
            "convnextv2_base",
            pretrained=pretrained,
            num_classes=0,          # Remove classification head; returns (B, 1024)
            global_pool="avg",      # Global average pooling over spatial dims
        )
        feature_dim = self.backbone.num_features  # 1024 for convnextv2_base
        logger.info(
            f"ConvNeXt V2-B backbone loaded  |  pretrained={pretrained}  "
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
        features = self.backbone(x)   # (B, 1024)
        return self.head(features)    # (B,)

    def get_optimizer_param_groups(
        self,
        backbone_lr: float = 2e-5,
        head_lr: float = 2e-4,
        weight_decay: float = 5e-2,
    ) -> List[Dict]:
        """
        Return two parameter groups for differential learning rates:
          - backbone  : lower LR to preserve pretrained representations
          - head      : higher LR for task-specific regression learning

        Matches the DINOv2 training convention for fair comparison.
        """
        backbone_params = list(self.backbone.parameters())
        head_params     = list(self.head.parameters())

        # Bias / norm terms should be excluded from weight decay
        def _split_wd(params):
            decay, no_decay = [], []
            for p in params:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1:          # bias, LayerNorm/BatchNorm gains
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        bb_decay, bb_no_decay = _split_wd(backbone_params)
        hd_decay, hd_no_decay = _split_wd(head_params)

        param_groups = [
            {"params": bb_decay,    "lr": backbone_lr, "weight_decay": weight_decay,  "name": "backbone_decay"},
            {"params": bb_no_decay, "lr": backbone_lr, "weight_decay": 0.0,            "name": "backbone_no_decay"},
            {"params": hd_decay,    "lr": head_lr,     "weight_decay": weight_decay,   "name": "head_decay"},
            {"params": hd_no_decay, "lr": head_lr,     "weight_decay": 0.0,            "name": "head_no_decay"},
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

def create_convnextv2_base(
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
    hidden_dim: int = 256,
) -> ConvNeXtV2Regressor:
    """
    Instantiate a ConvNeXt V2-B regressor.

    Parameters
    ----------
    pretrained     : Load ImageNet-22k FCMAE pretrained weights via timm.
    dropout_rate   : Dropout probability in the regression head.
    freeze_backbone: Freeze backbone for linear-probe evaluation.
    hidden_dim     : Hidden units in the two-layer MLP head.

    Returns
    -------
    ConvNeXtV2Regressor
    """
    model = ConvNeXtV2Regressor(
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        hidden_dim=hidden_dim,
    )
    n_total     = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"ConvNeXt V2-B Regressor  |  total={n_total:.1f}M  trainable={n_trainable:.1f}M")
    return model
