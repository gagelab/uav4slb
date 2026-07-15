"""
src/models/efficientnetv2s.py
-----------------------------
EfficientNet V2-S regression wrapper for UAV-based SLB disease severity prediction.

Architecture
------------
  Backbone : tf_efficientnetv2_s (timm)
             Mobile-inverted bottleneck CNN with Fused-MBConv stages.
             Feature dim = 1280.  Pretrained with supervised ImageNet-21k
             fine-tuned on ImageNet-1k — lightweight CNN baseline (~21 M params).
  Head     : Dropout → Linear(1280 → 256) → GELU → Dropout → Linear(256 → 1)

Differential learning rates
-----------------------------
  get_optimizer_param_groups() separates:
    - backbone (all conv/bn stages)  → lower LR (default: 5e-5)
    - regression head                → higher LR (default: 5e-4)

  This mirrors the pattern used in convnextv2.py and dinov2_vits14.py for a
  clean, fair architectural comparison.

Usage
-----
  from src.models.efficientnetv2s import create_efficientnetv2s

  model = create_efficientnetv2s(
      pretrained=True,
      dropout_rate=0.3,
      freeze_backbone=False,
  )
  param_groups = model.get_optimizer_param_groups(
      backbone_lr=5e-5, head_lr=5e-4, weight_decay=1e-2
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
        "timm is required for EfficientNet V2-S.  Install with:  pip install timm"
    ) from e

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EFFICIENTNETV2S_FEATURE_DIM = 1280   # num_features for tf_efficientnetv2_s in timm


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

class EfficientNetV2SRegressor(nn.Module):
    """
    EfficientNet V2-S backbone + regression head for disease severity scoring.

    Parameters
    ----------
    pretrained : bool
        Load timm's default pretrained weights (ImageNet-21k → ImageNet-1k).
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
            "tf_efficientnetv2_s",
            pretrained=pretrained,
            num_classes=0,          # Remove classification head; returns (B, 1280)
            global_pool="avg",      # Global average pooling over spatial dims
        )
        feature_dim = self.backbone.num_features  # 1280 for tf_efficientnetv2_s
        logger.info(
            f"EfficientNet V2-S backbone loaded  |  pretrained={pretrained}  "
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
        features = self.backbone(x)   # (B, 1280)
        return self.head(features)    # (B,)

    def get_optimizer_param_groups(
        self,
        backbone_lr: float = 5e-5,
        head_lr: float = 5e-4,
        weight_decay: float = 5e-2,
    ) -> List[Dict]:
        """
        Return two parameter groups for differential learning rates:
          - backbone  : lower LR to preserve pretrained representations
          - head      : higher LR for task-specific regression learning

        Matches the ConvNeXt V2-B and DINOv2 training convention for fair
        comparison.
        """
        backbone_params = list(self.backbone.parameters())
        head_params     = list(self.head.parameters())

        # Bias / norm terms should be excluded from weight decay
        def _split_wd(params):
            decay, no_decay = [], []
            for p in params:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1:          # bias, BatchNorm/LayerNorm gains
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

def create_efficientnetv2s(
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
    hidden_dim: int = 256,
) -> EfficientNetV2SRegressor:
    """
    Instantiate an EfficientNet V2-S regressor.

    Parameters
    ----------
    pretrained     : Load ImageNet-21k pretrained weights via timm.
    dropout_rate   : Dropout probability in the regression head.
    freeze_backbone: Freeze backbone for linear-probe evaluation.
    hidden_dim     : Hidden units in the two-layer MLP head.

    Returns
    -------
    EfficientNetV2SRegressor
    """
    model = EfficientNetV2SRegressor(
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        hidden_dim=hidden_dim,
    )
    n_total     = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"EfficientNet V2-S Regressor  |  total={n_total:.1f}M  trainable={n_trainable:.1f}M")
    return model
