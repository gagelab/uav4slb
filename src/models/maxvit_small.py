"""
src/models/maxvit_small.py
--------------------------
MaxViT-S regression wrapper for UAV-based SLB disease severity prediction.

Architecture
------------
  Backbone : maxvit_small_tf_224 (timm)
             Hybrid multi-axis vision transformer — interleaves local window
             attention, dilated grid attention, and MBConv blocks across 4
             hierarchical stages.  Feature dim = 768.
             Pretrained on ImageNet-1k (timm: maxvit_small_tf_224.in1k).
             Occupies a useful middle ground between ConvNeXt V2-B (~88 M,
             pure CNN) and DINOv2 ViT-B/14 (~86 M, pure ViT) — exploring
             whether the hybrid inductive bias aids SLB lesion detection.
  Head     : Dropout → Linear(768 → 256) → GELU → Dropout → Linear(256 → 1)

  Parameter count : ~69 M  (between DINOv2 ViT-S/14 ~22 M and ViT-B/14 ~86 M)
  Input size      : 224×224  (native training resolution for tf_224 variant)

Differential learning rates
-----------------------------
  get_optimizer_param_groups() separates:
    - backbone (all MaxViT stages)  → lower LR (default: 2e-5)
    - regression head               → higher LR (default: 2e-4)

  Mirrors the convention in convnextv2.py and dinov2_vitb14.py for a clean,
  fair architectural comparison.

Usage
-----
  from src.models.maxvit_small import create_maxvit_small

  model = create_maxvit_small(
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
        "timm is required for MaxViT.  Install with:  pip install timm"
    ) from e

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAXVIT_SMALL_FEATURE_DIM = 768   # num_features for maxvit_small_tf_224 in timm
MAXVIT_SMALL_TIMM_NAME   = "maxvit_small_tf_224"


# ===========================================================================
# Regression head
# ===========================================================================

class RegressionHead(nn.Module):
    """Two-layer MLP regression head with dropout and GELU activation.

    Identical in structure to the head used in ConvNeXtV2Regressor and the
    DINOv2 wrappers so that the only variable across model comparisons is the
    backbone.
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

class MaxViTSmallRegressor(nn.Module):
    """
    MaxViT-S backbone + regression head for disease severity scoring.

    MaxViT (Multi-Axis Vision Transformer) combines local window attention,
    dilated grid attention, and depthwise conv (MBConv) in each stage,
    giving it strong local-global feature extraction — potentially useful
    for capturing both individual SLB lesions (local) and plot-level disease
    gradients (global) within a single forward pass.

    Parameters
    ----------
    pretrained : bool
        Load timm's default pretrained weights (ImageNet-1k).
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
            MAXVIT_SMALL_TIMM_NAME,
            pretrained=pretrained,
            num_classes=0,      # Remove classifier; returns (B, 768) after GAP
            global_pool="avg",  # Global average pooling over the spatial grid
        )
        feature_dim = self.backbone.num_features  # 768 for maxvit_small_tf_224
        logger.info(
            f"MaxViT-S backbone loaded  |  pretrained={pretrained}  "
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
        features = self.backbone(x)   # (B, 768)
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

        Bias and norm parameters are excluded from weight decay in each group,
        matching the ConvNeXt V2 and DINOv2 training conventions.
        """
        backbone_params = list(self.backbone.parameters())
        head_params     = list(self.head.parameters())

        def _split_wd(params):
            decay, no_decay = [], []
            for p in params:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1:      # bias, LayerNorm/BatchNorm scale/shift
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

def create_maxvit_small(
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
    hidden_dim: int = 256,
) -> MaxViTSmallRegressor:
    """
    Instantiate a MaxViT-S regressor.

    Parameters
    ----------
    pretrained     : Load ImageNet-1k pretrained weights via timm.
    dropout_rate   : Dropout probability in the regression head.
    freeze_backbone: Freeze backbone for linear-probe evaluation.
    hidden_dim     : Hidden units in the two-layer MLP head.

    Returns
    -------
    MaxViTSmallRegressor
    """
    model = MaxViTSmallRegressor(
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        hidden_dim=hidden_dim,
    )
    n_total     = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"MaxViT-S Regressor  |  total={n_total:.1f}M  trainable={n_trainable:.1f}M")
    return model
