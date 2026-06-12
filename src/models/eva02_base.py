"""
src/models/eva02_base.py
------------------------
EVA-02-B regression wrapper for UAV-based SLB disease severity prediction.

Architecture
------------
  Backbone : eva02_base_patch14_448.mim_in22k_ft_in22k_in1k (timm)
             Pure ViT-B/14 with RoPE (Rotary Position Embedding) and SwiGLU
             FFN.  Feature dim = 768.  ~86 M params.
             Pretrained with CLIP-guided masked image modeling (MIM) on a
             merged-38M dataset, then fine-tuned on ImageNet-22k → ImageNet-1k.
             Input resolution: 448×448 → 32×32 = 1024 patches at patch size 14.
  Head     : Dropout → Linear(768 → 256) → GELU → Dropout → Linear(256 → 1)

Relationship to other models in this study
-------------------------------------------
  EVA-02-B and DINOv2 ViT-B/14 share the same ViT-B/14 macro-architecture
  (~86 M params, patch size 14, feature dim 768) but differ in three ways:
    1. Pretraining objective — EVA-02: CLIP-guided MIM;
                               DINOv2: self-distillation + iBOT MIM
    2. Input resolution     — EVA-02: 448px (32×32 patches);
                               DINOv2: 518px (37×37 patches)
    3. Positional encoding  — EVA-02: RoPE (relative, interpolation-free);
                               DINOv2: learnable absolute + interpolation
  This makes the EVA-02 vs. DINOv2 ViT-B/14 comparison a near-controlled
  experiment on pretraining objective and positional encoding strategy.

Differential learning rates
-----------------------------
  get_optimizer_param_groups() separates:
    - backbone (all ViT blocks)  → lower LR (default: 2e-5)
    - regression head            → higher LR (default: 2e-4)

  Mirrors the convention in dinov2_vitb14.py and convnextv2.py for a clean,
  fair architectural comparison across the full model suite.

Usage
-----
  from src.models.eva02_base import create_eva02_base

  model = create_eva02_base(
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
        "timm is required for EVA-02.  Install with:  pip install timm"
    ) from e

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVA02_BASE_FEATURE_DIM = 768
EVA02_BASE_TIMM_NAME   = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"


# ===========================================================================
# Regression head
# ===========================================================================

class RegressionHead(nn.Module):
    """Two-layer MLP regression head with dropout and GELU activation.

    Structurally identical to the head used in ConvNeXtV2Regressor,
    MaxViTSmallRegressor, and DINOv2 wrappers — the only variable across
    architectural comparisons is the backbone.
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

class EVA02BaseRegressor(nn.Module):
    """
    EVA-02-B backbone + regression head for disease severity scoring.

    EVA-02 combines RoPE positional embeddings (which generalize more cleanly
    to unseen resolutions than learned absolute positions) with CLIP-guided
    MIM pretraining, giving the model both rich semantic representations and
    a strong low-level texture prior.  These properties may complement SLB
    severity prediction, where both fine-grained lesion texture and higher-
    level plot structure matter.

    Parameters
    ----------
    pretrained : bool
        Load timm's default pretrained weights (IN-22k MIM → IN-22k FT → IN-1k FT).
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
            EVA02_BASE_TIMM_NAME,
            pretrained=pretrained,
            num_classes=0,       # Remove classifier head; returns (B, 768)
            global_pool="avg",   # Global average pool over the 32×32 patch grid
        )
        feature_dim = self.backbone.num_features  # 768 for eva02_base_patch14_448
        logger.info(
            f"EVA-02-B backbone loaded  |  pretrained={pretrained}  "
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

        Bias and LayerNorm/RMSNorm scale terms are excluded from weight decay
        in both groups, matching conventions across all models in this study.
        EVA-02 uses RMSNorm rather than LayerNorm in some sublayers; both
        produce 1-D scale vectors that should be excluded from weight decay.
        """
        backbone_params = list(self.backbone.parameters())
        head_params     = list(self.head.parameters())

        def _split_wd(params):
            decay, no_decay = [], []
            for p in params:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1:      # bias, LayerNorm/RMSNorm scale, 1-D gains
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

def create_eva02_base(
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
    hidden_dim: int = 256,
) -> EVA02BaseRegressor:
    """
    Instantiate an EVA-02-B regressor.

    Parameters
    ----------
    pretrained     : Load IN-22k MIM → IN-22k → IN-1k pretrained weights via timm.
    dropout_rate   : Dropout probability in the regression head.
    freeze_backbone: Freeze backbone for linear-probe evaluation.
    hidden_dim     : Hidden units in the two-layer MLP head.

    Returns
    -------
    EVA02BaseRegressor
    """
    model = EVA02BaseRegressor(
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        hidden_dim=hidden_dim,
    )
    n_total     = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"EVA-02-B Regressor  |  total={n_total:.1f}M  trainable={n_trainable:.1f}M")
    return model
