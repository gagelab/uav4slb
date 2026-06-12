"""
src/models/coatnet2.py
----------------------
CoAtNet-2 regression wrapper for UAV-based SLB disease severity prediction.

Architecture
------------
  Backbone : coatnet_2_rw_224.sw_in12k_ft_in1k (timm)
             Hybrid CNN+Transformer with 5 stages (S0–S4):
               S0 : 2× Conv stem (stride 2)
               S1 : MBConv blocks (depthwise conv, SE, stride 2)
               S2 : MBConv blocks (depthwise conv, SE, stride 2)
               S3 : Transformer blocks (relative attention, stride 2)
               S4 : Transformer blocks (relative attention, stride 2)
             Feature dim = 1024.  ~75 M params.
             Pretrained on ImageNet-12k (sw_in12k), fine-tuned on ImageNet-1k
             at 224×224.
  Head     : Dropout → Linear(1024 → 256) → GELU → Dropout → Linear(256 → 1)

CoAtNet design philosophy
---------------------------
  CoAtNet ("Combining Attention and Convolution Networks") uses depthwise
  convolution in early stages where local texture matters most, and global
  self-attention in deeper stages where long-range relationships are important.
  The transition point (MBConv → Transformer at S2/S3) is tuned so that
  convolutional stages handle fine-grained feature extraction while attention
  stages integrate global context — a different mixing strategy to MaxViT
  (which interleaves both in every stage) and SwinV2 (which uses only local
  shifted-window attention throughout).

  For SLB severity prediction this is appealing: early stages capture individual
  lesion texture via conv inductive biases; later stages integrate plot-level
  disease gradients via global attention on the downsampled feature grid.

Relationship to other models in this study
-------------------------------------------
  CoAtNet-2 is the second hybrid CNN+Transformer in the study after MaxViT-S,
  but uses a sequential (staged) rather than interleaved mixing strategy.
  Key comparisons:
    CoAtNet-2 vs. MaxViT-S      — staged vs. interleaved CNN+Transformer hybrid
                                   at ~69–75 M param range
    CoAtNet-2 vs. SwinV2-B      — global ViT stages vs. local shifted-window
                                   stages in the Transformer half
    CoAtNet-2 vs. ConvNeXt V2-B — staged hybrid vs. pure CNN at ~75–88 M scale

  Full architecture comparison
    EfficientNet-V2-S  : ~21 M,  384×384, supervised
    DINOv2 ViT-S/14   : ~22 M,  448×448, self-supervised
    MaxViT-S           : ~69 M,  224×224, supervised (IN-1k)
    CoAtNet-2          : ~75 M,  224×224, supervised (IN-12k → IN-1k) ← this
    EVA-02-B           : ~86 M,  448×448, CLIP-MIM
    DINOv2 ViT-B/14   : ~86 M,  518×518, self-supervised
    ConvNeXt V2-B     : ~88 M,  224×224, FCMAE self-supervised
    SwinV2-B           : ~88 M,  256×256, supervised (IN-22k → IN-1k)
    ConvNeXt V2-L     : ~198 M, 224×224, FCMAE self-supervised

Differential learning rates
-----------------------------
  get_optimizer_param_groups() separates:
    - backbone (all CoAtNet stages)  → lower LR (default: 2e-5)
    - regression head                → higher LR (default: 2e-4)

  Mirrors the convention used across all models in this study.

Usage
-----
  from src.models.coatnet2 import create_coatnet2

  model = create_coatnet2(
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
        "timm is required for CoAtNet.  Install with:  pip install timm"
    ) from e

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COATNET2_FEATURE_DIM = 1024
COATNET2_TIMM_NAME   = "coatnet_2_rw_224.sw_in12k_ft_in1k"


# ===========================================================================
# Regression head
# ===========================================================================

class RegressionHead(nn.Module):
    """Two-layer MLP regression head with dropout and GELU activation.

    Structurally identical to the head used across all other models in this
    study — the only variable in architectural comparisons is the backbone.
    in_features is 1024 for CoAtNet-2, matching ConvNeXt V2-B and SwinV2-B.
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

class CoAtNet2Regressor(nn.Module):
    """
    CoAtNet-2 backbone + regression head for disease severity scoring.

    CoAtNet-2 processes images through a staged hybrid pipeline:
      - Conv stem + two MBConv stages for local texture and edge features
      - Two Transformer stages with global relative-position-biased attention
        for long-range spatial context
    Global average pooling collapses the final 7×7 feature grid to a 1024-dim
    vector, which is then passed to the regression head.

    Parameters
    ----------
    pretrained : bool
        Load timm's default pretrained weights (IN-12k sw → IN-1k FT).
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
            COATNET2_TIMM_NAME,
            pretrained=pretrained,
            num_classes=0,       # Remove classifier; returns (B, 1024) after GAP
            global_pool="avg",   # Global average pooling over the 7×7 final grid
        )
        feature_dim = self.backbone.num_features  # 1024 for coatnet_2_rw_224
        logger.info(
            f"CoAtNet-2 backbone loaded  |  pretrained={pretrained}  "
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
            Input should be a multiple of 32 (5 stride-2 operations from stem
            through stage 4).  224 is the native pretrained resolution.

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

        Bias and LayerNorm scale terms are excluded from weight decay.
        CoAtNet uses LayerNorm in Transformer stages and BatchNorm in MBConv
        stages; both produce 1-D scale/bias vectors that should be no-decay.
        """
        backbone_params = list(self.backbone.parameters())
        head_params     = list(self.head.parameters())

        def _split_wd(params):
            decay, no_decay = [], []
            for p in params:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1:   # bias, LayerNorm/BatchNorm scale or shift
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

def create_coatnet2(
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
    hidden_dim: int = 256,
) -> CoAtNet2Regressor:
    """
    Instantiate a CoAtNet-2 regressor.

    Parameters
    ----------
    pretrained     : Load IN-12k sw → IN-1k pretrained weights via timm.
    dropout_rate   : Dropout probability in the regression head.
    freeze_backbone: Freeze backbone for linear-probe evaluation.
    hidden_dim     : Hidden units in the two-layer MLP head.

    Returns
    -------
    CoAtNet2Regressor
    """
    model = CoAtNet2Regressor(
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        hidden_dim=hidden_dim,
    )
    n_total     = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"CoAtNet-2 Regressor  |  total={n_total:.1f}M  trainable={n_trainable:.1f}M")
    return model
