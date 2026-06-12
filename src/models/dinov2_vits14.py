"""
DINOv2 ViT-S/8 model for Southern Leaf Blight severity regression.

Wraps the DINOv2 ViT-S/8 backbone with a regression head.
Loaded via torch.hub from facebookresearch/dinov2.

Key differences from ViT-B/14:
  - Patch size 8  → preferred input 448×448 (8×56) or 224×224 (8×28)
  - Feature dim : 384  (vs 768 for ViT-B/14) — smaller, faster backbone
  - More patches per image at the same resolution → finer spatial granularity
  - Lower VRAM footprint allows larger batch sizes or higher resolution

ViT-S/8 vs ViT-B/14 trade-offs
  - (+) ~4× fewer backbone parameters (21 M vs 86 M)
  - (+) patch-8 gives 56×56 = 3136 attention patches at 448 px
       (ViT-B/14 gives 37×37 = 1369 patches at 518 px)
  - (-) narrower feature space may capture less global context
  - Lesion-level interpretability may actually improve due to smaller patches

Author: Cole Hammett
Date: February 2026
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DINOv2vits14RegressionModel(nn.Module):
    """
    DINOv2 ViT-S/8 backbone with a regression head for disease severity scoring.

    Architecture
    ------------
    Backbone : DINOv2 ViT-S/8 (384-d CLS token)
    Head     : LayerNorm → Linear(384, 256) → GELU → Dropout → Linear(256, 1)
    """

    FEATURE_DIM = 384  # ViT-S hidden dimension

    def __init__(
        self,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
        pretrained: bool = True,
    ):
        """
        Args:
            dropout_rate:     Dropout probability in the regression head.
            freeze_backbone:  If True, backbone weights are frozen (linear-probe mode).
            pretrained:       Load pretrained DINOv2 weights via torch.hub.
        """
        super().__init__()

        # ------------------------------------------------------------------
        # Backbone
        # ------------------------------------------------------------------
        if pretrained:
            logger.info("Loading DINOv2 ViT-S/8 from torch.hub ...")
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                pretrained=True,
            )
            logger.info("DINOv2 ViT-S/8 backbone loaded successfully.")
        else:
            logger.warning(
                "pretrained=False: loading DINOv2 ViT-S/8 architecture WITHOUT weights. "
                "Only use this for architecture inspection, not real training."
            )
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                pretrained=False,
            )

        if freeze_backbone:
            logger.info("Freezing DINOv2 ViT-S/8 backbone (linear-probe mode).")
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ------------------------------------------------------------------
        # Regression head — same topology as ViT-B/14 but 384-d input
        # ------------------------------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(self.FEATURE_DIM),
            nn.Linear(self.FEATURE_DIM, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1),
        )

        self._init_head()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_head(self):
        """Xavier-uniform init for linear layers; const init for LN."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — H and W must be multiples of 8.
        Returns:
            Tensor of shape (B,) with predicted severity scores.
        """
        # DINOv2 forward_features returns the CLS token representation
        features = self.backbone.forward_features(x)["x_norm_clstoken"]  # (B, 384)
        out = self.head(features)  # (B, 1)
        return out.squeeze(1)      # (B,)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_optimizer_param_groups(
        self,
        backbone_lr: float,
        head_lr: float,
        weight_decay: float = 1e-4,
    ):
        """
        Return parameter groups with separate LRs for backbone and head.

        A 10× LR differential between backbone and head is preserved from the
        ViT-B/14 baseline, keeping the comparison apples-to-apples.

        Args:
            backbone_lr: LR for backbone parameters (e.g. 5e-5).
            head_lr:     LR for regression head (e.g. 5e-4).
            weight_decay: Weight decay applied to all groups.

        Returns:
            List of param-group dicts suitable for torch.optim constructors.
        """
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.head.parameters())

        return [
            {
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
                "name": "backbone",
            },
            {
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "head",
            },
        ]

    def __repr__(self):
        total = self.get_num_parameters()
        trainable = self.get_num_parameters(trainable_only=True)
        return (
            f"DINOv2vits14RegressionModel(\n"
            f"  backbone=DINOv2 ViT-S/8 (feature_dim={self.FEATURE_DIM}),\n"
            f"  head={self.head},\n"
            f"  total_params={total:,},\n"
            f"  trainable_params={trainable:,}\n"
            f")"
        )


def create_dinov2_vits14(
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
) -> DINOv2vits14RegressionModel:
    """
    Factory function — mirrors create_dinov2_vitb14() in dinov2.py.

    Args:
        pretrained:      Load DINOv2-pretrained ViT-S/8 weights.
        dropout_rate:    Head dropout probability.
        freeze_backbone: Freeze backbone for linear-probe experiments.

    Returns:
        DINOv2vits14RegressionModel instance.
    """
    model = DINOv2vits14RegressionModel(
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        pretrained=pretrained,
    )
    logger.info(repr(model))
    return model


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = create_dinov2_vits14(pretrained=True, dropout_rate=0.3)
    model = model.to(device)
    model.eval()

    # Test with 448×448 (recommended for patch-8) and 224×224
    for h, w in [(448, 448), (224, 224)]:
        x = torch.randn(2, 3, h, w, device=device)
        with torch.no_grad():
            out = model(x)
        print(f"Input {h}×{w}  →  output shape: {out.shape}, values: {out}")

    # Patch count comparison
    print(f"\nPatch counts at recommended resolutions:")
    print(f"  ViT-S/8  @ 448×448 : {(448 // 8) ** 2:,} patches  (56×56)")
    print(f"  ViT-B/14 @ 518×518 : {(518 // 14) ** 2:,} patches  (37×37)")

    # Check param groups
    groups = model.get_optimizer_param_groups(backbone_lr=5e-5, head_lr=5e-4)
    for g in groups:
        n_params = sum(p.numel() for p in g["params"])
        print(f"  Group '{g['name']}': {n_params:,} params, lr={g['lr']}")

    print("\n✓ DINOv2 ViT-S/8 smoke-test passed.")
