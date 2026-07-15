"""
DINOv2 ViT-B/14 model for Southern Leaf Blight severity regression.

Wraps the facebook/dinov2-base (ViT-B/14) backbone with a regression head.
DINOv2 is loaded via torch.hub from facebookresearch/dinov2.

Key differences from EfficientNet workflow:
  - Native patch size 14 → preferred input 518×518 (14×37) or 224×224 (14×16)
  - Feature dim: 768 (CLS token from the last transformer block)
  - Two-phase LR: backbone gets 10× lower LR than regression head (common ViT practice)
  - ImageNet normalization statistics are identical to what DINOv2 was trained with
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DINOv2RegressionModel(nn.Module):
    """
    DINOv2 ViT-B/14 backbone with a regression head for disease severity scoring.

    Architecture
    ------------
    Backbone : DINOv2 ViT-B/14 (768-d CLS token)
    Head     : LayerNorm → Linear(768, 256) → GELU → Dropout → Linear(256, 1)

    The two-layer head gives the model capacity to adapt the self-supervised
    features to the 1-9 severity scale without overfitting.
    """

    FEATURE_DIM = 768  # ViT-B hidden dimension

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
            logger.info("Loading DINOv2 ViT-B/14 from torch.hub ...")
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitb14",
                pretrained=True,
            )
            logger.info("DINOv2 backbone loaded successfully.")
        else:
            logger.warning(
                "pretrained=False: loading DINOv2 architecture WITHOUT weights. "
                "Only use this for architecture inspection, not real training."
            )
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitb14",
                pretrained=False,
            )

        if freeze_backbone:
            logger.info("Freezing DINOv2 backbone (linear-probe mode).")
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ------------------------------------------------------------------
        # Regression head
        # ------------------------------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(self.FEATURE_DIM),
            nn.Linear(self.FEATURE_DIM, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1),
        )

        # Initialise head weights
        self._init_head()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_head(self):
        """Truncated-normal init for linear layers; const init for LN."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — H and W must be multiples of 14.
        Returns:
            Tensor of shape (B,) with predicted severity scores.
        """
        # DINOv2 forward_features returns the CLS token representation
        features = self.backbone.forward_features(x)["x_norm_clstoken"]  # (B, 768)
        out = self.head(features)  # (B, 1)
        return out.squeeze(1)       # (B,)

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
        weight_decay: float = 5e-2,
    ):
        """
        Return parameter groups with separate LRs for backbone and head.

        Using a 10× lower LR for the backbone is standard practice when
        fine-tuning large self-supervised ViTs to avoid destroying pretrained
        representations early in training.

        Args:
            backbone_lr: LR for backbone parameters (e.g. 3e-5).
            head_lr:     LR for regression head (e.g. 3e-4).
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
            f"DINOv2RegressionModel(\n"
            f"  backbone=DINOv2 ViT-B/14 (feature_dim={self.FEATURE_DIM}),\n"
            f"  head={self.head},\n"
            f"  total_params={total:,},\n"
            f"  trainable_params={trainable:,}\n"
            f")"
        )


def create_dinov2_vitb14(
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
) -> DINOv2RegressionModel:
    """
    Factory function — mirrors create_efficientnetv2_s() in efficientnet.py.

    Args:
        pretrained:      Load ImageNet-pretrained DINOv2 weights.
        dropout_rate:    Head dropout probability.
        freeze_backbone: Freeze backbone for linear-probe experiments.

    Returns:
        DINOv2RegressionModel instance.
    """
    model = DINOv2RegressionModel(
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
    import sys

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = create_dinov2_vitb14(pretrained=True, dropout_rate=0.3)
    model = model.to(device)
    model.eval()

    # Test with 518×518 (recommended) and 224×224
    for h, w in [(518, 518), (224, 224)]:
        x = torch.randn(2, 3, h, w, device=device)
        with torch.no_grad():
            out = model(x)
        print(f"Input {h}×{w}  →  output shape: {out.shape}, values: {out}")

    # Check param groups
    groups = model.get_optimizer_param_groups(backbone_lr=3e-5, head_lr=3e-4)
    for g in groups:
        n_params = sum(p.numel() for p in g["params"])
        print(f"  Group '{g['name']}': {n_params:,} params, lr={g['lr']}")

    print("\n✓ DINOv2 smoke-test passed.")
