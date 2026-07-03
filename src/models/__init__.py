"""
src/models/__init__.py
======================
Central model registry for the uav4slb project.

All nine architecture factories are registered here under the string keys
used in the YAML configs (``model.name``).  Any script that needs to
instantiate a model by name should import MODEL_REGISTRY rather than
importing individual model modules directly.

Usage
-----
    from src.models import MODEL_REGISTRY

    factory = MODEL_REGISTRY["eva02_base"]
    model   = factory(pretrained=True, dropout_rate=0.3, freeze_backbone=False)

Registry keys (must match ``model.name`` in the per-architecture YAML configs)
-----------------------------------------------------------------------------
    coatnet2            → CoAtNet-2 (hybrid CNN-ViT)
    convnextv2_base     → ConvNeXt V2-B (~88 M params)
    convnextv2_large    → ConvNeXt V2-L (~198 M params)
    dinov2_vitb14       → DINOv2 ViT-B/14 (~86 M params)
    dinov2_vits14       → DINOv2 ViT-S/14 (~22 M params)
    efficientnetv2_s    → EfficientNet V2-S (~21 M params)
    eva02_base          → EVA-02-B (~86 M params; CLIP-MIM pretrained)
    maxvit_small        → MaxViT-S (~69 M params)
    swinv2_base         → Swin Transformer V2-B (~88 M params)
"""

from src.models.coatnet2        import create_coatnet2
from src.models.convnextv2      import create_convnextv2_base
from src.models.convnextv2_large import create_convnextv2_large
from src.models.dinov2_vitb14   import create_dinov2_vitb14
from src.models.dinov2_vits14   import create_dinov2_vits14
from src.models.efficientnetv2s import create_efficientnetv2s
from src.models.eva02_base      import create_eva02_base
from src.models.maxvit_small    import create_maxvit_small
from src.models.swinv2_base     import create_swinv2_base

# ---------------------------------------------------------------------------
# Registry: model name (string) → factory callable
#
# Every factory accepts the same three keyword arguments so that
# train_cv0.py and predict_cv0.py can instantiate any model uniformly:
#   pretrained      (bool)  — load pretrained weights from timm hub
#   dropout_rate    (float) — regression head dropout probability
#   freeze_backbone (bool)  — freeze backbone parameters (linear-probe mode)
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict = {
    "coatnet2":         create_coatnet2,
    "convnextv2_base":  create_convnextv2_base,
    "convnextv2_large": create_convnextv2_large,
    "dinov2_vitb14":    create_dinov2_vitb14,
    "dinov2_vits14":    create_dinov2_vits14,
    "efficientnetv2_s": create_efficientnetv2s,
    "eva02_base":       create_eva02_base,
    "maxvit_small":     create_maxvit_small,
    "swinv2_base":      create_swinv2_base,
}

__all__ = ["MODEL_REGISTRY"]
