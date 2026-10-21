"""Pretrained torchvision backbones with a 3-band head."""

from torch import nn
from torchvision import models

from hazeml.aqi import BANDS

ARCHS = ("efficientnet_b0", "mobilenet_v3_large")


def build_model(arch: str = "efficientnet_b0", pretrained: bool = True, num_outputs: int = len(BANDS)) -> nn.Module:
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_outputs)
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_outputs)
    else:
        raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")
    return model


def param_groups(model: nn.Module, lr_head: float, lr_backbone: float) -> list[dict]:
    """Higher LR for the new head, lower for the pretrained backbone."""
    head = list(model.classifier.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [{"params": backbone, "lr": lr_backbone}, {"params": head, "lr": lr_head}]
