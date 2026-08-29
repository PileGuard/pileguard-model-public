"""ResNet-based piling image classifier."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(*, pretrained: bool, num_classes: int = 2) -> nn.Module:
    """Build the reference ResNet18 architecture with a binary classification head."""

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
