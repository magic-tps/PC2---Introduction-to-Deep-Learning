from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torchvision import models


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        padding: int = 1,
        batch_norm: bool = False,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class LeNet5Adapted(nn.Module):
    """LeNet-5 adapted to 64x64 RGB images and 4 blood-cell classes."""

    def __init__(self, num_classes: int = 4, batch_norm: bool = False) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 6, kernel_size=5, padding=0, batch_norm=batch_norm),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(6, 16, kernel_size=5, padding=0, batch_norm=batch_norm),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 13 * 13, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class VGG11Small(nn.Module):
    """VGG-11 with half filters: 32, 64, 128, 256, 256."""

    def __init__(self, num_classes: int = 4, batch_norm: bool = False, dropout: float = 0.5) -> None:
        super().__init__()
        cfg: list[int | str] = [32, "M", 64, "M", 128, 128, "M", 256, 256, "M", 256, 256, "M"]
        self.features = self._make_layers(cfg, batch_norm=batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _make_layers(cfg: Iterable[int | str], batch_norm: bool) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_channels = 3
        for item in cfg:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue
            out_channels = int(item)
            layers.append(ConvBlock(in_channels, out_channels, batch_norm=batch_norm))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_model(name: str, num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    key = name.lower()
    if key == "lenet":
        return LeNet5Adapted(num_classes=num_classes, batch_norm=False)
    if key == "lenet_bn":
        return LeNet5Adapted(num_classes=num_classes, batch_norm=True)
    if key == "vgg11":
        return VGG11Small(num_classes=num_classes, batch_norm=False)
    if key == "vgg11_bn":
        return VGG11Small(num_classes=num_classes, batch_norm=True)
    if key in {"resnet18_feature", "resnet18_partial", "resnet18_full"}:
        return build_resnet18_transfer(num_classes=num_classes, strategy=key.split("_", 1)[1], pretrained=pretrained)
    if key in {"vgg16_feature", "vgg16_partial", "vgg16_full"}:
        return build_vgg16_transfer(num_classes=num_classes, strategy=key.split("_", 1)[1], pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def build_resnet18_transfer(num_classes: int, strategy: str, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if strategy == "feature":
        _set_trainable(model, False)
        _set_trainable(model.fc, True)
    elif strategy == "partial":
        _set_trainable(model, False)
        _set_trainable(model.layer3, True)
        _set_trainable(model.layer4, True)
        _set_trainable(model.fc, True)
    elif strategy == "full":
        _set_trainable(model, True)
    else:
        raise ValueError(f"Unknown transfer strategy: {strategy}")
    return model


def build_vgg16_transfer(num_classes: int, strategy: str, pretrained: bool = True) -> nn.Module:
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg16(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    if strategy == "feature":
        _set_trainable(model.features, False)
    elif strategy == "partial":
        _set_trainable(model.features[:10], False)
        _set_trainable(model.features[10:], True)
    elif strategy == "full":
        _set_trainable(model, True)
    else:
        raise ValueError(f"Unknown transfer strategy: {strategy}")
    _set_trainable(model.classifier, True)
    return model


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str


FROM_SCRATCH_MODELS = (
    ModelSpec("lenet", "LeNet"),
    ModelSpec("lenet_bn", "LeNet+BN"),
    ModelSpec("vgg11", "VGG-11 simplificado"),
    ModelSpec("vgg11_bn", "VGG-11 simplificado+BN"),
)

TRANSFER_MODELS = (
    ModelSpec("resnet18_feature", "ResNet-18 Feature Extraction"),
    ModelSpec("resnet18_partial", "ResNet-18 Fine-tuning parcial"),
    ModelSpec("resnet18_full", "ResNet-18 Fine-tuning total"),
)
