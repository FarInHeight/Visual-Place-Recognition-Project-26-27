"""ResNet descriptor scaffold with GeM or average pooling."""

import torch
import torch.nn.functional as F
from torchvision import models


class ResNetDescriptor(torch.nn.Module):
    def __init__(
        self,
        aggregation="gem",
        pretrained=False,
        backbone_training="full",
        backbone_name="resnet50",
    ):
        super().__init__()
        if backbone_training not in {"frozen", "full"}:
            raise ValueError(f"Unsupported backbone training mode: {backbone_training}")
        if aggregation not in {"gem", "avg"}:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.output_dim = 1024
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            self.output_dim = 256
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone_training = backbone_training
        self.backbone_name = backbone_name
        self.backbone = resnet
        self.aggregation = aggregation
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(backbone_training == "full")

    def forward(self, images):
        features = self.extract_features(images)
        descriptors = self.pool_features(features)
        return F.normalize(descriptors.flatten(1), p=2, dim=1)

    def extract_features(self, images):
        """Return the required spatial feature map from the ResNet backbone."""
        raise NotImplementedError(
            "Implement the required ResNet truncation and return its feature map."
        )

    def pool_features(self, features):
        """Aggregate a spatial feature map into one vector per image."""
        if self.aggregation == "avg":
            return F.adaptive_avg_pool2d(features, 1)
        return self.gem_pool(features)

    def gem_pool(self, features):
        """TODO: implement GeM pooling and return shape [batch, channels, 1, 1]."""
        raise NotImplementedError("Students should implement GeM pooling.")

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self
