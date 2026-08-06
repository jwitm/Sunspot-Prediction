"""Image-level convolutional neural network for sunspot classification.

This module wraps an ImageNet-pretrained VGG or ResNet backbone and appends a
small fully connected classifier that produces logits for the two classes used
by the project: no sunspot and sunspot.
"""

from torchvision.models import vgg16, resnet18, resnet34, resnet50, resnet101, resnet152, ResNet18_Weights, \
    ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights, VGG16_Weights
import torch.nn as nn


class Sunspot_CNN(nn.Module):
    """Classify individual solar images with a pretrained CNN backbone.

    The selected torchvision model produces a 1,000-element feature vector.
    A two-layer classification head maps that representation to two unnormalized
    class scores.
    """

    def __init__(self, base='vgg16', dropout=0.2):
        """Initialize the image classifier.

        Args:
            base: Backbone architecture. Supported values are ``"vgg16"``,
                ``"resnet18"``, ``"resnet34"``, ``"resnet50"``,
                ``"resnet101"``, and ``"resnet152"``.
            dropout: Dropout probability used in the classification head.

        Raises:
            ValueError: If ``base`` does not name a supported architecture.
        """
        super().__init__()

        # this model is used in thesis
        if base == 'vgg16':
            self.CNN = vgg16(weights=VGG16_Weights.DEFAULT)
        elif base == 'resnet18':
            self.CNN = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif base == 'resnet34':
            self.CNN = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif base == 'resnet50':
            self.CNN = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif base == 'resnet101':
            self.CNN = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif base == 'resnet152':
            self.CNN = resnet152(weights=ResNet152_Weights.DEFAULT)
        else:
            raise ValueError(
                'Invalid base model choose between "vgg16", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"')

        self.fully_connected = nn.Sequential(
            nn.Linear(1000, 100),
            nn.LayerNorm(normalized_shape=[(100)]),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(100, 2))

    def forward(self, x):
        """Compute class logits for a batch of images.

        Args:
            x: Image batch in the input format expected by the selected
                torchvision backbone.

        Returns:
            A tensor shaped ``[batch_size, 2]`` containing unnormalized class
            scores.
        """
        x = self.CNN(x)
        x = self.fully_connected(x)
        return x
