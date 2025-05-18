'''
* @name: ResNet-18
* @description: Pre-trained ResNet-18 model for image feature extraction.
* The model is used here as a method to extract visual features from video frames.
* Source: https://github.com/pytorch/vision/tree/main/torchvision/models
'''

import torch
from torch import nn
from .layer import Transformer, CrossTransformerEncoder
from torchvision import models


class VisionFeatureAggregator(nn.Module):
    """
    Aggregates a sequence of visual frame features using a Transformer encoder.

    Typically used after frame-wise CNN feature extraction to model temporal dependencies
    across visual tokens (e.g., from different time steps or regions).

    Args:
        dim (int): Dimensionality of each input visual feature vector

    Input:
        x (Tensor): Visual sequence of shape [B, T, D], where:
            - B: batch size
            - T: number of frames
            - D: feature dimension

    Returns:
        Tensor: Aggregated sequence of shape [B, T, D]
    """
    def __init__(self, dim):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=8),
            num_layers=1
        )

    def forward(self, x):  
        return self.transformer(x)  # Apply self-attention across frame sequence


class ResNetWithDropout(nn.Module):
    """
    ResNet-18 based feature extractor for images, with dropout applied to final features.

    Removes the final classification layer of ResNet-18 and adds dropout to the global average pooled features.
    Can be used as a backbone to extract features from individual video frames or images.

    Args:
        dropout_rate (float): Dropout probability for regularization (default: 0.3)

    Input:
        x (Tensor): Input images of shape [B, 3, 224, 224]

    Returns:
        Tensor: Extracted features of shape [B, 512]
    """
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet18.children())[:-1])  # Remove final FC layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.features(x)  # Shape: [B, 512, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # Flatten to [B, 512]
        x = self.dropout(x)
        return x