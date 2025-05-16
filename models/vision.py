import torch
from torch import nn
from .layer import Transformer,CrossTransformerEncoder
from torchvision import models



class VisionFeatureAggregator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=8),
            num_layers=1
        )

    def forward(self, x):  
        return self.transformer(x)
    
    
class ResNetWithDropout(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet18.children())[:-1])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        return x
