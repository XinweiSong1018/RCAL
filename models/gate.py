import torch
from torch import nn
from .layer import CrossTransformer
from torch.distributions import Bernoulli



class CrossAttentionTA(nn.Module):
    def __init__(self, dim=128, heads=8, mlp_dim=128):
        super().__init__()
        self.cross_attn = CrossTransformer(source_num_frames=8, tgt_num_frames=8, dim=dim, depth=1, heads=heads, mlp_dim=mlp_dim)

    def forward(self, h_t, h_a):
        out = self.cross_attn(h_t, h_a)  # [B, 9, D]
        return out  # keep [CLS] and body for downstream split

class GateController(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, state):
        temperature = 1.0
        p = torch.sigmoid(self.fc(state) / temperature)  
        dist = Bernoulli(probs=p)
        action = dist.sample()  # [B, 1]
        log_prob = dist.log_prob(action)  # [B, 1]
        return action, log_prob, p