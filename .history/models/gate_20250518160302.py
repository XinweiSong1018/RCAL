import torch
from torch import nn
from .layer import CrossTransformer  # Custom implementation of a cross-attention transformer block
from torch.distributions import Bernoulli


class CrossAttentionTA(nn.Module):
    """
    Cross-attention module between text and audio embeddings.

    This module leverages a custom CrossTransformer that applies attention from one modality
    (e.g., text) to another (e.g., audio), allowing the model to capture inter-modal alignment.

    Args:
        dim (int): Embedding dimension for both text and audio features.
        heads (int): Number of attention heads.
        mlp_dim (int): Hidden size of the MLP within transformer blocks.

    Forward Inputs:
        h_t (Tensor): Text embeddings of shape (B, T_text, D)
        h_a (Tensor): Audio embeddings of shape (B, T_audio, D)

    Returns:
        Tensor: Cross-attended features of shape (B, T_text, D)
    """
    def __init__(self, dim=128, heads=8, mlp_dim=128):
        super().__init__()
        self.cross_attn = CrossTransformer(
            source_num_frames=8,   # Number of source frames (e.g., audio)
            tgt_num_frames=8,      # Number of target frames (e.g., text)
            dim=dim,
            depth=1,               # Number of transformer layers
            heads=heads,
            mlp_dim=mlp_dim
        )

    def forward(self, h_t, h_a):
        out = self.cross_attn(h_t, h_a)  # Apply cross-attention: text attends to audio
        return out  # Return output including CLS token for downstream use


class GateController(nn.Module):
    """
    Learnable gate controller that stochastically selects whether to use a specific feature or not.

    This module is used for selective modality fusion or routing decisions. It learns a probability
    distribution over actions (select / not select) using a Bernoulli distribution.

    Args:
        dim (int): Input feature dimension (D)

    Forward Inputs:
        state (Tensor): Feature representation of shape (B, D)

    Returns:
        action (Tensor): Sampled binary gate decisions (B, 1)
        log_prob (Tensor): Log probability of selected actions (for policy gradient)
        p (Tensor): Gate activation probability (B, 1)
    """
    def __init__(self, dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, state):
        temperature = 1.0
        p = torch.sigmoid(self.fc(state) / temperature)  # Get Bernoulli probability
        dist = Bernoulli(probs=p)  # Create stochastic distribution
        action = dist.sample()     # Sample action: 1 (select), 0 (skip)
        log_prob = dist.log_prob(action)  # Compute log-probability for RL optimization
        return action, log_prob, p