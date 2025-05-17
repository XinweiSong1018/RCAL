'''
* @name: layer.py
* @description: Basic layers used in the model architecture.
*               Some components are adapted from Vision Transformer (ViT): 
*               https://github.com/gupta-abhay/pytorch-vit
'''

import torch
from torch import nn, einsum
from einops import rearrange, repeat


def pair(t):
    """
    Ensures that the input is a tuple (used for flexible dimension specification).
    """
    return t if isinstance(t, tuple) else (t, t)


# ======== PreNorm Wrappers =========

class PreNormForward(nn.Module):
    """
    Applies LayerNorm before a FeedForward module.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormAttention(nn.Module):
    """
    Applies separate LayerNorm to Q, K, V before Attention.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, return_attn=False):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)
        return self.fn(q, k, v, return_attn=return_attn)


class PreNormAHL(nn.Module):
    """
    Applies LayerNorm before passing multimodal embeddings (T, A, V, H) to HyperLayer.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, h_t, h_a, h_v, h_hyper):
        h_t = self.norm1(h_t)
        h_a = self.norm2(h_a)
        h_v = self.norm3(h_v)
        h_hyper = self.norm4(h_hyper)
        return self.fn(h_t, h_a, h_v, h_hyper)


# ======== Core Modules =========

class FeedForward(nn.Module):
    """
    Standard two-layer feedforward network with GELU and dropout.
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head self-attention mechanism with scaled dot-product attention.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v, return_attn=False):
        b, n, _, h = *q.shape, self.heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), 
                      (self.to_q(q), self.to_k(k), self.to_v(v)))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return (out, attn) if return_attn else out


class HhyperLearningLayer(nn.Module):
    """
    Cross-attention layer that combines text, audio, and visual information to update the hyper feature.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_tv = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_tv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, h_t, h_a, h_v, h_hyper):
        # Compute query from text and key/values from audio and visual
        b, n, _, h = *h_t.shape, self.heads
        q = self.to_q(h_t)
        k_ta, v_ta = self.to_k_ta(h_a), self.to_v_ta(h_a)
        k_tv, v_tv = self.to_k_tv(h_v), self.to_v_tv(h_v)

        # Reshape to multi-head format
        q, k_ta, k_tv, v_ta, v_tv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), 
                                        (q, k_ta, k_tv, v_ta, v_tv))

        # Cross-attention from text to audio
        attn_ta = self.attend(einsum('b h i d, b h j d -> b h i j', q, k_ta) * self.scale)
        out_ta = rearrange(einsum('b h i j, b h j d -> b h i d', attn_ta, v_ta), 'b h n d -> b n (h d)')

        # Cross-attention from text to visual
        attn_tv = self.attend(einsum('b h i d, b h j d -> b h i j', q, k_tv) * self.scale)
        out_tv = rearrange(einsum('b h i j, b h j d -> b h i d', attn_tv, v_tv), 'b h n d -> b n (h d)')

        # Combine and update hyper embedding
        h_hyper += self.to_out(out_ta + out_tv)
        return h_hyper


# ======== Transformer Encoder Blocks =========

class HhyperLearningEncoder(nn.Module):
    """
    Stack of HhyperLearningLayer blocks for multimodal fusion.
    """
    def __init__(self, dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            PreNormAHL(dim, HhyperLearningLayer(dim, heads, dim_head, dropout))
            for _ in range(depth)
        ])

    def forward(self, h_t_list, h_a, h_v, h_hyper):
        for i, layer in enumerate(self.layers):
            h_hyper = layer(h_t_list[i], h_a, h_v, h_hyper)
        return h_hyper


class TransformerEncoder(nn.Module):
    """
    Standard Transformer encoder stack (self-attention + feedforward).
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads, dim_head, dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout))
            ])
            for _ in range(depth)
        ])

    def forward(self, x, save_hidden=False):
        hidden_list = [x] if save_hidden else None
        for attn, ff in self.layers:
            x = attn(x, x, x) + x
            x = ff(x) + x
            if save_hidden:
                hidden_list.append(x)
        return hidden_list if save_hidden else x


class CrossTransformerEncoder(nn.Module):
    """
    Transformer block where target attends to source (cross-attention encoder).
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads, dim_head, dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout))
            ])
            for _ in range(depth)
        ])

    def forward(self, source_x, target_x, return_attn=False):
        all_attn = []
        for attn, ff in self.layers:
            if return_attn:
                target_x_tmp, attn_weights = attn(target_x, source_x, source_x, return_attn=True)
                all_attn.append(attn_weights)
            else:
                target_x_tmp = attn(target_x, source_x, source_x)
            target_x = target_x_tmp + target_x
            target_x = ff(target_x) + target_x
        return (target_x, all_attn) if return_attn else target_x


# ======== Embedding + Positional Encoding Wrappers =========

class Transformer(nn.Module):
    """
    Vision-style transformer encoder with optional [CLS]-like extra token.

    Args:
        token_len (int): Number of extra tokens (e.g., 1 for [CLS])
        num_frames (int): Number of input frames (patches)
    """
    def __init__(self, *, num_frames, token_len, save_hidden, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.token_len = token_len
        self.save_hidden = save_hidden

        if token_len:
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
        else:
            self.extra_token = None
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, x):
        b, n, _ = x.shape
        if self.token_len:
            extra = repeat(self.extra_token, '1 n d -> b n d', b=b)
            x = torch.cat((extra, x), dim=1)
            x = x + self.pos_embedding[:, :n + self.token_len]
        else:
            x = x + self.pos_embedding[:, :n]

        x = self.dropout(x)
        return self.encoder(x, self.save_hidden)


class CrossTransformer(nn.Module):
    """
    CrossTransformer: performs attention from target to source with independent positional embeddings.

    Used for cross-modal alignment (e.g., text attending to audio or visual).
    """
    def __init__(self, *, source_num_frames, tgt_num_frames, dim, depth, heads, mlp_dim,
                 pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames + 1, dim))
        self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames + 1, dim))
        self.extra_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.CrossTransformerEncoder = CrossTransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, source_x, target_x, return_attn=False):
        b, n_s, _ = source_x.shape
        b, n_t, _ = target_x.shape
        extra = repeat(self.extra_token, '1 1 d -> b 1 d', b=b)

        source_x = torch.cat((extra, source_x), dim=1) + self.pos_embedding_s[:, :n_s + 1]
        target_x = torch.cat((extra, target_x), dim=1) + self.pos_embedding_t[:, :n_t + 1]

        source_x, target_x = self.dropout(source_x), self.dropout(target_x)

        if return_attn:
            return self.CrossTransformerEncoder(source_x, target_x, return_attn=True)
        else:
            return self.CrossTransformerEncoder(source_x, target_x)