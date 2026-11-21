# CLIP text transformer and the temporally contextualized (TC) vision transformer.
# Adapted from CLIP (https://github.com/openai/CLIP, MIT License) and
# TC-CLIP (https://github.com/naver-ai/tc-clip, Copyright (c) 2024-present NAVER Cloud Corp., CC BY-NC 4.0).

from collections import OrderedDict

import torch
from einops import rearrange
from timm.layers import trunc_normal_
from torch import nn

from .tome import bipartite_soft_matching, merge_wavg, parse_r, schedule_r_constant


class LayerNorm(nn.LayerNorm):
    """LayerNorm computed in fp32."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        return super().forward(x.type(torch.float32)).type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def _mlp(d_model: int) -> nn.Sequential:
    return nn.Sequential(OrderedDict([
        ("c_fc", nn.Linear(d_model, d_model * 4)),
        ("gelu", QuickGELU()),
        ("c_proj", nn.Linear(d_model * 4, d_model)),
    ]))


# ---------------------------------------------------------------------------
# Text transformer
# ---------------------------------------------------------------------------
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = _mlp(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def forward(self, x: torch.Tensor):
        mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        y = self.ln_1(x)
        x = x + self.attn(y, y, y, need_weights=False, attn_mask=mask)[0]
        return x + self.mlp(self.ln_2(x))


class TextTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


# ---------------------------------------------------------------------------
# TC vision transformer
# ---------------------------------------------------------------------------
class TCAttention(nn.Module):
    """Temporally contextualized attention: frame-local tokens attend to shared context tokens."""

    def __init__(self, d_model: int, n_head: int, T: int, num_patches: int, seed_token_a: float,
                 first_layer: bool, local_global_bias: bool):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.num_heads = n_head
        self.head_dim = d_model // n_head
        self.T = T
        self.num_patches = num_patches
        self.top_s = int(num_patches * seed_token_a)  # seed tokens per frame
        self.first_layer = first_layer
        if not first_layer and local_global_bias:
            self.local_global_bias_table = nn.Parameter(torch.zeros(n_head, 1, 2))
            trunc_normal_(self.local_global_bias_table, std=.02)
        else:
            self.local_global_bias_table = None
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def get_seed_index(self, attn):
        B, T = attn.size(0) // self.T, self.T
        cls_attn = attn[:, :, 0, 1:1 + self.num_patches]
        cls_attn = cls_attn.reshape(B, T, self.num_heads, self.num_patches).mean(dim=2)
        return torch.topk(cls_attn, self.top_s, dim=2, largest=True, sorted=True)[1]

    def forward(self, x: torch.Tensor):
        B, _, C = x.shape
        T, N = self.T, 1 + self.num_patches
        BT, TN = B * T, T * N

        q = self.q_proj(x[:, :TN, :]).reshape(BT, N, C)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.first_layer:
            k = k.reshape(BT, N, C)
            v = v.reshape(BT, N, C)
        else:
            k_context = k[:, TN:, :].unsqueeze(1).repeat(1, T, 1, 1).reshape(BT, -1, C)
            v_context = v[:, TN:, :].unsqueeze(1).repeat(1, T, 1, 1).reshape(BT, -1, C)
            k = torch.cat([k[:, :TN, :].reshape(BT, N, C), k_context], dim=1)
            v = torch.cat([v[:, :TN, :].reshape(BT, N, C), v_context], dim=1)

        q = q.reshape(BT, q.size(1), self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(BT, k.size(1), self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(BT, v.size(1), self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q * self.head_dim ** -0.5) @ k.transpose(-2, -1)
        if self.local_global_bias_table is not None:
            local_bias = self.local_global_bias_table[:, :, 0:1].unsqueeze(0).repeat(
                BT, 1, attn.size(2), attn.size(2))
            global_bias = self.local_global_bias_table[:, :, 1:].unsqueeze(0).repeat(
                BT, 1, attn.size(2), attn.size(3) - attn.size(2))
            attn = attn + torch.cat([local_bias, global_bias], dim=-1)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, TN, C)
        return self.out_proj(x), self.get_seed_index(attn)


class TCAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, layer_idx: int, cfg: dict):
        super().__init__()
        self.T = cfg["temporal_length"]
        self.num_patches = cfg["num_patches"]
        self.num_context_token = cfg["context_token_k"]
        self.attn = TCAttention(d_model, n_head, T=self.T, num_patches=self.num_patches,
                                seed_token_a=cfg["seed_token_a"], first_layer=(layer_idx == 0),
                                local_global_bias=cfg["local_global_bias"])
        self.ln_1 = LayerNorm(d_model)
        self.mlp = _mlp(d_model)
        self.ln_2 = LayerNorm(d_model)

    def forward(self, x, merge_r: int):
        TN = self.T * (1 + self.num_patches)
        x_attn, seed_index = self.attn(self.ln_1(x))
        x = x[:, :TN, :] + x_attn
        x = torch.cat([x, self.summarize_context_tokens(x, seed_index, merge_r)], dim=1)
        return x + self.mlp(self.ln_2(x))

    def summarize_context_tokens(self, x, index, merge_r: int):
        """Gather seed tokens from all frames and merge them into context tokens."""
        B, T, D = x.size(0), self.T, x.size(-1)
        patch_tokens = rearrange(x, 'B (T N) D -> B T N D', T=T)[:, :, 1:1 + self.num_patches, :]
        Np, Ns = patch_tokens.size(2), index.size(2)

        offset = torch.arange(T, device=x.device).unsqueeze(0).unsqueeze(2).repeat(B, 1, Ns) * Np
        index_flatten = (index + offset).reshape(B, -1)
        patch_tokens = rearrange(patch_tokens, 'B T Np D -> B (T Np) D')
        tokens = patch_tokens.gather(dim=1, index=index_flatten.unsqueeze(-1).expand(-1, -1, D))

        size = None
        for r in schedule_r_constant(Ns * T, self.num_context_token, merge_r):
            merge = bipartite_soft_matching(tokens.clone(), r)
            tokens, size = merge_wavg(merge, tokens, size)
        return tokens


class TCVisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, cfg: dict):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.T = cfg["temporal_length"]
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.Sequential(*[TCAttentionBlock(width, heads, i, cfg) for i in range(layers)])
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.num_layers = layers
        self.tome_r = cfg["tome_r"]
        self.tome_d = cfg["tome_d"]

    def forward(self, x: torch.Tensor, return_layer_num):
        """
        Args:
            x: (B, T, 3, H, W) video frames.
            return_layer_num: indices of the layers whose outputs are returned.
        Returns:
            cls_tokens:     (B, n_layer, T, D) per-frame [CLS] embeddings.
            context_tokens: (B, n_layer, K, D) summarized context tokens.
        """
        B, T, C, H, W = x.size()
        x = self.conv1(x.reshape(-1, C, H, W))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(rearrange(x, '(B T) N D -> B (T N) D', B=B, T=T))

        merge_r = parse_r(self.num_layers, self.tome_r, self.tome_d)
        N = x.size(1) // T
        feats = []
        for i, block in enumerate(self.transformer.resblocks):
            x = block(x, merge_r[i])
            if i in return_layer_num:
                feat = rearrange(x[:, :T * N, :], 'B (T N) D -> B T N D', T=T, N=N)
                feats.append(torch.cat([feat[:, :, 0, :], x[:, T * N:, :]], dim=1))
        x = self.ln_post(torch.stack(feats, dim=1)) @ self.proj
        return x[:, :, :T, :], x[:, :, T:, :]
