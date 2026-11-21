"""BlockGCN skeleton encoder for COCO-17 skeleton sequences.

Adapted from BlockGCN (https://github.com/ZhouYuxuanYX/BlockGCN) without the
persistent-homology topology encoding.

Input : (N, C=3, T, V=17, M) skeleton sequence
Output: (N, 256) feature from ``encode``
"""

import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

# ---- Inline COCO-17 graph ------------------------------------------------

_NUM_NODE = 17
_SELF_LINK = [(i, i) for i in range(_NUM_NODE)]
_INWARD = [
    (2, 0), (4, 2), (1, 0), (3, 1),
    (6, 0), (8, 6), (10, 8), (5, 0), (7, 5), (9, 7),
    (12, 6), (14, 12), (16, 14), (11, 5), (13, 11), (15, 13),
]
_OUTWARD = [(j, i) for (i, j) in _INWARD]
_COCO_PAIRS = (
    (0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0),
    (7, 5), (8, 6), (9, 7), (10, 8), (11, 5), (12, 6),
    (13, 11), (14, 12), (15, 13), (16, 14),
)


def _edge2mat(link, n):
    A = np.zeros((n, n))
    for i, j in link:
        A[j, i] = 1
    return A


def _normalize_digraph(A):
    Dl = np.sum(A, 0)
    Dn = np.zeros_like(A)
    for i in range(A.shape[0]):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return A @ Dn


def _get_spatial_graph():
    I = _edge2mat(_SELF_LINK, _NUM_NODE)
    In = _normalize_digraph(_edge2mat(_INWARD, _NUM_NODE))
    Out = _normalize_digraph(_edge2mat(_OUTWARD, _NUM_NODE))
    return np.stack((I, In, Out))


# ---- Building blocks ------------------------------------------------------

def _conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def _bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def _weights_init(m):
    cn = m.__class__.__name__
    if cn.find("Conv") != -1:
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if hasattr(m, "bias") and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif cn.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, (kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1),
                              dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class MultiScale_TemporalConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,
                 dilations=(1, 2, 3, 4), residual=False, residual_kernel_size=1):
        super().__init__()
        n_branches = len(dilations) + 2
        assert out_ch % n_branches == 0
        branch_ch = out_ch // n_branches
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, branch_ch, 1), nn.BatchNorm2d(branch_ch), nn.ReLU(True),
                TemporalConv(branch_ch, branch_ch, ks, stride=stride, dilation=d),
            )
            for ks, d in zip(kernel_size, dilations)
        ])
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 1), nn.BatchNorm2d(branch_ch), nn.ReLU(True),
            nn.MaxPool2d((3, 1), (stride, 1), (1, 0)), nn.BatchNorm2d(branch_ch),
        ))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 1, stride=(stride, 1)), nn.BatchNorm2d(branch_ch),
        ))

        if not residual:
            self.residual = lambda x: 0
        elif in_ch == out_ch and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_ch, out_ch, residual_kernel_size, stride=stride)
        self.apply(_weights_init)

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], 1) + self.residual(x)


class UnitGCN(nn.Module):
    def __init__(self, in_ch, out_ch, A, adaptive=True, alpha=False):
        super().__init__()
        self.out_c, self.in_c = out_ch, in_ch
        self.num_heads = 8 if in_ch > 8 else 1

        self.fc1 = nn.Parameter(torch.stack([
            torch.stack([torch.eye(A.shape[-1]) for _ in range(self.num_heads)], 0)
            for _ in range(3)
        ], 0), requires_grad=True)
        self.fc2 = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 1, groups=self.num_heads) for _ in range(3)
        ])

        if in_ch != out_ch:
            self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch))
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                _conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                _bn_init(m, 1)
        _bn_init(self.bn, 1e-6)

        h1 = A.sum(0)
        h1[h1 != 0] = 1
        h = [None] * A.shape[-1]
        h[0] = np.eye(A.shape[-1])
        h[1] = h1
        self.hops = 0 * h[0]
        for i in range(2, A.shape[-1]):
            h[i] = h[i - 1] @ h1.T
            h[i][h[i] != 0] = 1
        for i in range(A.shape[-1] - 1, 0, -1):
            if np.any(h[i] - h[i - 1]):
                h[i] = h[i] - h[i - 1]
                self.hops += i * h[i]
        self.hops = torch.tensor(self.hops).long()
        self.rpe = nn.Parameter(torch.zeros(3, self.num_heads, self.hops.max() + 1))

        self.alpha = nn.Parameter(torch.ones(1, self.num_heads, 1, 1, 1)) if alpha else 1

    def L2_norm(self, w):
        return torch.norm(w, 2, dim=-2, keepdim=True) + 1e-4

    def forward(self, x):
        N, C, T, V = x.size()
        y = None
        pos_emb = self.rpe[:, :, self.hops]
        for i in range(3):
            wn = self.L2_norm(self.fc1[i])
            w1 = self.fc1[i] / wn + pos_emb[i] / self.L2_norm(pos_emb[i])
            x_in = x.view(N, self.num_heads, C // self.num_heads, T, V)
            z = torch.einsum("nhctv,hvw->nhctw", x_in, w1).contiguous().view(N, -1, T, V)
            z = self.fc2[i](z)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_ch, out_ch, A, stride=1, residual=True,
                 adaptive=True, kernel_size=5, dilations=(1, 2), alpha=False):
        super().__init__()
        self.gcn1 = UnitGCN(in_ch, out_ch, A, adaptive=adaptive, alpha=alpha)
        self.tcn1 = MultiScale_TemporalConv(out_ch, out_ch, kernel_size=kernel_size,
                                             stride=stride, dilations=dilations, residual=False)
        self.relu = nn.ReLU(True)
        if not residual:
            self.residual = lambda x: 0
        elif in_ch == out_ch and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        return self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))


# ---- Main model -----------------------------------------------------------

class BlockGCN(nn.Module):
    """BlockGCN for COCO 17-joint skeleton sequences.

    Architecture: 10 TCN-GCN blocks (128→128→128→128→256→256→256→256→256→256).
    Output: 256-d feature per sample (mean over time and persons).
    """

    def __init__(self, num_class=60, num_point=17, num_person=2,
                 in_channels=3, drop_out=0, adaptive=True, alpha=False):
        super().__init__()
        A = _get_spatial_graph()
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person

        self.data_bn = nn.BatchNorm1d(num_person * 128 * num_point)
        self.to_joint_embedding = nn.Linear(in_channels, 128)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_point, 128))

        self.l1 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, alpha=alpha)
        self.l2 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, alpha=alpha)
        self.l3 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, alpha=alpha)
        self.l4 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, alpha=alpha)
        self.l5 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive, alpha=alpha)
        self.l6 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, alpha=alpha)
        self.l7 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, alpha=alpha)
        self.l8 = TCN_GCN_unit(256, 256, A, stride=2, adaptive=adaptive, alpha=alpha)
        self.l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, alpha=alpha)
        self.l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, alpha=alpha)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        _bn_init(self.data_bn, 1)
        self.drop_out = nn.Dropout(drop_out) if drop_out else lambda x: x

    def encode(self, x, return_map: bool = False):
        """Extract 256-d features before the classification head.

        Args:
            x: (N, C=3, T, V=17, M) skeleton sequence
            return_map: if True, also return the temporal feature map before pooling
        Returns:
            if return_map is False: (N, 256) global feature vector
            if return_map is True : tuple of
                global_feat (N, 256)         — mean-pooled over time and persons
                feat_map    (N, M, 256, T')  — joint-pooled feature map per frame,
                                               T' = T//4 (two stride-2 layers)
        """
        N, C, T, V, M = x.size()
        x = rearrange(x, "n c t v m -> (n m t) v c", m=M, v=V).contiguous()
        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]
        x = rearrange(x, "(n m t) v c -> n (m v c) t", m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = x.view(N, M, V, 128, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, 128, T, V)

        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5,
                      self.l6, self.l7, self.l8, self.l9, self.l10]:
            x = layer(x)

        # x: (N*M, C_new, T', V)  where T' = T//4, C_new = 256
        c_new = x.size(1)
        T_new = x.size(2)

        if return_map:
            # Pool over joints → (N*M, C_new, T')
            feat_map = x.mean(-1)                           # (N*M, C_new, T')
            feat_map = feat_map.view(N, M, c_new, T_new)   # (N, M, C_new, T')
            global_feat = x.view(N, M, c_new, -1).mean(3).mean(1)  # (N, C_new)
            return global_feat, feat_map

        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)  # mean over time and persons → (N, 256)
        return x

    def forward(self, x, y=None):
        feat = self.encode(x)
        feat = self.drop_out(feat)
        return self.fc(feat), feat
