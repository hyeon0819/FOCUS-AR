# Token merging used to summarize context tokens in the TC-CLIP vision encoder.
# Adapted from https://github.com/facebookresearch/ToMe
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F


def parse_r(num_layers: int, r: int, inflect: float = 0.0) -> List[int]:
    """Per-layer number of merged tokens."""
    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)
    return [int(min_val + step * i) for i in range(num_layers)]


def schedule_r_constant(start_number: int, final_number: int, r: int) -> List[int]:
    """Merge schedule that reduces ``start_number`` tokens to ``final_number``."""
    cnt = start_number
    r_list = []
    for _ in range(int((start_number - final_number) / r)):
        r_list.append(min(r, cnt // 2))
        cnt -= r_list[-1]
    while cnt > final_number:
        r_list.append(min(cnt - final_number, cnt // 2))
        cnt -= r_list[-1]
    return r_list


def bipartite_soft_matching(metric: torch.Tensor, r: int) -> Callable:
    """Balanced bipartite matching; returns a function merging ``r`` tokens."""
    t = metric.shape[1]
    r = min(r, t // 2)
    if r <= 0:
        return lambda x: x

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True, stable=True)[..., None]
        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        # (n, t2, r) assignment of merged source tokens to destination tokens
        assign = F.one_hot(dst_idx[..., 0], b.shape[1]).transpose(1, 2).float()

    def merge(x: torch.Tensor) -> torch.Tensor:
        """Sum each merged source token into its destination (deterministic, unlike scatter_add)."""
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = (dst.float() + torch.bmm(assign, src.float())).to(x.dtype)
        return torch.cat([unm, dst], dim=1)

    return merge


def merge_wavg(merge: Callable, x: torch.Tensor, size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Size-weighted average merge; returns merged tokens and their sizes."""
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    x = merge(x * size)
    size = merge(size)
    return x / size, size
