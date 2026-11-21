import random
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .blending import CutmixMixupBlending
from .video_skeleton import VideoSkeletonDataset, load_class_descriptions

__all__ = ["CutmixMixupBlending", "VideoSkeletonDataset", "build_loader", "load_class_descriptions"]


def _worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.set_num_threads(1)


def build_loader(cfg, ann_file, label_file, video_root, skeleton_file, train, num_clips=1):
    """DataLoader over ``ann_file``; training shards with a shuffling DistributedSampler."""
    dataset = VideoSkeletonDataset(
        ann_file=ann_file, video_root=video_root, label_file=label_file, skeleton_file=skeleton_file,
        num_person=cfg.skeleton.num_person, num_frames=cfg.num_frames, num_clips=num_clips, train=train,
        input_size=cfg.input_size, color_jitter=cfg.aug.color_jitter, gray_scale=cfg.aug.gray_scale,
        cache_dir=cfg.data.cache_dir, seed=cfg.seed)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if train:
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        batch_size = cfg.batch_size
    else:
        sampler = list(range(rank, len(dataset), world_size))
        batch_size = cfg.test_batch_size
    num_workers = cfg.num_workers
    return DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        drop_last=train, persistent_workers=train and num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=partial(_worker_init_fn, num_workers=num_workers, rank=rank, seed=cfg.seed))
