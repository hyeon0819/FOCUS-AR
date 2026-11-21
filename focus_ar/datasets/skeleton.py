"""Skeleton-only clips for pretraining the BlockGCN encoder."""

import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def crop_resize(data, valid_len, p_interval, window):
    """Crop a temporal segment of the valid frames and resize it to ``window`` frames.

    ``p_interval`` is ``[p]`` (center crop of ratio p) or ``[p_min, p_max]`` (random crop ratio).
    """
    C, T, V, M = data.shape
    if valid_len <= 0:
        return np.zeros((C, window, V, M), dtype=data.dtype)
    if len(p_interval) == 1:
        bias = int((1 - p_interval[0]) * valid_len / 2)
        data = data[:, bias:valid_len - bias]
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        length = min(max(int(np.floor(valid_len * p)), 64), valid_len)
        bias = np.random.randint(0, valid_len - length + 1)
        data = data[:, bias:bias + length]
    length = data.shape[1]
    if length == 0:
        return np.zeros((C, window, V, M), dtype=data.dtype)
    x = torch.tensor(data, dtype=torch.float).permute(0, 2, 3, 1).reshape(1, 1, C * V * M, length)
    x = F.interpolate(x, size=(C * V * M, window), mode="bilinear", align_corners=False)
    return x.reshape(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()


class SkeletonClipDataset(Dataset):
    """Skeleton sequences of the videos listed in ``ann_file`` (videos without a skeleton are skipped)."""

    def __init__(self, skeleton_file, ann_file, num_person=5, window=16, train=True):
        self.num_person = num_person
        self.window = window
        self.p_interval = [0.5, 1.0] if train else [0.95]
        with open(skeleton_file, "rb") as f:
            anns = {a["frame_dir"]: a for a in pickle.load(f)["annotations"]}
        self.data, self.labels = [], []
        with open(ann_file) as f:
            for line in f:
                name, label = line.split()
                stem = Path(name).stem
                keys = [f"{Path(name).parent.name}/{stem}", stem, re.sub(r"_\d{6}_\d{6}$", "", stem)]
                key = next((k for k in keys if k in anns), None)
                if key is not None:
                    self.data.append(self._preprocess(anns[key]))
                    self.labels.append(int(label))

    def _preprocess(self, ann):
        """(P, T, 17, 2) keypoints -> (3, T, 17, M): normalized (x, y, score) of the top-M persons."""
        kp = np.array(ann["keypoint"], dtype=np.float32)
        score = np.array(ann["keypoint_score"], dtype=np.float32)
        h, w = ann.get("img_shape", (256, 340))
        kp[..., 0] /= w
        kp[..., 1] /= h
        top = np.argsort(-score.mean(axis=(1, 2)))[:self.num_person]
        data = np.concatenate([kp[top], score[top][..., None]], axis=-1)
        if len(top) < self.num_person:
            pad = np.zeros((self.num_person - len(top),) + data.shape[1:], dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)
        return data.transpose(3, 1, 2, 0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        valid_len = int(np.sum(data.sum(0).sum(-1).sum(-1) != 0)) or data.shape[1]
        data = crop_resize(data, valid_len, self.p_interval, self.window)
        # Coordinates relative to joint 0, which keeps its absolute trajectory.
        root = data[:, :, 0].copy()
        data = data - data[:, :, 0:1]
        data[:, :, 0] = root
        return torch.from_numpy(data), self.labels[idx]
