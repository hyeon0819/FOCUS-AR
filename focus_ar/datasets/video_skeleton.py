"""RGB video + 2D skeleton dataset.

Annotation files list ``<relative video path> <label>`` per line. Skeletons are read from a pose
pickle (released on Hugging Face, see README) and packed once into a disk-backed memmap cache so that
large datasets (e.g. full Kinetics-400) can be shared across DataLoader workers and ranks.
"""

import hashlib
import json
import os
import pickle
import re
from pathlib import Path

import decord
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from .transforms import TestTransform, TrainTransform, sample_frames


def load_class_descriptions(label_file):
    """Class descriptions ordered by label id (``{"0": "...", "1": "...", ...}``)."""
    with open(label_file) as f:
        labels = json.load(f)
    return [labels[k] for k in sorted(labels, key=int)]


def _is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


class VideoSkeletonDataset(Dataset):
    def __init__(self, ann_file, video_root, label_file, skeleton_file, num_person=5, num_frames=16,
                 num_clips=1, train=True, input_size=224, color_jitter=0.8, gray_scale=0.2,
                 cache_dir="data/cache", seed=1024):
        self.num_person = num_person
        self.num_frames = num_frames
        self.num_clips = num_clips
        self.train = train
        self.seed = seed
        self.classes = load_class_descriptions(label_file)
        self.transform = TrainTransform(input_size, color_jitter, gray_scale) if train else TestTransform(input_size)

        video_root = os.path.realpath(video_root)
        self.videos, self.labels = [], []
        n_missing = 0
        with open(ann_file) as f:
            for line in f:
                name, label = line.split()
                path = os.path.join(video_root, name)
                if not os.path.exists(path):
                    n_missing += 1
                    continue
                self.videos.append(path)
                self.labels.append(int(label))
        if n_missing and _is_main():
            print(f"[{Path(ann_file).name}] {n_missing} videos not found and skipped")

        self._skeleton_file = skeleton_file
        self._mmap = None
        self._setup_skeleton_cache(cache_dir)

    # ------------------------------------------------------------------
    # Skeleton cache
    # ------------------------------------------------------------------
    @staticmethod
    def _match_keys(path):
        stem = Path(path).stem
        return [stem, re.sub(r"_\d{6}_\d{6}$", "", stem), f"{Path(path).parent.name}/{stem}"]

    def _preprocess(self, ann):
        """(P, T, 17, 2) keypoints -> (T, 3, 17, M): normalized (x, y, score) of the top-M persons."""
        kp = np.array(ann["keypoint"], dtype=np.float32)
        score = np.array(ann["keypoint_score"], dtype=np.float32)
        h, w = ann.get("img_shape", (256, 340))
        kp[..., 0] /= w
        kp[..., 1] /= h
        top = np.argsort(-score.mean(axis=(1, 2)))[:self.num_person]
        data = np.concatenate([kp[top], score[top][..., None]], axis=-1)  # (M', T, 17, 3)
        if len(top) < self.num_person:
            pad = np.zeros((self.num_person - len(top),) + data.shape[1:], dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)
        return np.ascontiguousarray(data.transpose(1, 3, 2, 0))

    def _setup_skeleton_cache(self, cache_dir):
        h = hashlib.sha1()
        h.update(str(Path(self._skeleton_file).resolve()).encode())
        h.update(f"|{os.path.getmtime(self._skeleton_file)}|M{self.num_person}|".encode())
        for v in self.videos:
            h.update(v.encode())
        key = h.hexdigest()[:16]
        os.makedirs(cache_dir, exist_ok=True)
        self._cache_bin = os.path.join(cache_dir, f"{key}.bin")
        cache_meta = os.path.join(cache_dir, f"{key}.npz")

        if _is_main() and not os.path.exists(cache_meta):
            self._build_cache(cache_meta)
        if dist.is_initialized():
            dist.barrier()

        meta = np.load(cache_meta)
        self._offsets, self._lengths = meta["offsets"], meta["lengths"]
        self.has_skeleton = meta["has_skeleton"]
        self._shape = tuple(int(x) for x in meta["shape"])

    def _build_cache(self, cache_meta):
        print(f"Packing skeletons from {self._skeleton_file} -> {self._cache_bin}")
        with open(self._skeleton_file, "rb") as f:
            anns = {a["frame_dir"]: a for a in pickle.load(f)["annotations"]}
        offsets, lengths, has_skeleton = [0], [], []
        with open(self._cache_bin + ".tmp", "wb") as f:
            for path in self.videos:
                key = next((k for k in self._match_keys(path) if k in anns), None)
                if key is None:
                    data = np.zeros((1, 3, 17, self.num_person), dtype=np.float32)
                else:
                    data = self._preprocess(anns[key])
                f.write(data.tobytes())
                has_skeleton.append(key is not None)
                lengths.append(len(data))
                offsets.append(offsets[-1] + len(data))
        print(f"Skeletons found for {sum(has_skeleton)}/{len(has_skeleton)} videos")
        os.replace(self._cache_bin + ".tmp", self._cache_bin)
        shape = np.array([offsets[-1], 3, 17, self.num_person])
        np.savez(cache_meta[:-4] + ".tmp.npz", offsets=np.array(offsets[:-1]), lengths=np.array(lengths),
                 has_skeleton=np.array(has_skeleton), shape=shape)
        os.replace(cache_meta[:-4] + ".tmp.npz", cache_meta)

    def _load_skeleton(self, idx, frame_inds, total_frames):
        """Skeleton frames aligned with the sampled RGB frames: (3, len(frame_inds), 17, M)."""
        if self._mmap is None:
            self._mmap = np.memmap(self._cache_bin, dtype=np.float32, mode="r", shape=self._shape)
        start, length = int(self._offsets[idx]), int(self._lengths[idx])
        if length <= 1:
            inds = np.zeros_like(frame_inds)
        elif length == total_frames:
            inds = frame_inds
        else:  # pose and video frame rates may differ
            inds = np.round(frame_inds.astype(np.float64) * length / float(total_frames)).astype(np.int64)
        inds = np.clip(inds, 0, length - 1)
        skel = np.array(self._mmap[start:start + length][inds], dtype=np.float32).transpose(1, 0, 2, 3)
        # Coordinates relative to joint 0, which keeps its absolute trajectory.
        root = skel[:, :, 0].copy()
        skel = skel - skel[:, :, 0:1]
        skel[:, :, 0] = root
        return np.ascontiguousarray(skel)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        reader = decord.VideoReader(self.videos[idx], num_threads=1)
        total_frames = len(reader)
        # Multi-view test clips are drawn from a per-video generator so that results do not
        # depend on the number of GPUs or data-loading workers.
        rng = np.random if self.train else np.random.RandomState(self.seed + idx)
        frame_inds = sample_frames(total_frames, self.num_frames, self.num_clips, self.train, rng)
        decoded = {i: reader[i].asnumpy() for i in np.unique(frame_inds)}
        frames = [decoded[i] for i in frame_inds]
        del reader

        return {
            "video": torch.from_numpy(self.transform(frames)),
            "skeleton": torch.from_numpy(self._load_skeleton(idx, frame_inds, total_frames)),
            "has_skeleton": torch.tensor(bool(self.has_skeleton[idx])),
            "label": torch.tensor(self.labels[idx]),
        }
