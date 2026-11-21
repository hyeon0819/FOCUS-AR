"""Frame sampling and video augmentation (MMAction2 / ViFi-CLIP conventions)."""

import random

import cv2
import numpy as np
import torchvision
from PIL import Image

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------
def sample_segments(total_frames: int, num_segments: int, rng=np.random) -> np.ndarray:
    """One random frame from each of ``num_segments`` equal segments (TSN-style)."""
    avg_interval = total_frames // num_segments
    if avg_interval > 0:
        inds = np.arange(num_segments) * avg_interval + rng.randint(avg_interval, size=num_segments)
    elif total_frames > num_segments:
        inds = np.sort(rng.randint(total_frames, size=num_segments))
    elif avg_interval == 0:
        inds = np.around(np.arange(num_segments) * (total_frames / num_segments))
    else:
        inds = np.zeros(num_segments)
    return np.mod(inds.astype(int), total_frames)


def sample_segment_centers(total_frames: int, num_segments: int) -> np.ndarray:
    """The center frame of each of ``num_segments`` equal segments."""
    avg_interval = total_frames / float(num_segments)
    return (np.arange(num_segments) * avg_interval + avg_interval / 2.0).astype(int)


def sample_frames(total_frames: int, num_frames: int, num_clips: int, train: bool, rng=np.random) -> np.ndarray:
    """Frame indices of shape ``(num_clips * num_frames,)``.

    Training samples one random clip. Testing uses the segment centers for a single view and
    ``num_clips`` independently sampled clips for multi-view testing.
    """
    if train or num_clips > 1:
        return np.concatenate([sample_segments(total_frames, num_frames, rng) for _ in range(num_clips)])
    return sample_segment_centers(total_frames, num_frames)


# ---------------------------------------------------------------------------
# Spatial transforms (operate on a list of HxWx3 uint8 RGB frames)
# ---------------------------------------------------------------------------
def resize_short(frames, short_side: int):
    h, w = frames[0].shape[:2]
    scale = short_side / min(h, w)
    size = (int(w * scale + 0.5), int(h * scale + 0.5))
    return [cv2.resize(f, size, interpolation=cv2.INTER_LINEAR) for f in frames]


def center_crop(frames, size: int):
    h, w = frames[0].shape[:2]
    left, top = (w - size) // 2, (h - size) // 2
    return [f[top:top + size, left:left + size] for f in frames]


def multi_scale_crop(frames, input_size: int, scales=(1, 0.875, 0.75, 0.66), max_wh_scale_gap=1):
    """TSN multi-scale crop from one of five fixed positions, then resize to ``input_size``."""
    h, w = frames[0].shape[:2]
    crop_sizes = [int(min(h, w) * s) for s in scales]
    candidates = [[cw, ch] for i, ch in enumerate(crop_sizes) for j, cw in enumerate(crop_sizes)
                  if abs(i - j) <= max_wh_scale_gap]
    crop_w, crop_h = random.choice(candidates)
    crop_w = input_size if abs(crop_w - input_size) < 3 else crop_w
    crop_h = input_size if abs(crop_h - input_size) < 3 else crop_h
    w_step, h_step = (w - crop_w) // 4, (h - crop_h) // 4
    x, y = random.choice([(0, 0), (4 * w_step, 0), (0, 4 * h_step), (4 * w_step, 4 * h_step),
                          (2 * w_step, 2 * h_step)])
    frames = [f[y:y + crop_h, x:x + crop_w] for f in frames]
    return [cv2.resize(f, (input_size, input_size), interpolation=cv2.INTER_LINEAR) for f in frames]


def normalize(frames) -> np.ndarray:
    """uint8 frames -> float32 array (N, 3, H, W) normalized with ImageNet statistics."""
    imgs = np.stack(frames).astype(np.float32)
    for img in imgs:
        cv2.subtract(img, MEAN.astype(np.float64).reshape(1, -1), img)
        cv2.multiply(img, 1 / STD.astype(np.float64).reshape(1, -1), img)
    return np.ascontiguousarray(imgs.transpose(0, 3, 1, 2))


class TrainTransform:
    def __init__(self, input_size=224, color_jitter=0.8, gray_scale=0.2):
        self.input_size = input_size
        self.short_side = int(256 / 224 * input_size)
        self.p_jitter = color_jitter
        self.p_gray = gray_scale
        self.jitter = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        self.gray = torchvision.transforms.Grayscale(num_output_channels=3)

    def __call__(self, frames):
        frames = resize_short(frames, self.short_side)
        frames = multi_scale_crop(frames, self.input_size)
        if np.random.rand() < 0.5:
            frames = [cv2.flip(f, 1) for f in frames]
        if random.random() < self.p_jitter:
            frames = [np.asarray(self.jitter(Image.fromarray(f))) for f in frames]
        if random.random() < self.p_gray:
            frames = [np.asarray(self.gray(Image.fromarray(f))) for f in frames]
        return normalize(frames)


class TestTransform:
    def __init__(self, input_size=224):
        self.input_size = input_size
        self.short_side = int(256 / 224 * input_size)

    def __call__(self, frames):
        return normalize(center_crop(resize_short(frames, self.short_side), self.input_size))
