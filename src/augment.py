"""Training-time data augmentation on normalised lip positions (T, 40, 2).

Augmentation runs *after* normalisation on purpose: scaling or rotating the
raw landmarks would be undone by normalisation (which divides by eye distance
and levels the eyes), so the model would never see the variation.
"""
import numpy as np

import config
from src.features import resample
from src.landmarks import MIRROR_PERM, N_LIP_POINTS

_LIP_MIRROR = MIRROR_PERM[:N_LIP_POINTS]


def mirror(positions):
    """Flip left-right and swap each point with its twin.

    Without the swap, point 61 (left corner) would appear on the right, a
    shape the model never sees in real data.
    """
    out = positions[:, _LIP_MIRROR].copy()
    out[..., 0] *= -1
    return out


def time_stretch(positions, factor, rng):
    """Speak faster (<1) or slower (>1) inside the same recording window.

    Clips are resampled to a fixed length later, so plainly stretching the
    whole clip would be undone. Instead the motion is stretched and then
    padded with still frames or trimmed at the ends to keep the clip length,
    which is what a faster or slower speaker looks like in a 2.5 s window.
    """
    t = len(positions)
    new_t = max(2, int(round(t * factor)))
    stretched = resample(positions, new_t)
    if new_t >= t:
        start = rng.integers(0, new_t - t + 1)
        return stretched[start:start + t]
    pad = t - new_t
    before = int(rng.integers(0, pad + 1))
    return np.concatenate([np.repeat(stretched[:1], before, 0), stretched,
                           np.repeat(stretched[-1:], pad - before, 0)])


def frame_dropout(positions, rate, rng):
    """Freeze a few random frames (a dropped or blurry camera frame)."""
    t = len(positions)
    k = int(round(rate * t))
    if k == 0 or t < 3:
        return positions
    out = positions.copy()
    for i in np.sort(rng.choice(np.arange(1, t), size=min(k, t - 1), replace=False)):
        out[i] = out[i - 1]
    return out


def augment(positions, rng, cfg=None):
    """Return a randomly varied copy of one clip of normalised positions."""
    cfg = config.AUGMENT if cfg is None else cfg
    pos = np.asarray(positions, dtype=np.float32)

    if cfg.get("mirror_prob", 0) and rng.random() < cfg["mirror_prob"]:
        pos = mirror(pos)

    scale = 1.0 + rng.uniform(-1, 1) * cfg.get("scale", 0)
    theta = np.deg2rad(rng.uniform(-1, 1) * cfg.get("rotate_deg", 0))
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    shift = rng.uniform(-1, 1, size=2).astype(np.float32) * cfg.get("shift", 0)
    pos = (pos @ rot.T) * scale + shift

    if cfg.get("stretch", 0):
        pos = time_stretch(pos, 1.0 + rng.uniform(-1, 1) * cfg["stretch"], rng)
    if cfg.get("frame_dropout", 0):
        pos = frame_dropout(pos, rng.uniform(0, cfg["frame_dropout"]), rng)
    if cfg.get("noise_std", 0):
        pos = pos + rng.normal(0, cfg["noise_std"], pos.shape)
    return pos.astype(np.float32)
