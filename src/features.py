"""Turn raw landmark clips into model input.

raw clip (T, 42, 3) pixel units, NaN rows where the face was lost
  -> fill_missing   interpolate over lost frames
  -> normalise      (T, 40, 2): translate, scale and rotate away head pose
  -> [augment]      training only, see src/augment.py
  -> to_features    resample to 64 frames, add velocities -> (64, 160)
"""
import numpy as np

import config
from src.landmarks import (INNER_BOTTOM, INNER_TOP, N_LIP_POINTS, OUTER_LEFT,
                           OUTER_RIGHT)


def fill_missing(clip):
    """Linearly interpolate frames where the face was lost (NaN rows).

    Returns None when no frame has a face at all.
    """
    clip = np.asarray(clip, dtype=np.float32)
    valid = ~np.isnan(clip).any(axis=(1, 2))
    if not valid.any():
        return None
    if valid.all():
        return clip
    t = np.arange(len(clip))
    flat = clip.reshape(len(clip), -1).copy()
    for j in range(flat.shape[1]):
        flat[~valid, j] = np.interp(t[~valid], t[valid], flat[valid, j])
    return flat.reshape(clip.shape)


def _smooth(x, k=5):
    """Centred moving average with edge padding (reduces landmark jitter)."""
    if len(x) < 3:
        return x
    k = min(k, len(x) if len(x) % 2 else len(x) - 1)
    pad = k // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def normalise(clip):
    """Remove position, distance and head tilt so only mouth shape remains.

    1. Translate: subtract the centre of the lips.
    2. Scale: divide by the distance between the eye corners.
    3. Rotate: make the line between the eyes horizontal.

    Input (T, 42, 2 or 3) without NaNs; output (T, 40, 2) float32.
    """
    xy = np.asarray(clip, dtype=np.float32)[..., :2]
    lips = xy[:, :N_LIP_POINTS]
    eyes = xy[:, N_LIP_POINTS:N_LIP_POINTS + 2]

    centred = lips - lips.mean(axis=1, keepdims=True)
    eye_vec = eyes[:, 1] - eyes[:, 0]
    dist = _smooth(np.maximum(np.linalg.norm(eye_vec, axis=1), 1e-6))
    angle = _smooth(np.unwrap(np.arctan2(eye_vec[:, 1], eye_vec[:, 0])))

    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # (T, 2, 2)
    out = np.einsum("tij,tpj->tpi", rot, centred) / dist[:, None, None]
    return out.astype(np.float32)


def resample(seq, n):
    """Linearly resample a (T, ...) sequence to exactly n frames."""
    seq = np.asarray(seq, dtype=np.float32)
    t = len(seq)
    if t == n:
        return seq.copy()
    if t == 1:
        return np.repeat(seq, n, axis=0)
    src = np.linspace(0, t - 1, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, t - 1)
    w = (src - i0).reshape(-1, *([1] * (seq.ndim - 1))).astype(np.float32)
    return seq[i0] * (1 - w) + seq[i1] * w


def to_features(positions, use_velocity=True, n_frames=config.TARGET_FRAMES):
    """(T, 40, 2) normalised positions -> (n_frames, 160) features.

    80 position values per frame, plus 80 velocities (change since the
    previous frame, computed after resampling so the speed of mouthing is
    measured on the same time scale for every clip).
    """
    pos = resample(positions, n_frames).reshape(n_frames, -1)
    if not use_velocity:
        return pos.astype(np.float32)
    vel = np.diff(pos, axis=0, prepend=pos[:1])
    return np.concatenate([pos, vel], axis=1).astype(np.float32)


def build_features(clip, use_velocity=True, n_frames=config.TARGET_FRAMES):
    """Raw clip (T, 42, 3) -> model features, or None if the face never appeared."""
    filled = fill_missing(clip)
    if filled is None:
        return None
    return to_features(normalise(filled), use_velocity, n_frames)


def pad_to_length(clip, n):
    """Centre a short clip in an n-frame window by holding its first/last frame.

    Auto mode stops soon after the lips go still, so its clips are shorter
    than the fixed push-to-talk window the model was trained on. Padding with
    still frames makes them look like training clips again.
    """
    clip = np.asarray(clip, dtype=np.float32)
    if len(clip) >= n:
        return clip
    before = (n - len(clip)) // 2
    after = n - len(clip) - before
    return np.concatenate([np.repeat(clip[:1], before, 0), clip,
                           np.repeat(clip[-1:], after, 0)])


def lip_motion(positions):
    """How much the mouth moved in a clip of normalised positions.

    The larger of the ranges of lip opening and lip width, in eye-distance
    units. A clip where the speaker forgot to mouth anything scores ~0.
    """
    opening = np.linalg.norm(positions[:, INNER_TOP] - positions[:, INNER_BOTTOM], axis=1)
    width = np.linalg.norm(positions[:, OUTER_LEFT] - positions[:, OUTER_RIGHT], axis=1)
    return float(max(np.ptp(opening), np.ptp(width)))


def frame_motion(prev_positions, positions):
    """Mean lip-point movement between two normalised frames (40, 2)."""
    return float(np.abs(positions - prev_positions).mean())


def check_clip(clip, duration_s, max_lost=config.MAX_FACE_LOST_FRAC,
               min_fps=config.MIN_FPS, min_motion=config.MIN_LIP_MOTION):
    """Automatic quality check for a recorded clip.

    Returns (ok, reason, stats). `reason` is a short message for the screen.
    """
    clip = np.asarray(clip, dtype=np.float32)
    n = len(clip)
    stats = {"frames": n, "fps": n / duration_s if duration_s > 0 else 0.0}
    if n == 0:
        return False, "No frames captured", stats
    lost = float(np.isnan(clip).any(axis=(1, 2)).mean())
    stats["face_lost"] = lost
    if lost > max_lost:
        return False, f"Face lost in {lost:.0%} of frames - check light and position", stats
    if stats["fps"] < min_fps:
        return False, f"Camera too slow ({stats['fps']:.0f} FPS) - add light, close other apps", stats
    motion = lip_motion(normalise(fill_missing(clip)))
    stats["motion"] = motion
    if motion < min_motion:
        return False, "Lips barely moved - mouth the phrase clearly", stats
    return True, "OK", stats
