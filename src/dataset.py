"""Find recorded clips, split them honestly, and serve them to PyTorch.

Clips live at data/raw/<speaker>/<phrase_id>/<session>_<rep>.npy, so the
folder structure alone tells us the speaker, phrase and session.
"""
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from src.augment import augment
from src.features import fill_missing, normalise, to_features

_NAME = re.compile(r"(\d+)_(\d+)")


@dataclass(frozen=True)
class Clip:
    path: Path
    speaker: str
    phrase_id: str
    session: int
    rep: int


def scan_clips(raw_dir=None, phrase_ids=None, speakers=None):
    raw_dir = Path(raw_dir or config.RAW_DIR)
    clips = []
    for npy in sorted(raw_dir.glob("*/*/*.npy")):
        m = _NAME.fullmatch(npy.stem)
        if not m:
            continue
        speaker, pid = npy.parent.parent.name, npy.parent.name
        if phrase_ids is not None and pid not in phrase_ids:
            continue
        if speakers is not None and speaker not in speakers:
            continue
        clips.append(Clip(npy, speaker, pid, int(m[1]), int(m[2])))
    return clips


def make_splits(clips, holdout_speaker=None, test_sessions=config.TEST_SESSIONS,
                val_fraction=config.VAL_FRACTION, seed=config.SEED, max_reps=None):
    """Split by session, never randomly across sessions.

    train / val : sessions 1-2, val = 15% of each speaker's clips of each phrase
    test        : session 3 only (a new day and new lighting)
    unseen      : every clip of `holdout_speaker`, who is left out of training
    max_reps    : keep only this many train+val clips per speaker and phrase
                  (for the learning curve)
    """
    rng = np.random.default_rng(seed)
    unseen = [c for c in clips if c.speaker == holdout_speaker]
    pool = [c for c in clips if c.speaker != holdout_speaker]
    test = [c for c in pool if c.session in test_sessions]

    groups = defaultdict(list)
    for c in pool:
        if c.session not in test_sessions:
            groups[(c.speaker, c.phrase_id)].append(c)

    train, val = [], []
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda c: (c.session, c.rep))
        items = [items[i] for i in rng.permutation(len(items))]
        if max_reps:
            items = items[:max_reps]
        n_val = int(round(len(items) * val_fraction)) if len(items) >= 4 else 0
        val += items[:n_val]
        train += items[n_val:]
    return {"train": train, "val": val, "test": test, "unseen": unseen}


def blendshape_path(clip_path):
    """Sidecar file with the clip's 52 blendshape scores per frame: <session>_<rep>.bs.npy"""
    clip_path = Path(clip_path)
    return clip_path.with_name(clip_path.stem + ".bs.npy")


def load_blendshapes(clip_path):
    """(T, 52) blendshape scores for a clip (NaN where the face was lost), or None."""
    path = blendshape_path(clip_path)
    return np.load(path) if path.exists() else None


def load_positions(clips):
    """Load and normalise clips. Returns (positions list, kept clips)."""
    positions, kept = [], []
    for c in clips:
        filled = fill_missing(np.load(c.path))
        if filled is None or len(filled) < 2:
            continue
        positions.append(normalise(filled))
        kept.append(c)
    return positions, kept


def read_metadata(path=None):
    """{relative file path: row dict} from data/metadata.csv (may be empty)."""
    path = Path(path or config.METADATA_FILE)
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def clip_key(clip):
    """The `file` value used in metadata.csv for a clip."""
    return clip.path.relative_to(config.DATA_DIR).as_posix()


class LipDataset:
    """PyTorch-style dataset that builds (64, 160) features on the fly."""

    def __init__(self, positions, labels, use_velocity=True, augment_cfg=None, seed=config.SEED):
        self.positions = positions
        self.labels = labels
        self.use_velocity = use_velocity
        self.augment_cfg = augment_cfg
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        import torch

        pos = self.positions[i]
        if self.augment_cfg:
            pos = augment(pos, self.rng, self.augment_cfg)
        feats = to_features(pos, self.use_velocity)
        return torch.from_numpy(feats), int(self.labels[i])


def stack_features(positions, use_velocity=True):
    return np.stack([to_features(p, use_velocity) for p in positions]) if positions else \
        np.zeros((0, config.TARGET_FRAMES, 160 if use_velocity else 80), np.float32)
