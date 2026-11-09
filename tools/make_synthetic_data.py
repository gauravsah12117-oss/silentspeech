"""Generate a fake landmark dataset to smoke-test the pipeline without a webcam.

    set SILENTSPEECH_WORKDIR=C:\\temp\\ss_synthetic          (Windows)
    export SILENTSPEECH_WORKDIR=/tmp/ss_synthetic            (macOS / Linux)
    python tools/make_synthetic_data.py --speakers 4 --phrases 12 --reps 20
    python train.py --epochs 30 && python evaluate.py --split test

Each phrase gets its own made-up pattern of mouth opening, width and rounding,
played back with random speed, head pose, landmark noise and dropped frames.
It is NOT real lip reading: use it only to check that recording files,
training, evaluation and enrolment run end to end. Results on this data
mean nothing for the competition. It refuses to write into a data folder
that already holds real recordings.
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.utils import load_phrases  # noqa: E402

FIELDS = ["file", "speaker", "session", "phrase_id", "phrase_text", "frames", "fps",
          "lighting", "glasses", "timestamp"]
LIGHTING = {1: "daylight", 2: "tubelight", 3: "dim"}


def lip_shape(opening, width, rounding, sp):
    """40 lip points (eye-distance units, y down) in the same ring order as MediaPipe."""
    theta = np.pi - np.arange(20) * np.pi / 10          # 0: left corner, 5: bottom, 10: right, 15: top
    s = np.sin(theta)
    w = sp["w"] * width * (1 - 0.25 * rounding)
    outer = np.stack([np.cos(theta) * w / 2,
                      np.where(s >= 0, s * (sp["h_lo"] + 0.6 * opening), s * (sp["h_up"] + 0.4 * opening))], 1)
    inner = np.stack([np.cos(theta) * w * 0.42,
                      np.where(s >= 0, s * (0.01 + 0.55 * opening), s * (0.01 + 0.35 * opening))], 1)
    return np.concatenate([outer, inner])


def phrase_pattern(k):
    rng = np.random.default_rng(1000 + k)
    n = int(rng.integers(3, 7))
    keys = np.stack([rng.uniform(0.02, 0.35, n), rng.uniform(0.8, 1.15, n), rng.uniform(0, 1, n)], 1)
    neutral = np.array([[0.0, 1.0, 0.0]])
    return np.concatenate([neutral, keys, neutral])


def make_clip(pattern, sp, session, rng):
    fps = rng.uniform(27, 30)
    t = int(round(fps * config.RECORD_SECONDS))
    start = int(t * rng.uniform(0.12, 0.3))
    length = max(10, int(t * rng.uniform(0.45, 0.65) * sp["speed"]))
    end = min(t - 2, start + length)
    key_t = np.linspace(start, end, len(pattern))
    keys = pattern + rng.normal(0, 0.02, pattern.shape) + sp["bias"][: len(pattern)] * 0.5
    keys[[0, -1]] = [0.0, 1.0, 0.0]
    frames = np.arange(t)
    params = np.stack([np.interp(frames, key_t, keys[:, j]) for j in range(3)], 1)
    kernel = np.ones(5) / 5
    params = np.stack([np.convolve(np.pad(params[:, j], 2, mode="edge"), kernel, "valid") for j in range(3)], 1)

    d = rng.uniform(95, 140) * (1.0 if session != 3 else rng.uniform(0.85, 1.0))   # eye distance in px
    centre = np.array([320 + rng.uniform(-40, 40), 300 + rng.uniform(-30, 30)])
    roll0, drift = np.deg2rad(rng.uniform(-6, 6)), np.deg2rad(rng.uniform(-2, 2))
    noise = {1: 0.6, 2: 0.8, 3: 1.2}[session]
    clip = np.zeros((t, 42, 3), np.float32)
    for i in range(t):
        lips = lip_shape(params[i, 0] * sp["gain"], params[i, 1], np.clip(params[i, 2], 0, 1), sp)
        eyes = np.array([[-0.5, -0.9], [0.5, -0.9]])
        pts = np.concatenate([lips, eyes]) * d
        a = roll0 + drift * i / t
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        pts = pts @ rot.T + centre + rng.normal(0, noise, (42, 2))
        clip[i, :, :2] = pts
    if rng.random() < 0.15:                                   # a short tracking dropout
        j = int(rng.integers(1, t - 4))
        clip[j:j + int(rng.integers(1, 4))] = np.nan
    return clip, fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speakers", type=int, default=4)
    ap.add_argument("--phrases", type=int, default=12)
    ap.add_argument("--reps", type=int, default=20, help="per phrase per speaker, split 7/7/6 over 3 sessions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = config.RAW_DIR
    real = [p.name for p in raw.glob("*") if p.is_dir() and not p.name.startswith("syn")] if raw.exists() else []
    if real:
        sys.exit(f"{raw} already has real recordings ({', '.join(real)}). Set SILENTSPEECH_WORKDIR "
                 "to an empty folder for synthetic data.")

    phrases = list(load_phrases(phrase_set="full").values())[: args.phrases]
    rng = np.random.default_rng(args.seed)
    per_session = {1: int(np.ceil(args.reps / 3)), 2: int(np.ceil(args.reps / 3))}
    per_session[3] = args.reps - per_session[1] - per_session[2]
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / "SYNTHETIC.txt").write_text("Synthetic smoke-test data. Not real recordings.\n")

    rows = []
    for s in range(1, args.speakers + 1):
        speaker = f"syn{s:02d}"
        sp = {"w": rng.uniform(0.55, 0.7), "h_up": rng.uniform(0.06, 0.09), "h_lo": rng.uniform(0.08, 0.12),
              "gain": rng.uniform(0.85, 1.15), "speed": rng.uniform(0.85, 1.15),
              "bias": rng.normal(0, 0.03, (8, 3))}
        for k, p in enumerate(phrases):
            pattern = phrase_pattern(k)
            for session, n in per_session.items():
                for rep in range(1, n + 1):
                    clip, fps = make_clip(pattern, sp, session, rng)
                    path = raw / speaker / p["id"] / f"{session}_{rep:02d}.npy"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, clip)
                    rows.append({"file": path.relative_to(config.DATA_DIR).as_posix(), "speaker": speaker,
                                 "session": session, "phrase_id": p["id"], "phrase_text": p["text"],
                                 "frames": len(clip), "fps": f"{fps:.1f}", "lighting": LIGHTING[session],
                                 "glasses": 0, "timestamp": datetime.now().isoformat(timespec="seconds")})
    with open(config.METADATA_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} synthetic clips ({args.speakers} speakers x {len(phrases)} phrases x "
          f"{args.reps} reps) to {raw}")


if __name__ == "__main__":
    main()
