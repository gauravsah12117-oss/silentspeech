"""Record lip-landmark clips for each phrase.

    python collect_data.py --speaker s01 --session 1 --reps 7

The recorder shows a phrase, counts down 3-2-1, records 2.5 seconds (red dot),
then checks the clip automatically. Each clip is saved as soon as it passes,
so you can quit at any time and resume later with the same command.

Keys: SPACE record   R redo last clip   N next phrase   Q save and quit

Saved: data/raw/<speaker>/<phrase_id>/<session>_<rep>.npy, shape (frames, 42, 3)
(40 lip points + 2 eye corners, x/y/z in pixel units, NaN where the face was
lost), plus one row per clip in data/metadata.csv. Only coordinates are stored;
raw video is saved only if SAVE_VIDEO is turned on in config.py (with consent).
"""
import argparse
import csv
import sys
import time
from datetime import datetime

import numpy as np

import config
from src.features import check_clip
from src.landmarks import N_POINTS, LipTracker, draw_lips
from src.utils import draw_text, load_phrases, open_camera

WINDOW = "SilentSpeech - recorder"
METADATA_FIELDS = ["file", "speaker", "session", "phrase_id", "phrase_text", "frames", "fps",
                   "lighting", "glasses", "timestamp"]
SPEED_HINTS = ["normal speed", "a bit slower", "a bit faster"]


# --- Files -------------------------------------------------------------------

def clip_dir(speaker, phrase_id):
    return config.RAW_DIR / speaker / phrase_id


def existing_reps(speaker, phrase_id, session):
    return sorted(int(p.stem.split("_")[1])
                  for p in clip_dir(speaker, phrase_id).glob(f"{session}_*.npy"))


def append_metadata(row):
    config.METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    new = not config.METADATA_FILE.exists()
    with open(config.METADATA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def remove_metadata(file_key):
    if not config.METADATA_FILE.exists():
        return
    with open(config.METADATA_FILE, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["file"] != file_key]
    with open(config.METADATA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# --- Camera UI ---------------------------------------------------------------

class Recorder:
    """Webcam preview with lip overlay, countdown and timed capture."""

    PANEL_H = 150

    def __init__(self, cap, tracker, window=WINDOW):
        import cv2
        self.cv2 = cv2
        self.cap = cap
        self.tracker = tracker
        self.window = window
        self.flash = None          # (message, colour, until)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, config.FRAME_WIDTH, config.FRAME_HEIGHT + self.PANEL_H)

    def grab(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        if frame.shape[1] != config.FRAME_WIDTH or frame.shape[0] != config.FRAME_HEIGHT:
            frame = self.cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        return frame

    def read(self):
        frame = self.grab()
        return (None, None) if frame is None else (frame, self.tracker.process(frame))

    def show(self, frame, pts, lines, big=None, progress=None):
        cv2 = self.cv2
        view = frame.copy()
        draw_lips(view, pts)
        view = cv2.flip(view, 1)  # mirror view feels natural; landmarks are taken from the unflipped frame
        if pts is None and progress is None:
            draw_text(view, "No face detected", (20, 40), 0.8, (0, 0, 255), 2)
        if big:
            size = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 4, 8)[0]
            draw_text(view, big, ((view.shape[1] - size[0]) // 2, (view.shape[0] + size[1]) // 2),
                      4, (255, 255, 255), 8)
        if progress is not None:
            cv2.circle(view, (30, 30), 12, (0, 0, 255), -1, cv2.LINE_AA)
            draw_text(view, "REC", (50, 38), 0.7, (0, 0, 255), 2)
            cv2.rectangle(view, (0, view.shape[0] - 8),
                          (int(view.shape[1] * min(progress, 1.0)), view.shape[0]), (0, 0, 255), -1)

        panel = np.full((self.PANEL_H, view.shape[1], 3), 32, np.uint8)
        y = 32
        for i, line in enumerate(lines):
            scale, color, thick = (0.8, (255, 255, 255), 2) if i == 0 else (0.5, (200, 200, 200), 1)
            draw_text(panel, line, (14, y), scale, color, thick)
            y += 32 if i == 0 else 24
        if self.flash and time.time() < self.flash[2]:
            draw_text(panel, self.flash[0], (14, self.PANEL_H - 12), 0.55, self.flash[1], 1)
        cv2.imshow(self.window, np.vstack([view, panel]))
        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
            return ord("q")
        return key

    def notify(self, message, ok=True, seconds=2.5):
        self.flash = (message, (120, 220, 120) if ok else (60, 160, 255), time.time() + seconds)

    def wait_key(self, lines, keys):
        """Preview until one of `keys` (lower-case characters) is pressed."""
        while True:
            frame, pts = self.read()
            if frame is None:
                continue
            key = self.show(frame, pts, lines)
            if key == 255:
                continue
            ch = " " if key == 32 else chr(key).lower()
            if key == 27:
                ch = "q"
            if ch in keys:
                return ch

    def countdown(self, lines, seconds=config.COUNTDOWN_SECONDS):
        end = time.perf_counter() + seconds
        while (left := end - time.perf_counter()) > 0:
            frame, pts = self.read()
            if frame is not None and self.show(frame, pts, lines, big=str(int(left) + 1)) in (ord("q"), ord("Q"), 27):
                return False
        return True

    def capture(self, seconds, lines, keep_frames=False):
        """Record for `seconds`. Returns (clip (T, 42, 3), duration, frames).

        Frames are grabbed first and landmarks found afterwards, so a slow
        laptop still records at the camera's full frame rate.
        """
        frames = []
        start = time.perf_counter()
        while (elapsed := time.perf_counter() - start) < seconds:
            frame = self.grab()
            if frame is None:
                continue
            frames.append(frame)
            self.show(frame, None, lines, progress=elapsed / seconds)
        duration = time.perf_counter() - start
        if frames:
            self.show(frames[-1], None, lines[:1] + ["Checking the clip..."], progress=1.0)
        nan_row = np.full((N_POINTS, 3), np.nan, np.float32)
        points = [p if (p := self.tracker.process(f)) is not None else nan_row for f in frames]
        clip = np.stack(points).astype(np.float32) if points else np.zeros((0, N_POINTS, 3), np.float32)
        return clip, duration, frames if keep_frames else []

    def close(self):
        self.cv2.destroyWindow(self.window)


def save_video(frames, fps, path):
    import cv2
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1), (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


# --- Recording session -------------------------------------------------------

def record_phrases(speaker, session, reps, phrases, lighting="unknown", glasses=False,
                   camera_index=None, title="Recording"):
    """Record `reps` clips of every phrase for one speaker and session.

    phrases: {phrase_id: phrase_dict}. Returns {phrase_id: clips now on disk}.
    """
    cap = open_camera(camera_index)
    if cap is None:
        sys.exit("Could not open the webcam. Try --camera 1 and close Zoom/Teams. "
                 "Run python check_setup.py for details.")
    items = list(phrases.values())
    saved_stack = []  # (path, phrase index) so R can step back across phrases
    with LipTracker() as tracker:
        rec = Recorder(cap, tracker)
        pi = 0
        try:
            while pi < len(items):
                p = items[pi]
                have = existing_reps(speaker, p["id"], session)
                if len(have) >= reps:
                    pi += 1
                    continue
                n_done = len(have)
                header = [f"{pi + 1}/{len(items)}  {p['text']}",
                          f"Clip {n_done + 1} of {reps}  ({SPEED_HINTS[n_done % 3]})  |  "
                          f"{title}: speaker {speaker}, session {session}",
                          "Start and end with lips gently closed. Mouth silently, no sound.",
                          "SPACE record   R redo last   N next phrase   Q save and quit"]
                key = rec.wait_key(header, " rnq")
                if key == "q":
                    break
                if key == "n":
                    pi += 1
                    continue
                if key == "r":
                    if saved_stack:
                        path, idx = saved_stack.pop()
                        path.unlink(missing_ok=True)
                        remove_metadata(path.relative_to(config.DATA_DIR).as_posix())
                        pi = idx
                        rec.notify(f"Deleted {path.name} - record it again")
                    else:
                        rec.notify("Nothing to redo yet", ok=False)
                    continue

                # SPACE: countdown, capture, check, save
                if not rec.countdown(header):
                    break
                clip, duration, frames = rec.capture(config.RECORD_SECONDS, header,
                                                     keep_frames=config.SAVE_VIDEO)
                ok, reason, stats = check_clip(clip, duration)
                if not ok:
                    rec.notify(f"Rejected: {reason}. Press SPACE to try again.", ok=False, seconds=4)
                    continue
                rep = (have[-1] + 1) if have else 1
                path = clip_dir(speaker, p["id"]) / f"{session}_{rep:02d}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, clip)
                append_metadata({
                    "file": path.relative_to(config.DATA_DIR).as_posix(), "speaker": speaker,
                    "session": session, "phrase_id": p["id"], "phrase_text": p["text"],
                    "frames": len(clip), "fps": f"{stats['fps']:.1f}", "lighting": lighting,
                    "glasses": int(bool(glasses)), "timestamp": datetime.now().isoformat(timespec="seconds"),
                })
                if config.SAVE_VIDEO:
                    save_video(frames, stats["fps"],
                               config.VIDEO_DIR / speaker / p["id"] / f"{session}_{rep:02d}.mp4")
                saved_stack.append((path, pi))
                rec.notify(f"Saved {p['id']} clip {len(have) + 1}/{reps} "
                           f"({stats['frames']} frames, {stats['fps']:.0f} FPS)")
        finally:
            rec.close()
            cap.release()
    return {pid: len(existing_reps(speaker, pid, session)) for pid in phrases}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", required=True, help="speaker ID, e.g. s01")
    ap.add_argument("--session", type=int, required=True, help="1, 2 or 3 (record on different days)")
    ap.add_argument("--reps", type=int, default=7, help="clips per phrase in this session")
    ap.add_argument("--set", dest="phrase_set", default=config.PHRASE_SET, choices=["mvp", "full"])
    ap.add_argument("--phrases", help="comma-separated subset, e.g. P01,P12")
    ap.add_argument("--lighting", default="unknown", help="e.g. daylight, tubelight, dim")
    ap.add_argument("--glasses", action="store_true", help="speaker is wearing glasses")
    ap.add_argument("--camera", type=int, default=None, help="webcam index (default from config.py)")
    args = ap.parse_args()

    phrases = load_phrases(phrase_set=args.phrase_set)
    if args.phrases:
        wanted = [s.strip().upper() for s in args.phrases.split(",")]
        unknown = [w for w in wanted if w not in phrases]
        if unknown:
            sys.exit(f"Unknown phrase IDs: {', '.join(unknown)}")
        phrases = {pid: phrases[pid] for pid in wanted}

    counts = record_phrases(args.speaker, args.session, args.reps, phrases,
                            args.lighting, args.glasses, args.camera)
    done = sum(min(c, args.reps) for c in counts.values())
    print(f"\nSession {args.session} for {args.speaker}: {done}/{args.reps * len(counts)} clips recorded.")
    todo = [pid for pid, c in counts.items() if c < args.reps]
    if todo:
        print("Still to record: " + ", ".join(todo) + "  (run the same command again to resume)")


if __name__ == "__main__":
    main()
