"""Small helpers shared by the scripts."""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

import config


# --- Environment -------------------------------------------------------------

def load_env():
    """Load secrets from .env if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(config.ROOT / ".env")


def seed_everything(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


class _Tee:
    """Writes to the console and to a session log file."""

    def __init__(self, stream, log):
        self._stream, self._log = stream, log

    def write(self, text):
        n = self._stream.write(text)
        try:
            self._log.write(text)
            self._log.flush()
        except (OSError, ValueError):
            pass
        return n

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def start_session_log(prefix):
    """Copy everything printed (and any crash) to logs/<prefix>-<time>.log. Returns (path, file)."""
    from datetime import datetime

    folder = config.WORK_DIR / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}.log"
    log = open(path, "a", encoding="utf-8", errors="replace")
    sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
    return path, log


# --- Phrases -----------------------------------------------------------------

def load_phrases(path=None, phrase_set=None, custom=True):
    """Return {phrase_id: phrase_dict} in file order.

    phrase_set "mvp" keeps entries marked "set": "mvp"; "full" keeps all.
    The user's own phrases (add_phrase.py, ids C01, C02, ...) are always
    included unless custom=False.
    """
    path = Path(path or config.PHRASE_FILE)
    phrase_set = phrase_set or config.PHRASE_SET
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    problems = validate_phrases(items)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    if phrase_set not in ("mvp", "full"):
        raise ValueError(f"Unknown phrase set {phrase_set!r} (use 'mvp' or 'full')")
    phrases = {p["id"]: p for p in items if phrase_set == "full" or p.get("set", "mvp") == "mvp"}
    if custom:
        for p in load_custom_phrases():
            phrases.setdefault(p["id"], p)
    return phrases


def load_custom_phrases():
    """The user's own phrases, from phrases/custom.json (private, not in git)."""
    path = config.CUSTOM_PHRASE_FILE
    if not path.exists():
        return []
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [p for p in items if not validate_phrases([p])]


def save_custom_phrases(items):
    save_json(config.CUSTOM_PHRASE_FILE, items)


def validate_phrases(items):
    problems = []
    seen = set()
    for i, p in enumerate(items):
        for key in ("id", "text", "category", "emergency"):
            if key not in p:
                problems.append(f"entry {i} has no {key!r}")
        pid = p.get("id")
        if pid in seen:
            problems.append(f"duplicate id {pid}")
        seen.add(pid)
        if not isinstance(p.get("emergency", False), bool):
            problems.append(f"{pid}: 'emergency' must be true or false")
    return problems


# --- Models ------------------------------------------------------------------

def personal_model_path(speaker):
    return config.MODEL_DIR / f"lipreader_{speaker}.pt"


def holdout_model_path(speaker):
    return config.MODEL_DIR / f"lipreader_holdout_{speaker}.pt"


# --- Camera ------------------------------------------------------------------

def open_camera(index=None, width=None, height=None):
    """Open a webcam. On Windows DirectShow opens much faster than the default."""
    import cv2

    index = config.CAMERA_INDEX if index is None else index
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width or config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height or config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def close_window(name):
    """Close an OpenCV window, even if the user already closed it with its X button."""
    import cv2

    try:
        cv2.destroyWindow(name)
        cv2.waitKey(1)
    except cv2.error:
        pass


def screen_work_area():
    """(width, height) of the usable desktop (Windows), or None if unknown."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        rect = wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)   # SPI_GETWORKAREA
        return rect.right - rect.left, rect.bottom - rect.top
    except Exception:
        return None


def make_window(name, width, height):
    """A resizable OpenCV window that opens on top of the console and fits the screen."""
    import cv2

    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    area = screen_work_area()
    if area:
        scale = min(1.0, (area[0] - 20) / width, (area[1] - 50) / height)   # leave room for the title bar
        width, height = int(width * scale), int(height * scale)
    cv2.resizeWindow(name, width, height)
    try:
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:  # not supported by every OpenCV build
        pass


class FPSCounter:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.fps = 0.0
        self._last = None

    def tick(self):
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0 else (1 - self.alpha) * self.fps + self.alpha * inst
        self._last = now
        return self.fps


# --- Drawing -----------------------------------------------------------------

_UNICODE_FONTS = [
    "C:/Windows/Fonts/Nirmala.ttf",                                  # Windows: Indian scripts
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",  # Linux
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",           # macOS
]
_font_cache = {}


def _unicode_font(px):
    from PIL import ImageFont

    if px not in _font_cache:
        _font_cache[px] = None
        for path in _UNICODE_FONTS:
            if os.path.exists(path):
                _font_cache[px] = ImageFont.truetype(path, px)
                break
    return _font_cache[px]


def draw_text(img, text, org, scale=0.6, color=(255, 255, 255), thickness=1):
    """cv2.putText, falling back to Pillow for non-English scripts (e.g. Hindi)."""
    import cv2

    if text.isascii():
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return
    try:
        from PIL import Image, ImageDraw
        font = _unicode_font(max(10, int(scale * 34)))
    except ImportError:
        font = None
    if font is None:
        cv2.putText(img, text.encode("ascii", "replace").decode(), org,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return
    pil = Image.fromarray(img[:, :, ::-1])
    x, y = org
    ImageDraw.Draw(pil).text((x, y - int(scale * 30)), text, font=font, fill=color[::-1])
    img[:] = np.asarray(pil)[:, :, ::-1]


def wrap_text(text, max_chars):
    words, lines, line = text.split(), [], ""
    for w in words:
        if line and len(line) + 1 + len(w) > max_chars:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return lines


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
