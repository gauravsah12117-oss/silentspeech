"""Drawing toolkit for the live app: real fonts, rounded cards and live widgets.

Text is drawn with Pillow and TrueType fonts (Segoe UI on Windows, Nirmala UI
for Indian scripts), falling back to OpenCV's built-in font if neither is
available. Drawing text with Pillow is slow on a low-power CPU, so panels are
rendered into small images and cached until their content changes (see
PanelCache); only the camera, mouth close-up and muscle bars change every frame.
"""
import os

import numpy as np

# --- Palette (BGR) -------------------------------------------------------------
BG = (24, 20, 18)
CARD = (40, 35, 32)
CARD_EDGE = (70, 62, 56)
TEXT = (240, 240, 240)
MUTED = (175, 170, 165)
DIM = (115, 110, 105)
ACCENT = (190, 180, 40)          # teal, matches the face mesh
GREEN = (110, 215, 115)
AMBER = (45, 180, 250)
RED = (75, 75, 235)

FONT_FILES = {
    "regular": ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf"],
    "semibold": ["C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/segoeuib.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf"],
    "bold": ["C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/System/Library/Fonts/Supplemental/Arial Bold.ttf"],
    "indic": ["C:/Windows/Fonts/Nirmala.ttf", "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
              "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc"],
}
_fonts = {}


def font(style, size):
    """A Pillow TrueType font, or None if Pillow or the font files are missing."""
    key = (style, size)
    if key not in _fonts:
        _fonts[key] = None
        try:
            from PIL import ImageFont
        except ImportError:
            return None
        for path in FONT_FILES.get(style, []) + FONT_FILES["regular"]:
            if os.path.exists(path):
                try:
                    _fonts[key] = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
    return _fonts[key]


def _font_for(text, style, size):
    return font("indic", size) if not text.isascii() else font(style, size)


def text_width(text, size=16, style="regular"):
    f = _font_for(text, style, size)
    if f is not None:
        return int(f.getlength(text))
    import cv2
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size / 34, 1)[0][0]


def wrap(text, max_width, size=16, style="regular", max_lines=None):
    """Split text into lines that fit max_width pixels."""
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = f"{line} {w}".strip()
        if line and text_width(trial, size, style) > max_width:
            lines.append(line)
            line = w
        else:
            line = trial
    if line:
        lines.append(line)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and text_width(lines[-1] + "...", size, style) > max_width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += "..."
    return lines


class Painter:
    """Collects text, then draws it all onto an image in one Pillow pass."""

    def __init__(self):
        self.items = []

    def text(self, text, xy, size=16, color=TEXT, style="regular", anchor="la"):
        """anchor: Pillow anchors, e.g. 'la' left/ascender (top-left), 'mm' centred, 'ra' right."""
        if text:
            self.items.append((str(text), xy, size, color, style, anchor))

    def flush(self, img):
        if not self.items:
            return img
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            Image = None
        if Image is None or font("regular", 16) is None:
            import cv2
            for text, (x, y), size, color, _, _ in self.items:     # plain fallback
                cv2.putText(img, text.encode("ascii", "replace").decode(), (int(x), int(y + size * 0.8)),
                            cv2.FONT_HERSHEY_SIMPLEX, size / 34, color, 1, cv2.LINE_AA)
            self.items = []
            return img
        pil = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))
        draw = ImageDraw.Draw(pil)
        for text, xy, size, color, style, anchor in self.items:
            draw.text(xy, text, font=_font_for(text, style, size), fill=tuple(int(c) for c in color[::-1]),
                      anchor=anchor)
        img[:] = np.asarray(pil)[:, :, ::-1]
        self.items = []
        return img


# --- Shapes --------------------------------------------------------------------

def rounded_rect(img, x0, y0, x1, y1, r, color, thickness=-1):
    import cv2
    x0, y0, x1, y1, r = int(x0), int(y0), int(x1), int(y1), int(max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2)))
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        for a, b in (((x0 + r, y0), (x1 - r, y0)), ((x0 + r, y1), (x1 - r, y1)),
                     ((x0, y0 + r), (x0, y1 - r)), ((x1, y0 + r), (x1, y1 - r))):
            cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
        for (cx, cy), ang in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270), ((x0 + r, y1 - r), 90),
                              ((x1 - r, y1 - r), 0)):
            cv2.ellipse(img, (cx, cy), (r, r), ang, 0, 90, color, thickness, cv2.LINE_AA)
    return img


def card(img, painter, x0, y0, x1, y1, title=None, edge=CARD_EDGE, fill=CARD):
    rounded_rect(img, x0, y0, x1, y1, 10, fill)
    rounded_rect(img, x0, y0, x1, y1, 10, edge, 1)
    if title:
        painter.text(title.upper(), (x0 + 14, y0 + 9), 12, DIM, "semibold")


def bar(img, x0, y0, w, h, value, color, back=(62, 56, 52)):
    import cv2
    cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), back, -1)
    fill = int(w * float(np.clip(value, 0, 1)))
    if fill > 0:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + fill), int(y0 + h)), color, -1)


def level_color(p, accept=0.70, minimum=0.40):
    return GREEN if p >= accept else AMBER if p >= minimum else RED


# --- Live widgets ----------------------------------------------------------------

def mouth_inset(frame, lips, size):
    """Zoomed, mirrored close-up of the mouth with the lip contours drawn on it."""
    import cv2

    w, h = size
    out = np.full((h, w, 3), CARD, np.uint8)
    if lips is None or not np.isfinite(lips).all():
        return out
    ring = lips[:20, :2]
    cx, cy = ring.mean(0)
    span = max(np.ptp(ring[:, 0]), 1.0) * 1.7
    half_w, half_h = span / 2, span / 2 * h / w
    x0, y0 = int(cx - half_w), int(cy - half_h)
    x1, y1 = int(cx + half_w), int(cy + half_h)
    fh, fw = frame.shape[:2]
    if x1 - x0 < 8 or y1 - y0 < 8 or x0 < 0 or y0 < 0 or x1 > fw or y1 > fh:
        pad = max(0, -x0, -y0, x1 - fw, y1 - fh)
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
        lips = lips.copy()
        lips[:, :2] += pad
    crop = cv2.resize(frame[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_LINEAR)
    crop = cv2.convertScaleAbs(crop, alpha=0.75)
    sx, sy = w / (x1 - x0), h / (y1 - y0)
    pts = np.stack([(lips[:40, 0] - x0) * sx, (lips[:40, 1] - y0) * sy], 1).astype(np.int32)
    cv2.polylines(crop, [pts[:20], pts[20:40]], True, (120, 255, 160), 1, cv2.LINE_AA)
    for x, y in pts[:40]:
        cv2.circle(crop, (int(x), int(y)), 2, (90, 255, 150), -1, cv2.LINE_AA)
    return cv2.flip(crop, 1)


# Muscle-activity rows: label -> blendshape names averaged together.
MUSCLE_ROWS = [
    ("Jaw open", ["jawOpen"]),
    ("Lips closed", ["mouthClose"]),
    ("Pucker", ["mouthPucker"]),
    ("Funnel", ["mouthFunnel"]),
    ("Press", ["mouthPressLeft", "mouthPressRight"]),
    ("Smile", ["mouthSmileLeft", "mouthSmileRight"]),
    ("Lower lip down", ["mouthLowerDownLeft", "mouthLowerDownRight"]),
    ("Roll in", ["mouthRollLower", "mouthRollUpper"]),
]


def muscle_values(blend):
    """Blendshape vector (52,) -> one value per MUSCLE_ROWS row (0..1)."""
    from src.landmarks import BLENDSHAPE_NAMES

    if blend is None:
        return None
    idx = {n: i for i, n in enumerate(BLENDSHAPE_NAMES)}
    return np.array([np.mean([blend[idx[n]] for n in names]) for _, names in MUSCLE_ROWS], np.float32)


class PanelCache:
    """Re-render a panel only when its content key changes."""

    def __init__(self):
        self._store = {}

    def get(self, name, key, render):
        hit = self._store.get(name)
        if hit is None or hit[0] != key:
            hit = (key, render())
            self._store[name] = hit
        return hit[1]
