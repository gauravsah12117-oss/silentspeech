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
# Colour picture symbols (AAC-style phrase icons). Only Segoe UI Emoji scales to any size;
# elsewhere the icons are simply left out and the text stays.
EMOJI_FONT = "C:/Windows/Fonts/seguiemj.ttf"


def emoji_font(size):
    key = ("emoji", size)
    if key not in _fonts:
        _fonts[key] = None
        try:
            from PIL import ImageFont
            if os.path.exists(EMOJI_FONT):
                _fonts[key] = ImageFont.truetype(EMOJI_FONT, size)
        except (ImportError, OSError):
            pass
    return _fonts[key]


# Picture symbol for a phrase without its own "icon" (e.g. your own phrases).
ICON_FALLBACKS = {"emergency": "\U0001F198", "custom": "\u2B50", "needs": "\U0001F64B", "medical": "\U0001F3E5",
                  "responses": "\U0001F4AC", "feelings": "\U0001F4AD", "family": "\U0001F46A",
                  "social": "\U0001F44B", "requests": "\U0001F64B"}


def phrase_icon(phrase):
    """The phrase's picture symbol ("icon" in the phrase file), or one for its category."""
    if not phrase:
        return ""
    key = "emergency" if phrase.get("emergency") else phrase.get("category")
    return phrase.get("icon") or ICON_FALLBACKS.get(key, "\u2B50")


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

    def icon(self, symbol, xy, size=20, anchor="mm"):
        """A colour picture symbol (emoji); skipped where no colour emoji font exists."""
        if symbol and emoji_font(size) is not None:
            self.items.append((symbol, xy, size, TEXT, "emoji", anchor))

    def flush(self, img):
        if not self.items:
            return img
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            Image = None
        if Image is None or font("regular", 16) is None:
            import cv2
            for text, (x, y), size, color, style, _ in self.items:     # plain fallback
                if style == "emoji":
                    continue
                cv2.putText(img, text.encode("ascii", "replace").decode(), (int(x), int(y + size * 0.8)),
                            cv2.FONT_HERSHEY_SIMPLEX, size / 34, color, 1, cv2.LINE_AA)
            self.items = []
            return img
        pil = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))
        draw = ImageDraw.Draw(pil)
        for text, xy, size, color, style, anchor in self.items:
            if style == "emoji":
                draw.text(xy, text, font=emoji_font(size), anchor=anchor, embedded_color=True)
                continue
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
    import cv2
    rounded_rect(img, x0, y0, x1, y1, 10, fill)
    rounded_rect(img, x0, y0, x1, y1, 10, edge, 1)
    cv2.line(img, (int(x0) + 10, int(y0) + 1), (int(x1) - 10, int(y0) + 1), (84, 76, 70), 1)   # soft top light
    if title:
        painter.text(title.upper(), (x0 + 14, y0 + 9), 12, DIM, "semibold")


def _supersampled(img, x0, y0, x1, y1, draw, s=4):
    """Draw on a 4x copy of img[y0:y1, x0:x1] and scale it back down: smooth curves on any CPU."""
    import cv2
    patch = img[y0:y1, x0:x1]
    big = cv2.resize(patch, ((x1 - x0) * s, (y1 - y0) * s), interpolation=cv2.INTER_NEAREST)
    draw(big, s)
    img[y0:y1, x0:x1] = cv2.resize(big, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    return img


def ring_gauge(img, cx, cy, r, value, color, thickness=5, back=(62, 56, 52)):
    """A circular gauge filled clockwise from the top (value 0..1)."""
    import cv2

    def draw(big, s):
        c, R = (int((cx - x0) * s), int((cy - y0) * s)), int(r * s)
        cv2.circle(big, c, R, back, thickness * s, cv2.LINE_AA)
        end = -90 + 360 * float(np.clip(value, 0, 1))
        if end > -90:
            cv2.ellipse(big, c, (R, R), 0, -90, end, color, thickness * s, cv2.LINE_AA)

    x0, y0 = int(cx - r - thickness), int(cy - r - thickness)
    return _supersampled(img, x0, y0, int(cx + r + thickness + 1), int(cy + r + thickness + 1), draw)


def logo(img, x, y, size):
    """App mark: a rounded tile (teal to blue) with a mouth sending out sound waves."""
    import cv2

    def draw(big, s):
        S = size * s
        yy, xx = np.mgrid[0:S, 0:S]
        t = ((xx + yy) / (2.0 * S))[..., None]
        grad = (np.array(ACCENT) * (1 - t) + np.array((225, 120, 85)) * t).astype(np.uint8)
        mask = np.zeros((S, S), np.uint8)
        rounded_rect(mask, 0, 0, S - 1, S - 1, int(0.26 * S), 255)
        region = big[:S, :S]
        region[mask > 0] = grad[mask > 0]
        w = max(2, int(0.065 * S))
        white = (250, 250, 250)
        cv2.ellipse(region, (int(0.36 * S), int(0.5 * S)), (int(0.19 * S), int(0.12 * S)), 0, 0, 360, white, w,
                    cv2.LINE_AA)
        cv2.line(region, (int(0.19 * S), int(0.5 * S)), (int(0.53 * S), int(0.5 * S)), white, w, cv2.LINE_AA)
        for rad in (0.2, 0.32):
            cv2.ellipse(region, (int(0.56 * S), int(0.5 * S)), (int(rad * S), int(rad * S)), 0, -38, 38, white,
                        w, cv2.LINE_AA)

    return _supersampled(img, int(x), int(y), int(x + size), int(y + size), draw)


def badge(img, painter, x_right, cy, text, color):
    """A pill with a dot and a label, right-aligned at x_right. Returns its left edge."""
    import cv2
    tw = text_width(text, 11, "bold")
    x0 = int(x_right - tw - 28)
    rounded_rect(img, x0, cy - 10, x_right, cy + 10, 10, _mix(color, CARD, 0.22))
    cv2.circle(img, (x0 + 11, cy), 4, color, -1, cv2.LINE_AA)
    painter.text(text, (x0 + 19, cy), 11, color, "bold", "lm")
    return x0


def keycap(img, painter, x_right, cy, key, action, size=11):
    """'[key] action' right-aligned at x_right, the key drawn as a keyboard key. Returns the left edge."""
    aw = text_width(action, size)
    kw = text_width(key, size - 1, "semibold") + 12
    painter.text(action, (x_right, cy), size, DIM, "regular", "rm")
    kx1 = int(x_right - aw - 5)
    kx0 = int(kx1 - kw)
    rounded_rect(img, kx0, cy - 8, kx1, cy + 8, 4, (60, 54, 50))
    rounded_rect(img, kx0, cy - 8, kx1, cy + 8, 4, (92, 84, 78), 1)
    painter.text(key, ((kx0 + kx1) / 2, cy), size - 1, TEXT, "semibold", "mm")
    return kx0


def bar(img, x0, y0, w, h, value, color, back=(62, 56, 52)):
    """A rounded progress bar (value 0..1)."""
    r = int(h // 2)
    rounded_rect(img, x0, y0, x0 + w, y0 + h, r, back)
    fill = int(w * float(np.clip(value, 0, 1)))
    if fill >= max(2, h // 2):                # a sliver is not drawn as a dot
        rounded_rect(img, x0, y0, x0 + max(fill, h), y0 + h, r, color)


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
    ("Pucker", ["mouthPucker"]),
    ("Lips press", ["mouthPressLeft", "mouthPressRight"]),
    ("Smile", ["mouthSmileLeft", "mouthSmileRight"]),
    ("Brows up", ["browInnerUp", "browOuterUpLeft", "browOuterUpRight"]),
    ("Brows down", ["browDownLeft", "browDownRight"]),
    ("Eyes narrow", ["eyeSquintLeft", "eyeSquintRight"]),
    ("Nose wrinkle", ["noseSneerLeft", "noseSneerRight"]),
]


def muscle_values(blend):
    """Blendshape vector (52,) -> one value per MUSCLE_ROWS row (0..1)."""
    from src.landmarks import BLENDSHAPE_NAMES

    if blend is None:
        return None
    idx = {n: i for i, n in enumerate(BLENDSHAPE_NAMES)}
    return np.array([np.mean([blend[idx[n]] for n in names]) for _, names in MUSCLE_ROWS], np.float32)


# --- Emotion widgets -------------------------------------------------------------

def _mix(a, b, t):
    return tuple(int(round(x * t + y * (1 - t))) for x, y in zip(a, b))


def emotion_face(img, cx, cy, r, emotion, color):
    """A small drawn face showing the emotion. Drawn 4x larger and scaled down,
    so the strokes come out smooth."""
    import cv2

    s = 4
    x0, y0, x1, y1 = int(cx - r - 3), int(cy - r - 3), int(cx + r + 4), int(cy + r + 4)
    patch = img[y0:y1, x0:x1]
    big = cv2.resize(patch, ((x1 - x0) * s, (y1 - y0) * s), interpolation=cv2.INTER_NEAREST)
    c, R = ((cx - x0) * s, (cy - y0) * s), r * s
    t = max(2, int(0.1 * R))

    def P(dx, dy):
        return int(c[0] + dx * R), int(c[1] + dy * R)

    def line(a, b):
        cv2.line(big, P(*a), P(*b), color, t, cv2.LINE_AA)

    def arc(center, ax, ay, a0, a1):
        cv2.ellipse(big, P(*center), (int(ax * R), int(ay * R)), 0, a0, a1, color, t, cv2.LINE_AA)

    def dot(center, rad, filled=True):
        cv2.circle(big, P(*center), max(2, int(rad * R)), color, -1 if filled else t, cv2.LINE_AA)

    cv2.circle(big, c, R, _mix(color, CARD, 0.22), -1, cv2.LINE_AA)
    cv2.circle(big, c, R, color, t, cv2.LINE_AA)
    for side in (-1, 1):
        ex = 0.36 * side
        if emotion == "happy":
            arc((ex, -0.1), 0.15, 0.12, 180, 360)
        elif emotion in ("surprised", "afraid"):
            dot((ex, -0.14), 0.13, filled=False)
            dot((ex, -0.14), 0.04)
            if emotion == "surprised":
                arc((ex, -0.44), 0.17, 0.08, 200, 340)
            else:
                line((0.54 * side, -0.4), (0.18 * side, -0.56))
        elif emotion == "disgusted":
            line((0.5 * side, -0.16), (0.22 * side, -0.12))
            if side < 0:
                line((-0.56, -0.42), (-0.18, -0.34))
        elif emotion == "pain":                  # eyes squeezed shut, brows pulled down
            line((0.5 * side, -0.24), (0.24 * side, -0.14))
            line((0.24 * side, -0.14), (0.5 * side, -0.04))
            line((0.56 * side, -0.46), (0.16 * side, -0.32))
        else:
            dot((ex, -0.12), 0.085)
            if emotion == "sad":
                line((0.54 * side, -0.36), (0.2 * side, -0.48))
            elif emotion == "angry":
                line((0.56 * side, -0.46), (0.16 * side, -0.3))
    if emotion == "happy":
        arc((0, 0.14), 0.44, 0.36, 20, 160)
    elif emotion == "sad":
        arc((0, 0.64), 0.32, 0.22, 200, 340)
    elif emotion == "surprised":
        cv2.ellipse(big, P(0, 0.42), (int(0.14 * R), int(0.19 * R)), 0, 0, 360, color, t, cv2.LINE_AA)
    elif emotion == "afraid":
        cv2.ellipse(big, P(0, 0.45), (int(0.27 * R), int(0.12 * R)), 0, 0, 360, color, t, cv2.LINE_AA)
    elif emotion == "angry":
        arc((0, 0.62), 0.26, 0.14, 200, 340)
    elif emotion == "disgusted":
        pts = np.array([P(-0.32, 0.46), P(-0.12, 0.36), P(0.06, 0.48), P(0.3, 0.38)], np.int32)
        cv2.polylines(big, [pts], False, color, t, cv2.LINE_AA)
    elif emotion == "pain":                      # grimace with clenched teeth
        cv2.rectangle(big, P(-0.34, 0.30), P(0.34, 0.54), color, t, cv2.LINE_AA)
        line((-0.34, 0.42), (0.34, 0.42))
    else:
        line((-0.28, 0.4), (0.28, 0.4))
    img[y0:y1, x0:x1] = cv2.resize(big, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    return img


STRIP_BACK = (52, 47, 44)


def timeline_strip(img, x, y, w, h, timeline, now, colors, emotions, seconds=30.0):
    """Coloured strip: the strongest emotion over the last `seconds` (oldest on the left)."""
    import cv2

    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), STRIP_BACK, -1)
    if not timeline:
        return img
    times = np.array([t for t, _, _ in timeline])
    kinds = np.array([k for _, k, _ in timeline])
    confs = np.clip(np.array([c for _, _, c in timeline]), 0.0, 1.0)
    cols = now - seconds + (np.arange(w) + 0.5) * seconds / w
    pos = np.searchsorted(times, cols, side="right") - 1
    ok = (pos >= 0) & (cols - times[np.maximum(pos, 0)] <= 1.0)
    if ok.any():
        palette = np.array([colors[e] for e in emotions], np.float32)
        j = pos[ok]
        mix = (0.35 + 0.65 * confs[j])[:, None]
        rgb = palette[kinds[j]] * mix + np.array(STRIP_BACK, np.float32) * (1 - mix)
        img[y:y + h, x:x + w][:, ok] = np.round(rgb).astype(np.uint8)[None]
    return img


PAIN_STEPS = (0.12, 0.2, 0.35, 0.5, 0.65)
PAIN_COLORS = (GREEN, (70, 210, 190), AMBER, (40, 130, 245), RED)


def pain_meter(img, x, y, score, seg_w=16, seg_h=8, gap=4):
    """Five segments, green to red; how many light up follows the pain score."""
    for i, (step, color) in enumerate(zip(PAIN_STEPS, PAIN_COLORS)):
        x0 = x + i * (seg_w + gap)
        rounded_rect(img, x0, y, x0 + seg_w, y + seg_h, 3, color if score >= step else (62, 56, 52))
    return img


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
