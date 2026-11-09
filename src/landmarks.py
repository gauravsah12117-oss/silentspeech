"""Face and lip tracking with MediaPipe, plus the face-mesh "HUD" drawing.

Two trackers, same interface (make_tracker() picks one):

* FaceTracker - MediaPipe Face Landmarker (default). A 478-point face mesh
  plus 52 "blendshapes": scores from a second neural network for facial
  muscle movements such as jawOpen, mouthPucker, mouthFunnel, mouthPress.
  ~40 ms per frame on a low-power laptop CPU.
* LipTracker - the older MediaPipe Face Mesh (468 points, no blendshapes).
  ~13 ms per frame. Used if the Face Landmarker model can't be downloaded.

process(frame) returns the 42 points we read lips from: 20 outer-lip points,
20 inner-lip points and the two outer eye corners (for normalisation), in
*pixel units* (x * width, y * height, z * width). MediaPipe's own x and y are
fractions of the image width and height, which have different scales on a
4:3 frame; pixels keep angles and distances correct. After each call,
.last_mesh holds the full mesh and .last_blendshapes the 52 scores (or None).
"""
import os
import sys
import time
from contextlib import contextmanager

import numpy as np

# Rings run: left mouth corner -> along the lower lip -> right corner -> along
# the upper lip back to the start. Order matters for MIRROR_PERM below.
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
              291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
              308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
EYE_CORNERS = [33, 263]

LANDMARK_IDS = LIPS_OUTER + LIPS_INNER + EYE_CORNERS   # 42 points
N_LIP_POINTS = 40
N_POINTS = len(LANDMARK_IDS)

# Positions of useful points inside the 42-point array.
OUTER_LEFT, OUTER_RIGHT = 0, 10        # mouth corners (61, 291)
INNER_BOTTOM, INNER_TOP = 25, 35       # centre of the inner lips (14, 13)
EYE_LEFT, EYE_RIGHT = 40, 41


def _mirror_permutation():
    """Index map that swaps every point with its left-right twin.

    In both rings, position i and position (10 - i) mod 20 are mirror images
    (61<->291, 146<->375, ... 17 and 0 map to themselves). The eyes swap too.
    """
    ring = [(10 - i) % 20 for i in range(20)]
    return np.array(ring + [20 + j for j in ring] + [EYE_RIGHT, EYE_LEFT])


MIRROR_PERM = _mirror_permutation()

# The Face Landmarker's 52 blendshapes, in the model's output order.
BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]
# The ones that describe the mouth and jaw (what lip reading needs).
MOUTH_BLENDSHAPES = [i for i, n in enumerate(BLENDSHAPE_NAMES) if n.startswith(("mouth", "jaw")) or n == "cheekPuff"]
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")


@contextmanager
def _quiet_native_stderr():
    """Silence C/C++ log lines (TensorFlow Lite, absl) that look like errors but aren't."""
    try:
        fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    sys.stderr.flush()
    saved, devnull = os.dup(fd), os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd)
    try:
        yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)
        os.close(devnull)


class LipTracker:
    """MediaPipe Face Mesh (468 points, no blendshapes). Fast fallback tracker."""

    name = "Face Mesh"

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        import mediapipe as mp  # imported here so the rest of the code runs without it

        with _quiet_native_stderr():
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        self._warmed_up = False
        self.last_mesh = None
        self.last_blendshapes = None

    def process(self, frame_bgr):
        """Return a (42, 3) float32 array in pixel units, or None if no face."""
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        if not self._warmed_up:          # the first run prints the start-up chatter
            with _quiet_native_stderr():
                result = self._mesh.process(rgb)
            self._warmed_up = True
        else:
            result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            self.last_mesh = None
            return None
        lm = result.multi_face_landmarks[0].landmark
        self.last_mesh = np.array([(p.x * w, p.y * h, p.z * w) for p in lm], dtype=np.float32)
        return self.last_mesh[LANDMARK_IDS]

    def close(self):
        self._mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FaceTracker:
    """MediaPipe Face Landmarker: 478-point mesh + 52 blendshapes per frame."""

    name = "Face Landmarker"

    def __init__(self, model_path):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        self._mp = mp
        with _quiet_native_stderr():
            self._landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO, num_faces=1,
                output_face_blendshapes=True, output_facial_transformation_matrixes=False))
        self._ts = 0
        self._warmed_up = False
        self.last_mesh = None
        self.last_blendshapes = None

    def process(self, frame_bgr):
        """Return a (42, 3) float32 array in pixel units, or None if no face."""
        import cv2

        h, w = frame_bgr.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self._ts = max(self._ts + 1, int(time.perf_counter() * 1000))   # must keep increasing
        if not self._warmed_up:
            with _quiet_native_stderr():
                result = self._landmarker.detect_for_video(image, self._ts)
            self._warmed_up = True
        else:
            result = self._landmarker.detect_for_video(image, self._ts)
        if not result.face_landmarks:
            self.last_mesh = self.last_blendshapes = None
            return None
        self.last_mesh = np.array([(p.x * w, p.y * h, p.z * w) for p in result.face_landmarks[0]],
                                  dtype=np.float32)
        self.last_blendshapes = (np.array([c.score for c in result.face_blendshapes[0]], dtype=np.float32)
                                 if result.face_blendshapes else None)
        return self.last_mesh[LANDMARK_IDS]

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def ensure_landmarker_model(model_dir=None, quiet=False):
    """Path to face_landmarker.task, downloading it (3.7 MB) the first time. None if impossible."""
    import config

    path = (model_dir or config.MODEL_DIR) / "face_landmarker.task"
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    try:
        import urllib.request
        if not quiet:
            print("Downloading Google's Face Landmarker model (3.7 MB, one time)...")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        req = urllib.request.Request(LANDMARKER_URL, headers={"User-Agent": "SilentSpeech/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
            f.write(resp.read())
        tmp.replace(path)
        return path
    except Exception as e:
        if not quiet:
            print(f"Could not download the Face Landmarker model ({e}); using Face Mesh instead.")
        return None


def make_tracker(kind=None):
    """The configured tracker: Face Landmarker if possible, else Face Mesh."""
    import config

    kind = kind or config.TRACKER
    if kind == "landmarker":
        path = ensure_landmarker_model()
        if path is not None:
            try:
                return FaceTracker(path)
            except Exception as e:
                print(f"Face Landmarker failed to start ({e}); using Face Mesh instead.")
    return LipTracker()


def mouth_aspect_ratio(points):
    """Lip opening divided by lip width for one frame of 42 points."""
    opening = np.linalg.norm(points[INNER_TOP, :2] - points[INNER_BOTTOM, :2])
    width = np.linalg.norm(points[OUTER_LEFT, :2] - points[OUTER_RIGHT, :2])
    return float(opening / width) if width > 1e-6 else 0.0


def draw_lips(frame, points, color=(0, 255, 120)):
    """Draw both lip rings and the eye corners onto a BGR frame in place."""
    import cv2

    if points is None:
        return frame
    pts = np.round(points[:, :2]).astype(np.int32)
    cv2.polylines(frame, [pts[:20]], True, color, 1, cv2.LINE_AA)
    cv2.polylines(frame, [pts[20:40]], True, color, 1, cv2.LINE_AA)
    for x, y in pts[:40]:
        cv2.circle(frame, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
    for x, y in pts[40:]:
        cv2.circle(frame, (int(x), int(y)), 3, (255, 200, 0), -1, cv2.LINE_AA)
    return frame


# --- Face-mesh "HUD" view --------------------------------------------------------

HUD_MESH = (185, 175, 35)       # teal wireframe (BGR)
HUD_LINK = (150, 230, 90)       # lines between the glowing nodes
HUD_NODE = (90, 255, 150)       # node glow (bright green)
HUD_GRID = (38, 46, 26)
HUD_LIPS = (120, 255, 160)
# Glowing nodes: forehead, temples, eyes, nose, cheekbones, cheeks, mouth, chin, jaw.
HUD_NODES = [10, 54, 284, 127, 356, 33, 263, 133, 362, 168, 1, 234, 454, 205, 425,
             61, 291, 0, 17, 152, 172, 397, 136, 365]
HUD_LINKS = [(10, 54), (10, 284), (54, 127), (284, 356), (127, 234), (356, 454), (234, 172), (454, 397),
             (172, 136), (397, 365), (136, 152), (365, 152), (54, 33), (284, 263), (33, 133), (263, 362),
             (133, 168), (362, 168), (10, 168), (168, 1), (33, 205), (263, 425), (234, 205), (454, 425),
             (205, 61), (425, 291), (1, 61), (1, 291), (61, 0), (291, 0), (61, 17), (291, 17),
             (17, 152), (205, 1), (425, 1), (61, 136), (291, 365)]
_hud_cache = {}


def _hud_static(shape):
    """Grid lines (cached per frame size)."""
    import cv2

    if shape not in _hud_cache:
        grid = np.zeros(shape, np.uint8)
        h, w = shape[:2]
        for i, x in enumerate(range(0, w, 32)):
            cv2.line(grid, (x, 0), (x, h), HUD_GRID if i % 4 else tuple(int(c * 1.6) for c in HUD_GRID), 1)
        for i, y in enumerate(range(0, h, 32)):
            cv2.line(grid, (0, y), (w, y), HUD_GRID if i % 4 else tuple(int(c * 1.6) for c in HUD_GRID), 1)
        from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
        _hud_cache[shape] = grid
        _hud_cache["tess"] = np.array(sorted(FACEMESH_TESSELATION), dtype=np.int32)
    return _hud_cache[shape], _hud_cache["tess"]


def draw_face_hud(frame, mesh, lips=None, dim=0.35):
    """Holographic face-mesh view: dimmed camera, grid, wireframe, glowing nodes, lips.

    Returns a new image (the input frame is not changed). ~10 ms on a slow CPU.
    """
    import cv2

    out = cv2.add(cv2.convertScaleAbs(frame, alpha=dim), _hud_static(frame.shape)[0])
    if mesh is None:
        return out
    tess = _hud_static(frame.shape)[1]
    pts = np.round(mesh[:, :2]).astype(np.int32)
    layer = np.zeros_like(out)
    cv2.polylines(layer, list(pts[tess].reshape(-1, 2, 1, 2)), False, HUD_MESH, 1, cv2.LINE_AA)
    links = np.array(HUD_LINKS, dtype=np.int32)
    cv2.polylines(layer, list(pts[links].reshape(-1, 2, 1, 2)), False, HUD_LINK, 1, cv2.LINE_AA)
    out = cv2.addWeighted(out, 1.0, layer, 0.8, 0)

    # soft glow: draw the nodes small, blur at quarter size, add back
    h, w = out.shape[:2]
    glow = np.zeros((h // 4, w // 4, 3), np.uint8)
    for i in HUD_NODES:
        cv2.circle(glow, (int(pts[i, 0]) // 4, int(pts[i, 1]) // 4), 2, HUD_NODE, -1)
    glow = cv2.resize(cv2.GaussianBlur(glow, (0, 0), 1.5), (w, h), interpolation=cv2.INTER_LINEAR)
    out = cv2.add(out, glow)
    for i in HUD_NODES:
        cv2.circle(out, (int(pts[i, 0]), int(pts[i, 1])), 2, (225, 255, 230), -1, cv2.LINE_AA)

    if lips is not None:          # the part we actually read: bright lip contours
        lp = np.round(lips[:, :2]).astype(np.int32)
        cv2.polylines(out, [lp[:20], lp[20:40]], True, HUD_LIPS, 1, cv2.LINE_AA)

    # corner brackets around the face
    x0, y0 = pts[:, 0].min() - 12, pts[:, 1].min() - 12
    x1, y1 = pts[:, 0].max() + 12, pts[:, 1].max() + 12
    s = max(12, (x1 - x0) // 8)
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(out, (int(cx), int(cy)), (int(cx + dx * s), int(cy)), HUD_LINK, 2, cv2.LINE_AA)
        cv2.line(out, (int(cx), int(cy)), (int(cx), int(cy + dy * s)), HUD_LINK, 2, cv2.LINE_AA)
    return out
