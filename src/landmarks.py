"""Face and lip tracking with MediaPipe Face Mesh.

Face Mesh finds 468 face landmarks per frame on a CPU. We keep 42 of them:
20 outer-lip points, 20 inner-lip points and the two outer eye corners
(used only for normalisation).

Coordinates are returned in *pixel units* (x * width, y * height, z * width).
MediaPipe's own x and y are fractions of the image width and height, which
have different scales on a 4:3 frame; converting to pixels keeps angles and
distances correct, so normalisation can rotate the points properly.
"""
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


class LipTracker:
    """Wraps MediaPipe Face Mesh and returns the 42 points we use."""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        import mediapipe as mp  # imported here so the rest of the code runs without it

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        """Return a (42, 3) float32 array in pixel units, or None if no face."""
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0].landmark
        return np.array([(lm[i].x * w, lm[i].y * h, lm[i].z * w) for i in LANDMARK_IDS],
                        dtype=np.float32)

    def close(self):
        self._mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


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
