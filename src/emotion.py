"""Facial emotion and pain expression from the Face Landmarker's 52 muscle scores.

Psychologists describe facial expressions with the Facial Action Coding System
(FACS, Ekman & Friesen 1978): each *action unit* (AU) is one visible muscle
movement, e.g. AU12 = lip corners pulled up (zygomaticus major). The Face
Landmarker's blendshapes measure almost the same movements (mouthSmile ~ AU12,
cheekSquint ~ AU6 ...). Emotions are read from the AU combinations that EMFACS,
Ekman's emotion version of FACS, lists for each basic emotion:

    happy       AU12 lip corners up + AU6 cheeks raised (a real, "Duchenne" smile)
    sad         AU1 inner brows up + AU15 lip corners down (+ AU4 brows down, AU17 chin up)
    surprised   AU1 + AU2 brows up (+ AU5 eyes wide, AU26 jaw drop)
    afraid      AU1 + AU4 + AU5: brows up *and* drawn together, eyes wide (+ AU2, AU20 lips stretched)
    angry       AU4 brows down + AU7 eyes narrowed (+ AU23 lips tightened, AU5 glare)
    disgusted   AU9 nose wrinkled (+ AU10 upper lip up, AU15, AU17)

Pain uses the core of the Prkachin & Solomon Pain Intensity score (PSPI, 2008),
the standard FACS pain measure: brows lowered (AU4) + eyes tightened (AU6/AU7)
+ nose wrinkled or upper lip raised (AU9/AU10), with closed eyes (AU43) adding
to it. All three have to move together, so frowning in concentration is not
mistaken for pain.

Three things make it work on a real person in a live app:
  - Personal baseline: resting faces differ (some brows sit low, some eyes look
    narrow). Each muscle is measured against this person's own resting level,
    learned all the time from the last minute (a low percentile), so a
    naturally serious face doesn't read as "angry".
  - Speech-aware: mouthing a phrase moves the jaw and lips a lot, which would
    look like surprise or fear. While the user is mouthing, the mouth muscles
    count much less and the brows, eyes, cheeks and nose carry the reading.
  - Smoothing over about half a second, so a blink or one noisy frame doesn't
    flip the result.

It reads facial *expressions*, not feelings: a face can smile without being
happy, and some people (facial paralysis, Parkinson's) show little expression.
"""
import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.landmarks import BLENDSHAPE_NAMES

EMOTIONS = ("neutral", "happy", "sad", "surprised", "afraid", "angry", "disgusted")
LABELS = {"neutral": "Neutral", "happy": "Happy", "sad": "Sad", "surprised": "Surprised",
          "afraid": "Afraid", "angry": "Angry", "disgusted": "Disgusted"}
COLORS = {                                    # BGR
    "neutral": (190, 172, 150), "happy": (70, 205, 245), "sad": (220, 140, 70),
    "surprised": (200, 110, 230), "afraid": (220, 100, 160), "angry": (75, 75, 235),
    "disgusted": (80, 180, 120),
}

# FACS action unit -> (plain words, blendshapes averaged, score of a clear, strong movement)
ACTION_UNITS = {
    "AU1": ("inner brows up", ("browInnerUp",), 0.55),
    "AU2": ("outer brows up", ("browOuterUpLeft", "browOuterUpRight"), 0.45),
    "AU4": ("brows down", ("browDownLeft", "browDownRight"), 0.45),
    "AU5": ("eyes wide", ("eyeWideLeft", "eyeWideRight"), 0.35),
    "AU6": ("cheeks raised", ("cheekSquintLeft", "cheekSquintRight"), 0.45),
    "AU7": ("eyes narrowed", ("eyeSquintLeft", "eyeSquintRight"), 0.55),
    "AU9": ("nose wrinkled", ("noseSneerLeft", "noseSneerRight"), 0.35),
    "AU10": ("upper lip raised", ("mouthUpperUpLeft", "mouthUpperUpRight"), 0.40),
    "AU12": ("smile", ("mouthSmileLeft", "mouthSmileRight"), 0.60),
    "AU15": ("lip corners down", ("mouthFrownLeft", "mouthFrownRight"), 0.35),
    "AU17": ("chin raised", ("mouthShrugLower",), 0.50),
    "AU20": ("lips stretched", ("mouthStretchLeft", "mouthStretchRight"), 0.35),
    "AU23": ("lips tightened", ("mouthPressLeft", "mouthPressRight"), 0.45),
    "AU26": ("jaw dropped", ("jawOpen",), 0.50),
    "AU43": ("eyes closed", ("eyeBlinkLeft", "eyeBlinkRight"), 0.65),
}
AU_NAMES = list(ACTION_UNITS)
MOUTH_AUS = {"AU10", "AU12", "AU15", "AU17", "AU20", "AU23", "AU26"}   # also moved by mouthing words

# emotion -> (core AUs that must move together, supporting AUs that add confidence)
RULES = {
    "happy": (("AU12",), ("AU6",)),
    "sad": (("AU1", "AU15"), ("AU4", "AU17")),
    "surprised": (("AU1", "AU2"), ("AU5", "AU26")),
    "afraid": (("AU1", "AU4", "AU5"), ("AU2", "AU20", "AU26")),
    "angry": (("AU4", "AU7"), ("AU23", "AU5")),
    "disgusted": (("AU9",), ("AU10", "AU15", "AU17")),
}
# While mouthing a phrase the lips move anyway (a smile then has to show in the cheeks),
# and people raise their brows for emphasis (so surprise also needs wide eyes).
# Measured on a real clip: brows up while mouthing "I can't breathe" read as surprise before.
TALKING_RULES = {"happy": (("AU6",), ("AU12",)), "surprised": (("AU1", "AU2", "AU5"), ("AU26",))}

TALK_WEIGHT = 0.3        # how much mouth muscles count while the user is mouthing a phrase
NEUTRAL_LEVEL = 0.20     # evidence an emotion needs before it beats "neutral"
NEUTRAL_LEVEL_TALKING = 0.30   # ... and while mouthing, when the whole face moves more
SHARPNESS = 16.0         # evidence -> probability (higher = more decisive)
EPS = 0.04               # soft AND: one missing core muscle doesn't zero the score, but costs a lot
BASELINE_SECONDS = 60.0  # the resting face is learned from the last minute
BASELINE_PERCENTILE = 15
BASELINE_CAP = 0.7       # a resting level never absorbs more than 70% of a strong movement
PAIN_LEVELS = ((0.20, "none"), (0.35, "mild"), (0.50, "moderate"), (9.0, "strong"))

_IDX = {n: i for i, n in enumerate(BLENDSHAPE_NAMES)}
_COLS = [np.array([_IDX[n] for n in ACTION_UNITS[a][1]]) for a in AU_NAMES]
_FULL = np.array([ACTION_UNITS[a][2] for a in AU_NAMES], np.float64)


def action_units(blend):
    """Blendshape scores (52,) -> raw activation of each action unit (15,)."""
    b = np.asarray(blend, np.float64)
    return np.array([b[c].mean() for c in _COLS])


def soft_and(values, eps=EPS):
    """Geometric-style mean: high only when all values are high together."""
    v = np.clip(np.asarray(values, np.float64), 0.0, 1.0)
    return float(np.exp(np.mean(np.log(v + eps))) - eps)


def emotion_evidence(au, talking=False):
    """AU intensities {AU: 0..1} -> evidence (0..1) for each emotion except neutral."""
    x = {a: v * (TALK_WEIGHT if talking and a in MOUTH_AUS else 1.0) for a, v in au.items()}
    out = []
    for emo in EMOTIONS[1:]:
        core, support = (TALKING_RULES.get(emo) if talking else None) or RULES[emo]
        c = soft_and([x[a] for a in core])
        s = float(np.mean([x[a] for a in support])) if support else 0.0
        out.append(max(0.0, c) * (0.65 + 0.35 * s))
    return np.array(out)


def probabilities(evidence, talking=False):
    level = NEUTRAL_LEVEL_TALKING if talking else NEUTRAL_LEVEL
    logits = SHARPNESS * np.concatenate([[level], np.asarray(evidence, np.float64)])
    e = np.exp(logits - logits.max())
    return e / e.sum()


def pain_score(au, talking=False):
    """PSPI-style pain expression score, 0..1.

    A smile also narrows the eyes and lifts the upper lip (measured on a real
    smiling face: eyes 0.31, upper lip 0.55), so those count less while smiling.
    """
    smile = au["AU12"]
    levator = max(au["AU9"], au["AU10"] * (1.0 - smile) * (TALK_WEIGHT if talking else 1.0))
    orbit = max(au["AU7"], au["AU6"]) * (1.0 - 0.5 * smile)
    core = soft_and([au["AU4"], orbit, levator], eps=0.02)
    return float(min(1.0, max(0.0, core) * (1.0 + 0.15 * au["AU43"])))


def pain_level(score):
    return next(name for limit, name in PAIN_LEVELS if score < limit)


@dataclass
class Reading:
    emotion: str
    confidence: float
    probs: np.ndarray
    pain: float
    pain_level: str
    drivers: list = field(default_factory=list)     # [(plain words, intensity)] behind the emotion
    talking: bool = False
    learning: bool = True                           # still learning this person's resting face

    @property
    def label(self):
        return LABELS[self.emotion]


class EmotionReader:
    """Feed it every frame's blendshapes; it keeps a smoothed, personal reading."""

    def __init__(self, smooth_s=0.35, pain_smooth_s=0.8):
        self.smooth_s, self.pain_smooth_s = smooth_s, pain_smooth_s
        self.samples = deque()                      # (time, raw AUs) of resting frames
        self.baseline = 0.25 * _FULL                # until the person's own is learned
        self.learned = False
        self.probs = probabilities(np.zeros(len(EMOTIONS) - 1))
        self.pain = 0.0
        self.last_t = None
        self.history = deque()                      # (time, probs, pain) for the last 20 s
        self.timeline = deque(maxlen=120)           # (time, emotion index, confidence), 4 per second, 30 s
        self.reading = None

    def update(self, blend, now=None, talking=False):
        now = time.perf_counter() if now is None else now
        raw = action_units(blend)
        if not talking and (not self.samples or now - self.samples[-1][0] >= 0.2):
            self.samples.append((now, raw))
            while now - self.samples[0][0] > BASELINE_SECONDS:
                self.samples.popleft()
            if len(self.samples) >= 15:             # about 3 seconds of resting face
                rest = np.percentile(np.stack([r for _, r in self.samples]), BASELINE_PERCENTILE, axis=0)
                self.baseline = np.minimum(rest, BASELINE_CAP * _FULL)
                self.learned = True
        intensity = np.clip((raw - self.baseline) / np.maximum(_FULL - self.baseline, 0.1), 0.0, 1.0)
        au = dict(zip(AU_NAMES, intensity))
        target = probabilities(emotion_evidence(au, talking), talking)
        pain_now = pain_score(au, talking)
        dt = 1.0 if self.last_t is None else float(np.clip(now - self.last_t, 0.0, 1.0))
        a = 1.0 - math.exp(-dt / self.smooth_s)
        ap = 1.0 - math.exp(-dt / self.pain_smooth_s)
        self.probs = (1 - a) * self.probs + a * target
        self.pain = (1 - ap) * self.pain + ap * pain_now
        self.last_t = now
        self.history.append((now, self.probs.copy(), self.pain))
        while now - self.history[0][0] > 20.0:
            self.history.popleft()
        k = int(np.argmax(self.probs))
        if not self.timeline or now - self.timeline[-1][0] >= 0.25:
            self.timeline.append((now, k, float(self.probs[k])))
        self.reading = Reading(EMOTIONS[k], float(self.probs[k]), self.probs.copy(), self.pain,
                               pain_level(self.pain), self._drivers(EMOTIONS[k], au, talking), talking,
                               not self.learned)
        return self.reading

    @staticmethod
    def _drivers(emotion, au, talking):
        if emotion == "neutral":
            return []
        core, support = (TALKING_RULES.get(emotion) if talking else None) or RULES[emotion]
        # strongest first; on a tie the emotion's core muscle comes first
        moving = sorted(((round(au[a], 3), a in core, a) for a in core + support if au[a] >= 0.15), reverse=True)
        return [(ACTION_UNITS[a][0], float(au[a])) for _, _, a in moving[:2]]

    def summary(self, since, until=None):
        """Average reading over a time window, e.g. while a phrase was mouthed. None if no frames."""
        until = math.inf if until is None else until
        rows = [(p, pain) for t, p, pain in self.history if since <= t <= until]
        if not rows:
            return None
        probs = np.mean([p for p, _ in rows], axis=0)
        pain = max(pain for _, pain in rows)
        k = int(np.argmax(probs))
        return {"emotion": EMOTIONS[k], "label": LABELS[EMOTIONS[k]], "confidence": float(probs[k]),
                "pain": float(pain), "pain_level": pain_level(pain)}

    def reset(self):
        """Forget the learned resting face (e.g. a different person sits down)."""
        self.__init__(self.smooth_s, self.pain_smooth_s)


def describe(summary):
    """Short words for the transcript and the LLM, e.g. 'looked sad (pain expression: mild)'; None if calm."""
    if not summary:
        return None
    parts = []
    if summary["emotion"] != "neutral":
        parts.append(f"looked {summary['label'].lower()}")
    if summary["pain_level"] != "none":
        parts.append(f"pain expression: {summary['pain_level']}")
    if not parts:
        return None
    return parts[0] + (f" ({parts[1]})" if len(parts) > 1 else "")
