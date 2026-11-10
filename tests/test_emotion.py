"""Emotion and pain expression from facial muscle scores (src/emotion.py)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.emotion import (ACTION_UNITS, EmotionReader, describe, pain_level)  # noqa: E402
from src.landmarks import BLENDSHAPE_NAMES  # noqa: E402

IDX = {n: i for i, n in enumerate(BLENDSHAPE_NAMES)}


def face(**aus):
    """Blendshape vector with these action units at these raw scores (e.g. AU12=0.8)."""
    b = np.zeros(len(BLENDSHAPE_NAMES), np.float32)
    for au, v in aus.items():
        for name in ACTION_UNITS[au][1]:
            b[IDX[name]] = v
    return b


def hold(reader, blend, seconds=1.0, t0=0.0, talking=False, fps=13):
    """Show the same face for a while (as a webcam would); returns the last reading and the time."""
    t = t0
    r = None
    for i in range(int(seconds * fps)):
        t = t0 + i / fps
        r = reader.update(blend, t, talking=talking)
    return r, t


# Textbook expressions (EMFACS action units), at clear but not extreme strength
EXPRESSIONS = {
    "neutral": face(),
    "happy": face(AU12=0.8, AU6=0.5),
    "sad": face(AU1=0.5, AU15=0.35, AU4=0.25, AU17=0.4),
    "surprised": face(AU1=0.6, AU2=0.5, AU5=0.35, AU26=0.5),
    "afraid": face(AU1=0.55, AU4=0.4, AU5=0.4, AU20=0.35, AU2=0.3),
    "angry": face(AU4=0.5, AU7=0.6, AU23=0.4),
    "disgusted": face(AU9=0.35, AU10=0.35, AU15=0.2),
}


class TestEmotions(unittest.TestCase):
    def test_textbook_expressions(self):
        for emotion, blend in EXPRESSIONS.items():
            r, _ = hold(EmotionReader(), blend)
            self.assertEqual(r.emotion, emotion, f"{emotion}: read as {r.emotion} {np.round(r.probs, 2)}")
            self.assertGreater(r.confidence, 0.6, emotion)
            self.assertAlmostEqual(float(r.probs.sum()), 1.0, places=6)

    def test_drivers_explain_the_reading(self):
        r, _ = hold(EmotionReader(), EXPRESSIONS["happy"])
        self.assertEqual([w for w, _ in r.drivers], ["smile", "cheeks raised"])
        self.assertEqual(hold(EmotionReader(), EXPRESSIONS["neutral"])[0].drivers, [])

    def test_mouthing_a_phrase_is_not_an_emotion(self):
        # wide-open jaw, stretched and pursed lips: what mouthing "I need water" looks like
        mouthing = face(AU26=0.6, AU20=0.35, AU12=0.3, AU23=0.3, AU17=0.4, AU10=0.3)
        r, _ = hold(EmotionReader(), mouthing, talking=True)
        self.assertEqual(r.emotion, "neutral", np.round(r.probs, 2))
        # ...but a smile while mouthing still shows, through the cheeks
        r, _ = hold(EmotionReader(), mouthing + face(AU6=0.5, AU12=0.5) - face(AU12=0.3), talking=True)
        self.assertEqual(r.emotion, "happy")

    def test_learns_this_persons_resting_face(self):
        resting = face(AU4=0.3, AU7=0.4, AU23=0.25)        # naturally low brows and narrow eyes
        reader = EmotionReader()
        first, t = hold(reader, resting, seconds=0.5)
        r, t = hold(reader, resting, seconds=8, t0=t + 0.1)
        self.assertTrue(first.learning)
        self.assertFalse(r.learning)
        self.assertEqual(r.emotion, "neutral", np.round(r.probs, 2))
        frown = face(AU4=0.7, AU7=0.8, AU23=0.5)           # a real frown still reads as anger
        r, _ = hold(reader, frown, t0=t + 0.1)
        self.assertEqual(r.emotion, "angry")

    def test_one_noisy_frame_does_not_flip_it(self):
        reader = EmotionReader()
        _, t = hold(reader, EXPRESSIONS["neutral"])
        r = reader.update(EXPRESSIONS["surprised"], t + 0.077)
        self.assertEqual(r.emotion, "neutral")

    def test_summary_of_a_phrase(self):
        reader = EmotionReader()
        _, t = hold(reader, EXPRESSIONS["neutral"], seconds=2)
        start = t + 0.05
        _, t = hold(reader, EXPRESSIONS["sad"], seconds=2.5, t0=start)
        s = reader.summary(start, t)
        self.assertEqual(s["emotion"], "sad")
        self.assertEqual(describe(s), "looked sad")
        self.assertIsNone(reader.summary(t + 10))
        self.assertIsNone(describe({"emotion": "neutral", "label": "Neutral", "pain_level": "none"}))


class TestPain(unittest.TestCase):
    def test_pain_grimace(self):
        grimace = face(AU4=0.45, AU7=0.6, AU9=0.3, AU10=0.3, AU43=0.4)
        r, _ = hold(EmotionReader(), grimace, seconds=3)
        self.assertIn(r.pain_level, ("moderate", "strong"), r.pain)
        self.assertEqual(describe({"emotion": "neutral", "label": "Neutral", "pain_level": "strong"}),
                         "pain expression: strong")

    def test_not_pain(self):
        for name, blend in (("big smile", face(AU12=0.9, AU6=0.6)),
                            ("concentrating", face(AU4=0.45, AU7=0.6)),
                            ("eyes closed", face(AU43=0.9)),
                            ("calm", face())):
            r, _ = hold(EmotionReader(), blend, seconds=3)
            self.assertIn(r.pain_level, ("none", "mild"), f"{name}: {r.pain:.2f}")
        self.assertEqual(pain_level(0.0), "none")
        self.assertEqual(pain_level(0.99), "strong")


class TestPromptWithExpression(unittest.TestCase):
    def test_llm_prompt_mentions_the_face_only_when_given(self):
        from src.llm_corrector import build_prompt
        from src.utils import load_phrases
        phrases = load_phrases(phrase_set="full")
        cands = [("P10", 0.48), ("P28", 0.41)]
        plain = build_prompt(cands, [], phrases)
        self.assertNotIn("face", plain.lower())
        hinted = build_prompt(cands, [], phrases, expression="looked afraid (pain expression: moderate)")
        self.assertIn("looked afraid (pain expression: moderate)", hinted)
        self.assertIn('[P10] "Call the nurse"', hinted)


if __name__ == "__main__":
    unittest.main()
