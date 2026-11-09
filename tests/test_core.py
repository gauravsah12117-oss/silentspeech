"""Unit tests for the parts that do not need a webcam.

    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.augment import augment, mirror, time_stretch  # noqa: E402
from src.dataset import Clip, make_splits  # noqa: E402
from src.decision import decide  # noqa: E402
from src.features import (build_features, check_clip, fill_missing, normalise,  # noqa: E402
                          pad_to_length, resample)
from src.landmarks import EYE_CORNERS, LIPS_INNER, LIPS_OUTER, MIRROR_PERM  # noqa: E402
from src.llm_corrector import build_prompt, parse_choice  # noqa: E402
from src.utils import load_phrases  # noqa: E402
from tools.make_synthetic_data import lip_shape, make_clip, phrase_pattern  # noqa: E402

SP = {"w": 0.62, "h_up": 0.07, "h_lo": 0.1, "gain": 1.0, "speed": 1.0, "bias": np.zeros((8, 3))}


def raw_clip(seed=0, session=1):
    clip, _ = make_clip(phrase_pattern(seed), SP, session, np.random.default_rng(seed))
    return fill_missing(clip)


class TestLandmarks(unittest.TestCase):
    def test_lip_lists_are_rings_in_mediapipe_mesh(self):
        try:
            from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
        except ImportError:
            self.skipTest("mediapipe not installed")
        edges = {frozenset(e) for e in FACEMESH_LIPS}
        for ring in (LIPS_OUTER, LIPS_INNER):
            # every consecutive pair (and last -> first) is a lip edge in MediaPipe's own mesh
            for a, b in zip(ring, ring[1:] + ring[:1]):
                self.assertIn(frozenset((a, b)), edges, f"{a}-{b} is not a lip edge")
        self.assertEqual(len(set(LIPS_OUTER + LIPS_INNER + EYE_CORNERS)), 42)

    def test_mirror_perm_is_an_involution(self):
        self.assertTrue(np.array_equal(MIRROR_PERM[MIRROR_PERM], np.arange(42)))

    def test_mirror_maps_known_twins(self):
        ids = LIPS_OUTER + LIPS_INNER + EYE_CORNERS
        twins = {61: 291, 146: 375, 91: 321, 181: 405, 84: 314, 17: 17, 0: 0, 185: 409, 40: 270,
                 39: 269, 37: 267, 78: 308, 95: 324, 88: 318, 178: 402, 87: 317, 14: 14, 13: 13,
                 191: 415, 80: 310, 81: 311, 82: 312, 33: 263}
        for a, b in twins.items():
            self.assertEqual(ids[MIRROR_PERM[ids.index(a)]], b)

    def test_mirror_of_symmetric_mouth_is_itself(self):
        shape = lip_shape(0.2, 1.0, 0.3, SP)[None]
        np.testing.assert_allclose(mirror(shape), shape, atol=1e-6)


class TestFeatures(unittest.TestCase):
    def test_normalise_removes_head_pose(self):
        clip = raw_clip()
        a = np.deg2rad(17)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        moved = clip.copy()
        moved[..., :2] = (clip[..., :2] @ rot.T) * 1.7 + np.array([55.0, -20.0])
        np.testing.assert_allclose(normalise(clip), normalise(moved), atol=1e-4)

    def test_normalise_shape_and_scale(self):
        clip = raw_clip()
        pos = normalise(clip)
        self.assertEqual(pos.shape, (len(clip), 40, 2))
        width = np.linalg.norm(pos[:, 0] - pos[:, 10], axis=1)   # mouth corners, in eye distances
        self.assertTrue(((width > 0.3) & (width < 1.0)).all())

    def test_feature_shape(self):
        self.assertEqual(build_features(raw_clip()).shape, (config.TARGET_FRAMES, 160))
        self.assertEqual(build_features(raw_clip(), use_velocity=False).shape, (config.TARGET_FRAMES, 80))

    def test_resample_keeps_endpoints(self):
        seq = np.arange(10, dtype=np.float32)[:, None]
        out = resample(seq, 64)
        self.assertEqual(out.shape, (64, 1))
        self.assertAlmostEqual(out[0, 0], 0)
        self.assertAlmostEqual(out[-1, 0], 9)

    def test_fill_missing(self):
        clip = raw_clip()
        holed = clip.copy()
        holed[5:8] = np.nan
        filled = fill_missing(holed)
        self.assertFalse(np.isnan(filled).any())
        self.assertIsNone(fill_missing(np.full((5, 42, 3), np.nan)))

    def test_pad_to_length(self):
        clip = raw_clip()[:30]
        out = pad_to_length(clip, 75)
        self.assertEqual(len(out), 75)
        np.testing.assert_array_equal(out[0], clip[0])

    def test_quality_check(self):
        clip = raw_clip()
        ok, _, _ = check_clip(clip, 2.5)
        self.assertTrue(ok)
        still = np.repeat(clip[:1], len(clip), 0)
        ok, reason, _ = check_clip(still, 2.5)
        self.assertFalse(ok)
        self.assertIn("barely moved", reason)
        lost = clip.copy()
        lost[:20] = np.nan
        self.assertFalse(check_clip(lost, 2.5)[0])
        self.assertFalse(check_clip(clip, 6.0)[0])  # 75 frames in 6 s = too slow


class TestAugment(unittest.TestCase):
    def test_shapes_and_values(self):
        pos = normalise(raw_clip())
        rng = np.random.default_rng(0)
        for _ in range(20):
            out = augment(pos, rng)
            self.assertEqual(out.shape, pos.shape)
            self.assertTrue(np.isfinite(out).all())

    def test_time_stretch_keeps_length(self):
        pos = normalise(raw_clip())
        rng = np.random.default_rng(1)
        for f in (0.8, 1.0, 1.2):
            self.assertEqual(len(time_stretch(pos, f, rng)), len(pos))


class TestDecision(unittest.TestCase):
    def setUp(self):
        self.phrases = load_phrases(phrase_set="full")
        self.labels = ["P01", "P12", "P28"]

    def test_thresholds(self):
        self.assertEqual(decide([0.8, 0.1, 0.1], self.labels, self.phrases).action, "accept")
        self.assertEqual(decide([0.5, 0.3, 0.2], self.labels, self.phrases).action, "llm")
        self.assertEqual(decide([0.35, 0.33, 0.32], self.labels, self.phrases).action, "repeat")

    def test_emergency_alert_below_accept(self):
        d = decide([0.45, 0.48, 0.07], self.labels, self.phrases)   # P12 "I can't breathe"
        self.assertEqual(d.alert_phrase_id, "P12")
        self.assertEqual(d.action, "llm")
        self.assertIsNone(decide([0.9, 0.05, 0.05], self.labels, self.phrases).alert_phrase_id)


class TestLLMParsing(unittest.TestCase):
    def setUp(self):
        self.phrases = load_phrases(phrase_set="full")
        self.cands = [("P10", 0.48), ("P28", 0.41), ("P11", 0.06)]

    def test_valid_json(self):
        self.assertEqual(parse_choice('{"choice": "P10", "reason": "pain"}', self.cands, self.phrases)[0], "P10")

    def test_json_inside_text(self):
        text = 'Sure! ```json\n{"choice": "P28", "reason": "family"}\n```'
        self.assertEqual(parse_choice(text, self.cands, self.phrases)[0], "P28")

    def test_rejects_invented_phrase(self):
        self.assertIsNone(parse_choice('{"choice": "P99", "reason": "x"}', self.cands, self.phrases))
        self.assertIsNone(parse_choice('{"choice": "Bring me tea"}', self.cands, self.phrases))

    def test_phrase_text_instead_of_id(self):
        self.assertEqual(parse_choice('{"choice": "Call the nurse"}', self.cands, self.phrases)[0], "P10")

    def test_prompt_contains_candidates_and_history(self):
        prompt = build_prompt(self.cands, ["P09"], self.phrases)
        self.assertIn('[P10] "Call the nurse" (probability 0.48)', prompt)
        self.assertIn('- "I am in pain"', prompt)


class TestSplits(unittest.TestCase):
    def test_session_split_and_holdout(self):
        clips = [Clip(Path(f"{s}{p}{se}{r}"), s, p, se, r)
                 for s in ("s01", "s02", "s03") for p in ("P01", "P02")
                 for se in (1, 2, 3) for r in range(1, 8)]
        sp = make_splits(clips, holdout_speaker="s03")
        self.assertTrue(all(c.session != 3 for c in sp["train"] + sp["val"]))
        self.assertTrue(all(c.session == 3 for c in sp["test"]))
        self.assertTrue(all(c.speaker != "s03" for c in sp["train"] + sp["val"] + sp["test"]))
        self.assertEqual(len(sp["unseen"]), 42)
        self.assertFalse(set(sp["train"]) & set(sp["val"]))


class TestModel(unittest.TestCase):
    def test_forward_and_size(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from src.model import build_model, count_parameters
        for arch in ("conv_bigru", "bigru", "transformer"):
            m = build_model({"arch": arch, "n_classes": 30})
            self.assertEqual(m(torch.zeros(3, config.TARGET_FRAMES, 160)).shape, (3, 30))
        n = count_parameters(build_model({"arch": "conv_bigru", "n_classes": 30}))
        self.assertTrue(650_000 < n < 760_000, n)


if __name__ == "__main__":
    unittest.main()
