"""Deep model (src/deep.py): learns, calibrates, saves, and stays safe next to the templates."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.make_synthetic_data import make_clip, phrase_pattern  # noqa: E402

try:
    from src import deep             # needs PyTorch
except ImportError:  # torch is optional: the templates work without it
    deep = None

PIDS = ["P01", "P02", "P03", "P04"]
SP = {"w": 0.62, "h_up": 0.07, "h_lo": 0.1, "gain": 1.0, "speed": 1.0, "bias": np.zeros((8, 3))}
FAST = dict(n_nets=2, cv_nets=1, folds=2, epochs=60, bank_copies=6)   # seconds, not minutes


@unittest.skipIf(deep is None, "torch not installed")
class TestDeepModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(3)
        for k, pid in enumerate(PIDS):
            for session, reps in ((1, 4), (3, 3)):
                for rep in range(1, reps + 1):
                    path = cls.tmp / "raw" / "u1" / pid / f"{session}_{rep:02d}.npy"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, make_clip(phrase_pattern(k), SP, session, rng)[0])
        missed = make_clip(phrase_pattern(0), SP, 1, rng)[0]      # a missed cue: the lips never moved
        np.save(cls.tmp / "raw" / "u1" / "P01" / "5_01.npy", np.repeat(missed[:1], len(missed), 0))
        from src.dataset import scan_clips
        clips = scan_clips(cls.tmp / "raw")
        cls.train_clips = [c for c in clips if c.session == 1]
        cls.test_clips = [c for c in clips if c.session == 3]
        pos = [deep.clip_positions(np.load(c.path)) for c in cls.train_clips]
        labels = [PIDS.index(c.phrase_id) for c in cls.train_clips]
        cls.model = deep.train(pos, labels, PIDS, **FAST)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_recognises_clips_from_another_day(self):
        probs = np.stack([self.model.predict_clip(np.load(c.path))[0] for c in self.test_clips])
        truth = np.array([PIDS.index(c.phrase_id) for c in self.test_clips])
        self.assertGreaterEqual((probs.argmax(1) == truth).mean(), 0.75)
        np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-6)

    def test_honest_report(self):
        r = self.model.report
        self.assertEqual(r["n_clips"], len(self.train_clips))
        self.assertTrue(0.0 <= r["cv_accuracy"] <= 1.0)
        self.assertIn(r["temperature"], list(deep.TEMPERATURES))
        probs, truth = self.model.oof
        self.assertEqual(probs.shape, (len(self.train_clips), len(PIDS)))
        self.assertEqual(len(truth), len(self.train_clips))

    def test_save_and_load_give_the_same_answers(self):
        path = self.tmp / "models" / "deep_u1.pt"
        self.model.save(path, "abc")
        loaded, sig = deep.DeepModel.load(path)
        self.assertEqual(sig, "abc")
        self.assertEqual(loaded.label_map, PIDS)
        clip = np.load(self.test_clips[0].path)
        np.testing.assert_allclose(loaded.predict_clip(clip)[0], self.model.predict_clip(clip)[0], atol=1e-6)

    def test_face_lost_gives_no_answer(self):
        self.assertIsNone(self.model.predict_clip(np.full((40, 42, 3), np.nan))[0])

    def test_ranking_and_focus_accuracy(self):
        ranking = self.model.rank_phrases()
        self.assertEqual(sorted(p for p, _ in ranking), PIDS)
        self.assertEqual(self.model.subset_accuracy(PIDS), self.model.report["cv_accuracy"])

    def test_train_for_speaker_saves_and_loads(self):
        model = deep.train_for_speaker("u1", PIDS, raw_dir=self.tmp / "raw", model_dir=self.tmp / "m2", **FAST)
        self.assertIsNotNone(model)
        self.assertIn("templates_cv_accuracy", model.report)
        self.assertEqual(model.report["left_out"], ["P01 clip 5_01"])     # never learned from a still clip
        self.assertEqual(model.report["n_clips"], len(PIDS) * 7)
        self.assertTrue(deep.model_path("u1", self.tmp / "m2").exists())
        if model.report["better_than_templates"]:
            loaded, why = deep.load_for_speaker("u1", PIDS, self.tmp / "m2")
            self.assertIsNotNone(loaded, why)
        # a different phrase list: never used with the wrong phrases
        self.assertIsNone(deep.load_for_speaker("u1", PIDS[:3], self.tmp / "m2")[0])
        self.assertIsNone(deep.load_for_speaker("nobody", PIDS, self.tmp / "m2")[0])
        # up to date until a new clip arrives (e.g. one the app learned from 1/2/3)
        raw = self.tmp / "raw"
        self.assertIsNotNone(deep.up_to_date("u1", PIDS, raw, self.tmp / "m2"))
        extra = raw / "u1" / "P01" / "9_01.npy"
        np.save(extra, make_clip(phrase_pattern(0), SP, 1, np.random.default_rng(9))[0])
        try:
            self.assertIsNone(deep.up_to_date("u1", PIDS, raw, self.tmp / "m2"))
        finally:
            extra.unlink()
        self.assertGreater(deep.estimate_seconds("u1", PIDS, raw), 0)

    def test_too_few_clips_is_refused(self):
        pos = [deep.clip_positions(np.load(c.path)) for c in self.train_clips[:5]]
        with self.assertRaises(ValueError):
            deep.train(pos, [0, 0, 0, 0, 1], PIDS[:2], **FAST)


@unittest.skipIf(deep is None, "torch not installed")
class TestCalibrationAndSafety(unittest.TestCase):
    def test_temperature_softens_overconfident_nets(self):
        rng = np.random.default_rng(0)
        truth = rng.integers(0, 4, 200)
        logits = rng.normal(size=(2, 200, 4)) * 4.0
        right = rng.random(200) < 0.6                     # 60% right, but very sure every time
        logits[:, right, truth[right]] += 12
        t = deep.fit_temperature(logits, None, truth)
        self.assertGreater(t, 1.0)
        calibrated = deep.ensemble_probs(logits, None, t).max(1).mean()
        self.assertLess(abs(calibrated - 0.6), abs(deep.ensemble_probs(logits).max(1).mean() - 0.6))

    def test_templates_none_share_lowers_confidence(self):
        class Stub:                                       # templates that find the clip unlike any phrase
            label_map, report, name = PIDS, {}, "stub"

            def predict_signature(self, sig, bsig=None):
                return np.array([0.1, 0.05, 0.05, 0.1])   # 70% "none of these"

        class Sure:                                       # a deep model that is sure it is P02
            label_map, report, oof = PIDS, {"cv_accuracy": 0.8}, None

            def predict_positions(self, pos):
                return np.array([0.02, 0.94, 0.02, 0.02])

            def subset_accuracy(self, ids):
                return None

        rec = deep.DeepRecognizer(Sure(), Stub())
        clip = make_clip(phrase_pattern(1), SP, 1, np.random.default_rng(0))[0]
        probs, _ = rec.predict_clip(clip)
        self.assertEqual(int(np.argmax(probs)), 1)
        self.assertAlmostEqual(float(probs.sum()), 0.3, places=6)
        self.assertLess(probs.max(), 0.40)                # below CONF_MIN: the app asks to repeat
        self.assertEqual(rec.last_votes["deep"][0], "P02")


if __name__ == "__main__":
    unittest.main()
