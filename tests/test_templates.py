"""Instant template recogniser and voice clean-up."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import scan_clips  # noqa: E402
from src.speaker import trim_and_level  # noqa: E402
from src.templates import TemplateMatcher, dtw_distances, for_speaker  # noqa: E402
from tools.make_synthetic_data import make_clip, phrase_pattern  # noqa: E402

PIDS = ["P01", "P02", "P03", "P04", "P05"]
SP = {"w": 0.62, "h_up": 0.07, "h_lo": 0.1, "gain": 1.0, "speed": 1.0, "bias": np.zeros((8, 3))}


class TestTemplates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(5)
        for k, pid in enumerate(PIDS):
            for session, reps in ((1, 3), (3, 3)):
                for rep in range(1, reps + 1):
                    path = cls.tmp / "raw" / "u1" / pid / f"{session}_{rep:02d}.npy"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, make_clip(phrase_pattern(k), SP, session, rng)[0])
        cls.clips = scan_clips(cls.tmp / "raw")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_dtw_is_zero_for_identical_and_tolerates_speed(self):
        a = np.random.default_rng(0).normal(size=(32, 4)).astype(np.float32)
        self.assertAlmostEqual(float(dtw_distances(a, a[None])[0]), 0.0, places=3)   # float32 rounding
        shifted = np.concatenate([a[:1], a[:-1]])[None]   # same motion, one frame later
        noise = np.random.default_rng(1).normal(size=(1, 32, 4)).astype(np.float32)
        self.assertLess(dtw_distances(a, shifted)[0], dtw_distances(a, noise)[0])

    def test_recognises_another_day_with_calibrated_confidence(self):
        m = TemplateMatcher.from_clips([c for c in self.clips if c.session == 1], PIDS)
        self.assertEqual(m.label_map, PIDS)
        self.assertGreaterEqual(m.report["loo_accuracy"], 0.9)
        test = [c for c in self.clips if c.session == 3]
        probs = np.stack([m.predict_clip(np.load(c.path))[0] for c in test])
        truth = np.array([PIDS.index(c.phrase_id) for c in test])
        self.assertGreaterEqual((probs.argmax(1) == truth).mean(), 0.9)
        self.assertTrue(np.all(probs.sum(1) <= 1 + 1e-6))
        self.assertGreater(probs.max(1).mean(), 0.6)     # confident enough to be spoken

    def test_unknown_phrase_is_not_more_confident_than_known(self):
        # Same recording conditions for known and unknown queries, so only
        # "is it in the list?" differs. On the real calibration (2026-11-10)
        # unknown phrases averaged 0.37 confidence vs 0.49 for known ones; this
        # guards against unknown phrases looking *more* certain than real ones.
        refs = [c for c in self.clips if c.session == 1 and c.phrase_id != "P05" and c.rep <= 2]
        m = TemplateMatcher.from_clips(refs, PIDS[:4])
        known = [m.predict_clip(np.load(c.path))[0].max() for c in self.clips
                 if c.session == 1 and c.phrase_id != "P05" and c.rep == 3]
        unknown = [m.predict_clip(np.load(c.path))[0].max() for c in self.clips
                   if c.session == 1 and c.phrase_id == "P05"]
        self.assertLessEqual(np.mean(unknown), np.mean(known) + 0.05)

    def test_bad_clips_are_left_out(self):
        good = [c for c in self.clips if c.session == 1]
        still_path = self.tmp / "raw" / "u1" / "P02" / "1_08.npy"
        clip = np.load(good[0].path)
        np.save(still_path, np.repeat(clip[:1], len(clip), 0))         # lips never move
        try:
            from src.dataset import scan_clips as scan
            clips = [c for c in scan(self.tmp / "raw") if c.session == 1]
            m = TemplateMatcher.from_clips(clips, PIDS)
            self.assertIn(("P02 clip 1_08", "lips barely moved"), m.report["dropped"])
            self.assertEqual(len(m.feats), len(clips) - 1)
        finally:
            still_path.unlink()

    def test_cache_is_reused_until_clips_change(self):
        models = self.tmp / "models"
        m1 = for_speaker("u1", PIDS, raw_dir=self.tmp / "raw", model_dir=models)
        cache = models / "templates_u1.npz"
        stamp = cache.stat().st_mtime_ns
        m2 = for_speaker("u1", PIDS, raw_dir=self.tmp / "raw", model_dir=models)
        self.assertEqual(cache.stat().st_mtime_ns, stamp)            # loaded, not rebuilt
        np.testing.assert_allclose(m1.feats, m2.feats)
        extra = self.tmp / "raw" / "u1" / "P01" / "1_09.npy"
        np.save(extra, np.load(self.tmp / "raw" / "u1" / "P01" / "1_01.npy"))
        try:
            m3 = for_speaker("u1", PIDS, raw_dir=self.tmp / "raw", model_dir=models)
            self.assertEqual(len(m3.feats), len(m1.feats) + 1)           # rebuilt with the new clip
        finally:
            extra.unlink()
        self.assertIsNone(for_speaker("nobody", PIDS, raw_dir=self.tmp / "raw", model_dir=models))


class TestVoiceCleanup(unittest.TestCase):
    def test_trims_silence_and_levels(self):
        sr = 16000
        rng = np.random.default_rng(0)
        silence = lambda s: rng.normal(0, 0.001, int(sr * s))
        tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(int(sr * 0.6)) / sr)
        clip = trim_and_level(np.concatenate([silence(0.7), tone, silence(0.7)]), sr, noise_rms=0.001)
        self.assertLess(abs(len(clip) / sr - 0.8), 0.06)            # 0.6 s of sound + 2 x 0.1 s margin
        self.assertLessEqual(np.abs(clip).max(), 0.951)
        self.assertGreater(np.sqrt((clip ** 2).mean()), 0.05)

    def test_speech_in_a_noisy_room_is_kept(self):
        # Measured on the team laptop: room noise RMS ~0.03 (fan / mic gain).
        sr, rng = 16000, np.random.default_rng(1)
        speech = 0.14 * np.sin(2 * np.pi * 200 * np.arange(int(sr * 0.7)) / sr)   # RMS 0.1, ~3.3x the noise
        audio = rng.normal(0, 0.03, int(sr * 2.0))
        audio[int(sr * 0.6):int(sr * 0.6) + len(speech)] += speech
        clip = trim_and_level(audio, sr, noise_rms=0.03)
        self.assertIsNotNone(clip)
        self.assertLess(len(clip) / sr, 1.3)                          # trimmed around the speech
        self.assertIsNone(trim_and_level(rng.normal(0, 0.03, sr * 2), sr, noise_rms=0.03))

    def test_silence_is_rejected(self):
        self.assertIsNone(trim_and_level(np.random.default_rng(0).normal(0, 0.001, 16000), 16000, 0.001))


if __name__ == "__main__":
    unittest.main()
