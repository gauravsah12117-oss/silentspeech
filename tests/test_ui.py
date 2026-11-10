"""Interface toolkit, the live screen, custom phrases and the conversation transcript."""
import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import ui  # noqa: E402

HAVE_CV = bool(importlib.util.find_spec("cv2") and importlib.util.find_spec("mediapipe"))


class TestToolkit(unittest.TestCase):
    def test_wrap_respects_width_and_line_limit(self):
        text = "Please stay with me while the doctor is coming to see me today"
        lines = ui.wrap(text, 150, 16)
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(ui.text_width(line, 16) <= 150 or " " not in line for line in lines))
        short = ui.wrap(text, 150, 16, max_lines=2)
        self.assertEqual(len(short), 2)
        self.assertTrue(short[-1].endswith("..."))

    @unittest.skipUnless(HAVE_CV, "opencv/mediapipe not installed")
    def test_painter_draws_latin_and_devanagari(self):
        img = np.zeros((60, 300, 3), np.uint8)
        p = ui.Painter()
        p.text("Call the nurse", (10, 10), 16)
        p.text("राम लाई बोलाउनुहोस्", (10, 34), 16)
        p.flush(img)
        self.assertGreater(img[:30].sum(), 0)
        self.assertGreater(img[30:].sum(), 0)

    def test_every_phrase_has_a_picture_symbol(self):
        from src.utils import load_phrases
        phrases = load_phrases(phrase_set="full", custom=False)
        self.assertEqual([pid for pid, ph in phrases.items() if not ph.get("icon")], [])
        self.assertEqual(ui.phrase_icon(phrases["P01"]), "\U0001F4A7")            # water drop
        custom = {"id": "C01", "text": "Turn on the news", "category": "custom", "emergency": False}
        self.assertEqual(ui.phrase_icon(custom), "⭐")
        self.assertEqual(ui.phrase_icon({**custom, "emergency": True}), "\U0001F198")
        self.assertEqual(ui.phrase_icon(None), "")

    @unittest.skipUnless(HAVE_CV, "opencv/mediapipe not installed")
    def test_new_widgets_draw(self):
        img = np.zeros((70, 420, 3), np.uint8)
        p = ui.Painter()
        ui.ring_gauge(img, 30, 35, 20, 0.6, ui.GREEN)
        ui.logo(img, 60, 16, 38)
        left = ui.keycap(img, p, 410, 35, "SPACE", "speak")
        self.assertLess(left, 410)
        self.assertLess(ui.badge(img, p, left - 10, 35, "READY", ui.ACCENT), left - 10)
        p.icon("\U0001F4A7", (120, 35), 20)
        p.flush(img)
        for x0, x1 in ((8, 52), (60, 98), (left, 410)):
            self.assertGreater(img[:, x0:x1].sum(), 0, (x0, x1))
        ui.bar(img, 10, 60, 100, 6, 0.004, ui.RED)              # a 0.4% bar draws no red dot
        self.assertLess(int(img[60:66, 10:16, 2].max()), 100)    # red channel: only the grey background

    @unittest.skipUnless(HAVE_CV, "opencv/mediapipe not installed")
    def test_mouth_inset_and_muscles(self):
        frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        ring = np.stack([320 + 60 * np.cos(theta), 330 + 25 * np.sin(theta), np.zeros(20)], 1)
        lips = np.concatenate([ring, ring * [1, 1, 0] + [0, 0, 0], [[260, 200, 0], [380, 200, 0]]]).astype(np.float32)
        out = ui.mouth_inset(frame, lips, (240, 124))
        self.assertEqual(out.shape, (124, 240, 3))
        edge = lips.copy()
        edge[:, 0] -= 300                       # mouth partly outside the frame: still works
        self.assertEqual(ui.mouth_inset(frame, edge, (240, 124)).shape, (124, 240, 3))
        self.assertEqual(ui.mouth_inset(frame, None, (240, 124)).shape, (124, 240, 3))
        blend = np.zeros(52, np.float32)
        blend[25] = 0.8                          # jawOpen
        values = ui.muscle_values(blend)
        self.assertAlmostEqual(float(values[0]), 0.8, places=5)
        self.assertIsNone(ui.muscle_values(None))

    def test_panel_cache_rerenders_only_on_change(self):
        cache, calls = ui.PanelCache(), []
        render = lambda: calls.append(1) or len(calls)
        self.assertEqual(cache.get("a", 1, render), 1)
        self.assertEqual(cache.get("a", 1, render), 1)
        self.assertEqual(cache.get("a", 2, render), 2)
        self.assertEqual(len(calls), 2)


class TestCustomPhrases(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.saved = config.CUSTOM_PHRASE_FILE
        config.CUSTOM_PHRASE_FILE = self.tmp / "custom.json"

    def tearDown(self):
        config.CUSTOM_PHRASE_FILE = self.saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_merge_duplicate_remove(self):
        import add_phrase
        from src.utils import load_phrases
        p1 = add_phrase.add("Turn on the news")
        p2 = add_phrase.add("राम लाई बोलाउनुहोस्", emergency=True)
        self.assertEqual((p1["id"], p2["id"]), ("C01", "C02"))
        for phrase_set in ("mvp", "full"):
            ph = load_phrases(phrase_set=phrase_set)
            self.assertIn("C01", ph)
            self.assertTrue(ph["C02"]["emergency"])
        with self.assertRaises(SystemExit):
            add_phrase.add("turn ON the   news")      # same words: refused
        with self.assertRaises(SystemExit):
            add_phrase.add("x" * 61)                   # too long
        add_phrase.main(["--remove", "c01"])
        self.assertNotIn("C01", load_phrases(phrase_set="full"))
        self.assertEqual(add_phrase.next_id(json.loads(config.CUSTOM_PHRASE_FILE.read_text(encoding="utf-8"))),
                         "C03")


@unittest.skipUnless(HAVE_CV, "opencv/mediapipe not installed")
class TestLiveScreen(unittest.TestCase):
    """The live screen renders in every state, clicks work, and the transcript is written."""

    def setUp(self):
        import app as appmod
        self.tmp = Path(tempfile.mkdtemp())
        a = appmod.App(argparse.Namespace(mode="ptt", voice="default", no_llm=True, speaker="nobody-here",
                                          camera=None, model=None, recognizer="templates"))
        a.speaker.say = lambda *x, **k: None
        a.speaker.alert = lambda: None
        a.transcript = self.tmp / "conversation.txt"
        self.app, self.appmod = a, appmod
        self.frame = np.full((480, 640, 3), 90, np.uint8)

    def tearDown(self):
        self.app.speaker.close(timeout=1)
        self.app.llm.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_renders_and_records(self):
        a, now = self.app, time.perf_counter()
        self.assertEqual(a.render(self.frame, None, 0, 0, now).shape, (self.appmod.H, self.appmod.W, 3))
        a.commit("P01", "Spoke it.", now, [("P01", 0.9), ("P22", 0.05)], source="lips", confidence=0.9)
        a.tap("P28", now)
        a.alert = "P12"
        out = a.render(self.frame, None, 0, 0, now)
        self.assertEqual(out.shape, (self.appmod.H, self.appmod.W, 3))
        self.assertEqual([e["source"] for e in a.log], ["lips", "tap"])
        lines = a.transcript.read_text(encoding="utf-8").splitlines()
        self.assertIn("I need water  [lips 90%]", lines[0])
        self.assertIn("Call my mother  [tapped]", lines[1])
        a.clicks.append((50, 20))                          # clicking the red banner acknowledges
        a.handle_clicks(now)
        self.assertIsNone(a.alert)
        self.assertIn("alert acknowledged", a.transcript.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
