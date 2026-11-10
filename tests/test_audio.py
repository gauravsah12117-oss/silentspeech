"""Audio engine (src/audio.py) and the speaker's cues, alarms and volume - no sound card needed."""
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import audio  # noqa: E402


class FakeStream:
    """Stands in for sounddevice.OutputStream: the test calls the audio callback itself."""
    last = None

    def __init__(self, samplerate, channels, dtype, callback):
        self.callback, self.samplerate = callback, samplerate
        FakeStream.last = self

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass

    def pull(self, frames):
        out = np.zeros((frames, 1), np.float32)
        self.callback(out, frames, None, None)
        return out[:, 0]


def fake_sounddevice():
    module = types.ModuleType("sounddevice")
    module.OutputStream = FakeStream
    return mock.patch.dict(sys.modules, {"sounddevice": module})


def speech_like(level_db, sr=16000, seconds=1.0, seed=0):
    """Noise shaped like syllables (bursts), at a chosen loudness."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    x = rng.normal(size=t.size) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2)
    x = np.concatenate([np.zeros(sr // 5), x, np.zeros(sr // 5)]).astype(np.float32)   # silence around it
    return x * 10 ** ((level_db - audio.speech_level_db(x, sr)) / 20)


class TestSignals(unittest.TestCase):
    def test_speech_is_levelled_to_one_loudness(self):
        quiet, loud = speech_like(-26), speech_like(-12, seed=1)
        for clip in (quiet, loud):
            self.assertAlmostEqual(audio.speech_level_db(audio.level_speech(clip, 16000), 16000),
                                   audio.SPEECH_LEVEL_DB, delta=0.5)

    def test_limiter_never_clips_and_leaves_quiet_sound_alone(self):
        x = np.linspace(-3, 3, 2001).astype(np.float32)
        y = audio.soft_limit(x)
        self.assertLessEqual(float(np.abs(y).max()), 1.0)
        quiet = np.abs(x) < audio.LIMIT_START
        np.testing.assert_allclose(y[quiet], x[quiet])
        self.assertTrue(np.all(np.diff(y) >= 0))              # still smooth and in order: no distortion jumps

    def test_cues_are_clean_and_distinct(self):
        lengths = {}
        for name in audio.CUES:
            c = audio.cue(name)
            self.assertGreater(len(c), 1000, name)
            self.assertLessEqual(float(np.abs(c).max()), 1.0, name)
            self.assertLess(abs(float(c[0])), 0.01, name)            # soft start
            self.assertLess(float(np.abs(c[-20:]).max()), 0.01, name)   # no click at the end
            lengths[name] = len(c)
        self.assertLess(lengths["done"], lengths["ready"])          # short cues stay short

    def test_pain_alarm_sounds_different_from_emergency(self):
        em, pain = audio.alarm("emergency"), audio.alarm("pain")
        self.assertLessEqual(float(np.abs(em).max()), 1.0)
        spectrum = lambda x: np.abs(np.fft.rfft(x))                      # noqa: E731
        f = lambda x: np.fft.rfftfreq(len(x), 1 / audio.SAMPLE_RATE)[np.argmax(spectrum(x))]  # noqa: E731
        self.assertGreater(f(em), f(pain) * 1.2)                          # emergency is higher pitched
        self.assertNotEqual(len(em), len(pain))

    def test_resample(self):
        x = np.sin(np.linspace(0, 20, 22050)).astype(np.float32)
        y = audio.resample(x, 22050, 44100)
        self.assertEqual(len(y), 44100)
        self.assertAlmostEqual(float(y[-1]), float(x[-1]), places=4)


class TestMixer(unittest.TestCase):
    def test_mixes_sounds_and_reports_when_done(self):
        with fake_sounddevice():
            m = audio.Mixer(sr=8000)
        stream = FakeStream.last
        first = m.play(np.full(100, 0.1, np.float32))
        second = m.play(np.full(50, 0.2, np.float32))      # starts while the first is playing
        out = stream.pull(64)
        np.testing.assert_allclose(out[:50], 0.3, atol=1e-6)
        np.testing.assert_allclose(out[50:], 0.1, atol=1e-6)
        self.assertTrue(second.is_set())
        self.assertFalse(first.is_set())
        self.assertGreater(m.level, 0)
        stream.pull(64)
        self.assertTrue(first.is_set())
        self.assertFalse(m.playing)

    def test_volume_and_limiter(self):
        with fake_sounddevice():
            m = audio.Mixer(sr=8000, volume=2.0)
        m.play(np.full(80, 0.8, np.float32))
        out = FakeStream.last.pull(80)
        self.assertLessEqual(float(np.abs(out).max()), 1.0)        # 1.6 is bent below full scale
        self.assertGreater(float(out.min()), 0.9)                  # but still much louder than 0.8
        with fake_sounddevice():
            self.assertEqual(audio.Mixer(volume=9).volume, audio.VOLUME_RANGE[1])


class TestSpeakerSound(unittest.TestCase):
    def test_cues_alarms_and_volume(self):
        from src.speaker import Speaker
        with fake_sounddevice():
            sp = Speaker({"P01": {"text": "I need water"}}, "nobody", mode="default", warm_cache=False,
                         volume=1.0, sounds=False)
        try:
            mixer = sp._mixer
            self.assertIsNotNone(mixer)
            sp.cue("listen")                                       # cues are off: nothing plays
            self.assertFalse(mixer.playing)
            sp.alert("pain")                                       # alarms always sound
            self.assertTrue(mixer.playing)
            mixer.stop()
            sp.sounds = True
            sp.cue("listen")
            self.assertTrue(mixer.playing)
            self.assertEqual(sp.set_volume(5), audio.VOLUME_RANGE[1])
            self.assertEqual(mixer.volume, audio.VOLUME_RANGE[1])
            self.assertEqual(sp.set_volume(0), audio.VOLUME_RANGE[0])
        finally:
            sp.close(timeout=2)

    def test_settings_are_remembered(self):
        from src.utils import load_settings, save_settings
        tmp = Path(tempfile.mkdtemp())
        saved = config.SETTINGS_FILE
        config.SETTINGS_FILE = tmp / "settings.json"
        try:
            self.assertEqual(load_settings(), {})
            save_settings(volume=1.3)
            save_settings(sounds=False)
            self.assertEqual(load_settings(), {"volume": 1.3, "sounds": False})
        finally:
            config.SETTINGS_FILE = saved
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
