"""Speech output: the user's cloned voice, or an offline TTS fallback.

Cloned audio is rendered in advance (notebooks/voice_cloning_colab.ipynb) as
one WAV per phrase in audio/phrase_audio/<speaker>/<phrase_id>.wav, so playback
starts almost instantly. A sentence is played phrase by phrase. If a phrase
has no cloned clip, pyttsx3 speaks it instead, so the demo never goes silent.

All audio runs on one background thread so the camera view never freezes.
"""
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

import config


class Speaker:
    def __init__(self, phrases, speaker_id=None, mode=None, audio_dir=None):
        self.phrases = phrases
        self.speaker_id = speaker_id or config.SPEAKER_ID
        self.mode = mode or config.VOICE_MODE
        self.audio_dir = Path(audio_dir or config.PHRASE_AUDIO_DIR) / self.speaker_id
        self.last_latency_ms = None
        self._queue = queue.Queue()
        self._busy = threading.Event()
        self._sd = self._sf = None
        try:
            import sounddevice as sd
            import soundfile as sf
            self._sd, self._sf = sd, sf
        except Exception as e:  # missing library or no audio device
            print(f"[speaker] audio playback unavailable ({e}); using text-to-speech only")
        self._thread = threading.Thread(target=self._worker, name="speaker", daemon=True)
        self._thread.start()

    # -- public API ---------------------------------------------------------

    @property
    def label(self):
        if self.mode == "cloned":
            return "cloned voice (AI-generated)"
        return "default TTS voice"

    @property
    def busy(self):
        return self._busy.is_set() or not self._queue.empty()

    def wav_path(self, phrase_id):
        return self.audio_dir / f"{phrase_id}.wav"

    def missing_audio(self):
        return [pid for pid in self.phrases if not self.wav_path(pid).exists()]

    def say(self, phrase_ids, t_ref=None):
        """Queue phrases to be spoken. t_ref (perf_counter) is used to measure
        latency until the first sound starts."""
        self._queue.put(("say", list(phrase_ids), t_ref))

    def alert(self):
        """Caregiver alert tone."""
        self._queue.put(("alert", None, None))

    def close(self, timeout=5.0):
        self._queue.put(("stop", None, None))
        self._thread.join(timeout)

    # -- worker -------------------------------------------------------------

    def _worker(self):
        if sys.platform == "win32":  # pyttsx3 uses COM; each thread must initialise it
            try:
                import comtypes
                comtypes.CoInitialize()
            except Exception:
                pass
        while True:
            kind, ids, t_ref = self._queue.get()
            if kind == "stop":
                return
            self._busy.set()
            try:
                if kind == "alert":
                    self._beep()
                    continue
                for n, pid in enumerate(ids):
                    if n:
                        time.sleep(config.PHRASE_GAP_S)
                    wav = self.wav_path(pid)
                    if self.mode == "cloned" and wav.exists() and self._sd:
                        data, sr = self._sf.read(str(wav), dtype="float32")
                        self._mark_start(t_ref, n)
                        self._sd.play(data, sr)
                        self._sd.wait()
                    else:
                        self._tts(self.phrases[pid]["text"], t_ref, n)
            except Exception as e:
                print(f"[speaker] playback error: {e}")
            finally:
                self._busy.clear()

    def _mark_start(self, t_ref, n):
        if t_ref is not None and n == 0:
            self.last_latency_ms = (time.perf_counter() - t_ref) * 1000

    def _tts(self, text, t_ref, n):
        try:
            import pyttsx3
        except ImportError:
            print(f"[speaker] (no TTS installed) would say: {text}")
            return
        # A fresh engine for every utterance: a reused engine silently skips
        # everything after its first utterance (seen with pyttsx3 2.99 on Windows).
        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", config.TTS_RATE)
            self._mark_start(t_ref, n)
            engine.say(text)
            engine.runAndWait()
        finally:
            engine.stop()
            del engine

    def _beep(self):
        if not self._sd:
            print("\a[speaker] ALERT")
            return
        sr = 22050
        t = np.arange(int(sr * 0.18)) / sr
        fade = np.minimum(1, np.minimum(t, t[::-1]) * 60)
        tone = lambda f: 0.4 * np.sin(2 * np.pi * f * t) * fade
        gap = np.zeros(int(sr * 0.06))
        pattern = np.concatenate([tone(1046), gap, tone(784), gap] * 3).astype(np.float32)
        self._sd.play(pattern, sr)
        self._sd.wait()
