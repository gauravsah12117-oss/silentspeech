"""Speech output: the user's own voice, or an offline computer voice.

Own voice = one WAV per phrase in audio/phrase_audio/<speaker>/<phrase_id>.wav,
either recorded directly with record_voice.py (voice banking) or rendered by
the cloned-voice notebook (notebooks/voice_cloning_colab.ipynb). A manifest
(voice.json) remembers which clips are genuine recordings, so the app can
label AI-generated audio honestly.

If a phrase has no clip, the offline computer voice (pyttsx3) speaks it. Its
clips are rendered once into audio/tts_cache/ (in the background when the app
is idle, or up front by start.py), so after that they also start instantly.

All audio runs on one background thread so the camera view never freezes.
"""
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

import config

MANIFEST = "voice.json"


# --- Files ---------------------------------------------------------------------

def voice_dir(speaker_id, audio_dir=None):
    return Path(audio_dir or config.PHRASE_AUDIO_DIR) / speaker_id


def load_manifest(directory):
    path = Path(directory) / MANIFEST
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def mark_recorded(directory, phrase_id):
    """Note that this phrase's WAV is a genuine recording, not an AI clone."""
    data = load_manifest(directory)
    data[phrase_id] = "recorded"
    path = Path(directory) / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def tts_cache_dir():
    return config.AUDIO_DIR / "tts_cache" / f"rate{config.TTS_RATE}"


# --- Audio clean-up (used by record_voice.py) ----------------------------------

def trim_and_level(audio, sr, noise_rms=0.0, target_rms=0.08, peak=0.95, margin_s=0.1):
    """Cut leading/trailing silence, remove DC, set a consistent loudness.

    Returns None if there is no clear sound above the background noise.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    block = max(1, int(sr * 0.02))
    n = len(audio) // block
    if n < 3:
        return None
    rms = np.sqrt((audio[: n * block].reshape(n, block) ** 2).mean(1))
    if rms.max() < max(4 * noise_rms, 0.004):
        return None
    threshold = max(3 * noise_rms, 0.1 * rms.max())
    voiced = np.where(rms > threshold)[0]
    a = max(0, voiced[0] * block - int(margin_s * sr))
    b = min(len(audio), (voiced[-1] + 1) * block + int(margin_s * sr))
    seg = audio[a:b] - audio[a:b].mean()
    gain = min(target_rms / max(float(np.sqrt((seg ** 2).mean())), 1e-6),
               peak / max(float(np.abs(seg).max()), 1e-6))
    seg = seg * gain
    fade = min(len(seg) // 4, int(sr * 0.01))
    if fade > 1:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        seg[:fade] *= ramp
        seg[-fade:] *= ramp[::-1]
    return seg.astype(np.float32)


def _init_com():
    """pyttsx3 uses COM on Windows; every thread must initialise it."""
    if sys.platform == "win32":
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass


def render_tts(text, path):
    """Render text to a WAV with the offline voice. Returns True on success."""
    import pyttsx3

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.wav")
    # A fresh engine every time: a reused engine silently skips everything
    # after its first job (seen with pyttsx3 2.99 on Windows).
    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", config.TTS_RATE)
        engine.save_to_file(text, str(tmp))
        engine.runAndWait()
    finally:
        engine.stop()
        del engine
    if tmp.exists() and tmp.stat().st_size > 1000:
        tmp.replace(path)
        return True
    tmp.unlink(missing_ok=True)
    return False


def render_tts_cache(phrases, progress=None):
    """Render the offline voice for every phrase that isn't cached yet."""
    _init_com()
    todo = [pid for pid in phrases if not (tts_cache_dir() / f"{pid}.wav").exists()]
    for n, pid in enumerate(todo, 1):
        try:
            render_tts(phrases[pid]["text"], tts_cache_dir() / f"{pid}.wav")
        except Exception as e:
            print(f"[speaker] could not render {pid}: {e}")
            return
        if progress:
            progress(n, len(todo))


# --- Speaker -------------------------------------------------------------------

class Speaker:
    def __init__(self, phrases, speaker_id=None, mode=None, audio_dir=None, warm_cache=True):
        self.phrases = phrases
        self.speaker_id = speaker_id or config.SPEAKER_ID
        self.mode = mode or config.VOICE_MODE
        self.audio_dir = voice_dir(self.speaker_id, audio_dir)
        self.last_latency_ms = None
        self._queue = queue.Queue()
        self._busy = threading.Event()
        self._warm = warm_cache
        self._sd = self._sf = None
        try:
            import sounddevice as sd
            import soundfile as sf
            self._sd, self._sf = sd, sf
        except Exception as e:  # missing library or no audio device
            print(f"[speaker] audio playback unavailable ({e}); using live text-to-speech only")
        self._thread = threading.Thread(target=self._worker, name="speaker", daemon=True)
        self._thread.start()

    # -- public API ---------------------------------------------------------

    def own_clip(self, phrase_id):
        """The phrase's own-voice WAV if voice mode allows it and it exists."""
        path = self.audio_dir / f"{phrase_id}.wav"
        return path if self.mode == "cloned" and path.exists() else None

    @property
    def label(self):
        """How the output voice is described on screen (cached: files are read once)."""
        if getattr(self, "_label", None) is None:
            self._label = self._describe_voice()
        return self._label

    def _describe_voice(self):
        if self.mode != "cloned":
            return "computer voice (offline)"
        have = [pid for pid in self.phrases if (self.audio_dir / f"{pid}.wav").exists()]
        if not have:
            return "computer voice (no voice clips yet)"
        manifest = load_manifest(self.audio_dir)
        if all(manifest.get(pid) == "recorded" for pid in have):
            return "own recorded voice"
        return "cloned voice (AI-generated)"

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

    def _uncached(self):
        return [pid for pid in self.phrases
                if self.own_clip(pid) is None and not (tts_cache_dir() / f"{pid}.wav").exists()]

    def _worker(self):
        _init_com()
        todo = self._uncached() if self._warm and self._sd else []
        while True:
            try:
                kind, ids, t_ref = self._queue.get(timeout=0.3)
            except queue.Empty:
                if todo:  # idle: render one offline-voice clip so it is instant next time
                    pid = todo.pop(0)
                    try:
                        render_tts(self.phrases[pid]["text"], tts_cache_dir() / f"{pid}.wav")
                    except Exception as e:
                        print(f"[speaker] offline voice cache failed ({e}); speaking live instead")
                        todo = []
                continue
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
                    wav = self.own_clip(pid) or tts_cache_dir() / f"{pid}.wav"
                    if wav.exists() and self._sd:
                        data, sr = self._sf.read(str(wav), dtype="float32")
                        self._mark_start(t_ref, n)
                        self._sd.play(data, sr)
                        self._sd.wait()
                    else:
                        self._tts_live(self.phrases[pid]["text"], t_ref, n)
            except Exception as e:
                print(f"[speaker] playback error: {e}")
            finally:
                self._busy.clear()

    def _mark_start(self, t_ref, n):
        if t_ref is not None and n == 0:
            self.last_latency_ms = (time.perf_counter() - t_ref) * 1000

    def _tts_live(self, text, t_ref, n):
        try:
            import pyttsx3
        except ImportError:
            print(f"[speaker] (no TTS installed) would say: {text}")
            return
        engine = pyttsx3.init()   # fresh engine per utterance, see render_tts
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
