"""Audio engine: one always-open output stream that mixes speech, sound cues and alarms.

Why a mixer instead of playing files one after another:
  - a cue or an alarm can sound *over* speech instead of waiting for it;
  - the output device is opened once, not for every phrase, so sound starts sooner;
  - one master volume, a soft limiter (nothing clips, even at 200%), and a live
    output level for the on-screen meter.

Speech is levelled to the same loudness (SPEECH_LEVEL_DB) so the own-voice clips
and the computer voice sound equally loud. Measured on this laptop's clips: own
voice -20.6 dBFS, computer voice -17.6 dBFS before levelling.

The sound cues (earcons) are synthesised here: short bell-like tones, no audio
files needed. Each one means one thing, so they can be learned quickly.
"""
import threading

import numpy as np

SAMPLE_RATE = 44100
SPEECH_LEVEL_DB = -16.0          # speech loudness after levelling (RMS of the voiced parts, dBFS)
LIMIT_START = 0.7                # the soft limiter starts bending the waveform here (about -3 dBFS)
VOLUME_RANGE = (0.3, 2.0)


# --- signal helpers --------------------------------------------------------------

def resample(data, sr_from, sr_to):
    """Linear resampling (plenty for speech and cues)."""
    if sr_from == sr_to or len(data) < 2:
        return np.asarray(data, np.float32)
    n = int(round(len(data) * sr_to / sr_from))
    x = np.linspace(0, len(data) - 1, n)
    return np.interp(x, np.arange(len(data)), data).astype(np.float32)


def speech_level_db(data, sr):
    """Loudness of the voiced part of a clip (20 ms blocks above 10% of the loudest), dBFS."""
    data = np.asarray(data, np.float32)
    block = max(1, int(sr * 0.02))
    n = len(data) // block
    if n == 0:
        return -120.0
    rms = np.sqrt((data[:n * block].reshape(n, block) ** 2).mean(1))
    voiced = rms[rms > 0.1 * rms.max()] if rms.max() > 0 else rms
    return float(20 * np.log10(np.sqrt(np.mean(voiced ** 2)) + 1e-9))


def level_speech(data, sr, target_db=SPEECH_LEVEL_DB):
    """Scale a speech clip to the target loudness (peaks are handled by the limiter)."""
    data = np.asarray(data, np.float32)
    if data.ndim > 1:
        data = data.mean(1)
    gain = 10 ** ((target_db - speech_level_db(data, sr)) / 20)
    return (data * min(gain, 8.0)).astype(np.float32)


def soft_limit(x, start=LIMIT_START):
    """Leave quiet audio alone; bend peaks smoothly so the output never exceeds 1.0."""
    a = np.abs(x)
    over = a > start
    if over.any():
        x = x.copy()
        x[over] = np.sign(x[over]) * (start + (1 - start) * np.tanh((a[over] - start) / (1 - start)))
    return x


# --- sound cues --------------------------------------------------------------------

def _note(freq, dur, sr, amp=0.35, decay=7.0, partials=((1, 1.0), (2, 0.3), (3, 0.12), (4.2, 0.05))):
    """A soft bell / marimba-like note: quick attack, natural decay, no click at the end."""
    t = np.arange(int(sr * dur)) / sr
    env = np.minimum(1.0, t / 0.004) * np.exp(-decay * t)
    tail = min(len(t), int(sr * 0.01))
    if tail > 1:
        env[-tail:] *= np.linspace(1, 0, tail)
    wave = sum(a * np.sin(2 * np.pi * freq * h * t) for h, a in partials)
    return (amp * env * wave / sum(a for _, a in partials)).astype(np.float32)


def _sequence(notes, sr):
    """notes: [(start s, freq, dur s, amp)] -> one mixed waveform."""
    end = max(start + dur for start, _, dur, _ in notes)
    out = np.zeros(int(sr * end) + 1, np.float32)
    for start, freq, dur, amp in notes:
        n = _note(freq, dur, sr, amp)
        i = int(sr * start)
        out[i:i + len(n)] += n
    return out


C5, D5, E5, G5, A4, E4, A5, C6, E6, G6, D6 = 523.3, 587.3, 659.3, 784.0, 440.0, 329.6, 880.0, 1046.5, 1318.5, 1568.0, 1174.7
CUES = {
    "ready":   [(0.0, C5, 0.5, 0.30), (0.09, E5, 0.5, 0.30), (0.18, G5, 0.7, 0.32)],   # app is ready
    "listen":  [(0.0, E5, 0.25, 0.34), (0.07, A5, 0.35, 0.36)],                         # listening starts
    "done":    [(0.0, A4, 0.16, 0.30)],                                                  # listening window ended
    "unsure":  [(0.0, A4, 0.3, 0.28), (0.11, E4, 0.4, 0.28)],                           # not sure, try again
    "learned": [(0.0, C6, 0.25, 0.16), (0.05, E6, 0.25, 0.16), (0.10, G6, 0.4, 0.18)],  # new example kept
    "on":      [(0.0, G5, 0.18, 0.26)],
    "off":     [(0.0, D5, 0.18, 0.24)],
    "volume":  [(0.0, D6, 0.14, 0.30)],
}


def cue(name, sr=SAMPLE_RATE):
    return _sequence(CUES[name], sr)


def alarm(kind="emergency", sr=SAMPLE_RATE):
    """Emergency: a fast high two-tone, 3 times. Pain: slower falling three-note, twice (easy to tell apart)."""
    if kind == "pain":
        notes = [(0.62 * k + 0.2 * i, f, 0.35, 0.42) for k in range(2) for i, f in enumerate((A5, E5, A4))]
    else:
        notes = [(0.48 * k + 0.24 * i, f, 0.22, 0.5) for k in range(3) for i, f in enumerate((C6, G5))]
    return _sequence(notes, sr)


# --- mixer --------------------------------------------------------------------------

class Mixer:
    """Keeps one output stream open and mixes everything played into it."""

    def __init__(self, sr=SAMPLE_RATE, volume=1.0):
        import sounddevice as sd
        self.sr = sr
        self.volume = float(np.clip(volume, *VOLUME_RANGE))
        self.level = 0.0                     # smoothed output level, for the speaking meter
        self._lock = threading.Lock()
        self._voices = []                    # [samples, position, done event, gain]
        self._stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32", callback=self._callback)
        self._stream.start()

    def _callback(self, outdata, frames, time_info, status):
        out = np.zeros(frames, np.float32)
        with self._lock:
            keep = []
            for v in self._voices:
                data, pos = v[0], v[1]
                n = min(frames, len(data) - pos)
                if n > 0:
                    out[:n] += data[pos:pos + n] * v[3]
                v[1] = pos + max(n, 0)
                if v[1] >= len(data):
                    v[2].set()
                else:
                    keep.append(v)
            self._voices = keep
        out = soft_limit(out * self.volume)
        outdata[:, 0] = out
        rms = float(np.sqrt(np.mean(out ** 2))) if frames else 0.0
        self.level = max(rms, 0.85 * self.level)

    def play(self, data, sr=None, gain=1.0, wait=False):
        """Mix a sound in; with wait=True return when it has finished playing."""
        data = resample(np.asarray(data, np.float32).reshape(-1), sr or self.sr, self.sr)
        done = threading.Event()
        with self._lock:
            self._voices.append([data, 0, done, float(gain)])
        if wait:
            done.wait(len(data) / self.sr + 2.0)
        return done

    @property
    def playing(self):
        return bool(self._voices)

    def stop(self):
        with self._lock:
            for v in self._voices:
                v[2].set()
            self._voices = []

    def close(self):
        self.stop()
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
