"""Record each phrase in your own voice (voice banking, no GPU needed).

    python record_voice.py --speaker s01                # one clip per phrase
    python record_voice.py --speaker s01 --phrases P01,P28
    python record_voice.py --speaker s01 --redo         # record every phrase again
    python record_voice.py --speaker s01 --sample       # a 45-60 s sample for voice cloning

Press SPACE, then say the phrase naturally. Recording stops by itself a moment
after you finish (or press SPACE again). Each clip is trimmed, levelled and
played back, then saved to audio/phrase_audio/<speaker>/<phrase_id>.wav - the
same place the cloned-voice notebook writes to - so app.py plays it at once.

Keys: SPACE record / stop   P play again   R redo   N skip   Q quit

For someone who has already lost their voice, use the cloning notebook with
old voice notes instead (README section 18). --sample records a clean sample
for that notebook from someone who can still speak.
"""
import argparse
import sys
import time

import numpy as np

import config
from src.speaker import mark_recorded, trim_and_level, voice_dir
from src.utils import draw_text, load_phrases, make_window, wrap_text

WINDOW = "SilentSpeech - record your voice"
W, H = 800, 330
MAX_PHRASE_S = 5.0
MAX_SAMPLE_S = 90.0
SPEECH_OVER_NOISE = 2.5    # speech must be ~8 dB above the room noise (works in noisy rooms too)
SAMPLE_TEXT = (
    "Every morning I open the window and listen to the street waking up. A scooter rattles past, "
    "a dog barks twice, and somebody's radio plays an old song about the sea. I make a cup of tea, "
    "check my phone, and think about the day ahead. Some days are quiet and slow; others are busy, "
    "loud and full of surprises. Either way, I like to begin calmly, with a deep breath and a smile. "
    "Thank you for helping me keep my voice, so the people I love can still hear me."
)


class VoiceRecorder:
    def __init__(self):
        import cv2
        import sounddevice as sd
        import soundfile as sf
        self.cv2, self.sd, self.sf = cv2, sd, sf
        try:
            dev = sd.query_devices(kind="input")
        except Exception as e:
            sys.exit(f"No microphone found ({e}). Plug one in or check Windows sound settings.")
        self.mic_name = dev["name"]
        self.sr = int(dev["default_samplerate"]) or 44100
        self.noise = 0.0
        make_window(WINDOW, W, H)

    # -- screen -------------------------------------------------------------
    def show(self, title, lines, level=None, color=(255, 255, 255), progress=None):
        cv2 = self.cv2
        img = np.full((H, W, 3), (34, 30, 28), np.uint8)
        y = 52
        for line in wrap_text(title, 34)[:3]:
            draw_text(img, line, (24, y), 1.05, color, 2)
            y += 44
        y = max(y + 6, 170)
        for line in lines:
            draw_text(img, line, (24, y), 0.55, (200, 200, 200), 1)
            y += 26
        if level is not None:
            cv2.rectangle(img, (24, H - 46), (W - 24, H - 26), (70, 70, 70), -1)
            fill = int((W - 48) * min(1.0, level * 6))
            cv2.rectangle(img, (24, H - 46), (24 + fill, H - 26), (60, 60, 230), -1)
        if progress is not None:
            cv2.rectangle(img, (0, H - 6), (int(W * min(progress, 1.0)), H), (60, 60, 230), -1)
        draw_text(img, f"Microphone: {self.mic_name}", (24, H - 8), 0.42, (130, 130, 130), 1)
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(15) & 0xFF
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return "q"
        if key == 255:
            return None
        return " " if key == 32 else "q" if key == 27 else chr(key).lower()

    # -- audio --------------------------------------------------------------
    def measure_noise(self):
        chunks = []
        with self.sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                 callback=lambda d, *_: chunks.append(d[:, 0].copy())):
            end = time.time() + 1.2
            while time.time() < end:
                self.show("Measuring room noise...", ["Please stay quiet for a second."])
        audio = np.concatenate(chunks) if chunks else np.zeros(1)
        block = int(self.sr * 0.02)
        n = max(1, len(audio) // block)
        rms = np.sqrt((audio[: n * block].reshape(n, -1) ** 2).mean(1)) if len(audio) >= block else np.zeros(1)
        self.noise = float(np.median(rms))

    def record(self, title, lines, max_s, auto_stop=True):
        """Record until speech ends (auto_stop), SPACE, or max_s. None if Q pressed."""
        chunks, level = [], [0.0]

        def callback(data, *_):
            chunks.append(data[:, 0].copy())
            level[0] = float(np.sqrt((data[:, 0] ** 2).mean()))

        threshold = max(SPEECH_OVER_NOISE * self.noise, 0.01)
        heard_at, loud_run = None, 0
        with self.sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                 blocksize=int(self.sr * 0.03), callback=callback):
            start = time.time()
            while True:
                el = time.time() - start
                # Speech = sustained sound after the key click has died away.
                loud_run = loud_run + 1 if level[0] > threshold else 0
                if el > 0.25 and (loud_run >= 3 or (heard_at is not None and loud_run)):
                    heard_at = el
                key = self.show(title, lines, level=level[0], color=(80, 80, 255), progress=el / max_s)
                if key == "q":
                    return None
                if key == " " and el > 0.4:
                    break
                if auto_stop and heard_at is not None and el - heard_at > 0.8:
                    break
                if el >= max_s:
                    break
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def play(self, audio):
        self.sd.play(audio, self.sr)
        self.sd.wait()

    def close(self):
        self.cv2.destroyWindow(WINDOW)


def record_phrase_voices(speaker, phrases, redo=False):
    """Record a clip for each phrase. Returns the number saved."""
    out = voice_dir(speaker)
    out.mkdir(parents=True, exist_ok=True)
    rec = VoiceRecorder()
    saved = 0
    try:
        rec.measure_noise()
        items = [p for p in phrases.values() if redo or not (out / f"{p['id']}.wav").exists()]
        i = 0
        while i < len(items):
            p = items[i]
            head = f"Say: {p['text']}"
            info = [f"Phrase {i + 1} of {len(items)}. Press SPACE, then say it in your normal voice.",
                    "SPACE record   N skip   Q quit"]
            key = None
            while key not in (" ", "n", "q"):
                key = rec.show(head, info)
            if key == "q":
                break
            if key == "n":
                i += 1
                continue
            audio = rec.record(head, ["Recording... speak now. It stops by itself when you finish."], MAX_PHRASE_S)
            if audio is None:
                break
            clip = trim_and_level(audio, rec.sr, rec.noise, over_noise=SPEECH_OVER_NOISE)
            if clip is None:
                end, quit_now = time.time() + 2.0, False
                while time.time() < end and not quit_now:
                    quit_now = rec.show("Didn't hear anything", [
                        "Check the microphone is on and speak a little louder.", "Q quit"]) == "q"
                if quit_now:
                    break
                continue
            rec.show(head, ["Playing it back..."])
            rec.play(clip)
            choice = None
            while choice not in (" ", "r", "p", "q", "n"):
                choice = rec.show(head, ["Keep it?  SPACE yes, next phrase   R record again   P play again",
                                         "Q quit"])
                if choice == "p":
                    rec.play(clip)
                    choice = None
            if choice == "q":
                break
            if choice == "r":
                continue
            rec.sf.write(str(out / f"{p['id']}.wav"), clip, rec.sr)
            mark_recorded(out, p["id"])
            saved += 1
            i += 1
    finally:
        rec.close()
    return saved


def record_sample(speaker):
    out = config.VOICE_SAMPLE_DIR / f"{speaker}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = VoiceRecorder()
    try:
        rec.measure_noise()
        lines = wrap_text(SAMPLE_TEXT, 92)
        key = None
        while key not in (" ", "q"):
            key = rec.show("Read this aloud, calmly", lines + ["", "SPACE start (then SPACE again to stop)   Q quit"])
        if key == "q":
            return False
        audio = rec.record("Recording... read the text", lines + ["", "Press SPACE when you finish."],
                           MAX_SAMPLE_S, auto_stop=False)
        if audio is None:
            return False
        clip = trim_and_level(audio, rec.sr, rec.noise, margin_s=0.3, over_noise=SPEECH_OVER_NOISE)
        if clip is None or len(clip) < rec.sr * 10:
            print("The sample is too short or silent; aim for 30-60 seconds of clear speech.")
            return False
        rec.sf.write(str(out), clip, rec.sr)
        print(f"Saved {out} ({len(clip) / rec.sr:.0f} s). Upload it to the cloning notebook.")
        return True
    finally:
        rec.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", default=config.SPEAKER_ID)
    ap.add_argument("--set", dest="phrase_set", default="full", choices=["mvp", "full"])
    ap.add_argument("--phrases", help="comma-separated subset, e.g. P01,P28")
    ap.add_argument("--redo", action="store_true", help="record phrases that already have a clip too")
    ap.add_argument("--sample", action="store_true", help="record a voice sample for the cloning notebook")
    args = ap.parse_args()

    if args.sample:
        sys.exit(0 if record_sample(args.speaker) else 1)
    phrases = load_phrases(phrase_set=args.phrase_set)
    if args.phrases:
        wanted = [s.strip().upper() for s in args.phrases.split(",")]
        missing = [w for w in wanted if w not in phrases]
        if missing:
            sys.exit(f"Unknown phrase IDs: {', '.join(missing)}")
        phrases = {pid: phrases[pid] for pid in wanted}
    n = record_phrase_voices(args.speaker, phrases, redo=args.redo)
    print(f"Saved {n} clip(s) in {voice_dir(args.speaker)}. Run: python app.py --speaker {args.speaker}")


if __name__ == "__main__":
    main()
