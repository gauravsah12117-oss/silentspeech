"""SilentSpeech live demo.

    python app.py                      # push-to-talk, cloned voice, LLM on
    python app.py --mode auto          # automatic start/stop detection
    python app.py --voice default      # offline pyttsx3 voice
    python app.py --no-llm             # fully offline mode
    python app.py --speaker s02        # another teammate's voice and model

Keys
    SPACE      capture a phrase (push-to-talk; also works in auto mode)
    1 / 2 / 3  pick an option when the system is unsure, or correct the last phrase
    ENTER      speak the whole sentence
    BACKSPACE  remove the last phrase          C  clear the sentence
    M          push-to-talk <-> auto mode      L  LLM layer on/off
    A          acknowledge an emergency alert  Q  quit
"""
import argparse
import sys
import time
from collections import deque

import numpy as np

import config
from src.decision import decide
from src.features import fill_missing, frame_motion, lip_motion, normalise, pad_to_length
from src.landmarks import N_POINTS, LipTracker, draw_lips, mouth_aspect_ratio
from src.llm_corrector import LLMCorrector
from src.recognizer import load_recognizer
from src.speaker import Speaker
from src.utils import FPSCounter, draw_text, load_env, load_phrases, open_camera, wrap_text

WINDOW = "SilentSpeech"
W, H = 1280, 640
VX, VY, VW, VH = 16, 70, 640, 480      # video area
PX = VX + VW + 24                      # right panel x
WHITE, GREY, DIM = (255, 255, 255), (185, 185, 185), (120, 120, 120)
GREEN, AMBER, RED = (90, 200, 90), (40, 170, 250), (60, 60, 230)
NAN_ROW = np.full((N_POINTS, 3), np.nan, np.float32)


class App:
    def __init__(self, args):
        import cv2
        self.cv2 = cv2
        load_env()
        self.args = args
        self.mode = args.mode
        self.phrases = load_phrases(phrase_set="full")

        self.runtime, why = load_recognizer(args.speaker, self.phrases, getattr(args, "recognizer", "auto"),
                                            args.model)
        if self.runtime is not None:
            print(f"Recogniser: {self.runtime.describe()} - {why}")
        else:
            print(f"Tracking-only mode: {why}")

        known = self.runtime.label_map if self.runtime else list(self.phrases)
        self.speaker = Speaker({pid: self.phrases[pid] for pid in known}, args.speaker, args.voice)
        print(f"Voice: {self.speaker.label}")
        if args.voice == "cloned":
            missing = self.speaker.missing_audio()
            if missing:
                print(f"No own-voice clip for {len(missing)} phrase(s) in {self.speaker.audio_dir}; "
                      "the offline computer voice speaks those (record_voice.py adds them).")
        self.brightness = None

        self.llm = LLMCorrector()      # cheap: threads start only on the first call
        self.llm_on = config.USE_LLM and not args.no_llm and self.llm.available
        if config.USE_LLM and not args.no_llm and not self.llm.available:
            print(f"LLM layer off: {self.llm.unavailable_reason}")

        # capture state
        self.capturing = False
        self.cap_kind = None
        self.cap_points = []
        self.cap_start = 0.0
        self.prebuffer = deque(maxlen=max(1, int(config.AUTO_PREROLL_SECONDS * 30)))
        self.active_run = 0
        self.still_since = None
        self.cooldown_until = 0.0
        self.prev_norm = None

        # conversation state
        self.decision = None
        self.options = None            # candidates waiting for 1/2/3
        self.pending = None            # LLM call in flight
        self.last_committed = None     # (phrase_id, candidates) for corrections
        self.note = "Press SPACE and mouth a phrase." if self.mode == "ptt" else "Auto mode: just mouth a phrase."
        self.alert = None
        self.sentence = []
        self.history = deque(maxlen=config.LLM_CONTEXT_PHRASES)
        self.fps = FPSCounter()

    # ------------------------------------------------------------------ text
    def text(self, pid):
        return self.phrases[pid]["text"]

    # --------------------------------------------------------------- capture
    def start_capture(self, kind, now, seed_rows=()):
        self.capturing, self.cap_kind, self.cap_start = True, kind, now
        self.cap_points = list(seed_rows)
        self.still_since = None
        self.options = None

    def finish_capture(self, now):
        self.capturing = False
        self.cooldown_until = now + config.AUTO_COOLDOWN_SECONDS
        duration = now - self.cap_start
        clip = np.stack(self.cap_points) if self.cap_points else np.zeros((0, N_POINTS, 3), np.float32)
        if self.cap_kind == "auto" and duration < config.AUTO_MIN_SECONDS:
            return
        if len(clip) < 5 or np.isnan(clip).any(axis=(1, 2)).mean() > 0.3:
            self.note = "Face not visible enough - please repeat."
            return
        if lip_motion(normalise(fill_missing(clip))) < config.MIN_LIP_MOTION:
            self.note = "No lip movement seen, so nothing was spoken. Please repeat."
            return
        if self.runtime is None:
            self.note = "No trained model yet - run collect_data.py and train.py."
            return
        if self.cap_kind == "auto":  # look like a fixed push-to-talk window again
            fps = len(clip) / max(duration, 1e-3)
            clip = pad_to_length(clip, int(round(config.RECORD_SECONDS * fps)))
        probs, _ = self.runtime.predict_clip(clip)
        if probs is None:
            self.note = "Face not visible enough - please repeat."
            return
        d = decide(probs, self.runtime.label_map, self.phrases)
        self.decision = d
        top = ", ".join(f"{pid} {p:.2f}" for pid, p in d.candidates)
        print(f"[{d.action}] {top}" + (f"  ALERT {d.alert_phrase_id}" if d.alert_phrase_id else ""))

        if d.alert_phrase_id:
            self.alert = d.alert_phrase_id
            self.speaker.alert()
        if d.action == "accept":
            self.commit(d.phrase_id, f"Spoke it (confidence {d.confidence:.0%}).", now, d.candidates)
        elif d.action == "llm":
            if self.llm_on:
                self.pending = {"future": self.llm.submit(d.candidates, list(self.history), self.phrases),
                                "deadline": now + config.LLM_TIMEOUT_S, "t_ref": now, "decision": d}
                self.options = d.candidates
                self.note = "Not fully sure - asking the LLM to use the conversation..."
            else:
                self.commit(d.candidates[0][0], f"Medium confidence ({d.confidence:.0%}) - spoke the top guess.",
                            now, d.candidates)
        else:
            self.options = d.candidates
            self.note = "Not sure, so nothing was spoken. Please repeat, or press 1/2/3."

    def commit(self, pid, note, t_ref, candidates=None):
        self.sentence.append(pid)
        self.history.append(pid)
        self.options = None
        self.pending = None
        self.note = note
        self.last_committed = (pid, candidates or [])
        self.speaker.say([pid], t_ref)

    def resolve_llm(self, now):
        p = self.pending
        if not p or not (p["future"].done() or now >= p["deadline"]):
            return
        result = None
        if p["future"].done():
            try:
                result = p["future"].result()
            except Exception:
                result = None
        d = p["decision"]
        if result:
            pid, reason = result
            note = f"LLM picked from the top-3 using context: {reason}" if reason else "LLM picked from the top-3."
        else:
            pid, note = d.candidates[0][0], "LLM too slow or unavailable - spoke the top guess."
        self.commit(pid, note, p["t_ref"], d.candidates)

    # ----------------------------------------------------------------- input
    def pick(self, k, now):
        if self.options and k <= len(self.options):
            pid = self.options[k - 1][0]
            cands = self.options
            self.pending = None
            self.commit(pid, f"You picked option {k}.", now, cands)
        elif self.last_committed and self.sentence and self.sentence[-1] == self.last_committed[0]:
            cands = self.last_committed[1]
            if k <= len(cands) and cands[k - 1][0] != self.sentence[-1]:
                pid = cands[k - 1][0]
                self.sentence[-1] = pid
                if self.history:
                    self.history[-1] = pid
                self.last_committed = (pid, cands)
                self.note = f"Corrected to option {k}."
                self.speaker.say([pid], now)

    def handle_key(self, key, now):
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key == 32 and not self.capturing and not self.pending:
            self.start_capture("ptt", now)
        elif key in (ord("1"), ord("2"), ord("3")):
            self.pick(key - ord("0"), now)
        elif key in (13, 10):
            if self.sentence:
                self.speaker.say(self.sentence, now)
                self.note = "Speaking the whole sentence."
        elif key in (8, 127):
            if self.sentence:
                self.sentence.pop()
                self.last_committed = None
        elif key in (ord("c"), ord("C")):
            self.sentence.clear()
            self.options = None
            self.last_committed = None
            self.note = "Sentence cleared."
        elif key in (ord("m"), ord("M")):
            self.mode = "auto" if self.mode == "ptt" else "ptt"
            self.active_run = 0
            self.note = "Auto mode: just mouth a phrase." if self.mode == "auto" else "Push-to-talk: press SPACE."
        elif key in (ord("l"), ord("L")):
            if self.llm_on:
                self.llm_on = False
            elif self.llm and self.llm.available:
                self.llm_on = True
            else:
                self.note = "LLM unavailable: " + self.llm.unavailable_reason
        elif key in (ord("a"), ord("A")):
            self.alert = None
        return True

    # --------------------------------------------------------- per-frame step
    def step(self, pts, now):
        row = pts if pts is not None else NAN_ROW
        motion, mar = 0.0, 0.0
        if pts is not None:
            norm = normalise(pts[None])[0]
            motion = frame_motion(self.prev_norm, norm) if self.prev_norm is not None else 0.0
            mar = mouth_aspect_ratio(pts)
            self.prev_norm = norm
        else:
            self.prev_norm = None

        if self.capturing:
            self.cap_points.append(row)
            elapsed = now - self.cap_start
            if self.cap_kind == "ptt":
                if elapsed >= config.RECORD_SECONDS:
                    self.finish_capture(now)
            else:
                if pts is not None and motion > config.AUTO_MOTION_THRESHOLD:
                    self.still_since = None
                elif self.still_since is None:
                    self.still_since = now
                if (self.still_since and now - self.still_since >= config.AUTO_STILL_SECONDS) \
                        or elapsed >= config.AUTO_MAX_SECONDS:
                    self.finish_capture(now)
        elif self.mode == "auto" and not self.pending and now >= self.cooldown_until:
            self.prebuffer.append(row)
            opening = mar > config.AUTO_MAR_THRESHOLD and motion > config.AUTO_MOTION_THRESHOLD / 2
            active = pts is not None and (motion > config.AUTO_MOTION_THRESHOLD or opening)
            self.active_run = self.active_run + 1 if active else 0
            if self.active_run >= config.AUTO_START_FRAMES:
                self.active_run = 0
                self.start_capture("auto", now, self.prebuffer)
                self.prebuffer.clear()
        self.resolve_llm(now)
        return mar, motion

    # ---------------------------------------------------------------- render
    def render(self, frame, pts, mar, motion, now):
        cv2 = self.cv2
        canvas = np.full((H, W, 3), (28, 24, 22), np.uint8)

        # banner
        if self.alert:
            cv2.rectangle(canvas, (0, 0), (W, 56), (0, 0, 210) if int(now * 2) % 2 == 0 else (0, 0, 140), -1)
            draw_text(canvas, f'EMERGENCY: "{self.text(self.alert)}" - caregiver alerted. Press A to acknowledge.',
                      (16, 37), 0.75, WHITE, 2)
        else:
            draw_text(canvas, "SilentSpeech", (16, 38), 0.95, WHITE, 2)
            draw_text(canvas, "silent lips -> phrase -> their own voice", (230, 37), 0.55, GREY, 1)
            draw_text(canvas, f"Output: {self.speaker.label}", (W - 400, 37), 0.5, GREY, 1)

        # video
        view = frame.copy()
        draw_lips(view, pts)
        view = cv2.flip(view, 1)
        if pts is None:
            draw_text(view, "No face detected - face the camera, light from the front", (12, 30), 0.6, RED, 2)
        if self.brightness is not None and self.brightness < 50:
            draw_text(view, "Too dark: put a light in front of you", (12, 58), 0.6, AMBER, 2)
        if self.capturing:
            cv2.circle(view, (VW - 90, 28), 11, (0, 0, 255), -1, cv2.LINE_AA)
            draw_text(view, "REC", (VW - 72, 36), 0.7, (0, 0, 255), 2)
            frac = (now - self.cap_start) / (config.RECORD_SECONDS if self.cap_kind == "ptt" else config.AUTO_MAX_SECONDS)
            cv2.rectangle(view, (0, VH - 8), (int(VW * min(frac, 1)), VH), (0, 0, 255), -1)
        elif self.mode == "auto":
            draw_text(view, "AUTO - listening for lip movement", (12, VH - 16), 0.55, AMBER, 1)
        canvas[VY:VY + VH, VX:VX + VW] = view

        # prediction panel
        y = VY + 24
        draw_text(canvas, "PREDICTION", (PX, y), 0.5, DIM, 1)
        y += 40
        d = self.decision
        if d:
            if self.pending:
                headline = self.text(d.candidates[0][0]) + " ?"
            elif self.options:
                headline = "Not sure - pick 1, 2 or 3"
            elif self.last_committed and self.sentence:
                headline = self.text(self.sentence[-1])
            else:
                headline = self.text(d.candidates[0][0]) + " ?"
            draw_text(canvas, headline, (PX, y), 1.0, WHITE, 2)
            y += 30
            for k, (pid, p) in enumerate(d.candidates, 1):
                col = GREEN if p >= config.CONF_ACCEPT else AMBER if p >= config.CONF_MIN else RED
                draw_text(canvas, f"[{k}] {self.text(pid)}", (PX, y + 16), 0.55, WHITE, 1)
                bx, bw = PX + 330, 180
                cv2.rectangle(canvas, (bx, y + 4), (bx + bw, y + 20), (60, 60, 60), -1)
                cv2.rectangle(canvas, (bx, y + 4), (bx + int(bw * p), y + 20), col, -1)
                draw_text(canvas, f"{p:.0%}", (bx + bw + 8, y + 18), 0.5, GREY, 1)
                y += 30
        else:
            draw_text(canvas, "-", (PX, y), 1.0, DIM, 2)
            y += 30
        y += 10
        for line in wrap_text(self.note, 60)[:3]:
            draw_text(canvas, line, (PX, y), 0.52, AMBER if self.pending or self.options else GREY, 1)
            y += 22

        # sentence
        y = max(y + 18, VY + 290)
        draw_text(canvas, "SENTENCE", (PX, y), 0.5, DIM, 1)
        y += 32
        sentence = ". ".join(self.text(pid) for pid in self.sentence) + ("." if self.sentence else "")
        for line in (wrap_text(sentence, 42) or ["(empty)"])[-4:]:
            draw_text(canvas, line, (PX, y), 0.75, WHITE if self.sentence else DIM, 2 if self.sentence else 1)
            y += 32

        draw_text(canvas, "SPACE capture  1/2/3 pick  ENTER speak  BKSP undo  C clear",
                  (PX, VY + VH - 26), 0.45, DIM, 1)
        draw_text(canvas, "M mode  L LLM  A acknowledge alert  Q quit", (PX, VY + VH - 6), 0.45, DIM, 1)

        # status bar
        cv2.rectangle(canvas, (0, H - 36), (W, H), (48, 44, 42), -1)
        llm = f"ON ({self.llm.provider})" if self.llm_on else "OFF"
        lat = self.speaker.last_latency_ms
        rec = "-" if self.runtime is None else self.runtime.kind
        status = (f"Mode: {'PUSH-TO-TALK' if self.mode == 'ptt' else 'AUTO'}   FPS {self.fps.fps:4.1f}   "
                  f"Recogniser: {rec}   LLM: {llm}   Speaker: {self.args.speaker}   "
                  f"Latency: {'-' if lat is None else f'{lat:.0f} ms'}   MAR {mar:.2f}  motion {motion:.3f}")
        draw_text(canvas, status, (16, H - 12), 0.5, GREY, 1)
        return canvas

    # ------------------------------------------------------------------ main
    def run(self):
        cv2 = self.cv2
        cap = open_camera(self.args.camera)
        if cap is None:
            sys.exit("Could not open the webcam. Try --camera 1, close Zoom/Teams, or run python check_setup.py.")
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, W, H)
        failures = 0
        try:
            with LipTracker() as tracker:
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        failures += 1
                        if failures > 60:
                            sys.exit("The webcam stopped sending frames.")
                        continue
                    failures = 0
                    if frame.shape[:2] != (VH, VW):
                        frame = cv2.resize(frame, (VW, VH))
                    self.fps.tick()
                    self.brightness = float(frame[::8, ::8].mean())   # every 8th pixel: microseconds
                    now = time.perf_counter()
                    pts = tracker.process(frame)
                    mar, motion = self.step(pts, now)
                    cv2.imshow(WINDOW, self.render(frame, pts, mar, motion, now))
                    key = cv2.waitKey(1) & 0xFF
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key != 255 and not self.handle_key(key, now):
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.speaker.close(timeout=1.0)
            self.llm.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["ptt", "auto"], default="ptt")
    ap.add_argument("--voice", choices=["cloned", "default"], default=config.VOICE_MODE)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--speaker", default=config.SPEAKER_ID)
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--model", help="use this model file instead of the speaker's or the base model")
    ap.add_argument("--recognizer", choices=["auto", "neural", "templates"], default="auto",
                    help="auto: enrolled/trained neural model if it knows this speaker, else calibration templates")
    App(ap.parse_args(argv)).run()


if __name__ == "__main__":
    main()
