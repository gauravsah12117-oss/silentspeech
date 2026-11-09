"""SilentSpeech live demo.

    python app.py                      # push-to-talk, cloned voice, LLM on
    python app.py --mode auto          # automatic start/stop detection
    python app.py --voice default      # offline pyttsx3 voice
    python app.py --no-llm             # fully offline mode
    python app.py --speaker s02        # another teammate's voice and model

Keys
    SPACE      capture a phrase (push-to-talk; also works in auto mode)
    1 / 2 / 3  pick an option when the system is unsure, or correct / confirm the last phrase
               (with instant templates, every choice is kept as a new example: it learns as you use it)
    ENTER      speak the whole sentence
    BACKSPACE  remove the last phrase          C  clear the sentence
    M          push-to-talk <-> auto mode      L  LLM layer on/off
    A          acknowledge an emergency alert  Q  quit
    F          focus: switch to your most reliable phrases (higher accuracy) and back
    V          view: face-mesh hologram <-> plain camera
"""
import argparse
import sys
import threading
import time
from collections import deque

import numpy as np

import config
from src.decision import combine_attempts, decide, top_k
from src.features import fill_missing, frame_motion, lip_motion, normalise, pad_to_length
from src.dataset import blendshape_path
from src.landmarks import (BLENDSHAPE_NAMES, N_POINTS, draw_face_hud, draw_lips, make_tracker,
                           mouth_aspect_ratio)
from src.llm_corrector import LLMCorrector
from src.recognizer import load_recognizer
from src.speaker import Speaker
from src import ui
from src.ui import PanelCache
from src.utils import (FPSCounter, draw_text, load_env, load_phrases, make_window, open_camera,
                       start_session_log)

WINDOW = "SilentSpeech"
W, H = 1280, 720
VX, VY, VW, VH = 16, 72, 640, 480      # video area
# panels: (x0, y0, x1, y1)
HEADER = (0, 0, W, 62)
BOARD = (16, 562, 656, 690)
PRED = (672, 72, 1264, 292)
WIDGETS = (672, 300, 1264, 456)
SENT = (672, 464, 1264, 560)
HIST = (672, 568, 1264, 690)
STATUS = (0, 694, W, H)
WHITE, GREY = (255, 255, 255), (185, 185, 185)
GREEN, AMBER, RED = (90, 200, 90), (40, 170, 250), (60, 60, 230)
NAN_ROW = np.full((N_POINTS, 3), np.nan, np.float32)
SOURCE_LABEL = {"lips": "lips", "llm": "lips + context", "pick": "picked", "corrected": "corrected",
                "tap": "tapped", "fallback": "lips (top guess)"}


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
        self.retry = None               # evidence from earlier "not sure" tries (they add up)
        self.options = None            # candidates waiting for 1/2/3
        self.pending = None            # LLM call in flight
        self.last_committed = None     # (phrase_id, candidates) for corrections
        self.note = "Press SPACE and mouth a phrase." if self.mode == "ptt" else "Auto mode: just mouth a phrase."
        self.alert = None
        self.sentence = []
        self.history = deque(maxlen=config.LLM_CONTEXT_PHRASES)
        self.fps = FPSCounter()

        # face-mesh view (V)
        self.hud = config.HUD_VIEW
        self.mesh = None
        self.has_blend = False
        self.last_blend_live = None
        self.tracker_name = ""
        self.cap_blend = []

        # focus mode (F): only the speaker's most reliable phrases
        self.full_runtime = self.runtime
        self.focus = None
        self._ranking = None

        # learning from the user's 1/2/3 choices
        self.last_clip = None
        self.last_blend = None
        self.learned_path = None
        self._learn_lock = threading.Lock()
        self._learn_again = False
        self._learn_thread = None

        # interface: cached panels, mouse clicks, conversation log, alert repeats
        self.ui = PanelCache()
        self.clicks = []
        self.tiles = []                 # [(rect, phrase_id)] on the phrase board
        self.log = deque(maxlen=60)     # {"time", "pid", "source", "confidence"}
        self.alert_last_beep = 0.0
        self.muscles = None             # smoothed muscle-activity values for the bars
        self.transcript = config.WORK_DIR / "logs" / f"conversation-{time.strftime('%Y-%m-%d')}.txt"

    # ------------------------------------------------------------------ text
    def text(self, pid):
        return self.phrases[pid]["text"]

    # --------------------------------------------------------------- capture
    def start_capture(self, kind, now, seed=()):
        self.capturing, self.cap_kind, self.cap_start = True, kind, now
        self.cap_points = [row for row, _ in seed]
        self.cap_blend = [b for _, b in seed]
        self.still_since = None
        self.options = None

    def finish_capture(self, now):
        self.capturing = False
        self.cooldown_until = now + config.AUTO_COOLDOWN_SECONDS
        duration = now - self.cap_start
        clip = np.stack(self.cap_points) if self.cap_points else np.zeros((0, N_POINTS, 3), np.float32)
        blend = None
        if any(b is not None for b in self.cap_blend):          # the 52 muscle scores, if the tracker has them
            nan_bs = np.full(len(BLENDSHAPE_NAMES), np.nan, np.float32)
            blend = np.stack([b if b is not None else nan_bs for b in self.cap_blend]).astype(np.float32)
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
        print(f"[capture] {len(clip)} frames in {duration:.1f} s ({len(clip) / max(duration, 1e-3):.0f} FPS), "
              f"lip motion {lip_motion(normalise(fill_missing(clip))):.2f}")
        if self.cap_kind == "auto":  # look like a fixed push-to-talk window again
            fps = len(clip) / max(duration, 1e-3)
            clip = pad_to_length(clip, int(round(config.RECORD_SECONDS * fps)))
            if blend is not None:
                blend = pad_to_length(blend, len(clip))
        probs, _ = self.runtime.predict_clip(clip, blend)
        if probs is None:
            self.note = "Face not visible enough - please repeat."
            return
        self.last_clip, self.last_blend, self.learned_path = clip, blend, None
        # "Not sure - please repeat": a quick repeat counts together with the earlier try (or tries)
        tries, labels = 1, list(self.runtime.label_map)
        r = self.retry
        if r and now - r["time"] <= config.RETRY_COMBINE_SECONDS and r["labels"] == labels:
            single = top_k(probs, labels, 1)[0]
            probs, tries = combine_attempts(r["probs"], probs), r["tries"] + 1
            print(f"[combine] try {tries}: this try alone {single[0]} {single[1]:.2f}; "
                  f"together {top_k(probs, labels, 1)[0][0]} {probs.max():.2f}")
        d = decide(probs, labels, self.phrases)
        self.retry = {"probs": probs, "tries": tries, "time": now, "labels": labels} if d.action == "repeat" else None
        self.decision = d
        top = ", ".join(f"{pid} {p:.2f}" for pid, p in d.candidates)
        print(f"[{d.action}] {top}" + (f"  ALERT {d.alert_phrase_id}" if d.alert_phrase_id else ""))
        together = f" (your {tries} tries together)" if tries > 1 else ""

        if d.alert_phrase_id:
            self.alert = d.alert_phrase_id
            self.alert_last_beep = now
            self.speaker.alert()
            self.write_transcript(f"ALERT raised: {self.text(d.alert_phrase_id)}")
        if d.action == "accept":
            self.commit(d.phrase_id, f"Spoke it (confidence {d.confidence:.0%}{together}).", now, d.candidates,
                        source="lips", confidence=d.confidence)
        elif d.action == "llm":
            if self.llm_on:
                self.pending = {"future": self.llm.submit(d.candidates, list(self.history), self.phrases),
                                "deadline": now + config.LLM_TIMEOUT_S, "t_ref": now, "decision": d}
                self.options = d.candidates
                self.note = f"Not fully sure{together} - asking {self.llm.provider.title()} to use the conversation..."
            else:
                self.commit(d.candidates[0][0], f"Medium confidence ({d.confidence:.0%}{together}) - spoke the "
                            "top guess.", now, d.candidates, source="fallback", confidence=d.confidence)
        else:
            self.options = d.candidates
            self.note = ("Not sure, so nothing was spoken. Mouth it again (your tries add up), "
                         "or press / click 1, 2 or 3.")

    def commit(self, pid, note, t_ref, candidates=None, source="lips", confidence=None):
        self.sentence.append(pid)
        self.history.append(pid)
        self.options = None
        self.pending = None
        self.retry = None
        self.note = note
        self.last_committed = (pid, candidates or [])
        self.speaker.say([pid], t_ref)
        self.log.append({"time": time.localtime(), "pid": pid, "source": source, "confidence": confidence})
        conf = f" {confidence:.0%}" if confidence is not None else ""
        self.write_transcript(f"{self.text(pid)}  [{SOURCE_LABEL.get(source, source)}{conf}]")

    def write_transcript(self, line):
        """Keep a plain-text record of the conversation for caregivers (logs/conversation-<date>.txt)."""
        if not config.SAVE_TRANSCRIPT:
            return
        try:
            self.transcript.parent.mkdir(parents=True, exist_ok=True)
            with open(self.transcript, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')}  {line}\n")
        except OSError:
            pass

    def tap(self, pid, now):
        """Speak a phrase chosen on the phrase board (a fallback that needs no lip reading)."""
        self.pending = None
        self.commit(pid, f"You tapped \"{self.text(pid)}\".", now, source="tap")

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
        name = self.llm.provider.title()
        if result:
            pid, reason = result
            note = f"{name} picked from the top 3 using the conversation: {reason}" if reason \
                else f"{name} picked from the top 3."
            source = "llm"
        else:
            pid, note, source = d.candidates[0][0], f"{name} too slow or unavailable - spoke the top guess.", "fallback"
        conf = dict(d.candidates).get(pid)
        self.commit(pid, note, p["t_ref"], d.candidates, source=source, confidence=conf)

    # -------------------------------------------------------------- learning
    def learn(self, pid):
        """Keep a clip the user confirmed (1/2/3) as a new example of that phrase.

        Only explicit choices are learned, never automatic guesses, so mistakes
        are not reinforced. The templates are rebuilt in the background.
        """
        if self.last_clip is None or not config.LEARN_FROM_CHOICES or getattr(self.runtime, "kind", "") != "templates":
            return
        from collect_data import append_metadata, remove_metadata
        if self.learned_path is not None:          # changed their mind about the same clip
            self.learned_path.unlink(missing_ok=True)
            blendshape_path(self.learned_path).unlink(missing_ok=True)
            remove_metadata(self.learned_path.relative_to(config.DATA_DIR).as_posix())
        folder = config.RAW_DIR / self.args.speaker / pid
        folder.mkdir(parents=True, exist_ok=True)
        session = config.LEARN_SESSION
        reps = [int(p.stem.split("_")[1]) for p in folder.glob(f"{session}_*.npy")]
        path = folder / f"{session}_{max(reps, default=0) + 1:02d}.npy"
        np.save(path, self.last_clip.astype(np.float32))
        if self.last_blend is not None:
            np.save(blendshape_path(path), self.last_blend)
        append_metadata({"file": path.relative_to(config.DATA_DIR).as_posix(), "speaker": self.args.speaker,
                         "session": session, "phrase_id": pid, "phrase_text": self.text(pid),
                         "frames": len(self.last_clip), "fps": "", "lighting": "live", "glasses": 0,
                         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.learned_path = path
        print(f"[learn] saved your choice as a new example of {pid} ({path.name})")
        with self._learn_lock:
            self._learn_again = True
            if self._learn_thread is None or not self._learn_thread.is_alive():
                self._learn_thread = threading.Thread(target=self._refresh_templates, daemon=True)
                self._learn_thread.start()

    def _refresh_templates(self):
        from src.templates import for_speaker
        while True:
            with self._learn_lock:
                if not self._learn_again:
                    self._learn_thread = None
                    return
                self._learn_again = False
            try:
                full = for_speaker(self.args.speaker, self.full_runtime.label_map, rebuild=True)
                if full is not None and full.label_map == self.full_runtime.label_map:
                    self.full_runtime, self._ranking = full, None
                    self.runtime = full.restrict(self.focus) if self.focus else full
                    print(f"[learn] recogniser updated: {self.runtime.describe()}")
            except Exception as e:
                print(f"[learn] update failed: {e}")

    # ------------------------------------------------------------------ focus
    def toggle_focus(self, size=None):
        """Switch between all phrases and the speaker's most reliable ones."""
        if getattr(self.full_runtime, "kind", "") != "templates":
            self.note = "Focus mode needs the instant templates (run start.py to calibrate)."
            return
        if self.focus and size is None:
            self.focus, self.runtime = None, self.full_runtime
            self.note = f"All {len(self.runtime.label_map)} phrases."
            return
        size = size or config.FOCUS_PHRASES
        if self._ranking is None:
            self._ranking = self.full_runtime.rank_phrases()
        top = self._ranking[:max(2, min(size, len(self._ranking)))]
        self.focus = [pid for pid, _ in top]
        self.runtime = self.full_runtime.restrict(self.focus)
        self.note = (f"Focus: your {len(self.focus)} most reliable phrases (about {top[-1][1]:.0%} accurate). "
                     "F for all phrases.")
        print(f"[focus] {', '.join(self.text(p) for p in self.focus)} - estimated {top[-1][1]:.0%}")

    # ----------------------------------------------------------------- input
    def pick(self, k, now):
        if self.options and k <= len(self.options):
            pid = self.options[k - 1][0]
            cands = self.options
            self.pending = None
            self.commit(pid, f"You picked option {k} - learned from it.", now, cands, source="pick")
            self.learn(pid)
        elif self.last_committed and self.sentence and self.sentence[-1] == self.last_committed[0]:
            cands = self.last_committed[1]
            if k > len(cands):
                return
            pid = cands[k - 1][0]
            if pid == self.sentence[-1]:            # "yes, that was right"
                self.note = f"Confirmed \"{self.text(pid)}\" - learned from it."
                self.learn(pid)
                return
            self.sentence[-1] = pid
            if self.history:
                self.history[-1] = pid
            self.last_committed = (pid, cands)
            self.note = f"Corrected to option {k} - learned from it."
            self.speaker.say([pid], now)
            if self.log:
                self.log[-1] = {**self.log[-1], "pid": pid, "source": "corrected"}
            self.write_transcript(f"  corrected to: {self.text(pid)}")
            self.learn(pid)

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
                self.write_transcript("SENTENCE: " + ". ".join(self.text(p) for p in self.sentence) + ".")
        elif key in (8, 127):
            if self.sentence:
                self.sentence.pop()
                self.last_committed = None
        elif key in (ord("c"), ord("C")):
            self.sentence.clear()
            self.options = None
            self.retry = None
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
            if self.alert:
                self.write_transcript(f"alert acknowledged: {self.text(self.alert)}")
            self.alert = None
        elif key in (ord("f"), ord("F")) and not self.capturing and not self.pending:
            self.toggle_focus()
        elif key in (ord("v"), ord("V")):
            self.hud = not self.hud
        return True

    # --------------------------------------------------------- per-frame step
    def step(self, pts, now, blend=None):
        row = pts if pts is not None else NAN_ROW
        blend = blend if pts is not None else None
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
            self.cap_blend.append(blend)
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
            self.prebuffer.append((row, blend))
            opening = mar > config.AUTO_MAR_THRESHOLD and motion > config.AUTO_MOTION_THRESHOLD / 2
            active = pts is not None and (motion > config.AUTO_MOTION_THRESHOLD or opening)
            self.active_run = self.active_run + 1 if active else 0
            if self.active_run >= config.AUTO_START_FRAMES:
                self.active_run = 0
                self.start_capture("auto", now, list(self.prebuffer))
                self.prebuffer.clear()
        if self.alert and now - self.alert_last_beep >= config.ALERT_REPEAT_SECONDS:
            self.alert_last_beep = now          # nobody acknowledged yet: sound it again
            self.speaker.alert()
        self.resolve_llm(now)
        return mar, motion

    # ---------------------------------------------------------------- render
    @staticmethod
    def _blank(rect):
        x0, y0, x1, y1 = rect
        return np.full((y1 - y0, x1 - x0, 3), ui.BG, np.uint8)

    @staticmethod
    def _paste(canvas, rect, img):
        x0, y0, x1, y1 = rect
        canvas[y0:y1, x0:x1] = img

    def _candidates(self):
        return self.decision.candidates if self.decision else []

    def _candidate_rects(self):
        """Clickable rows of the prediction card, in window coordinates."""
        x0, y0, x1, _ = PRED
        return [(x0 + 8, y0 + 76 + 32 * i, x1 - 8, y0 + 106 + 32 * i) for i in range(len(self._candidates()))]

    def _headline(self, now):
        d = self.decision
        if self.capturing:
            return "Listening...", ui.ACCENT
        if self.pending and d:
            return self.text(d.candidates[0][0]) + " ?", ui.AMBER
        if self.options:
            return "Not sure - pick 1, 2 or 3", ui.AMBER
        if self.last_committed and self.sentence:
            return self.text(self.sentence[-1]), ui.GREEN
        if d:
            return self.text(d.candidates[0][0]) + " ?", ui.MUTED
        return "Press SPACE, then mouth a phrase", ui.DIM

    def _render_header(self, blink):
        img, p = self._blank(HEADER), ui.Painter()
        if self.alert:
            ui.rounded_rect(img, 8, 7, W - 8, 55, 12, ui.RED if blink else (40, 40, 150))
            p.text(f"EMERGENCY  —  “{self.text(self.alert)}”  —  caregiver alerted.   "
                   "Press A (or click here) to acknowledge.", (24, 31), 21, ui.TEXT, "bold", "lm")
            return p.flush(img)
        p.text("SilentSpeech", (20, 31), 28, ui.TEXT, "bold", "lm")
        p.text("silent lips  ›  phrase  ›  their own voice", (214, 33), 15, ui.MUTED, "regular", "lm")
        chips = []
        if self.runtime is not None:
            acc = getattr(self.runtime, "report", {}).get("loo_accuracy")
            chips.append(((f"Focus {len(self.focus)} phrases" if self.focus else
                           f"{len(self.runtime.label_map)} phrases") + (f" · ~{acc:.0%}" if acc else ""), ui.ACCENT))
        chips.append((f"{self.llm.provider.title()} context on" if self.llm_on else "Context (LLM) off",
                      ui.GREEN if self.llm_on else ui.DIM))
        chips.append((f"Voice: {self.speaker.label}", ui.GREEN if "own" in self.speaker.label else ui.AMBER))
        x = W - 16
        for label, color in reversed(chips):
            tw = ui.text_width(label, 13, "semibold")
            ui.rounded_rect(img, x - tw - 34, 17, x, 45, 14, ui.CARD)
            self.cv2.circle(img, (x - tw - 20, 31), 5, color, -1, self.cv2.LINE_AA)
            p.text(label, (x - tw - 10, 31), 13, ui.TEXT, "semibold", "lm")
            x -= tw + 44
        return p.flush(img)

    def _render_prediction(self, now):
        img, p = self._blank(PRED), ui.Painter()
        w, h = PRED[2] - PRED[0], PRED[3] - PRED[1]
        ui.card(img, p, 0, 0, w - 1, h - 1, "Prediction")
        headline, color = self._headline(now)
        p.text(ui.wrap(headline, w - 32, 28, "bold", max_lines=1)[0], (16, 30), 28, color, "bold")
        for i, (pid, prob) in enumerate(self._candidates()):
            y = 76 + 32 * i
            ui.rounded_rect(img, 16, y + 2, 42, y + 26, 6, (70, 62, 56))
            p.text(str(i + 1), (29, y + 14), 14, ui.TEXT, "bold", "mm")
            p.text(ui.wrap(self.text(pid), w - 270, 17, "semibold", max_lines=1)[0], (54, y + 14), 17, ui.TEXT,
                   "semibold", "lm")
            ui.bar(img, w - 200, y + 9, 130, 10, prob, ui.level_color(prob, config.CONF_ACCEPT, config.CONF_MIN))
            p.text(f"{prob:.0%}", (w - 16, y + 14), 14, ui.MUTED, "regular", "rm")
        note_color = ui.AMBER if (self.pending or self.options) else ui.MUTED
        for j, line in enumerate(ui.wrap(self.note, w - 32, 14, "regular", max_lines=2)):
            p.text(line, (16, h - 42 + 18 * j), 14, note_color)
        return p.flush(img)

    def _render_widgets_frame(self):
        img, p = self._blank(WIDGETS), ui.Painter()
        w, h = WIDGETS[2] - WIDGETS[0], WIDGETS[3] - WIDGETS[1]
        ui.card(img, p, 0, 0, 260, h - 1, "Mouth")
        ui.card(img, p, 270, 0, w - 1, h - 1, "Muscle activity")
        if self.has_blend:
            for i, (label, _) in enumerate(ui.MUSCLE_ROWS):
                p.text(label, (284, 30 + 15 * i), 12, ui.MUTED, "regular", "lm")
        else:
            p.text("Needs the Face Landmarker tracker", (284, 60), 13, ui.DIM)
        return p.flush(img)

    def _render_sentence(self):
        img, p = self._blank(SENT), ui.Painter()
        w, h = SENT[2] - SENT[0], SENT[3] - SENT[1]
        ui.card(img, p, 0, 0, w - 1, h - 1, "Sentence")
        p.text("ENTER says it all", (w - 14, 15), 12, ui.DIM, "regular", "rm")
        if self.sentence:
            text = ". ".join(self.text(pid) for pid in self.sentence) + "."
            lines = ui.wrap(text, w - 32, 20, "semibold")[-2:]
            for j, line in enumerate(lines):
                p.text(line, (16, 32 + 26 * j), 20, ui.TEXT, "semibold")
        else:
            p.text("(empty) - phrases you say are collected here", (16, 40), 15, ui.DIM)
        return p.flush(img)

    def _render_history(self):
        img, p = self._blank(HIST), ui.Painter()
        w, h = HIST[2] - HIST[0], HIST[3] - HIST[1]
        ui.card(img, p, 0, 0, w - 1, h - 1, "History")
        p.text(f"saved to logs/{self.transcript.name}" if config.SAVE_TRANSCRIPT else "", (w - 14, 15), 11,
               ui.DIM, "regular", "rm")
        entries = list(self.log)[-4:][::-1]
        if not entries:
            p.text("Nothing said yet.", (16, 40), 14, ui.DIM)
        for j, e in enumerate(entries):
            y = 34 + 21 * j
            p.text(time.strftime("%H:%M", e["time"]), (16, y), 13, ui.DIM)
            p.text(ui.wrap(self.text(e["pid"]), w - 250, 15, "semibold", max_lines=1)[0], (66, y - 1), 15,
                   ui.TEXT if j == 0 else ui.MUTED, "semibold")
            tag = SOURCE_LABEL.get(e["source"], e["source"])
            if e.get("confidence") is not None and e["source"] in ("lips", "llm", "fallback"):
                tag += f" {e['confidence']:.0%}"
            p.text(tag, (w - 14, y), 12, ui.DIM, "regular", "ra")
        return p.flush(img)

    def _board_ids(self):
        """Phrases shown on the tap-to-speak board: current candidates first, at most 10."""
        if self.runtime is None:
            return []
        ids = [pid for pid, _ in self._candidates()]
        ids += [pid for pid in self.runtime.label_map if pid not in ids]
        return ids[:10]

    def _tile_rects(self, n):
        x0, y0, x1, y1 = BOARD
        cols, rows, gap = 5, 2, 6
        tw = (x1 - x0 - 24 - gap * (cols - 1)) / cols
        th = (y1 - y0 - 34 - gap * (rows - 1)) / rows
        return [(int(x0 + 12 + (i % cols) * (tw + gap)), int(y0 + 28 + (i // cols) * (th + gap)),
                 int(x0 + 12 + (i % cols) * (tw + gap) + tw), int(y0 + 28 + (i // cols) * (th + gap) + th))
                for i in range(n)]

    def _render_board(self, ids, highlight):
        img, p = self._blank(BOARD), ui.Painter()
        w, h = BOARD[2] - BOARD[0], BOARD[3] - BOARD[1]
        ui.card(img, p, 0, 0, w - 1, h - 1, "Tap to speak")
        p.text("click a phrase to say it", (w - 14, 15), 12, ui.DIM, "regular", "rm")
        for (x0, y0, x1, y1), pid in zip(self._tile_rects(len(ids)), ids):
            x0, y0, x1, y1 = x0 - BOARD[0], y0 - BOARD[1], x1 - BOARD[0], y1 - BOARD[1]
            emergency = self.phrases[pid].get("emergency")
            fill = (55, 45, 88) if emergency else (58, 52, 47)
            ui.rounded_rect(img, x0, y0, x1, y1, 8, fill)
            if pid in highlight:
                color, number = highlight[pid]
                ui.rounded_rect(img, x0, y0, x1, y1, 8, color, 2)
                if number:
                    p.text(str(number), (x0 + 7, y0 + 4), 11, color, "bold")
            lines = ui.wrap(self.text(pid), x1 - x0 - 12, 13, "semibold", max_lines=2)
            for j, line in enumerate(lines):
                yy = (y0 + y1) / 2 + (j - (len(lines) - 1) / 2) * 15
                p.text(line, ((x0 + x1) / 2, yy), 13, ui.TEXT, "semibold", "mm")
        return p.flush(img)

    def render(self, frame, pts, mar, motion, now):
        cv2 = self.cv2
        canvas = np.full((H, W, 3), ui.BG, np.uint8)
        blink = bool(self.alert) and int(now * 2) % 2 == 0
        self._paste(canvas, HEADER, self.ui.get(
            "header", (self.alert, blink, self.focus and len(self.focus), self.llm_on, self.speaker.label,
                       id(self.runtime)), lambda: self._render_header(blink)))

        # video: face-mesh view (V toggles) or the plain camera with lip contours
        view = draw_face_hud(frame, self.mesh, pts) if self.hud else draw_lips(frame.copy(), pts)
        view = cv2.flip(view, 1)
        if self.hud and self.mesh is not None:
            label = f"{self.tracker_name}: {len(self.mesh)} points" + (
                " + 52 muscle scores" if self.has_blend else "")
            draw_text(view, label, (12, VH - (40 if self.mode == "auto" else 16)), 0.45, (150, 230, 90), 1)
        if pts is None:
            draw_text(view, "No face detected - face the camera, light from the front", (12, 30), 0.6, RED, 2)
        if self.brightness is not None and self.brightness < 50:
            draw_text(view, "Too dark: put a light in front of you", (12, 58), 0.6, AMBER, 2)
        if self.capturing:
            pulse = 9 + int(3 * abs(np.sin(now * 6)))
            cv2.circle(view, (VW - 92, 28), pulse, (0, 0, 255), -1, cv2.LINE_AA)
            draw_text(view, "REC", (VW - 74, 36), 0.7, (0, 0, 255), 2)
            frac = (now - self.cap_start) / (config.RECORD_SECONDS if self.cap_kind == "ptt" else config.AUTO_MAX_SECONDS)
            cv2.rectangle(view, (0, VH - 8), (int(VW * min(frac, 1)), VH), (0, 0, 255), -1)
        elif self.mode == "auto":
            draw_text(view, "AUTO - listening for lip movement", (12, VH - 16), 0.55, AMBER, 1)
        ui.rounded_rect(canvas, VX - 2, VY - 2, VX + VW + 1, VY + VH + 1, 6, ui.CARD_EDGE, 1)
        canvas[VY:VY + VH, VX:VX + VW] = view

        # prediction, sentence, history: re-drawn only when their content changes
        cands = tuple((pid, round(p, 2)) for pid, p in self._candidates())
        self._paste(canvas, PRED, self.ui.get(
            "pred", (self._headline(now), cands, self.note, bool(self.options), bool(self.pending)),
            lambda: self._render_prediction(now)))
        self._paste(canvas, SENT, self.ui.get("sentence", tuple(self.sentence), self._render_sentence))
        self._paste(canvas, HIST, self.ui.get(
            "history", tuple((e["pid"], e["source"], time.mktime(e["time"])) for e in self.log), self._render_history))

        # mouth close-up + live muscle bars (every frame)
        widgets = self.ui.get("widgets", self.has_blend, self._render_widgets_frame).copy()
        widgets[24:24 + 124, 10:10 + 240] = ui.mouth_inset(frame, pts, (240, 124))
        values = ui.muscle_values(self.last_blend_live) if self.has_blend else None
        if values is not None:
            self.muscles = values if self.muscles is None else 0.5 * self.muscles + 0.5 * values
            for i, v in enumerate(self.muscles):
                ui.bar(widgets, 400, 25 + 15 * i, WIDGETS[2] - WIDGETS[0] - 416, 8, v, ui.ACCENT)
        self._paste(canvas, WIDGETS, widgets)

        # tap-to-speak board
        ids = self._board_ids()
        self.tiles = list(zip(self._tile_rects(len(ids)), ids))
        cands = self._candidates()
        if self.options:          # unsure: the three choices, numbered, in amber
            highlight = {pid: (ui.AMBER, k) for k, (pid, _) in enumerate(self.options, 1)}
        elif cands:               # otherwise just the best guess, coloured by confidence
            highlight = {cands[0][0]: (ui.level_color(cands[0][1], config.CONF_ACCEPT, config.CONF_MIN), None)}
        else:
            highlight = {}
        self._paste(canvas, BOARD, self.ui.get(
            "board", (tuple(ids), tuple(sorted(highlight.items()))), lambda: self._render_board(ids, highlight)))

        # status bar: refreshed twice a second
        lat = self.speaker.last_latency_ms
        status_key = (int(now * 2), self.mode, round(self.fps.fps), lat and round(lat))
        self._paste(canvas, STATUS, self.ui.get("status", status_key, lambda: self._render_status(lat)))
        return canvas

    def _render_status(self, lat):
        img, p = self._blank(STATUS), ui.Painter()
        img[:] = (34, 30, 28)
        left = (f"{'PUSH-TO-TALK' if self.mode == 'ptt' else 'AUTO MODE'}   ·   {self.fps.fps:4.1f} FPS   ·   "
                f"latency {'-' if lat is None else f'{lat:.0f} ms'}   ·   {self.tracker_name or 'tracker'}   ·   "
                f"speaker {self.args.speaker}")
        p.text(left, (16, 13), 12, ui.MUTED, "regular", "lm")
        p.text("SPACE speak  ·  1/2/3 pick  ·  ENTER say all  ·  BKSP undo  ·  C clear  ·  "
               "F focus  ·  V view  ·  M mode  ·  L context  ·  Q quit",
               (W - 16, 13), 12, ui.DIM, "regular", "rm")
        return p.flush(img)

    # ----------------------------------------------------------------- mouse
    def on_mouse(self, event, x, y, flags, param):
        if event == self.cv2.EVENT_LBUTTONDOWN:
            self.clicks.append((x, y))

    def handle_clicks(self, now):
        clicks, self.clicks = self.clicks, []
        for x, y in clicks:
            if self.alert and y < HEADER[3]:
                self.handle_key(ord("a"), now)
                continue
            for k, (x0, y0, x1, y1) in enumerate(self._candidate_rects(), 1):
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self.pick(k, now)
                    break
            else:
                for (x0, y0, x1, y1), pid in self.tiles:
                    if x0 <= x <= x1 and y0 <= y <= y1 and not self.capturing:
                        self.tap(pid, now)
                        break

    # ------------------------------------------------------------------ main
    def run(self):
        cv2 = self.cv2
        cap = open_camera(self.args.camera)
        if cap is None:
            sys.exit("Could not open the webcam. Try --camera 1, close Zoom/Teams, or run python check_setup.py.")
        make_window(WINDOW, W, H)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        failures = 0
        try:
            with make_tracker() as tracker:
                self.tracker_name = tracker.name
                print(f"Face tracking: {tracker.name}")
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
                    self.mesh, blend = tracker.last_mesh, tracker.last_blendshapes
                    self.last_blend_live = blend
                    self.has_blend = self.has_blend or blend is not None
                    mar, motion = self.step(pts, now, blend)
                    self.handle_clicks(now)
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
    ap.add_argument("--focus", type=int, metavar="N",
                    help="start with only your N most reliable phrases (press F in the app to switch)")
    args = ap.parse_args(argv)
    log_path, _ = start_session_log("app")
    try:
        app = App(args)
        if args.focus:
            app.toggle_focus(args.focus)
        app.run()
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"\nSomething went wrong. The details are saved in:\n  {log_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
