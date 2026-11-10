"""SilentSpeech live demo.

    python app.py                      # push-to-talk, cloned voice, LLM on
    python app.py --mode auto          # automatic start/stop detection
    python app.py --voice default      # offline pyttsx3 voice
    python app.py --no-llm             # fully offline mode
    python app.py --speaker s02        # another teammate's voice and model

Keys
    SPACE      capture a phrase (push-to-talk; also works in auto mode)
    1 / 2 / 3  pick an option when the system is unsure, or correct / confirm the last phrase
               (every choice is kept as a new example: it learns as you use it, and the deep
               model catches up with those examples when you close the app)
    ENTER      speak the whole sentence
    BACKSPACE  remove the last phrase          C  clear the sentence
    M          push-to-talk <-> auto mode      L  LLM layer on/off
    A          acknowledge an emergency alert  Q  quit
    F          focus: switch to your most reliable phrases (higher accuracy) and back
    V          view: face-mesh hologram <-> plain camera
    E          emotion reading on/off (facial expression and pain, from the muscle scores)
    - / +      volume down / up (remembered)       S  sound cues on/off (alarms always sound)
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
from src.emotion import COLORS as EMO_COLORS
from src.emotion import EMOTIONS, EmotionReader
from src.emotion import describe as describe_expression
from src.landmarks import (BLENDSHAPE_NAMES, N_POINTS, draw_face_hud, draw_lips, make_tracker,
                           mouth_aspect_ratio)
from src.llm_corrector import LLMCorrector
from src.recognizer import load_recognizer, with_deep
from src.speaker import Speaker
from src import ui
from src.ui import PanelCache
from src.utils import (FPSCounter, draw_text, load_env, load_phrases, load_settings, make_window, open_camera,
                       save_settings, start_session_log)

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
MOUTH_CARD, EMO_CARD, MUSCLE_CARD = (0, 176), (186, 405), (415, 591)   # x ranges inside WIDGETS
KEYS = [("SPACE", "speak"), ("1 2 3", "pick"), ("ENTER", "say all"), ("BKSP", "undo"), ("C", "clear"),
        ("F", "focus"), ("E", "emotion"), ("V", "view"), ("M", "mode"), ("L", "context"), ("- +", "volume"),
        ("S", "sounds"), ("Q", "quit")]
VOICE = (235, 150, 120)          # colour of the SPEAKING state (BGR)
VIDEO_RADIUS = 12
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
        saved = load_settings()
        self.speaker = Speaker({pid: self.phrases[pid] for pid in known}, args.speaker, args.voice,
                               volume=saved.get("volume"), sounds=saved.get("sounds"))
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
        self.note = ""                 # the steps on the prediction card explain how to start
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
        self.mar_hist = deque(maxlen=150)       # (time, mouth opening) for the lip-signal trace

        # facial emotion and pain expression from the 52 muscle scores (E toggles)
        self.emotion = EmotionReader()
        self.emotion_on = config.EMOTION
        self.reading = None
        self.utter_emotion = None       # the face around the last mouthed phrase
        self.pain_since = None          # when the current pain expression started
        self.pain_alert = None          # when a held pain expression alerted the caregiver
        self.pain_quiet_until = 0.0

        # focus mode (F): only the speaker's most reliable phrases
        self.full_runtime = self.runtime
        self.focus = None
        self._ranking = None

        # learning from the user's 1/2/3 choices
        self.last_clip = None
        self.last_blend = None
        self.learned_path = None
        self.learned_count = 0          # new examples this session (the deep model is updated on exit)
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

    def icon(self, pid):
        """The phrase's picture symbol (AAC style), e.g. a water drop for "I need water"."""
        return ui.phrase_icon(self.phrases.get(pid))

    # --------------------------------------------------------------- capture
    def start_capture(self, kind, now, seed=()):
        self.capturing, self.cap_kind, self.cap_start = True, kind, now
        self.cap_points = [row for row, _ in seed]
        self.cap_blend = [b for _, b in seed]
        self.still_since = None
        self.options = None
        self.speaker.cue("listen")

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
        self.speaker.cue("done")                 # the listening window is over
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
        # the face just before and while mouthing it (src/emotion.py)
        self.utter_emotion = self.emotion.summary(self.cap_start - 1.5, now) if self.emotion_on else None
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
        if self.utter_emotion:
            print(f"[face] {describe_expression(self.utter_emotion) or 'calm'}")
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
                self.pending = {"future": self.llm.submit(d.candidates, list(self.history), self.phrases,
                                                          self._expression_for_llm()),
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
            self.speaker.cue("unsure")

    def commit(self, pid, note, t_ref, candidates=None, source="lips", confidence=None):
        self.sentence.append(pid)
        self.history.append(pid)
        self.options = None
        self.pending = None
        self.retry = None
        self.note = note
        self.last_committed = (pid, candidates or [])
        self.speaker.say([pid], t_ref)
        emotion = self._phrase_emotion(source, t_ref)
        self.log.append({"time": time.localtime(), "pid": pid, "source": source, "confidence": confidence,
                         "emotion": emotion})
        conf = f" {confidence:.0%}" if confidence is not None else ""
        face = describe_expression(emotion)
        self.write_transcript(f"{self.text(pid)}  [{SOURCE_LABEL.get(source, source)}{conf}]"
                              + (f"  ({face})" if face else ""))

    def _phrase_emotion(self, source, t_ref):
        """The facial expression that goes with a spoken phrase (None when emotion reading is off)."""
        if not self.emotion_on:
            return None
        if source == "tap":                       # no mouthing: the face over the last 3 seconds
            return self.emotion.summary(t_ref - 3.0, t_ref)
        return self.utter_emotion

    def _expression_for_llm(self):
        if not (self.emotion_on and config.EMOTION_TO_LLM):
            return None
        return describe_expression(self.utter_emotion)

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
        learner = getattr(self.runtime, "kind", "") in ("templates", "deep")
        if self.last_clip is None or not config.LEARN_FROM_CHOICES or not learner:
            return
        from collect_data import append_metadata, remove_metadata
        if self.learned_path is not None:          # changed their mind about the same clip
            self.learned_path.unlink(missing_ok=True)
            blendshape_path(self.learned_path).unlink(missing_ok=True)
            remove_metadata(self.learned_path.relative_to(config.DATA_DIR).as_posix())
        else:
            self.learned_count += 1
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
        self.speaker.cue("learned", after_speech=True)
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
                    if getattr(self.full_runtime, "kind", "") == "deep":   # keep the deep model in front
                        full, _ = with_deep(full, self.args.speaker, quiet=True)
                    self.full_runtime, self._ranking = full, None
                    self.runtime = full.restrict(self.focus) if self.focus else full
                    print(f"[learn] recogniser updated: {self.runtime.describe()}")
            except Exception as e:
                print(f"[learn] update failed: {e}")

    # ------------------------------------------------------------------ focus
    def toggle_focus(self, size=None):
        """Switch between all phrases and the speaker's most reliable ones."""
        if getattr(self.full_runtime, "kind", "") not in ("templates", "deep"):
            self.note = "Focus mode needs your own calibration clips (run start.py to calibrate)."
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
            self.speaker.cue("on" if self.mode == "auto" else "off")
        elif key in (ord("l"), ord("L")):
            if self.llm_on:
                self.llm_on = False
            elif self.llm and self.llm.available:
                self.llm_on = True
            else:
                self.note = "LLM unavailable: " + self.llm.unavailable_reason
            self.speaker.cue("on" if self.llm_on else "off")
        elif key in (ord("+"), ord("=")):
            self.change_volume(+0.1)
        elif key in (ord("-"), ord("_")):
            self.change_volume(-0.1)
        elif key in (ord("s"), ord("S")):
            if self.speaker.sounds:
                self.speaker.cue("off")
            self.speaker.sounds = not self.speaker.sounds
            self.speaker.cue("on")
            save_settings(sounds=self.speaker.sounds)
            self.note = ("Sound cues on: a chime when it starts listening, a tick when it stops, a low tone when "
                         "unsure." if self.speaker.sounds else
                         "Sound cues off (alarms still sound). S turns them back on.")
        elif key in (ord("a"), ord("A")):
            if self.alert:
                self.write_transcript(f"alert acknowledged: {self.text(self.alert)}")
            if self.pain_alert:
                self.write_transcript("alert acknowledged: pain expression")
                self.pain_quiet_until = now + config.PAIN_ALERT_QUIET_SECONDS
            self.alert = None
            self.pain_alert = self.pain_since = None
        elif key in (ord("e"), ord("E")):
            self.emotion_on = not self.emotion_on
            self.reading = self.pain_alert = self.pain_since = None
            self.note = ("Emotion reading on: facial expression and pain, from the face's muscle scores."
                         if self.emotion_on else "Emotion reading off. E turns it back on.")
            self.speaker.cue("on" if self.emotion_on else "off")
        elif key in (ord("f"), ord("F")) and not self.capturing and not self.pending:
            self.toggle_focus()
            self.speaker.cue("on" if self.focus else "off")
        elif key in (ord("v"), ord("V")):
            self.hud = not self.hud
        return True

    def change_volume(self, step):
        v = self.speaker.set_volume(round(self.speaker.volume + step, 2))
        save_settings(volume=v)
        limit = " (maximum)" if v >= 2.0 else " (minimum)" if v <= 0.3 else ""
        self.note = f"Volume {v:.0%}{limit}. - and + change it; it is remembered."
        self.speaker.cue("volume")

    def check_pain(self, now):
        """A strong pain expression held for a few seconds (not while mouthing) alerts the caregiver."""
        r = self.reading
        if not config.PAIN_ALERT or r is None or self.capturing or now < self.pain_quiet_until:
            return
        if r.pain >= config.PAIN_ALERT_LEVEL:
            self.pain_since = self.pain_since or now
            held = now - self.pain_since
            if self.pain_alert is None and held >= config.PAIN_ALERT_SECONDS:
                self.pain_alert = self.alert_last_beep = now
                self.speaker.alert("pain")
                self.write_transcript(f"ALERT raised: pain expression ({r.pain_level}) held for {held:.0f} s")
                print(f"[pain] {r.pain_level} pain expression held for {held:.0f} s - caregiver alerted")
        elif r.pain < 0.6 * config.PAIN_ALERT_LEVEL:
            self.pain_since = None

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
            self.mar_hist.append((now, mar))
        else:
            self.prev_norm = None

        if self.emotion_on and blend is not None:
            self.reading = self.emotion.update(blend, now, talking=self.capturing)
            self.check_pain(now)
        elif self.reading is not None and self.emotion.last_t is not None and now - self.emotion.last_t > 1.0:
            self.reading = None                     # face lost for a second: no stale reading

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
        if (self.alert or self.pain_alert) and now - self.alert_last_beep >= config.ALERT_REPEAT_SECONDS:
            self.alert_last_beep = now          # nobody acknowledged yet: sound it again
            self.speaker.alert("pain" if self.pain_alert and not self.alert else "emergency")
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
        return [(x0 + 8, y0 + 84 + 30 * i, x1 - 8, y0 + 110 + 30 * i) for i in range(len(self._candidates()))]

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
        return "Ready when you are", ui.TEXT

    def _votes(self):
        """How the deep model and the templates voted on the last clip (a second opinion)."""
        v = getattr(self.runtime, "last_votes", None)
        if not v or self.capturing or self.decision is None:
            return None
        if v["deep"][0] == v["templates"][0]:
            return "neural net + templates agree", ui.GREEN
        return f"templates guessed “{self.text(v['templates'][0])}”", ui.AMBER

    def _render_header(self, blink):
        img, p = self._blank(HEADER), ui.Painter()
        if self.alert:
            ui.rounded_rect(img, 8, 7, W - 8, 55, 12, ui.RED if blink else (40, 40, 150))
            p.text(f"EMERGENCY  —  “{self.text(self.alert)}”  —  caregiver alerted.   "
                   "Press A (or click here) to acknowledge.", (24, 31), 21, ui.TEXT, "bold", "lm")
            return p.flush(img)
        if self.pain_alert:
            ui.rounded_rect(img, 8, 7, W - 8, 55, 12, (30, 115, 210) if blink else (35, 75, 135))
            p.text("PAIN EXPRESSION  —  held for a few seconds, please check on them.   "
                   "Press A (or click here) to dismiss.", (24, 31), 21, ui.TEXT, "bold", "lm")
            return p.flush(img)
        ui.logo(img, 16, 12, 38)
        p.text("SilentSpeech", (64, 30), 26, ui.TEXT, "bold", "lm")
        p.text("silent lips  ›  phrase  ›  their own voice", (80 + ui.text_width("SilentSpeech", 26, "bold"), 32),
               14, ui.MUTED, "regular", "lm")
        chips = []
        if self.runtime is not None:
            acc = getattr(self.runtime, "report", {}).get("loo_accuracy")
            engine = "Deep AI · " if getattr(self.runtime, "kind", "") == "deep" else ""
            chips.append((engine + (f"Focus {len(self.focus)} phrases" if self.focus else
                                    f"{len(self.runtime.label_map)} phrases") + (f" · ~{acc:.0%}" if acc else ""),
                          ui.ACCENT))
        chips.append((f"{self.llm.provider.title()} context on" if self.llm_on else "Context (LLM) off",
                      ui.GREEN if self.llm_on else ui.DIM))
        chips.append(("Emotion on" if self.emotion_on else "Emotion off", ui.GREEN if self.emotion_on else ui.DIM))
        cues = "" if self.speaker.sounds else " · cues off"
        chips.append((f"Voice: {self.speaker.label} · {self.speaker.volume:.0%}{cues}",
                      ui.GREEN if "own" in self.speaker.label else ui.AMBER))
        x = W - 16
        for label, color in reversed(chips):
            tw = ui.text_width(label, 13, "semibold")
            ui.rounded_rect(img, x - tw - 34, 17, x, 45, 14, ui.CARD)
            self.cv2.circle(img, (x - tw - 20, 31), 5, color, -1, self.cv2.LINE_AA)
            p.text(label, (x - tw - 10, 31), 13, ui.TEXT, "semibold", "lm")
            x -= tw + 44
        return p.flush(img)

    def _state(self):
        """What the app is doing right now (the badge on the prediction card)."""
        if self.runtime is None:
            return "TRACKING ONLY", ui.DIM
        if self.capturing:
            return "LISTENING", ui.RED
        if self.pending:
            return "THINKING", ui.AMBER
        if self.options:
            return "YOUR CHOICE", ui.AMBER
        if self.speaker.busy:
            return "SPEAKING", VOICE
        if self.last_committed and self.sentence:
            return "SPOKEN", ui.GREEN
        return "READY", ui.ACCENT

    def _headline_icon(self):
        if self.capturing or (not self.decision and not self.sentence):
            return "\U0001F444"                  # lips
        if self.options:
            return "\U0001F914"                  # thinking face
        if self.last_committed and self.sentence:
            return self.icon(self.sentence[-1])
        return self.icon(self.decision.candidates[0][0]) if self.decision else ""

    def _ring(self):
        """(fraction, colour, label) for the gauge beside the headline, or None."""
        if self.capturing or not (self.decision or self.log):
            return None
        if self.last_committed and self.sentence and self.log and not (self.options or self.pending):
            c = self.log[-1].get("confidence")
            if c is None:                         # tapped, picked or corrected: the user chose it
                return 1.0, ui.GREEN, "you"
            return c, ui.level_color(c, config.CONF_ACCEPT, config.CONF_MIN), f"{c:.0%}"
        if self.decision:
            c = self.decision.candidates[0][1]
            return c, ui.level_color(c, config.CONF_ACCEPT, config.CONF_MIN), f"{c:.0%}"
        return None

    def _draw_steps(self, img, p, y):
        """Three steps for a first-time user (or a judge): how to use it, at a glance."""
        if self.mode == "auto":
            first = ("\U0001F440", "Face the camera", "auto mode is on")
            second = ("\U0001F444", "Mouth a phrase", "it starts by itself")
        else:
            first = ("⌨", "Press SPACE", "it starts listening")
            second = ("\U0001F444", "Mouth a phrase", "silently, about 2 s")
        voice = "in your own voice" if "own" in self.speaker.label else "out loud"
        steps = [first, second, ("\U0001F50A", "Hear it spoken", voice)]
        bw, gap = 172, 24
        for k, (icon, title, sub) in enumerate(steps):
            x = 16 + k * (bw + gap)
            ui.rounded_rect(img, x, y, x + bw, y + 52, 10, (52, 46, 42))
            p.icon(icon, (x + 23, y + 26), 22)
            p.text(title, (x + 44, y + 18), 14, ui.TEXT, "semibold", "lm")
            p.text(sub, (x + 44, y + 36), 11, ui.MUTED, "regular", "lm")
            if k < len(steps) - 1:
                p.text("›", (x + bw + gap / 2, y + 25), 22, ui.DIM, "regular", "mm")

    def _render_prediction(self, now):
        img, p = self._blank(PRED), ui.Painter()
        w, h = PRED[2] - PRED[0], PRED[3] - PRED[1]
        state, scolor = self._state()
        busy = state in ("LISTENING", "THINKING", "YOUR CHOICE")
        ui.card(img, p, 0, 0, w - 1, h - 1, "Prediction", edge=ui._mix(scolor, ui.CARD_EDGE, 0.6) if busy else ui.CARD_EDGE)
        bx = ui.badge(img, p, w - 12, 16, state, scolor)
        votes = self._votes()
        if votes:
            label, vcolor = votes
            p.text(ui.wrap(label, bx - 130, 11, "regular", max_lines=1)[0], (bx - 10, 16), 11, vcolor, "regular", "rm")
        headline, color = self._headline(now)
        ring = self._ring()
        p.icon(self._headline_icon(), (34, 55), 30)
        p.text(ui.wrap(headline, w - (140 if ring or self.capturing else 76), 25, "bold", max_lines=1)[0], (60, 55), 25,
               color, "bold", "lm")
        if ring:
            value, rcolor, label = ring
            ui.ring_gauge(img, w - 42, 55, 21, value, rcolor, 5)
            p.text(label, (w - 42, 55), 11 if len(label) > 3 else 12, rcolor, "bold", "mm")
        for i, (pid, prob) in enumerate(self._candidates()):
            y = 84 + 30 * i
            ui.rounded_rect(img, 16, y + 2, 40, y + 24, 6, (70, 62, 56))
            p.text(str(i + 1), (28, y + 13), 13, ui.TEXT, "bold", "mm")
            p.icon(self.icon(pid), (57, y + 13), 16)
            p.text(ui.wrap(self.text(pid), w - 300, 16, "semibold", max_lines=1)[0], (72, y + 13), 16, ui.TEXT,
                   "semibold", "lm")
            ui.bar(img, w - 192, y + 9, 124, 8, prob, ui.level_color(prob, config.CONF_ACCEPT, config.CONF_MIN))
            p.text(f"{prob:.0%}", (w - 16, y + 13), 13, ui.MUTED, "regular", "rm")
        if self.capturing:
            p.text("Silently mouth your phrase now", (16, 100), 17, ui.TEXT, "semibold", "lm")
            p.text("keep your head still and your face in the light", (16, 124), 12, ui.MUTED, "regular", "lm")
        elif not self._candidates():
            self._draw_steps(img, p, 92)
        note_color = ui.AMBER if (self.pending or self.options) else ui.MUTED
        for j, line in enumerate(ui.wrap("" if self.capturing else self.note, w - 32, 13, "regular", max_lines=2)):
            p.text(line, (16, h - 40 + 18 * j), 13, note_color)
        return p.flush(img)

    def _render_widgets_frame(self):
        img, p = self._blank(WIDGETS), ui.Painter()
        h = WIDGETS[3] - WIDGETS[1]
        ui.card(img, p, MOUTH_CARD[0], 0, MOUTH_CARD[1], h - 1, "Mouth")
        x0, x1 = MUSCLE_CARD
        ui.card(img, p, x0, 0, x1, h - 1, "Muscles")
        if self.has_blend:
            for i, (label, _) in enumerate(ui.MUSCLE_ROWS):
                p.text(label, (x0 + 12, 30 + 15 * i), 11, ui.MUTED, "regular", "lm")
        else:
            p.text("Needs Face Landmarker", (x0 + 12, 60), 12, ui.DIM)
        return p.flush(img)

    @staticmethod
    def _emotion_display(r):
        """(icon, label, colour, score). A strong pain expression is shown as pain: to the FACS
        emotion rules a pain grimace looks like anger or disgust, which would mislead a caregiver."""
        if r.pain_level == "strong":
            return "pain", "In pain", ui.RED, r.pain
        return r.emotion, r.label, EMO_COLORS[r.emotion], r.confidence

    def _emotion_key(self):
        r = self.reading if self.emotion_on else None
        if r is None:
            return (self.emotion_on, self.has_blend)
        icon, _, _, score = self._emotion_display(r)
        return (icon, round(score * 20), tuple(w for w, _ in r.drivers), r.pain_level,
                sum(r.pain >= s for s in ui.PAIN_STEPS), r.learning)

    def _render_emotion(self):
        """The emotion card: a drawn face, the emotion, the muscles behind it, and the pain meter."""
        x0, x1 = EMO_CARD
        w, h = x1 - x0 + 1, WIDGETS[3] - WIDGETS[1]
        img, p = np.full((h, w, 3), ui.BG, np.uint8), ui.Painter()
        ui.card(img, p, 0, 0, w - 1, h - 1, "Emotion")
        r = self.reading if self.emotion_on else None
        if r is None:
            msg = ("Emotion reading is off (press E)" if not self.emotion_on else
                   "Needs Face Landmarker" if not self.has_blend else "Looking for a face...")
            p.text(msg, (14, 66), 12, ui.DIM)
            return p.flush(img)
        icon, label, color, score = self._emotion_display(r)
        p.text("from face muscles", (w - 12, 10), 10, ui.DIM, "regular", "ra")
        ui.emotion_face(img, 40, 62, 24, icon, color)
        p.text(label, (76, 52), 20, color, "bold", "lm")
        p.text(f"{score:.0%}", (w - 12, 52), 13, ui.MUTED, "semibold", "rm")
        why = ("learning your resting face..." if r.learning else
               " · ".join(word for word, _ in r.drivers) if r.drivers else "relaxed face")
        p.text(ui.wrap(why, w - 86, 10, max_lines=1)[0], (76, 76), 10, ui.MUTED, "regular", "lm")
        p.text("LAST 30 SECONDS", (14, 91), 9, ui.DIM, "semibold")
        p.text("PAIN", (14, 129), 11, ui.DIM, "semibold", "lm")
        ui.pain_meter(img, 52, 125, r.pain)
        lit = sum(r.pain >= s for s in ui.PAIN_STEPS)
        level_color = ui.PAIN_COLORS[lit - 1] if r.pain_level != "none" and lit else ui.MUTED
        p.text(r.pain_level, (w - 12, 129), 12, level_color, "semibold", "rm")
        return p.flush(img)

    def _render_sentence(self):
        img, p = self._blank(SENT), ui.Painter()
        w, h = SENT[2] - SENT[0], SENT[3] - SENT[1]
        ui.card(img, p, 0, 0, w - 1, h - 1, "Sentence")
        ui.keycap(img, p, w - 14, 16, "ENTER", "says it all")
        if not self.sentence:
            p.text("(empty) - phrases you say are collected here", (16, 44), 15, ui.DIM)
            return p.flush(img)
        # one chip per phrase, newest last; two rows, the oldest drop off when it is full
        avail, gap = w - 32, 8
        widths = [min(ui.text_width(self.text(pid), 15, "semibold"), avail - 60) + 44 for pid in self.sentence]
        rows, used, r = [[], []], 0, 1
        for i in range(len(self.sentence) - 1, -1, -1):
            if rows[r] and used + widths[i] > avail:
                r, used = r - 1, 0
                if r < 0:
                    break
            rows[r].insert(0, i)
            used += widths[i] + gap
        rows = [row for row in rows if row]
        for j, row in enumerate(rows):
            x, cy = 16, 44 + 32 * j
            for i in row:
                pid, cw = self.sentence[i], widths[i]
                newest = i == len(self.sentence) - 1
                ui.rounded_rect(img, x, cy - 13, x + cw, cy + 13, 13, (60, 54, 49))
                if newest:
                    ui.rounded_rect(img, x, cy - 13, x + cw, cy + 13, 13, ui.ACCENT, 1)
                p.icon(self.icon(pid), (x + 17, cy), 15)
                p.text(ui.wrap(self.text(pid), cw - 44, 15, "semibold", max_lines=1)[0], (x + 32, cy), 15,
                       ui.TEXT, "semibold", "lm")
                x += cw + gap
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
            p.icon(self.icon(e["pid"]), (75, y + 9), 14)
            p.text(ui.wrap(self.text(e["pid"]), w - 290, 15, "semibold", max_lines=1)[0], (90, y - 1), 15,
                   ui.TEXT if j == 0 else ui.MUTED, "semibold")
            tag = SOURCE_LABEL.get(e["source"], e["source"])
            if e.get("confidence") is not None and e["source"] in ("lips", "llm", "fallback"):
                tag += f" {e['confidence']:.0%}"
            p.text(tag, (w - 14, y), 12, ui.DIM, "regular", "ra")
            emo = e.get("emotion")
            if emo and emo["emotion"] != "neutral":
                p.text(emo["label"].lower(), (w - 24 - ui.text_width(tag, 12), y), 12, EMO_COLORS[emo["emotion"]],
                       "semibold", "ra")
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
            color, number = highlight.get(pid, (None, None))
            p.icon(self.icon(pid), (x0 + 19, (y0 + y1) / 2), 20)
            lines = ui.wrap(self.text(pid), x1 - x0 - 42 - (12 if number else 0), 12, "semibold", max_lines=2)
            for j, line in enumerate(lines):
                yy = (y0 + y1) / 2 + (j - (len(lines) - 1) / 2) * 14
                p.text(line, (x0 + 35, yy), 12, ui.TEXT, "semibold", "lm")
            if color is not None:
                ui.rounded_rect(img, x0, y0, x1, y1, 8, color, 2)
            if number:
                self.cv2.circle(img, (x1 - 10, y0 + 10), 8, color, -1, self.cv2.LINE_AA)
                p.text(str(number), (x1 - 10, y0 + 10), 10, (30, 26, 24), "bold", "mm")
        return p.flush(img)

    def render(self, frame, pts, mar, motion, now):
        cv2 = self.cv2
        # Every panel is painted over its own area each frame, so the background is filled
        # once and the same canvas reused (a fresh 1280x720 fill cost ~16 ms on a slow CPU).
        if getattr(self, "_canvas", None) is None:
            self._canvas = np.full((H, W, 3), ui.BG, np.uint8)
        canvas = self._canvas
        blink = bool(self.alert or self.pain_alert) and int(now * 2) % 2 == 0
        self._paste(canvas, HEADER, self.ui.get(
            "header", (self.alert, blink, self.focus and len(self.focus), self.llm_on, self.speaker.label,
                       id(self.runtime), bool(self.pain_alert), self.emotion_on, self.speaker.volume,
                       self.speaker.sounds),
            lambda: self._render_header(blink)))

        # video: face-mesh view (V toggles) or the plain camera with lip contours
        view = draw_face_hud(frame, self.mesh, pts) if self.hud else draw_lips(frame.copy(), pts)
        view = cv2.flip(view, 1)
        if self.hud and self.mesh is not None:
            label = f"{self.tracker_name}: {len(self.mesh)} points" + (
                " + 52 muscle scores" if self.has_blend else "")
            draw_text(view, label, (12, VH - (40 if self.mode == "auto" else 16)), 0.45, (150, 230, 90), 1)
        if self.emotion_on and self.reading is not None and self.mesh is not None:
            self._draw_emotion_tag(view)
        if pts is None:
            draw_text(view, "No face detected - face the camera, light from the front", (12, 62), 0.6, RED, 2)
        if self.brightness is not None and self.brightness < 50:
            draw_text(view, "Too dark: put a light in front of you", (12, 90), 0.6, AMBER, 2)
        if self.capturing:
            pulse = 9 + int(3 * abs(np.sin(now * 6)))
            cv2.circle(view, (VW - 92, 28), pulse, (0, 0, 255), -1, cv2.LINE_AA)
            draw_text(view, "REC", (VW - 74, 36), 0.7, (0, 0, 255), 2)
            frac = (now - self.cap_start) / (config.RECORD_SECONDS if self.cap_kind == "ptt" else config.AUTO_MAX_SECONDS)
            cv2.rectangle(view, (0, VH - 8), (int(VW * min(frac, 1)), VH), (0, 0, 255), -1)
        elif self.mode == "auto":
            draw_text(view, "AUTO - listening for lip movement", (12, VH - 16), 0.55, AMBER, 1)
        self._draw_live_badge(view, now)
        self._draw_lip_signal(view, now)
        if self.capturing and self.mesh is not None:
            self._draw_scan_line(view, now)
        canvas[VY:VY + VH, VX:VX + VW] = view
        self._round_video_corners(canvas)
        ui.rounded_rect(canvas, VX - 2, VY - 2, VX + VW + 1, VY + VH + 1, VIDEO_RADIUS + 2,
                        ui.RED if self.capturing else ui.CARD_EDGE, 1)

        # prediction, sentence, history: re-drawn only when their content changes
        cands = tuple((pid, round(p, 2)) for pid, p in self._candidates())
        self._paste(canvas, PRED, self.ui.get(
            "pred", (self._headline(now), cands, self.note, bool(self.options), bool(self.pending), self._votes(),
                     self._state(), self._ring(), self._headline_icon()),
            lambda: self._render_prediction(now)))
        if self._state()[0] == "SPEAKING":
            self._draw_speaking_meter(canvas, now)
        if self.capturing:                    # recording progress in the gauge, live
            frac = (now - self.cap_start) / (config.RECORD_SECONDS if self.cap_kind == "ptt" else config.AUTO_MAX_SECONDS)
            ui.ring_gauge(canvas, PRED[2] - 42, PRED[1] + 55, 21, min(1.0, frac), ui.RED, 5)
        self._paste(canvas, SENT, self.ui.get("sentence", tuple(self.sentence), self._render_sentence))
        self._paste(canvas, HIST, self.ui.get(
            "history", tuple((e["pid"], e["source"], time.mktime(e["time"]), (e.get("emotion") or {}).get("emotion"))
                             for e in self.log), self._render_history))

        # mouth close-up + live muscle bars (every frame)
        widgets = self.ui.get("widgets", self.has_blend, self._render_widgets_frame).copy()
        mx0, mx1 = MOUTH_CARD
        widgets[24:24 + 124, mx0 + 10:mx1 - 10] = ui.mouth_inset(frame, pts, (mx1 - mx0 - 20, 124))
        ex0, ex1 = EMO_CARD
        widgets[:, ex0:ex1 + 1] = self.ui.get("emotion", self._emotion_key(), self._render_emotion)
        if self.emotion_on and self.reading is not None:
            ui.timeline_strip(widgets, ex0 + 14, 104, ex1 - ex0 - 26, 6, list(self.emotion.timeline), now,
                              EMO_COLORS, EMOTIONS)
        values = ui.muscle_values(self.last_blend_live) if self.has_blend else None
        if values is not None:
            self.muscles = values if self.muscles is None else 0.5 * self.muscles + 0.5 * values
            for i, v in enumerate(self.muscles):
                ui.bar(widgets, MUSCLE_CARD[0] + 92, 26 + 15 * i, MUSCLE_CARD[1] - MUSCLE_CARD[0] - 104, 7, v,
                       ui.ACCENT)
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

    def _draw_speaking_meter(self, canvas, now):
        """Four bars that move with the voice, in place of the dot on the SPEAKING badge."""
        cv2 = self.cv2
        x0 = int(PRED[2] - 12 - ui.text_width("SPEAKING", 11, "bold") - 28)
        cy = PRED[1] + 16
        cv2.rectangle(canvas, (x0 + 4, cy - 8), (x0 + 18, cy + 8), ui._mix(VOICE, ui.CARD, 0.22), -1)
        level = min(1.0, self.speaker.level / 0.12)
        for i in range(4):
            amp = 0.25 + 0.75 * level * (0.55 + 0.45 * abs(np.sin(now * 13 + i * 1.9)))
            half = max(1, int(7 * amp))
            cv2.rectangle(canvas, (x0 + 5 + 3 * i, cy - half), (x0 + 6 + 3 * i, cy + half), VOICE, -1)

    def _draw_live_badge(self, view, now):
        """'LIVE' with a pulsing dot and the frame rate, top-left of the video."""
        cv2 = self.cv2
        roi = view[10:38, 10:150]
        roi[:] = (roi * 0.35).astype(np.uint8)
        pulse = 0.55 + 0.45 * abs(np.sin(now * 3))
        cv2.circle(view, (24, 24), 5, (int(60 * pulse), int(60 * pulse), int(235 * pulse)), -1, cv2.LINE_AA)
        draw_text(view, "LIVE", (36, 29), 0.45, (235, 235, 235), 1)
        draw_text(view, f"{self.fps.fps:4.1f} FPS", (80, 29), 0.4, (170, 170, 170), 1)

    def _draw_lip_signal(self, view, now, seconds=3.0):
        """A small oscilloscope of mouth opening over the last 3 seconds (bottom-right of the video)."""
        cv2 = self.cv2
        x0, y0, w, h = VW - 216, VH - 76, 200, 54
        roi = view[y0:y0 + h, x0:x0 + w]
        roi[:] = (roi * 0.3).astype(np.uint8)
        color = (90, 90, 255) if self.capturing else (150, 230, 90)
        cv2.rectangle(view, (x0, y0), (x0 + w - 1, y0 + h - 1), tuple(int(c * 0.6) for c in color), 1)
        draw_text(view, "LIP SIGNAL", (x0 + 6, y0 + 13), 0.33, (190, 190, 190), 1)
        pts = [(x0 + int((t - now + seconds) / seconds * (w - 8)) + 4,
                y0 + h - 5 - int(min(1.0, max(0.0, m / 0.6)) * (h - 22)))
               for t, m in self.mar_hist if now - t <= seconds]
        if len(pts) >= 2:
            cv2.polylines(view, [np.array(pts, np.int32)], False, color, 2 if self.capturing else 1, cv2.LINE_AA)

    def _draw_scan_line(self, view, now):
        """A line sweeping down the face while it listens."""
        cv2 = self.cv2
        xs, ys = self.mesh[:, 0], self.mesh[:, 1]
        x0, x1 = int(max(0, VW - xs.max() - 12)), int(min(VW - 1, VW - xs.min() + 12))
        y0, y1 = int(max(0, ys.min() - 12)), int(min(VH - 1, ys.max() + 12))
        if y1 - y0 < 10 or x1 - x0 < 10:
            return
        y = int(y0 + ((now * 0.8) % 1.0) * (y1 - y0))
        for k, alpha in enumerate((0.5, 0.3, 0.15)):
            yy = y - 3 * k
            if y0 <= yy < y1:
                row = view[yy, x0:x1].astype(np.float32)
                view[yy, x0:x1] = (row * (1 - alpha) + np.array((90, 90, 255)) * alpha).astype(np.uint8)
        cv2.line(view, (x0, y), (x1, y), (120, 120, 255), 1, cv2.LINE_AA)

    def _round_video_corners(self, canvas):
        r = VIDEO_RADIUS
        if getattr(self, "_corner", None) is None:
            yy, xx = np.mgrid[0:r, 0:r]
            self._corner = (xx - r + 0.5) ** 2 + (yy - r + 0.5) ** 2 > r * r      # top-left, outside the arc
        c = self._corner
        canvas[VY:VY + r, VX:VX + r][c] = ui.BG
        canvas[VY:VY + r, VX + VW - r:VX + VW][c[:, ::-1]] = ui.BG
        canvas[VY + VH - r:VY + VH, VX:VX + r][c[::-1]] = ui.BG
        canvas[VY + VH - r:VY + VH, VX + VW - r:VX + VW][c[::-1, ::-1]] = ui.BG

    def _draw_emotion_tag(self, view):
        """Emotion above the face's top-left corner, pain expression below it (HUD style)."""
        r = self.reading
        xs, ys = self.mesh[:, 0], self.mesh[:, 1]
        x0 = int(np.clip(VW - (xs.max() + 12), 4, VW - 170))          # the view is mirrored
        y0, y1 = int(ys.min() - 12), int(ys.max() + 12)
        _, label, color, score = self._emotion_display(r)
        draw_text(view, f"{label.upper()} {score:.0%}", (x0, max(18, y0 - 8)), 0.5, color, 1)
        if r.pain_level != "none":
            lit = sum(r.pain >= s for s in ui.PAIN_STEPS)
            draw_text(view, f"PAIN EXPRESSION: {r.pain_level.upper()}", (x0, min(VH - 30, y1 + 20)), 0.45,
                      ui.PAIN_COLORS[max(0, lit - 1)], 1)

    def _render_status(self, lat):
        img, p = self._blank(STATUS), ui.Painter()
        img[:] = (34, 30, 28)
        cy = (STATUS[3] - STATUS[1]) // 2
        x = W - 14
        for key, action in reversed(KEYS):
            x = ui.keycap(img, p, x, cy, key, action) - 12
        mode = "PUSH-TO-TALK" if self.mode == "ptt" else "AUTO MODE"
        left = (f"{mode}   ·   {self.fps.fps:4.1f} FPS   ·   latency {'-' if lat is None else f'{lat:.0f} ms'}   ·   "
                f"{self.tracker_name or 'tracker'}   ·   speaker {self.args.speaker}")
        if ui.text_width(left, 12) > x - 30:                  # not enough room: the essentials only
            left = f"{mode}   ·   {self.fps.fps:4.1f} FPS"
        p.text(left, (16, cy), 12, ui.MUTED, "regular", "lm")
        return p.flush(img)

    # ----------------------------------------------------------------- mouse
    def on_mouse(self, event, x, y, flags, param):
        if event == self.cv2.EVENT_LBUTTONDOWN:
            self.clicks.append((x, y))

    def handle_clicks(self, now):
        clicks, self.clicks = self.clicks, []
        for x, y in clicks:
            if (self.alert or self.pain_alert) and y < HEADER[3]:
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
        self.speaker.cue("ready")
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


def update_deep_after_session(app):
    """The app kept new examples (1/2/3): retrain the deep model with them now that the camera is free."""
    n = getattr(app, "learned_count", 0)
    import importlib.util
    if not config.DEEP_MODEL or n < config.DEEP_RETRAIN_AFTER or importlib.util.find_spec("torch") is None:
        return
    print(f"\nThe app learned {n} new example(s) this session. Updating the deep model with them.")
    print("This takes a few minutes. Closing this window skips it; nothing breaks, it just keeps the old model.")
    from train_deep import train_deep
    train_deep(app.args.speaker)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["ptt", "auto"], default="ptt")
    ap.add_argument("--voice", choices=["cloned", "default"], default=config.VOICE_MODE)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--speaker", default=config.SPEAKER_ID)
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--model", help="use this model file instead of the speaker's or the base model")
    ap.add_argument("--recognizer", choices=["auto", "neural", "deep", "templates"], default="auto",
                    help="auto: enrolled/trained neural model if it knows this speaker, else the deep model "
                         "trained on their clips (train_deep.py) with the calibration templates, else the templates")
    ap.add_argument("--focus", type=int, metavar="N",
                    help="start with only your N most reliable phrases (press F in the app to switch)")
    args = ap.parse_args(argv)
    log_path, _ = start_session_log("app")
    try:
        app = App(args)
        if args.focus:
            app.toggle_focus(args.focus)
        app.run()
        update_deep_after_session(app)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"\nSomething went wrong. The details are saved in:\n  {log_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
