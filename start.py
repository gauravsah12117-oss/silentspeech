"""Get SilentSpeech working on this laptop: one command, about 10 minutes.

    python start.py                  (Windows: double-click start.bat)
    python start.py --speaker s02    (set up another person)
    python start.py --recalibrate    (record another round of calibration clips)

It walks you through:
  1. a camera, face and sound check
  2. calibration: silently mouth each phrase 4 times (hands-free, ~6 s each)
  3. a check of the calibration, with a chance to add clips for weak phrases
  4. the voice: record each phrase aloud in your own voice, or use the computer voice
  5. the live app
Finished steps are skipped, so you can run it again any time.

No training is needed: the "instant templates" recogniser compares what you
mouth with your own calibration clips. For the competition dataset and the
neural model, see README sections 13-15.
"""
import argparse
import re
import sys
import traceback
from datetime import datetime

import config
from src.utils import load_env, load_phrases

CAL_SESSION = 1
_LOG = None   # session log file (everything shown in the window, plus the answers)


class _Tee:
    """Writes to the console and to the session log."""

    def __init__(self, stream, log):
        self._stream, self._log = stream, log

    def write(self, text):
        n = self._stream.write(text)
        try:
            self._log.write(text)
            self._log.flush()
        except (OSError, ValueError):
            pass
        return n

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def start_log():
    global _LOG
    folder = config.WORK_DIR / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"start-{datetime.now():%Y%m%d-%H%M%S}.log"
    _LOG = open(path, "a", encoding="utf-8", errors="replace")
    sys.stdout, sys.stderr = _Tee(sys.stdout, _LOG), _Tee(sys.stderr, _LOG)
    return path


def _log_answer(prompt, answer):
    if _LOG:
        _LOG.write(f"{prompt.strip()} -> {answer!r}\n")
        _LOG.flush()


def ask(prompt, default=""):
    text = f"{prompt} [{default}]: " if default else f"{prompt}: "
    try:
        answer = input(text).strip()
    except EOFError:
        answer = ""
    _log_answer(text, answer or default)
    return answer or default


def wait_enter(prompt):
    try:
        input(prompt)
    except EOFError:
        pass
    _log_answer(prompt, "Enter")


def yes(prompt, default=True):
    return ask(prompt + (" (Y/n)" if default else " (y/N)"), "y" if default else "n").lower().startswith("y")


def heading(n, title):
    print(f"\n=== Step {n} of 5: {title} ===")


def step_check(camera):
    import check_setup as cs
    heading(1, "camera, face and sound check")
    if not cs.check_libraries():
        sys.exit("\nSome libraries are missing. Run: pip install -r requirements.txt")
    while True:
        cs.RESULTS.clear()
        cam = cs.check_camera(camera, show=True)
        if cam is None:
            sys.exit("\nThe webcam did not work - see the message above, then run start.py again.")
        if cam["face_rate"] >= 0.5:
            break
        print("\nYour face wasn't found clearly." + (" The picture is very dark." if cam["brightness"] < 50 else ""))
        print("Sit 40-70 cm from the screen, look at the camera, and put a light in FRONT of you.")
        if not yes("Try again?", True):
            sys.exit("Fix the camera view, then run start.py again.")
    cs.check_audio(play=False)


def choose_phrases(all_phrases):
    print("\nWhich phrases do you want to use?")
    minutes = lambda n: round(n * config.QUICKSTART_REPS * 6.3 / 60)
    print(f"  1) {len(config.QUICKSTART_PHRASES)} demo phrases       "
          f"about {minutes(len(config.QUICKSTART_PHRASES))} minutes of calibration")
    print(f"  2) all 30 MVP phrases    about {minutes(30)} minutes")
    print(f"  3) all 50 phrases        about {minutes(50)} minutes")
    choice = ask("Choose 1, 2 or 3", "1")
    if choice == "3":
        return dict(all_phrases)
    if choice == "2":
        return {pid: p for pid, p in all_phrases.items() if p.get("set", "mvp") == "mvp"}
    return {pid: all_phrases[pid] for pid in config.QUICKSTART_PHRASES}


def clip_counts(speaker, phrase_ids):
    from src.dataset import scan_clips
    counts = {pid: 0 for pid in phrase_ids}
    for c in scan_clips(phrase_ids=set(phrase_ids), speakers={speaker}):
        counts[c.phrase_id] += 1
    return counts


def record(speaker, session, reps, phrases, camera, title):
    from collect_data import record_phrases
    print(f"\nA camera window opens for {len(phrases)} phrase(s), {reps} clips each. For every clip:")
    print("  - keep your lips gently closed,")
    print("  - when the red REC dot appears, SILENTLY mouth the phrase shown (no sound),")
    print("  - close your lips again. It moves on by itself.")
    print("  Keys in that window: P pause, R redo the last clip, N skip a phrase, Q stop (progress is kept).")
    wait_enter("Press Enter to open the camera window...")
    record_phrases(speaker, session, reps, phrases, camera_index=camera, title=title,
                   auto_advance=True, order="round", countdown=config.QUICKSTART_COUNTDOWN)


def build_templates(speaker, phrase_ids):
    from src.templates import for_speaker

    def progress(i, n, secs):
        print(f"\r  checking clips {i}/{n}", end="", flush=True)

    m = for_speaker(speaker, phrase_ids, rebuild=True, progress=progress)
    print()
    return m


def report(m, phrases):
    acc = m.report.get("loo_accuracy")
    if acc is None:
        print("Not enough clips to check the calibration yet (each phrase needs 2 or more).")
        return []
    print(f"\nCalibration check: {acc:.0%} of your clips were recognised correctly from the others.")
    print("(This is measured on clips from the same sitting, so live use is usually a little lower.)")
    for name, reason in m.report.get("dropped", []):
        pid = name.split()[0]
        print(f"  left out one clip of \"{phrases[pid]['text']}\": {reason}")
    if acc < 0.7:
        print("Tips for a better score:")
        print("  - mouth each phrase the SAME way every time: clearly, at your normal speed")
        print("  - start right when the red REC dot appears and finish before the red bar ends")
        print("  - keep your head still and your face well lit from the front")
    weak = [pid for pid, a in m.report.get("per_phrase", {}).items() if a < 0.67]
    for pair, n in list(m.report.get("confusions", {}).items())[:3]:
        a, b = pair.split(">")
        print(f"  mixed up {n}x: \"{phrases[a]['text']}\" looked like \"{phrases[b]['text']}\"")
    if weak:
        print("Weakest phrases: " + ", ".join(f"\"{phrases[p]['text']}\"" for p in weak))
    return weak


def step_voice(speaker, phrase_ids, phrases):
    from src.speaker import render_tts_cache, voice_dir
    heading(4, "the voice")
    missing = [pid for pid in phrase_ids if not (voice_dir(speaker) / f"{pid}.wav").exists()]
    if missing:
        print("How should SilentSpeech speak?")
        print(f"  1) Record your own voice now - say each of the {len(missing)} phrase(s) once (about "
              f"{max(1, round(len(missing) * 8 / 60))} min)")
        print("  2) Use the computer voice for now (later: python record_voice.py)")
        if ask("Choose 1 or 2", "2") == "1":
            from record_voice import record_phrase_voices
            n = record_phrase_voices(speaker, {pid: phrases[pid] for pid in missing})
            print(f"Saved {n} recording(s).")
    else:
        print("Your own voice clips are ready.")
    rest = {pid: phrases[pid] for pid in phrase_ids if not (voice_dir(speaker) / f"{pid}.wav").exists()}
    if rest:
        print("Preparing the computer voice for phrases without a recording...")
        try:
            render_tts_cache(rest, progress=lambda i, n: print(f"\r  {i}/{n}", end="", flush=True))
            print()
        except Exception as e:  # the app can still speak live, just more slowly
            print(f"\n  (skipped: {e})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", help="short ID for the person using it, e.g. s01")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--recalibrate", action="store_true", help="record another round of calibration clips")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args(argv)
    load_env()
    all_phrases = load_phrases(phrase_set="full")

    print("SilentSpeech quick start. Answer the questions here; press Enter to accept [the default].")
    speaker = args.speaker or ask("\nShort ID for the person using it (e.g. s01)", config.SPEAKER_ID)
    speaker = re.sub(r"[^A-Za-z0-9_-]", "", speaker) or config.SPEAKER_ID

    if not args.skip_check:
        step_check(args.camera)

    # --- Step 2: calibration ------------------------------------------------
    heading(2, "calibration")
    counts = clip_counts(speaker, list(all_phrases))
    ready = [pid for pid, n in counts.items() if n >= config.QUICKSTART_REPS]
    recorded_now = False
    if ready and not args.recalibrate and yes(f"Found calibration for {len(ready)} phrase(s) for {speaker}. Use it?", True):
        phrases = {pid: all_phrases[pid] for pid in all_phrases if pid in ready}
    else:
        recorded_now = True
        phrases = choose_phrases(all_phrases)
        reps = config.QUICKSTART_REPS
        if args.recalibrate:   # another round on top of what is there
            reps = max(counts[pid] for pid in phrases) + config.QUICKSTART_REPS
        record(speaker, CAL_SESSION, reps, phrases, args.camera, "Calibration")
    counts = clip_counts(speaker, list(phrases))
    phrases = {pid: p for pid, p in phrases.items() if counts[pid] >= 2}
    if len(phrases) < 2:
        sys.exit("\nAt least 2 phrases need 2+ clips each. Run start.py again to finish calibration.")

    # --- Step 3: check -----------------------------------------------------
    heading(3, "checking the calibration")
    m = build_templates(speaker, list(phrases))
    weak = report(m, all_phrases)
    if weak and yes(f"Record 2 more clips of the {len(weak)} weakest phrase(s)? It usually helps.", recorded_now):
        extra = max(counts[pid] for pid in weak) + 2
        record(speaker, CAL_SESSION, extra, {pid: all_phrases[pid] for pid in weak}, args.camera, "Extra clips")
        m = build_templates(speaker, list(phrases))
        report(m, all_phrases)

    step_voice(speaker, m.label_map, all_phrases)

    # --- Step 5: go --------------------------------------------------------
    heading(5, "the live app")
    from src.llm_corrector import LLMCorrector
    llm = LLMCorrector()
    if not llm.available:
        print("(Optional) For context-aware correction, put an API key in .env - see README section 17.")
    llm.close()
    print("In the app: press SPACE, then silently mouth a phrase within 2.5 seconds.")
    print("1/2/3 pick an option when it is unsure, ENTER speaks the whole sentence, Q quits.")
    wait_enter("Press Enter to start...")
    from app import App
    App(argparse.Namespace(mode="ptt", voice="cloned", no_llm=False, speaker=speaker, camera=args.camera,
                           model=None, recognizer="auto")).run()
    print("\nThe live app is closed. Run start.bat again any time.")


if __name__ == "__main__":
    log_path = start_log()
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception:
        traceback.print_exc()
        print(f"\nSomething went wrong. The details are saved in:\n  {log_path}\n"
              "Send that file to whoever is helping you.")
        sys.exit(1)
