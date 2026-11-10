"""Get SilentSpeech working on this laptop: one command, about 10 minutes.

    python start.py                  (Windows: double-click start.bat)
    python start.py --speaker s02    (set up another person)
    python start.py --recalibrate    (record another round of calibration clips)

It walks you through:
  1. a camera, face and sound check
  2. calibration: silently mouth each phrase 4 times (hands-free, ~6 s each)
  3. a check of the calibration, with a chance to add clips for weak phrases,
     then deep training: neural networks learn your lip movements (a few minutes)
  4. the voice: record each phrase aloud in your own voice, or use the computer voice
  5. the live app
Finished steps are skipped, so you can run it again any time.

The "instant templates" recogniser works straight after calibration: it
compares what you mouth with your own calibration clips. The deep model
(train_deep.py) then learns from the same clips and takes over when it is
measurably better. For the competition dataset and the big neural model, see
README sections 13-15.
"""
import argparse
import re
import sys
import traceback

import config
from src.utils import load_env, load_phrases, start_session_log

CAL_SESSION = 1
TRAIN_SESSION = 2      # extra examples from --train-more
_LOG = None   # session log file (everything shown in the window, plus the answers)


def start_log():
    global _LOG
    path, _LOG = start_session_log("start")
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
    each = "more clips" if isinstance(reps, dict) else "clips"
    n = config.TRAIN_MORE_REPS if isinstance(reps, dict) else reps
    print(f"\nA camera window opens for {len(phrases)} phrase(s), {n} {each} each. For every clip:")
    print("  - keep your lips gently closed,")
    print("  - when the red REC dot appears, SILENTLY mouth the phrase shown (no sound),")
    print("  - close your lips again. It moves on by itself.")
    print("  Keys in that window: P pause, R redo the last clip, N skip a phrase,")
    print("  Q (or closing the window) stops - everything recorded so far is kept.")
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


def step_deep(speaker, all_phrases, ask_first=True):
    """Deep training on all of the speaker's clips (skipped when the model is already up to date)."""
    if not config.DEEP_MODEL:
        return None
    try:
        from src import deep
    except ImportError:
        print("(Deep model skipped: PyTorch is not installed. The templates work without it.)")
        return None
    current = deep.up_to_date(speaker, list(all_phrases))
    if current is not None:
        acc = current.report.get("cv_accuracy")
        print("\nYour deep model is up to date" + (f" ({acc:.0%} on clips it never saw)." if acc else "."))
        return current
    minutes = max(1, round(deep.estimate_seconds(speaker, list(all_phrases)) / 60))
    if ask_first and not yes(f"\nTrain the deep model on your clips now? Neural networks learn your lip "
                             f"movements; about {minutes} min on this laptop.", True):
        print("Skipped. train_deep.bat does it any time.")
        return None
    from train_deep import train_deep
    return train_deep(speaker, all_phrases)


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


def deep_accuracy(speaker):
    """Cross-validated accuracy of the speaker's current deep model, or None."""
    try:
        from src.deep import DeepModel, model_path
        model, _ = DeepModel.load(model_path(speaker))
    except Exception:  # no torch, or no model yet
        return None
    return None if model is None else model.report.get("cv_accuracy")


def train_more(args, speaker, all_phrases):
    """Record more examples of every calibrated phrase and show the gain."""
    from collect_data import existing_reps
    counts = clip_counts(speaker, list(all_phrases))
    phrases = {pid: all_phrases[pid] for pid in all_phrases if counts[pid] >= 2}
    if len(phrases) < 2:
        sys.exit(f"\n{speaker} has no calibration yet. Run start.bat first.")
    print(f"\n=== Train more: {len(phrases)} phrases ===")
    before = build_templates(speaker, list(phrases)).report.get("loo_accuracy") or 0.0
    deep_before = deep_accuracy(speaker)
    print(f"Right now your clips are recognised {before:.0%} of the time by the templates"
          + (f" and {deep_before:.0%} by the deep model" if deep_before is not None else "")
          + f" ({min(counts[p] for p in phrases)}-{max(counts[p] for p in phrases)} examples per phrase).")
    print("More examples is what makes it more accurate. Mouth each phrase clearly, the same way every time.")
    targets = {pid: len(existing_reps(speaker, pid, TRAIN_SESSION)) + config.TRAIN_MORE_REPS for pid in phrases}
    record(speaker, TRAIN_SESSION, targets, phrases, args.camera, "Training")
    m = build_templates(speaker, list(phrases))
    report(m, all_phrases)
    after = m.report.get("loo_accuracy") or 0.0
    print(f"\nTemplates on your clips: {before:.0%} before -> {after:.0%} now.")
    model = step_deep(speaker, all_phrases, ask_first=False)
    ranking = m.rank_phrases()
    if model is not None and model.report.get("better_than_templates"):
        if deep_before is not None:
            print(f"Deep model on clips it never saw: {deep_before:.0%} before -> "
                  f"{model.report['cv_accuracy']:.0%} now.")
        ranking = model.rank_phrases()
    n = min(config.FOCUS_PHRASES, len(ranking))
    print(f"Your {n} most reliable phrases (about {ranking[n - 1][1]:.0%} accurate together): "
          + ", ".join(f"\"{all_phrases[p]['text']}\"" for p, _ in ranking[:n]))
    print("In the live app, press F to use only those (and F again for all phrases).")
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", help="short ID for the person using it, e.g. s01")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--recalibrate", action="store_true", help="record another round of calibration clips")
    ap.add_argument("--train-more", action="store_true",
                    help=f"record {config.TRAIN_MORE_REPS} more examples of every phrase to raise accuracy")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args(argv)
    load_env()
    all_phrases = load_phrases(phrase_set="full")

    print("SilentSpeech quick start. Answer the questions here; press Enter to accept [the default].")
    speaker = args.speaker or ask("\nShort ID for the person using it (e.g. s01)", config.SPEAKER_ID)
    speaker = re.sub(r"[^A-Za-z0-9_-]", "", speaker) or config.SPEAKER_ID

    if not args.skip_check:
        step_check(args.camera)

    if args.train_more:
        train_more(args, speaker, all_phrases)
        if yes("\nOpen the live app now?", True):
            launch_app(speaker, args.camera)
        return

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
        from collect_data import existing_reps
        extra = max(len(existing_reps(speaker, pid, CAL_SESSION)) for pid in weak) + 2
        record(speaker, CAL_SESSION, extra, {pid: all_phrases[pid] for pid in weak}, args.camera, "Extra clips")
        m = build_templates(speaker, list(phrases))
        report(m, all_phrases)
    step_deep(speaker, all_phrases)

    step_voice(speaker, m.label_map, all_phrases)

    # --- Step 5: go --------------------------------------------------------
    heading(5, "the live app")
    from src.llm_corrector import LLMCorrector
    llm = LLMCorrector()
    if not llm.available:
        print("(Optional) For context-aware correction, put an API key in .env - see README section 17.")
    llm.close()
    wait_enter("Press Enter to start...")
    launch_app(speaker, args.camera)


def launch_app(speaker, camera):
    print("In the app: press SPACE, then silently mouth a phrase within 2.5 seconds.")
    print("1/2/3 pick, correct or confirm a phrase (it learns from that), F focuses on your")
    print("most reliable phrases, ENTER speaks the whole sentence, Q quits.")
    from app import App, update_deep_after_session
    app = App(argparse.Namespace(mode="ptt", voice="cloned", no_llm=False, speaker=speaker, camera=camera,
                                 model=None, recognizer="auto"))
    app.run()
    update_deep_after_session(app)
    print("\nThe live app is closed. app.bat opens it again; train_more.bat makes it more accurate.")


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
