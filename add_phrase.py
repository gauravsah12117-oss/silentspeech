"""Add your own phrase - any words, any language - and teach it in a minute.

    python add_phrase.py                          (Windows: double-click add_phrase.bat)
    python add_phrase.py "Turn on the news"
    python add_phrase.py "Call Ram" --emergency   (raise the caregiver alert for it)
    python add_phrase.py --list
    python add_phrase.py --remove C03

It saves the phrase (ids C01, C02, ...) to phrases/custom.json, then records
4 silent clips of it hands-free (like calibration) and, if you want, your
voice saying it once. From then on it is recognised, shown on the tap board,
and spoken - in your own voice if you recorded it, otherwise by the computer
voice (Hindi/Nepali text needs your own recording).

Custom phrases are private: phrases/custom.json is not uploaded to GitHub.
"""
import argparse
import re
import sys

import config
from src.utils import load_custom_phrases, load_env, load_phrases, save_custom_phrases

MAX_LEN = 60


def ask(prompt, default=""):
    try:
        answer = input(f"{prompt} [{default}]: " if default else f"{prompt}: ").strip()
    except EOFError:
        answer = ""
    return answer or default


def next_id(items):
    numbers = [int(m.group(1)) for p in items if (m := re.fullmatch(r"C(\d+)", p["id"]))]
    return f"C{max(numbers, default=0) + 1:02d}"


def add(text, emergency=False):
    """Validate and save a new phrase. Returns the phrase dict (or exits with a message)."""
    text = " ".join(text.split())
    if not text:
        sys.exit("The phrase is empty.")
    if len(text) > MAX_LEN:
        sys.exit(f"Please keep it to {MAX_LEN} characters or less - short phrases are easier to lip-read.")
    existing = load_phrases(phrase_set="full")
    for p in existing.values():
        if p["text"].casefold() == text.casefold():
            sys.exit(f"\"{text}\" already exists ({p['id']}).")
    items = load_custom_phrases()
    phrase = {"id": next_id(items), "text": text, "category": "custom", "emergency": bool(emergency),
              "set": "custom"}
    items.append(phrase)
    save_custom_phrases(items)
    print(f"Saved {phrase['id']}: \"{text}\"" + ("  (emergency)" if emergency else ""))
    return phrase


def teach(phrase, speaker, camera=None, voice=None):
    """Record calibration clips (and optionally the voice) for one phrase, then rebuild."""
    from collect_data import record_phrases
    from src.templates import for_speaker

    print(f"\nA camera window opens. Silently mouth \"{phrase['text']}\" when the red REC dot appears,")
    print(f"{config.QUICKSTART_REPS} times - the same way each time. It moves on by itself.")
    ask("Press Enter to start")
    record_phrases(speaker, 1, config.QUICKSTART_REPS, {phrase["id"]: phrase}, camera_index=camera,
                   title="New phrase", auto_advance=True, countdown=config.QUICKSTART_COUNTDOWN)
    if voice is None:
        voice = ask("Record your own voice saying it? (y/n)", "y").lower().startswith("y")
    if voice:
        from record_voice import record_phrase_voices
        record_phrase_voices(speaker, {phrase["id"]: phrase})
    m = for_speaker(speaker, list(load_phrases(phrase_set="full")), rebuild=True)
    if m is not None and phrase["id"] in m.label_map:
        acc = m.report.get("per_phrase", {}).get(phrase["id"])
        print(f"\nDone. \"{phrase['text']}\" is now part of your {len(m.label_map)} phrases"
              + (f" (recognised {acc:.0%} of the time in the self-test)." if acc is not None else "."))
        import importlib.util
        if config.DEEP_MODEL and importlib.util.find_spec("torch") is not None and ask(
                "Retrain the deep model so it knows the new phrase too? A few minutes; until then the app "
                "uses the templates for all phrases. (y/n)", "y").lower().startswith("y"):
            from train_deep import train_deep
            train_deep(speaker)
    else:
        print("\nNot enough clips were recorded for it yet; run add_phrase.py again later "
              "or train_more.bat to record more of all phrases.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="the phrase, e.g. \"Turn on the news\"")
    ap.add_argument("--emergency", action="store_true", help="raise the caregiver alert when it is recognised")
    ap.add_argument("--speaker", default=config.SPEAKER_ID)
    ap.add_argument("--camera", type=int)
    ap.add_argument("--no-record", action="store_true", help="only save the phrase")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--remove", metavar="ID")
    args = ap.parse_args(argv)
    load_env()

    if args.list:
        items = load_custom_phrases()
        print("\n".join(f"{p['id']}  {p['text']}" + ("  (emergency)" if p["emergency"] else "") for p in items)
              or "No custom phrases yet.")
        return
    if args.remove:
        items = load_custom_phrases()
        keep = [p for p in items if p["id"] != args.remove.upper()]
        if len(keep) == len(items):
            sys.exit(f"No custom phrase {args.remove}.")
        save_custom_phrases(keep)
        print(f"Removed {args.remove.upper()}. (Its recorded clips stay on disk but are no longer used.)")
        return

    text = args.text or ask("Type the phrase to add (any language, up to 60 characters)")
    emergency = args.emergency or (not args.text and ask("Is it an emergency phrase that should alert a "
                                                         "caregiver? (y/n)", "n").lower().startswith("y"))
    phrase = add(text, emergency)
    if not args.no_record:
        teach(phrase, args.speaker, args.camera)


if __name__ == "__main__":
    main()
