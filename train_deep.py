"""Deep training: teach a neural-network ensemble this person's own lip movements.

    python train_deep.py                  (Windows: double-click train_deep.bat)
    python train_deep.py --speaker s02

It learns from every clip recorded for the speaker: calibration (start.py),
train_more, and the examples the live app kept when you pressed 1/2/3. On a
slow laptop it takes a few minutes. start.py and train_more run it for you,
and the app runs it when it closes after learning new examples.

Before it is used, it is checked honestly: 5-fold cross-validation, so every
clip is judged by networks that never saw it, and the instant templates are
scored on exactly the same clips. The app uses the deep model only if it wins.
"""
import argparse
import sys
import time

import config
from src.utils import load_phrases, start_session_log


def _duration(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60} min {seconds % 60:02d} s" if seconds >= 60 else f"{seconds} s"


class ProgressBar:
    """One console line per stage; the training stage shows the time left."""

    def __init__(self, width=24):
        self.width, self.stage, self.t0 = width, None, 0.0

    def __call__(self, stage, done, total):
        now = time.time()
        if stage != self.stage:
            if self.stage is not None:
                print()
            self.stage, self.t0 = stage, now
        frac = done / max(total, 1)
        n = int(self.width * frac)
        eta = ""
        if stage.startswith("training") and 0 < done < total:
            eta = f"  about {_duration((now - self.t0) / done * (total - done))} left"
        print(f"\r  {stage:<40s} [{'#' * n}{'-' * (self.width - n)}] {frac:4.0%}{eta}      ", end="", flush=True)

    def close(self):
        if self.stage is not None:
            print()


def summarize(model, phrases):
    r = model.report
    text = (lambda pid: phrases[pid]["text"] if pid in phrases else pid)
    print(f"\nDeep model for {r.get('speaker', '?')}: {len(model.label_map)} phrases, {r['n_clips']} clips, "
          f"{len(model.nets)} networks (trained in {_duration(r['train_seconds'])}).")
    for name in r.get("left_out", []):
        pid = name.split()[0]
        print(f"  left out one clip of \"{text(pid)}\": the lips barely moved")
    if r.get("cv_accuracy") is None:
        return
    print(f"Honest check - {r['folds']}-fold cross-validation, every clip judged by networks that never saw it:")
    print(f"  deep model          {r['cv_accuracy']:.0%}")
    if r.get("templates_cv_accuracy") is not None:
        print(f"  instant templates   {r['templates_cv_accuracy']:.0%}   (same clips, same folds)")
    if r.get("spoken_directly_right") is not None:
        print(f"When it is sure ({config.CONF_ACCEPT:.0%}+) it speaks straight away: {r['spoken_directly']:.0%} of "
              f"clips, and {r['spoken_directly_right']:.0%} of those were right. Otherwise it asks or offers options.")
    weak = sorted((a, pid) for pid, a in r.get("per_phrase", {}).items() if a < 0.67)
    if weak:
        print("Weakest phrases: " + ", ".join(f"\"{text(pid)}\" {a:.0%}" for a, pid in weak))
    for pair, n in list(r.get("confusions", {}).items())[:3]:
        a, b = pair.split(">")
        print(f"  mixed up {n}x: \"{text(a)}\" looked like \"{text(b)}\"")
    if r.get("better_than_templates", True):
        print("The live app now uses the deep model, with the templates as a safety check.")
    else:
        print("It did not beat the templates on your clips, so the app keeps using the templates.\n"
              "More clips usually changes that: train_more.bat records 4 more of each phrase.")


def train_deep(speaker, phrases=None, **kw):
    """Train, save and report. Returns the model, or None (with the reason printed)."""
    try:
        from src import deep
    except ImportError:
        print("The deep model needs PyTorch: venv\\Scripts\\pip install torch  (the templates work without it)")
        return None
    phrases = phrases or load_phrases(phrase_set="full")
    print(f"\nDeep training for {speaker}: {deep.N_NETS} neural networks learn your own lip movements.")
    bar = ProgressBar()
    try:
        model = deep.train_for_speaker(speaker, list(phrases), progress=bar, **kw)
    except ValueError as e:
        bar.close()
        print(f"Deep training skipped: {e}.")
        return None
    bar.close()
    if model is None:
        print("Deep training needs your calibration clips first: run start.bat.")
        return None
    summarize(model, phrases)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", default=config.SPEAKER_ID)
    ap.add_argument("--nets", type=int, help="networks in the ensemble (default from src/deep.py)")
    ap.add_argument("--epochs", type=int, help="training passes per network (default from src/deep.py)")
    args = ap.parse_args(argv)
    kw = {k: v for k, v in (("n_nets", args.nets), ("epochs", args.epochs)) if v}
    model = train_deep(args.speaker, **kw)
    sys.exit(0 if model is not None else 1)


if __name__ == "__main__":
    log_path, _ = start_session_log("train_deep")
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. The previous deep model (if any) is unchanged.")
