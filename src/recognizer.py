"""Choose how to recognise a speaker: the neural model or instant templates.

auto (default):
  1. the speaker's enrolled neural model (models/lipreader_<speaker>.pt)
  2. the base neural model, if this speaker was in its training data
  3. the speaker's own clips: the deep model trained on them (train_deep.py,
     models/deep_<speaker>.pt) with the instant templates as a safety check,
     or the instant templates alone until a deep model is trained
  4. the base neural model for someone it has never seen (less accurate)
"""
from pathlib import Path

import config
from src.utils import personal_model_path


def with_deep(templates, speaker, model_dir=None, quiet=False):
    """Put the speaker's deep model in front of these templates when it fits them.

    Returns (recogniser, note). The deep model is used only if it was trained on
    exactly these phrases and beat the templates on held-out clips.
    """
    if templates is None or getattr(templates, "kind", "") != "templates":
        return templates, ""
    try:
        from src.deep import DeepRecognizer, load_for_speaker
    except ImportError:  # no torch: the templates alone still work
        return templates, "the deep model needs PyTorch"
    model, why = load_for_speaker(speaker, templates.label_map, model_dir)
    if model is None:
        if not quiet:
            print(f"Deep model not used: {why}")
        return templates, why
    return DeepRecognizer(model, templates), "deep model trained on this speaker's own clips"


def load_recognizer(speaker, phrases, kind="auto", model=None, progress=None):
    """Return (recognizer or None, note). phrases: {id: phrase} in display order."""
    if model:
        from src.model import LipReaderRuntime
        return LipReaderRuntime(Path(model)), "model given with --model"

    neural_ok = kind in ("auto", "neural")
    if neural_ok:
        try:
            from src.model import LipReaderRuntime
        except ImportError:  # torch missing: templates still work
            neural_ok = False
    if neural_ok and personal_model_path(speaker).exists():
        return LipReaderRuntime(personal_model_path(speaker)), "enrolled model"
    base = None
    if neural_ok and config.BASE_MODEL.exists():
        base = LipReaderRuntime(config.BASE_MODEL)
        if speaker in base.ckpt.get("speakers", []) or kind == "neural":
            return base, "trained on this speaker" if kind == "auto" else "chosen with --recognizer neural"

    if kind in ("auto", "templates", "deep"):
        from src.templates import for_speaker
        m = for_speaker(speaker, list(phrases), progress=progress)
        if m is not None:
            if kind != "templates":
                rec, why = with_deep(m, speaker)
                if rec is not m:
                    return rec, why
            return m, "calibrated with this speaker's own clips"
    if base is not None:
        return base, "WARNING: this speaker is new to the model - run start.py to calibrate"
    return None, "no model and no calibration clips yet - run start.py"
