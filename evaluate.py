"""Evaluate a trained model and make charts for slides and the poster.

    python evaluate.py --split test                         # session 3: new day, new light
    python evaluate.py --split val                          # + threshold table to tune CONF_ACCEPT
    python evaluate.py --split unseen_speaker --speaker s04 # needs: python train.py --holdout-speaker s04
    python evaluate.py --split test --llm                   # LLM benefit on medium-confidence clips

Personalisation gain (before vs after enrolment), on a session not used for enrolment:
    python evaluate.py --split unseen_speaker --speaker new01 --session 2
    python evaluate.py --split unseen_speaker --speaker new01 --session 2 --model models/lipreader_new01.pt

Outputs in results/: metrics.json, confusion_matrix.png, per_phrase_accuracy.png
(other splits get a suffix, e.g. confusion_matrix_unseen_s04.png).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import config
from src.dataset import clip_key, load_positions, make_splits, read_metadata, scan_clips, stack_features
from src.decision import top_k
from src.metrics import compute_metrics, threshold_table, top_confusions
from src.model import LipReaderRuntime, predict_proba
from src.utils import holdout_model_path, load_env, load_phrases


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "val", "unseen_speaker", "all"])
    ap.add_argument("--speaker", help="speaker for --split unseen_speaker")
    ap.add_argument("--session", type=int, help="only clips from this session")
    ap.add_argument("--model", help="model file (default depends on --split)")
    ap.add_argument("--llm", action="store_true", help="measure LLM re-ranking on medium-confidence clips")
    return ap.parse_args(argv)


def plot_confusion(cm, label_map, phrases, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(label_map)
    size = max(6, 0.32 * n + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    names = [f"{pid} {phrases[pid]['text']}" for pid in label_map]
    ax.set_xticks(range(n), label_map, rotation=90, fontsize=7)
    ax.set_yticks(range(n), names, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true phrase")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_per_phrase(per_phrase, phrases, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(per_phrase.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(items) + 1)))
    colors = ["#c0392b" if phrases[pid]["emergency"] else "#2e86c1" for pid, _ in items]
    ax.barh([f"{pid} {phrases[pid]['text']}" for pid, _ in items], [v for _, v in items], color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("top-1 accuracy (red = emergency phrase)")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def group_accuracy(kept, y, pred, key_fn):
    groups = {}
    for c, yt, yp in zip(kept, y, pred):
        g = key_fn(c)
        if g in (None, ""):
            continue
        n, k = groups.get(g, (0, 0))
        groups[g] = (n + 1, k + int(yt == yp))
    return {g: {"n": n, "top1_accuracy": k / n} for g, (n, k) in sorted(groups.items())}


def main(argv=None):
    args = parse_args(argv)
    load_env()
    if args.split == "unseen_speaker" and not args.speaker:
        sys.exit("--split unseen_speaker needs --speaker, e.g. --speaker s04")

    if args.model:
        model_path = Path(args.model)
    elif args.split == "unseen_speaker" and holdout_model_path(args.speaker).exists():
        model_path = holdout_model_path(args.speaker)
    else:
        model_path = config.BASE_MODEL
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}. Train one first: python train.py")

    runtime = LipReaderRuntime(model_path)
    ckpt, label_map = runtime.ckpt, runtime.label_map
    phrases = load_phrases(phrase_set="full")
    index = {pid: i for i, pid in enumerate(label_map)}
    trained_on = set(ckpt.get("speakers", []))
    clips = scan_clips(phrase_ids=set(label_map))

    if args.split == "unseen_speaker":
        eval_clips = [c for c in clips if c.speaker == args.speaker]
        if args.speaker in trained_on:
            print(f"WARNING: {args.speaker} was in this model's training data, so this is not an "
                  f"unseen-speaker test. Train one without them: python train.py --holdout-speaker {args.speaker}")
        enrolled = ckpt.get("enrolled", {})
        if enrolled.get("speaker") == args.speaker and args.session in (None, *enrolled.get("sessions", [])):
            print("WARNING: these clips were used for enrolment; pass --session with a different session.")
        split_key = f"unseen_{args.speaker}"
    elif args.split == "all":
        eval_clips, split_key = clips, "all"
    else:
        s = ckpt.get("split", {})
        splits = make_splits(clips, s.get("holdout_speaker"), tuple(s.get("test_sessions", config.TEST_SESSIONS)),
                             s.get("val_fraction", config.VAL_FRACTION), s.get("seed", config.SEED),
                             s.get("max_reps"))
        eval_clips, split_key = splits[args.split], args.split
    if args.session is not None:
        eval_clips = [c for c in eval_clips if c.session == args.session]
        split_key += f"_session{args.session}"

    positions, kept = load_positions(eval_clips)
    if not kept:
        sys.exit(f"No clips to evaluate for split {split_key!r}. "
                 "(The test split is session 3; record it with collect_data.py --session 3.)")
    y = np.array([index[c.phrase_id] for c in kept])
    probs = predict_proba(runtime.model, stack_features(positions, runtime.use_velocity))
    m = compute_metrics(y, probs, label_map, phrases)
    pred = probs.argmax(1)

    # Processing latency per clip: features + model, one clip at a time, as in the app.
    times = []
    for c in kept[:60]:
        t0 = time.perf_counter()
        runtime.predict_clip(np.load(c.path))
        times.append((time.perf_counter() - t0) * 1000)
    m["processing_latency_ms_median"] = float(np.median(times))

    meta = read_metadata()
    m["per_speaker"] = group_accuracy(kept, y, pred, lambda c: c.speaker)
    m["per_lighting"] = group_accuracy(kept, y, pred, lambda c: meta.get(clip_key(c), {}).get("lighting"))
    m["per_session"] = group_accuracy(kept, y, pred, lambda c: c.session)
    m["threshold_table"] = threshold_table(y, probs)
    m["model"] = str(model_path)
    m["label_map"] = label_map

    # --- LLM benefit on medium-confidence predictions -----------------------
    if args.llm:
        from src.llm_corrector import LLMCorrector
        llm = LLMCorrector()
        if not llm.available:
            print(f"LLM not available: {llm.unavailable_reason}")
        else:
            conf = probs.max(1)
            medium = np.where((conf >= config.CONF_MIN) & (conf < config.CONF_ACCEPT))[0]
            print(f"Asking {llm.provider} ({llm.model}) about {len(medium)} medium-confidence clips "
                  "(no conversation context, so this is a lower bound)...")
            before = after = fallbacks = 0
            for i in medium:
                cands = top_k(probs[i], label_map, 3)
                choice, _, source = llm.choose(cands, [], phrases)
                before += cands[0][0] == label_map[y[i]]
                after += choice == label_map[y[i]]
                fallbacks += source == "fallback"
            if len(medium):
                m["llm"] = {"provider": llm.provider, "model": llm.model, "n_medium": int(len(medium)),
                            "top1_without_llm": before / len(medium), "top1_with_llm": after / len(medium),
                            "fallbacks": fallbacks}
            llm.close()

    # --- Save -------------------------------------------------------------------
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = config.RESULTS_DIR / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    all_metrics[split_key] = m
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    suffix = "" if split_key == "test" else f"_{split_key}"
    title = {"test": "Test set (session 3)", "val": "Validation set"}.get(split_key, split_key)
    plot_confusion(m["confusion_matrix"], label_map, phrases,
                   config.RESULTS_DIR / f"confusion_matrix{suffix}.png", f"Confusion matrix - {title}")
    plot_per_phrase(m["per_phrase_accuracy"], phrases,
                    config.RESULTS_DIR / f"per_phrase_accuracy{suffix}.png", f"Per-phrase accuracy - {title}")

    # --- Report -----------------------------------------------------------------
    print(f"\n{title}: {m['n']} clips, model {model_path.name}")
    print(f"  top-1 accuracy        {m['top1_accuracy']:.3f}")
    print(f"  top-3 accuracy        {m['top3_accuracy']:.3f}")
    print(f"  macro F1              {m['macro_f1']:.3f}")
    if m["emergency_recall"] is not None:
        print(f"  emergency recall      {m['emergency_recall']:.3f}   "
              f"(false alarms on normal phrases {m['emergency_false_alarm_rate']:.3f})")
    print(f"  processing latency    {m['processing_latency_ms_median']:.1f} ms per clip (median; "
          "end-to-end latency is shown live in app.py)")
    if len(m["per_speaker"]) > 1:
        print("  per speaker:   " + "  ".join(f"{k} {v['top1_accuracy']:.2f}" for k, v in m["per_speaker"].items()))
    if len(m["per_lighting"]) > 1:
        print("  per lighting:  " + "  ".join(f"{k} {v['top1_accuracy']:.2f}" for k, v in m["per_lighting"].items()))
    if "llm" in m:
        l = m["llm"]
        print(f"  LLM on {l['n_medium']} medium clips: top-1 {l['top1_without_llm']:.3f} -> "
              f"{l['top1_with_llm']:.3f} ({l['fallbacks']} fallbacks)")

    worst = sorted(m["per_phrase_accuracy"].items(), key=lambda kv: kv[1])[:5]
    print("\n  weakest phrases: " + ", ".join(f"{pid} {phrases[pid]['text']!r} {acc:.2f}" for pid, acc in worst))
    conf_pairs = top_confusions(m["confusion_matrix"], label_map)
    if conf_pairs:
        print("  most confused (look-alike candidates; consider renaming one of each pair):")
        for n, a, b in conf_pairs[:6]:
            print(f"    {n:3d} x  {phrases[a]['text']!r} -> {phrases[b]['text']!r}")

    print("\n  threshold   spoken directly   accuracy when spoken")
    for row in m["threshold_table"]:
        acc = "-" if row["accuracy_when_spoken"] is None else f"{row['accuracy_when_spoken']:.3f}"
        mark = "  <- CONF_ACCEPT" if abs(row["threshold"] - config.CONF_ACCEPT) < 1e-9 else ""
        print(f"    {row['threshold']:.1f}         {row['coverage']:6.1%}            {acc}{mark}")
    if args.split == "val":
        good = [r for r in m["threshold_table"]
                if r["accuracy_when_spoken"] is not None and r["accuracy_when_spoken"] >= 0.95]
        if good:
            print(f"  -> lowest threshold with >= 95% accuracy when spoken: {good[0]['threshold']:.1f}")
    print(f"\nSaved {metrics_path} and charts in {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
