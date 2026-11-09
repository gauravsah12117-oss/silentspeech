"""Train the lip-reading model.

    python train.py                          # main model -> models/lipreader.pt
    python train.py --holdout-speaker s04    # leave s04 out -> models/lipreader_holdout_s04.pt
    python train.py --set full               # all 50 phrases

Experiments (each adds a row to results/experiments.csv; --tag names it):
    python train.py --no-velocity --tag no_velocity
    python train.py --no-augment  --tag no_augment
    python train.py --no-attention --tag no_attention
    python train.py --arch bigru       --tag bigru_only
    python train.py --arch transformer --tag transformer
    python train.py --learning-curve     # -> results/learning_curve.png

Experiment runs save to models/experiments/<tag>.pt so they never replace
the main model, unless you pass --out.

Splits are by session (see src/dataset.py): sessions 1-2 for train/val,
session 3 only for the test set.
"""
import argparse
import copy
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from src.dataset import LipDataset, load_positions, make_splits, scan_clips, stack_features
from src.metrics import compute_metrics
from src.model import (ARCHS, build_model, count_parameters, predict_proba,
                       save_checkpoint)
from src.utils import holdout_model_path, load_phrases, save_json, seed_everything


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="phrase_set", default=config.PHRASE_SET, choices=["mvp", "full"])
    ap.add_argument("--holdout-speaker", help="leave this speaker out entirely (unseen-speaker test)")
    ap.add_argument("--arch", default="conv_bigru", choices=ARCHS)
    ap.add_argument("--no-velocity", action="store_true", help="positions only (80 features)")
    ap.add_argument("--no-attention", action="store_true", help="mean pooling instead of attention")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--reps", type=int, help="max train+val clips per speaker and phrase")
    ap.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--tag", help="experiment name for results/experiments.csv")
    ap.add_argument("--out", help="where to save the model")
    ap.add_argument("--learning-curve", action="store_true",
                    help="train with increasing reps per phrase and plot test accuracy")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def train_model(train_pos, train_y, val_pos, val_y, model_config, augment_cfg, epochs,
                seed=config.SEED, quiet=False):
    """Train one model. Returns (model with best weights, history dict)."""
    seed_everything(seed)
    use_vel = model_config["use_velocity"]
    model = build_model(model_config)

    train_feats = stack_features(train_pos, use_vel)       # un-augmented, for normalisation stats
    flat = train_feats.reshape(-1, train_feats.shape[-1])
    model.set_normalisation(flat.mean(0), flat.std(0))

    loader = DataLoader(LipDataset(train_pos, train_y, use_vel, augment_cfg, seed),
                        batch_size=config.BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    val_x = torch.from_numpy(stack_features(val_pos, use_vel))
    val_t = torch.as_tensor(np.asarray(val_y, dtype=np.int64))
    has_val = len(val_pos) > 0

    opt = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                            weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=config.LR_FACTOR,
                                                       patience=config.LR_PATIENCE)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr")}
    best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
        tr_loss, tr_acc = loss_sum / total, correct / total

        if has_val:
            model.eval()
            with torch.no_grad():
                logits = model(val_x)
                va_loss = criterion(logits, val_t).item()
                va_acc = (logits.argmax(1) == val_t).float().mean().item()
        else:
            va_loss, va_acc = tr_loss, tr_acc
        sched.step(va_loss)
        lr = opt.param_groups[0]["lr"]
        for k, v in zip(hist, (tr_loss, tr_acc, va_loss, va_acc, lr)):
            hist[k].append(v)
        if not quiet:
            print(f"epoch {epoch:3d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.3f} acc {va_acc:.3f} | lr {lr:.1e}")

        if va_loss < best_loss - 1e-4:
            best_loss, best_state, best_epoch, bad = va_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= config.EARLY_STOP_PATIENCE:
                if not quiet:
                    print(f"Early stopping: no val improvement for {bad} epochs.")
                break

    model.load_state_dict(best_state)
    model.eval()
    hist["best_epoch"] = best_epoch
    return model, hist


def plot_curves(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = range(1, len(hist["train_loss"]) + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(ep, hist["train_loss"], label="train")
    a.plot(ep, hist["val_loss"], label="validation")
    a.set_title("Loss")
    b.plot(ep, hist["train_acc"], label="train")
    b.plot(ep, hist["val_acc"], label="validation")
    b.set_title("Accuracy")
    for ax in (a, b):
        ax.axvline(hist["best_epoch"], color="grey", ls="--", lw=1, label="best epoch")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def append_experiment(row):
    path = config.RESULTS_DIR / "experiments.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def prepare(args, max_reps=None):
    """Load phrases and clips and build the splits."""
    phrases = load_phrases(phrase_set=args.phrase_set)
    clips = scan_clips(phrase_ids=set(phrases))
    if not clips:
        sys.exit(f"No clips found in {config.RAW_DIR}. Record some first:\n"
                 "  python collect_data.py --speaker s01 --session 1 --reps 7")
    if args.holdout_speaker and not any(c.speaker == args.holdout_speaker for c in clips):
        sys.exit(f"Speaker {args.holdout_speaker!r} has no clips.")
    splits = make_splits(clips, args.holdout_speaker, seed=args.seed, max_reps=max_reps)
    if not splits["train"]:
        sys.exit("No training clips: sessions 1-2 are used for training, session 3 for testing.")

    trained_ids = {c.phrase_id for c in splits["train"]}
    label_map = [pid for pid in phrases if pid in trained_ids]
    missing = [pid for pid in phrases if pid not in trained_ids]
    if missing:
        print(f"Warning: no training clips for {', '.join(missing)}; they are left out of the model.")
    index = {pid: i for i, pid in enumerate(label_map)}

    data = {}
    for name, clips_ in splits.items():
        pos, kept = load_positions([c for c in clips_ if c.phrase_id in index])
        data[name] = (pos, [index[c.phrase_id] for c in kept], kept)
    return phrases, label_map, data


def run(args, max_reps=None, save=True):
    phrases, label_map, data = prepare(args, max_reps)
    model_config = {"arch": args.arch, "n_classes": len(label_map),
                    "use_velocity": not args.no_velocity, "use_attention": not args.no_attention,
                    "n_frames": config.TARGET_FRAMES}
    augment_cfg = None if args.no_augment else config.AUGMENT
    tr, va, te = data["train"], data["val"], data["test"]
    speakers = sorted({c.speaker for c in tr[2]})
    if not args.quiet:
        print(f"{len(label_map)} phrases | speakers {', '.join(speakers)} | "
              f"train {len(tr[0])}  val {len(va[0])}  test (session 3) {len(te[0])}"
              + (f"  unseen {args.holdout_speaker}: {len(data['unseen'][0])}" if args.holdout_speaker else ""))

    t0 = time.time()
    model, hist = train_model(tr[0], tr[1], va[0], va[1], model_config, augment_cfg,
                              args.epochs, args.seed, args.quiet)
    minutes = (time.time() - t0) / 60

    results = {}
    for name in ("val", "test", "unseen"):
        pos, y, _ = data[name]
        if pos:
            probs = predict_proba(model, stack_features(pos, model_config["use_velocity"]))
            results[name] = compute_metrics(y, probs, label_map, phrases)

    tag = args.tag or ("main" if not args.holdout_speaker else f"holdout_{args.holdout_speaker}")
    if save:
        if args.out:
            out = Path(args.out)
        elif args.holdout_speaker:
            out = holdout_model_path(args.holdout_speaker)
        elif args.tag:
            out = config.MODEL_DIR / "experiments" / f"{args.tag}.pt"
        else:
            out = config.BASE_MODEL
        save_checkpoint(out, model, model_config, label_map,
                        phrase_set=args.phrase_set, speakers=speakers,
                        split={"holdout_speaker": args.holdout_speaker, "seed": args.seed,
                               "max_reps": max_reps, "val_fraction": config.VAL_FRACTION,
                               "test_sessions": list(config.TEST_SESSIONS)},
                        history={k: v for k, v in hist.items()},
                        trained_at=datetime.now().isoformat(timespec="seconds"))
        if out == config.BASE_MODEL:
            save_json(config.LABEL_MAP_FILE, {"label_map": label_map,
                                              "phrases": {pid: phrases[pid]["text"] for pid in label_map}})
            plot_curves(hist, config.RESULTS_DIR / "training_curves.png")
        else:
            plot_curves(hist, config.RESULTS_DIR / f"training_curves_{tag}.png")
        print(f"\nSaved {out}  ({count_parameters(model):,} parameters, {minutes:.1f} min, "
              f"best epoch {hist['best_epoch']})")

    get = lambda split, key: (results.get(split) or {}).get(key)
    append_experiment({
        "timestamp": datetime.now().isoformat(timespec="seconds"), "tag": tag, "arch": args.arch,
        "velocity": not args.no_velocity, "attention": not args.no_attention,
        "augment": not args.no_augment, "max_reps": max_reps or "", "phrases": len(label_map),
        "holdout": args.holdout_speaker or "", "n_train": len(tr[0]), "n_val": len(va[0]),
        "n_test": len(te[0]), "val_top1": get("val", "top1_accuracy"),
        "test_top1": get("test", "top1_accuracy"), "test_top3": get("test", "top3_accuracy"),
        "test_macro_f1": get("test", "macro_f1"), "test_emergency_recall": get("test", "emergency_recall"),
        "unseen_top1": get("unseen", "top1_accuracy"), "epochs": len(hist["train_loss"]),
        "minutes": round(minutes, 2),
    })
    for name, m in results.items():
        label = {"val": "Validation", "test": "Test (session 3)", "unseen": f"Unseen speaker {args.holdout_speaker}"}[name]
        er = m.get("emergency_recall")
        print(f"{label:28s} top-1 {m['top1_accuracy']:.3f}  top-3 {m['top3_accuracy']:.3f}  "
              f"macro F1 {m['macro_f1']:.3f}" + (f"  emergency recall {er:.3f}" if er is not None else ""))
    return results


def learning_curve(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, _, data = prepare(args)
    counts = {}
    for c in data["train"][2] + data["val"][2]:
        counts[(c.speaker, c.phrase_id)] = counts.get((c.speaker, c.phrase_id), 0) + 1
    most = max(counts.values())
    steps = sorted({r for r in (3, 5, 8, 11, 14, most) if 2 <= r <= most})
    split = "test" if data["test"][0] else "val"
    print(f"Learning curve over {steps} clips per phrase per speaker, scored on the {split} set")

    xs, ys = [], []
    for r in steps:
        args.tag = f"learning_curve_{r}"
        args.quiet = True
        res = run(args, max_reps=r, save=False)
        if split in res:
            xs.append(r)
            ys.append(res[split]["top1_accuracy"])
            print(f"  {r:2d} reps -> top-1 {ys[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, "o-")
    ax.set_xlabel("training clips per phrase per speaker (sessions 1-2)")
    ax.set_ylabel(f"top-1 accuracy ({split} set)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.set_title("Learning curve")
    fig.tight_layout()
    out = config.RESULTS_DIR / "learning_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    if args.learning_curve:
        learning_curve(args)
    else:
        run(args, max_reps=args.reps)


if __name__ == "__main__":
    main()
