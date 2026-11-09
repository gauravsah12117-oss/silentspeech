"""Quick personalisation for a new user (about 10 minutes for 30 phrases).

    python enroll.py --speaker new01 --reps 5

1. Records `reps` clips of every phrase the model knows (same recorder as
   collect_data.py; skip with --skip-record if they are already recorded).
2. Freezes the convolution and GRU layers and fine-tunes only the attention
   pooling and the final layers, for about one minute.
3. Saves models/lipreader_new01.pt, which app.py picks up automatically:
       python app.py --speaker new01

It prints accuracy before and after on one held-back clip per phrase. For
the presentation, also measure on a separate session:
       python evaluate.py --split unseen_speaker --speaker new01 --session 2 [--model models/lipreader_new01.pt]
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from src.dataset import LipDataset, load_positions, scan_clips, stack_features
from src.model import load_checkpoint, predict_proba, save_checkpoint
from src.utils import load_phrases, personal_model_path, seed_everything


def fine_tune(model, positions, labels, use_velocity, seconds, seed=config.SEED, quiet=False):
    """Train only model.enrolment_modules(); everything else stays frozen."""
    seed_everything(seed)
    for p in model.parameters():
        p.requires_grad = False
    trainable = [p for m in model.enrolment_modules() for p in m.parameters()]
    for p in trainable:
        p.requires_grad = True
    opt = torch.optim.AdamW(trainable, lr=config.ENROLL_LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    loader = DataLoader(LipDataset(positions, labels, use_velocity, config.AUGMENT, seed),
                        batch_size=config.BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    end = time.time() + seconds
    epoch = 0
    while time.time() < end and epoch < config.ENROLL_MAX_EPOCHS:
        model.eval()                       # frozen BatchNorm and dropout stay fixed
        for m in model.enrolment_modules():
            m.train()
        total = correct = 0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            criterion(logits, yb).backward()
            opt.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
        epoch += 1
        if not quiet and epoch % 5 == 0:
            print(f"  epoch {epoch:3d}  train acc {correct / total:.3f}  ({max(0, end - time.time()):.0f}s left)")
    for p in model.parameters():
        p.requires_grad = True
    model.eval()
    return epoch


def accuracy(model, positions, labels, use_velocity):
    if not positions:
        return None
    probs = predict_proba(model, stack_features(positions, use_velocity))
    return float((probs.argmax(1) == np.asarray(labels)).mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--session", type=int, default=1, help="session number for the new recordings")
    ap.add_argument("--base", default=str(config.BASE_MODEL), help="model to start from")
    ap.add_argument("--seconds", type=float, default=config.ENROLL_SECONDS, help="fine-tuning time budget")
    ap.add_argument("--skip-record", action="store_true", help="use clips already on disk")
    ap.add_argument("--lighting", default="unknown")
    ap.add_argument("--camera", type=int)
    args = ap.parse_args(argv)

    if not Path(args.base).exists():
        sys.exit(f"Base model not found: {args.base}. Train it first: python train.py")
    base_model, ckpt = load_checkpoint(args.base)
    label_map = ckpt["label_map"]
    use_vel = ckpt["model_config"].get("use_velocity", True)
    phrases = load_phrases(phrase_set="full")
    model_phrases = {pid: phrases[pid] for pid in label_map}
    if args.speaker in ckpt.get("speakers", []):
        print(f"Note: {args.speaker} is already in the base model's training data, "
              "so the 'before' score is not an unseen-speaker score.")

    if not args.skip_record:
        from collect_data import record_phrases
        print(f"Recording {args.reps} clips of {len(model_phrases)} phrases for {args.speaker}...")
        record_phrases(args.speaker, args.session, args.reps, model_phrases,
                       lighting=args.lighting, camera_index=args.camera, title="Enrolment")

    clips = scan_clips(phrase_ids=set(label_map), speakers={args.speaker})
    if not clips:
        sys.exit(f"No clips for {args.speaker}. Record them (drop --skip-record).")
    index = {pid: i for i, pid in enumerate(label_map)}

    # Hold back the most recent clip of each phrase (if it has 2+) to measure before/after.
    by_phrase = {}
    for c in sorted(clips, key=lambda c: (c.session, c.rep)):
        by_phrase.setdefault(c.phrase_id, []).append(c)
    train_clips, check_clips = [], []
    for items in by_phrase.values():
        if len(items) >= 2:
            check_clips.append(items[-1])
            train_clips += items[:-1]
        else:
            train_clips += items
    tr_pos, tr_kept = load_positions(train_clips)
    ck_pos, ck_kept = load_positions(check_clips)
    tr_y = [index[c.phrase_id] for c in tr_kept]
    ck_y = [index[c.phrase_id] for c in ck_kept]

    before = accuracy(base_model, ck_pos, ck_y, use_vel)
    print(f"\nFine-tuning on {len(tr_pos)} clips for up to {args.seconds:.0f} s "
          "(convolution and GRU layers frozen)...")
    model = copy.deepcopy(base_model)
    epochs = fine_tune(model, tr_pos, tr_y, use_vel, args.seconds)
    after = accuracy(model, ck_pos, ck_y, use_vel)

    out = personal_model_path(args.speaker)
    sessions = sorted({c.session for c in clips})
    save_checkpoint(out, model, ckpt["model_config"], label_map,
                    phrase_set=ckpt.get("phrase_set"), speakers=ckpt.get("speakers", []),
                    split=ckpt.get("split", {}),
                    enrolled={"speaker": args.speaker, "sessions": sessions, "base": str(args.base),
                              "n_clips": len(tr_pos), "epochs": epochs,
                              "before": before, "after": after})
    print(f"\nSaved {out} ({epochs} epochs).")
    if before is not None:
        print(f"Held-back clips ({len(ck_pos)}): accuracy {before:.3f} before -> {after:.3f} after enrolment")
    print(f"Use it with: python app.py --speaker {args.speaker}")


if __name__ == "__main__":
    main()
