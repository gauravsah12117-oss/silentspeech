"""Evaluation metrics shared by train.py, evaluate.py and enroll.py."""
import numpy as np

import config
from src.decision import decide


def compute_metrics(y_true, probs, label_map, phrases, conf_min=config.CONF_MIN,
                    emergency_min=config.EMERGENCY_MIN):
    """y_true: class indices (N,), probs: (N, C). Returns a JSON-friendly dict."""
    from sklearn.metrics import confusion_matrix, f1_score

    y_true = np.asarray(y_true)
    n = len(y_true)
    if n == 0:
        return {"n": 0}
    pred = probs.argmax(1)
    top3 = np.argsort(probs, axis=1)[:, ::-1][:, :3]
    labels = list(range(len(label_map)))

    per_phrase = {}
    for i, pid in enumerate(label_map):
        mask = y_true == i
        if mask.any():
            per_phrase[pid] = float((pred[mask] == i).mean())

    # Emergency safety: a clip of an emergency phrase is "caught" when the
    # decision logic raises an alert for that same phrase.
    emerg_idx = [i for i, pid in enumerate(label_map) if phrases.get(pid, {}).get("emergency")]
    caught = total_emerg = false_alarms = total_normal = 0
    for yt, p in zip(y_true, probs):
        alert = decide(p, label_map, phrases, emergency_min=emergency_min).alert_phrase_id
        if yt in emerg_idx:
            total_emerg += 1
            caught += alert == label_map[yt]
        else:
            total_normal += 1
            false_alarms += alert is not None

    return {
        "n": int(n),
        "top1_accuracy": float((pred == y_true).mean()),
        "top3_accuracy": float((top3 == y_true[:, None]).any(1).mean()),
        "macro_f1": float(f1_score(y_true, pred, labels=sorted(set(y_true)), average="macro",
                                   zero_division=0)),
        "emergency_recall": float(caught / total_emerg) if total_emerg else None,
        "emergency_false_alarm_rate": float(false_alarms / total_normal) if total_normal else None,
        "per_phrase_accuracy": per_phrase,
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
    }


def threshold_table(y_true, probs, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """For each threshold: share of clips spoken directly and their accuracy."""
    y_true = np.asarray(y_true)
    conf = probs.max(1)
    correct = probs.argmax(1) == y_true
    rows = []
    for t in thresholds:
        acc_mask = conf >= t
        rows.append({"threshold": t, "coverage": float(acc_mask.mean()),
                     "accuracy_when_spoken": float(correct[acc_mask].mean()) if acc_mask.any() else None})
    return rows


def top_confusions(cm, label_map, k=8):
    """The k most frequent (true, predicted) mix-ups, as look-alike candidates."""
    cm = np.asarray(cm)
    pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j]:
                pairs.append((int(cm[i, j]), label_map[i], label_map[j]))
    return sorted(pairs, reverse=True)[:k]
