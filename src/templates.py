"""Instant recogniser: nearest neighbours with dynamic time warping (DTW).

The neural model (train.py) needs hundreds of clips and a long training run,
which takes hours on a low-power laptop. This recogniser needs only 3-5 clips
of each phrase from the person who will use it and no training at all: a new
clip is compared with every stored example of every phrase, allowing for
differences in speed (DTW), and the closest phrase wins. It is ready seconds
after calibration, on any CPU.

Confidence is calibrated with leave-one-out (LOO): each stored clip is
recognised using all the others, and the "temperature" that turns distances
into probabilities is chosen so the probabilities match how often that is
right. The same pass gives an accuracy estimate. Because calibration clips
come from one sitting, the estimate is optimistic.

A clip that is far from everything stored (a phrase that isn't in the list)
gets low confidence, so the decision logic shows options instead of speaking.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import config
from src.dataset import scan_clips
from src.features import fill_missing, normalise, resample

N_FRAMES = 32          # frames per signature
BAND = 8               # DTW warping window (frames)
TAU_MIN = 0.15         # softest allowed confidence temperature (relative margin units)
REJECT_FACTOR = 2.0    # "none of these" beyond 2x the typical distance between phrases
VERSION = 3            # bump to invalidate caches when the signature or scoring changes


# --- Signatures ----------------------------------------------------------------

def _rest_and_active(pos, margin=2, frac=0.2):
    """Resting lip shape, and the frames where the mouth moves away from it."""
    n = len(pos)
    k = max(1, min(3, n // 4))
    rest = np.median(np.concatenate([pos[:k], pos[-k:]]), axis=0)
    dev = np.abs(pos - rest).mean(axis=(1, 2))
    lo, hi = float(dev.min()), float(dev.max())
    active = np.where(dev > lo + frac * (hi - lo))[0]
    if hi - lo < 1e-6 or len(active) < 3:
        return rest, 0, n
    a, b = max(0, active[0] - margin), min(n, active[-1] + margin + 1)
    return (rest, a, b) if b - a >= 5 else (rest, 0, n)


def signature(positions):
    """Normalised positions (T, 40, 2) -> (features (32, 80), duration fraction).

    Still frames before and after the phrase are cut off, and each frame is
    stored as its change from the resting mouth shape, which makes clips from
    different days easier to compare.
    """
    rest, a, b = _rest_and_active(positions)
    feats = resample(positions[a:b] - rest, N_FRAMES).reshape(N_FRAMES, -1)
    return feats.astype(np.float32), (b - a) / len(positions)


def raw_signature(clip):
    filled = fill_missing(clip)
    if filled is None or len(filled) < 4:
        return None
    return signature(normalise(filled))


def dtw_distances(q, templates, band=BAND):
    """DTW distance from one sequence q (L, D) to each template (N, L, D)."""
    n, length, _ = templates.shape
    qq = (q * q).sum(-1)
    tt = (templates * templates).sum(-1)
    cross = np.einsum("id,njd->nij", q, templates)
    cost = np.sqrt(np.maximum(qq[None, :, None] + tt[:, None, :] - 2 * cross, 0))
    acc = np.full((n, length + 1, length + 1), np.inf, np.float32)
    acc[:, 0, 0] = 0
    for i in range(1, length + 1):
        for j in range(max(1, i - band), min(length, i + band) + 1):
            acc[:, i, j] = cost[:, i - 1, j - 1] + np.minimum(
                np.minimum(acc[:, i - 1, j], acc[:, i, j - 1]), acc[:, i - 1, j - 1])
    return acc[:, length, length] / (2 * length)


def _class_scores(dist, labels, n_classes, k):
    """Per phrase: mean of the k smallest distances to its examples."""
    scores = np.full(n_classes, np.inf)
    for c in range(n_classes):
        d = dist[labels == c]
        if len(d):
            scores[c] = np.sort(d)[:min(k, len(d))].mean()
    return scores


# --- Matcher -------------------------------------------------------------------

class TemplateMatcher:
    kind = "templates"

    def __init__(self, feats, durs, labels, label_map, params, report=None, name="templates"):
        self.feats = np.asarray(feats, np.float32)
        self.durs = np.asarray(durs, np.float32)
        self.labels = np.asarray(labels, np.int64)
        self.label_map = list(label_map)
        self.params = dict(params)
        self.report = report or {}
        self.name = name

    # -- building -----------------------------------------------------------

    @classmethod
    def from_clips(cls, clips, phrase_ids=None, calibrate=True, max_queries=300, progress=None,
                   name="templates"):
        """Build from Clip objects (src.dataset.Clip). phrase_ids keeps file order."""
        sigs, pids = [], []
        for c in clips:
            s = raw_signature(np.load(c.path))
            if s is not None:
                sigs.append(s)
                pids.append(c.phrase_id)
        order = list(phrase_ids) if phrase_ids else sorted(set(pids))
        label_map = [pid for pid in order if pid in set(pids)]
        if not label_map:
            raise ValueError("No usable clips to build templates from")
        index = {pid: i for i, pid in enumerate(label_map)}
        keep = [i for i, pid in enumerate(pids) if pid in index]
        feats = np.stack([sigs[i][0] for i in keep])
        durs = np.array([sigs[i][1] for i in keep], np.float32)
        labels = np.array([index[pids[i]] for i in keep])
        m = cls(feats, durs, labels, label_map, {"T": 0.05, "k": 1, "dur_weight": 0.0, "reject": np.inf},
                name=name)
        if calibrate:
            m.calibrate(max_queries=max_queries, progress=progress)
        return m

    def _distances(self, feat, dur):
        d = dtw_distances(feat, self.feats)
        if self.params["dur_weight"]:
            d = d + self.params["dur_weight"] * np.abs(np.log(max(dur, 1e-3) / np.maximum(self.durs, 1e-3)))
        return d

    def calibrate(self, max_queries=300, progress=None, seed=0):
        """Leave-one-out: choose k, duration weight and temperature; estimate accuracy."""
        n, n_cls = len(self.feats), len(self.label_map)
        counts = np.bincount(self.labels, minlength=n_cls)
        candidates = np.where(counts[self.labels] >= 2)[0]   # needs another example of its phrase
        if len(candidates) == 0:
            self.report = {"loo_accuracy": None, "n_templates": n}
            return
        rng = np.random.default_rng(seed)
        queries = candidates if len(candidates) <= max_queries else np.sort(
            rng.choice(candidates, max_queries, replace=False))

        dtw_rows, dur_rows = [], []
        t0 = time.time()
        for qi, i in enumerate(queries):
            row = dtw_distances(self.feats[i], self.feats)
            row[i] = np.inf
            dtw_rows.append(row)
            dur_rows.append(np.abs(np.log(max(self.durs[i], 1e-3) / np.maximum(self.durs, 1e-3))))
            if progress and (qi % 20 == 0 or qi == len(queries) - 1):
                progress(qi + 1, len(queries), time.time() - t0)
        dtw_rows, dur_rows = np.array(dtw_rows), np.array(dur_rows)
        truth = self.labels[queries]
        scale = float(np.median(dtw_rows[np.isfinite(dtw_rows)]))

        def evaluate(k, w):
            dist = dtw_rows + w * dur_rows
            return np.array([_class_scores(d, self.labels, n_cls, k) for d in dist])

        best = None
        for k in (1, 2):
            for lam in (0.0, 0.25, 0.5, 1.0):
                scores = evaluate(k, lam * scale)
                acc = float((scores.argmin(1) == truth).mean())
                if best is None or acc > best[0] + 1e-9:
                    best = (acc, k, lam * scale, scores)
        acc, k, w, scores = best
        rows = np.arange(len(truth))
        true_scores = scores[rows, truth]
        floor = 0.25 * float(np.median(true_scores[np.isfinite(true_scores)]))

        # Gross mismatch (talking, laughing, a phrase not in the list): far
        # beyond the distance to the nearest *other* phrase.
        wrong = scores.copy()
        wrong[rows, truth] = np.inf
        nearest_wrong = wrong.min(1)
        reject = float(REJECT_FACTOR * np.median(nearest_wrong[np.isfinite(nearest_wrong)])) \
            if np.isfinite(nearest_wrong).any() else np.inf

        # Confidence comes from how clearly the best phrase beats the others,
        # relative to its own distance (clips from another day are all ~1.5x
        # farther from the stored examples; a relative margin is unaffected).
        # The temperature is set so the average top-1 confidence matches the
        # leave-one-out accuracy: confidence then means "how often it's right".
        rel = self._relative(np.column_stack([scores, np.full(len(scores), reject)]), floor)

        def mean_confidence(t):
            logits = -rel / t
            p = np.exp(logits - logits.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            return float(p[:, :-1].max(1).mean())

        tau = TAU_MIN
        for t in np.logspace(-2, 0.7, 80):          # confidence falls as t rises
            if mean_confidence(t) <= acc:
                tau = float(t)
                break
        # Calibration clips come from one sitting, so they look easier than
        # real use; never let the temperature collapse below a safe floor.
        tau = max(tau, TAU_MIN)

        self.params = {"T": tau, "k": int(k), "dur_weight": float(w), "reject": reject, "floor": floor}
        pred = scores.argmin(1)
        per_phrase, confusions = {}, {}
        for c, pid in enumerate(self.label_map):
            mask = truth == c
            if mask.any():
                per_phrase[pid] = float((pred[mask] == c).mean())
        for t_, p_ in zip(truth, pred):
            if t_ != p_:
                key = f"{self.label_map[t_]}>{self.label_map[p_]}"
                confusions[key] = confusions.get(key, 0) + 1
        self.report = {"loo_accuracy": acc, "n_queries": int(len(queries)), "n_templates": int(n),
                       "per_phrase": per_phrase,
                       "confusions": dict(sorted(confusions.items(), key=lambda kv: -kv[1]))}

    # -- prediction ---------------------------------------------------------

    @staticmethod
    def _relative(scores, floor):
        """(score - best) / best for each row; inf (no examples) becomes very large."""
        scores = np.atleast_2d(scores)
        best = scores.min(1, keepdims=True)
        rel = (scores - best) / np.maximum(best, max(floor, 1e-9))
        return np.where(np.isfinite(rel), rel, 1e3)

    def predict_signature(self, sig):
        feat, dur = sig
        scores = _class_scores(self._distances(feat, dur), self.labels, len(self.label_map), self.params["k"])
        floor, tau = self.params.get("floor", 0.0), self.params["T"]
        logits = -self._relative(scores, floor)[0] / tau
        reject = self.params["reject"]
        reject_logit = -self._relative(np.append(scores, reject), floor)[0][-1] / tau \
            if np.isfinite(reject) else -np.inf
        top = max(logits.max(), reject_logit)
        e = np.exp(logits - top)
        total = e.sum() + (np.exp(reject_logit - top) if np.isfinite(reject_logit) else 0.0)
        return (e / total).astype(np.float64)   # may sum to < 1: the rest is "none of these"

    def predict_clip(self, clip):
        """Raw clip (T, 42, 3) -> (probabilities, milliseconds), or (None, ms)."""
        t0 = time.perf_counter()
        sig = raw_signature(clip)
        probs = None if sig is None else self.predict_signature(sig)
        return probs, (time.perf_counter() - t0) * 1000

    def predict_positions(self, positions):
        return np.stack([self.predict_signature(signature(p)) for p in positions])

    def describe(self):
        acc = self.report.get("loo_accuracy")
        per = np.bincount(self.labels, minlength=len(self.label_map))
        est = f", calibration estimate {acc:.0%}" if acc is not None else ""
        return (f"instant templates ({len(self.label_map)} phrases, "
                f"{per.min()}-{per.max()} clips each{est})")

    # -- cache --------------------------------------------------------------

    def save(self, path, signature_hash):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, feats=self.feats, durs=self.durs, labels=self.labels,
                 label_map=np.array(self.label_map), params=np.array(json.dumps(self.params)),
                 report=np.array(json.dumps(self.report)), signature=np.array(signature_hash),
                 version=np.array(VERSION))

    @classmethod
    def load(cls, path, name="templates"):
        with np.load(path, allow_pickle=False) as z:
            if int(z["version"]) != VERSION:
                return None, None
            m = cls(z["feats"], z["durs"], z["labels"], [str(s) for s in z["label_map"]],
                    json.loads(str(z["params"])), json.loads(str(z["report"])), name=name)
            return m, str(z["signature"])


def clips_hash(clips):
    h = hashlib.sha1(f"v{VERSION}".encode())
    for c in sorted(clips, key=lambda c: str(c.path)):
        st = c.path.stat()
        h.update(f"{c.path.as_posix()}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def cache_path(speaker, model_dir=None):
    return Path(model_dir or config.MODEL_DIR) / f"templates_{speaker}.npz"


def for_speaker(speaker, phrase_ids, raw_dir=None, model_dir=None, rebuild=False, progress=None):
    """Templates from all of this speaker's clips, cached until the clips change.

    Returns None if the speaker has no usable clips.
    """
    clips = scan_clips(raw_dir, phrase_ids=set(phrase_ids), speakers={speaker})
    if not clips:
        return None
    path = cache_path(speaker, model_dir)
    sig = clips_hash(clips)
    if path.exists() and not rebuild:
        m, cached_sig = TemplateMatcher.load(path, name=f"templates_{speaker}")
        if m is not None and cached_sig == sig:
            return m
    m = TemplateMatcher.from_clips(clips, phrase_ids, progress=progress, name=f"templates_{speaker}")
    m.save(path, sig)
    return m
