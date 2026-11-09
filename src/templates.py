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
from src.dataset import blendshape_path, load_blendshapes, scan_clips
from src.features import fill_missing, lip_motion, normalise, resample
from src.landmarks import MOUTH_BLENDSHAPES

N_FRAMES = 32          # frames per signature
BAND = 8               # DTW warping window (frames)
TAU_MIN = 0.15         # softest allowed confidence temperature (relative margin units)
REJECT_FACTOR = 2.0    # "none of these" beyond 2x the typical distance between phrases
OUTLIER_FACTOR = 2.0   # a clip this much farther from its own phrase than usual is left out
VELOCITY_WEIGHT = 2.0  # how much the speed of the lips counts next to their shape
BLEND_WEIGHTS = (0.0, 0.5, 1.0, 2.0)   # muscle-score weights tried at calibration (0 = not used)
VERSION = 6            # bump to invalidate caches when the signature or scoring changes


# --- Signatures ----------------------------------------------------------------

def active_fraction(pos, frac=0.15):
    """Share of the clip during which the lips are moving (used as a duration cue)."""
    if len(pos) < 4:
        return 1.0
    speed = np.abs(np.diff(pos, axis=0)).mean(axis=(1, 2))
    speed = np.convolve(speed, np.ones(3) / 3, mode="same")
    lo, hi = float(speed.min()), float(speed.max())
    active = np.where(speed > lo + frac * (hi - lo))[0]
    if hi - lo < 1e-9 or len(active) < 2:
        return 1.0
    return (active[-1] - active[0] + 1) / len(speed)


def signature(positions):
    """Normalised positions (T, 40, 2) -> (features (32, 160), duration fraction).

    The whole clip is resampled to 32 frames and the clip's average mouth
    shape is subtracted, so the signature describes how the shape changes;
    the frame-to-frame speed of every point is added (weight VELOCITY_WEIGHT).
    Real users are often already moving their lips when recording starts and
    still moving when it stops, so nothing here assumes a resting mouth at
    either end.

    Measured on the first real calibration (50 clips, 10 phrases, leave-one-
    out): change from "rest" frames 35%, shape only 66%, shape + speed 70%.
    Learned whitening/LDA transforms did worse (16-52%): too few clips.
    """
    feats = resample(positions, N_FRAMES).reshape(N_FRAMES, -1)
    feats = feats - feats.mean(0)
    # Speed is measured on a lightly smoothed copy: differencing raw frames
    # magnifies landmark jitter (dim light), which hurt clips from another day.
    padded = np.pad(feats, ((1, 1), (0, 0)), mode="edge")
    smooth = (padded[:-2] + padded[1:-1] + padded[2:]) / 3
    speed = np.diff(smooth, axis=0, prepend=smooth[:1]) * VELOCITY_WEIGHT
    return np.concatenate([feats, speed], 1).astype(np.float32), active_fraction(positions)


def raw_signature(clip):
    filled = fill_missing(clip)
    if filled is None or len(filled) < 4:
        return None
    return signature(normalise(filled))


def blend_signature(blendshapes):
    """(T, 52) Face Landmarker muscle scores -> (32, 28) mouth/jaw signature, or None.

    Same treatment as the lip shape: resampled to 32 frames and the clip's
    average removed, so it describes how the mouth muscles change.
    """
    if blendshapes is None:
        return None
    b = np.asarray(blendshapes, dtype=np.float32)[:, MOUTH_BLENDSHAPES]
    filled = fill_missing(b[:, :, None])
    if filled is None or len(filled) < 4:
        return None
    f = resample(filled[:, :, 0], N_FRAMES)
    return (f - f.mean(0)).astype(np.float32)


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

    def __init__(self, feats, durs, labels, label_map, params, report=None, name="templates", bsfeats=None):
        self.feats = np.asarray(feats, np.float32)
        self.durs = np.asarray(durs, np.float32)
        self.labels = np.asarray(labels, np.int64)
        self.label_map = list(label_map)
        self.params = {"blend_weight": 0.0, **dict(params)}
        self.report = report or {}
        self.name = name
        self.bsfeats = None if bsfeats is None else np.asarray(bsfeats, np.float32)   # (N, 32, 28) or None

    def _uses_blend(self):
        return self.bsfeats is not None and self.params.get("blend_weight", 0) > 0

    def _template_feats(self, blend_weight=None):
        """Template signatures, with the muscle scores appended when they are used."""
        w = self.params.get("blend_weight", 0) if blend_weight is None else blend_weight
        if self.bsfeats is None or w <= 0:
            return self.feats
        return np.concatenate([self.feats, w * self.bsfeats], axis=2)

    # -- building -----------------------------------------------------------

    @classmethod
    def from_clips(cls, clips, phrase_ids=None, calibrate=True, max_queries=300, progress=None,
                   name="templates", prune=True):
        """Build from Clip objects (src.dataset.Clip). phrase_ids keeps file order.

        prune: leave out clips where the lips barely moved, and clips that look
        nothing like the other examples of their phrase (a missed cue, a laugh,
        a turned head). Each phrase always keeps at least 2 examples.
        """
        sigs, bsigs, pids, names, motion = [], [], [], [], []
        for c in clips:
            filled = fill_missing(np.load(c.path))
            if filled is None or len(filled) < 4:
                continue
            pos = normalise(filled)
            sigs.append(signature(pos))
            bsigs.append(blend_signature(load_blendshapes(c.path)))
            pids.append(c.phrase_id)
            names.append(f"{c.phrase_id} clip {c.session}_{c.rep:02d}")
            motion.append(lip_motion(pos))
        order = list(phrase_ids) if phrase_ids else sorted(set(pids))
        label_map = [pid for pid in order if pid in set(pids)]
        if not label_map:
            raise ValueError("No usable clips to build templates from")
        index = {pid: i for i, pid in enumerate(label_map)}
        keep = [i for i, pid in enumerate(pids) if pid in index]
        dropped = []
        if prune and len(keep) <= 600:
            keep, dropped = cls._prune(keep, sigs, pids, names, motion)
        feats = np.stack([sigs[i][0] for i in keep])
        durs = np.array([sigs[i][1] for i in keep], np.float32)
        labels = np.array([index[pids[i]] for i in keep])
        # muscle scores only if every clip has them (clips from the older tracker don't)
        bsfeats = np.stack([bsigs[i] for i in keep]) if all(bsigs[i] is not None for i in keep) else None
        m = cls(feats, durs, labels, label_map, {"T": 0.05, "k": 1, "dur_weight": 0.0, "reject": np.inf},
                name=name, bsfeats=bsfeats)
        if calibrate:
            m.calibrate(max_queries=max_queries, progress=progress)
        m.report["dropped"] = dropped
        return m

    @staticmethod
    def _prune(keep, sigs, pids, names, motion):
        """Return (indices to keep, [(name, reason)] left out)."""
        dropped = []

        def removable(i, kept):
            return sum(pids[j] == pids[i] for j in kept) > 2

        kept = list(keep)
        for i in sorted(kept, key=lambda i: motion[i]):          # lips barely moved
            if motion[i] < config.MIN_LIP_MOTION and removable(i, kept):
                kept.remove(i)
                dropped.append((names[i], "lips barely moved"))
        if len(kept) < 3:
            return kept, dropped
        feats = np.stack([sigs[i][0] for i in kept])
        own = []
        for a, i in enumerate(kept):
            d = dtw_distances(feats[a], feats)
            d[a] = np.inf
            same = [d[b] for b, j in enumerate(kept) if pids[j] == pids[i]]
            own.append(min(same) if same else np.inf)
        own = np.array(own)
        typical = float(np.median(own[np.isfinite(own)])) if np.isfinite(own).any() else np.inf
        outliers = [kept[a] for a in np.argsort(-own) if own[a] > OUTLIER_FACTOR * typical]   # worst first
        for i in outliers:
            if removable(i, kept):
                kept.remove(i)
                dropped.append((names[i], "looks unlike the other clips of this phrase"))
        return kept, dropped

    def restrict(self, phrase_ids, progress=None):
        """A matcher that only knows these phrases (re-calibrated on their clips)."""
        keep_ids = [pid for pid in self.label_map if pid in set(phrase_ids)]
        old = {pid: i for i, pid in enumerate(self.label_map)}
        mask = np.isin(self.labels, [old[pid] for pid in keep_ids])
        new = {old[pid]: j for j, pid in enumerate(keep_ids)}
        m = TemplateMatcher(self.feats[mask], self.durs[mask], [new[c] for c in self.labels[mask]], keep_ids,
                            self.params, name=f"{self.name}_focus{len(keep_ids)}",
                            bsfeats=None if self.bsfeats is None else self.bsfeats[mask])
        m.calibrate(progress=progress)
        return m

    def rank_phrases(self):
        """Phrases from most to least reliable, with the estimated accuracy of each top-n set.

        Greedy: repeatedly drop the phrase whose removal helps leave-one-out
        accuracy most. Returns [(phrase_id, accuracy of the set up to it)].
        """
        n = len(self.feats)
        feats = self._template_feats()
        dist = np.stack([dtw_distances(feats[i], feats) for i in range(n)])
        np.fill_diagonal(dist, np.inf)

        def accuracy(classes):
            idx = [i for i in range(n) if self.labels[i] in classes
                   and np.sum(self.labels == self.labels[i]) >= 2]
            if not idx:
                return 1.0
            hits = 0
            for i in idx:
                scores = {c: dist[i, self.labels == c].min() for c in classes}
                hits += min(scores, key=scores.get) == self.labels[i]
            return hits / len(idx)

        current = list(range(len(self.label_map)))
        removed = []
        while len(current) > 2:
            drop = max(current, key=lambda c: accuracy([x for x in current if x != c]))
            removed.append((drop, accuracy(current)))
            current = [x for x in current if x != drop]
        ranking = [(self.label_map[c], accuracy(current[:i + 1])) for i, c in enumerate(current)]
        ranking += [(self.label_map[c], acc) for c, acc in reversed(removed)]
        return ranking

    def _distances(self, feat, dur, bsfeat=None):
        if self._uses_blend() and bsfeat is not None:
            w = self.params["blend_weight"]
            d = dtw_distances(np.concatenate([feat, w * bsfeat], axis=1), self._template_feats())
        else:                      # no muscle scores for this clip: compare lip shapes only
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

        truth = self.labels[queries]
        dur_rows = np.array([np.abs(np.log(max(self.durs[i], 1e-3) / np.maximum(self.durs, 1e-3)))
                             for i in queries])
        # Muscle scores (when every clip has them) are tried at several weights;
        # the weight that recognises this person's own clips best wins (0 = unused).
        blend_weights = BLEND_WEIGHTS if self.bsfeats is not None else (0.0,)
        t0, done, total = time.time(), 0, len(queries) * len(blend_weights)
        best, accuracy_by_weight = None, {}
        for bw in blend_weights:
            feats = self._template_feats(bw)
            dtw_rows = []
            for i in queries:
                row = dtw_distances(feats[i], feats)
                row[i] = np.inf
                dtw_rows.append(row)
                done += 1
                if progress and (done % 20 == 0 or done == total):
                    progress(done, total, time.time() - t0)
            dtw_rows = np.array(dtw_rows)
            scale = float(np.median(dtw_rows[np.isfinite(dtw_rows)]))
            for k in (1, 2):
                for lam in (0.0, 0.25, 0.5, 1.0):
                    dist = dtw_rows + lam * scale * dur_rows
                    scores = np.array([_class_scores(d, self.labels, n_cls, k) for d in dist])
                    acc = float((scores.argmin(1) == truth).mean())
                    accuracy_by_weight[bw] = max(acc, accuracy_by_weight.get(bw, 0.0))
                    if best is None or acc > best[0] + 1e-9:
                        best = (acc, k, lam * scale, scores, bw)
        acc, k, w, scores, bw = best
        self.params["blend_weight"] = float(bw)
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

        self.params = {"T": tau, "k": int(k), "dur_weight": float(w), "reject": reject, "floor": floor,
                       "blend_weight": float(bw)}
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
                       "confusions": dict(sorted(confusions.items(), key=lambda kv: -kv[1])),
                       "accuracy_by_blend_weight": {str(k_): v for k_, v in accuracy_by_weight.items()}}

    # -- prediction ---------------------------------------------------------

    @staticmethod
    def _relative(scores, floor):
        """(score - best) / best for each row; inf (no examples) becomes very large."""
        scores = np.atleast_2d(scores)
        best = scores.min(1, keepdims=True)
        rel = (scores - best) / np.maximum(best, max(floor, 1e-9))
        return np.where(np.isfinite(rel), rel, 1e3)

    def predict_signature(self, sig, bsig=None):
        feat, dur = sig
        scores = _class_scores(self._distances(feat, dur, bsig), self.labels, len(self.label_map),
                               self.params["k"])
        floor, tau = self.params.get("floor", 0.0), self.params["T"]
        logits = -self._relative(scores, floor)[0] / tau
        reject = self.params["reject"]
        reject_logit = -self._relative(np.append(scores, reject), floor)[0][-1] / tau \
            if np.isfinite(reject) else -np.inf
        top = max(logits.max(), reject_logit)
        e = np.exp(logits - top)
        total = e.sum() + (np.exp(reject_logit - top) if np.isfinite(reject_logit) else 0.0)
        return (e / total).astype(np.float64)   # may sum to < 1: the rest is "none of these"

    def predict_clip(self, clip, blendshapes=None):
        """Raw clip (T, 42, 3) [+ (T, 52) muscle scores] -> (probabilities, ms), or (None, ms)."""
        t0 = time.perf_counter()
        sig = raw_signature(clip)
        probs = None if sig is None else self.predict_signature(sig, blend_signature(blendshapes))
        return probs, (time.perf_counter() - t0) * 1000

    def predict_positions(self, positions):
        return np.stack([self.predict_signature(signature(p)) for p in positions])

    def describe(self):
        acc = self.report.get("loo_accuracy")
        per = np.bincount(self.labels, minlength=len(self.label_map))
        est = f", calibration estimate {acc:.0%}" if acc is not None else ""
        muscles = " + facial muscle scores" if self._uses_blend() else ""
        return (f"instant templates{muscles} ({len(self.label_map)} phrases, "
                f"{per.min()}-{per.max()} clips each{est})")

    # -- cache --------------------------------------------------------------

    def save(self, path, signature_hash):
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {} if self.bsfeats is None else {"bsfeats": self.bsfeats}
        np.savez(path, feats=self.feats, durs=self.durs, labels=self.labels,
                 label_map=np.array(self.label_map), params=np.array(json.dumps(self.params)),
                 report=np.array(json.dumps(self.report)), signature=np.array(signature_hash),
                 version=np.array(VERSION), **extra)

    @classmethod
    def load(cls, path, name="templates"):
        with np.load(path, allow_pickle=False) as z:
            if int(z["version"]) != VERSION:
                return None, None
            m = cls(z["feats"], z["durs"], z["labels"], [str(s) for s in z["label_map"]],
                    json.loads(str(z["params"])), json.loads(str(z["report"])), name=name,
                    bsfeats=z["bsfeats"] if "bsfeats" in z.files else None)
            return m, str(z["signature"])


def clips_hash(clips):
    h = hashlib.sha1(f"v{VERSION}".encode())
    for c in sorted(clips, key=lambda c: str(c.path)):
        st = c.path.stat()
        h.update(f"{c.path.as_posix()}|{st.st_size}|{st.st_mtime_ns}".encode())
        bs = blendshape_path(c.path)
        if bs.exists():
            h.update(f"bs|{bs.stat().st_size}|{bs.stat().st_mtime_ns}".encode())
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
    # Older clips have no muscle scores. Once every phrase has 3+ newer clips that
    # do, also try those alone with muscle scores, and keep whichever recognises
    # this person's clips better.
    with_scores = [c for c in clips if blendshape_path(c.path).exists()]
    if m.bsfeats is None and with_scores:
        per_phrase = {pid: sum(c.phrase_id == pid for c in with_scores) for pid in m.label_map}
        if min(per_phrase.values()) >= 3:
            alt = TemplateMatcher.from_clips(with_scores, phrase_ids, progress=progress,
                                             name=f"templates_{speaker}")
            better = (alt.report.get("loo_accuracy") or 0) > (m.report.get("loo_accuracy") or 0)
            if alt.label_map == m.label_map and alt._uses_blend() and better:
                alt.report["chosen_over"] = {"all_clips_without_muscle_scores": m.report.get("loo_accuracy")}
                m = alt
    m.save(path, sig)
    return m
