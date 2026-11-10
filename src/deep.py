"""Deep model: an ensemble of small neural networks trained on one person's own clips.

The instant templates (src/templates.py) compare a new clip with every stored
example. This model instead *learns* what tells this person's phrases apart:
several 1-D convolutional networks are trained on thousands of varied copies of
the same clips, and their answers are averaged.

Measured on real clips (s01: 58 clips of 10 phrases, 2026-11-10), 5-fold
cross-validation - every clip judged by networks that never saw it:

                         accuracy   spoken straight away (right)   "please repeat"
    instant templates       66%          9%  (100%)                     47%
    deep model              76%         43%  ( 96%)                     10%

Only 4-8 clips of each phrase is very little for deep learning, so everything
here is built to learn the movement rather than memorise the clips:
  - a small network (~39k weights) on the same 32-frame lip signature the
    templates use (lip shape relative to the clip's average, plus lip speed);
  - every clip is seen as BANK pre-made variations: mirrored, tilted, scaled,
    faster/slower, jittered, with dropped frames (src/augment.py);
  - dropout, weight decay, label smoothing, a cosine learning-rate schedule;
  - N_NETS networks from different random starts, averaged, and each clip is
    also judged as its mirror image;
  - confidence calibrated on held-out folds (temperature scaling), so "70%"
    means right about 70% of the time, like the templates' confidence.
Round 1 (a 2x larger network, 400 full-batch epochs) scored the same 76% but
took 3x longer to train; the scripts are summarised in README "Deep model".

The templates stay on beside it as a safety check (DeepRecognizer): when a
movement looks like none of the phrases, their "none of these" share lowers
the confidence, so the app asks to repeat instead of saying something wrong.
On real phrases that share was 0.99+ for 90% of clips, so it costs nothing.
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import config
from src.augment import augment, mirror
from src.dataset import scan_clips
from src.features import fill_missing, lip_motion, normalise
from src.templates import blend_signature, clips_hash, signature

VERSION = 1
N_NETS = 5             # networks in the ensemble
CV_NETS = 2            # networks per fold when estimating accuracy and calibrating
FOLDS = 5              # cross-validation folds
EPOCHS = 150
BATCH = 16
BANK = 64              # augmented copies of every clip
WIDTH = 48
DROPOUT = 0.3
LR = 3e-3
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.1
TEMPERATURES = np.round(np.arange(0.3, 3.01, 0.1), 2)
MIN_CLIPS = 2          # per phrase (the same rule as the templates)
SECONDS_PER_CLIP_NET = 0.24   # measured on a Celeron N4500 (2 cores); used only for the time estimate


def model_path(speaker, model_dir=None):
    return Path(model_dir or config.MODEL_DIR) / f"deep_{speaker}.pt"


class LipNet(nn.Module):
    """1x1 projection of the 160 lip features, three 1-D convolutions over time
    (the middle one dilated: it sees 9 frames), then average + max pooling."""

    def __init__(self, n_in, n_cls, width=WIDTH, p=DROPOUT):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(n_in, width, 1), nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(p),
            nn.Conv1d(width, width, 5, padding=4, dilation=2), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(p),
            nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width), nn.GELU())
        self.head = nn.Linear(2 * width, n_cls)
        self.p = p

    def forward(self, x):                          # (batch, frames, features) -> (batch, phrases)
        h = self.body(x.transpose(1, 2))
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        return self.head(F.dropout(h, self.p, self.training))


# --- features -----------------------------------------------------------------

def clip_positions(clip):
    """Raw clip (T, 42, 3) -> normalised lip positions, or None if the face was lost."""
    filled = fill_missing(np.asarray(clip, dtype=np.float32))
    if filled is None or len(filled) < 4:
        return None
    return normalise(filled)


def features(positions):
    return signature(positions)[0].astype(np.float32)          # (32, 160)


def make_bank(positions, copies=BANK, seed=0, progress=None):
    """Clean, mirrored and `copies` augmented feature sets for every clip."""
    rng = np.random.default_rng(seed)
    clean = np.stack([features(p) for p in positions])
    mirrored = np.stack([features(mirror(p)) for p in positions])
    bank = np.empty((len(positions), copies) + clean.shape[1:], np.float32)
    for i, p in enumerate(positions):
        for j in range(copies):
            bank[i, j] = features(augment(p, rng, dict(config.AUGMENT)))
        if progress:
            progress(i + 1, len(positions))
    return clean, mirrored, bank


def softmax(z):
    z = np.asarray(z, np.float64)
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def ensemble_probs(logits, mirrored_logits=None, temperature=1.0):
    """Average of the networks' (and mirror images') probabilities. logits: (nets, ..., phrases)."""
    p = softmax(np.asarray(logits) / temperature).mean(0)
    if mirrored_logits is not None:
        p = 0.5 * p + 0.5 * softmax(np.asarray(mirrored_logits) / temperature).mean(0)
    return p


def fit_temperature(logits, mirrored_logits, truth):
    """Temperature with the lowest log-loss on held-out predictions."""
    rows = np.arange(len(truth))

    def nll(t):
        p = ensemble_probs(logits, mirrored_logits, t)[rows, truth]
        return -np.mean(np.log(np.clip(p, 1e-9, 1)))

    return float(min(TEMPERATURES, key=nll))


def stratified_folds(labels, k, seed=0):
    """Fold number per clip, with every phrase spread evenly over the folds."""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(labels), int)
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        fold[idx] = np.arange(len(idx)) % k
    return fold


def train_net(clean, bank, labels, n_cls, seed, epochs=EPOCHS, batch=BATCH, tick=None):
    """One network on these clips. Returns (net, mean, std) - inputs are standardised."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dim = clean.shape[-1]
    mu = clean.reshape(-1, dim).mean(0)
    sd = clean.reshape(-1, dim).std(0) + 1e-6
    xb = torch.as_tensor((bank - mu) / sd, dtype=torch.float32)        # (clips, copies, 32, 160)
    yt = torch.as_tensor(labels, dtype=torch.long)
    net = LipNet(dim, n_cls)
    n = len(labels)
    steps = epochs * max(1, -(-n // batch))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    net.train()
    for e in range(epochs):
        order = rng.permutation(n)
        pick = rng.integers(bank.shape[1], size=n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            if len(idx) < 4 and n >= 4:      # a tiny last batch upsets batch normalisation
                continue
            rows = torch.as_tensor(idx)
            loss = F.cross_entropy(net(xb[rows, torch.as_tensor(pick[idx])]), yt[rows],
                                   label_smoothing=LABEL_SMOOTHING)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if tick and (e + 1) % 10 == 0:
            tick(10)
    if tick and epochs % 10:
        tick(epochs % 10)
    net.eval()
    return net, mu.astype(np.float32), sd.astype(np.float32)


def net_logits(nets, mu, sd, feats):
    x = torch.as_tensor((np.asarray(feats, np.float32) - mu) / sd, dtype=torch.float32)
    with torch.no_grad():
        return np.stack([net(x).numpy() for net in nets])


# --- the model ----------------------------------------------------------------

class DeepModel:
    """N_NETS trained networks + their calibration and the honest accuracy estimate."""

    def __init__(self, nets, mu, sd, label_map, temperature, report, oof=None):
        self.nets, self.mu, self.sd = nets, np.asarray(mu, np.float32), np.asarray(sd, np.float32)
        self.label_map = list(label_map)
        self.temperature = float(temperature)
        self.report = dict(report)
        self.oof = oof                 # (held-out probabilities (N, C), true labels (N,)) from cross-validation

    def predict_positions(self, positions):
        """Normalised lip positions -> probabilities over label_map (they sum to 1)."""
        f = features(positions)[None]
        m = features(mirror(positions))[None]
        p = ensemble_probs(net_logits(self.nets, self.mu, self.sd, f),
                           net_logits(self.nets, self.mu, self.sd, m), self.temperature)
        return p[0]

    def predict_clip(self, clip, blendshapes=None):
        t0 = time.perf_counter()
        pos = clip_positions(clip)
        probs = None if pos is None else self.predict_positions(pos)
        return probs, (time.perf_counter() - t0) * 1000

    def subset_accuracy(self, phrase_ids):
        """Cross-validated accuracy when only these phrases are allowed (focus mode)."""
        if self.oof is None:
            return None
        probs, truth = self.oof
        cls = [self.label_map.index(p) for p in phrase_ids if p in self.label_map]
        sel = np.isin(truth, cls)
        if not sel.any():
            return None
        pred = np.array(cls)[probs[sel][:, cls].argmax(1)]
        return float((pred == truth[sel]).mean())

    def rank_phrases(self):
        """Phrases from most to least reliable, with the accuracy of each top-n set.

        Greedy, like the templates: keep dropping the phrase whose removal helps
        the held-out accuracy most. Returns [(phrase_id, accuracy up to it)].
        """
        current, removed = list(self.label_map), []
        while len(current) > 2:
            drop = max(current, key=lambda c: self.subset_accuracy([x for x in current if x != c]) or 0)
            removed.append((drop, self.subset_accuracy(current)))
            current = [x for x in current if x != drop]
        ranking = [(pid, self.subset_accuracy(current[:i + 1])) for i, pid in enumerate(current)]
        return ranking + list(reversed(removed))

    # -- files --------------------------------------------------------------

    def save(self, path, signature_hash=""):
        """Write atomically: an interrupted save never leaves a broken model behind."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({
            "version": VERSION, "kind": "deep_ensemble", "label_map": self.label_map,
            "n_in": int(self.mu.shape[-1]), "width": WIDTH, "dropout": DROPOUT,
            "nets": [net.state_dict() for net in self.nets],
            "mu": torch.as_tensor(self.mu), "sd": torch.as_tensor(self.sd),
            "temperature": self.temperature, "report": self.report, "signature": signature_hash,
            "oof_probs": None if self.oof is None else torch.as_tensor(self.oof[0]),
            "oof_labels": None if self.oof is None else torch.as_tensor(self.oof[1]),
        }, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        """(model, signature hash), or (None, None) if missing or from another version."""
        path = Path(path)
        if not path.exists():
            return None, None
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck.get("version") != VERSION:
            return None, None
        nets = []
        for state in ck["nets"]:
            net = LipNet(ck["n_in"], len(ck["label_map"]), ck["width"], ck["dropout"])
            net.load_state_dict(state)
            nets.append(net.eval())
        oof = None
        if ck.get("oof_probs") is not None:
            oof = (ck["oof_probs"].numpy(), ck["oof_labels"].numpy())
        return cls(nets, ck["mu"].numpy(), ck["sd"].numpy(), ck["label_map"], ck["temperature"],
                   ck["report"], oof), ck.get("signature", "")


# --- training -----------------------------------------------------------------

def train(positions, labels, label_map, n_nets=N_NETS, cv_nets=CV_NETS, folds=FOLDS, epochs=EPOCHS,
          bank_copies=BANK, compare=None, progress=None, seed=0):
    """Train the ensemble on these clips and measure it honestly.

    positions: normalised clips; labels: index into label_map for each clip.
    compare(train_idx, test_idx) -> template probabilities for test_idx (optional),
    so the templates are scored on exactly the same held-out clips.
    progress(stage, done, total): optional callback for a progress bar.
    """
    t_start = time.time()
    labels = np.asarray(labels, int)
    n_cls = len(label_map)
    counts = np.bincount(labels, minlength=n_cls)
    if n_cls < 2 or counts.min() < MIN_CLIPS:
        raise ValueError(f"every phrase needs {MIN_CLIPS}+ clips (fewest: {counts.min()})")
    say = progress or (lambda *a: None)
    clean, mirrored, bank = make_bank(positions, bank_copies, seed,
                                      progress=lambda i, n: say("varying the clips", i, n))
    # Every phrase has 2+ clips, so each fold's training part still has every phrase,
    # even when a phrase has fewer clips than there are folds.
    folds = max(2, min(folds, len(labels) // 2)) if folds else 0
    report = {"n_clips": int(len(labels)), "clips_per_phrase": [int(c) for c in counts], "n_nets": n_nets,
              "epochs": epochs, "bank": bank_copies, "folds": folds, "architecture": f"LipNet width {WIDTH}"}
    fold = stratified_folds(labels, folds, seed) if folds else None

    tmpl = None
    if folds and compare is not None:
        tmpl = np.zeros((len(labels), n_cls))
        for f in range(folds):
            say("scoring the templates the same way", f, folds)
            tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
            tmpl[te] = compare(tr, te)
        say("scoring the templates the same way", folds, folds)

    total = (folds * cv_nets + n_nets) * epochs
    done = [0]

    def tick(n):
        done[0] += n
        say("training the neural networks", done[0], total)

    oof, temperature = None, 1.0
    if folds:
        logits = np.zeros((cv_nets, len(labels), n_cls))
        mlogits = np.zeros_like(logits)
        for f in range(folds):
            tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
            for s in range(cv_nets):
                net, mu, sd = train_net(clean[tr], bank[tr], labels[tr], n_cls, seed + 100 * f + s, epochs,
                                        tick=tick)
                logits[s, te] = net_logits([net], mu, sd, clean[te])[0]
                mlogits[s, te] = net_logits([net], mu, sd, mirrored[te])[0]
        temperature = fit_temperature(logits, mlogits, labels)
        probs = ensemble_probs(logits, mlogits, temperature)
        pred = probs.argmax(1)
        conf = probs.max(1)
        sure = conf >= config.CONF_ACCEPT
        confusions = {}
        for t_, p_ in zip(labels, pred):
            if t_ != p_:
                key = f"{label_map[t_]}>{label_map[p_]}"
                confusions[key] = confusions.get(key, 0) + 1
        report.update({
            "cv_accuracy": float((pred == labels).mean()), "temperature": temperature,
            "mean_confidence": float(conf.mean()),
            "spoken_directly": float(sure.mean()),
            "spoken_directly_right": float((pred == labels)[sure].mean()) if sure.any() else None,
            "asks_to_repeat": float((conf < config.CONF_MIN).mean()),
            "per_phrase": {pid: float((pred[labels == c] == c).mean()) for c, pid in enumerate(label_map)},
            "confusions": dict(sorted(confusions.items(), key=lambda kv: -kv[1]))})
        if tmpl is not None:
            t_pred = tmpl.argmax(1)
            report["templates_cv_accuracy"] = float((t_pred == labels).mean())
            report["agree_with_templates"] = float((t_pred == pred).mean())
        oof = (probs.astype(np.float32), labels.astype(np.int64))

    nets, mu, sd = [], None, None
    for s in range(n_nets):
        net, mu, sd = train_net(clean, bank, labels, n_cls, seed + 1000 + s, epochs, tick=tick)
        nets.append(net)
    report["train_seconds"] = round(time.time() - t_start, 1)
    report["trained"] = time.strftime("%Y-%m-%d %H:%M")
    return DeepModel(nets, mu, sd, label_map, temperature, report, oof)


def _speaker_clips(speaker, phrase_ids, raw_dir=None):
    """(usable clips, their positions, label_map, all scanned clips, [left-out clip names]).

    Like the templates, clips where the lips barely moved (a missed cue) are left
    out: the live app never classifies such a clip, so the networks should not
    learn from it or be scored on it. Every phrase keeps at least MIN_CLIPS.
    """
    clips = scan_clips(raw_dir, phrase_ids=set(phrase_ids), speakers={speaker})
    loaded = []
    for c in clips:
        pos = clip_positions(np.load(c.path))
        if pos is not None:
            loaded.append((c, pos, lip_motion(pos)))
    dropped = set()
    for pid in {c.phrase_id for c, _, _ in loaded}:
        mine = sorted((m, i) for i, (c, _, m) in enumerate(loaded) if c.phrase_id == pid)
        for m, i in mine[:max(0, len(mine) - MIN_CLIPS)]:
            if m < config.MIN_LIP_MOTION:
                dropped.add(i)
    keep = [c for i, (c, _, _) in enumerate(loaded) if i not in dropped]
    positions = [p for i, (_, p, _) in enumerate(loaded) if i not in dropped]
    left_out = [f"{loaded[i][0].phrase_id} clip {loaded[i][0].session}_{loaded[i][0].rep:02d}" for i in sorted(dropped)]
    label_map = [pid for pid in phrase_ids if any(c.phrase_id == pid for c in keep)]
    return keep, positions, label_map, clips, left_out


def estimate_seconds(speaker, phrase_ids, raw_dir=None, n_nets=N_NETS, cv_nets=CV_NETS, folds=FOLDS):
    """Rough training time on a slow laptop (faster computers finish sooner)."""
    n = len(scan_clips(raw_dir, phrase_ids=set(phrase_ids), speakers={speaker}))
    nets = folds * cv_nets * (1 - 1 / max(folds, 1)) + n_nets
    return n * (SECONDS_PER_CLIP_NET * nets * EPOCHS / 150 + 0.1) + 10


def train_for_speaker(speaker, phrase_ids, raw_dir=None, model_dir=None, progress=None, compare_templates=True,
                      **kw):
    """Train on all of this speaker's clips of these phrases; save models/deep_<speaker>.pt.

    phrase_ids: the phrases in display order (the app passes every phrase; only
    those with clips are used, exactly as the templates do). Returns the model,
    or None when there are no clips. Raises ValueError when a phrase has too few.
    """
    from src.templates import TemplateMatcher
    keep, positions, label_map, scanned, left_out = _speaker_clips(speaker, phrase_ids, raw_dir)
    if not keep:
        return None
    labels = np.array([label_map.index(c.phrase_id) for c in keep])
    counts = np.bincount(labels, minlength=len(label_map))
    if len(label_map) < 2:
        raise ValueError("it needs clips of at least 2 phrases")
    few = [pid for pid, n in zip(label_map, counts) if n < MIN_CLIPS]
    if few:
        raise ValueError(f"every phrase needs {MIN_CLIPS}+ clips; {', '.join(few)} "
                         f"{'has' if len(few) == 1 else 'have'} fewer (train_more.bat records more)")

    def compare(tr, te):
        m = TemplateMatcher.from_clips([keep[i] for i in tr], label_map)
        out = np.zeros((len(te), len(label_map)))
        idx = [label_map.index(p) for p in m.label_map]
        for row, i in enumerate(te):
            out[row, idx] = m.predict_signature(signature(positions[i]))
        return out

    model = train(positions, labels, label_map, compare=compare if compare_templates else None,
                  progress=progress, **kw)
    model.report["speaker"] = speaker
    model.report["left_out"] = left_out
    if "templates_cv_accuracy" in model.report:
        model.report["better_than_templates"] = model.report["cv_accuracy"] >= model.report["templates_cv_accuracy"]
    model.save(model_path(speaker, model_dir), clips_hash(scanned))
    return model


def up_to_date(speaker, phrase_ids, raw_dir=None, model_dir=None):
    """The saved deep model if it was trained on exactly the clips there are now, else None."""
    try:
        model, sig = DeepModel.load(model_path(speaker, model_dir))
    except Exception:
        return None
    if model is None:
        return None
    clips = scan_clips(raw_dir, phrase_ids=set(phrase_ids), speakers={speaker})
    return model if clips and sig == clips_hash(clips) else None


def load_for_speaker(speaker, label_map, model_dir=None):
    """(model, note): the saved deep model if it knows exactly these phrases and beat the templates."""
    path = model_path(speaker, model_dir)
    if not path.exists():
        return None, "no deep model yet (train_deep.bat trains one)"
    try:
        model, _ = DeepModel.load(path)
    except Exception as e:  # damaged file, or saved by an incompatible PyTorch
        return None, f"the deep model could not be read ({e}); train_deep.bat makes a new one"
    if model is None:
        return None, "the deep model is from an older version; train_deep.bat makes a new one"
    if list(model.label_map) != list(label_map):
        return None, "the deep model was trained on other phrases; train_deep.bat updates it"
    if not model.report.get("better_than_templates", True):
        return None, (f"the deep model ({model.report.get('cv_accuracy', 0):.0%}) did not beat the templates "
                      f"({model.report.get('templates_cv_accuracy', 0):.0%}) on your clips, so it is not used")
    return model, "deep model"


# --- deep model + templates ------------------------------------------------------

class DeepRecognizer:
    """The deep model decides; the templates add a "none of these" safety check.

    probability = deep model's probability x the templates' share for "one of
    my phrases". For a real phrase that share is close to 1 and changes nothing;
    for a yawn, a laugh or a phrase not in the list it drops, and the app asks
    to repeat instead of speaking.
    """
    kind = "deep"

    def __init__(self, model, templates):
        self.model, self.templates = model, templates
        self.label_map = list(templates.label_map)       # all phrases, or the focus subset
        self._idx = [model.label_map.index(pid) for pid in self.label_map]
        acc = model.report.get("cv_accuracy")
        if len(self.label_map) < len(model.label_map):
            acc = model.subset_accuracy(self.label_map)
        self.report = {**templates.report, "loo_accuracy": acc, "deep": model.report}
        self.name = f"deep+{templates.name}"
        self.last_votes = None

    def predict_clip(self, clip, blendshapes=None):
        t0 = time.perf_counter()
        pos = clip_positions(clip)
        if pos is None:
            return None, (time.perf_counter() - t0) * 1000
        p_tmpl = self.templates.predict_signature(signature(pos), blend_signature(blendshapes))
        p_deep = self.model.predict_positions(pos)[self._idx]
        p_deep = p_deep / max(p_deep.sum(), 1e-12)
        inside = float(np.clip(p_tmpl.sum(), 0.0, 1.0))
        t_norm = p_tmpl / max(p_tmpl.sum(), 1e-12)
        self.last_votes = {"deep": (self.label_map[int(p_deep.argmax())], float(p_deep.max())),
                           "templates": (self.label_map[int(t_norm.argmax())], float(t_norm.max())),
                           "inside": inside}
        return p_deep * inside, (time.perf_counter() - t0) * 1000

    def restrict(self, phrase_ids, progress=None):
        return DeepRecognizer(self.model, self.templates.restrict(phrase_ids, progress=progress))

    def rank_phrases(self):
        return self.model.rank_phrases() if self.model.oof is not None else self.templates.rank_phrases()

    def describe(self):
        r = self.model.report
        est = f", {r['cv_accuracy']:.0%} on clips it never saw" if r.get("cv_accuracy") is not None else ""
        vs = f" vs templates {r['templates_cv_accuracy']:.0%}" if r.get("templates_cv_accuracy") is not None else ""
        return (f"deep model: {len(self.model.nets)} neural networks + template safety check "
                f"({len(self.label_map)} phrases{est}{vs})")
