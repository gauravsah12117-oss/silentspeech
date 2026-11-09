"""The lip-reading network and helpers to save, load and run it.

Main model (arch "conv_bigru", about 0.7 M parameters):

    input (64 frames x 160 features), standardised with training-set stats
    -> Conv1D(160->128, k5) + BatchNorm + ReLU + Dropout 0.3
    -> Conv1D(128->128, k5) + BatchNorm + ReLU
    -> BiGRU(hidden 128, 2 layers, dropout 0.3)
    -> attention pooling over time
    -> Linear(256->128) + ReLU + Dropout 0.3 -> Linear(128->N)   (logits)

Softmax is applied at prediction time; training uses cross-entropy on logits.
Comparison models for the experiments: "bigru" (no conv front-end) and
"transformer" (a small 2-layer Transformer encoder).
"""
import time

import numpy as np
import torch
from torch import nn

import config
from src.features import build_features


class AttentionPool(nn.Module):
    """Learns a weight per frame and returns the weighted average."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h):
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)
        return (w.unsqueeze(-1) * h).sum(dim=1)


class MeanPool(nn.Module):
    def forward(self, h):
        return h.mean(dim=1)


def _head(dim, n_classes, dropout):
    return nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(128, n_classes))


class _Base(nn.Module):
    """Holds feature mean/std as buffers so they travel with the weights."""

    def __init__(self, n_features):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(n_features))
        self.register_buffer("feat_std", torch.ones(n_features))

    def set_normalisation(self, mean, std):
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.as_tensor(np.maximum(std, 1e-6), dtype=torch.float32))

    def _standardise(self, x):
        return (x - self.feat_mean) / self.feat_std

    def enrolment_modules(self):
        """Modules that stay trainable during new-user enrolment."""
        return [self.pool, self.head]


class LipReader(_Base):
    def __init__(self, n_classes, n_features=160, use_conv=True, use_attention=True,
                 channels=128, hidden=128, dropout=0.3):
        super().__init__(n_features)
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Sequential(
                nn.Conv1d(n_features, channels, 5, padding=2), nn.BatchNorm1d(channels),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(channels, channels, 5, padding=2), nn.BatchNorm1d(channels), nn.ReLU(),
            )
        self.gru = nn.GRU(channels if use_conv else n_features, hidden, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.pool = AttentionPool(2 * hidden) if use_attention else MeanPool()
        self.head = _head(2 * hidden, n_classes, dropout)

    def forward(self, x):                      # x: (batch, frames, features)
        x = self._standardise(x)
        if self.use_conv:
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(x)
        return self.head(self.pool(h))


class TransformerReader(_Base):
    def __init__(self, n_classes, n_features=160, n_frames=config.TARGET_FRAMES,
                 use_attention=True, d_model=128, heads=4, layers=2, dropout=0.1):
        super().__init__(n_features)
        self.inp = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_frames, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=256,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.pool = AttentionPool(d_model) if use_attention else MeanPool()
        self.head = _head(d_model, n_classes, 0.3)

    def forward(self, x):
        h = self.encoder(self.inp(self._standardise(x)) + self.pos[:, : x.shape[1]])
        return self.head(self.pool(h))


ARCHS = ("conv_bigru", "bigru", "transformer")


def build_model(model_config):
    """model_config: arch, n_classes, use_velocity, use_attention, n_frames."""
    arch = model_config.get("arch", "conv_bigru")
    n_features = 160 if model_config.get("use_velocity", True) else 80
    kwargs = dict(n_classes=model_config["n_classes"], n_features=n_features,
                  use_attention=model_config.get("use_attention", True))
    if arch == "conv_bigru":
        return LipReader(use_conv=True, **kwargs)
    if arch == "bigru":
        return LipReader(use_conv=False, **kwargs)
    if arch == "transformer":
        return TransformerReader(n_frames=model_config.get("n_frames", config.TARGET_FRAMES), **kwargs)
    raise ValueError(f"Unknown arch {arch!r}; choose from {ARCHS}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path, model, model_config, label_map, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_config": model_config,
                "label_map": list(label_map), **extra}, path)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_proba(model, features, batch_size=256):
    """features (N, frames, F) numpy -> probabilities (N, classes) numpy."""
    model.eval()
    out = []
    for i in range(0, len(features), batch_size):
        logits = model(torch.from_numpy(np.ascontiguousarray(features[i:i + batch_size])))
        out.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(out) if out else np.zeros((0, len(model.head[-1].bias)))


class LipReaderRuntime:
    """Everything the live app needs: load once, then predict from raw clips."""

    kind = "neural"

    def __init__(self, path):
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self.path = path
        self.model, self.ckpt = load_checkpoint(path)
        self.label_map = self.ckpt["label_map"]
        self.use_velocity = self.ckpt["model_config"].get("use_velocity", True)
        self.n_frames = self.ckpt["model_config"].get("n_frames", config.TARGET_FRAMES)

    def describe(self):
        return f"neural model {getattr(self.path, 'name', self.path)} ({len(self.label_map)} phrases)"

    def predict_positions(self, positions):
        from src.features import to_features
        feats = np.stack([to_features(p, self.use_velocity, self.n_frames) for p in positions])
        return predict_proba(self.model, feats)

    def predict_clip(self, clip):
        """Raw clip (T, 42, 3) -> (probabilities, milliseconds), or (None, ms)."""
        t0 = time.perf_counter()
        feats = build_features(clip, self.use_velocity, self.n_frames)
        if feats is None:
            return None, (time.perf_counter() - t0) * 1000
        probs = predict_proba(self.model, feats[None])[0]
        return probs, (time.perf_counter() - t0) * 1000
