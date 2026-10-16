"""Training-time augmentation (never applied to validation or test).

Design follows the research review:
  * Crop selection by activity (BirdCLEF 2024 1st place, Moummad et al.):
    a random 5 s crop of a 10 s clip can land on silence while keeping the label.
  * Cross-group background mixing (Lasseck 2018 - the single biggest gain):
    quiet segments from OTHER source groups are mixed in, so a recording's
    background stops being a clue to its species.
  * Mild random EQ (simulates different microphones) - kept small because
    frequency cues identify species (Stowell 2022).
  * SpecAugment masking.
  * Mixup is available but OFF by default (it hurt the BirdCLEF 2024 winner).
  * No pitch shift / time stretch: they change species-defining cues.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


def loop_pad(y, n):
    """Repeat a short clip until it is n samples long (cyclic padding)."""
    if len(y) >= n:
        return y
    reps = int(np.ceil(n / max(len(y), 1)))
    return np.tile(y, reps)[:n]


def window_scores(curve, n_samples, stride_sec=0.25):
    """Mean activity of every candidate window start (in samples)."""
    win_f = max(1, int(round(n_samples / C.HOP)))
    stride_f = max(1, int(round(stride_sec * C.SR / C.HOP)))
    if len(curve) <= win_f:
        return np.array([0]), np.array([curve.mean() if len(curve) else 0.0])
    cs = np.concatenate([[0.0], np.cumsum(curve, dtype=np.float64)])
    starts_f = np.arange(0, len(curve) - win_f + 1, stride_f)
    scores = (cs[starts_f + win_f] - cs[starts_f]) / win_f
    return starts_f * C.HOP, scores


def choose_crop(y, curve, n, rng, p_random=0.2, temperature=0.5):
    """Pick an n-sample crop, favouring windows that contain calls.

    With probability p_random the crop is uniform-random (keeps diversity);
    otherwise windows are sampled with softmax(score / temperature * scale).
    """
    y = loop_pad(y, n)
    max_start = len(y) - n
    if max_start <= 0:
        return y[:n]
    if curve is None or rng.random() < p_random:
        s = int(rng.integers(0, max_start + 1))
        return y[s:s + n]
    starts, scores = window_scores(curve, n)
    starts = np.minimum(starts, max_start)
    z = (scores - scores.max()) / (scores.std() + 1e-6) / temperature
    p = np.exp(z); p /= p.sum()
    s = int(rng.choice(starts, p=p))
    return y[s:s + n]


def best_crop(y, curve, n):
    """Deterministic: the single most active window (for previews / embeddings)."""
    y = loop_pad(y, n)
    if curve is None or len(y) <= n:
        return y[:n]
    starts, scores = window_scores(curve, n)
    s = int(min(starts[int(np.argmax(scores))], len(y) - n))
    return y[s:s + n]


def rms(x):
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12)


def random_gain(y, rng, db=(-6.0, 6.0)):
    return y * 10 ** (rng.uniform(*db) / 20)


def random_eq(y, rng, max_db=4.0, n_points=6):
    """Smooth random gain curve over log-frequency, applied in the FFT domain."""
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1 / C.SR)
    knots = np.geomspace(50, C.SR / 2, n_points)
    gains_db = rng.uniform(-max_db, max_db, n_points)
    g = 10 ** (np.interp(np.log(np.maximum(f, 50)), np.log(knots), gains_db) / 20)
    return np.fft.irfft(Y * g, n=len(y)).astype(np.float32)


class NoiseBank:
    """Quiet, call-free segments cut from training clips (scripts/04)."""

    def __init__(self, table: pd.DataFrame, allowed_groups=None):
        t = table
        if allowed_groups is not None:
            t = t[t["source_group"].isin(set(allowed_groups))]
        self.table = t.reset_index(drop=True)
        self._cache = {}

    @classmethod
    def load(cls, allowed_groups=None):
        path = C.OUT_DIR / "noise_bank.csv"
        return cls(pd.read_csv(path), allowed_groups)

    def __len__(self):
        return len(self.table)

    def get(self, i):
        if i not in self._cache:
            self._cache[i] = np.load(C.NOISE_CACHE / self.table.at[i, "noise_file"]).astype(np.float32)
        return self._cache[i]

    def sample(self, n, rng, exclude_group=None):
        idx = np.arange(len(self.table))
        if exclude_group is not None:
            idx = idx[self.table["source_group"].values != exclude_group]
        if len(idx) == 0:
            return None
        x = loop_pad(self.get(int(rng.choice(idx))), n)
        s = int(rng.integers(0, len(x) - n + 1))
        return x[s:s + n]


def mix_noise(y, noise, snr_db):
    """Add noise so that signal RMS / noise RMS = snr_db."""
    scale = rms(y) / (rms(noise) * 10 ** (snr_db / 20))
    return (y + scale * noise).astype(np.float32)


def spec_augment(S, rng, n_time=2, max_t=40, n_freq=2, max_f=12):
    S = S.copy()
    fill = S.mean()
    F, T = S.shape
    for _ in range(n_freq):
        w = int(rng.integers(0, max_f + 1)); f0 = int(rng.integers(0, max(1, F - w)))
        S[f0:f0 + w, :] = fill
    for _ in range(n_time):
        w = int(rng.integers(0, max_t + 1)); t0 = int(rng.integers(0, max(1, T - w)))
        S[:, t0:t0 + w] = fill
    return S


def mixup(x1, y1, x2, y2, rng, alpha=0.4):
    lam = rng.beta(alpha, alpha)
    return lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2


@dataclass
class AugConfig:
    p_noise: float = 0.8          # cross-group background mixing
    snr_db: tuple = (3.0, 20.0)
    n_noise: int = 2              # up to this many noise layers
    p_gain: float = 0.5
    p_eq: float = 0.5
    eq_db: float = 4.0
    p_specaug: float = 0.5
    p_crop_random: float = 0.2
    use_mixup: bool = False       # off by default - test before enabling


def augment_wave(y, group, rng, noise_bank: NoiseBank = None, cfg: AugConfig = AugConfig()):
    """Waveform-level augmentation for one already-cropped window."""
    if noise_bank is not None and len(noise_bank) and rng.random() < cfg.p_noise:
        for _ in range(int(rng.integers(1, cfg.n_noise + 1))):
            nz = noise_bank.sample(len(y), rng, exclude_group=group)
            if nz is not None:
                y = mix_noise(y, nz, rng.uniform(*cfg.snr_db))
    if rng.random() < cfg.p_eq:
        y = random_eq(y, rng, cfg.eq_db)
    if rng.random() < cfg.p_gain:
        y = random_gain(y, rng)
    return np.clip(y, -1.0, 1.0).astype(np.float32)
