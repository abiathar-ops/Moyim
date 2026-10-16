"""Loading cached audio and turning it into model inputs.

Framework-agnostic: works with plain numpy (scikit-learn) and becomes a
torch Dataset automatically when torch is installed.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import augment as A

try:
    from torch.utils.data import Dataset as _Base
except Exception:  # torch not installed - still usable as a plain sequence
    _Base = object


def load_audio(split, file_id):
    return np.load(C.AUDIO_CACHE / split / f"{file_id}.npy").astype(np.float32)


def load_activity(file_id):
    p = C.ACTIVITY_CACHE / f"{file_id}.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


def load_train_table(with_folds=True):
    meta = pd.read_csv(C.TRAIN_META)
    if with_folds and (C.OUT_DIR / "folds.csv").exists():
        meta = meta.merge(pd.read_csv(C.OUT_DIR / "folds.csv")[["file_id", "fold"]], on="file_id")
    meta["y"] = meta["label"].map(C.CLASS_TO_IDX)
    return meta


def eval_windows(y, n=C.WIN, hop=int(C.SR * C.EVAL_HOP_SEC)):
    """All sliding windows covering the clip (short clips are loop-padded)."""
    if len(y) <= n:
        return [A.loop_pad(y, n)]
    starts = list(range(0, len(y) - n + 1, hop))
    if starts[-1] != len(y) - n:
        starts.append(len(y) - n)          # make sure the tail is covered
    return [y[s:s + n] for s in starts]


def class_balanced_weights(labels):
    """Per-sample weights for a balanced sampler (inverse class frequency)."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=len(C.CLASSES)).astype(float)
    return (1.0 / counts[labels]) / (1.0 / counts[labels]).sum()


class BirdWindowDataset(_Base):
    """One random (activity-weighted) 5 s window per clip per epoch.

    train=True  -> activity-weighted crop + augmentation
    train=False -> most active window, no augmentation
    Returns (spectrogram[1, n_mels, frames] float32, class index).
    """

    def __init__(self, table, frontend="pcen_norm", train=True, noise_bank=None,
                 aug_cfg=A.AugConfig(), split="train", seed=C.SEED):
        self.t = table.reset_index(drop=True)
        self.fe = F.frontend(frontend)
        self.train, self.noise_bank, self.cfg, self.split = train, noise_bank, aug_cfg, split
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        row = self.t.iloc[i]
        y = load_audio(self.split, row.file_id)
        curve = load_activity(row.file_id)
        if self.train:
            w = A.choose_crop(y, curve, C.WIN, self.rng, p_random=self.cfg.p_crop_random)
            w = A.augment_wave(w, row.source_group, self.rng, self.noise_bank, self.cfg)
        else:
            w = A.best_crop(y, curve, C.WIN)
        S = self.fe(w)
        if self.train and self.rng.random() < self.cfg.p_specaug:
            S = A.spec_augment(S, self.rng)
        label = int(row.y) if "y" in row and not pd.isna(row.y) else -1
        return S[None].astype(np.float32), label
