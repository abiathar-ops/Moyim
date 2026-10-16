"""Audio front-ends: waveform (32 kHz mono) -> 2-D time-frequency image.

Variants (compared honestly in scripts/05_compare_frontends.py):
  logmel       absolute log-mel in dB (keeps loudness -> can leak recording level)
  logmel_norm  log-mel standardised per clip (removes gain differences between sources)
  pcen         per-channel energy normalisation (suppresses stationary background noise)
  pcen_norm    PCEN followed by per-clip standardisation
"""
import numpy as np
import librosa

from . import config as C

_MEL_KW = dict(sr=C.SR, n_fft=C.N_FFT, hop_length=C.HOP, n_mels=C.N_MELS,
               fmin=C.FMIN, fmax=C.FMAX, center=True)


def mel_power(y):
    return librosa.feature.melspectrogram(y=y, power=2.0, **_MEL_KW)


def logmel(y):
    return librosa.power_to_db(mel_power(y), ref=1.0, amin=1e-10).astype(np.float32)


def clip_norm(X):
    return ((X - X.mean()) / (X.std() + 1e-6)).astype(np.float32)


def pcen(y):
    # librosa expects magnitude spectra of int-scaled audio
    S = librosa.feature.melspectrogram(y=y, power=1.0, **_MEL_KW)
    P = librosa.pcen(S * (2 ** 31), sr=C.SR, hop_length=C.HOP, **C.PCEN)
    return P.astype(np.float32)


FRONTENDS = {
    "logmel": logmel,
    "logmel_norm": lambda y: clip_norm(logmel(y)),
    "pcen": pcen,
    "pcen_norm": lambda y: clip_norm(pcen(y)),
}


def frontend(name):
    if name not in FRONTENDS:
        raise KeyError(f"unknown frontend {name!r}; options: {list(FRONTENDS)}")
    return FRONTENDS[name]


def activity_curve(y):
    """Per-frame 'something is calling' score.

    For each mel band in the bird range, subtract that band's median level
    (its noise floor) and keep only the excess. Summing the excess across
    bands gives a curve that rises when a transient call is present and stays
    low for stationary noise (insects, steady rain, hum).
    """
    S = librosa.feature.melspectrogram(y=y, sr=C.SR, n_fft=C.N_FFT, hop_length=C.HOP,
                                       n_mels=64, fmin=C.ACT_FMIN, fmax=C.ACT_FMAX, power=2.0)
    D = librosa.power_to_db(S, ref=1.0, amin=1e-10)
    excess = np.clip(D - np.median(D, axis=1, keepdims=True), 0, None).mean(axis=0)
    k = max(1, int(C.ACT_SMOOTH_SEC * C.SR / C.HOP))
    return np.convolve(excess, np.ones(k) / k, mode="same").astype(np.float32)
