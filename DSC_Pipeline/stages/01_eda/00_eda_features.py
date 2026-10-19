"""Step 0 - EDA features, computed from the ORIGINAL 48 kHz FLAC files.

Nothing here depends on preprocessing decisions: we look at the raw data so
the later choices (sample rate, frequency range, noise handling, validation)
can be justified from evidence.

Per clip (train and test):
  quality    duration, RMS level, peak, clipping fraction, DC offset
  spectrum   mean / 10th-percentile ("background floor") / 95th-percentile
             ("loud events") power per frequency, 0-24 kHz
  bands      share of energy <0.5k, 0.5-2k, 2-8k, 8-16k, >16 kHz
  texture    spectral centroid, flatness, onset rate, SNR estimate
  noise      pulse_strength (rhythmic insect-like pulses), low_share_300 (wind/rumble)
  summary    128-band log-mel mean + std (for maps and train-vs-test checks)

Outputs (outputs/eda/)
  clip_stats.csv   one row per clip
  spectra.npz      freqs + mean/floor/p95 spectra (rows aligned with clip_stats)
  mel_summary.npy  256-dim log-mel summary per clip (rows aligned with clip_stats)
"""
import _common  # noqa: F401
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from joblib import Parallel, delayed

from src import config as C

OUT = C.OUT_DIR / "eda"
SR = 48_000
N_FFT, HOP = 2048, 512
BANDS = [(0, 500), (500, 2000), (2000, 8000), (8000, 16000), (16000, 24000)]
FREQS = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-14))


def one(split, file_id, path, label, group):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SR
    y = x.mean(axis=1)
    P = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    mean_s = P.mean(axis=1)
    tot = mean_s.sum() + 1e-14
    frame_e = P.sum(axis=0)
    onset_env = librosa.onset.onset_strength(S=librosa.power_to_db(P), sr=SR, hop_length=HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR, hop_length=HOP)
    # rhythmic pulse noise (insect-like): periodicity of 8-16 kHz energy, lag 0.3-2 s
    hb = np.log(P[(FREQS >= 8000) & (FREQS < 16000)].sum(0) + 1e-12); hb -= hb.mean()
    ac = np.correlate(hb, hb, "full")[len(hb) - 1:]; ac /= ac[0] + 1e-12
    lo, hi = int(0.3 * SR / HOP), min(int(2.0 * SR / HOP), len(ac))
    k = lo + int(np.argmax(ac[lo:hi])) if hi > lo else lo
    pulse_strength = float(ac[k]) if hi > lo else 0.0
    M = librosa.power_to_db(librosa.feature.melspectrogram(S=P, sr=SR, n_mels=128, fmin=0, fmax=SR / 2))
    row = dict(
        split=split, file_id=file_id, label=label, source_group=group,
        duration=len(y) / SR,
        rms_db=float(db(np.mean(y.astype(np.float64) ** 2))),
        peak=float(np.abs(x).max()),
        clip_frac=float((np.abs(x) >= 0.999).mean()),
        dc_offset=float(y.mean()),
        snr_est_db=float(np.percentile(db(frame_e), 95) - np.percentile(db(frame_e), 10)),
        centroid_hz=float((FREQS[:, None] * P).sum(0).mean() / (P.sum(0).mean() + 1e-14)),
        flatness=float(librosa.feature.spectral_flatness(S=np.sqrt(P)).mean()),
        onset_rate=len(onsets) / (len(y) / SR),
        pulse_strength=pulse_strength, pulse_period_s=k * HOP / SR,
        low_share_300=float(mean_s[FREQS < 300].sum() / tot),
        **{f"band_{a / 1000:g}k_{b / 1000:g}k": float(mean_s[(FREQS >= a) & (FREQS < b)].sum() / tot)
           for a, b in BANDS},
    )
    spec = np.stack([db(mean_s), db(np.percentile(P, 10, axis=1)), db(np.percentile(P, 95, axis=1))]).astype(np.float32)
    mel = np.concatenate([M.mean(1), M.std(1)]).astype(np.float32)
    return row, spec, mel


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(C.TRAIN_META)
    sub = pd.read_csv(C.SAMPLE_SUB)
    jobs = [("train", r.file_id, C.DATA_DIR / r.audio_path, r.label, r.source_group) for r in meta.itertuples()]
    jobs += [("test", f, C.TEST_AUDIO / f"{f}.flac", None, None) for f in sub.file_id]
    res = Parallel(n_jobs=4, batch_size=8)(delayed(one)(*j) for j in jobs)
    stats = pd.DataFrame([r[0] for r in res])
    stats.to_csv(OUT / "clip_stats.csv", index=False)
    S = np.stack([r[1] for r in res])
    np.savez_compressed(OUT / "spectra.npz", freqs=FREQS.astype(np.float32),
                        mean=S[:, 0], floor=S[:, 1], p95=S[:, 2])
    np.save(OUT / "mel_summary.npy", np.stack([r[2] for r in res]))
    print(f"EDA features for {len(stats)} clips -> {OUT}")
    stats["label"] = stats["label"].fillna("(test)")
    cols = ["duration", "rms_db", "snr_est_db", "centroid_hz", "flatness", "onset_rate",
            "band_0k_0.5k", "band_0.5k_2k", "band_2k_8k", "band_8k_16k", "band_16k_24k"]
    cols = [c for c in cols if c in stats]
    print(stats.groupby("label")[cols].median().round(3).T.to_string())


if __name__ == "__main__":
    main()
