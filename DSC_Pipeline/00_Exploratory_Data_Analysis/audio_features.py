"""Per-file audio feature extraction (numpy / scipy / soundfile only)."""
from __future__ import annotations

import hashlib
import os
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, welch

import config as C

BAND_EDGES = [0, 200, 500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, np.inf]
BAND_NAMES = ["bf_0_200", "bf_200_500", "bf_500_1k", "bf_1k_2k", "bf_2k_4k",
              "bf_4k_6k", "bf_6k_8k", "bf_8k_12k", "bf_12k_16k", "bf_16k_up"]

_FB = None


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mel_pts)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / (c - lo)
        down = (hi - bins) / (hi - c)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    fb *= (2.0 / (hz[2:n_mels + 2] - hz[:n_mels]))[:, None]
    return fb


def _fb():
    global _FB
    if _FB is None:
        _FB = mel_filterbank(C.TARGET_SR, C.N_FFT, C.N_MELS, C.FMIN, C.FMAX)
    return _FB


def mel_centers_hz():
    pts = np.linspace(hz_to_mel(C.FMIN), hz_to_mel(C.FMAX), C.N_MELS + 2)
    return mel_to_hz(pts[1:-1])


def to_target_sr(x, sr):
    if int(sr) == C.TARGET_SR:
        return x
    g = gcd(int(sr), C.TARGET_SR)
    return resample_poly(x, C.TARGET_SR // g, int(sr) // g)


def _frames(x):
    n_fft, hop = C.N_FFT, C.HOP
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    x = np.ascontiguousarray(x, dtype=np.float64)
    n = 1 + (len(x) - n_fft) // hop
    return np.lib.stride_tricks.as_strided(
        x, (n, n_fft), (x.strides[0] * hop, x.strides[0]), writeable=False)


def power_spectrogram_and_rms(x):
    """Return (power[freq, time], frame_rms[time]). Power is normalised by window gain."""
    fr = _frames(x)
    rms = np.sqrt(np.mean(fr ** 2, axis=1))
    win = np.hanning(C.N_FFT)
    spec = np.fft.rfft(fr * win, axis=1)
    power = (np.abs(spec) ** 2).T / (win.sum() ** 2)
    return power, rms


def logmel_from_power(power):
    return 10.0 * np.log10(_fb() @ power + 1e-10)


def load_logmel(path):
    """Load a file and return (log-mel [n_mels, time], duration_seconds). Used for plots."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.astype(np.float64).mean(axis=1)
    y = to_target_sr(mono, sr)
    power, _ = power_spectrogram_and_rms(y)
    return logmel_from_power(power), len(mono) / sr


def _db(v, floor=1e-9):
    return float(20.0 * np.log10(max(float(v), floor)))


def analyze_file(job):
    """job = (file_id, path). Returns (record dict, (mel_mean, mel_std, mel_active_mean) or None)."""
    file_id, path = job
    rec = {"file_id": file_id, "error": ""}
    try:
        info = sf.info(path)
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:  # corrupt / unreadable
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec, None
    if data.shape[0] == 0:
        rec["error"] = "empty audio"
        return rec, None

    rec.update(sr=int(sr), channels=int(data.shape[1]), frames=int(data.shape[0]),
               duration=data.shape[0] / sr, subtype=info.subtype, format=info.format,
               file_bytes=os.path.getsize(path))

    mono = data.astype(np.float64).mean(axis=1)
    peak = float(np.abs(data).max())
    rms = float(np.sqrt(np.mean(mono ** 2)))
    rec.update(peak_dbfs=_db(peak), rms_dbfs=_db(rms), crest_db=_db(peak) - _db(rms),
               clip_frac=float((np.abs(data) >= 0.999).mean()),
               dc_offset=float(mono.mean()),
               pcm_md5=hashlib.md5(data.tobytes()).hexdigest())

    if data.shape[1] >= 2:
        L = data[:, 0].astype(np.float64)
        R = data[:, 1].astype(np.float64)
        ok = L.std() > 1e-9 and R.std() > 1e-9
        rec["stereo_corr"] = float(np.corrcoef(L, R)[0, 1]) if ok else np.nan
        rec["lr_identical"] = bool(np.max(np.abs(L - R)) < 1e-6)
        rec["lr_rms_diff_db"] = _db(np.sqrt(np.mean(L ** 2))) - _db(np.sqrt(np.mean(R ** 2)))
    else:
        rec["stereo_corr"] = np.nan
        rec["lr_identical"] = False
        rec["lr_rms_diff_db"] = np.nan

    # ---- native-rate spectrum (Welch) ----
    nper = int(min(4096, len(mono)))
    f, psd = welch(mono, fs=sr, nperseg=nper)
    tot = psd.sum() + 1e-20
    for name, lo, hi in zip(BAND_NAMES, BAND_EDGES[:-1], BAND_EDGES[1:]):
        rec[name] = float(psd[(f >= lo) & (f < hi)].sum() / tot)
    cen = float((f * psd).sum() / tot)
    rec["spec_centroid_hz"] = cen
    rec["spec_bandwidth_hz"] = float(np.sqrt((((f - cen) ** 2) * psd).sum() / tot))
    cs = np.cumsum(psd) / tot
    rec["rolloff85_hz"] = float(f[min(np.searchsorted(cs, 0.85), len(f) - 1)])
    rec["rolloff95_hz"] = float(f[min(np.searchsorted(cs, 0.95), len(f) - 1)])
    hi_mask = f >= 100
    rec["peak_freq_hz"] = float(f[hi_mask][np.argmax(psd[hi_mask])]) if hi_mask.any() else np.nan
    rec["lf_ratio"] = float(psd[f < 300].sum() / tot)
    rec["hf_ratio"] = float(psd[f > 10000].sum() / tot)
    rec["global_flatness"] = float(np.exp(np.mean(np.log(psd + 1e-20))) / (np.mean(psd) + 1e-20))

    # ---- frame-level features at the target rate ----
    y = to_target_sr(mono, sr)
    power, rms_f = power_spectrogram_and_rms(y)
    fdb = 20.0 * np.log10(rms_f + 1e-9)
    p95, p90, p50, p10 = np.percentile(fdb, [95, 90, 50, 10])
    thr = max(-70.0, p95 - 30.0)
    active = fdb > thr
    rec.update(noise_floor_db=float(p10), signal_db=float(p95), dyn_range_db=float(p95 - p10),
               snr_est_db=float(p90 - p10), silence_ratio=float(1.0 - active.mean()),
               n_frames=int(len(fdb)),
               rms_cv=float(np.std(rms_f) / (np.mean(rms_f) + 1e-12)))

    closed = np.convolve(active.astype(float), np.ones(10), mode="same") > 0   # merge gaps < 100 ms
    edges = np.diff(np.concatenate([[0], closed.astype(int)]))
    rec["n_segments"] = int((edges == 1).sum())
    rec["active_frac_closed"] = float(closed.mean())
    longest = cur = 0
    for v in ~closed:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    rec["longest_silence_s"] = longest * C.HOP / C.TARGET_SR

    # Events: frames clearly above this file's own noise floor (bird calls, bursts). Clips here are
    # loudness-normalised with continuous background noise, so "digital silence" never triggers;
    # the floor-relative rule is what separates calls from background.
    ev = fdb > (p10 + 6.0)
    ev_closed = np.convolve(ev.astype(float), np.ones(5), mode="same") > 0
    ev_edges = np.diff(np.concatenate([[0], ev_closed.astype(int)]))
    rec["event_ratio"] = float(ev.mean())
    rec["n_events"] = int((ev_edges == 1).sum())
    use = ev if ev.sum() >= 5 else (active if active.sum() >= 5 else np.ones_like(active))
    freqs = np.fft.rfftfreq(C.N_FFT, 1.0 / C.TARGET_SR)
    pu = power[:, use]
    cen_f = (freqs[:, None] * pu).sum(0) / (pu.sum(0) + 1e-20)
    rec["frame_centroid_hz"] = float(np.median(cen_f))
    band = (freqs >= 1000) & (freqs <= 8000)
    pb = pu[band] + 1e-20
    rec["flatness_active"] = float(np.median(np.exp(np.log(pb).mean(0)) / pb.mean(0)))

    # Periodic gating / stripe artifacts (seen in some recording sources): strong repetition of the
    # 8-14 kHz energy envelope at 0.4-2 s lags, and a "striped" long-term spectrum.
    hf = (freqs >= 8000) & (freqs <= 14000)
    env = 10.0 * np.log10(power[hf].sum(0) + 1e-12)
    rec["hf_env_depth_db"] = float(np.percentile(env, 90) - np.percentile(env, 10))
    e = env - env.mean()
    if len(e) > 250 and e.std() > 1e-6:
        ac = np.correlate(e, e, mode="full")[len(e) - 1:]
        ac = ac / ac[0]
        rec["hf_periodicity"] = float(ac[40:min(200, len(ac) - 1) + 1].max())
    else:
        rec["hf_periodicity"] = np.nan

    mel = logmel_from_power(power)
    mm = mel.mean(1)
    rec["mel_roughness_db"] = float(np.std((mm - np.convolve(mm, np.ones(5) / 5, mode="same"))[2:-2]))
    rec["mel_flux"] = float(np.mean(np.abs(np.diff(mel, axis=1)))) if mel.shape[1] > 1 else np.nan
    rec["zcr"] = float(np.mean(np.abs(np.diff(np.signbit(y).astype(np.int8)))))
    return rec, (mel.mean(1), mel.std(1), mel[:, use].mean(1))
