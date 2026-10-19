"""Step 4 - Where are the calls, and where is just background?

For every clip (train and test) we compute an activity curve: per-band
energy above that band's own noise floor (src/features.activity_curve).
  * Training crops are later sampled where the curve is high, so a 5 s window
    labelled "Kutilang" actually contains a Kutilang.
  * Quiet stretches (>= 1 s, bottom 30% of the curve) from TRAIN clips are cut
    out as a noise bank (Lasseck 2018). During training these are mixed into
    clips from other source groups, so backgrounds stop predicting species.

Outputs
  cache/activity/<file_id>.npy     activity curve (one value per 10 ms frame)
  cache/noise/<file_id>_<k>.npy    noise segments (train only)
  outputs/activity_summary.csv     per clip: activity ratio, peak score, SNR estimate
  outputs/noise_bank.csv           index of noise segments with label + source_group
"""
import _common  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import config as C
from src import features as F
from src.augment import window_scores
from src.dataset import load_audio


def runs_below(mask):
    """(start, end) frame index pairs of consecutive True values."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def process(split, file_id, label, group):
    y = load_audio(split, file_id)
    curve = F.activity_curve(y)
    np.save(C.ACTIVITY_CACHE / f"{file_id}.npy", curve.astype(np.float16))

    med = float(np.median(curve))
    mad = float(np.median(np.abs(curve - med))) + 1e-6
    thr = med + 3 * mad
    _, scores = window_scores(curve, C.WIN)
    n = len(y) // C.HOP * C.HOP
    frame_db = 10 * np.log10((y[:n].astype(np.float64) ** 2).reshape(-1, C.HOP).mean(1) + 1e-12)
    row = dict(split=split, file_id=file_id, label=label, source_group=group,
               duration=len(y) / C.SR, activity_ratio=float((curve > thr).mean()),
               peak_window_score=float(scores.max()), curve_median=med,
               snr_est_db=float(np.percentile(frame_db, 95) - np.percentile(frame_db, 10)))

    noise_rows = []
    if split == "train":
        quiet = curve <= np.quantile(curve, C.NOISE_QUANTILE)
        min_f = int(C.NOISE_MIN_SEC * C.SR / C.HOP)
        max_f = int(C.NOISE_MAX_SEC * C.SR / C.HOP)
        runs = sorted([r for r in runs_below(quiet) if r[1] - r[0] >= min_f],
                      key=lambda r: r[0] - r[1])[:C.NOISE_PER_CLIP]
        for k, (a, b) in enumerate(runs):
            if b - a > max_f:                       # keep the centre of long runs
                c = (a + b) // 2; a, b = c - max_f // 2, c + max_f // 2
            seg = y[a * C.HOP:b * C.HOP]
            name = f"{file_id}_{k}.npy"
            np.save(C.NOISE_CACHE / name, seg.astype(np.float16))
            noise_rows.append(dict(noise_file=name, file_id=file_id, label=label,
                                   source_group=group, duration=len(seg) / C.SR,
                                   rms_db=float(20 * np.log10(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-12))))
    return row, noise_rows


def main():
    for d in (C.ACTIVITY_CACHE, C.NOISE_CACHE):
        d.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(C.TRAIN_META)
    sub = pd.read_csv(C.SAMPLE_SUB)
    jobs = [("train", r.file_id, r.label, r.source_group) for r in meta.itertuples()]
    jobs += [("test", f, None, None) for f in sub.file_id]

    out = Parallel(n_jobs=4, batch_size=16, verbose=0)(delayed(process)(*j) for j in jobs)
    summ = pd.DataFrame([o[0] for o in out])
    noise = pd.DataFrame([r for o in out for r in o[1]])
    summ.to_csv(C.OUT_DIR / "activity_summary.csv", index=False)
    noise.to_csv(C.OUT_DIR / "noise_bank.csv", index=False)

    summ["label"] = summ["label"].fillna("(test)")
    print("activity summary by class (train) and test:")
    print(summ.groupby("label")[["activity_ratio", "peak_window_score", "snr_est_db"]]
              .median().round(3).to_string())
    print(f"\nnoise bank: {len(noise)} segments, {noise.duration.sum() / 60:.1f} min total, "
          f"from {noise.file_id.nunique()} clips in {noise.source_group.nunique()} groups")
    print(noise.groupby("label").agg(segments=("noise_file", "size"),
                                     minutes=("duration", lambda s: s.sum() / 60)).round(1).to_string())


if __name__ == "__main__":
    main()
