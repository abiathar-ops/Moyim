"""Step 2 - Decode every FLAC once: stereo 48 kHz -> mono 32 kHz float16 .npy

Why: decoding 5 GB of FLAC every epoch is the slowest part of training.
  * mono   - average of L and R (see channel_check.csv from step 1)
  * 32 kHz - exact 2/3 polyphase resampling (scipy.signal.resample_poly);
             keeps everything up to 16 kHz, which covers the bird range
  * level is NOT normalised here - that is a front-end choice tested in step 5

Also writes outputs/audio_stats.csv (rms, peak, exact-duplicate hash).
Re-running skips files that are already cached.
"""
import _common  # noqa: F401
import argparse
import hashlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from src import config as C


def process(job):
    split, file_id, src = job
    out = C.AUDIO_CACHE / split / f"{file_id}.npy"
    if out.exists():
        y = np.load(out).astype(np.float32)
    else:
        x, sr = sf.read(str(src), dtype="float32", always_2d=True)
        y = x.mean(axis=1)
        if sr != C.SR:
            if sr == C.ORIG_SR:
                y = resample_poly(y, C.RESAMPLE_UP, C.RESAMPLE_DOWN)
            else:
                from math import gcd
                g = gcd(C.SR, sr)
                y = resample_poly(y, C.SR // g, sr // g)
        y = y.astype(np.float16)
        np.save(out, y)
        y = y.astype(np.float32)
    return dict(split=split, file_id=file_id, n_samples=len(y), duration=len(y) / C.SR,
                rms_db=float(20 * np.log10(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12)),
                peak=float(np.abs(y).max()),
                md5=hashlib.md5(np.round(y, 3).tobytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    meta = pd.read_csv(C.TRAIN_META)
    sub = pd.read_csv(C.SAMPLE_SUB)
    jobs = [("train", f, C.DATA_DIR / p) for f, p in zip(meta.file_id, meta.audio_path)]
    jobs += [("test", f, C.TEST_AUDIO / f"{f}.flac") for f in sub.file_id]
    for s in ("train", "test"):
        (C.AUDIO_CACHE / s).mkdir(parents=True, exist_ok=True)

    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process, jobs, chunksize=8), 1):
            rows.append(r)
            if i % 250 == 0 or i == len(jobs):
                print(f"  cached {i}/{len(jobs)}", flush=True)

    st = pd.DataFrame(rows)
    st.to_csv(C.OUT_DIR / "audio_stats.csv", index=False)

    dup = st[st.duplicated("md5", keep=False)].sort_values("md5")
    n_tr = dup[dup.split == "train"].md5.nunique()
    cross = set(dup[dup.split == "train"].md5) & set(dup[dup.split == "test"].md5)
    print(f"\nexact duplicate audio: {len(dup)} files in {dup.md5.nunique()} sets "
          f"({n_tr} sets involve train, {len(cross)} sets shared between train and test)")
    if len(dup):
        dup.merge(meta[["file_id", "label", "source_group"]], on="file_id", how="left") \
           .to_csv(C.OUT_DIR / "duplicates.csv", index=False)
        print("  -> details in outputs/duplicates.csv")
    print("\nlevel (RMS dBFS) by split:")
    print(st.groupby("split").rms_db.describe()[["mean", "std", "min", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()
