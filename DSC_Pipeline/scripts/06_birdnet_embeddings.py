"""Step 6 - BirdNET v2.4 embeddings for every train and test clip.

BirdNET splits each clip into 3 s windows (48 kHz; it resamples and downmixes
internally) and returns a 1024-d embedding per window. With a 1.5 s overlap
the hop is 1.5 s, so a 10 s clip gives about 6 windows. No high-pass filter
(bandpass_fmin=0), for the same reason as FMIN in config.py: keep the
low-pitched Cendrawasih and possible Unknown calls.

Only the embeddings are used. BirdNET's own species predictions are not.

This runs as a script, not inside the notebook, because BirdNET starts worker
processes and those are unreliable when launched from a Jupyter kernel.

Outputs  outputs/03_birdnet/birdnet_windows.npz
         emb (n_windows, 1024) float32, file_id, win, split (one row per window)
Usage    python 06_birdnet_embeddings.py [--workers N] [--force]
"""
import _common  # noqa: F401
import argparse, os, time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

OUT = C.OUT_DIR / "03_birdnet"
CACHE = OUT / "birdnet_windows.npz"
OVERLAP_SEC = 1.5


def load_birdnet():
    import birdnet
    try:
        return birdnet.load("acoustic", "2.4", "tf")
    except ValueError as e:                      # TensorFlow not installed
        print("TensorFlow not available, using the TFLite runtime instead:", str(e).split(".")[0])
        return birdnet.load("acoustic", "2.4", "tf", library="litert")


def encode(model, paths, workers):
    res = model.encode([str(p) for p in paths], overlap_duration_s=OVERLAP_SEC,
                       bandpass_fmin=0, bandpass_fmax=C.FMAX,
                       batch_size=16, n_workers=workers, show_stats="minimal")
    bad = np.asarray(res.unprocessable_inputs())   # a method in birdnet 1.3; mask or indices
    if bad.dtype == bool:
        bad = np.flatnonzero(bad)
    if bad.size:
        raise RuntimeError(f"BirdNET could not read {bad.size} files, e.g. {[str(res.inputs[i]) for i in bad[:3]]}")
    emb = res.embeddings                          # (n_files, max_windows, 1024)
    valid = ~res.embeddings_masked.all(axis=2)    # mask is True on padding windows
    fids, wins, rows = [], [], []
    for i, p in enumerate(res.inputs):            # match by path, not by input order
        fid = Path(str(p)).stem
        for w in np.flatnonzero(valid[i]):
            fids.append(fid); wins.append(int(w)); rows.append(emb[i, w])
    return fids, wins, np.vstack(rows).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--force", action="store_true", help="re-extract even if the cache exists")
    a = ap.parse_args()
    if CACHE.exists() and not a.force:
        print("cache exists:", CACHE.relative_to(C.PIPELINE_DIR), "(use --force to re-extract)")
        return
    OUT.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(C.TRAIN_META)
    test_ids = pd.read_csv(C.SAMPLE_SUB).file_id
    test_paths = {p.stem: p for p in C.TEST_AUDIO.iterdir()}
    missing = [f for f in test_ids if f not in test_paths]
    if missing:
        raise FileNotFoundError(f"{len(missing)} test files missing, e.g. {missing[:3]}")

    model = load_birdnet()
    parts = []
    for split, paths in [("train", [C.DATA_DIR / p for p in meta.audio_path]),
                         ("test", [test_paths[f] for f in test_ids])]:
        t = time.time()
        fids, wins, E = encode(model, paths, a.workers)
        print(f"{split}: {len(paths)} clips -> {len(E)} windows in {time.time() - t:.0f} s")
        parts.append((fids, wins, [split] * len(fids), E))

    fids = np.concatenate([p[0] for p in parts]); wins = np.concatenate([p[1] for p in parts])
    split = np.concatenate([p[2] for p in parts]); E = np.vstack([p[3] for p in parts])
    expected = set(meta.file_id) | set(test_ids)
    got = set(fids)
    if expected - got:
        raise RuntimeError(f"{len(expected - got)} clips produced no windows")
    np.savez_compressed(CACHE, emb=E, file_id=fids.astype(str), win=wins, split=split.astype(str))
    print("saved", CACHE.relative_to(C.PIPELINE_DIR), E.shape)


if __name__ == "__main__":
    main()
