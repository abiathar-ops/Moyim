"""Step 1: scan every audio file once and store per-file features.

Outputs (outputs/tables/):
    features.csv          one row per file: split, label, source_group + audio features
    mel_stats.npz         log-mel summary vectors (mean / std / active-frame mean) per file
    file_inventory.json   missing / extra files and sample_submission consistency
"""
import argparse
import hashlib
import json
import pickle
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import audio_features as AF
import config as C


def build_jobs(limit):
    meta = pd.read_csv(C.ROOT / "train" / "metadata.csv")
    sub = pd.read_csv(C.ROOT / "sample_submission.csv", keep_default_na=False)
    inv = {"train_rows": int(len(meta)), "submission_rows": int(len(sub)),
           "submission_unique_ids": int(sub["file_id"].nunique())}

    train_dir = C.ROOT / "train" / "audio"
    test_dir = C.ROOT / "test" / "audio"
    train_on_disk = {p.stem for p in train_dir.glob("*.flac")} if train_dir.exists() else set()
    test_on_disk = {p.stem for p in test_dir.glob("*.flac")} if test_dir.exists() else set()

    meta["path"] = meta["audio_path"].map(lambda p: str(C.ROOT / p))
    meta_ids = set(meta["file_id"])
    sub_ids = set(sub["file_id"])
    inv.update(
        train_missing_files=sorted(meta_ids - train_on_disk)[:50],
        n_train_missing_files=len(meta_ids - train_on_disk),
        n_train_extra_files=len(train_on_disk - meta_ids),
        n_test_on_disk=len(test_on_disk),
        n_test_missing_files=len(sub_ids - test_on_disk),
        n_test_extra_files=len(test_on_disk - sub_ids),
        submission_label_values=sorted(map(str, sub["label"].unique()))[:10],
        train_label_values=sorted(map(str, meta["label"].unique())),
        n_duplicate_train_ids=int(meta["file_id"].duplicated().sum()),
        n_null_train_labels=int(meta["label"].isna().sum()),
        n_null_train_groups=int(meta["source_group"].isna().sum()),
    )

    meta = meta[meta["file_id"].isin(train_on_disk)]
    test_ids = sorted(sub_ids & test_on_disk)
    if limit:
        meta = meta.sample(min(limit, len(meta)), random_state=C.SEED)
        test_ids = list(pd.Series(test_ids).sample(min(limit, len(test_ids)), random_state=C.SEED))

    jobs = [(r.file_id, r.path) for r in meta.itertuples()]
    jobs += [(i, str(test_dir / f"{i}.flac")) for i in test_ids]
    info = pd.concat([
        meta[["file_id", "label", "source_group"]].assign(split="train"),
        pd.DataFrame({"file_id": test_ids}).assign(split="test", label=np.nan, source_group=np.nan),
    ], ignore_index=True)
    return jobs, info, inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only scan N random files per split (smoke test)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="recompute even if features.csv exists")
    args = ap.parse_args()

    C.seed_everything()
    C.ensure_dirs()
    out_csv = C.TABLES / "features.csv"
    if out_csv.exists() and not args.force:
        print(f"[01] {out_csv} exists, skipping (use --force to recompute)")
        return

    jobs, info, inv = build_jobs(args.limit)
    print(f"[01] scanning {len(jobs)} files with {args.workers} workers", flush=True)

    # Resumable: results are cached in batches; an interrupted run continues where it stopped.
    # The cache is discarded automatically if the feature code or the file list changes.
    sig = hashlib.md5(Path(AF.__file__).read_bytes() + json.dumps([j[0] for j in jobs]).encode()).hexdigest()
    cache = C.TABLES / "_cache" / sig[:12]      # keyed by code + file list, so stale caches are never reused
    cache.mkdir(parents=True, exist_ok=True)

    recs, mels = [], []
    t0 = time.time()
    BATCH = 200
    with Pool(args.workers) as pool:
        for b in range(0, len(jobs), BATCH):
            part = cache / f"part_{b:06d}.pkl"
            if part.exists():
                with open(part, "rb") as fh:
                    out = pickle.load(fh)
            else:
                out = pool.map(AF.analyze_file, jobs[b:b + BATCH], chunksize=4)
                with open(part, "wb") as fh:
                    pickle.dump(out, fh)
            for rec, mel in out:
                recs.append(rec)
                mels.append(mel)
            print(f"[01] {len(recs)}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    shutil.rmtree(cache, ignore_errors=True)

    feats = pd.DataFrame(recs).merge(info, on="file_id", how="left")
    n_mels = C.N_MELS
    blank = np.full(n_mels, np.nan, dtype=np.float32)
    mel_mean = np.stack([m[0] if m else blank for m in mels]).astype(np.float32)
    mel_std = np.stack([m[1] if m else blank for m in mels]).astype(np.float32)
    mel_act = np.stack([m[2] if m else blank for m in mels]).astype(np.float32)

    feats.to_csv(out_csv, index=False)
    np.savez_compressed(C.TABLES / "mel_stats.npz", file_ids=feats["file_id"].values.astype(str),
                        mel_mean=mel_mean, mel_std=mel_std, mel_act=mel_act,
                        mel_centers_hz=AF.mel_centers_hz())
    inv["n_unreadable"] = int((feats["error"].fillna("") != "").sum())
    with open(C.TABLES / "file_inventory.json", "w") as f:
        json.dump(inv, f, indent=2)
    print(f"[01] wrote {out_csv} ({len(feats)} rows, {inv['n_unreadable']} unreadable)")


if __name__ == "__main__":
    main()
