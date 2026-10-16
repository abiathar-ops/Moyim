"""Step 1 - Inventory: what is in the data, and do train and test match?

Outputs
  outputs/inventory_train.csv, outputs/inventory_test.csv  per-file header info
  outputs/group_summary.csv                                clips per source_group x label
  outputs/channel_check.csv                                are the 2 stereo channels different?
"""
import _common  # noqa: F401
import numpy as np
import pandas as pd
import soundfile as sf

from src import config as C


def header(path):
    i = sf.info(str(path))
    return dict(sr=i.samplerate, channels=i.channels, frames=i.frames,
                duration=i.frames / i.samplerate, subtype=i.subtype)


def main():
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(C.TRAIN_META)
    sub = pd.read_csv(C.SAMPLE_SUB)

    tr = pd.concat([meta, pd.DataFrame([header(C.DATA_DIR / p) for p in meta.audio_path])], axis=1)
    te = pd.concat([sub[["file_id"]],
                    pd.DataFrame([header(C.TEST_AUDIO / f"{f}.flac") for f in sub.file_id])], axis=1)
    tr.to_csv(C.OUT_DIR / "inventory_train.csv", index=False)
    te.to_csv(C.OUT_DIR / "inventory_test.csv", index=False)

    print(f"train clips: {len(tr)}   test clips: {len(te)}")
    print("\nclass balance (clips / groups):")
    cls = tr.groupby("label").agg(clips=("file_id", "size"), groups=("source_group", "nunique"),
                                  median_dur=("duration", "median"))
    print(cls.to_string())

    g = (tr.groupby(["source_group", "label"]).size().rename("clips").reset_index()
           .sort_values("clips", ascending=False))
    g.to_csv(C.OUT_DIR / "group_summary.csv", index=False)
    multi = g.groupby("source_group")["label"].nunique()
    print(f"\nsource groups: {len(multi)}   groups with >1 species: {(multi > 1).sum()}")
    if (multi == 1).all():
        print("  -> every group is a single species: group identity == species in train.")
        print("     Validation MUST hold out whole groups, or recording conditions leak.")

    print("\naudio properties  train vs test:")
    for col in ["sr", "channels", "subtype"]:
        print(f"  {col:9s} train={ {k: int(v) for k, v in tr[col].value_counts().items()} }  test={ {k: int(v) for k, v in te[col].value_counts().items()} }")
    q = [0, .1, .5, .9, 1]
    print("  duration quantiles (s):")
    print("   train", np.round(tr.duration.quantile(q).values, 2))
    print("   test ", np.round(te.duration.quantile(q).values, 2))

    # stereo check on a sample: if L and R are nearly identical, mono loses nothing
    rng = np.random.default_rng(C.SEED)
    rows = []
    for p in rng.choice(tr.audio_path.values, size=min(80, len(tr)), replace=False):
        x, _ = sf.read(str(C.DATA_DIR / p), dtype="float32", always_2d=True)
        if x.shape[1] < 2:
            continue
        l, r = x[:, 0], x[:, 1]
        corr = float(np.corrcoef(l, r)[0, 1]) if l.std() > 0 and r.std() > 0 else 1.0
        lvl = 20 * np.log10((np.sqrt(np.mean(l ** 2)) + 1e-12) / (np.sqrt(np.mean(r ** 2)) + 1e-12))
        rows.append(dict(audio_path=p, lr_corr=corr, lr_level_diff_db=lvl,
                         identical=bool(np.allclose(l, r, atol=1e-6))))
    ch = pd.DataFrame(rows)
    ch.to_csv(C.OUT_DIR / "channel_check.csv", index=False)
    print(f"\nstereo check on {len(ch)} clips: identical L/R = {ch.identical.mean():.0%}, "
          f"median L/R correlation = {ch.lr_corr.median():.3f}, "
          f"median |level diff| = {ch.lr_level_diff_db.abs().median():.2f} dB")


if __name__ == "__main__":
    main()
