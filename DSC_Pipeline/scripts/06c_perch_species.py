"""Step 6c - Perch v2 species scores for every train and test clip (run on the Mac).

Perch v2 also classifies ~15,000 species. Here we keep, for each clip, the
highest score each species reaches in any of its 5 s windows. These are used
to check whether Perch recognises our three species by name (on training
clips) and to spot related species in test, the way step 7 did with BirdNET.
They are referee/features only, never the final label.

Needs the same environment as step 6b (DSC_Pipeline/.venv313 with perch-hoplite)
and the 32 kHz audio cache from step 2. Takes about as long as step 6b;
progress is saved every 200 clips and a rerun continues where it stopped.

Output   outputs/03_birdnet/perch_species_clip.npz
           logit     (n_clips, n_species) float16, max over windows
                     (Perch's calibrated logits; sigmoid gives a 0-1 score)
           file_id, split, species (n_species,) names
Usage    python 06c_perch_species.py [--force]
"""
import argparse, importlib.util, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s6b", HERE / "06b_perch_embeddings.py")
s6b = importlib.util.module_from_spec(spec); spec.loader.exec_module(s6b)

OUT = s6b.OUT
CACHE = OUT / "perch_species_clip.npz"
PARTIAL = OUT / "perch_species_clip.partial.npz"
SAVE_EVERY = 200


def species_logits(model, frames):
    out = model.batch_embed(np.asarray(frames, np.float32))
    keys = [k for k in out.logits if np.asarray(out.logits[k]).shape[-1] > 1000]
    if not keys:
        raise RuntimeError(f"no species logits found, keys: {list(out.logits)}")
    L = np.asarray(out.logits[keys[0]])
    return keys[0], L.reshape(len(frames), L.shape[-1]).astype(np.float32)


def class_names(model, key, n):
    cl = model.class_list
    for k in [key] + list(cl):
        if k in cl and len(cl[k].classes) == n:
            return np.array(cl[k].classes, str)
    raise RuntimeError(f"no class list with {n} entries; available: {[(k, len(v.classes)) for k, v in cl.items()]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop", type=float, default=2.5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if CACHE.exists() and not a.force:
        print("cache exists:", CACHE, "(use --force to recompute)"); return
    if a.force and PARTIAL.exists():
        PARTIAL.unlink()

    meta = pd.read_csv(s6b.DATA / "train" / "metadata.csv")
    test_ids = pd.read_csv(s6b.DATA / "sample_submission.csv").file_id.astype(str)
    items = [("train", f) for f in meta.file_id] + [("test", f) for f in test_ids]

    rows, fids, splits, names = [], [], [], None
    if PARTIAL.exists():
        z = np.load(PARTIAL)
        rows, fids, splits, names = list(z["logit"]), list(z["file_id"]), list(z["split"]), z["species"]
        print(f"resuming: {len(fids)} clips already done")
    todo = [(s, f) for s, f in items if f not in set(fids)]

    print("loading Perch v2...", flush=True)
    model = s6b.load_model(a.hop)

    def save(path):
        np.savez_compressed(path, logit=np.vstack(rows).astype(np.float16), file_id=np.array(fids, str),
                            split=np.array(splits, str), species=names)

    buf, owner, t0 = [], [], time.time()
    def flush():
        nonlocal names
        if not buf:
            return
        key, L = species_logits(model, buf)
        if names is None:
            names = class_names(model, key, L.shape[1])
        clip_max = pd.DataFrame(L).groupby(np.array(owner)).max()   # max over each clip's windows
        for o in sorted(set(owner)):                                   # owner = index into this batch's clips
            rows.append(clip_max.loc[o].to_numpy()); fids.append(pending[o][1]); splits.append(pending[o][0])
        buf.clear(); owner.clear(); pending.clear()

    pending = []
    for i, (split, fid) in enumerate(todo, 1):
        audio = np.load(s6b.AUDIO / split / f"{fid}.npy").astype(np.float32)
        fr = s6b.frames_of(model, audio, a.hop)
        if len(buf) + len(fr) > a.batch and buf:
            flush()
        pending.append((split, fid))
        buf.extend(fr); owner.extend([len(pending) - 1] * len(fr))
        if i % SAVE_EVERY == 0 or i == len(todo):
            flush(); save(PARTIAL)
            rate = i / (time.time() - t0)
            print(f"{len(items) - len(todo) + i}/{len(items)} clips, ~{(len(todo) - i) / rate / 60:.0f} min left", flush=True)

    missing = set(f for _, f in items) - set(fids)
    if missing:
        raise RuntimeError(f"{len(missing)} clips missing")
    save(CACHE); PARTIAL.unlink(missing_ok=True)
    print(f"saved {CACHE.name}: {len(fids)} clips x {len(names)} species")


if __name__ == "__main__":
    main()
