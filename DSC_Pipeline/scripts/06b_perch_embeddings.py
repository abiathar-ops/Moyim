"""Step 6b - Perch v2 embeddings for every train and test clip (run on the Mac).

Uses Google's perch-hoplite package, which downloads Perch v2 from Kaggle:
    pip install git+https://github.com/google-research/perch-hoplite.git
(installed in DSC_Pipeline/.venv313; that environment also has everything
step 8 needs).

Input    cache/audio/{train,test}/<file_id>.npy  (32 kHz mono, from step 2;
         exactly the sample rate Perch expects, so no resampling here)
Output   outputs/03_birdnet/perch_windows.npz  (emb, file_id, win, split),
         same format as BirdNET's birdnet_windows.npz, 1536-d per window.

Perch v2 sees 5 s windows. With a 2.5 s hop a 10 s clip gives 3 windows;
clips shorter than 5 s are zero-padded (centred), as perch-hoplite does.
Each window is peak-normalised inside perch-hoplite. Only embeddings are
kept, never Perch's species predictions.

The run takes a while on CPU. Progress is saved every 200 clips, so if it is
interrupted, running it again continues where it stopped.

Usage    cd DSC_Pipeline/scripts
         source ../.venv313/bin/activate
         python 06b_perch_embeddings.py            # then: python 08_open_set_v2.py --emb both
         python 06b_perch_embeddings.py --force    # start over
"""
import argparse, time
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[1]
DATA = PIPE.parent / "DSC_Dataset"
AUDIO = PIPE / "cache" / "audio"
OUT = PIPE / "outputs" / "03_birdnet"
CACHE = OUT / "perch_windows.npz"
PARTIAL = OUT / "perch_windows.partial.npz"
SR, WIN_S = 32000, 5.0
SAVE_EVERY = 200


def load_model(hop_s):
    from perch_hoplite.zoo import model_configs
    preset = model_configs.get_preset_model_config("perch_v2")   # picks the CPU or GPU build
    preset.model_config.hop_size_s = hop_s
    return preset.load_model()


def frames_of(model, audio, hop_s):
    win = int(WIN_S * SR)
    if len(audio) < win:
        pad = win - len(audio)
        audio = np.pad(audio, (pad // 2, pad - pad // 2))
    return model.frame_audio(audio, WIN_S, hop_s)          # [n_windows, 160000]


def embed_frames(model, frames):
    out = model.batch_embed(np.asarray(frames, np.float32))  # each row is exactly one window
    emb = np.asarray(out.embeddings).reshape(len(frames), -1)
    return emb.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop", type=float, default=2.5, help="seconds between window starts")
    ap.add_argument("--batch", type=int, default=32, help="windows per model call")
    ap.add_argument("--force", action="store_true", help="ignore existing results and start over")
    a = ap.parse_args()
    if CACHE.exists() and not a.force:
        print("cache exists:", CACHE, "(use --force to re-extract)"); return
    if a.force and PARTIAL.exists():
        PARTIAL.unlink()

    meta = pd.read_csv(DATA / "train" / "metadata.csv")
    test_ids = pd.read_csv(DATA / "sample_submission.csv").file_id.astype(str)
    items = [("train", f) for f in meta.file_id] + [("test", f) for f in test_ids]
    missing = [f for s, f in items if not (AUDIO / s / f"{f}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} clips missing from {AUDIO} (run step 2 first), e.g. {missing[:2]}")

    done = {"emb": [], "file_id": [], "win": [], "split": []}
    if PARTIAL.exists():
        z = np.load(PARTIAL)
        done = {k: list(z[k]) for k in done}
        print(f"resuming: {len(set(done['file_id']))} clips already embedded")
    seen = set(done["file_id"])
    todo = [(s, f) for s, f in items if f not in seen]

    print("loading Perch v2 (first run downloads it from Kaggle)...", flush=True)
    model = load_model(a.hop)

    def save(path):
        np.savez_compressed(path, emb=np.vstack(done["emb"]) if done["emb"] else np.zeros((0, 1536), np.float32),
                            file_id=np.array(done["file_id"], str), win=np.array(done["win"]),
                            split=np.array(done["split"], str))

    buf, meta_buf, t0 = [], [], time.time()
    def flush():
        if buf:
            E = embed_frames(model, buf)
            done["emb"].extend(E); 
            for (s, f, w) in meta_buf:
                done["split"].append(s); done["file_id"].append(f); done["win"].append(w)
            buf.clear(); meta_buf.clear()

    for i, (split, fid) in enumerate(todo, 1):
        audio = np.load(AUDIO / split / f"{fid}.npy").astype(np.float32)
        for w, fr in enumerate(frames_of(model, audio, a.hop)):
            buf.append(fr); meta_buf.append((split, fid, w))
            if len(buf) >= a.batch:
                flush()
        if i % SAVE_EVERY == 0 or i == len(todo):
            flush(); save(PARTIAL)
            rate = i / (time.time() - t0)
            print(f"{len(seen) + i}/{len(items)} clips, {len(done['file_id'])} windows, "
                  f"~{(len(todo) - i) / rate / 60:.0f} min left", flush=True)

    got = set(done["file_id"])
    lost = set(f for _, f in items) - got
    if lost:
        raise RuntimeError(f"{len(lost)} clips produced no windows")
    save(CACHE)
    PARTIAL.unlink(missing_ok=True)
    E = np.vstack(done["emb"])
    print(f"saved {CACHE.name}: {E.shape[0]} windows x {E.shape[1]} dims for {len(got)} clips")


if __name__ == "__main__":
    main()
