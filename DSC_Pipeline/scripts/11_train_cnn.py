"""Step 11 - Fine-tune a CNN on PCEN spectrograms (run on a GPU machine).

Why: frozen BirdNET/Perch embeddings blur two things we need: (a) the same
species sounding different across recordings (held-out Bondol/Cendrawasih
recordings get wrongly rejected), and (b) close relatives that sit inside our
known clusters (~40% of test "Kutilang" are likely other bulbuls). A network
trained on our own spectrograms can learn both.

Design (all choices from steps 2-5 of the pipeline):
  input      5 s crops of the 32 kHz mono cache (cache/audio), crops chosen
             where the activity curve (cache/activity) shows calls
  front-end  128-band mel (40 Hz-15 kHz, n_fft 1024, hop 320) -> PCEN -> per-clip
             standardisation ("pcen_norm", best in step 5), computed on the GPU
  augment    noise from OTHER recording groups (noise bank, step 4), mild EQ,
             gain, SpecAugment. No pitch shift / time stretch.
  sampling   per clip weight 1/sqrt(class size) * 1/sqrt(group size), so the rare
             Cendrawasih and small recording groups are seen more often
  model      EfficientNet-B0 (timm, ImageNet-pretrained, 1 input channel)
  validate   group folds from step 3: whole recording groups held out
  inference  5 s windows every 2.5 s, logits averaged per clip; embeddings are
             the mean and max of the network's last feature layer over windows
  optional   --pseudo-unknown CSV of test file_ids treated as a 4th "Unknown"
             class (test audio only, no test labels)

Outputs  outputs/04_cnn/<run>/fold<k>.npz   (fold k held out)
         outputs/04_cnn/<run>/full.npz      (--fold full: trained on all clips)
         each holds, for ALL 2,640 clips: file_id, split, probs, emb (mean+max),
         heldout (True for the clips this model never trained on), classes
         plus <name>.pt (weights) and <name>_log.csv (per-epoch validation)

Usage    python 11_train_cnn.py --fold 0           (then 1, 2, 3)
         python 11_train_cnn.py --fold all          (4 folds in a row, prints OOF summary)
         python 11_train_cnn.py --fold full
         quick check: python 11_train_cnn.py --fold 0 --epochs 1 --limit 200
Needs    torch (CUDA build on the 3060), timm, librosa, numpy, pandas, scikit-learn
"""
import argparse, math, os, platform, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PIPE = Path(__file__).resolve().parents[1]
DATA = PIPE.parent / "DSC_Dataset"
CACHE = PIPE / "cache"
OUTS = PIPE / "outputs"
CLASSES = ["Bondol Jawa", "Cendrawasih", "Kutilang"]
SR, WIN = 32000, 160000                      # 5 s
N_FFT, HOP, N_MELS, FMIN, FMAX = 1024, 320, 128, 40, 15000
PCEN = dict(gain=0.8, bias=10.0, power=0.25, time_constant=0.06, eps=1e-6)


# ----------------------------------------------------------------- front-end
class PCENFrontend(nn.Module):
    """librosa-equivalent mel (power=1, centred, Hann) -> PCEN -> per-clip z-score."""

    def __init__(self):
        super().__init__()
        import librosa
        mel = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
        self.register_buffer("mel", torch.tensor(mel, dtype=torch.float32))
        self.register_buffer("window", torch.hann_window(N_FFT))
        t_frames = PCEN["time_constant"] * SR / HOP
        self.b = (math.sqrt(1 + 4 * t_frames ** 2) - 1) / (2 * t_frames ** 2)

    @torch.no_grad()
    def forward(self, wav):                                   # (B, samples)
        # fp32 only: under autocast the mel matmul runs in fp16 and * 2**31 overflows to inf -> NaN logits
        with torch.autocast(wav.device.type, enabled=False):
            return self._forward(wav.float())

    def _forward(self, wav):
        X = torch.stft(wav.float(), N_FFT, HOP, window=self.window, center=True,
                       pad_mode="constant", return_complex=True).abs()
        S = (self.mel @ X) * (2 ** 31)                         # (B, mels, T), as librosa pcen input
        M = torch.empty_like(S); m = torch.ones_like(S[..., 0])
        for t in range(S.shape[-1]):                          # IIR smoother; state starts at 1 like librosa
            m = (1 - self.b) * m + self.b * S[..., t]
            M[..., t] = m
        g, bias, p, eps = PCEN["gain"], PCEN["bias"], PCEN["power"], PCEN["eps"]
        P = (S / (eps + M) ** g + bias) ** p - bias ** p
        mu = P.mean(dim=(1, 2), keepdim=True); sd = P.std(dim=(1, 2), keepdim=True)
        return ((P - mu) / (sd + 1e-6)).unsqueeze(1)          # (B, 1, mels, T)


def spec_augment(S, rng, n_time=2, max_t=40, n_freq=2, max_f=12):
    for i in range(S.shape[0]):
        if rng.random() < 0.5:
            continue
        for _ in range(n_freq):
            w = int(rng.integers(0, max_f + 1)); f0 = int(rng.integers(0, max(1, N_MELS - w)))
            S[i, :, f0:f0 + w, :] = 0.0
        for _ in range(n_time):
            w = int(rng.integers(0, max_t + 1)); t0 = int(rng.integers(0, max(1, S.shape[-1] - w)))
            S[i, :, :, t0:t0 + w] = 0.0
    return S


# ----------------------------------------------------------------- waveform side
def loop_pad(y, n):
    return y if len(y) >= n else np.tile(y, int(np.ceil(n / max(len(y), 1))))[:n]


def choose_crop(y, curve, rng, p_random=0.2, temperature=0.5):
    y = loop_pad(y, WIN); max_start = len(y) - WIN
    if max_start <= 0:
        return y[:WIN]
    if curve is None or rng.random() < p_random:
        s = int(rng.integers(0, max_start + 1)); return y[s:s + WIN]
    win_f = WIN // HOP; stride_f = 25
    if len(curve) <= win_f:
        s = int(rng.integers(0, max_start + 1)); return y[s:s + WIN]
    cs = np.concatenate([[0.0], np.cumsum(curve, dtype=np.float64)])
    starts_f = np.arange(0, len(curve) - win_f + 1, stride_f)
    sc = (cs[starts_f + win_f] - cs[starts_f]) / win_f
    z = (sc - sc.max()) / (sc.std() + 1e-6) / temperature; p = np.exp(z); p /= p.sum()
    s = int(min(rng.choice(starts_f, p=p) * HOP, max_start))
    return y[s:s + WIN]


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


def augment(y, group, rng, noise, noise_groups):
    if len(noise) and rng.random() < 0.8:                     # cross-group background mixing
        idx = np.flatnonzero(noise_groups != group)
        for _ in range(int(rng.integers(1, 3))):
            nz = loop_pad(noise[int(rng.choice(idx))], WIN)
            s = int(rng.integers(0, len(nz) - WIN + 1)); nz = nz[s:s + WIN]
            y = y + rms(y) / (rms(nz) * 10 ** (rng.uniform(3, 20) / 20)) * nz
    if rng.random() < 0.5:                                    # mild random EQ (+-4 dB)
        Y = np.fft.rfft(y); f = np.fft.rfftfreq(len(y), 1 / SR)
        knots = np.geomspace(50, SR / 2, 6)
        g = 10 ** (np.interp(np.log(np.maximum(f, 50)), np.log(knots), rng.uniform(-4, 4, 6)) / 20)
        y = np.fft.irfft(Y * g, n=len(y))
    if rng.random() < 0.5:
        y = y * 10 ** (rng.uniform(-6, 6) / 20)
    return np.clip(y, -1, 1).astype(np.float32)


def windows(y, hop=WIN // 2):
    y = loop_pad(y, WIN)
    starts = list(range(0, len(y) - WIN + 1, hop))
    return np.stack([y[s:s + WIN] for s in starts])


# ----------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, n_classes, pretrained):
        super().__init__()
        import timm
        try:
            self.body = timm.create_model("efficientnet_b0", pretrained=pretrained, in_chans=1, num_classes=0)
        except Exception as e:
            raise SystemExit(f"could not load pretrained weights ({e}); check internet, or pass --no-pretrained")
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.body.num_features, n_classes))

    def forward(self, x):
        e = self.body(x)
        return self.head(e), e


# ----------------------------------------------------------------- data
def load_all(limit=None, seed=0):
    folds = pd.read_csv(OUTS / "folds.csv")
    test_ids = pd.read_csv(DATA / "sample_submission.csv").file_id.astype(str)
    if limit:                                                 # quick-check mode: stratified subset
        folds = pd.concat([d.sample(min(len(d), max(8, limit * len(d) // len(folds))), random_state=seed)
                           for _, d in folds.groupby("label")])
        test_ids = test_ids.sample(min(len(test_ids), limit // 2), random_state=seed)
    items = [("train", f) for f in folds.file_id] + [("test", f) for f in test_ids]
    audio = {f: np.load(CACHE / "audio" / s / f"{f}.npy") for s, f in items}
    act = {f: np.load(CACHE / "activity" / f"{f}.npy").astype(np.float32) for _, f in items}
    nb = pd.read_csv(OUTS / "noise_bank.csv")
    noise = [np.load(CACHE / "noise" / n).astype(np.float32) for n in nb.noise_file]
    return folds.reset_index(drop=True), list(test_ids), audio, act, noise, nb.source_group.to_numpy()


def macro_f1(t, p, labels):
    f = []
    for c in labels:
        tp = np.sum((p == c) & (t == c)); fp = np.sum((p == c) & (t != c)); fn = np.sum((p != c) & (t == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


@torch.no_grad()
def infer(model, fe, ids, audio, dev, batch=64):
    model.eval(); probs, embs = [], []
    for f in ids:
        W = torch.from_numpy(windows(audio[f].astype(np.float32))).to(dev)
        lo, em = [], []
        for i in range(0, len(W), batch):
            x = fe(W[i:i + batch])
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                l, e = model(x)
            lo.append(l.float()); em.append(e.float())
        lo, em = torch.cat(lo), torch.cat(em)
        probs.append(F.softmax(lo.mean(0), -1).cpu().numpy())
        embs.append(torch.cat([em.mean(0), em.max(0).values]).cpu().numpy())
    return np.stack(probs), np.stack(embs)


# ----------------------------------------------------------------- training
def train_one(fold, a, data, dev):
    folds, test_ids, audio, act, noise, noise_groups = data
    rng = np.random.default_rng(a.seed + (0 if fold == "full" else int(fold)))
    torch.manual_seed(a.seed)
    known = [c for c in CLASSES if c != a.exclude_class]
    pseudo = load_pseudo(a, audio)
    classes = known + (["Unknown"] if (pseudo.label == "Unknown").any() else [])
    tr = folds if fold == "full" else folds[folds.fold != int(fold)]
    va = folds.iloc[0:0] if fold == "full" else folds[folds.fold == int(fold)]
    tr = tr[tr.label.isin(known)]                             # --exclude-class: that bird is never seen
    va = va[va.label.isin(known)]

    ids = list(tr.file_id); labels = [classes.index(l) for l in tr.label]; groups = list(tr.source_group)
    pseudo = pseudo[pseudo.label.isin(classes)]
    if len(pseudo):
        ids += list(pseudo.file_id); labels += [classes.index(l) for l in pseudo.label]
        groups += ["test_" + l for l in pseudo.label]         # one sampling group per pseudo label
        print("pseudo-labelled test clips:", pseudo.label.value_counts().to_dict(), flush=True)
    lab = np.array(labels); grp = np.array(groups)
    n_cls = pd.Series(lab).value_counts(); n_grp = pd.Series(grp).value_counts()
    w = 1 / np.sqrt(n_cls[lab].to_numpy()) / np.sqrt(n_grp[grp].to_numpy()); w /= w.sum()

    name = f"fold{fold}" if fold != "full" else "full"
    outdir = OUTS / "04_cnn" / a.run; outdir.mkdir(parents=True, exist_ok=True)
    model = Net(len(classes), a.pretrained and not a.reinfer).to(dev); fe = PCENFrontend().to(dev)
    if a.reinfer:
        model.load_state_dict(torch.load(outdir / f"{name}.pt", map_location=dev))
        print(f"loaded {outdir / name}.pt (re-running inference only)", flush=True)
        return finish(model, fe, fold, folds, test_ids, va, audio, dev, outdir, name, classes)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    steps = a.epochs * a.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    log = []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for _ in range(a.steps_per_epoch):
            pick = rng.choice(len(ids), size=a.batch, p=w)
            wav = np.stack([augment(choose_crop(audio[ids[i]].astype(np.float32), act[ids[i]], rng),
                                    grp[i], rng, noise, noise_groups) for i in pick])
            x = spec_augment(fe(torch.from_numpy(wav).to(dev)), rng)
            y = torch.from_numpy(lab[pick]).to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                logits, _ = model(x); loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
        row = {"epoch": ep, "loss": tot / a.steps_per_epoch, "sec": round(time.time() - t0)}
        if len(va) and (ep % a.eval_every == 0 or ep == a.epochs):
            P, _ = infer(model, fe, list(va.file_id), audio, dev)
            pred = np.array(known)[P[:, :len(known)].argmax(1)]; t = va.label.to_numpy()
            row["val_macroF1"] = macro_f1(t, pred, known)
            for c in known:
                row[f"recall_{c.split()[0]}"] = float(np.mean(pred[t == c] == c))
        log.append(row); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    torch.save(model.state_dict(), outdir / f"{name}.pt"); pd.DataFrame(log).to_csv(outdir / f"{name}_log.csv", index=False)
    return finish(model, fe, fold, folds, test_ids, va, audio, dev, outdir, name, classes)


def load_pseudo(a, audio):
    """Test clips used as extra training data: --pseudo CSV (file_id,label) and/or --pseudo-unknown CSV (file_id)."""
    parts = []
    if a.pseudo:
        parts.append(pd.read_csv(a.pseudo, dtype=str)[["file_id", "label"]])
    if a.pseudo_unknown:
        parts.append(pd.read_csv(a.pseudo_unknown, dtype=str)[["file_id"]].assign(label="Unknown"))
    if not parts:
        return pd.DataFrame(columns=["file_id", "label"])
    p = pd.concat(parts).drop_duplicates("file_id")
    bad = set(p.label) - set(CLASSES + ["Unknown"])
    if bad:
        raise SystemExit(f"unknown pseudo labels: {bad}")
    return p[p.file_id.isin(audio.keys())].reset_index(drop=True)


def finish(model, fe, fold, folds, test_ids, va, audio, dev, out, name, classes):
    all_ids = list(folds.file_id) + test_ids
    P, E = infer(model, fe, all_ids, audio, dev)
    va_set = set(va.file_id)
    held = np.array([f in va_set for f in all_ids], bool)
    np.savez_compressed(out / f"{name}.npz", file_id=np.array(all_ids, str),
                        split=np.array(["train"] * len(folds) + ["test"] * len(test_ids)),
                        probs=P.astype(np.float32), emb=E.astype(np.float16), heldout=held, classes=np.array(classes))
    print(f"saved {out / name}.npz", flush=True)
    return P[: len(folds)], held[: len(folds)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="all", help="0-3, all, or full")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=60)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run", default="b0_pcen")
    ap.add_argument("--pseudo-unknown", default=None, help="CSV with a file_id column of test clips")
    ap.add_argument("--pseudo", default=None, help="CSV file_id,label of pseudo-labelled test clips (any of the 4 labels)")
    ap.add_argument("--exclude-class", default=None, choices=CLASSES, help="open-set check: never train on this bird")
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--limit", type=int, default=None, help="quick check on a small subset")
    ap.add_argument("--device", default=None)
    ap.add_argument("--reinfer", action="store_true", help="load saved fold<k>.pt and only redo inference")
    a = ap.parse_args()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else
                                    "mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"device: {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else platform.processor()})")
    if dev.type == "cpu" and not a.limit:
        print("WARNING: no GPU found - this will be very slow")
    t0 = time.time(); data = load_all(a.limit, a.seed)
    print(f"loaded {len(data[0])} train + {len(data[1])} test clips, {len(data[4])} noise segments in {time.time() - t0:.0f} s")

    folds_to_run = ["0", "1", "2", "3"] if a.fold == "all" else [a.fold]
    known = [c for c in CLASSES if c != a.exclude_class]
    oof = np.zeros((len(data[0]), 4)); seen = np.zeros(len(data[0]), bool)
    for k in folds_to_run:
        print(f"\n===== fold {k} =====", flush=True)
        P, held = train_one(k, a, data, dev)
        held = held & data[0].label.isin(known).to_numpy()
        oof[held, :P.shape[1]] = P[held]; seen |= held
    if seen.any():
        t = data[0].label.to_numpy()[seen]; p = np.array(known)[oof[seen][:, :len(known)].argmax(1)]
        print(f"\nout-of-fold over {seen.sum()} held-out clips: macro-F1 {macro_f1(t, p, known):.3f} |",
              {c: round(float(np.mean(p[t == c] == c)), 3) for c in known})


if __name__ == "__main__":
    main()
