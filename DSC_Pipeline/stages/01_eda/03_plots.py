"""Step 3: figures for the EDA report (saved to outputs/figures/)."""
import json
import sys
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import audio_features as AF
import config as C

COL = {"Bondol Jawa": "#1f77b4", "Cendrawasih": "#d62728", "Kutilang": "#2ca02c", "Test": "#7f7f7f"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(C.FIGS / name, dpi=130)
    plt.close(fig)
    print(f"[03] {name}")


def safe(fn):
    def w(*a, **k):
        try:
            fn(*a, **k)
        except Exception:
            print(f"[03] FAILED {fn.__name__}")
            traceback.print_exc()
    return w


def main():
    C.seed_everything()
    C.ensure_dirs()
    T = C.TABLES
    feats = pd.read_csv(T / "features.csv")
    feats["error"] = feats["error"].fillna("")
    ok = feats[feats["error"] == ""].copy()
    tr = ok[ok["split"] == "train"].reset_index(drop=True)
    te = ok[ok["split"] == "test"].reset_index(drop=True)
    npz = np.load(T / "mel_stats.npz", allow_pickle=True)
    idx = {f: i for i, f in enumerate(npz["file_ids"].astype(str))}
    MEL = npz["mel_act"]
    centers = npz["mel_centers_hz"]
    gid = C.short_group_ids(tr)
    summ = json.load(open(T / "summary.json")) if (T / "summary.json").exists() else {}
    ok["plot_group"] = np.where(ok["split"] == "test", "Test", ok["label"])

    @safe
    def class_counts():
        fig, ax = plt.subplots(figsize=(6, 4))
        vc = tr["label"].value_counts().reindex(C.KNOWN_CLASSES).fillna(0)
        ax.bar(vc.index, vc.values, color=[COL[c] for c in vc.index])
        for i, v in enumerate(vc.values):
            ax.text(i, v, f"{int(v)}", ha="center", va="bottom")
        ax.set_ylabel("train files")
        ax.set_title(f"Class balance (train n={len(tr)}, test n={len(te)})")
        save(fig, "01_class_counts.png")

    @safe
    def group_heatmap():
        ct = pd.crosstab(tr["source_group"].map(gid), tr["label"]).reindex(columns=C.KNOWN_CLASSES).fillna(0)
        fig, ax = plt.subplots(figsize=(5, max(3, 0.35 * len(ct) + 1)))
        im = ax.imshow(ct.values, aspect="auto", cmap="Blues")
        ax.set_xticks(range(ct.shape[1]), ct.columns, rotation=20)
        ax.set_yticks(range(ct.shape[0]), ct.index)
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                if ct.values[i, j] > 0:
                    ax.text(j, i, int(ct.values[i, j]), ha="center", va="center", fontsize=8)
        ax.set_title("Files per source_group x class")
        fig.colorbar(im, ax=ax)
        save(fig, "02_group_by_class.png")

    @safe
    def durations():
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        bins = np.linspace(0, max(ok["duration"].quantile(0.99), 1), 40)
        for k in C.KNOWN_CLASSES + ["Test"]:
            axs[0].hist(ok.loc[ok.plot_group == k, "duration"], bins=bins, histtype="step", density=True,
                        color=COL[k], label=k, lw=1.6)
        axs[0].set_xlabel("duration (s)")
        axs[0].legend()
        axs[0].set_title("Duration distribution")
        order = sorted(gid.items(), key=lambda kv: kv[1])
        data = [tr.loc[tr.source_group == g, "duration"].values for g, _ in order]
        axs[1].boxplot(data, labels=[n for _, n in order], showfliers=False)
        axs[1].tick_params(axis="x", rotation=90)
        axs[1].set_title("Duration by source group")
        save(fig, "03_duration.png")

    @safe
    def quality_boxes():
        feats_ = [("rms_dbfs", "RMS (dBFS)"), ("snr_est_db", "SNR estimate (dB)"), ("noise_floor_db", "noise floor (dB)"),
                  ("event_ratio", "frames > floor+6 dB"), ("flatness_active", "spectral flatness (1-8 kHz)"),
                  ("lf_ratio", "energy < 300 Hz"), ("hf_ratio", "energy > 10 kHz"), ("hf_periodicity", "periodicity of 8-14 kHz envelope")]
        fig, axs = plt.subplots(2, 4, figsize=(15, 7))
        groups = C.KNOWN_CLASSES + ["Test"]
        for ax, (c, t) in zip(axs.ravel(), feats_):
            data = [ok.loc[ok.plot_group == k, c].dropna().values for k in groups]
            bp = ax.boxplot(data, labels=["BJ", "CW", "KT", "Test"], showfliers=False, patch_artist=True)
            for patch, k in zip(bp["boxes"], groups):
                patch.set_facecolor(COL[k])
                patch.set_alpha(0.6)
            ax.set_title(t, fontsize=10)
        fig.suptitle("Recording quality by class (BJ=Bondol Jawa, CW=Cendrawasih, KT=Kutilang)")
        save(fig, "04_quality_by_class.png")

    @safe
    def mean_mel():
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for k in C.KNOWN_CLASSES + ["Test"]:
            ids = ok.loc[ok.plot_group == k, "file_id"]
            M = MEL[[idx[f] for f in ids]]
            m, s = np.nanmean(M, 0), np.nanstd(M, 0)
            ax.plot(centers, m, color=COL[k], label=k, ls="--" if k == "Test" else "-")
            if k != "Test":
                ax.fill_between(centers, m - s, m + s, color=COL[k], alpha=0.12)
        ax.set_xscale("log")
        ax.set_xlabel("frequency (Hz, mel-band centre)")
        ax.set_ylabel("mean log-mel level over active frames (dB)")
        ax.set_title("Average spectral profile by class")
        ax.legend()
        save(fig, "05_mean_spectrum.png")

    @safe
    def band_fractions():
        bands = AF.BAND_NAMES
        fig, ax = plt.subplots(figsize=(10, 4.5))
        w = 0.2
        for i, k in enumerate(C.KNOWN_CLASSES + ["Test"]):
            v = ok.loc[ok.plot_group == k, bands].mean().values
            ax.bar(np.arange(len(bands)) + i * w, v, w, color=COL[k], label=k)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(bands)) + 1.5 * w, [b[3:] for b in bands], rotation=30)
        ax.set_ylabel("mean energy fraction")
        ax.set_title("Energy by frequency band (rain/wind = low bands, insects = high bands)")
        ax.legend()
        save(fig, "06_band_fractions.png")

    def design_pca():
        cols = [c for c in summ.get("feature_columns", []) if c in ok.columns]
        med = tr[cols].median()
        def D(df):
            A = df[cols].astype(float).fillna(med).values
            M = np.nan_to_num(MEL[[idx[f] for f in df["file_id"]]])
            return np.hstack([A, M])
        sc = StandardScaler().fit(D(tr))
        pca = PCA(n_components=2, random_state=C.SEED).fit(sc.transform(D(tr)))
        return pca.transform(sc.transform(D(tr))), pca.transform(sc.transform(D(te))), pca

    @safe
    def pca_scatter():
        Ptr, Pte, pca = design_pca()
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
        axs[0].scatter(Pte[:, 0], Pte[:, 1], s=8, c=COL["Test"], alpha=0.35, label="Test")
        for k in C.KNOWN_CLASSES:
            m = tr["label"].values == k
            axs[0].scatter(Ptr[m, 0], Ptr[m, 1], s=8, c=COL[k], alpha=0.6, label=k)
        axs[0].legend()
        axs[0].set_title("PCA of features: class + test")
        cmap = plt.get_cmap("tab20")
        for i, (g, name) in enumerate(sorted(gid.items(), key=lambda kv: kv[1])):
            m = tr["source_group"].values == g
            axs[1].scatter(Ptr[m, 0], Ptr[m, 1], s=8, color=cmap(i % 20), label=name)
        axs[1].legend(ncol=2, fontsize=7)
        axs[1].set_title("PCA of features: source group (train)")
        for a in axs:
            a.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
            a.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
        save(fig, "07_pca.png")

    @safe
    def variance_explained():
        t = pd.read_csv(T / "feature_class_effects.csv").dropna(subset=["eta2_class"]).head(15)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        y = np.arange(len(t))
        ax.barh(y - 0.2, t["eta2_class"], 0.4, label="class effect (eta^2)", color="#4c72b0")
        ax.barh(y + 0.2, t["adj_r2_group_within_class"].clip(lower=0), 0.4,
                label="source-group effect within class (adj R^2)", color="#dd8452")
        ax.set_yticks(y, t["feature"])
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.set_title("Class signal vs recording-source signal (top class features)")
        save(fig, "08_class_vs_group_effect.png")

    @safe
    def cv_gap():
        r = summ["D"]["results"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        keys = [k for k in r if k.endswith("cv")]
        ax.bar([k.replace("_", "\n") for k in keys], [r[k]["macro_f1"] for k in keys],
               color=["#8da0cb" if "random" in k else "#66c2a5" for k in keys])
        for i, k in enumerate(keys):
            ax.text(i, r[k]["macro_f1"], f"{r[k]['macro_f1']:.2f}", ha="center", va="bottom")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("macro F1 (3 known classes)")
        ax.set_title("Random CV (leaky) vs group-aware CV")
        save(fig, "09_cv_leakage_gap.png")

    @safe
    def shift():
        imp = pd.read_csv(T / "adversarial_importance.csv").head(15)
        ksd = pd.read_csv(T / "shift_ks.csv").head(15)
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        axs[0].barh(imp["feature"], imp["importance"], color="#8172b3")
        axs[0].invert_yaxis()
        axs[0].set_title(f"Train-vs-test classifier (AUC={summ['E']['adversarial_auc']:.2f}): top features")
        axs[1].barh(ksd["feature"], ksd["ks"], color="#c44e52")
        axs[1].invert_yaxis()
        axs[1].set_title("KS statistic, train vs test (top 15)")
        save(fig, "10_train_test_shift.png")

    @safe
    def novelty():
        d_lgo = np.load(T / "nn_dist_train_lgo.npy")
        d_te = np.load(T / "nn_dist_test.npy")
        ref = d_lgo[~np.isnan(d_lgo)]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bins = np.linspace(0, np.percentile(np.r_[ref, d_te], 99.5), 50)
        ax.hist(ref, bins=bins, alpha=0.55, density=True, label="train (nearest file from OTHER sources)", color="#4c72b0")
        ax.hist(d_te, bins=bins, alpha=0.55, density=True, label="test (nearest train file)", color="#dd8452")
        ax.axvline(np.percentile(ref, 95), color="k", ls="--", lw=1, label="train p95")
        ax.set_xlabel("mean distance to 3 nearest train files (PCA space)")
        ax.legend(fontsize=8)
        ax.set_title("Novelty preview: how far are test files from known training audio?")
        save(fig, "11_novelty_preview.png")

    @safe
    def loco():
        t = pd.read_csv(T / "loco_novelty_auroc.csv")
        p = t.pivot(index="held_out_class", columns="method", values="auroc")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        p.plot.bar(ax=ax, rot=0)
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUROC (held-out class vs held-out known)")
        ax.set_title("Can we reject a species we never trained on?")
        save(fig, "12_loco_novelty.png")

    @safe
    def examples():
        rs = np.random.RandomState(C.SEED)
        picks = []
        for k in C.KNOWN_CLASSES:
            sub = tr[tr.label == k]
            for r in sub.sample(min(2, len(sub)), random_state=C.SEED).itertuples():
                picks.append((f"{k} [{gid[r.source_group]}]", r.file_id, "train"))
        for r in te.sample(min(4, len(te)), random_state=C.SEED).itertuples():
            picks.append(("test", r.file_id, "test"))
        fig, axs = plt.subplots(5, 2, figsize=(13, 14))
        for ax, (title, fid, split) in zip(axs.ravel(), picks):
            path = C.ROOT / split / "audio" / f"{fid}.flac"
            mel, dur = AF.load_logmel(str(path))
            vmax = np.percentile(mel, 99.5)
            ax.imshow(mel, origin="lower", aspect="auto", cmap="magma", vmin=vmax - 70, vmax=vmax,
                      extent=[0, dur, 0, mel.shape[0]])
            tick = [0, 16, 32, 48, 63]
            ax.set_yticks(tick, [f"{centers[i] / 1000:.1f}k" for i in tick])
            ax.set_title(f"{title}  ({fid[:8]}...)", fontsize=9)
        for ax in axs.ravel()[len(picks):]:
            ax.axis("off")
        fig.suptitle("Random log-mel spectrograms (seeded)")
        save(fig, "13_example_spectrograms.png")

    for fn in [class_counts, group_heatmap, durations, quality_boxes, mean_mel, band_fractions,
               pca_scatter, variance_explained, cv_gap, shift, novelty, loco, examples]:
        fn()


if __name__ == "__main__":
    main()
