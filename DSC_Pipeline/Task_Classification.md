# Task Brief: Open-Set Bird Sound Classification (Schematics NPC DSC 2026)

> Purpose of this file: give an LLM (or an engineer) a complete, self-contained description of the problem so it can analyze the audio data, design a pipeline, and classify recordings into the required labels.

## 1. Background

Pak Pisi is a senior forest ranger protecting Indonesian tropical rainforest. The forest is habitat to endemic birds whose presence indicates ecosystem health. Acoustic sensors at many locations produce thousands of recordings daily. Each recording mixes bird calls with rain, insects, wind, and other environmental noise. Listening manually is too costly.

**Goal:** build an audio classification system that identifies bird species from short audio recordings.

## 2. Task

Given a short audio clip, predict exactly one label:

| Label | Meaning |
|---|---|
| `Bondol Jawa` | Known class (in training data) |
| `Cendrawasih` | Known class (in training data) |
| `Kutilang` | Known class (in training data) |
| `Unknown` | Any bird sound NOT from the three known species. **No labeled examples exist in the training data.** |

Labels must be written exactly as above (case and spacing matter).

This is an **open-set recognition / novelty detection** problem: the model must classify three known species and reject samples that do not resemble them.

## 3. Objectives

1. Recognize the three species available in training data.
2. Be robust to rain, wind, insects, and other environmental interference.
3. Detect recordings of unseen species and output `Unknown`.
4. Produce consistent predictions across varied recording conditions.

## 4. Data

- **Size:** 2,642 files total, 5.22 GB, format `flac` + `csv`. License CC BY-NC 4.0.
- **Layout:**

```
train/
├── audio/            # <file_id>.flac
└── metadata.csv
test/
└── audio/            # unlabeled recordings (814 file_ids)
sample_submission.csv # all test file_ids, label = NA
```

- **`train/metadata.csv` columns:**
  - `file_id`: SHA-256 hash, also the audio filename.
  - `audio_path`: path relative to dataset root, e.g. `train/audio/<file_id>.flac`.
  - `label`: one of `Bondol Jawa`, `Cendrawasih`, `Kutilang`.
  - `source_group`: hash of the original source. Recordings sharing it come from the same source recording.
- **Test data:** audio with **no labels and no `source_group`**. May contain the three known species and unseen species (to be labeled `Unknown`).
- **`sample_submission.csv`:** 814 unique `file_id`s with `label` = `NA` as a format example.

### Critical data-handling note: `source_group`
Segments from the same source are highly correlated. Random train/validation splitting will leak information and give inflated scores. **Use `GroupKFold` / `StratifiedGroupKFold` on `source_group`** for validation so that no group appears in both train and validation.

## 5. Evaluation

**Metric: Macro F1 over 4 classes** (`Bondol Jawa`, `Cendrawasih`, `Kutilang`, `Unknown`).

For each class c, F1 is the harmonic mean of precision and recall:

F1_c = 2 · P_c · R_c / (P_c + R_c)

Final score = mean of F1_c over the four classes (each class weighted equally regardless of sample count).

Implications:
- `Unknown` is worth 25% of the score even if it is rare; the model must not ignore it.
- Class imbalance must be handled; do not optimize plain accuracy.
- Both false `Unknown` (missing known birds) and missed `Unknown` hurt.

**Leaderboard:** public LB uses ~30% of test data; private LB (~70%) decides final ranking. Avoid overfitting thresholds to the public LB.

## 6. Submission Format

CSV with exactly two columns, `file_id` and `label`:

```
file_id,label
02a1b3c4d5e6f7...,Bondol Jawa
14b2c8d9e0f1a2...,Unknown
25c3d9e1f2a4b6...,Kutilang
```

Rules:
- `file_id` values must match `sample_submission.csv` exactly.
- Each `file_id` appears exactly once; none missing, none extra.
- `label` must be one of the four allowed strings, exactly.
- Row order does not matter.
- Column names must be `file_id` and `label`.

## 7. Competition Rules (binding constraints)

**Data**
- Use **only the data provided by the organizers. No external data.** This means no extra bird recordings, no external noise datasets (rain/wind/insect), and no external audio to act as synthetic `Unknown`. Noise augmentation must use noise derived from the provided audio or synthetically generated (e.g., Gaussian/colored noise).
- No hand labeling or human prediction of validation/test records.

**Submissions**
- Max **5 submissions per day** per team on Kaggle.

**Notebook & reproducibility**
- The submitted notebook **must set seeds / random states** so results are reproducible.
- An unexplained significant gap between notebook score and Kaggle score may lead to disqualification.

**Tools & models**
- Any tool, library, or framework is allowed.
- **Full AutoML or any process that cannot be explained leads to disqualification.**
- Gated-repository models are allowed if an access token is provided that stays valid until results are announced, and the **model identity is stated clearly**.
- Pretrained models: not explicitly banned (gated models are mentioned), but they were trained on external data. Confirm with the organizers before relying on them, and always document the exact model.

**Gen AI / LLM**
- AI may be used as an aid, but every approach and result must be explainable and defensible by the team.
- **LLMs (or similar) must not be used to generate the submission directly.** Predictions must come from a trained model/pipeline, not from an LLM labeling files.

**Kaggle foundational rules (highlights)**
- One Kaggle account per person; no private code sharing outside the team (public sharing on the competition forum is OK).
- Open-source code used must have an OSI-approved license that permits commercial use.
- Foundational rules override competition-specific rules on conflict.

## 8. Allowed Methods (free choice within the rules above)

- Silence removal / voice-activity-style trimming
- Audio augmentation (time shift, pitch/time stretch, gain, SpecAugment, noise mixing using provided or synthetic noise only)
- Resampling
- Mel-spectrogram, MFCC, or other representations
- Classical ML or deep learning
- Open-set / novelty detection strategies

## 9. Suggested Analysis Plan for the LLM

> The LLM's role is analysis and code assistance. It must not label test files itself (see rules). Set seeds everywhere (`numpy`, `random`, `torch`, etc.).

### Step 1: Inspect the data
- Count files per class and per `source_group`; check duration, sample rate, channels, and format.
- Identify corrupt/empty/clipped files; measure silence ratio.
- Compare train vs. test distributions (duration, loudness, spectral stats) to anticipate domain shift and gauge how much `Unknown` may exist in test.

### Step 2: Preprocess
- Resample to a common rate; convert to mono; normalize loudness.
- Trim silence; segment long clips into fixed windows (e.g., 3-5 s) with overlap; aggregate window predictions per file (mean/max of logits or probabilities).
- Compute log-mel spectrograms (or embeddings from a pretrained bird/audio model).

### Step 3: Closed-set classifier
- Train on the three known classes with group-aware cross-validation.
- Use augmentation, especially background noise (rain, wind, insects) to improve robustness.
- Consider fine-tuning or using frozen embeddings + a light classifier (logistic regression / SVM / MLP).

### Step 4: Unknown detection (no labeled Unknown data)
Options, can be combined:
- **Confidence thresholding:** max softmax probability or energy score below a threshold gives `Unknown`.
- **Embedding distance:** Mahalanobis or kNN distance to class centroids/training embeddings; far samples give `Unknown`.
- **One-class models:** Isolation Forest, One-Class SVM, or GMM on embeddings.
- **Simulated unknowns (no external data):** hold out one known class at a time in group-aware CV and treat it as `Unknown` to tune the rejection threshold. Synthetic outliers built from provided audio (heavy distortion, noise-only segments cut from the training clips) can also help.
- **Test-set structure:** clustering test embeddings can reveal tight clusters far from the training classes (likely unseen species). Using unlabeled test audio this way is not hand labeling, but keep it fully automated and documented.
- Tune the threshold to maximize macro F1 in group-aware CV with simulated unknowns, not on the public LB alone.

### Step 5: Validate
- Report per-class precision/recall/F1, macro F1, and a confusion matrix over all four labels.
- Test robustness on noisy/augmented versions of validation clips.
- Check prediction consistency (same clip, different crops, similar output).

### Step 6: Produce the submission
- Predict every `file_id` in `sample_submission.csv`.
- Validate the file: columns, uniqueness, completeness, label spelling.

## 10. Pitfalls to Avoid

- Random train/val split ignoring `source_group` (leakage, inflated CV).
- Treating `Unknown` as an afterthought; it is 25% of the metric.
- Threshold overfit to public LB (only ~30% of test).
- Misclassifying noise-only clips as a species; consider whether rain/insect-dominated clips should be `Unknown` per the data definition (verify against any provided guidance).
- Label typos or whitespace differences in the submission.
- Using external data (including external noise or other-species audio) or an LLM to label the test set: both are disqualifying.
- Forgetting to set seeds; notebook and Kaggle scores must match.
- Silence removal that deletes short calls; check that trimming keeps the vocalization.

## 11. Deliverables Checklist

- [ ] Data inspection summary
- [ ] Preprocessing pipeline
- [ ] Closed-set model with group-aware CV score
- [ ] Unknown-detection method with tuned threshold
- [ ] Per-class and macro F1 report
- [ ] Seeded, reproducible notebook; model identities documented
- [ ] Valid `submission.csv` (`file_id,label`, 814 rows)

## 12. Citation

Schematics. NPC DSC 2026. https://www.kaggle.com/competitions/schematics-2026, 2026. Kaggle.
