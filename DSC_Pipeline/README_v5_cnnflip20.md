# v5_cnnflip20: public LB 0.69491 (team best)

`submission_v5_cnnflip20.csv` = our v3 submission (LB 0.682), with **20 clips moved from a known bird to Unknown**. The 20 clips are the ones a CNN with an "Unknown" class is most sure are Unknown. Mix: Unknown 486 (60%), Bondol Jawa 157, Kutilang 140, Cendrawasih 31.

## The pipeline, step by step

| Step | Script | What it does | Result |
|---|---|---|---|
| 6 | `06_birdnet_embeddings.py`, `06b_perch_embeddings.py` | Pretrained bird models turn audio into feature vectors: BirdNET v2.4 (1024-d per 3 s window) and Perch v2 (1536-d per 5 s window). Each clip is pooled with mean + max over its windows. | features |
| 6c | `06c_perch_species.py` | Perch's species scores (14,795 species) for every clip; used for the vetoes in step 9. | species scores |
| 8 | `08_open_set_v2.py --emb both` | **Known-bird classifier:** logistic regression on BirdNET + Perch features, self-trained with confident test clips. **Unknown detector:** a clip is Unknown if it is far from every training clip but close to other test clips (kNN distance ratio in 64-d PCA space), combined with BirdNET species novelty and smoothed over clusters of similar test clips. The top 45% most novel → Unknown. | v2 r45 |
| 9 | `09_cendrawasih_trim.py` | Cendrawasih had become a catch-all for loud Unknown birds. Keep only the 45 most Cendrawasih-like clips, and veto any whose Perch top-3 species include a parrot/cockatoo or White-rumped Shama. Result: 33 Cendrawasih, 57% Unknown. | **v3: LB 0.682** |
| 11b | `11b_pseudo_unknown.py` | Pick 200 test clips to teach the CNN what "Unknown" sounds like: the clips v3 already calls Unknown that sit furthest from the training data (BirdNET distance ratio). Validator anchor clips are excluded. | `pseudo_unknown.csv` |
| 11 | `11_train_cnn.py --run b0_pu --pseudo-unknown outputs/pseudo_unknown.csv` | **CNN:** EfficientNet-B0 (timm, ImageNet-pretrained, 1 input channel) fine-tuned on 5 s spectrograms (128-band mel, 32 kHz, PCEN + per-clip normalisation). 4 classes: the 3 birds + Unknown (the 200 clips above). Augmentation: crops weighted toward bird activity, background noise mixed in from other recordings, random EQ and gain, SpecAugment. 30 epochs, AdamW + one-cycle LR, label smoothing 0.05, mixed precision. 4 folds that hold out whole recordings; the test predictions are the average of the 4 fold models. About 1 h on an RTX 3060. | CNN P(Unknown) |
| 12 | `12_cnn_unknown_flip.py --k 20` | Take v3; among the clips v3 calls a known bird, move the 20 with the highest CNN P(Unknown) (all ≥ 0.963) to Unknown: 11 Kutilang, 7 Bondol Jawa, 2 Cendrawasih. | **v5: LB 0.69491** |

## Why the flip is safe
- **Low cost to real birds.** On training recordings the CNN never saw, only **0.2%** of real birds get P(Unknown) ≥ 0.963. Real birds almost never look this Unknown to it.
- **Better than BirdNET at spotting Unknowns.** On the validator's anchor clips (separation, 1.0 = perfect): CNN P(Unknown) 0.926 vs BirdNET distance 0.906. Combining the two gives 0.94.
- **Confirmed on the leaderboard:** +0.013 over v3.

## What didn't work (so nobody repeats it)
- **CNN without an Unknown class:** known-bird macro-F1 0.847 on held-out recordings vs 0.897 for BirdNET + logistic regression, and weaker at spotting Unknowns.
- **Rebuilding every decision from the CNN instead of tweaking v3:** worse on the validator, because it throws away the Perch-based steps.
- **No recording artifacts to exploit:** none of 11 technical audio features separates test from train or Unknown from known.

## Reproduce (from `DSC_Pipeline/`)
```
python scripts/06_birdnet_embeddings.py; python scripts/06b_perch_embeddings.py; python scripts/06c_perch_species.py
python scripts/08_open_set_v2.py --emb both
python scripts/09_cendrawasih_trim.py
python scripts/11b_pseudo_unknown.py
python scripts/11_train_cnn.py --fold all --run b0_pu --pseudo-unknown outputs/pseudo_unknown.csv   # GPU
python scripts/12_cnn_unknown_flip.py --k 20
```
Setup for the CNN: install CUDA PyTorch first, then `timm librosa numpy pandas scikit-learn`.

**Folds:** the CNN (step 11) was trained with `outputs/folds_cnn.csv` (the fold split from the `main` branch), not `outputs/folds.csv` (the split steps 3–10 used on the Mac). Both have the same recording groups, but the groups are assigned to folds differently. To reproduce the CNN exactly, copy `folds_cnn.csv` over `folds.csv` on the GPU machine before running step 11.

## Notes for the write-up
- **Pretrained models used:** BirdNET, Perch v2 and ImageNet EfficientNet-B0. All are public and ungated, and no external audio was used.
- **Test clips as training signal:** unlabelled test clips are used as pseudo-labels (self-training in step 8, pseudo-Unknown in step 11). The Perch species scores drive the step-9 vetoes.
