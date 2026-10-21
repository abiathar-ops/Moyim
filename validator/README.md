# NPC DSC 2026 – offline submission validator (team only, keep private)

Check a submission **before** spending a leaderboard submission. One command tells you:

1. whether the file meets the competition's format rules (Kaggle would otherwise reject or mis-score it), and
2. how it scores on **121 test clips whose label we know with high confidence** ("core anchors"), with an uncertainty interval, so you can rank candidate submissions against each other, and
3. how it labels **50 clips that are still an open question** (see *The White-rumped Munia question* below).

> This repo contains labels derived from the test set. Share it **only with our team**. Kaggle-style rules forbid sharing test-derived data with other teams.

## Setup

```bash
pip install -r requirements.txt     # just pandas and numpy
```

## Usage

```bash
python validate.py my_submission.csv
python validate.py a.csv b.csv c.csv                 # rank several + pairwise comparison
python validate.py my_submission.csv --details d.csv # see which anchor clips it gets wrong
```

Example output (files in `examples/`):

```
=== Ranking (estimated public LB | core anchor macro-F1 [95% interval])
  1. ~0.621 | 0.781 [0.672-0.866]  45% Unknown, munia->Unknown 47/50  novelty_r45.csv
  2. ~0.598 | 0.752 [0.651-0.835]  51% Unknown, munia->Unknown 16/50  lb0577_msp_q85.csv
  3. ~0.472 | 0.594 [0.503-0.681]  16% Unknown, munia->Unknown 0/50  lb0531_msp_q97.csv
```

The script exits with code 1 if any file has a format problem.

## How to read the output

| Line | Meaning |
|---|---|
| `FORMAT` | Exact columns `file_id,label`, all 814 test `file_id`s exactly once, labels spelled exactly `Bondol Jawa`, `Cendrawasih`, `Kutilang`, `Unknown`. Typos get a "did you mean" hint. |
| `Prediction mix` | Share of each label over all 814 clips. Leaderboard evidence so far suggests roughly **40–50% of test is `Unknown`**; you get a note outside 30–60%. |
| `Estimated public LB` | 0.769 × core anchor F1, calibrated on eight real leaderboard scores. Typically within ±0.01–0.03. |
| `Core anchor macro-F1` | Same metric as Kaggle (macro-F1 over the 4 labels), computed only on the 121 core anchor clips. The interval is a 95% bootstrap interval. |
| `White-rumped Munia clips` | How the file labels the 50 unresolved clips. Not part of the score. |
| `Anchors correct` | Per label: correct / number of anchors. |
| `Known-bird anchors wrongly called Unknown` | Cost of rejecting too aggressively. |
| Pairwise comparison | Resamples the anchor clips 2,000 times and counts how often the top file beats each other file. Under 80% means *too close to call*: don't spend a submission to choose between them on this basis alone. |

**Calibration.** Eight submissions with known public scores. The core anchor score puts all eight in the **same order as the public leaderboard**:

| Submission | Public LB | Core anchor macro-F1 | Estimate (0.769×) |
|---|---|---|---|
| v3_cend45_veto (Danta) | 0.682 | 0.885 | 0.681 |
| v2_r50 (Danta) | 0.632 | 0.826 | 0.635 |
| submission.csv (Arif) | 0.621 | 0.769 | 0.591 |
| submission_v3_pi40.csv (Arif) | 0.589 | 0.758 | 0.583 |
| SVM + confidence rejection, 51% Unknown (Danta) | 0.577 | 0.752 | 0.578 |
| submission_v2_pi25.csv (Arif) | 0.566 | 0.743 | 0.571 |
| submission_v2.csv (Arif) | 0.552 | 0.727 | 0.559 |
| SVM + confidence rejection, 16% Unknown (Danta) | 0.531 | 0.594 | 0.457 |

The validator prints an **estimated public LB = 0.769 × core anchor F1** (median ratio over these eight, refitted 2026-10-20). Leave-one-out, it is within ±0.008 for six of them; it underestimates files with very many Unknowns (−0.03 at 65%) or very few (−0.075 at 16%). The public LB itself (~244 clips) moves about ±0.03, so treat differences under ~0.02 as noise and trust the ranking more than the exact number. The White-rumped Munia split was decided after seeing early results, so keep checking it against new leaderboard scores.

**Blind spots found so far:** the validator can't judge Cendrawasih (5 anchors) and can't see Unknown birds that BirdNET can't name (its Unknown anchors are BirdNET-nameable species). Changes that mainly remove such hidden Unknowns, e.g. the Kutilang veto, look slightly worse here than they probably are.

## The White-rumped Munia question

BirdNET confidently hears **White-rumped Munia** in 50 anchor test clips (51 overall) and never in any training clip, so they were first anchored as `Unknown`. With them scored as `Unknown`, the validator matched the leaderboard order in only 10 of 15 pairs; scored as `Bondol Jawa`, also 10 of 15; left out, 15 of 15. Top public submissions disagree completely on them (45/50 vs 4/50 called Unknown) yet score close together.

The most likely reading: **few of these clips are in the public 30% of test**, so the public leaderboard can't tell us what they are, but they could move the **private** score a lot. Evidence that they are a separate species (so `Unknown`): they form the test cluster farthest from all training audio, and BirdNET never hears White-rumped Munia in any of the 1,124 Bondol Jawa training clips. Evidence for caution: White-rumped Munia is a close relative of Javan Munia (Bondol Jawa). Decide as a team before the final submission, and consider making the two final picks differ on this.

## How the anchors were built

BirdNET v2.4's own species predictions were run on every train and test clip. BirdNET is **not** reliable on our classes by name (it never recognises Bondol Jawa as Javan Munia), so it is only trusted where training data shows it is reliable:

| Anchor | Rule | Test clips |
|---|---|---|
| Kutilang | A BirdNET species that, on training clips, means Kutilang ≥ 95% of the time (≥ 8 clips, ≥ 2 recording groups), detected with confidence > 0.5. Mostly bulbuls. | 42 |
| Bondol Jawa | Same rule for Bondol Jawa (BirdNET consistently mislabels it as a fixed set of other species). | 29 |
| Cendrawasih | Lesser Bird-of-Paradise > 0.3 (never fires on other training classes). | 5 |
| Unknown | A species detected confidently (> 0.5) in ≥ 3 test clips but in **no** training clip, birds-of-paradise excluded. | 45 core + 50 White-rumped Munia (reported separately) |

Clips matching more than one rule are dropped. Rebuilding the anchors needs the audio and BirdNET: `DSC_Pipeline/scripts/07_anchor_check.py` writes `outputs/03_birdnet/anchors.csv`, which is copied to `data/anchors.csv`.

## Limitations

- **Biased subset.** Anchors are clips where BirdNET is sure; hard clips are not in it.
- **Cendrawasih has only 5 anchors**, so that class is barely checked. Bondol Jawa has 29.
- **Core Unknown anchors cover about 10 species**, with a few clips each. Hard Unknowns that BirdNET can't name are not checked.
- **Don't tune to the anchors.** If you repeatedly tweak a model until the anchor score goes up, it stops being an independent check. Use it to choose between a few genuinely different candidates.
- BirdNET predictions are used only to **check** submissions, never to make them. If this check influences the final choice, mention it in the write-up.

## Files

```
validate.py          the validator (pandas + numpy only)
data/anchors.csv     171 anchor clips: file_id, anchor label, group (core / white_rumped_munia)
data/test_ids.csv    all 814 test file_ids (from sample_submission.csv)
examples/            three earlier submissions, two with known public scores
```
