# Train/Test Split Indices

This directory contains the exact train/test partitions used in the paper,
released as identifier lists only — no audio, and no location or
deployment-time information.

| File | Data | Rule | Train/Test |
|---|---|---|---|
| `PAM_split_index.csv` | White-headed langur PAM clips (PAM-0001 – PAM-2250; anonymized identifiers) | Random split, Python `random.seed(42)`, 8:2 | 1,800 / 450 |
| `FROG_split_index.csv` | Anuran recordings (public Zenodo dataset, Cañas et al. 2023; source-recording filenames) | Sorted by recording date–time (encoded in the filename) and split sequentially, 8:2 | 89 / 23 |
| `BIRD_split_index.csv` | Avian recordings (Aldfly, Kaggle Cornell Birdcall Identification; xeno-canto file identifiers) | Sorted by file identifier and split sequentially, 8:2 | 80 / 20 |

## Verification

- The PAM split is exactly reproducible with Python `random.seed(42)` and
  `random.shuffle` (verified: 2,250/2,250 assignments match):

```python
import random
random.seed(42)
ids = [f"PAM-{i:04d}" for i in range(1, 2251)]
random.shuffle(ids)
train, test = ids[:1800], ids[1800:]
```

- The anuran split is strictly chronological: the latest training recording
  (2020-01-11 19:15) precedes the earliest test recording (2020-01-11 20:15);
  no recording contributes to both subsets.
- The avian files are independent field recordings (one file per xeno-canto
  recording), so the identifier-order file-level split guarantees that no
  recording appears in both subsets.

## CSV format

```csv
clip_id,subset
PAM-0001,train
PAM-0002,test
```
