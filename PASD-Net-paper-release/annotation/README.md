# Evaluation records (`annotation/`)

These workbooks contain the **per-segment evaluation records** behind the
objective-acoustic results in the paper: Table 5 (transfer learning),
Supplementary Tables S1–S3 (stratified noise regimes), the Table 6 / Figure 2
model comparisons, and the VAD segmentation used for metric computation.

| File | Taxon / dataset | Models compared |
|---|---|---|
| `anuran-index.xlsx` | Anuran (public Zenodo dataset) | Noisy vs fine-tuned PASD-Net |
| `bird-index.xlsx`   | Avian (Aldfly, Kaggle dataset) | Noisy vs fine-tuned PASD-Net |
| `data-index.xlsx`   | White-headed langur PAM test partition | Noisy vs T-1 … T-7 |
| `langur_pamd_labels.csv` | White-headed langur PAMD collection (independent detection dataset) | Clip-level labels for DeepADN training and evaluation (Table 4) |

`langur_pamd_labels.csv` contains one row per 12-s clip (`clip_id, subset,
label`). **Label definition:** positive = the clip contains at least one
target long-distance call (snort, roar, or wahoo); negative = no target call
is present. Recall and F1 for the downstream detection experiment (Table 4)
were computed at the clip level and averaged over the five runs.

## Sheet structure (identical across files)

- **meta** — evaluation configuration: sample rate (48 kHz), bit depth (16-bit),
  mono; VAD settings (frame 20 ms, hop 10 ms, threshold 0.1, minimum segment
  duration 0.1 s); baseline label.
- **compare_summary** — mean and variance of each metric per condition
  (Noisy baseline vs denoised/model outputs).
- **compare_segments** — one row per VAD segment: segment boundaries
  (`start_sample` / `end_sample`, in samples at 48 kHz; **offsets relative to
  the evaluated stream**) and all metrics for every condition, including
  per-segment SNR, SegSNR, SI-SNR, PESQ, STOI and MSE.
- **snr_binning** — the four input-SNR regimes (<0, 0–5, 5–10, >10 dB) with
  per-regime means/variances and deltas relative to the noisy baseline.

Note on row counts: rows correspond to VAD-detected vocal segments, not raw
clips — e.g., the white-headed langur independent test partition comprises 450
clips, from which VAD extracts 336 voiced segments (clips without a detected
vocal segment contribute no rows).

## Identifier namespaces

Two disjoint PAM collections appear in this release, with deliberately distinct
identifier prefixes:

| Prefix | Collection | Where |
|---|---|---|
| `PAM-NNNN` (PAM-0001 – PAM-2250) | PAM partition used to train and evaluate the PASD-Net **denoising** model | `splits/PAM_split_index.csv` |
| `PAMD-NNNN` | Independent PAM dataset collected separately, used to evaluate **downstream detection** (DeepADN, Table 4) | clip-level label file in this folder |

The two collections share no recordings (the detection dataset was collected
separately from the denoising training/test material); the distinct prefixes
make this explicit.

## Anonymization statement

Local absolute paths have been removed from the `meta` sheets. Segment
boundaries are sample offsets within each evaluated stream and contain **no
location, device-identifier, or deployment-time information**. The white-headed
langur file covers the independent PAM test partition only; no training audio
or deployment metadata is included.
