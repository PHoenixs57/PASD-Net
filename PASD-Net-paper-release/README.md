# PASD-Net (paper release)

This directory is a slimmed-down, GitHub-friendly “paper release” extracted from the original project. It contains only what is needed to reproduce the paper results:

- Core PASD-Net C implementation (`src/`, `include/`)
- Demo program (`examples/`)
- PyTorch training / inference / weight export scripts (`torch/pasd_net/`)
- Pre-trained models for all three taxa (`weights/`)
- The exact train/test partitions, per-segment evaluation records and
  clip-level labels used in the paper (`splits/`, `annotation/`)
- Non-sensitive recording metadata (`docs/dataset-metadata.md`)
- Evaluation and figure scripts (`scripts/`)
- Autotools build entrypoints (`autogen.sh`, `configure.ac`, `Makefile.am`, `m4/`)

## Repository contents

```text
<repository root>
├── PASD-Net-paper-release/          code, weights and paper artifacts (this README)
│   ├── src/, include/, examples/      core C implementation, feature tools, demo
│   ├── torch/pasd_net/                PyTorch training / inference / weight export
│   ├── weights/                       pre-trained langur / anuran / avian models (+ SHA-256)
│   ├── splits/                        exact train/test partitions used in the paper (CSV)
│   ├── annotation/                    per-segment evaluation records and clip-level labels
│   ├── scripts/                       SNR & SI-SDR, F0 analysis, spectrogram, split generation
│   ├── docs/dataset-metadata.md       non-sensitive recording metadata (no coordinates)
│   └── datasets.txt                   training-data sources for the three taxa
└── data/                            paired noisy/denoised example clips (langur / anuran / avian)
```

**Reproducibility scope.** The scripts in `scripts/` recompute the SNR (and
optional SI-SDR) metrics, the pitch-fidelity analysis (paper Figures 4–5) and
the spectrogram figures (Figure 3) for any audio, including the fully public
anuran and avian datasets. The per-segment values behind the objective-acoustic
results (Table 5, Table 6 / Figure 2, Supplementary Tables S1–S3; including
SegSNR, PESQ, STOI and MSE) are released in `annotation/`, the clip-level
labels for the downstream detection experiment (Table 4) in
`annotation/langur_pamd_labels.csv`, and the exact partitions behind every
experiment in `splits/` — see the README files in those folders. The raw
white-headed langur recordings themselves cannot be redistributed for
conservation reasons (see `docs/dataset-metadata.md`), so langur-side results
can be audited against these released artifacts but cannot be recomputed from
raw audio by third parties.

## 1) Build (C library + demo)

```bash
./autogen.sh
./configure
make -j
```

Run the demo (48 kHz, 16-bit, mono RAW PCM):

```bash
./examples/pasdnet_demo <noisy.pcm> <out.pcm>
```

## 2) Feature extraction (generate `features.f32` for training)

Training uses a single binary float32 feature file (commonly named `features.f32`). Each frame stores:

- `features` (65 dims)
- `ideal gain` target (32 dims)
- `VAD` target (1 dim)

Total: 98 float32 values per frame.

### 2.1 Generate features with the built-in tool

After building, the repository provides a tool named `dump_features`:

```bash
./dump_features [-rir_list <rir_list.txt>] <clean.pcm> <noisy.pcm> <features.f32> <count>
```

Inputs are raw 16-bit little-endian mono PCM at 48 kHz:

- `clean.pcm`: clean reference recording (high-purity target vocalizations)
- `noisy.pcm`: noisy field recording containing the target vocalizations plus background noise

`count` controls how many sequences to generate. Each sequence is about 10 seconds (1000 frames at 10 ms per frame). The output is appended as binary float32.

Example:

```bash
./dump_features clean.pcm noisy.pcm features.f32 5000
```

The training-data sources for the three studied taxa (white-headed langur, anuran, avian) are listed in `datasets.txt`.

### 2.2 (Optional) Room impulse response (RIR) augmentation

`dump_features` accepts an optional `-rir_list <rir_list.txt>` flag that
convolves the generated training material with room impulse responses. Each
line of `rir_list.txt` gives the path to one RIR stored as a raw float32 array
(e.g. written with NumPy's `ndarray.tofile()`); one RIR is drawn at random for
each generated sequence.

## 3) PyTorch training / inference (optional)

See `requirements.txt` for suggested dependencies.

Train:

```bash
cd torch/pasd_net
python3 train_pasd_net.py <features.f32> <output_dir>
```

Inference (example script; input/output are int16 RAW PCM):

```bash
cd torch/pasd_net
python3 inference_pasd_net.py <checkpoint.pth> <noisy.pcm> <out.pcm>
```

## 4) Model files

Pre-trained weights are included in this release under `weights/`:

- `weights/pasd_net_langur.pth` — white-headed langur model (`cond_size=128`, `gru_size=384`, `use_attention=True`; SHA-256 in `weights/checksums.txt`). Build and use with the default configuration (§5)
- `weights/pasd_net_anuran.pth`, `weights/pasd_net_avian.pth` — fine-tuned anuran and avian transfer models (anuran uses the default configuration; avian uses the avian configuration, §5)

The checkpoints store their `model_kwargs`, so `inference_pasd_net.py` reconstructs the correct architecture automatically:

```bash
cd torch/pasd_net
python3 inference_pasd_net.py ../../weights/pasd_net_langur.pth <noisy.pcm> <out.pcm>
```

External hosting (GitHub Release / object storage) is also supported for larger bundles: set `PASDNET_MODEL_URL` and run `./download_model.sh`, which verifies the archive against the SHA-256 in `model_version` before unpacking.

## 5) Model configurations (pitch search & silence gating)

Two configurations of the pitch-related parameters are used in the paper:

| Configuration | PASDNET_PITCH_MIN_PERIOD | PASDNET_SILENCE_E_THRESHOLD | F0 search ceiling | Used for |
|---|---|---|---|---|
| Default | 60 (`PITCH_MIN_PERIOD`) | 0.04 | ~1 kHz | White-headed langur model; anuran fine-tuned model |
| Avian | 20 | 0.01 | ~2.4 kHz | Avian fine-tuned model |

The default build corresponds to the langur/anuran configuration. To build the avian configuration:

```bash
./configure CPPFLAGS="-DPASDNET_PITCH_MIN_PERIOD=20 -DPASDNET_SILENCE_E_THRESHOLD=0.01f"
make -j
```

Features must be extracted with the same configuration used to train the model: use the matching build of `dump_features` and `pasdnet_demo` for the langur/anuran and avian models respectively.

## 6) License

See `LICENSE`.

