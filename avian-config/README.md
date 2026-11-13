# Avian (bird) transfer-learning configuration

This folder contains the complete configuration for the bird-vocalization
transfer-learning experiment reported in the PASD-Net paper. It is kept in a
separate `avian-config/` directory so that the experiment can be reviewed and
reproduced without searching through the rest of the repository.

## Contents

| File | Description |
|---|---|
| [`bird.yaml`](bird.yaml) | Complete configuration: data layout, feature front-end, network, optimizer, loss, and the code settings that differ from the speech defaults. |
| [`train_bird.sh`](train_bird.sh) | Launcher with the exact training arguments used by the run. |
| `README.md` | This document: provenance, reproduction steps, and compatibility notes. |

## Placement

`bird.yaml` and `train_bird.sh` use paths relative to the **PASD-Net source
tree root** — the directory containing `src/`, `torch/`, `feature.f32`, and
`models/`. Keep this folder at that root:

```text
PASD-Net/
├── avian-config/
│   ├── bird.yaml
│   ├── train_bird.sh
│   └── README.md
├── src/
├── torch/
│   └── rnnoise/
│       ├── train_rnnoise.py
│       └── rnnoise.py
├── feature.f32        (generated; not distributed)
└── models/            (generated; not distributed)
```

The public paper release keeps the C sources under `PASD-Net-paper-release/`.
When applying the code settings below, use the corresponding file in that
tree (`PASD-Net-paper-release/src/denoise.c`). The RNNoise training scripts
(`torch/rnnoise/train_rnnoise.py` and `torch/rnnoise/rnnoise.py`) are part of
the PASD-Net source tree used for the run; they are required to retrain the
model.

## Experiment at a glance

| Item | Value |
|---|---|
| Task | Bird-vocalization transfer-learning run |
| Training data | `feature.f32`, 98 float32 values per frame |
| Frames | 200,000,000 (100,000 sequences × 2,000 frames) |
| Input / output | 65 front-end features → 32 gains + 1 VAD |
| Network | `rnnoise.RNNoise`, `cond_size=128`, `gru_size=384`, 3 GRU layers, attention gate |
| Parameters | 3,032,609 |
| Initialization | random (no initial checkpoint), no sparsification |
| Optimizer | AdamW, lr 5e-4, betas (0.8, 0.98), eps 1e-8, weight decay 0.01 |
| Schedule | 5,000-step warm-up, then `1/(1 + 5e-5·(step−5000))` |
| Batch / epochs | 128 / 50 (781 steps per epoch, 39,050 total) |
| Loss | perceptual gain loss (γ = 0.25) + 0.001 × VAD loss |
| Final training loss | 0.010415596585177008 (epoch 50) |

## Provenance

No configuration file was stored with the original run. Every value in
`bird.yaml` was recovered from the run artifacts:

| Evidence | Value established |
|---|---|
| `models/birds-model/checkpoints/rnnoise_1.pth` … `rnnoise_50.pth` | 50 epochs; final checkpoint at epoch 50 |
| `torch/rnnoise/nohup.out` | 781 batches/epoch, 50 epochs, `model: 3032609 weights` |
| `feature.f32` size: 78,400,000,000 bytes | 200,000,000 frames × 98 float32 |
| checkpoint `model_kwargs` | `cond_size=128`, `gru_size=384`, attention enabled |
| checkpoint epoch 50 | loss `0.010415596585177008` |
| final checkpoint weights | 0% zeros → no `--sparse` |

The 200,000,000 frames correspond to 100,000 training sequences of 2,000
frames. With batch size 128, that gives `floor(100000 / 128) = 781` steps per
epoch. `dump_features` generated 200,000 sequences of 1,000 frames each,
which concatenate to the same 200,000,000 frames.

## Reproduce

The commands below assume the PASD-Net source tree used for the run
(including `torch/rnnoise/train_rnnoise.py`, `torch/rnnoise/rnnoise.py`, and
the C front-end under `src/`) is at the repository root, with this folder as
`avian-config/`.

In the public paper release the C sources live under
`PASD-Net-paper-release/` instead. In that layout, run the build and
`dump_features` steps from `PASD-Net-paper-release/` and adjust the relative
paths below accordingly. The RNNoise training scripts referenced by
`train_bird.sh` are part of the authors' source tree and are required to
retrain the model.

### 1. Build the C front-end

From the source-tree root (the repository root in the authors' layout, or
`PASD-Net-paper-release/` in the public release):

```bash
./autogen.sh
./configure
make -j
```

### 2. Apply the bird-specific code settings

These are the only code differences from the speech-default configuration:

| File | Setting |
|---|---|
| `src/denoise.c` | `const int minperiod = 20;` (speech default: `PITCH_MIN_PERIOD = 60`) |
| `src/denoise.c` | `const float silence_E_threshold = 0.01f;` (speech default: `0.04f`; inference only) |
| `torch/rnnoise/rnnoise.py` | `attn_out = x * (0.5 + 0.5 * sigmoid(0.5 * attn_dense(x)))` |

The third item is essential: the bird checkpoints were trained with the
0.5–1.0 attention gate, which is also what `src/nnet.c::compute_attention_gate`
implements. The current `torch/rnnoise/rnnoise.py` in the PASD-Net tree scales
the gate to 0.95–1.0; restore the 0.5–1.0 form before retraining or before
loading the bird checkpoints for inference.

### 3. Generate `feature.f32`

The feature file is about 78.4 GB and is not distributed:

```bash
./dump_features speech.pcm background_noise.pcm foreground_noise.pcm feature.f32 200000
```

`speech.pcm`, `background_noise.pcm`, and `foreground_noise.pcm` must be raw
48 kHz / 16-bit / mono PCM. The bird-vocalization corpus used for the paper is
site-specific and is not redistributed with the repository; substitute the
corresponding recordings. No RIR list was used for this run.

### 4. Train

From the repository root:

```bash
./avian-config/train_bird.sh
```

This is equivalent to:

```bash
cd torch/rnnoise
python3 train_rnnoise.py ../../feature.f32 ../../models/birds-model \
    --epochs 50 --batch-size 128 --lr 5e-4 \
    --sequence-length 2000 --lr-decay 5e-5 --gamma 0.25 \
    --cond-size 128 --gru-size 384
```

Set `PYTHON=/path/to/python` to select a specific interpreter. The launcher
also accepts optional `FEATURES.f32` and `OUTPUT_DIR` arguments:

```bash
./avian-config/train_bird.sh /data/feature.f32 /data/birds-model
```

Expected artifacts:

```text
models/birds-model/checkpoints/rnnoise_1.pth ... rnnoise_50.pth
models/birds-model/train_loss_curves.png
```

## Data availability

- The raw bird recordings are not included in this folder or in the public
  repository.
- `feature.f32` (~78.4 GB) is generated locally from those recordings with
  `dump_features`; it is not committed.
- The original run also produced `bird.csv`, a 3.4 MB `dump_single_features`
  export of one recording. It is not redistributed; `bird.yaml` records its
  format so that an equivalent file can be regenerated with
  `./dump_single_features --pcm -o bird.csv <bird.pcm>`.
