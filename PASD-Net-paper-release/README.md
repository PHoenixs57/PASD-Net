# PASD-Net (paper release)

This directory is a slimmed-down, GitHub-friendly “paper release” extracted from the original project. It contains only what is needed to reproduce the paper results:

- Core PASD-Net C implementation (`src/`, `include/`)
- Demo program (`examples/`)
- PyTorch training / inference / weight export scripts (`torch/pasd_net/`)
- Autotools build entrypoints (`autogen.sh`, `configure.ac`, `Makefile.am`, `m4/`)

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
./dump_features [-rir_list <rir_list.txt>] <speech.pcm> <noisy.pcm> <features.f32> <count>
```

Inputs are raw 16-bit little-endian mono PCM at 48 kHz:

- `speech.pcm`: clean speech
- `noisy.pcm`: already-mixed noisy speech (speech + noise, etc.)

`count` controls how many sequences to generate. Each sequence is about 10 seconds (1000 frames at 10 ms per frame). The output is appended as binary float32.

Example:

```bash
./dump_features clean_speech.pcm noisy_speech.pcm features.f32 5000
```

### 2.2 (Optional) Parallel feature dumping

If you have GNU `parallel` installed, you can use `scripts/dump_features_parallel.sh` to run many `dump_features` jobs and concatenate the results:

```bash
scripts/dump_features_parallel.sh ./dump_features clean_speech.pcm noisy_speech.pcm features.f32 100 rir_list.txt
```

Note: the script runs 400 parallel shards by default, so the total number of generated sequences is approximately `400 * count`.

### 2.3 (Optional) Create an RIR list

`dump_features` supports `-rir_list <rir_list.txt>` to apply random room impulse responses (RIR) during data generation. The RIR files are expected to be raw float32 arrays (written with `.tofile()`), one per line in `rir_list.txt`.

This repository includes a helper to extract an RIR from a recorded sweep WAV:

```bash
python3 scripts/rir_deconv.py <recorded_sweep.wav> <rir.f32>
echo "$PWD/rir.f32" > rir_list.txt
```

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

To keep the repository small, this release does not include `models/` (it is hundreds of MB in the original project and contains training artifacts).

Download the model (with hash verification):

```bash
./download_model.sh
```

## 5) License

See `LICENSE`.

