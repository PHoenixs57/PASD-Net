# PASD-Net

Official code and paper artifacts for **"PASD-Net: A sensing and denoise
network for wildlife passive acoustic monitoring"** (Ecological Informatics).

| Directory | Contents |
|---|---|
| [`PASD-Net-paper-release/`](PASD-Net-paper-release/) | Source code (C + PyTorch), pre-trained weights, train/test splits, per-segment evaluation records, clip-level labels, evaluation scripts and dataset metadata — see its [README](PASD-Net-paper-release/README.md) |
| [`data/`](data/) | Paired noisy/denoised example clips for the three studied taxa (white-headed langur, anuran, avian) |

Raw white-headed langur recordings cannot be redistributed because location
information of this critically endangered species is sensitive; see
[`PASD-Net-paper-release/docs/dataset-metadata.md`](PASD-Net-paper-release/docs/dataset-metadata.md)
for the non-sensitive recording metadata and what is withheld.

## Quick start

```bash
cd PASD-Net-paper-release
./autogen.sh && ./configure && make -j          # C library + demo
cd torch/pasd_net
python3 inference_pasd_net.py ../../weights/pasd_net_langur.pth noisy.pcm out.pcm
```

See [`PASD-Net-paper-release/README.md`](PASD-Net-paper-release/README.md) for
feature extraction, training, model configurations and the reproducibility
scope of the released artifacts.
