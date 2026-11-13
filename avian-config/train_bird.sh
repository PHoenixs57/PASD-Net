#!/usr/bin/env bash
# =============================================================================
# Reproduce the avian (bird) transfer-learning run.
#
# Full configuration and provenance: avian-config/bird.yaml
# Documentation:                      avian-config/README.md
#
# Usage:
#   ./avian-config/train_bird.sh [FEATURES.f32] [OUTPUT_DIR]
#
# Defaults (resolved relative to this script, i.e. the repository root):
#   FEATURES   = <repo-root>/feature.f32
#   OUTPUT_DIR = <repo-root>/models/birds-model
#
# This script assumes avian-config/ is placed at the root of the PASD-Net
# source tree, next to torch/ and src/.
#
# NOTE: the bird checkpoints were trained with the bird-branch attention gate
#       attn_out = x * (0.5 + 0.5 * sigmoid(0.5 * attn_dense(x)))
#   and with the bird pitch settings (minperiod = 20, silence threshold =
#   0.01f) in src/denoise.c. See bird.yaml -> required_code_settings.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CALLER_PWD="$PWD"

FEATURES="${1:-$SCRIPT_DIR/../feature.f32}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/../models/birds-model}"

# Resolve user-supplied relative paths against the caller's directory, not
# against avian-config/.
case "$FEATURES" in
    /*) ;;
    *) FEATURES="$CALLER_PWD/$FEATURES" ;;
esac
case "$OUTPUT_DIR" in
    /*) ;;
    *) OUTPUT_DIR="$CALLER_PWD/$OUTPUT_DIR" ;;
esac

PYTHON="${PYTHON:-python3}"

# Run from the training directory, as in the original run.
cd "$SCRIPT_DIR/../torch/rnnoise"

"$PYTHON" ./train_rnnoise.py \
    "$FEATURES" "$OUTPUT_DIR" \
    --epochs 50 \
    --batch-size 128 \
    --lr 5e-4 \
    --sequence-length 2000 \
    --lr-decay 5e-5 \
    --gamma 0.25 \
    --cond-size 128 \
    --gru-size 384
