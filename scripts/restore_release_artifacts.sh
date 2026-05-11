#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SRC_DIR="${SRC_DIR:-_handoff/nsd_encoder_ablation_checkpoints}"
DST_DIR="${DST_DIR:-results/train_no_interdiff_288_gate00625_seed1}"

MAIN_CKPT="$SRC_DIR/no_interdiff_288_gate00625_seed1_best_model.pth"
RAW_CKPT="$SRC_DIR/no_interdiff_288_seed0_best_model.pth"

if [[ ! -f "$MAIN_CKPT" ]]; then
  echo "Missing checkpoint: $MAIN_CKPT" >&2
  echo "Set SRC_DIR to the handoff checkpoint directory or restore the _handoff archive." >&2
  exit 1
fi

mkdir -p "$DST_DIR"
cp "$MAIN_CKPT" "$DST_DIR/best_model.pth"

if [[ -f "$RAW_CKPT" ]]; then
  mkdir -p results/train_no_interdiff_288_seed0
  cp "$RAW_CKPT" results/train_no_interdiff_288_seed0/best_model.pth
fi

echo "Restored paper checkpoint to $DST_DIR/best_model.pth"
