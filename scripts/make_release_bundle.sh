#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION="${VERSION:-800d-$(date +%Y%m%d)}"
OUT_DIR="${OUT_DIR:-dist}"
ARCHIVE="${OUT_DIR}/nsd-${VERSION}.tar.gz"

mkdir -p "$OUT_DIR"

bash scripts/verify_release_smoke.sh

tar \
  --exclude=".git" \
  --exclude=".pytest_cache" \
  --exclude="__pycache__" \
  --exclude="*/__pycache__" \
  --exclude="data" \
  --exclude="logs" \
  --exclude="outputs" \
  --exclude="results" \
  --exclude="checkpoints" \
  --exclude="baselines/weights" \
  --exclude="_handoff" \
  --exclude="*.pth" \
  --exclude="*.pt" \
  --exclude="*.pyc" \
  --exclude="dist" \
  -czf "$ARCHIVE" \
  README.md RELEASE_800D.md RELEASE_CHECKLIST.md DATA.md EXPERIMENT_HANDOFF.md \
  artifacts configs scripts src tests \
  requirements.txt setup.py train_multi_dataset.py LICENSE

echo "created ${ARCHIVE}"
