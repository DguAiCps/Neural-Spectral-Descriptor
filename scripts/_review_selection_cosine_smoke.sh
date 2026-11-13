#!/bin/bash
set -euo pipefail
python3 -u scripts/_review_selection_cosine_snapshot.py
exec python3 -u scripts/_review_selection_full_cosine.py --manifest results/pdv2_review_selection_full_20261105/cosine_acceleration_audit/manifest.json
