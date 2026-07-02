#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

docs=(
  README.md
  RELEASE_800D.md
  RELEASE_CHECKLIST.md
  DATA.md
  EXPERIMENT_HANDOFF.md
  configs/README.md
  artifacts/MANIFEST.md
)

required_files=(
  "${docs[@]}"
  scripts/run_paper_kitti_closed_form.sh
  scripts/run_paper_kitti_residual.sh
  scripts/run_paper_nclt_physics3_control.sh
  scripts/run_retrain_combine_eval.sh
  scripts/make_release_bundle.sh
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing consistency target: $path" >&2
    exit 1
  fi
done

if grep -R -n -E "INCLUDE_DATA|DATA_DIRS|source \\+ training|training-data bundle|nsd-800d-[0-9]+-data" \
  "${docs[@]}" scripts/make_release_bundle.sh scripts/verify_release_smoke.sh >/tmp/nsd_release_data_mode.txt; then
  cat /tmp/nsd_release_data_mode.txt >&2
  echo "data-bundle wording or code found; release must be code-only" >&2
  exit 1
fi

if grep -R -n -E "0\\.9496|zero-shot performance|older README|Current Sequential|current main upgrade|GAT learns phase|NSD full|full NSD" \
  "${docs[@]}" >/tmp/nsd_release_stale.txt; then
  cat /tmp/nsd_release_stale.txt >&2
  echo "stale release wording found" >&2
  exit 1
fi

for script in scripts/run_paper_kitti_closed_form.sh scripts/run_paper_kitti_residual.sh scripts/run_paper_nclt_physics3_control.sh; do
  grep -q -- "--encoder-preset no_interdiff" "$script" || {
    echo "paper runner missing no_interdiff override: $script" >&2
    exit 1
  }
  grep -q -- "--use-gated-context --gate-initial-alpha 0.0625" "$script" || {
    echo "paper runner missing fixed-alpha GAT override: $script" >&2
    exit 1
  }
done

grep -qi "appendix" scripts/run_retrain_combine_eval.sh || {
  echo "run_retrain_combine_eval.sh must be explicitly marked appendix-only" >&2
  exit 1
}

if compgen -G "dist/nsd-*.tar.gz" >/dev/null; then
  for archive in dist/nsd-*.tar.gz; do
    if tar -tzf "$archive" | grep -E '(^|/)(data|results|logs|_handoff|checkpoints)(/|$)|DATA_BUNDLE|\.pth$|\.pt$|__pycache__|\.pytest_cache' >/tmp/nsd_release_archive_forbidden.txt; then
      echo "forbidden payload in archive: $archive" >&2
      cat /tmp/nsd_release_archive_forbidden.txt >&2
      exit 1
    fi
  done
fi

echo "release consistency audit passed"
