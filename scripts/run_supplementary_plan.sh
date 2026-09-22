#!/usr/bin/env bash
# Safe launcher for docs/SUPPLEMENTARY_EXPERIMENT_PLAN.md.
# Default mode prints commands only.  GPU training requires an explicit
# NSD_ALLOW_GPU_TRAINING=1 execution acknowledgement.
set -euo pipefail

TASK="${1:-list}"
MODE="${2:---dry-run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NSD_DATA_ROOT:-$REPO/data}"

run() {
  printf '+ '; printf '%q ' "$@"; printf '\n'
  if [[ "$MODE" == "--execute" ]]; then
    "$@"
  fi
}

need_gpu_eval() {
  [[ "$MODE" != "--execute" || "${NSD_ALLOW_GPU_EVALUATION:-0}" == "1" ]] || {
    echo "Refusing GPU evaluation: set NSD_ALLOW_GPU_EVALUATION=1 with --execute." >&2; exit 2; }
}

need_gpu_training() {
  [[ "$MODE" != "--execute" || "${NSD_ALLOW_GPU_TRAINING:-0}" == "1" ]] || {
    echo "Refusing GPU training: set NSD_ALLOW_GPU_TRAINING=1 with --execute to confirm execution." >&2; exit 2; }
}

case "$TASK" in
  list)
    cat <<'EOF'
Tasks: audit, raw-grid-height, ringsharp-n13, baseline-latency,
       causal-retrain, component-variance, paired-headline-scpp,
       bevplace-headline, paired-headline-bevplace.
Use --dry-run (default) to inspect.  --execute never starts GPU work unless
the matching NSD_ALLOW_GPU_EVALUATION=1 or NSD_ALLOW_GPU_TRAINING=1 is set.
EOF
    ;;
  audit)
    run python3 "$REPO/scripts/verify_reproducibility_manifest.py"
    run python3 "$REPO/scripts/audit_bispectrum_provenance.py" \
      --output "$REPO/results/supplementary/bispectrum_provenance.json"
    run env NSD_DATA_ROOT="$DATA_ROOT" python3 "$REPO/scripts/audit_headline_data_lineage.py" \
      --output "$REPO/results/supplementary/headline_data_lineage.json"
    run python3 "$REPO/scripts/collect_supplementary_provenance.py"
    ;;
  raw-grid-height)
    run python3 "$REPO/scripts/_height_candidate_fusion.py" --device cpu --pool-k 100 \
      --variants legacy240 raw_height240 height240 \
      --output "$REPO/results/supplementary/raw_grid_height_seed1.json"
    ;;
  ringsharp-n13)
    need_gpu_eval
    if [[ "$MODE" == "--execute" ]]; then
      : "${RINGSHARP_WEIGHT:?set RINGSHARP_WEIGHT to the same-split RING#-L checkpoint}"
    fi
    ringsharp_weight="${RINGSHARP_WEIGHT:-/path/to/same_split_ringsharp_checkpoint.pth}"
    run env NSD_DATA_ROOT="$DATA_ROOT" python3 "$REPO/scripts/_ringsharp_eval.py" \
      --weight "$ringsharp_weight" nclt/2013-01-10 --n-coarse 1000 --n-trans 100 \
      --out "$REPO/results/supplementary/ringsharp_n13_k1000_t100.json"
    ;;
  baseline-latency)
    need_gpu_eval
    run env NSD_DATA_ROOT="$DATA_ROOT" python3 "$REPO/scripts/_bench_baseline_latency.py" \
      --methods bevplace seqot
    ;;
  causal-retrain)
    need_gpu_training
    for seed in 1 2 3; do
      ckpt="$REPO/checkpoints/supplementary_causal_seed${seed}/best_model.pth"
      run env NSD_DATA_ROOT="$DATA_ROOT" python3 "$REPO/train_multi_dataset.py" \
        --config "$REPO/configs/training_multi_dataset.yaml" --data-root "$DATA_ROOT" \
        --encoder-preset no_interdiff --use-gated-context --gate-initial-alpha 0.0625 \
        --temporal-edge-mode past_to_current --checkpoint-dir "${ckpt%/best_model.pth}" --seed "$seed"
      run python3 "$REPO/scripts/_eval_deployed_ablation.py" baseline \
        --config "$REPO/configs/training_multi_dataset.yaml" --checkpoint "$ckpt" \
        --label "causal retraining (past_to_current), seed ${seed}" \
        --training-seed "$seed" \
        --temporal-edge-mode past_to_current \
        --out "$REPO/results/supplementary/causal_retrain_seed${seed}.json"
    done
    ;;
  component-variance)
    need_gpu_training
    for variant in v0 v1 v2 v3 v4 v5; do
      for seed in 2 3; do
        run env NSD_DATA_ROOT="$DATA_ROOT" python3 "$REPO/train_multi_dataset.py" \
          --config "$REPO/configs/_abl_deployed_${variant}.yaml" --data-root "$DATA_ROOT" \
          --encoder-preset no_interdiff --use-gated-context --gate-initial-alpha 0.0625 \
          --checkpoint-dir "$REPO/checkpoints/supplementary_abl_${variant}_seed${seed}" --seed "$seed"
        run python3 "$REPO/scripts/_eval_deployed_ablation.py" "$variant" \
          --config "$REPO/configs/_abl_deployed_${variant}.yaml" \
          --checkpoint "$REPO/checkpoints/supplementary_abl_${variant}_seed${seed}/best_model.pth" \
          --training-seed "$seed" \
          --out "$REPO/results/supplementary/component_${variant}_seed${seed}.json"
      done
    done
    run python3 "$REPO/scripts/aggregate_component_variance.py" \
      --seed1 "$REPO/results/deployed_ablation.json" \
      --supplementary-dir "$REPO/results/supplementary" \
      --output "$REPO/results/supplementary/component_variance_3seed.json"
    ;;
  paired-headline-scpp)
    need_gpu_eval
    headline_seed="${HEADLINE_SEED:-1}"
    headline_root="$REPO/results/per_query_headline/seed${headline_seed}"
    run env NSD_DATA_ROOT="$DATA_ROOT" PYTHONDONTWRITEBYTECODE=1 \
      python3 "$REPO/scripts/_headline_perquery_dump.py" --seed "$headline_seed" \
      --output-root "$headline_root"
    if [[ "$headline_seed" == "1" ]]; then
      run env NSD_DATA_ROOT="$DATA_ROOT" PYTHONDONTWRITEBYTECODE=1 \
        python3 "$REPO/scripts/_scpp_nclt_fix8_perq.py"
    fi
    stats_args=(python3 "$REPO/scripts/_paired_headline_stats.py"
      --headline-root "$headline_root"
      --output "$REPO/results/supplementary/paired_headline_scpp_seed${headline_seed}.json")
    if [[ "$headline_seed" != "1" ]]; then
      stats_args+=(--skip-table1-gate)
    fi
    run "${stats_args[@]}"
    ;;
  bevplace-headline)
    need_gpu_eval
    run env NSD_DATA_ROOT="$DATA_ROOT" PYTHONDONTWRITEBYTECODE=1 \
      python3 "$REPO/scripts/_bevplace_headline_perquery.py" \
      --output-dir "$REPO/results/per_query_baselines_regen/bevplace"
    ;;
  paired-headline-bevplace)
    nsd_root="$REPO/results/per_query_headline/seed1"
    bev_root="$REPO/results/per_query_baselines_regen/bevplace"
    required_files=(KITTI_00.json KITTI_05.json KITTI_08.json
      NCLT_2012-01-08.json NCLT_2013-01-10.json HeLiPR_Town01.json
      MulRan_DCC03.json MulRan_KAIST03.json MulRan_Riverside03.json)
    if [[ "$MODE" == "--execute" && ! -f "$REPO/results/supplementary/paired_headline_scpp_seed1.json" ]]; then
      echo "Refusing learned paired test: complete the seed-1 Table-1 gate first." >&2
      exit 2
    fi
    run python3 "$REPO/scripts/paired_success_stats.py" \
      --left-dir "$nsd_root/nsd" --right-dir "$bev_root" \
      --left-label NSD --right-label BEVPlace++ \
      --require-files "${required_files[@]}" \
      --output "$REPO/results/supplementary/paired_headline_nsd_bevplace_seed1.json"
    run python3 "$REPO/scripts/paired_success_stats.py" \
      --left-dir "$nsd_root/nsd_h" --right-dir "$bev_root" \
      --left-label NSD-H --right-label BEVPlace++ \
      --require-files "${required_files[@]}" \
      --output "$REPO/results/supplementary/paired_headline_nsdh_bevplace_seed1.json"
    ;;
  *) echo "Unknown task: $TASK" >&2; exit 2 ;;
esac
