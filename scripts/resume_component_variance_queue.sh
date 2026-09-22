#!/usr/bin/env bash
# Continue the interrupted component-variance queue after a host restart.
# Training checkpoints are intentionally not treated as resumable because the
# trainer reloads only model weights, not optimizer/epoch state.
set -euo pipefail

repo=/workspace/Neural-Spectral-Codec
cd "$repo"
export NSD_DATA_ROOT=/workspace/data
export PYTHONDONTWRITEBYTECODE=1

# Preserve the interrupted v2/seed3 run and restart it as a valid full seed.
partial_dir=checkpoints/supplementary_abl_v2_seed3
partial_backup=checkpoints/interrupted_supplementary_abl_v2_seed3_reboot_20260730
if [[ -d "$partial_dir" && ! -f results/supplementary/component_v2_seed3.json ]]; then
    if [[ -e "$partial_backup" ]]; then
        echo "Refusing to overwrite preserved partial checkpoint: $partial_backup" >&2
        exit 2
    fi
    mv "$partial_dir" "$partial_backup"
    echo "Preserved interrupted v2/seed3 checkpoint at $partial_backup"
fi

for pair in "v2 3" "v3 2" "v3 3" "v4 2" "v4 3" "v5 2" "v5 3"; do
    set -- $pair
    variant=$1
    seed=$2
    result="results/supplementary/component_${variant}_seed${seed}.json"
    checkpoint_dir="checkpoints/supplementary_abl_${variant}_seed${seed}"

    if [[ -f "$result" ]]; then
        echo "${variant}/seed${seed}: existing result, skipping"
        continue
    fi
    if [[ -d "$checkpoint_dir" ]]; then
        echo "Refusing to reuse incomplete checkpoint directory: $checkpoint_dir" >&2
        exit 2
    fi

    python3 train_multi_dataset.py \
        --config "configs/_abl_deployed_${variant}.yaml" \
        --data-root "$NSD_DATA_ROOT" \
        --encoder-preset no_interdiff \
        --use-gated-context \
        --gate-initial-alpha 0.0625 \
        --checkpoint-dir "$checkpoint_dir" \
        --seed "$seed"
    python3 scripts/_eval_deployed_ablation.py "$variant" \
        --config "configs/_abl_deployed_${variant}.yaml" \
        --checkpoint "$checkpoint_dir/best_model.pth" \
        --training-seed "$seed" \
        --out "$result"
done

chown -R 1000:1000 results/supplementary
python3 scripts/aggregate_component_variance.py \
    --seed1 results/deployed_ablation.json \
    --supplementary-dir results/supplementary \
    --output results/supplementary/component_variance_3seed.json
python3 scripts/collect_supplementary_provenance.py
