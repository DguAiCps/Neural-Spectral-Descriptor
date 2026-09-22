#!/usr/bin/env bash
set -euo pipefail

# NSD-only environment-disjoint supplementary control. Run inside the
# ring_baseline container, whose repository mount is /workspace/Neural-Spectral-Codec.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python3 scripts/prepare_place_disjoint_nsd.py

config="results/supplementary/place_disjoint_nsd/fixed_protocol.yaml"
sequences=(
  kitti/00 kitti/05 kitti/08
  nclt/2012-01-08 nclt/2013-01-10
  helipr/Town01
  mulran/DCC03 mulran/KAIST03 mulran/Riverside03
)

for seed in 1 2 3; do
  checkpoint_dir="checkpoints/place_disjoint_nsd_seed${seed}"
  final_checkpoint="${checkpoint_dir}/final_model.pth"
  if [[ ! -f "${final_checkpoint}" ]]; then
    python3 train_multi_dataset.py \
      --config "${config}" \
      --encoder-preset no_interdiff \
      --use-gated-context \
      --gate-initial-alpha 0.0625 \
      --checkpoint-dir "${checkpoint_dir}" \
      --seed "${seed}" \
      2>&1 | tee "logs/place_disjoint_nsd_seed${seed}.log"
  fi

  python3 scripts/_eval_deployed_ablation.py v0 \
    --config "${config}" \
    --checkpoint "${final_checkpoint}" \
    --label "NSD environment-disjoint, fixed epoch 100, seed ${seed}" \
    --training-seed "${seed}" \
    --sequences "${sequences[@]}" \
    --out "results/supplementary/place_disjoint_nsd/eval_seed${seed}.json"
done

python3 scripts/summarize_place_disjoint_nsd.py
