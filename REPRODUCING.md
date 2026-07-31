# Reproducing the Paper's Experiments

This guide maps every table in the paper (main text and appendix) to the exact
command, configuration, checkpoints, and result artifact that produce it. It is
written for reviewers working from the code-only release bundle.

Machine-checked source of truth: [docs/reproducibility_manifest.yaml](docs/reproducibility_manifest.yaml)
(validated by `python3 scripts/verify_reproducibility_manifest.py`, which is part
of the release smoke check). The per-table map below is rendered from that file.

## 1. Environment

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

Exact library and driver versions are listed in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).
Training and the heavier evaluations assume a single 16 GB GPU; retrieval-only
diagnostics run on CPU.

## 2. Data

Dataset layout, sensor metadata, and the NCLT evaluation protocol are specified
in [DATA.md](DATA.md). All commands read the dataset root from `NSD_DATA_ROOT`
(default `./data`).

## 3. Checkpoints and fitted artifacts

The bundle is code-only. The three frozen refiner checkpoints and the two
post-hoc artifacts are restored per [artifacts/MANIFEST.md](artifacts/MANIFEST.md):

```bash
bash scripts/restore_release_artifacts.sh
```

Expected paths (referenced by every evaluation command):

- `checkpoints/800d_4sensor_20260511_161726/best_model.pth` (seed 1)
- `checkpoints/800d_4sensor_seed2_20260604_125426/best_model.pth` (seed 2)
- `checkpoints/800d_4sensor_seed3_20260623_132834/best_model.pth` (seed 3)
- `artifacts/key_remetrize_r.npy` (key re-metrization vector)
- `artifacts/edge_classifier.pt` (learned edge selector)

To regenerate the two artifacts instead of restoring them:
`python3 scripts/fit_key_remetrize.py` and
`python3 scripts/build_and_train_edge_classifier.py`.

## 4. Sanity check

```bash
bash scripts/verify_release_smoke.sh
```

This verifies required files, shell syntax, stale wording, Python compilation,
and the reproducibility manifest. `RUN_TESTS=1` additionally runs the unit-test
subset (requires `torch_geometric`).

## 5. Mandatory CLI overrides

The YAML defaults describe an older 544D encoder policy. **Every** paper-main
training or evaluation command needs:

```bash
--encoder-preset no_interdiff                    # 288D magnitude key
--use-gated-context --gate-initial-alpha 0.0625  # gated 128D context
```

Without these you reproduce the historical 544D/672D ablation baseline, not the
reported system.

## 6. Headline results (Table 1)

Both operating points reuse the same three frozen checkpoints; there is no
retraining. Per seed:

```bash
python3 scripts/_pbev_branch_fusion.py --seed 1        # NSD (800-float state)
python3 scripts/_hspec_branch_fusion.py --seed 1 --freqs 12   # NSD-H (1,024-float state)
```

Repeat for seeds 2 and 3, then aggregate (fusion-tuple selection on the six
KITTI/NCLT training sequences + 3-seed summary over the nine validation
sequences):

```bash
python3 scripts/_final_fix8_agg.py
```

Retraining a refiner seed from scratch (about 1.6 h on one 16 GB GPU):

```bash
python3 train_multi_dataset.py --config configs/training_multi_dataset.yaml \
  --encoder-preset no_interdiff --use-gated-context --gate-initial-alpha 0.0625
```

Baseline rows: analytic baselines are deterministic re-evaluations; learned
baselines (BEVPlace++, SeqOT, P-GAT, RING#-L) use the external repositories
pinned by commit in `docs/reproducibility_manifest.yaml`
(`external_repositories`), which must be cloned separately at those commits.
The BEVPlace++ multi-sensor fine-tuning protocol is specified in the paper
appendix and implemented by `scripts/finetune_bevplace.py`.

## 7. Per-table reproduction map

Rendered from `docs/reproducibility_manifest.yaml`; regenerate after edits with
the smoke check to keep it synchronized with the paper's table labels.

| Paper table | Scope | Command | Result artifacts | Seeds |
|---|---|---|---|---|
| `tab:results` | main | `python3 scripts/_pbev_branch_fusion.py --seed {1,2,3}`<br>`python3 scripts/_hspec_branch_fusion.py --seed {1,2,3} --freqs 12`<br>`python3 scripts/_final_fix8_agg.py`<br>`python3 scripts/_ringpp_eval.py --variant {ring,ringpp}`<br>`python3 scripts/_ringsharp_eval.py --weight <RINGSharp checkpoint>`<br>`python3 scripts/_seqot_eval.py --seqot-checkpoint <...> --gem-checkpoint <...>`<br>`python3 scripts/_pgat_eval_rerank.py --checkpoint <...>`<br>`python3 scripts/finetune_bevplace.py` | `results/height_sota/pbev_branch_fusion{,_seed2,_seed3}.json`<br>`results/height_sota/hspec_branch_fusion{,_seed2,_seed3}.json`<br>`results/ring_eval.json`<br>`results/ringpp_eval.json`<br>`results/ringsharp_eval_fix8.json`<br>`results/seqot_eval.json`<br>`results/pgat_eval*.json`<br>`results/per_query_baselines/bevplace/` | NSD, NSD-H, BEVPlace++, SeqOT, and P-GAT: 3-seed means; RING#-L: one checkpoint; analytic rows: deterministic |
| `tab:rate_mini` | main | `python3 scripts/_tables23_fix8.py && python scripts/_verify_alias_source_ladder.py` | `results/tables23_fix8.json`<br>`results/_alias_source/ladder.json` | 288D deterministic; context-dependent values use 3 seeds |
| `tab:ctx_summary` | main | `python3 scripts/_ctx_perm_fix8.py` | `results/ctx_perm_fix8.json` | 3 seeds; 10 within-sequence permutations per seed |
| `tab:spectrum_roles` | main | `python3 scripts/_pbev_branch_fusion.py --seed {1,2,3}` | `results/height_sota/pbev_branch_fusion{,_seed2,_seed3}.json` | 3-seed means; train-only fusion tuple selection |
| `tab:context_iso` | appendix | `python3 scripts/_tables23_fix8.py`<br>`python3 scripts/_ctx_only_fix8.py` | `results/tables23_fix8.json`<br>`results/ctx_only_fix8.json` | 288D deterministic; 416D/context-only columns are 3-seed means |
| `tab:edge_headline` | appendix | `python3 scripts/_edge_ablation_fix8.py` | `results/edge_ablation_fix8.json` | 3-seed means |
| `tab:edgefeat` | appendix | `python3 scripts/build_and_train_edge_classifier.py --help` | `results/_edge_cls/classifier.pt`<br>`results/_edge_cls/keepk_holdout.json` | single post-hoc classifier trained on the training split |
| `tab:refine_ladder` | appendix | `python3 scripts/run_remetrize_twopass_eval.py seed1 seed2 seed3`<br>`python3 scripts/run_remetrize_twopass_rerank.py raw seed1 seed2 seed3`<br>`python3 scripts/run_oracle_ceiling_eval.py seed1` | `results/_remetrize_twopass/seed*.json`<br>`results/_remetrize_twopass/rerank_raw_seed*.json`<br>`results/_remetrize_twopass/oracle_seed1.json` | reported rungs are 3-seed means except the explicitly marked seed-1 oracle |
| `tab:ablation` | appendix | `bash scripts/_run_deployed_ablation.sh` | `results/deployed_ablation.json` | one retrain per row |
| `tab:datasets` | appendix | — | `DATA.md` | not_applicable |
| `tab:hyperparams` | appendix | `python3 train_multi_dataset.py --config configs/training_multi_dataset.yaml --encoder-preset no_interdiff --use-gated-context --gate-initial-alpha 0.0625` | `artifacts/key_remetrize_r.npy`<br>`artifacts/edge_classifier.pt` | fixed constants; no aggregation |
| `tab:noninteger_yaw` | appendix | `python3 scripts/_verify_noninteger_yaw.py` | `results/_alias_source/noninteger_yaw.json` | deterministic |
| `tab:rotation` | appendix | `python3 scripts/compute_yaw_recall.py` | `results/aliasrate_yaw_split.json`<br>`results/per_query_b3/` | historical control; see table caption |
| `tab:aliasrate` | appendix | `python3 scripts/_dump_544_672.py`<br>`python3 scripts/_aliasrate_544672_recompute.py`<br>`python3 scripts/_verify_aliasrate_main.py` | `results/_alias_source/aliasrate_544672.json`<br>`results/_verify_aliasrate_main.json` | 288D deterministic; 416D uses 3 seeds |
| `tab:miss_decomp` | appendix | `python3 scripts/_verify_samespread.py` | `results/_verify_samespread.json` | 288D deterministic; 416D uses 3 seeds |
| `tab:rate_ladder_full` | appendix | `python3 scripts/_tables23_fix8.py`<br>`python3 scripts/_rate_rungs_fix8.py`<br>`python3 scripts/_verify_alias_source_ladder.py` | `results/tables23_fix8.json`<br>`results/rate_rungs_fix8.json`<br>`results/_alias_source/ladder.json` | deterministic |
| `tab:alias_source` | appendix | `python3 scripts/_verify_alias_source_ladder.py` | `results/_alias_source/ladder.json` | encoder values deterministic; context values use dumped seed-1 state where labeled |
| `tab:identifiability` | appendix | `python3 scripts/_dump_final416_seed1.py`<br>`python3 scripts/_verify_identifiability.py` | `results/_final416_seed1/`<br>`results/_alias_source/identifiability.json` | seed 1 |
| `tab:causal` | appendix | `python3 scripts/_causal_eval_only.py` | `results/_remetrize_twopass/causal_eval_only.json` | 3-seed means |
| `tab:ratedist` | appendix | `python3 scripts/_final_fix8_agg.py`<br>`python3 scripts/_ringsharp_eval.py --weight <RINGSharp checkpoint>` | `results/height_sota/*branch_fusion*.json`<br>`results/ringsharp_eval_fix8.json` | headline operating points: 3-seed means; RING#-L: one checkpoint |
| `tab:transfer` | appendix | `bash scripts/_run_transfer.sh` | `results/_task3/joint_s{1,2,3}.json`<br>`results/_task3/transfer_lonclt_s{1,2}.json` | leave-HDL-32E-out: 2 seeds; matched joint comparison: 3 seeds; other rows as reported |
| `tab:yaw_quad` | appendix | `python3 scripts/_dump_b3_yaw_perquery.py seed1 seed2 seed3` | `results/per_query_b3/` | 3-seed means |
| `tab:encoder_preset` | appendix | `python3 scripts/evaluate_kitti_checkpoint.py --config configs/training_multi_dataset.yaml --encoder-preset <preset> --skip-checkpoint --sequences 00 05 08` | `results/encoder_ablation_*_phase256_n400.json` | deterministic encoder control |
| `tab:e2e_latency` | appendix | `NSC_N_COARSE=800 python scripts/_bench_latency_memory.py` | `results/latency_memory_bench.json` | seed 1 |
| `tab:ksweep` | appendix | `python3 scripts/_ksweep_eval.py` | `results/_remetrize_twopass/ksweep_{seed1,seed2,seed3}.json`<br>`results/_remetrize_twopass/ksweep_3seed_summary.json` | Recall: 3-seed summary; latency: seed-1 instrumentation |
| `tab:fixed_budget` | appendix | `python3 scripts/_fixed_budget_fusion.py --tags K00 K05 K08 N12 N13 TOWN DCC KAI RIV --require-complete-nine --candidate-budgets 50 100 200 --height-variant raw_height240 --seed 1` | `results/supplementary/fixed_budget_9seq_regenerated_candidate_seed1.json`<br>`results/supplementary/fixed_budget_9seq_summary.json` | seed 1 only; candidate-set oracle recall, not end-to-end Recall@1 |

## 8. Supplementary controls (appendix)

Guarded runner for the heavier supplementary jobs (dry-run by default; GPU jobs
require `NSD_ALLOW_GPU_EVALUATION=1` / `NSD_ALLOW_GPU_TRAINING=1`):

```bash
bash scripts/run_supplementary_plan.sh list
```

Standalone commands for the appendix controls:

```bash
# Fixed-total-candidate-budget control (candidate-set oracle)
python3 scripts/_fixed_budget_fusion.py --tags K00 K05 K08 N12 N13 TOWN DCC KAI RIV \
  --require-complete-nine --candidate-budgets 50 100 200 --height-variant raw_height240 --seed 1

# Raw-grid height control + fusion-tuple sensitivity
bash scripts/run_supplementary_plan.sh raw-grid-height --execute
python3 scripts/_summarize_fusion_tuple_sensitivity.py \
  --input results/supplementary/raw_grid_height_seed1.json --variant raw_height240

# Temporal-edge pose-attribute sensitivity
python3 scripts/_pose_noise_sensitivity.py --device cuda \
  --conditions clean translation_0p25m translation_1m yaw_2deg yaw_10deg joint_1m_10deg \
  --noise-seeds 0 1 2 --output results/supplementary/pose_noise_sensitivity.json

# Simple learned temporal baseline (mean-pool + MLP)
python3 scripts/_mean_pool_mlp_baseline.py --device cuda --seeds 1 2 3 --epochs 40 \
  --triplets 60000 --batch-size 512 --checkpoint-root checkpoints/supplementary_b4 \
  --output results/supplementary/mean_pool_mlp_3seed.json

# Causal retraining (three seeds; GPU training)
NSD_ALLOW_GPU_TRAINING=1 bash scripts/run_supplementary_plan.sh causal-retrain --execute

# RING#-L candidate-budget sensitivity (requires the RING#-L checkpoint)
RINGSHARP_WEIGHT=<path> NSD_ALLOW_GPU_EVALUATION=1 \
  bash scripts/run_supplementary_plan.sh ringsharp-n13 --execute
```

Protocol labels for these controls (candidate-set oracle, regenerated-cache
protocol, six-sequence raw-416D protocol) are defined in the paper appendix's
protocol map; the numbers they produce are not Table 1 values.

## 9. Notes and caveats

- **Seeds.** Headline NSD/NSD-H values are 3-seed means (seed standard
  deviations 0.0020 / 0.0009). Single-seed runs land within that band.
- **NCLT caches.** The NCLT evaluations preprocess from raw NCLT data per
  DATA.md. Appendix results labeled *regenerated-cache protocol* used a
  regenerated candidate cache; they are internal comparisons, not Table 1
  reproductions.
- **Ablation-only paths.** Configs and flags marked ablation/negative-result in
  [configs/README.md](configs/README.md) (sensor_gate, dual_stream, physics3,
  two-pass train-time refinement, learnable re-metrization) are not part of the
  reported system.
- **Determinism.** Analytic components (encoder, re-metrization vector,
  cyclic alignment) are closed-form; only refiner training and the edge
  classifier involve stochastic optimization.
