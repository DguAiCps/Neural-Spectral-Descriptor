# NSD Encoder/GAT Ablation Handoff

Date: 2026-05-10 (reported 800D rows frozen)

This branch is for continuing NSD encoder compression and gated-GAT experiments.
It is intentionally separated from the main paper repository state.

## Reported 800D configuration (FROZEN as of 2026-05-10)

There is exactly one deployed NSD architecture; it applies identically to KITTI, NCLT, HeLiPR, and MulRan.

- Encoder: `no_interdiff` preset (288D invariant magnitude key)
- GAT: fixed-alpha gated DiffAttnConv (`gate_initial_alpha=0.0625`),
  NO sensor-aware gate, NO sensor-balanced sampling
- Retrieval key total: 416D = 288D magnitude + 128D gated context
- Train config: `configs/training_multi_dataset.yaml`
- Phase-alignment sketch: cylindrical+BEV 384D
  (`range 16×4×2 = 128D` + `BEV 16×8×2 = 256D`)
- Reranker: closed-form cyclic-shift cosine (Eq. 6)
- Results source: paper Table 1, full 9 held-out sequences

Eval entry point:

```bash
scripts/evaluate_kitti_checkpoint.py \
  --enable-bev-layout --enable-phase-sketch --phase-rerank-mode sketch_fft
```

## Ablations (paper appendix only)

The following are reported as ablations and ARE NOT part of either reported
800D main row. Two are pure negatives; physics3 is a sensor-dependent trade-off.

### Pure negatives

- Sensor-aware GAT (`gnn.sensor_gate.enabled=true`) — gate stabilizes alpha
  but does not improve retrieval (Appendix `app:sensor_gate`).
- Dual-stream phase GAT (`gnn.dual_stream.enabled=true`) — per-node phase
  summaries cannot substitute for candidate-level cyclic-shift alignment
  (Appendix `app:dual_stream`).

### Trade-off (sensor-dependent)

- Physics3 BEV (`bev.height_encoding=physics3`) — under the same 384D
  phase-sketch budget, reallocating 16×12 → 48×4 frequencies:
  - KITTI 08 reverse-loop: -6.81 %p (max-only 0.8468 → physics3 0.7787).
  - NCLT held-out (KITTI+NCLT checkpoint, n=800): +14.0 %p macro
    (max-only 0.3218 → physics3 0.4619; 2012-01-08 0.2805 → 0.4499,
    2013-01-10 0.3631 → 0.4738).
  The four-sensor row keeps cylindrical+BEV. Physics3 is reported as a
  physically grounded sensor-adaptation ablation, not a pure negative
  (Appendix `app:physics3`, Table `tab:physics3_nclt`).

The runner `scripts/run_retrain_combine_eval.sh` reproduces these
negative ablations and is annotated as ABLATION-ONLY in its header
comment. DO NOT use it as the paper-main reproduction script.

## Remote Workspace

Use the server-side isolated copy:

```bash
cd /rise/RISE1/workspace/impl/Neural-Spectral-Descriptor_encoder_ablation
```

Do not run these experiments in the original remote repository unless you intend
to merge this branch.

## Archived ablation runner: sensor-aware GAT + physics3

The sensor-aware GAT + physics3 appendix chain is automated in:

```bash
bash scripts/run_retrain_combine_eval.sh
```

It runs the following stages in order:

1. Retrain the 416D retrieval key with sensor-aware GAT + `abs_diff` messages.
2. Evaluate GAT-only KITTI 00/05/08.
3. Evaluate the 800D analytic state: 416D GAT key + 384D physics3 phase sketch.

Default config:

```bash
configs/training_multi_dataset_sensor_gat_absdiff.yaml
```

Default result summary:

```bash
results/summary_sensor_gat_absdiff_physics3_seed1.json
```

Useful overrides:

```bash
SEED=2 bash scripts/run_retrain_combine_eval.sh
RUN_NCLT=0 bash scripts/run_retrain_combine_eval.sh
RUN_TESTS=1 bash scripts/run_retrain_combine_eval.sh
```

## What Changed

- Added compact encoder presets:
  - `no_interdiff`: 288D, keeps 16 rows x 9 octave bins x mean/std, removes 256D inter-bin diff.
  - `mean_diff`: 272D, mean plus adjacent-bin diff.
  - `rows12_full`: 408D, reduces vertical rows to 12.
  - `cross4_no_interdiff`: 408D, 288D magnitude + 120D adjacent-row cross-spectrum.
  - `cross8_no_interdiff`: 528D, 288D magnitude + 240D adjacent-row cross-spectrum.
- Added yaw-invariant adjacent-row cross-spectrum:
  - `src/encoding/cross_spectrum.py`
  - Formula: `F_e(k) * conj(F_{e+1}(k)) / (|F_e(k)| |F_{e+1}(k)| + eps)`
  - DC is skipped; only low non-DC frequencies are retained.
- Added learned GAT context gate:
  - `raw_norm + alpha * ctx_norm` via concatenation `[raw_norm, alpha * ctx_norm]`.
  - `--use-gated-context --gate-initial-alpha 0.0625`
  - `forward()` and `forward_with_attention()` now both apply the same gate.
- Added fast KITTI evaluation support:
  - `--skip-checkpoint` for encoder-only ablations.
  - `--ctx-weights` for post-hoc context weighting.
  - phase sketch evaluation with variable frequency budget.

## Current Completed Results

SC++ reference from paper table:

```text
KITTI 00: 0.962
KITTI 05: 0.942
KITTI 08: 0.843
```

Encoder-only preset sweep with 256D phase sketch:

```text
no_interdiff 288D:
  raw       00 0.9209 / 05 0.8223 / 08 0.3957
  +phase256 00 0.9763 / 05 0.9416 / 08 0.7787

mean_diff 272D:
  raw       00 0.9114 / 05 0.8037 / 08 0.3191
  +phase256 00 0.9763 / 05 0.9416 / 08 0.7660

rows12_full 408D:
  raw       00 0.9225 / 05 0.7905 / 08 0.3532
  +phase256 00 0.9731 / 05 0.9416 / 08 0.7787

cross4_no_interdiff 408D:
  raw       00 0.9209 / 05 0.8223 / 08 0.3957
  +phase256 00 0.9763 / 05 0.9416 / 08 0.7787

cross8_no_interdiff 528D:
  raw       00 0.9209 / 05 0.8223 / 08 0.3957
  +phase256 00 0.9763 / 05 0.9416 / 08 0.7787
```

Gated 288D GAT checkpoint:

```text
checkpoint: results/train_no_interdiff_288_gate00625_seed1/best_model.pth
eval:       results/checkpoint_eval_nointerdiff288_gate00625_phase384_n800.json

raw:
  00 0.9209 / 05 0.8223 / 08 0.3957

gated final:
  00 0.9209 / 05 0.8196 / 08 0.3957

best post-hoc context weight:
  00 0.9225 / 05 0.8249 / 08 0.4000

best final + phase384:
  00 0.9810 / 05 0.9523 / 08 0.8383
```

Historical note: this block is an older phase-budget sweep, not the frozen
reported 800D row. Use the top "Reported 800D configurations" section and
`RELEASE_800D.md` for paper numbers.

The handoff archive also includes downloaded copies:

```text
_handoff/nsd_encoder_ablation_checkpoints/no_interdiff_288_seed0_best_model.pth
_handoff/nsd_encoder_ablation_checkpoints/no_interdiff_288_gate00625_seed1_best_model.pth
```

KITTI 08 phase budget sweep using ungated 288D GAT checkpoint:

```text
results/phase_budget_08_no_interdiff288_r4_b8_n800.json: best 08 R@1 = 0.84255
results/phase_budget_08_no_interdiff288_r8_b8_n800.json: best 08 R@1 = 0.84255
```

Interpretation: KITTI 08 is effectively at SC++ level but not cleanly above it
yet. Do not claim "beats SC++ on KITTI 08" unless a subsequent run exceeds
0.843 with the same evaluation protocol.

## Files To Inspect First

```text
src/encoding/cross_spectrum.py
src/encoding/bev_image.py
src/encoding/spectral_encoder.py
src/gnn/model.py
train_multi_dataset.py
scripts/evaluate_kitti_checkpoint.py
scripts/evaluate_nclt_checkpoint.py
scripts/summarize_retrain_combine_eval.py
scripts/run_kitti_operating_point.py
tests/test_cross_spectrum.py
tests/test_gnn_gate.py
```

## Re-run Fast Encoder Ablations

```bash
for preset in no_interdiff mean_diff rows12_full cross4_no_interdiff cross8_no_interdiff; do
  CUDA_VISIBLE_DEVICES=3 python3 scripts/evaluate_kitti_checkpoint.py \
    --config configs/training_multi_dataset.yaml \
    --encoder-preset "$preset" \
    --skip-checkpoint \
    --sequences 00 05 08 \
    --device cuda \
    --cache-dir "data/preprocessed_kitti_encoder_ablation_${preset}" \
    --enable-bev-layout \
    --bev-row-pool 16 \
    --enable-phase-sketch \
    --phase-sketch-only \
    --phase-range-freqs 4 \
    --phase-bev-freqs 4 \
    --n-coarse 400 \
    --output "results/encoder_ablation_${preset}_phase256_n400.json"
done
```

## Re-train Gated 288D GAT

```bash
CUDA_VISIBLE_DEVICES=3 python3 train_multi_dataset.py \
  --config configs/training_multi_dataset.yaml \
  --encoder-preset no_interdiff \
  --use-gated-context \
  --gate-initial-alpha 0.0625 \
  --checkpoint-dir results/train_no_interdiff_288_gate00625_seedX \
  --seed X
```

## Evaluate Gated 288D GAT With Phase

```bash
CUDA_VISIBLE_DEVICES=3 python3 scripts/evaluate_kitti_checkpoint.py \
  --config configs/training_multi_dataset.yaml \
  --encoder-preset no_interdiff \
  --use-gated-context \
  --gate-initial-alpha 0.0625 \
  --checkpoint results/train_no_interdiff_288_gate00625_seedX/best_model.pth \
  --sequences 00 05 08 \
  --device cuda \
  --cache-dir data/preprocessed_kitti_encoder_ablation_no_interdiff \
  --enable-bev-layout \
  --bev-row-pool 16 \
  --enable-phase-sketch \
  --phase-sketch-only \
  --phase-range-freqs 4 \
  --phase-bev-freqs 8 \
  --n-coarse 800 \
  --ctx-weights 0 0.0625 0.125 0.25 0.5 1.0 \
  --phase-sketch-bev-weights 0.5 1.0 2.0 4.0 8.0 \
  --phase-sketch-range-weights 0.0 0.125 0.25 0.5 1.0 \
  --output results/checkpoint_eval_nointerdiff288_gate00625_phase384_n800_seedX.json
```

## Immediate Next Experiments

1. Run at least 3 seeds for `no_interdiff + gated context`.
2. Try `gate_initial_alpha=0.03` and `0.125`; current best post-hoc context weight often lands below or around `0.5` after learned gating.
3. Add phase-aware edge features next:
   - `phase_consistency`
   - `sin(best_shift)`
   - `cos(best_shift)`
   - `phase_residual`
4. Do not put `5.7K layout rerank` in the main method table.
5. If using phase sketch in the main method, report total descriptor state:
   - `288D raw + 128D gated context + 384D phase = 800D`
   - `288D raw + 128D gated context + 512D phase = 928D`
   - SC++ reference state is about `20D ring key + 1200D layout = 1220D`.

## Validation Commands

Local syntax:

```bash
python3 -m py_compile \
  src/encoding/cross_spectrum.py \
  src/encoding/spectral_encoder.py \
  src/gnn/model.py \
  train_multi_dataset.py \
  scripts/evaluate_kitti_checkpoint.py \
  scripts/run_kitti_operating_point.py
```

Local tests:

```bash
PYTHONPATH=src pytest -q tests/test_cross_spectrum.py tests/test_gnn_gate.py
```

`tests/test_gnn_gate.py` skips if `torch_geometric` is unavailable.

## Phase-aware GAT upgrades (NeurIPS round)

Two new opt-in modules layered on the existing magnitude GAT. Both consume
`graph.x_phase` (already produced by `phase_features_from_layouts`) and are
flag-gated, so existing checkpoints continue to load and run unchanged.

### 1. Closed-form phase-coherence edge bias

`src/encoding/phase_coherence.py::ClosedFormPhaseEdgeBias` —
parameter-free, math-guaranteed yaw-invariant bias added to attention logits
inside `DiffAttnConv`. Score: peak of phase-only correlation (PoC, IFFT of
unit-magnitude cross spectrum). Stacks additively with the learned
`PhaseEdgeBias`.

YAML knob:

```yaml
gnn:
  phase_coherence:
    enabled: true
    n_rows: 16          # rows in the underlying phase sketch
    n_freqs: 8          # per-row non-DC frequency budget
    scale: 2.0
    mode: poc           # 'poc' (default, sharpest) or 'ncc'
    pad_factor: 4       # IFFT zero-pad for peak resolution
    similarity_only: true
    center: true
```

Disabled by default; setting `enabled: true` adds a single deterministic
edge logit per similarity edge with no extra parameters.

### 2. Dual-stream magnitude + phase GNN

`src/gnn/model.py::DualStreamSpectralGNN` — a wrapper around the existing
`SpectralGNN` (magnitude stream) plus a sibling `PhaseStreamGNN` operating
on yaw-invariant features derived from `data.x_phase`:

* `log(1 + |z|^2)` per-node
* optional bispectrum coefficients `B[r;k1,k2] = z[r,k1] z[r,k2] z̄[r,k1+k2]`

Both helpers in `src/gnn/phase_diff_conv.py`. Per-node features are
exactly yaw-invariant (verified: max deviation ≤ 1e-5 under random
per-node yaw shifts), so the two `DiffAttnConv` layers in the phase
stream produce a yaw-invariant context vector. Final retrieval key is
`cat(raw_mag, ctx_fused)` where `ctx_fused = (1 − α) · ctx_mag + α ·
ctx_phase` and `α` is a per-node sigmoid gate initialised at
`fuse_initial_alpha` (0.1 by default).

YAML knob:

```yaml
gnn:
  dual_stream:
    enabled: true
    n_rows: 16
    n_freqs: 8
    use_bispectrum: true        # adds bispectral coefficients to phase features
    hidden_dim: 128
    context_dim: 128            # must equal gnn.context_dim for fusion
    n_layers: 2
    n_heads: 4
    fuse_initial_alpha: 0.1     # initial phase contribution; sigmoid(bias) at init
    fuse_per_node: true         # per-node gate (vs. one scalar α)
    fuse_hidden_dim: 64
```

When enabled, `create_spectral_gnn` wraps the existing magnitude
`SpectralGNN` so older checkpoints can be loaded with `strict=False`
(only the new phase-stream + fuse-gate weights need fresh training).

### Recommended ablation order

1. Train baseline (existing 288D + 128D gated, no phase upgrades).
2. Add `phase_coherence.enabled: true` only — should improve KITTI 08
   without retraining the magnitude weights (closed-form, parameter-free).
3. Add `dual_stream.enabled: true` and retrain — expected to subsume the
   handcrafted phase-rerank step into a single learned 416D key.

## 2026-05-09 Update: GAT phase experiments

The latest KITTI experiments show that invariant phase summaries are not
enough to replace pairwise phase-sketch alignment.

| Variant | Stored state | KITTI 00 | KITTI 05 | KITTI 08 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| compact raw + gated context | 416D | 0.9209 | 0.8196 | 0.3957 | no phase |
| power dual-stream GAT | 416D | 0.9304 | 0.8302 | 0.3872 | no 08 gain |
| bispectrum dual-stream GAT | 416D | 0.9272 | 0.8223 | 0.3915 | no 08 gain |
| BEV phase-sketch rerank | 800D | 0.9794 | 0.9523 | 0.8468 | closed-form pairwise alignment |

Key conclusion: the strong signal is not generic "phase identity". It is the
candidate-specific yaw/column-shift alignment curve. `|z|^2` and bispectrum are
yaw-invariant, but they do not perform pairwise shift matching, so a single-pass
GAT key remains far below the closed-form phase-sketch reranker on KITTI 08.

The reranker uses the closed-form phase peak as the score (Eq. 6):

```text
score(q,c) = w_phase * max_s corr_phase(q,c,s)
           + w_emb   * cosine(emb_q, emb_c)
```

Cross/magcross note: keep `cross_spectrum` and `bev_cross` paths as ablation
code only. The main config already uses `cross_spectrum.enabled: false` and
`phase_features.source: bev_complex`. NCLT phase-token evidence favors no cross
(`phase64token` R@1 0.5046 vs. `phase64token_magcross` R@1 0.4985), so cross is
not part of the headline method.

## GAT Sensor-Bandwidth Upgrade

The GAT should be framed as a sensor-bandwidth compensator, not as a phase
learner. The code now has three opt-in upgrades for this narrative:

1. `gnn.sensor_gate`: residual context gate conditioned on per-node
   `sensor_id` plus continuous `beam_count`. Missing metadata falls back to an
   unknown sensor token and 64 beams.
2. `keyframe.graph.sensor_similarity`: dataset-specific similarity graph
   density. This can keep KITTI conservative while giving sparse sensors more
   candidate edges.
3. `gnn.diffattn_value_source: abs_diff`: keeps diff-based attention scores but
   adds an absolute-neighbor value branch `W_abs h_j` to prevent pure
   `(h_j-h_i)` messages from losing keyframe identity.

Suggested ablation config:

```yaml
gnn:
  use_residual_gate: true
  gate_initial_alpha: 0.0625
  sensor_gate:
    enabled: true
    num_sensors: 4       # kitti, nclt, helipr, mulran
    sensor_embed_dim: 8
    use_beam_count: true
    beam_embed_dim: 4
    default_beam_count: 64
  diffattn_value_source: abs_diff

keyframe:
  graph:
    similarity_max_k: 10
    similarity_min_k: 0
    sensor_similarity:
      enabled: true
      kitti:  {max_k: 10, min_k: 0}
      nclt:   {max_k: 16, min_k: 4}
      helipr: {max_k: 24, min_k: 6}
      mulran: {max_k: 12, min_k: 2}
```

Backward compatibility: all three changes are disabled by default. Old configs
and old checkpoints still load on the previous path (`diff` value branch,
scalar graph k, no sensor gate). New sensor-aware checkpoints must be evaluated
with scripts that attach `sensor_id`/`beam_count`; the KITTI/NCLT eval scripts
now do this.

## Physics-Aware 384D Phase Encoding

To absorb some of the multi-sensor fine-tuning benefit without a large learned
encoder, BEV projection now supports `height_encoding: physics3`. It stacks
three deterministic LiDAR physical channels:

1. normalized max height;
2. polar-cell-area-normalized log occupancy density;
3. normalized vertical span.

The channels are row-stacked, so no downstream phase-sketch code changes are
needed. To keep the same 384D phase budget as BEV-only 384:

```text
old BEV-only:   16 rows * 12 freqs * 2 = 384D
physics3 BEV:   48 rows *  4 freqs * 2 = 384D
```

Recommended quick ablation:

```bash
python3 scripts/evaluate_kitti_checkpoint.py \
  ... \
  --enable-bev-layout \
  --enable-phase-sketch --phase-sketch-only \
  --bev-height-encoding physics3 \
  --bev-row-pool 48 \
  --phase-bev-freqs 4 \
  --phase-range-freqs 0 \
  --phase-sketch-bev-weights 0.5 1.0 2.0 4.0 8.0 \
  --phase-sketch-range-weights 0.0
```

For phase-sketch eval, use the same `--bev-height-encoding
physics3 --bev-row-pool 48 --bev-freqs 4`. This preserves the 800D stored state
(`416D retrieval key + 384D physics-aware phase sketch`) while injecting
domain knowledge that should matter most for sparse sensors.

Implementation note: `physics3` is row-stacked, so both BEV interpolation and
row pooling are channel-aware. Empty-ring interpolation and pooling must never
cross height/density/span channel boundaries; the scripts pass `n_channels=3`
automatically when `height_encoding=physics3`.

### 2026-05-10 status: physics3 debug and main-path decision

The first `sensor_gate_reg_kitti_nclt_seed1` summary under-reported the
physics3 phase-sketch result because `summarize_retrain_combine_eval.py`
recursively selected the first nested `R@1` metric, which was the baseline
`raw`/`final` retrieval value rather than the nested
`final_phase_sketch.phase_sketch_fusion_*` rows. The summarizer now selects the
best phase-sketch fusion row per sequence and records the selected key.

Corrected sensor-gate-reg physics3 sketch result:

```text
KITTI 00: 0.9763  (best: bev1)
KITTI 05: 0.9655  (best: bev4)
KITTI 08: 0.7745  (best: bev2)
```

Fixed-alpha control using the original
`train_no_interdiff_288_gate00625_seed1/best_model.pth`:

```text
max-BEV 384D sketch:      00 0.9794 / 05 0.9523 / 08 0.8468
physics3 384D sketch:     00 0.9794 / 05 0.9655 / 08 0.7787
```

Conclusion: physics3 is correctly wired, but the 3-channel/4-frequency
allocation hurts KITTI 08 reverse-loop discrimination. It improves KITTI 05
and preserves KITTI 00, but the 08 drop is too large. Report physics3 as an
ablation/future-work direction for multi-sensor physical encoding, not as a
reported 800D main row.

NCLT fixed-alpha held-out control with the KITTI+NCLT checkpoint
(`train_kitti_nclt_nointerdiff288_gate00625_seed1/best_model.pth`):

```text
max-BEV 384D:
  2012-01-08 0.2805 / 2013-01-10 0.3631 / macro 0.3218

physics3 384D:
  2012-01-08 0.4499 / 2013-01-10 0.4738 / macro 0.4619
```

Updated interpretation: physics3 is not globally bad. It is a
sensor-adaptation win on HDL-32/NCLT under the same 384D budget (+14.0 %p
macro over max-BEV in the held-out control), but it loses too much
high-frequency azimuthal evidence for KITTI 08 reverse loops. The four-sensor
NSD row keeps cylindrical+BEV. Physics3 should be reported as a
physics-grounded sparse-sensor ablation and follow-up direction, with a larger
budget (e.g. 48 rows * 8 freqs * 2 = 768D) or sensor-conditional phase budget
as the next experiment.

Sensor-aware GAT result:

```text
GAT-only:          00 0.9209 / 05 0.8223 / 08 0.3957
physics3 sketch:   00 0.9763 / 05 0.9655 / 08 0.7745
NCLT zero-shot:    12-01 0.3376 / 13-01 0.4062
```

The regularized sensor-aware gate stabilized alpha separation but did not
improve retrieval. Treat this as a Section 6 ablation/negative result:
regularization can prevent gate collapse, but the shared GAT operator plus
balanced downsampling did not yield a stronger descriptor. The reported main
path should remain fixed-alpha GAT with the cylindrical+BEV 384D phase sketch
scored by the closed-form reranker.
