# PAPER MAIN (FROZEN): 800D + fixed re-metrization + learned edge selection (B3)

Date frozen: 2026-07-09 · Branch: `feat/remetrize-twopass` · Decision: user-approved paper-main.
This file is the canonical spec for the AAAI-27 paper main configuration. It supersedes the
Table-1 v1.5 main (which becomes the "base NSD" ablation row). Written so a fresh session can
write the paper from this file + `docs/analysis/ab3_changes_report_ko.tex` alone.

## 1. Configuration summary (what the paper reports as "NSD")

Stored state per keyframe: **800D, unchanged** = 288D magnitude key `d` + 128D context `c` + 384D phase sketch `p`.
Retrieval key: 416D. Everything below is inference-time except the 20 KB edge classifier.

Pipeline (four stages, all uniform across KITTI/NCLT/HeLiPR/MulRan):

1. **Key re-metrization (A)** — fixed diagonal whitening of the key block:
   `f_i = [ normalize(r ⊙ d_i) ; g_i · normalize(c_i) ]`,
   `r_k = (Var_within[k] + eps)^(-1/2)` fit once on TRAIN revisit pairs (<5 m, >=30-frame gap, ~38k pairs).
   One global 288-vector: `artifacts/key_remetrize_r.npy`. No training; yaw invariance preserved
   (diagonal reweight of an invariant statistic). GNN input stays raw `d`.
2. **Pass-1 forward** — frozen 800D checkpoint on the v1.5 paper graph (raw-d similarity edges,
   threshold 0.993) -> embeddings `f^(1)`.
3. **Learned edge selection (B3)** — candidates `C_i = top-30 by <f^(1)_i, f^(1)_j>` excluding
   |i-j|<30; 14-feature vector per candidate (three cosines f/d~/c, rank, reciprocal rank,
   query margin, two densities, cylindrical-sketch cyclic-shift score/sin/cos/peak-margin/entropy,
   index gap); MLP classifier `h` (14-64-64-1, 5,185 params, `artifacts/edge_classifier.pt`)
   scores each; keep top-10 per node as similarity edges (attr `[0,0,cos,log1p(l2)/5,prob]`);
   pass-2 forward on temporal + selected edges -> final 416D key.
4. **Phase rerank** — closed-form cyclic-shift cosine on the 384D sketch (cyl 4 + BEV 8 freqs,
   16x60 layouts), top-800 pool, `sketch_fft`, **raw-scale fusion** (`fusion_norm="raw"`:
   `delta_emb + w_b*delta_bev + w_r*delta_rng`, w grid includes 0). The legacy per-query minmax
   fusion is deprecated (it amplified flat sketch channels into noise on NCLT/HeLiPR).

Classifier training (the only learned addition): BCE on GT-pose labels (pos <5 m, neg >=25 m,
5-25 m band dropped) over 23 TRAIN sequences (2.52M candidate edges), holdout AUC 0.862.
No validation-sequence supervision anywhere. Rebuild: `scripts/build_and_train_edge_classifier.py`.

## 2. Frozen numbers (3-seed, 9 validation sequences, full pipeline)

Aggregates (query-weighted / sequence-balanced / population-sigma over 4 sensor macros / worst sensor):

| Config | R^q | R^s | sigma_cross | R_min |
| --- | --- | --- | --- | --- |
| base NSD (Table-1 v1.5 main) | 0.765 | 0.737 | 0.243 | 0.398 |
| + A (re-metrization) | 0.773 | 0.744 | 0.233 | 0.426 |
| + B2 + raw fusion (ablation step) | 0.784 | 0.758 | 0.216 | 0.469 |
| **+ B3 (paper main)** | **0.7956** [0.794, 0.798] | **0.7674** | **0.2005** | **0.4672** |
| SC++ / BEVPlace++ ms-ft | 0.760 / 0.859 | 0.707 / 0.872 | 0.286 / 0.082 | 0.218 / 0.742 |
| GT-pose-edge oracle (coarse, seed1) | 0.958 | — | — | — |

Per-sequence 3-seed means (paper main):

| 00 | 05 | 08 | 12-01 | 13-01 | Town01 | DCC03 | KAIST03 | Riv03 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.981 | 0.959 | 0.843 | 0.542 | 0.392 | 0.550 | 0.774 | 0.998 | 0.867 |

Coarse (pre-rerank) 3-seed for the ablation ladder: base 0.729 -> A 0.742 -> B2 0.749 -> B3 0.774.
Source JSONs: `results/_remetrize_twopass/` (`{seedN,rerank_raw_seedN,b3_seedN,b3_rerank_seedN}.json`,
oracle_seed1.json, native_*.json). Seeds = the three 800D checkpoints in `checkpoints/800d_4sensor_*`.

## 3. Protocol disclosures the paper MUST state

- **Per-sequence key selection**: reported numbers take, per sequence, the best of
  {coarse, (w_b,w_r) grid} — this matches the v1.5 release summarizer that produced Table 1,
  so all rows are protocol-consistent. Under one global weight pair the base NSD drops to 0.739
  (B3 to ~0.757); if reviewers demand a single fixed weight, per-sensor calibration on the train
  split is the defensible middle ground.
- **sigma_cross** uses the population (1/S, S=4) definition everywhere (v1.5 bug where the main
  row used 1/(S-1) was fixed; recompute scripts must not use torch.std default unbiased=True).
- Eval protocol: revisit query = prior keyframe <5 m and >=30 frames older; R@1 via cosine;
  9 val sequences, n = 632/377/235/1834/181/1586/2344/2909/2796.
- Storage stays 800D: all classifier features derive from the 416D key + the cylindrical 128D
  half of the stored sketch. Added model constants: r (288 floats) + classifier (~20 KB).
- Extra inference cost: one additional GNN forward (two-pass) + 30 edge scorings per node at
  indexing time (a few ms per keyframe; update app:latency accordingly).

## 4. Negative results ledger (for honest ablations / rebuttals)

| Attempt | Outcome |
| --- | --- |
| B1: similarity edges from re-metrized d cosine | KITTI08 -18 pp coarse; edge-attr distribution shift |
| B0: top-10 raw-d edges (density control) | worse than A; proves gain is edge *quality*, not density |
| Learned diagonal r (trained head, step-2) | below closed-form diag_wccn; r stays closed-form |
| Joint EdgeConfidenceGate + aux BCE + two-pass training | diverges (val 0.672->0.51); two real bugs fixed en route (AMP overflow in gate path; stale edge_pose_label after edge rebuild) |
| Retrain with frozen r + train-time two-pass (stage-5) | +0.7 pp under its native Bayesian-refined edges (3/3 seeds) but does NOT stack with B3 (0.791 vs 0.796); keep as two-pass-training ablation only |
| minmax fusion (v1.5 implementation) | harms NCLT/HeLiPR; replaced by raw-scale (Eq. 6 semantics) |

## 5. Reproduction commands (container `nvcr.io/nvidia/pyg:26.01-py3`, repo at /workspace)

```bash
# (once) refit / verify the re-metrization vector (host numpy is enough)
python scripts/fit_key_remetrize.py            # verify mode; --write to regenerate
# (once) rebuild classifier from train sequences
python scripts/build_and_train_edge_classifier.py
# paper-main eval: coarse + rerank grid, per seed
python scripts/run_b3_rerank.py seed1 seed2 seed3
# ablation ladder (P/A/B0/B1/B2 coarse + rerank)
python scripts/run_remetrize_twopass_eval.py seed1 seed2 seed3
python scripts/run_remetrize_twopass_rerank.py raw seed1 seed2 seed3
python scripts/summarize_remetrize_twopass.py raw            # + "global" for fixed-weight table
# supporting analyses
python scripts/run_oracle_ceiling_eval.py seed1              # graph-topology ceiling
python scripts/run_native_refine_eval.py seed1 s5 s5_2 s5_3  # two-pass-training ablation
```
Training-side runs (stage-5 etc.) additionally require the mandatory CLI overrides:
`--encoder-preset no_interdiff --use-gated-context --gate-initial-alpha 0.0625`.

## 6. Paper integration checklist (from docs/analysis/ab3_changes_report_ko.tex)

1. New subsection after §3.3 "learned edge selection": the C_i / phi / h / E''_sim equations
   (LaTeX already drafted in the report tex; keep paper notation).
2. Eq. 10 gains r with its one-line closed-form definition + invariance sentence.
3. Eq. 6 implementation wording corrected to raw-scale fusion.
4. Table 1: replace the NSD main row numbers with §2 above (Dim stays 800); move the v1.5 main
   to the ablation ladder (appendix table with the coarse ladder + oracle row).
5. Declare h as a learned component (no external pretraining -> pretraining-free narrative holds);
   update limitations (NCLT 13-01 seed fragility) and app:latency (+1 forward, +edge scoring).
Paper-writing pitfalls: submission build must stay 9 pages; wrap new text in \edited{} per the
working-build convention; smoke check greps ban stale phrases (see repo CLAUDE.md); the paper tex
lives in docs/paper/body_v1.5/aaai2027_body.tex (gitignored).
