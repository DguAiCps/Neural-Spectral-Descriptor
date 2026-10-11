# BASELINE RETRAIN PLAN — post NCLT 8-byte parser fix (2026-07-28)

Context: `src/data/nclt_loader.py` was fixed from a wrong 12-byte record parser to the
official 8-byte format (x,y,z `<u2` ×0.005−100, intensity `u1`, laser `u1`; then z=−z for
z-up). NCLT **val** keyframe caches (`data/preprocessed/cache_3a73ece2_nclt_val_*` =
`557cafe4`/`056e0a02` symlinks) were regenerated with the fixed loader (verified: scan_ids
and descriptors differ from the `.oldz` backups). Corrected NCLT operating caches live in
`data/preprocessed_nclt_fix8/` (do not touch).

Eval-only passes of the existing (contaminated-trained) checkpoints on corrected NCLT
clouds were run on 2026-07-28 (logs `~/baseline_fix8_*.log`, results
`outputs/*_nclt_fix8.csv`, `results/{seqot,pgat}_eval_nclt_fix8.json`). Those are
lower-bound references; the retrains below give the fair numbers.

All commands: host, `cd /rise/RISE1/workspace/impl/NSD-paper`,
`PY=/rise/RISE1/miniconda3/envs/py310/bin/python` (torch 2.6.0+cu124, torch_radon and
torch_geometric both import OK on host). Use **GPU1 only** (`CUDA_VISIBLE_DEVICES=1`);
GPU0/2 run other groups' jobs.

## 0. Common prerequisite — NCLT TRAIN keyframe caches (REQUIRED for every retrain below)

The 5 NCLT train sequences (2012-05-11, 2012-08-04, 2012-11-04, 2012-11-16, 2013-02-23)
have **no** keyframe caches on this server (`data/preprocessed_train/` has none for nclt);
any that are synced from elsewhere predate the parser fix. Regenerate with the canonical
pipeline (same encoder/selector/cache-key as training):

    $PY gen_val_caches_canonical.py --config configs/eval_host_paths.yaml   # val pattern
    # train sequences: run the same driver against the train split (see gen_train_caches.log
    # for the exact invocation used on 2026-07-27, which wrote data/preprocessed_train/
    # cache_de201724_* + 557cafe4 aliases) — point it at data.datasets.train nclt entries.

Estimated: ~28k scans/date × 5 dates, encode ≈ 10 ms/scan → **2–4 h** (GPU1 or CPU).
SeqOT/P-GAT adapters expect alias keys `056e0a02` (kitti/nclt/helipr), `33919e6e` (mulran),
`76c42cd7` (nclt_2012-11-16, 2013-02-23), `de201724` (kitti_01) — create symlinks like the
existing `data/preprocessed/cache_056e0a02_nclt_val_*` ones if the computed key differs.

## 1. BEVPlace++ (ms-finetuned) — RETRAIN REQUIRED

Current `baselines/weights/bevplace_finetune.pth` was fine-tuned on a 30k multi-sensor
subset whose NCLT quarter was 12-byte-corrupted.

    CUDA_VISIBLE_DEVICES=1 $PY scripts/finetune_bevplace.py \
        --config configs/eval_host_paths.yaml \
        --output baselines/weights/bevplace_finetune_fix8.pth \
        --epochs 5 --subset-per-sensor 7500 > ~/bevplace_ft_fix8_train.log 2>&1

Estimated runtime: BEV precompute over ~30k clouds + 5 epochs ≈ **4–6 h** on GPU1.
Then eval (same harness as the 07-28 eval-only pass):

    NSD_BEVPLACE_WEIGHTS=$PWD/baselines/weights/bevplace_finetune_fix8.pth \
    CUDA_VISIBLE_DEVICES=1 $PY baselines/evaluate_baselines.py --methods bevplace \
        --config configs/eval_host_paths.yaml \
        --dataset-filter NCLT_2012-01-08 NCLT_2013-01-10 \
        --output outputs/bevplace_ft_fix8_retrained.csv          # ~15 min

Note: the eval-only pass of the contaminated-trained ckpt on corrected clouds already
scores R@1 0.897/0.906 (N12/N13), so the retrain is expected to move numbers only mildly.

## 2. SeqOT — RETRAIN REQUIRED (both stages)

Checkpoints `_external/seqot_runs/full/{seqot,gem}_latest.pth.tar` were trained on range
images projected from corrupted NCLT clouds (5/24 train sequences affected). Official code
now restored at `_external/SeqOT` (clone of github.com/BIT-MJY/SeqOT, 2026-07-28).

    # (a) regenerate range images for the 5 NCLT train sequences (after step 0):
    $PY scripts/_seqot_prepare.py nclt/2012-05-11 nclt/2012-08-04 nclt/2012-11-04 \
        nclt/2012-11-16 nclt/2013-02-23 --config configs/eval_host_paths.yaml   # ~1 h CPU
    # (non-NCLT train sequences also need preparing once on this server — the old
    # _external/seqot_data was not synced here; add the kitti/helipr/mulran train seqs.)

    # (b) two-stage training (docstring budget: <= 12 GPU-h; H200 likely ~6-8 h):
    CUDA_VISIBLE_DEVICES=1 $PY scripts/_seqot_train.py \
        --run-dir _external/seqot_runs/fix8 --stage both > ~/seqot_fix8_train.log 2>&1

    # (c) eval (protocol identical to 07-28 eval-only pass):
    CUDA_VISIBLE_DEVICES=1 $PY scripts/_seqot_eval.py \
        --sequences nclt/2012-01-08 nclt/2013-01-10 --config configs/eval_host_paths.yaml \
        --seqot-checkpoint _external/seqot_runs/fix8/seqot_latest.pth.tar \
        --gem-checkpoint _external/seqot_runs/fix8/gem_latest.pth.tar \
        --out results/seqot_eval_nclt_fix8_retrained.json        # ~1 min

Total ≈ **8–13 h** (mostly GPU1).

## 3. P-GAT — RETRAIN REQUIRED (features + model)

P-GAT node features are the 544-D cache descriptors; the NCLT train features and the
checkpoint `_external/pgat_runs/full/attentional_graph.pt` are contamination-derived.
Official code restored at `_external/P-GAT` (github.com/csiro-robotics/P-GAT @ a66a7018 =
the pinned commit).

    # (a) after step 0, rebuild training tensors (CPU, minutes):
    $PY scripts/_pgat_make_tensors.py            # writes _external/pgat_data/
    # (b) train:
    CUDA_VISIBLE_DEVICES=1 $PY scripts/_pgat_train.py \
        --config_file configs/_pgat_train.yml > ~/pgat_fix8_train.log 2>&1   # est. 6-12 h
    # (c) eval:
    CUDA_VISIBLE_DEVICES=1 $PY scripts/_pgat_eval.py \
        --checkpoint _external/pgat_runs/fix8/attentional_graph.pt \
        --sequences nclt_2012-01-08,nclt_2013-01-10 \
        --out results/pgat_eval_nclt_fix8_retrained.json         # ~2 min

Note: 07-28 eval-only pass on corrected NCLT confirms the known embedding-readout collapse
(R^q@1 = 0.036); retraining is for completeness, the Table-1 qualitative story is unlikely
to change.

## 4. RING#-L (RING# trained on our split, official recipe) — NOT YET TRAINED (Table 1 row currently "---")

State on this server: NO checkpoint exists; `_external/RINGSharp` contains only
`tools/results` (51 MB) — the glnet source tree, the custom `glnet/datasets/nsc/
nsc_dataset.py` shim (make_nsc_loader, NSC_BOUNDS, NSC_X/Y/Z) and the sm_120 torch-radon
build were on the original training machine and are gitignored, so they were never synced.
`torch_radon` DOES import in the host py310 env, so no docker/rebuild is needed here.

    # (a) restore code (keep tools/results):
    cd _external && git clone https://github.com/lus6-Jenny/RINGSharp RINGSharp_src && \
        rsync -a RINGSharp_src/ RINGSharp/ && cd ..
    # (b) restore or reimplement the nsc shim glnet/datasets/nsc/nsc_dataset.py
    #     (wraps OUR loaders — so the fixed NCLT parser is inherited automatically;
    #      _ringpp_eval.py/_ringsharp_eval.py document its API). Recover from the
    #     student's machine if possible; reimplementation ~0.5 day.
    # (c) after step 0, build training pickles (CPU, ~1-2 h):
    $PY scripts/_ringsharp_prepare.py
    # (d) official RING#-L training on train_nsc.pickle, student config:
    #     grid 160x160, search-space descriptor 160-D, stored ~51.4k floats/frame.
    #     Entry point: RINGSharp repo training script for RING#-L (LiDAR config), e.g.
    #     cd _external/RINGSharp && CUDA_VISIBLE_DEVICES=1 python training/train.py \
    #         --config <ringsharp-L lidar cfg pointing at ../ringsharp_data/train_nsc.pickle>
    #     Estimated: **1-2 GPU-days** on GPU1 (their TRO'25 recipe, 24-seq split).
    # (e) eval on corrected NCLT (torch_radon OK on host):
    CUDA_VISIBLE_DEVICES=1 $PY scripts/_ringsharp_eval.py --weight <model_..._final.pth>
    #     (_ringsharp_eval expects cache key 056e0a02 for nclt — symlinks already created.)

Parser check: RING# input path = our caches (scan_ids/poses) + `make_nsc_loader` → OUR
`src/data/nclt_loader.py` (fixed). No own NCLT binary parser exists in the synced portion
of `_external/RINGSharp` (grepped for `fromfile`/`<u2`: none).

## 5. Analytic baselines — NO RETRAIN

M2DP / FreSCo / LiDAR-Iris / SC++ / RING / RING++ have no trained parameters. Corrected
NCLT re-evals were launched/completed 2026-07-28 (see `~/baseline_fix8_*.log`,
`outputs/*_nclt_fix8.csv`). SC++ fix8 verification runs separately (`~/scpp_fix8.log`,
another lane — do not relaunch).

## Suggested order (GPU1 serial)

1. Step 0 caches (2–4 h) → 2. SeqOT (8–13 h) → 3. BEVPlace++ ft (4–6 h) →
4. P-GAT (6–12 h) → 5. RING#-L (needs shim restore first; 1–2 d training).
Steps 2–4 total ≈ 1.5 GPU-days; RING#-L is the long pole and can run in parallel on GPU1
only after the others finish (do not use GPU2/GPU3).
