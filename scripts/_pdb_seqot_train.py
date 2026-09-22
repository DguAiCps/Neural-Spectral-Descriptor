#!/usr/bin/env python3
"""SeqOT pd2026 retrain driver: scripts/_seqot_train.py on the pd TRAIN split.

Thin wrapper that runs the unchanged two-stage batched driver
(scripts/_seqot_train.py: official featureExtracter + GeM, upstream
triplet_loss, pose-distance surrogate index) with pd2026 defaults, plus a
place-disjoint safety guard:

  * --data-root defaults to _external/seqot_data_pd2026 (written by
    scripts/_pdb_seqot_prepare.py, which emits train_index.npy ONLY for the
    16 pd TRAIN sequences);
  * before handing over, every train dir under data-root (any dir containing
    train_index.npy) is checked against the pd TRAIN split of
    configs/training_place_disjoint.yaml — the run aborts if anything else
    (a test or val sequence) would be trained on.

All other flags pass through to _seqot_train.py (upstream-equivalent
defaults: --stage both, --epochs1 15, --epochs2 10, --anchors-per-seq 300,
--anchors-per-step 4, lr 5e-6 — same recipe as the paper-era full run in
_external/seqot_runs/full).

Full pd retrain (inside the ring_baseline container):
  python scripts/_pdb_seqot_train.py --run-dir _external/seqot_runs_pd2026/full
Smoke:
  python scripts/_pdb_seqot_train.py --run-dir _external/seqot_runs_pd2026/smoke \
      --sequences kitti_02 --epochs1 2 --epochs2 1 \
      --anchors-per-seq 40 --anchors-per-step 4
"""
from __future__ import annotations
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

DATA_ROOT_DEFAULT = REPO / "_external" / "seqot_data_pd2026"
PD_CONFIG = REPO / "configs" / "training_place_disjoint.yaml"


def pd_train_dirs():
    cfg = yaml.safe_load(open(PD_CONFIG))
    return {f"{ds['type']}_{seq}"
            for ds in cfg["data"]["datasets"]["train"]
            for seq in ds["sequences"]}


def main():
    if "--data-root" not in sys.argv:
        sys.argv += ["--data-root", str(DATA_ROOT_DEFAULT)]
    data_root = Path(sys.argv[sys.argv.index("--data-root") + 1])

    allowed = pd_train_dirs()
    import _seqot_train
    found = set(_seqot_train.find_train_seqs(data_root)) if data_root.is_dir() \
        else set()
    if "--sequences" in sys.argv:  # explicit list: guard those instead
        i = sys.argv.index("--sequences") + 1
        found = set()
        while i < len(sys.argv) and not sys.argv[i].startswith("--"):
            found.add(sys.argv[i])
            i += 1
    illegal = found - allowed
    assert not illegal, (
        f"PLACE-DISJOINT GUARD: train_index.npy found for non-pd-train "
        f"sequences under {data_root}: {sorted(illegal)} — refusing to train")
    if "--allow-truncated" in sys.argv:  # smoke runs only
        sys.argv.remove("--allow-truncated")
    else:
        import json
        truncated = [s for s in sorted(found)
                     if json.loads((data_root / s / "meta.json").read_text())
                     .get("truncated")]
        assert not truncated, (
            f"TRUNCATION GUARD: {truncated} were prepared with --max-frames "
            f"(smoke); re-run scripts/_pdb_seqot_prepare.py on them without "
            f"--max-frames, or pass --allow-truncated for a smoke train")
    print(f"[pdb guard] {len(found)}/{len(allowed)} pd TRAIN sequences "
          f"prepared under {data_root}; none illegal", flush=True)

    _seqot_train.main()


if __name__ == "__main__":
    main()
