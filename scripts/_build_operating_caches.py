#!/usr/bin/env python3
"""Build evaluator-native operating caches for the sequences that lack them
(KITTI 00/05/08 with the evaluator's own builder; MulRan DCC03/KAIST03/
Riverside03 via a loader-swapped call to the same builder)."""
import sys
from pathlib import Path

REPO = Path('/rise/RISE1/workspace/impl/NSD-paper')
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

import run_kitti_operating_point as op
from data.mulran_loader_old import MulRanLoader

ROOT = Path('/rise/RISE1/workspace/data')
config = op._load_config(REPO / 'configs/training_multi_dataset.yaml')

for seq in ('00', '05', '08'):
    print(f'=== kitti {seq} ===', flush=True)
    p = op._build_sequence_cache(
        root=ROOT / 'kitti/dataset', sequence=seq, config=config,
        cache_dir=REPO / 'data/preprocessed_kitti_operating', device='cuda')
    print(f'built {p}', flush=True)

_orig = op.KITTILoader
op.KITTILoader = lambda root, seq, lazy_load=True: MulRanLoader(
    str(ROOT / 'mulran'), seq, lazy_load=True)
for seq in ('DCC03', 'KAIST03', 'Riverside03'):
    print(f'=== mulran {seq} ===', flush=True)
    p = op._build_sequence_cache(
        root=ROOT / 'mulran', sequence=seq, config=config,
        cache_dir=REPO / 'data/preprocessed_mulran_operating', device='cuda')
    print(f'built {p}', flush=True)
op.KITTILoader = _orig
print('OPERATING_CACHES_DONE', flush=True)
