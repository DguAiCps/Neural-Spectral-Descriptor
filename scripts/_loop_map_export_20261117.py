#!/usr/bin/env python3
"""궤적 루프 폐합 지도용 정답 위치(poses_gt 평행이동) 내보내기. GT는 그림 표시에만 쓴다.
docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_loop_map_export_20261117.py SEQ [SEQ...]"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _pdv2_scratch_search_20261116 as S  # noqa: E402
OUT = ROOT / "results/pdv2_loop_map_20261117/positions"
OUT.mkdir(parents=True, exist_ok=True)
for seq in sys.argv[1:]:
    cache, pos, minimum, *_ = S.P.inputs("test", seq)
    np.savez_compressed(OUT / (seq.replace("/", "_") + ".npz"), pos=pos, minimum=minimum)
    print(seq, pos.shape)
