"""SC++ KITTI00 재방문 정의 대조(정답 계수만, 서술자·점수 미사용).

공식 코드의 두 정의를 사전 고정 값으로 센다.
- D_radar_past: fast_evaluator_radar/main.m:34,80,83 (과거 1:end-49, 거리 < 기준)
- D_tro_bidir: tro2022/misc/is_revisit.m:8,16 (abs(q-i) < N 건너뜀, 과거·미래 전체)
N에 따른 곡선은 서술용이며 조건 선택에 쓰지 않는다.

python3 scripts/_baseline_closure_scpp_revisits.py POSES LEDGER_CSV OUT_JSON
"""
import hashlib
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

RADIUS_M = 8.0
PAPER_REVISITS = 852
PAPER_NONSAME = 22


def sha256(path: str) -> str:
    """파일 SHA-256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pair_lists(xy: np.ndarray) -> list:
    """프레임마다 기준 거리 < RADIUS_M 인 다른 프레임 번호 목록."""
    tree = cKDTree(xy)
    out = []
    for i, nb in enumerate(tree.query_ball_point(xy, RADIUS_M + 1e-6)):
        nb = np.asarray([j for j in nb if j != i], dtype=np.int64)
        d = np.linalg.norm(xy[nb] - xy[i], axis=1) if len(nb) else np.zeros(0)
        out.append(nb[d < RADIUS_M])
    return out


def count_past(nbrs: list, n: int, gap: int) -> int:
    """과거 노드 0..q-gap(0-기준) 안에 8 m 미만이 있는 질의 수. 질의는 q >= gap."""
    return int(sum(1 for q in range(gap, n) if np.any(nbrs[q] <= q - gap)))


def count_bidir(nbrs: list, n: int, apart: int) -> int:
    """is_revisit.m: |q-i| >= apart 인 모든 노드 중 8 m 미만이 있는 프레임 수."""
    return int(sum(1 for q in range(n) if np.any(np.abs(nbrs[q] - q) >= apart)))


def main() -> None:
    poses_path, ledger_path, out_path = sys.argv[1:4]
    poses = np.loadtxt(poses_path)
    xy = poses[:, [3, 11]]  # 카메라 좌표 x,z (SingleTestInfo.m:85-86)
    n = len(xy)
    nbrs = pair_lists(xy)
    ledger = np.loadtxt(ledger_path, delimiter=",")
    ledger_pos = int(np.sum(ledger[:, 4] < RADIUS_M))
    pre = {
        "D_radar_past_gap49": count_past(nbrs, n, 49),
        "D_tro_bidir_N50": count_bidir(nbrs, n, 50),
    }
    curve_past = {g: count_past(nbrs, n, g) for g in
                  [1, 25, 49, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000]}
    curve_bidir = {a: count_bidir(nbrs, n, a) for a in
                   [1, 25, 50, 100, 150, 200, 300, 500, 1000, 1500, 2000]}
    exact_past = [g for g in range(1, 2001) if count_past(nbrs, n, g) == PAPER_REVISITS]
    exact_bidir = [a for a in range(1, 3001) if count_bidir(nbrs, n, a) == PAPER_REVISITS]
    report = {
        "inputs": {"poses": poses_path, "poses_sha256": sha256(poses_path), "frames": n,
                   "ledger": ledger_path, "ledger_sha256": sha256(ledger_path)},
        "fixed": {"radius_m": RADIUS_M, "compare": "<", "axes": "camera x,z"},
        "sanity_existing_ledger_positive": ledger_pos,
        "sanity_matches_recount": ledger_pos == pre["D_radar_past_gap49"],
        "paper_table_iii": {"revisits": PAPER_REVISITS, "total": 4541,
                            "non_same_direction": PAPER_NONSAME},
        "preregistered_counts": pre,
        "preregistered_hit_852": {k: v == PAPER_REVISITS for k, v in pre.items()},
        "descriptive_curve_past_gap": curve_past,
        "descriptive_curve_bidir_apart": curve_bidir,
        "descriptive_exact_852_past_gaps": exact_past,
        "descriptive_exact_852_bidir_apart": exact_bidir,
        "note": "서술 곡선과 정확 일치 목록은 조건 선택에 쓰지 않는다(protocol.json C1)",
    }
    with open(out_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: report[k] for k in
                      ["sanity_existing_ledger_positive", "sanity_matches_recount",
                       "preregistered_counts", "preregistered_hit_852",
                       "descriptive_curve_past_gap", "descriptive_curve_bidir_apart"]},
                     ensure_ascii=False))
    print("exact852 past", exact_past[:20], len(exact_past))
    print("exact852 bidir", exact_bidir[:20], len(exact_bidir))


if __name__ == "__main__":
    main()
