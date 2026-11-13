#!/usr/bin/env python3
"""공식 KittiPointCloudSelect.cpp(d94040e)에 고정 두 줄 회귀 수정을 적용한 사본을 만든다 (2026-11-08).

  python3 scripts/fresco_official/cpp/kitti_groundfix.py <공식 KittiPointCloudSelect.cpp> <출력 .cpp>
공식 원본은 읽기만 한다. 원본·결과 해시가 고정값과 다르면 아무것도 쓰지 않고 멈춘다.

원인: Point-Cloud-Preprocessing-Tools 1974440("removed usage of non-trivial designated initers", 컴파일러 호환
정리)이 resize(N, PointXYZIRCT{.intensity = -1})를 resize(N)과 실제 점마다 intensity = -1로 바꿨다
(d94040e KittiPointCloudSelect.cpp:207,238). BatchMultiBevGen.cpp:146-166 markGroundPoints는 intensity == -1을
"측정 없음"으로 보므로 KITTI 계열에서는 지면 제거가 사실상 꺼진다. 수정은 1974440 이전 의미 두 가지뿐이다.
  (a) structured_cloud->points.resize(N) 뒤 모든 구조 칸 intensity = -1 (빈 칸 표시)
  (b) 실제 점의 intensity = -1 대입 삭제
결과 SHA256은 공통46 KITTI 지면 수정 재평가 규약(fresco_common46_kitti_groundfix_v1)의 패치 원천과 같다.
다른 TU·센서·MATLAB·매칭·상수는 바꾸지 않는다. codec.py의 KITTI intensity 0 대체는 이 수정 뒤에도 무손실이다.
intensity는 == -1 비교에만 읽히고 KITTI·KITTI-360 원시 intensity는 [0, 1]이기 때문이다.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BASE_COMMIT = "d94040ed626c7275b6f428deddde2ddca74419d7"
REGRESSION_COMMIT = "1974440"
BASE_SHA256 = "98bc2d6593dea6cedbed29025eeeac1c3bf4d204fd88a8ee035c548742dca67f"
PATCHED_SHA256 = "5054c7c642c44c91dd1c63b8484589252c175c1632d1fb86262478f0bcd442c3"
OLD_LINE = "    structured_cloud->points.resize(N_SCAN * Horizon_SCAN);\n"
NEW_LINE = ("    structured_cloud->points.resize(N_SCAN * Horizon_SCAN);\n"
            "    for (auto &fresco_pt_p : structured_cloud->points) { fresco_pt_p.intensity = -1; }"
            "  // pre-1974440: resize(N, PointXYZIRCT{.intensity = -1})\n")
DEL_LINE = "            cloud->points[i].intensity = -1;\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch(src: bytes) -> bytes:
    """공식 원본 바이트 -> 수정 사본 바이트. 두 줄이 정확히 한 번씩 있어야 한다."""
    assert sha256(src) == BASE_SHA256, "official KittiPointCloudSelect.cpp differs from d94040e"
    text = src.decode()
    assert text.count(OLD_LINE) == 1 and text.count(DEL_LINE) == 1
    out = text.replace(OLD_LINE, NEW_LINE).replace(DEL_LINE, "").encode()
    assert sha256(out) == PATCHED_SHA256, "patched source hash differs from the fixed correction"
    return out


def main() -> None:
    src, dst = map(Path, sys.argv[1:3])
    out = patch(src.read_bytes())
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(out)
    rec = {"base_commit": BASE_COMMIT, "regression_commit": REGRESSION_COMMIT,
           "base_sha256": BASE_SHA256, "patched_sha256": sha256(dst.read_bytes()),
           "patch_tool_sha256": sha256(Path(__file__).read_bytes()),
           "patch": "pre-1974440 empty-cell semantics: structured cells intensity=-1; real points keep intensity"}
    (dst.parent / "patch.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
