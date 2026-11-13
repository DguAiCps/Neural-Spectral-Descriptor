#!/usr/bin/env python3
"""대기한 지역 특징 진단을 회수·검산하고 범위가 명시된 기록을 남긴다."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time

from _review_native_collect import sync, REPO, NATIVE
import _review_native_ring_verify as verify
from _review_native_pdv2_ringpp import write_json


def main() -> None:
    """완료 마커가 있을 때만 전체 수치와 제출 제안을 생성한다."""
    root = REPO / NATIVE / "ring_feature_metric"
    while True:
        try:
            sync(NATIVE + "/ring_feature_metric", ["*.json", "pair_dists_*.npy", "*.log"], optional=True)
            if (root / "comparison.json").exists():
                verify.OUT = root
                verify.main()
                report = json.loads((root / "comparison.json").read_text())
                row = next(r for r in report["rows"] if r["condition"]=="all_ringpp" and r["radius_m"]==10)
                before = 100*row["original_anisotropic"]["official_R@1"]
                after = 100*row["isotropic_feature_metric"]["official_R@1"]
                text = r"\paragraph{Coordinates used for local geometric features.}" + "\n"
                text += "The released NCLT loader divides horizontal coordinates by 70 and the vertical range by 19; its local 30-neighbor search and geometric features use these anisotropically scaled coordinates. As a separate diagnostic, we retain voxel addresses and exact maximum aggregation but use the same length unit for all axes during local feature computation. "
                text += f"Full-pair 10\\,m R@1 changes from {before:.2f}\\% to {after:.2f}\\% ({after-before:+.2f} percentage points). "
                text += "The RING distance matrices remain bitwise unchanged, and independent recomputation verifies all 64 metrics and 1,413 raw-input hashes. The paper does not specify this coordinate convention, so the diagnostic is not claimed as a correction to the published benchmark. It is not applied to the common 46-sequence evaluation.\n"
                (root.parent / "paper_proposal/ring_feature_metric_addendum.tex").write_text(text)
                date = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
                log = REPO / "docs/RAW/실험일지" / (date + "_RING-지역특징-좌표대조.md")
                if not log.exists():
                    log.write_text(f"# RING++ 지역 특징 좌표 대조\n\n## 왜\n\n공식 NCLT 정규화는 XY와 Z에 서로 다른 단위를 쓰며 지역 특징도 이 좌표에서 계산한다. 이 단일요인의 영향을 분리한다. 논문은 좌표 단위를 명시하지 않아 확정 오류나 저자 수치의 정확 재현이라고 부르지 않는다.\n\n## 설정\n\n`{root.relative_to(REPO)}/protocol.json`에서 결과 전에 고정했다. 격자 주소·정확 max·각도·정답·집계는 유지하고 지역 특징의 Z에19/70만 곱한다. 공통46과 HeLiOS 재생성 평가가 완료된 뒤에만 GPU1을 사용했다.\n\n## 결과\n\nFull NCLT10m R@1은{before:.6f}%→{after:.6f}%({after-before:+.6f}%p)다. `{root.relative_to(REPO)}/comparison.json`과 `verification.json`에서 RING 행렬 불변, 원시1,413개 해시, 독립64지표 검산을 확인한다.\n\n## 결론 / 다음\n\n이 조건은 별도 원인 진단이며 공통46·주표의 설정으로 채택하지 않는다. `{NATIVE}/paper_proposal/ring_feature_metric_addendum.tex`에 영문 삽입 제안을 남겼다.\n")
                state = {"complete": True, "pid": os.getpid(), "unix_time": time.time(), "r1_before_percent": before, "r1_after_percent": after, "delta_pp": after-before}
            else:
                state = {"complete": False, "pid": os.getpid(), "unix_time": time.time(), "waiting": "주요 실험 완료 후 단일요인 진단"}
        except Exception as exc:
            state = {"complete": False, "pid": os.getpid(), "unix_time": time.time(), "error": repr(exc)}
        write_json(root / "collection_status.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        if state["complete"]:
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
