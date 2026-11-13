#!/usr/bin/env python3
"""기하만으로 예시를 고른 뒤 현재 순환 정렬 함수의 응답을 저장한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python scripts/_paper_structure_figure_example.py
"""
import sys
from pathlib import Path

import numpy as np

from _paper_structure_figure_review import OUT, REPO, digest, read, write

sys.path.insert(0, str(REPO / "src"))
from _xs_eval_lib import column_align_scores, variant_inputs
from retrieval.spectral_evidence import correlation_profiles


def main() -> None:
    """점수와 독립적인 한 쌍의 원점수·기하·출처를 기록한다."""
    summary = read(OUT / "summary.json")
    source = REPO / "results/pdv2_pair_visualization_20261030/all_pair_scores.npz"
    options = []
    sequences = ("kitti/08", "mulran/DCC01", "nclt/2012-01-08")
    with np.load(source) as data:
        for sid, sequence in enumerate(sequences):
            with np.load(REPO / summary["paths"][sequence]["cache"]) as cache:
                pose = cache["poses_gt"]
            yaw = np.arctan2(pose[:, 1, 0], pose[:, 0, 0])
            indices = np.flatnonzero((data["sequence"] == sid) & ((data["flags"] & 1) != 0))
            for index in indices:
                q, j = int(data["query"][index]), int(data["reference"][index])
                angle = float(np.degrees(abs(np.angle(np.exp(1j * (yaw[q]-yaw[j]))))))
                if angle >= 150:
                    options.append(dict(sequence=sequence, query=q, reference=j,
                                        distance_m=float(data["distance"][index]), heading_deg=angle,
                                        **summary["paths"][sequence]))
    example = min(options, key=lambda r: (r["distance_m"], r["sequence"], r["query"], r["reference"]))
    write(OUT / "example_selection.json", {
        "selection": "기존 균등 표본의 양성 중 GT 방향 차이 150도 이상에서 공간 거리 최소. 점수·성공 여부를 쓰지 않는다.",
        "eligible": len(options), "example": example,
        "pair_source_sha256": digest(source),
    })
    with np.load(REPO / example["spectrum"]) as data:
        rows = [example["query"], example["reference"]]
        values = (data["sk_re"][rows, :, :8].astype(np.float32)
                  + 1j * data["sk_im"][rows, :, :8].astype(np.float32))
    bank = variant_inputs(values, 8)
    profile = correlation_profiles(bank, np.array([0]), np.array([1]), "c_col")[0]
    np.testing.assert_allclose(profile.max(), column_align_scores(bank[3], 0, np.array([1]))[0],
                               rtol=0, atol=1e-12)
    shift = int(np.argmax(profile))
    angle = shift * 6.
    report = dict(**example, channel="c_col", zero_shift_score=float(profile[0]),
                  maximum_score=float(profile.max()), peak_bin=shift, peak_shift_deg=angle,
                  absolute_yaw_error_deg=abs(min(angle, 360-angle)-example["heading_deg"]),
                  auxiliary_magnitude_cosine=float(bank[0][0] @ bank[0][1]),
                  profile=profile.tolist(), angles_deg=(np.arange(60)*6.).tolist(),
                  scope="고정 실제 쌍의 중심 스펙트럼 응답. 최종 시간 판정 점수와 구분한다.",
                  source_sha256={example["spectrum"]: digest(REPO / example["spectrum"]),
                                 "scripts/_xs_eval_lib.py": digest(REPO / "scripts/_xs_eval_lib.py"),
                                 "src/retrieval/spectral_evidence.py": digest(REPO / "src/retrieval/spectral_evidence.py")})
    previous = OUT / "example_profile.json"
    if previous.exists():
        old = read(previous)
        assert old["query"] == report["query"] and old["reference"] == report["reference"]
        np.testing.assert_array_equal(old["profile"], report["profile"])
    write(previous, report)
    print("기하 선택과 정렬 응답 재현 완료", example["sequence"], example["query"], example["reference"])


if __name__ == "__main__":
    main()
