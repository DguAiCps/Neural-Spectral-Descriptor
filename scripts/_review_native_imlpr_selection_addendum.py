#!/usr/bin/env python3
"""검증 선택 ImLPR의 같은 native 입력 대조를 영문 제안과 일지에 기록한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105"


def main() -> None:
    """선택을 바꾸지 않고 독립 검산된 결과의 차이만 보고한다."""
    root = OUT / "imlpr/common_selection_native"
    path = root / "independent_verification.json"
    report = json.loads(path.read_text())
    manifest = json.loads((root / "selection_manifest.json").read_text())
    assert report["complete"] and report["passed"] and len(report["rows"]) == 24
    assert sum(c["original_alias"] is None for c in manifest["candidates"]) == 1
    difference = []
    for k in [15, 25]:
        for sequence in ["DCC01", "DCC02", "DCC03"]:
            ap = report["groups"]["AP"][str(k)][sequence]["r1"]
            r1 = report["groups"]["R@1"][str(k)][sequence]["r1"]
            difference.append({"k": k, "sequence": sequence,
                "AP_selected_mean_percent": 100*ap["mean"], "AP_selected_sample_sd_pp": 100*ap["sample_sd"],
                "R1_selected_mean_percent": 100*r1["mean"], "R1_selected_sample_sd_pp": 100*r1["sample_sd"],
                "difference_pp": 100*(ap["mean"]-r1["mean"])})
    result = {"passed": True, "selection_sha256": manifest["selection_sha256"],
              "verification_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": difference,
              "scope": "보존 best/final 두 후보의 공통검증 선택 민감도; 같은 native 입력·배치1·규약. 시험으로 선택 안 함. 최적수렴 미입증."}
    (root / "selection_sensitivity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    text = "\\paragraph{ImLPR checkpoint-selection sensitivity on unchanged native inputs.}\n"
    text += ("Common-validation AP selects epochs 99/100/74, while common-validation R@1 selects 99/94/74. "
             "Only seed two changes; the other three unique checkpoints reuse hash-verified native ledgers. "
             "The new seed-two checkpoint is evaluated with the same batch-one inference, inputs, labels, and both fixed normal-estimation settings. "
             "The resulting three-seed native R@1 is shown below (mean and sample SD, percent).\n")
    text += "\\begin{center}\\begin{tabular}{llrrr}\nNormal & Query & AP selection & R@1 selection & Difference \\\\\n\\hline\n"
    for row in difference:
        text += (f"$k={row['k']}$ & {row['sequence']} & ${row['AP_selected_mean_percent']:.2f}\\pm{row['AP_selected_sample_sd_pp']:.2f}$ & "
                 f"${row['R1_selected_mean_percent']:.2f}\\pm{row['R1_selected_sample_sd_pp']:.2f}$ & {row['difference_pp']:+.3f} \\\\\n")
    text += "\\end{tabular}\\end{center}\nThis comparison covers only the retained best/final checkpoints. It does not recover missing intermediate epochs or establish optimal convergence, and native test results do not select the checkpoint.\n"
    (OUT / "paper_proposal/imlpr_selection_addendum.tex").write_text(text)
    journal = REPO / "docs/RAW/실험일지/2026-11-06_ImLPR-검증선택-native-대조.md"
    content = "# ImLPR 검증 선택 native 대조\n\n## 왜\n\n공통 검증의 AP/R@1 선택 차이가 원래 native 저자/v2 차이를 얼마나 바꾸는지 같은 입력에서 확인한다.\n\n## 설정\n\n검증54원장 완료 뒤 선택을 동결했다. AP99/100/74, R@1 99/94/74이며 원래 선택은 R@1과 같다. 실제 파일·텐서 SHA를 확인해 4개 고유 모델 중 원래3개의 native원장은 재사용하고 시드2 epoch100만 새로 추론했다. k15/k25와 DCC01/02/03의 공식 재생성 입력·batch1·정답·검색을 유지한다. 새 모델은 smoke 뒤 6조건을 평가했다. 시험/native 수치로 선택하지 않았다.\n\n## 결과\n\n24개 고유 모델/조건 원장을 공식 평가함수 없이 재계산했다. 6개 신규와18개 보존원장이다.\n\n| k | 시퀀스 | AP선택 평균 R@1 | R@1선택 평균 R@1 | 차이(%p) |\n|---|---|---:|---:|---:|\n"
    for row in difference:
        content += f"| {row['k']} | {row['sequence']} | {row['AP_selected_mean_percent']:.6f} | {row['R1_selected_mean_percent']:.6f} | {row['difference_pp']:+.6f} |\n"
    content += "\n## 결론·다음\n\n원천은 `results/pdv2_review_native_20261105/imlpr/common_selection_native/independent_verification.json`과 `selection_sensitivity.json`이다. 보존 best/final 두 후보 한정의 선택 민감도이며 누락 중간epoch·최적수렴을 확인한 결과는 아니다. 저자학습/native대전의 장소 겹침과 공식재생성 입력 범위를 유지한다. 원고·위키 통합은 부모 담당자가 수행한다.\n"
    cross_host = root / "cross_host_verification.json"
    if cross_host.exists():
        assert json.loads(cross_host.read_text())["passed"]
        content += "\n서버와 로컬의24개 원장행·각지표·R@1집계평균·파일SHA는 정확히같다. 처음 그룹 JSON 전체 동일성은 표본표준편차9개에서만 실패했고, Python3.12/3.10의 `statistics.stdev` 마지막비트 최대2.7755576e−17 차이였다. 독립 산식 허용오차1e−12 이내임을 별도 `cross_host_verification.json`에 기록했다. 평가 원장이나 수치를 조정하지 않았다.\n"
    if journal.exists():
        assert journal.read_text() == content
    else:
        journal.write_text(content)
    print(json.dumps({"passed": True, "new_ledgers": 6, "all_verified_ledgers": 24, "differences_pp": [r["difference_pp"] for r in difference]}))


if __name__ == "__main__":
    main()
