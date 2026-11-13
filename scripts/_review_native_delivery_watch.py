#!/usr/bin/env python3
"""완료된 전체 원장의 고유 일지와 통합 검토 제안을 생성한다."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from _review_native_pdv2_ringpp import OUT as RING, REPO, sha256, write_json

OUT = REPO / "results/pdv2_review_native_20261105"


def record_ring() -> dict:
    """전체 지표 및 기하 검산이 끝난 뒤 공통46 완료 일지를 남긴다."""
    summary = json.loads((RING / "summary_verified.json").read_text())
    geometry = json.loads((RING / "geometry_independent_verification.json").read_text())
    assert summary["checks_passed"] and geometry["passed"] and geometry["complete"]
    assert geometry["common_queries"] == 195726 and geometry["sequences"] == 46
    current = summary["ringpp_maxfix"]["macro"]
    original = summary["previous_ringpp"]
    delta = summary["delta_previous_pp"]
    path = REPO / "docs/RAW/실험일지/2026-11-06_RINGpp-정확max-공통46-완료.md"
    content = "# RING++ 정확 max 공통46 완료\n\n## 왜\n\n공식 CUDA 격자 최댓값의 비원자 갱신에서 확인한 경쟁 상태의 영향을 공통 시험 규약에서도 검증한다. 기존 공식 원장과 표는 보존한다.\n\n## 설정\n\n시험 결과로 고른 설정은 없다. 결과 전에 동결한 정확 max 연산만 바꾸고 원래 점 특징·각도·entropy·키프레임·과거 후보·정답·집계를 유지한다. 시험46시퀀스197,796키프레임을 입력으로 사용하며 공통질의195,726개, 양성61,682개다. 양성 없는KITTI07을 제외한45시퀀스를6도시로 평균한다. GPU1의 기존 작업을 보존하며 Docker 안에서 실행했다.\n\n## 결과\n\n| 지표 | 보존 RING++ | 정확 max | 차이(%p) |\n|---|---:|---:|---:|\n"
    for key in ["AP", "max_F1", "R@100P", "R@1"]:
        content += f"| {key} | {100*original[key]:.6f} | {100*current[key]:.6f} | {delta[key]:+.6f} |\n"
    content += "\n`results/pdv2_ringpp_maxfix_20261105/summary_verified.json`에서 저장 예측의 지표와 도시 집계를 독립 산식으로 재계산했다. `geometry_independent_verification.json`에서는 모든 과거 후보의3D거리를 다시 계산해 후보 수·최소 거리·선택 승자의 거리·5m/25m 정답을 검산했다. 전체46시퀀스와195,726공통질의를 확인했다. 원래Table3시간·메모리를 수정본의 측정치로 간주하지 않는다.\n\n## 결론·다음\n\n최종 수치와 변경폭은 위 독립 검산 결과로만 보고한다. 원고 반영 제안은 `results/pdv2_review_native_20261105/paper_proposal/ring_common46_addendum.tex`이며 원고·위키의 직접 수정은 부모 담당자가 한다. 각도·entropy·지역좌표 대조는 이 결과에 합치지 않는다.\n"
    if not path.exists():
        path.write_text(content)
    else:
        assert path.read_text() == content
    return {"complete": True, "summary_sha256": sha256(RING / "summary_verified.json"),
            "geometry_sha256": sha256(RING / "geometry_independent_verification.json"),
            "raw_journal": str(path.relative_to(REPO)), "macro": current, "delta_previous_pp": delta}


def build_proposal() -> dict:
    """검산된 추가안만 모아 검토 PDF를 생성한다."""
    proposal = OUT / "paper_proposal"
    helios = json.loads((OUT / "helios/full_release_final_verification.json").read_text())
    assert helios["passed"] and helios["independently_verified_ledgers"] == 96
    comparisons = helios["public_vs_regenerated_velodyne_comparisons"]
    assert len(comparisons) == 24
    changes = [r["delta_regenerated_minus_public_pp"] for r in comparisons]
    original = (proposal / "helios_regeneration_addendum.tex").read_text()
    revised = original.replace("Regenerating unavailable Velodyne inputs.", "Sensitivity to point-cloud regeneration.")
    revised = revised[:revised.index("Across the eight matched")]
    revised += ("The full 24-pair benchmark is evaluated separately with all released inputs. "
                "Across 24 matched comparisons spanning three Velodyne sessions, two databases, and four models, "
                f"R@1 on regenerated inputs differs from R@1 on released inputs by {min(changes):+.3f} to "
                f"{max(changes):+.3f} percentage points. "
                "This is a preprocessing sensitivity check with fixed weights and query labels.\n")
    context_path = proposal / "helios_regeneration_full_context_v2.tex"
    if context_path.exists():
        assert context_path.read_text() == revised
    else:
        context_path.write_text(revised)
    required = ["body_insert_completed_20261106.tex", "native_compact_table_20261106.tex",
                "helios_author_reproduction.tex", "helios_full_coverage_addendum.tex", "helios_regeneration_full_context_v2.tex",
                "seqot_training_place_note.tex", "seqot_selection_addendum.tex", "helios_same_sensor_addendum.tex", "imlpr_overlap_note.tex",
                "imlpr_archive_scope_note.tex",
                "ring_common46_addendum.tex", "ring_angle_addendum.tex", "ring_entropy_addendum.tex", "label_conventions_addendum.tex",
                "training_recipe_note.tex", "seqot_common_native_bridge.tex",
                "ring_score_separation_roc_addendum.tex"]
    for optional in ["ring_feature_metric_addendum.tex", "imlpr_selection_addendum.tex"]:
        if (proposal / optional).exists():
            required.append(optional)
    assert all((proposal / f).exists() for f in required)
    text = "\\documentclass[10pt]{article}\n\\usepackage[margin=0.75in]{geometry}\n\\usepackage{booktabs,amsmath,hyperref}\n\\title{Verified Baseline Diagnostics: Final Additions}\n\\author{}\n\\date{6 November 2026}\n\\begin{document}\n\\maketitle\n"
    text += "\n".join("\\input{" + f + "}" for f in required) + "\n\\end{document}\n"
    (proposal / "completed_additions_20261106.tex").write_text(text)
    with (proposal / "completed_additions_build.log").open("w") as log:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "completed_additions_20261106.tex"], cwd=proposal, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "completed_additions_20261106.tex"], cwd=proposal, stdout=log, stderr=subprocess.STDOUT, check=True)
    tex_log = (proposal / "completed_additions_20261106.log").read_text()
    assert "Overfull" not in tex_log
    return {"pdf": str((proposal / "completed_additions_20261106.pdf").relative_to(REPO)),
            "pdf_sha256": sha256(proposal / "completed_additions_20261106.pdf"),
            "included_source_sha256": {f: sha256(proposal / f) for f in required}, "latex_overfull": False}


def main() -> None:
    """기존 GPU 실행을 건드리지 않고 완료 자료에만 반응한다."""
    pid = OUT / "delivery_watch.pid"
    if pid.exists():
        try:
            os.kill(int(pid.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit("완료 기록기가 이미 실행 중이다")
    pid.write_text(str(os.getpid()))
    finished = {}
    while True:
        state = {"pid": os.getpid(), "unix_time": time.time(), "finished": finished, "complete": False}
        try:
            if (RING / "summary_verified.json").exists() and (RING / "geometry_independent_verification.json").exists():
                geo = json.loads((RING / "geometry_independent_verification.json").read_text())
                if geo.get("complete"):
                    finished["common46"] = record_ring()
                    diagnostic = RING / "score_separation_diagnostic.json"
                    fixed = json.loads((RING / "score_diagnostic_protocol.json").read_text())
                    assert sha256(REPO / "scripts/_review_native_ringpp_score_diagnostic.py") == fixed["script_sha256"]
                    if not diagnostic.exists():
                        subprocess.run(["python3", "scripts/_review_native_ringpp_score_diagnostic.py"], cwd=REPO, check=True)
                    assert json.loads(diagnostic.read_text())["passed"]
                    finished["score_diagnostic"] = {"sha256": sha256(diagnostic)}
            helios = OUT / "helios/full_release_final_verification.json"
            if helios.exists():
                hp = json.loads(helios.read_text())
                assert hp["passed"] and hp["complete"] and hp["independently_verified_ledgers"] == 96
                finished["helios_full"] = {"sha256": sha256(helios), "released_pairs": 24}
            metric_path = OUT / "ring_feature_metric/collection_status.json"
            metric = json.loads(metric_path.read_text()) if metric_path.exists() else {}
            runner_path = OUT / "ring_feature_metric/runner_status.json"
            runner = json.loads(runner_path.read_text()) if runner_path.exists() else {}
            metric_failed = runner.get("stage") == "failed"
            if "common46" in finished and "helios_full" in finished and (metric.get("complete") or metric_failed):
                finished["feature_metric"] = metric if metric.get("complete") else {"complete": False, "failed": True, "runner": runner}
                subprocess.run(["python3", "scripts/_review_native_final_packet.py"], cwd=REPO, check=True)
                finished["native_packet"] = {"sha256": sha256(OUT / "final_native_delivery_20261106.json")}
                finished["proposal"] = build_proposal()
                subprocess.run(["python3", "docs/도구/check_wiki.py", "docs"], cwd=REPO, check=True)
                state["complete"] = True
                state["scope"] = "공통46와 공개HeLiOS24쌍 검산/일지/추가안 전달 완료. 지역좌표 진단의 성공/실패는 finished.feature_metric에 별도 기록한다. 보류된 장기수렴학습은 별개 미완료이며 이 마커가 모든 논문학습조건 재현이나 최적수렴을 의미하지 않는다."
        except Exception as exc:
            state["error"] = repr(exc)
        write_json(OUT / "delivery_watch_status.json", state)
        with (OUT / "delivery_watch_history.jsonl").open("a") as stream:
            stream.write(json.dumps(state, ensure_ascii=False) + "\n")
        print(json.dumps(state, ensure_ascii=False), flush=True)
        if state["complete"]:
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
