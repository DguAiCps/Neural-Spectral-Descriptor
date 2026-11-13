#!/usr/bin/env python3
"""전체 공개 HeLiOS 원장과 보존 검사를 회수하고 최종 범위를 검증한다."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from _review_native_collect import NATIVE, REMOTE, sync
import _review_native_helios as native

MODELS = ["author", "v2_s1", "v2_s2", "v2_s3"]
OUT = native.OUT


def finish() -> dict:
    """공개96원장·기존52파일 보존·재생성24짝을 함께 확인한다."""
    coverage = json.loads((OUT / "coverage_with_regeneration.json").read_text())
    assert coverage["all_pairs_complete"] and coverage["released_input_pairs"] == 24
    assert coverage["new_regenerated_input_pairs"] == 0
    independent = json.loads((OUT.parent / "learned_independent_verification.json").read_text())["helios"]
    assert len(independent) == 96 and all(r["passed"] for r in independent)
    assert all(sum(r["model"] == m for r in independent) == 24 for m in MODELS)
    transition = json.loads((OUT / "score_queue_transition_20261106.json").read_text())
    for name, expected in transition["before_sha256"].items():
        assert native.sha(OUT / name) == expected, name
    checks = {}
    for model in MODELS:
        path = OUT / model / "full_release_preservation.json"
        checks[model] = json.loads(path.read_text())
        assert checks[model]["passed"]
        assert len(checks[model]["reused_prediction_files"]) == 12
        assert len(checks[model]["new_prediction_files"]) == 12
    manifest = json.loads((OUT / "input_manifest.json").read_text())
    assert len(manifest["sequences"]) == 12
    assert all(r["available"] and len(r["point_sha256"]) == r["n_frames"] for r in manifest["sequences"].values())
    assert sum(r["n_frames"] for r in manifest["sequences"].values()) == 19342
    comparisons = []
    for model in MODELS:
        for db in native.DATABASES:
            for session in [1, 2, 3]:
                name = f"Roundabout{session:02d}-Velodyne"
                filename = f"{db}__{name}.npz"
                public_path = OUT / model / "native_pairs" / filename
                regen_path = OUT.parent / "helios_regenerated_velodyne" / model / "native_pairs" / filename
                with np.load(public_path) as p, np.load(regen_path) as r:
                    same = p["evaluated"] & r["evaluated"]
                    assert same.any()
                    public = float(p["first_correct"][same].mean())
                    regen = float(r["first_correct"][same].mean())
                    comparisons.append({"model": model, "database": db, "query": name,
                        "common_positive": int(same.sum()), "public_r1": public, "regenerated_r1": regen,
                        "delta_regenerated_minus_public_pp": 100 * (regen-public),
                        "top1_changes": int(np.sum(p["indices"][:, 0] != r["indices"][:, 0])),
                        "public_prediction_sha256": native.sha(public_path),
                        "regenerated_prediction_sha256": native.sha(regen_path)})
    assert len(comparisons) == 24
    report = {"passed": True, "complete": True, "created_unix": time.time(),
        "released_input_sequences": 12, "released_input_frames": 19342,
        "benchmark_pairs_per_model": 24, "models": 4, "independently_verified_ledgers": 96,
        "old_files_preserved_by_sha256": len(transition["before_sha256"]),
        "new_prediction_files": 48, "groups": coverage["groups"],
        "public_vs_regenerated_velodyne_comparisons": comparisons,
        "input_manifest_sha256": native.sha(OUT / "input_manifest.json"),
        "coverage_sha256": native.sha(OUT / "coverage_with_regeneration.json"),
        "independent_sha256": native.sha(OUT.parent / "learned_independent_verification.json"),
        "scope": "전체 공개24쌍. 모든 그룹은6쌍의 공식분모 평균. v2검증장소이며 가중치선택/주표교체 아님. 기존재생성40원장과부분12쌍결과 보존."}
    native.write(OUT / "full_release_final_verification.json", report)
    journal = native.REPO / "docs/RAW/실험일지/2026-11-06_HeLiOS-공개24쌍-완료.md"
    text = "# HeLiOS 공개24쌍 완료\n\n## 왜\n\n공개 자료 quota로 남은 센서 범위를 해소하고 논문의 전체 센서군과 비교한다.\n\n## 설정\n\n02:48 재시도에서 누락6개 archive를 확보했다. 새9,713점군은 D:에 풀고 서버 전수SHA를 확인했다. 기존6개와 합해12센서세션19,342점군이다. 고정 저자가중치와 원래 v2선택3시드, 공개 overlap>0.5와공식검색을 유지했다. 새가중치선택이나 추가학습은 하지 않았다.\n\n## 결과\n\n24쌍×4모델=96원장을 독립 검산했다. 기존48예측과4부분보고서는 SHA까지 보존하고 신규48예측만 저장했다.\n\n| 지도 | 질의센서군 | 논문 R@1 | 저자 R@1 | v2 R@1 평균±표본표준편차 |\n|---|---|---:|---:|---:|\n"
    for group in coverage["groups"]:
        values = group["official_mean"]
        seeds = [100*values[m]["r1"] for m in MODELS[1:]]
        text += f"| {group['database']} | {group['group']} | {100*group['published_reference']['r1']:.6f} | {100*values['author']['r1']:.6f} | {np.mean(seeds):.6f} ± {np.std(seeds,ddof=1):.6f} |\n"
    delta = [r["delta_regenerated_minus_public_pp"] for r in comparisons]
    text += f"\nVelodyne3세션×2지도×4모델의 같은양성질의24대조에서 재생성−공개R@1 차이는{min(delta):+.6f}~{max(delta):+.6f}%p다. 자료출처는 성능으로 선택하지 않았으며 원래benchmark는 전부 공개입력을 쓴다.\n\n## 결론·다음\n\n원천은 `results/pdv2_review_native_20261105/helios/full_release_final_verification.json`과 `coverage_with_regeneration.json`이다. 전체 공개센서 범위를 채웠지만 Roundabout은v2검증장소이므로 새로운 장소분리 시험으로 주장하지 않는다. 저자/v2학습장소·센서·학습조건 변화의 개별 인과효과도 분리한 것이 아니다. 영문 제안은 `paper_proposal/helios_full_coverage_addendum.tex`이며 원고·위키는 부모가 통합한다.\n"
    if not journal.exists():
        journal.write_text(text)
    else:
        assert journal.read_text() == text
    scope = OUT.parent / "native_completion_scope.json"
    current = json.loads(scope.read_text())
    current["methods"]["HeLiOS"]["actual_run"] = "전체 공개12센서세션·24쌍×4모델=96원장 독립검산 완료"
    current["methods"]["HeLiOS"]["external_remaining"] = "없음: 공개12archive quota해제후확보·추론·검산완료"
    current["methods"]["HeLiOS"]["effective_original_benchmark_coverage"] = "24/24쌍, 모두 공개가공입력; 재생성대조는 별도보존"
    current["updated_unix"] = time.time()
    native.write(scope, current)
    return {"complete": True, "verification_sha256": native.sha(OUT / "full_release_final_verification.json"),
            "journal": str(journal.relative_to(native.REPO)), "delta_regenerated_pp_range": [min(delta), max(delta)]}


def main() -> None:
    """작은 상태·보존 검사만 추가 회수하고 주 수집기의96원장 검산을 기다린다."""
    pidfile = OUT / "full_finalize.pid"
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit("전체 HeLiOS 마무리기가 이미 살아 있다")
    pidfile.write_text(str(os.getpid()))
    while True:
        state = {"pid": os.getpid(), "unix_time": time.time(), "complete": False}
        try:
            # 최상위 진행 파일만 회수한다. 입력 하위 폴더의 D: 링크에는 접근하지 않는다.
            subprocess.run(["rsync", "-az", "--omit-dir-times", "--include=full_score*.log",
                            "--include=full_score_queue_status.json", "--include=score_queue_transition_20261106.json",
                            "--exclude=*", REMOTE + NATIVE + "/helios/", str(OUT) + "/"], check=True)
            completed = []
            for model in MODELS:
                sync(NATIVE + "/helios/" + model, ["full_release_preservation.json"])
                if (OUT / model / "full_release_preservation.json").exists():
                    completed.append(model)
            state["models_with_preservation_check"] = completed
            coverage = json.loads((OUT / "coverage_with_regeneration.json").read_text())
            if len(completed) == 4 and coverage["released_input_pairs"] == 24:
                state.update(finish())
        except Exception as exc:
            state["error"] = repr(exc)
        native.write(OUT / "full_finalize_status.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        if state["complete"]:
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
