#!/usr/bin/env python3
"""비교 방법 원장의 원본 코드 대조와 학습 선택 기록을 감사한다."""
import json
from pathlib import Path

import numpy as np
import _paper_fixed_threshold as f

ROOT = f.REPO
OUT = ROOT/"results/pdv2_review_response_20261104/baseline_audit.json"


def main() -> None:
    """원장 검사 통과와 원 논문 성능 재현을 구분하여 보고한다."""
    manifest = f.read(ROOT/"results/pdv2_baseline_seeds/summary.json")
    methods = {}
    for method, folder, prefix, score, _, _ in f.METHODS:
        runs = [(None, folder)] if method in f.TRAINING_FREE else f.SEED_FOLDERS[method]
        records, examples, native_checks = [], {}, []
        for seed, run in runs:
            for sensor, seq in f.TEST:
                path = ROOT/run/f"{prefix}_{sensor}_{seq}.npz"
                assert f.digest(path) == manifest["input_sha256"][str(path.relative_to(ROOT))]
                with np.load(path) as d:
                    meta = json.loads(str(d["meta"]))
                    q, j, values = d["query_idx"], d["j_star"], d[score]
                    assert ((j < 0) | (j <= q-30)).all()
                    assert np.isfinite(values[j >= 0]).all()
                    if "scorer_crosscheck_vs_batch_circorr" in meta:
                        cross = meta["scorer_crosscheck_vs_batch_circorr"]
                        assert all(r["same_argmax"] for r in cross)
                        native_checks.extend(cross)
                    for key in ("l2_check", "kdtree_check"):
                        if key in meta:
                            check = meta[key]
                            count = check.get("n_checked")
                            matches = [v for k, v in check.items() if "equals_j_star" in k]
                            if count is not None and matches: assert all(v == count for v in matches), (path, check)
                    examples.setdefault(sensor, {k: meta[k] for k in ("dataset_table_row", "descriptor_norm", "eval_window",
                                             "window", "deviation", "protocol", "encoder", "l2_check", "kdtree_check") if k in meta})
                    records.append(dict(seed=seed, sequence=f"{sensor}/{seq}", queries=len(q), finite_returned_scores=True,
                                        prefix_valid=True, sha256=f.digest(path)))
        methods[method] = dict(records=records, examples=examples, native_scorer_crosschecks=len(native_checks),
                              max_crosscheck_error=max((r["max_abs_diff"] for r in native_checks), default=None),
                              main_metrics=manifest["methods"][method]["mean_std"])
    selections = {}
    for method in ("bevplace", "helios", "imlpr", "disco"):
        for seed in (1, 2, 3):
            suffix = "" if seed == 1 else f"_s{seed}"
            p = ROOT/f"results/pdv2_{method}_official{suffix}/train_selection.json"
            d = f.read(p)
            best = d.get("best", d.get("val_best"))
            selections[f"{method}/s{seed}"] = dict(source=str(p.relative_to(ROOT)), sha256=f.digest(p), chosen=best,
                    available_epochs=len(d.get("epochs", d.get("candidates", []))))
    for seed in (1, 2, 3):
        suffix = "" if seed == 1 else f"_s{seed}"
        p = ROOT/f"results/pdv2_seqot_official_centered{suffix}/train_selection.json"
        d = f.read(p)
        selections[f"seqot/s{seed}"] = dict(source=str(p.relative_to(ROOT)), sha256=f.digest(p),
                    chosen=d["stage2"]["chosen"], available_epochs=len(d["stage2"]["epochs"]))
    checks = {}
    campaign = OUT.parent
    for name, p in (("ring", campaign/"ring_checks/check_ring_official_cpu.json"),
                    ("helios", campaign/"helios_selftest.json"),
                    ("imlpr", campaign/"imlpr_selftest.json"),
                    ("seqot", campaign/"seqot_selftest.json")):
        assert p.exists(), p
        checks[name] = dict(source=str(p.relative_to(ROOT)), sha256=f.digest(p))
    log = campaign/"disco_selftest.log"
    assert "SELFTEST_OK" in log.read_text()
    checks["disco"] = dict(source=str(log.relative_to(ROOT)), sha256=f.digest(log))
    f.write(OUT, dict(methods=methods, training_selection=selections, rerun_checks=checks,
        findings=["10방법의 원장 해시·후보 접두·반환 점수 유한성을 재검산했다.",
                  "RING/RING++ 저장 원장의 공식 상관 함수 대조는 모든 확인 질의에서 같은 승자다.",
                  "SeqOT 주표는 공식 가운데 창과 원래 L2 점수다. 과거 창으로 낮아졌다는 설명은 현 주표에 적용되지 않는다.",
                  "ImLPR는 100에포크 실행 및 75/95/100에포크 선택 기록이 있다. 모든 낮은 수치를 짧은 학습만으로 설명할 수 없다."],
        unresolved=["검사한 입력·계산 경로의 정합성은 원 논문 벤치마크 성능 재현을 보장하지 않는다.",
                    "8개 훈련 시퀀스·장소와 센서 분포·학습 예산·선택 지표의 영향은 분리되지 않았다.",
                    "저자 가중치의 학습 장소 오염을 표시한 별도 원 논문 규약 평가가 남아 있다.",
                    "RING++ 낮은 AP의 원인을 구현 오류 또는 방법 열세로 단정하지 않는다."]))
    print("AUDIT_PASSED", sum(len(v["records"]) for v in methods.values()), "ledgers", flush=True)


if __name__ == "__main__": main()
