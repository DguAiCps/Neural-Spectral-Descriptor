#!/usr/bin/env python3
"""baseline별 보존 가중치와 검증 선택 기록을 읽기 전용으로 감사한다."""
from __future__ import annotations

from pathlib import Path
import re

from _review_selection_bev import OUT, ROOT, read, sha, write


def relative(path: str) -> str:
    """서버·컨테이너 절대 경로를 저장소 상대 경로로 맞춘다."""
    if "Neural-Spectral-Codec/" in path:
        return path.split("Neural-Spectral-Codec/", 1)[1]
    return path


def main() -> None:
    """가중치 보유와 평가 기록을 서로 다른 수량으로 보고한다."""
    inventory = read(OUT / "remote_inventory.json")
    files = [r for rows in inventory["files"].values() for r in rows]
    report = dict(source_inventory_sha256=sha(OUT / "remote_inventory.json"), methods={})
    scores = {"bevplace": "score_negl2: 공식 8192D descriptor -L2", "helios": "score_cos: 공식 8448D 단위 descriptor 코사인",
        "imlpr": "score_cos: 공식 8448D SALAD 단위 descriptor 코사인", "disco": "score_disco: 정규화하지 않은 공식1024D signature의 -L2",
        "seqot": "score_l2: 공식 가운데 창256D GeM descriptor -L2", "ringsharp": "score_trans: 공식 이동 재정렬 오차의 음수",
        "deepring": "score_corr: 순환상관 C의 학습 logit W*C+b에서 상수1을 뺀 값"}
    score_sources = {"bevplace": "_bevplace_official_dump.py", "helios": "_helios_official_dump.py",
        "imlpr": "_imlpr_official_dump.py", "disco": "_disco_official_dump.py", "seqot": "_seqot_official_dump.py",
        "ringsharp": "_ringsharp_official.py", "deepring": "_deepring_reimpl.py"}
    for method in scores:
        rows = {}
        for seed in ((1, 3, 4) if method == "deepring" else (1, 2, 3)):
            suffix = "" if seed == 1 else f"_s{seed}"
            if method == "ringsharp":
                path = ROOT / f"results/pdv2_ring_official{suffix}/ringsharp_ckpt_selection.json"
            elif method == "deepring":
                path = ROOT / f"results/pdv2_deepring_reimpl{suffix}/train_selection.json"
            else:
                base = method + ("_official_centered" if method == "seqot" else "_official")
                path = ROOT / f"results/pdv2_{base}{suffix}/train_selection.json"
            record = read(path)
            if method == "ringsharp":
                chosen = dict(epoch=record["selected_epoch"], checkpoint=record["selected_weight"], sha256=record["weight_sha256"])
                count = len(record["epochs"])
                rule = record["rule"]
            elif method == "seqot":
                chosen = record["stage2"]["chosen"]
                count = len(record["stage2"]["epochs"])
                rule = record["_meta"]["rule"]
            elif method == "helios":
                chosen = record[record["chosen"]]
                count = len(record["candidates"])
                rule = record["_meta"]["rule_val_best"]
            else:
                chosen = record.get("best", record.get("val_best"))
                count = len(record["evals"])+1 if method == "imlpr" else len(record["epochs"])
                rule = record["_meta"]["selection_rule"]
            ckpt = relative(chosen.get("checkpoint", chosen.get("path", "")))
            folder = str(Path(ckpt).parent)
            stored = [r for r in files if str(Path(r["path"]).parent) == folder]
            if method == "bevplace":
                useful = [r for r in stored if r["path"].endswith(("model_best.pth.tar", "checkpoint.pth.tar"))]
                feasibility = "두 후보(best와 마지막 label39)만 가능; 중간38epoch 가중치 없음"
            elif method == "imlpr":
                useful = [r for r in stored if r["path"].endswith(("_best.pth", "_final.pth"))]
                feasibility = "best/final만 가능; 5epoch 간격 평가21회 중 중간 평가상태 가중치 없음"
            elif method == "helios":
                useful = [r for r in stored if Path(r["path"]).name in record["candidates"]]
                feasibility = "기존 후보파일10개 보존; 같은 epoch·다른 직렬화가 있어 모델 텐서 중복을 따로 확인해야 함"
            elif method == "disco":
                useful = [r for r in stored if re.fullmatch(r"epoch_\d+\.ckpt", Path(r["path"]).name)]
                feasibility = "원래20epoch 후보 보존; KISS 최종 검증 점수 재추출 필요"
            elif method == "seqot":
                useful = [r for r in stored if re.fullmatch(r"amodel_gem\d+\.pth.tar", Path(r["path"]).name)]
                feasibility = "공식 가운데 창 stage2 99후보 보존; 고정 stage1 아래 KISS 최종 검증 재추출 필요"
            elif method == "ringsharp":
                useful = [r for r in stored if re.search(r"_\d+\.pth$", r["path"])]
                feasibility = "30epoch 후보 보존; 기존 검증은 재방문 stride4, 전체 KISS 최종 점수 재추출 필요"
            else:
                useful = [r for r in stored if re.fullmatch(r"epoch_\d+\.pth", Path(r["path"]).name)]
                feasibility = "20epoch 후보 보존; KISS 최종 검증 점수 재추출 필요"
            rows[str(seed)] = dict(selection_file=str(path.relative_to(ROOT)), selection_sha256=sha(path),
                original_validation_records=count, original_choice=chosen, original_rule=rule,
                retained_candidate_count=len(useful), retained_candidates=useful, feasibility=feasibility,
                existing_validation_output="선택 JSON은 재방문 R@1 집계만 있으며 비재방문 수락점수/AP를 복원할 수 없음",
                kiss_validation_best_ledgers=9 if method == "bevplace" else "이 감사에서 전체 후보에 대응한 KISS 최종 원장을 찾지 못함")
        score_source = ROOT / "scripts" / score_sources[method]
        report["methods"][method] = dict(primary_score=scores[method], seeds=rows,
            score_source=str(score_source.relative_to(ROOT)), score_source_sha256=sha(score_source))
    for method in ("scpp", "ring", "ringpp"):
        report["methods"][method] = dict(training_free=True, checkpoint_selection="해당 없음")
    report["findings"] = [
        "기존 감사 available_epochs는 학습 기록/후보파일 개수이며 실제 보존 epoch 가중치 개수와 다르다.",
        "BEVPlace++ 공식 saveCheckpoint(_external/BEVPlace2/main.py:231)는 checkpoint.pth.tar를 덮어쓰고 개선 시 model_best.pth.tar만 복사한다. epoch별40개를 저장하는 코드가 아니다.",
        "BEVPlace++ label4/3/2와 마지막39는 0-based이며 완료한 학습 횟수는5/4/3와40이다.",
        "ImLPR 선택 at_start_of_epoch100/95/75는 실제99/94/74epoch 학습 상태다.",
        "HeLiOS10개는 epoch1..10이 아니다. 주기 저장10,20,...80와 best/final 파일이다.",
        "시험 원장으로 AP를 선택하지 않는다. 기존 검증 R@1만으로 AP 목적 선택의 효과를 계산할 수 없다.",
        "이번 실측은 최우선 직접 비교인 BEVPlace++ 두 후보×3시드다. 다른 보존후보의 전체 재평가는 미수행이다."]
    nsd_path = ROOT / "results/pdv2_diffattn_contrast_recalibrated_20261104/summary.json"
    nsd = read(nsd_path)
    tags = ("diff_lr5_4", "diff_lr5_8", "diff_lr4_4", "diff_lr4_8")
    report["current_nsd_selection"] = dict(source=str(nsd_path.relative_to(ROOT)), source_sha256=sha(nsd_path),
        first_seed_candidates={tag: nsd["results"][tag]["macro"] for tag in tags}, selected=nsd["best"]["diff"],
        fixed_winner_R1_identical=len({nsd["results"][tag]["macro"]["R@1"] for tag in tags}) == 1,
        interpretation="첫시드 원키 검색 승자를 고정한 판정 단계에서 lr/epoch를 AP로 선택하고, 이후 matching_only 검색 연결을 별도로 검증했다. 현재 최종 파이프라인에서 모든 epoch를 AP로 훑은 선택이 아니다. 시드2·3은 첫 선택을 그대로 사용한다.")
    write(OUT / "baseline_inventory_audit.json", report)


if __name__ == "__main__":
    main()
