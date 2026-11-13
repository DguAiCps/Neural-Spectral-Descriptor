#!/usr/bin/env python3
"""게재 논문·공개 코드·보존 v2 학습 예산을 구별해 기록한다."""
from __future__ import annotations

import configparser
import hashlib
import json
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105"


def digest(path: Path) -> str:
    """검토한 원천 파일 전체의 SHA256을 반환한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """실제 세 시드 기록을 확인하고 추가 학습 없이 해석 자료를 쓴다."""
    sources = [
        "scripts/_review_native_training_recipe_audit.py",
        "scripts/_helios_official_train.py",
        "scripts/_imlpr_official_train.py",
        "scripts/_seqot_official_train.py",
        "_external/HeLiOS/config/config_baseline_helios.txt",
        "_external/ImLPR/config/config_helipr.txt",
        "_external/ImLPR/misc/utils.py",
        "_external/ImLPR/training/train.py",
        "_external/SeqOT/config/config.yml",
        "_external/SeqOT/train/training_seqot.py",
        "_external/SeqOT/train/training_gem.py",
    ]
    records = {"helios": [], "imlpr": [], "seqot": []}
    for seed in [1, 2, 3]:
        suffix = "" if seed == 1 else f"_s{seed}"
        hp = REPO / f"results/pdv2_helios_official{suffix}/train_record.json"
        ip = REPO / f"results/pdv2_imlpr_official{suffix}/train_selection.json"
        sp = REPO / f"results/pdv2_seqot_official_centered{suffix}/train_selection.json"
        h, i, s = [json.loads(p.read_text()) for p in [hp, ip, sp]]
        assert len(h["epochs"]) == 80 and h["finished"]
        assert h["epochs"][0]["lr"] == 1e-3
        cp = configparser.ConfigParser()
        cp.read_string(i["_meta"]["config"])
        assert cp["TRAIN"].getfloat("lr") == 5e-4
        assert cp["TRAIN"].getint("epochs") == 100
        assert len(i["epochs"]) == 100 and i["finished"]
        assert all(len(s[k]["epochs"]) == 99 and s[k]["finished"]
                   for k in ["stage1", "stage2"])
        records["helios"].append({"seed": seed, "completed_epochs": 80,
                                  "first_logged_lr": h["epochs"][0]["lr"]})
        records["imlpr"].append({"seed": seed, "completed_epochs": 100,
                                 "configured_lr": cp["TRAIN"].getfloat("lr"),
                                 "optimizer": "Adam (공개 parser 기본값; config에 override 없음)"})
        assert not cp["TRAIN"].get("optimizer")
        records["seqot"].append({"seed": seed, "completed_epochs_per_stage": [99, 99]})
        sources.extend(str(p.relative_to(REPO)) for p in [hp, ip, sp])
    papers = {
        "helios": ("learned_sources/helios_author_icra2025.txt", "https://minwoo0611.github.io/assets/pdf/icra2025-helios.pdf"),
        "imlpr": ("learned_sources/imlpr_pmlr_jung25c.txt", "https://proceedings.mlr.press/v305/jung25c.html"),
        "seqot": ("learned_sources/seqot.txt", "https://arxiv.org/abs/2209.07951"),
    }
    paper_sources = {}
    for key, (relative, url) in papers.items():
        path = OUT / relative
        paper_sources[key] = {"text": str(path.relative_to(REPO)), "sha256": digest(path), "url": url}
    report = {
        "checked_unix": time.time(), "passed": True,
        "purpose": "동일 epoch 수와 원 논문 학습 레시피 재현·최적 수렴을 혼동하지 않는다. 설정 선택이나 추가 학습 없음.",
        "records": records,
        "paper_sources": paper_sources,
        "source_sha256": {name: digest(REPO / name) for name in sources},
        "methods": {
            "helios": {
                "paper": "§IV-A: 80 epochs, 초기 lr 1e-3, MultiStepLR",
                "released_and_v2": "80 epochs, lr 1e-3, Adam(weight_decay=1e-4), milestones40/70, loss weights0.8/0.1/0.1",
                "interpretation": "학습 예산과 초기 lr는 논문에 명시된 값과 같다. 학습 장소·센서와 일부 loss 표현은 다르므로 논문 전체 학습조건 일치나 최적 수렴을 뜻하지 않는다.",
                "later_author_guidance": "별도 author_training_guidance.json의 150epoch/lr권고는 후속 안내이며 원 논문의80epoch와 구분한다.",
            },
            "imlpr": {
                "paper": "§4.1: 100 epochs, AdamW, lr 5e-4, 1/10 warmup, cosine",
                "released_defaults": "100 epochs, lr 5e-5, optimizer 미지정→Adam, 명시적 weight_decay 전달 없음",
                "v2": "100 epochs, lr 5e-4, 공개 Adam 경로; 다른 차이는 train_selection._meta.deviations에 보존",
                "interpretation": "v2의100epoch는 논문보다 짧지 않다. 공개 코드 경로와 논문의 optimizer는 다르다. 끝의 검증 상승 때문에 추가 수렴 가능성이 남지만150epoch가 논문 필수조건이라고 하지 않는다.",
            },
            "seqot": {
                "paper": "§IV-A: transformer20epochs/lr5e-6, GeM은 유사 설정에 lr5e-5; GeM20회는 본문에서 별도 숫자로 반복 명시하지 않음",
                "released_and_v2": "각 단계 range(1,100)→99회, lr5e-7/5e-6, Adam, StepLR(5,0.9), stage1양/음3/3·stage2 6/6",
                "interpretation": "v2는 보존 공개 코드 레시피를 따른다. 논문 레시피 재학습을 수행했다고 하지 않는다. stage2 공통 AP/R@1 선택 대조는 완료했으나 stage1 재선택·재학습과 별개다.",
            },
        },
        "scope": "학습 원천·예산 대조와 수렴 검토. 논문 레시피 대비 성능의 인과 분리는 새 학습을 하지 않아 미검증.",
    }
    (OUT / "training_recipe_conformance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    text = "\\paragraph{Training budgets and recipe conformance.}\n"
    text += "Our HeLiOS and ImLPR runs complete 80 and 100 epochs, respectively, matching the epoch counts stated in their papers. Equal epoch counts do not imply identical training recipes or convergence: the released ImLPR path uses Adam whereas the paper specifies AdamW. Our fixed learning rate is $5\\times10^{-4}$, while the released configuration lists $5\\times10^{-5}$. SeqOT follows the released 99-epoch loops for both stages and initial learning rates $5\\times10^{-7}$ and $5\\times10^{-6}$; its paper states 20 transformer-training epochs and learning rates $5\\times10^{-6}$ and $5\\times10^{-5}$. These distinctions are retained in the audit, and the checkpoint diagnostics are not presented as a reproduction of every paper-training setting. No new training configuration is chosen using the diagnostic evaluation.\n"
    (OUT / "paper_proposal/training_recipe_note.tex").write_text(text)
    print(json.dumps({"passed": True, "records": records}, ensure_ascii=False))


if __name__ == "__main__":
    main()
