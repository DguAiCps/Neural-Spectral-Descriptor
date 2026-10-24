#!/usr/bin/env python3
"""공식 코드 비교 방법의 시드별 시험 기록을 주표 공통 질의로 재집계하고 시드 평균·표본 표준편차를 낸다.

- 실행마다 _pdv2_main_baselines_official.aggregate를 그대로 쓴다(질의·정답 거리·후보 규칙과
  방법 자체 요약 재현을 같은 방식으로 확인한다).
- 학습 없는 SC++·RING·RING++는 실행이 하나다(시드 없음).
- DeepRING 2시드는 발산 규칙(2026-10-21 사용자 결정: 20에폭 검증 최고 R@1 < 0.1)으로 시험 기록을
  만들지 않았고, 시드 번호 순서대로 다음인 4시드가 대체한다. 판정 근거는 2시드 선택 기록에서 다시 확인한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_main_baselines_official import METHODS, METRICS, aggregate, sha256, write_json

OUT = REPO / "results/pdv2_baseline_seeds"
TRAINING_FREE = ("scpp", "ring", "ringpp")
SEED_FOLDERS = {   # method -> [(seed, ledger folder)]
    "ringsharp": [(1, "results/pdv2_ring_official/odom"), (2, "results/pdv2_ring_official_s2/odom"),
                  (3, "results/pdv2_ring_official_s3/odom")],
    "deepring": [(1, "results/pdv2_deepring_reimpl/odom"), (3, "results/pdv2_deepring_reimpl_s3/odom"),
                 (4, "results/pdv2_deepring_reimpl_s4/odom")],
    "disco": [(k, f"results/pdv2_disco_official{s}/odom") for k, s in ((1, ""), (2, "_s2"), (3, "_s3"))],
    "helios": [(k, f"results/pdv2_helios_official{s}/odom") for k, s in ((1, ""), (2, "_s2"), (3, "_s3"))],
    "seqot": [(k, f"results/pdv2_seqot_official_centered{s}/odom") for k, s in ((1, ""), (2, "_s2"), (3, "_s3"))],
    "imlpr": [(k, f"results/pdv2_imlpr_official{s}/odom") for k, s in ((1, ""), (2, "_s2"), (3, "_s3"))],
    "bevplace": [(k, f"results/pdv2_bevplace_official{s}/odom") for k, s in ((1, ""), (2, "_s2"), (3, "_s3"))],
}
DEEPRING_DIVERGED = REPO / "results/pdv2_deepring_reimpl_s2_diverged/train_selection.json"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}

    def track(path: Path) -> Path:
        sources[str(path.relative_to(REPO))] = sha256(path)
        return path

    track(Path(__file__))
    track(REPO / "scripts/_pdv2_main_baselines_official.py")
    div = json.loads(track(DEEPRING_DIVERGED).read_text())
    assert div["finished"] and div["best"]["family_macro_r1"] < 0.1, div["best"]
    report = {"scope": "공식 코드 비교 방법의 시드별 시험 기록 재집계(주표 공통 질의); 새 선택·추론 없음",
              "aggregate": "시드마다 6도시 매크로 → 시드 평균, 표본 표준편차(ddof=1)",
              "deepring_divergence": {"rule": "20에폭 검증 가족 평균 R@1 최고값 < 0.1이면 발산, 시드 번호 순서로 대체 "
                                              "(2026-10-21 사용자 결정, 2시드 학습 3에폭째에 고정)",
                                      "diverged_seed": 2, "best_val_r1": div["best"]["family_macro_r1"],
                                      "replacement_seed": 4},
              "methods": {}}
    for method, folder, prefix, score_name, summary_name, summary_key in METHODS:
        runs = [(None, folder)] if method in TRAINING_FREE else SEED_FOLDERS[method]
        per_seed = {}
        for seed, f in runs:
            entry = aggregate(f, prefix, score_name, summary_name, summary_key, track)
            assert entry is not None, (method, seed, f)
            per_seed["-" if seed is None else str(seed)] = {"source": entry["source"], "macro": entry["macro"]}
        vals = np.array([[r["macro"][k] for k in METRICS] for r in per_seed.values()])
        stats = {k: {"mean": float(vals[:, i].mean()),
                     "std": float(vals[:, i].std(ddof=1)) if len(vals) > 1 else None,
                     "values": vals[:, i].tolist()} for i, k in enumerate(METRICS)}
        report["methods"][method] = {"score": score_name, "training_free": method in TRAINING_FREE,
                                     "seeds": list(per_seed), "per_seed": per_seed, "mean_std": stats}
        print(f"{method:10s} seeds {','.join(per_seed):6s} " + " ".join(
            f"{k} {stats[k]['mean']:.4f}" + (f"±{stats[k]['std']:.4f}" if stats[k]["std"] is not None else "")
            for k in ("AP", "max_F1", "R@100P", "R@1")), flush=True)
    report["input_sha256"] = sources
    report["checks_passed"] = True
    write_json(OUT / "summary.json", report)


if __name__ == "__main__":
    main()
