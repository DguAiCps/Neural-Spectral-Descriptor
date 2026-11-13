#!/usr/bin/env python3
"""표2의 SeqOT 행을 논문 학습법 + 공식 GeM 수치 안정화(log-domain) 3시드 원장으로 바꾼 채택 요약을 만든다.

- 공통 원천(results/pdv2_baseline_seeds/summary.json)과 기존 RING++ 정확max 사본은 고치지 않는다.
- SeqOT 138원장을 `_pdv2_main_baselines_official.aggregate`로 다시 집계하고, 독립 검산을 마친
  records/summary.json의 시드별 매크로·평균·SD와 1e-12 안에서 같은지 확인한다.
- 그 요약을 기반으로 RING++ 행만 정확max 46원장으로 바꾼 사본을 기존 생성기로 다시 만들고,
  SeqOT 외 모든 방법 값이 기존 채택 사본과 같은지 확인한다.
출력은 D의 results/pdv2_seqot_stable_gem_20261108/manuscript_integration_20261108/에 둔다. 새 추론·선택·학습은 없다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_main_baselines_official import METRICS, aggregate, sha256  # noqa: E402

LABEL = "paper recipe + numerical stabilization of official GeM"
STABLE = REPO / "results/pdv2_seqot_stable_gem_20261108"
RECORDS = STABLE / "server_mirror/records/summary.json"
RECORDS_SHA256 = "22767634596c2799273636e0a0231acdc05b735ef1952d668c89f3c7ac50c80f"
COMMON = REPO / "results/pdv2_baseline_seeds/summary.json"
COMMON_SHA256 = "53cb0b724111661f113c49911e07b655b35d7f0645634aaada9fbc1da3f9dada"
PREVIOUS_RINGPP = REPO / "results/pdv2_baseline_revision_20261107/ringpp/baseline_seeds_summary_ringpp_exactmax.json"
PREVIOUS_RINGPP_SHA256 = "551a60847fe8c0d38e37adc3b182033379cba4544b0a0826cd1aea74f52203d7"
OUT = STABLE / "manuscript_integration_20261108"
BASE_OUT = OUT / "baseline_seeds_summary_seqot_stable.json"
ADOPTED_OUT = OUT / "baseline_seeds_summary_seqot_stable_ringpp_exactmax.json"
FOLDERS = [(k, f"results/pdv2_seqot_stable_gem_20261108/server_mirror/records/s{k}/odom") for k in (1, 2, 3)]


def main() -> None:
    assert sha256(RECORDS) == RECORDS_SHA256 and sha256(COMMON) == COMMON_SHA256
    assert sha256(PREVIOUS_RINGPP) == PREVIOUS_RINGPP_SHA256
    records = json.loads(RECORDS.read_text())
    assert records["label"] == LABEL
    common = json.loads(COMMON.read_text())
    sources = {}

    def track(path: Path) -> Path:
        sources[str(path.relative_to(REPO))] = sha256(path)
        return path

    track(Path(__file__))
    per_seed = {}
    for seed, folder in FOLDERS:
        entry = aggregate(folder, "dump_seqotl2", "score_l2", "pr_summary_baselines.json", "seqotl2", track)
        assert entry is not None, folder
        for k in METRICS:
            assert abs(entry["macro"][k] - records["per_seed"][str(seed)]["macro"][k]) < 1e-12, (seed, k)
        per_seed[str(seed)] = {"source": entry["source"], "macro": entry["macro"]}
    ledgers = [p for p in sources if "/dump_seqotl2_" in p]
    assert len(ledgers) == 138
    for p in ledgers:   # 원장 점수 유한성
        with np.load(REPO / p) as z:
            assert np.all(np.isfinite(z["score_l2"])), p
    vals = np.array([[r["macro"][k] for k in METRICS] for r in per_seed.values()])
    stats = {k: {"mean": float(vals[:, i].mean()), "std": float(vals[:, i].std(ddof=1)),
                 "values": vals[:, i].tolist()} for i, k in enumerate(METRICS)}
    for k in METRICS:
        assert abs(stats[k]["mean"] - records["mean_std"][k]["mean"]) < 1e-12, k
        assert abs(stats[k]["std"] - records["mean_std"][k]["std"]) < 1e-12, k

    base = dict(common)
    base["methods"] = dict(common["methods"])
    base["methods"]["seqot"] = {"score": "score_l2", "training_free": False, "seeds": ["1", "2", "3"],
                                "per_seed": per_seed, "mean_std": stats, "recipe": LABEL,
                                "selected_epochs": records["selected_epochs"]}
    # 원천 해시 목록에서 이전 SeqOT(공개 코드 학습법) 원장을 빼고 채택 원장을 넣는다.
    base["input_sha256"] = {k: v for k, v in common["input_sha256"].items()
                            if not k.startswith("results/pdv2_seqot_official_centered")}
    base["input_sha256"].update(sources)
    base["seqot_replacement"] = {
        "label": LABEL, "decided": "2026-11-08",
        "authorization": "OVN coordination.json seqot_stable_integration_authorization",
        "base_summary": str(COMMON.relative_to(REPO)), "base_summary_sha256": COMMON_SHA256,
        "records_summary": str(RECORDS.relative_to(REPO)), "records_summary_sha256": RECORDS_SHA256,
        "previous_seqot_mean_std": common["methods"]["seqot"]["mean_std"],
        "previous_seqot_recipe": "released-code recipe (99+99 epochs), results/pdv2_seqot_official_centered{,_s2,_s3}/odom",
        "seqot_inputs_sha256": sources, "source_sha256": sha256(Path(__file__))}
    assert all(base["methods"][m] == common["methods"][m] for m in common["methods"] if m != "seqot")
    OUT.mkdir(parents=True, exist_ok=True)
    BASE_OUT.write_text(json.dumps(base, indent=1, ensure_ascii=False) + "\n")

    # RING++ 행만 정확max로 바꾼 사본을 기존 생성기로 다시 만든다(출력 경로만 바꾼다).
    import _ringpp_exactmax_seeds_summary_20261107 as ringpp
    ringpp.BASE, ringpp.OUT = BASE_OUT, ADOPTED_OUT
    ringpp.main()
    adopted, previous = json.loads(ADOPTED_OUT.read_text()), json.loads(PREVIOUS_RINGPP.read_text())
    assert adopted["methods"].keys() == previous["methods"].keys()
    assert all(adopted["methods"][m] == previous["methods"][m] for m in previous["methods"] if m != "seqot")
    assert adopted["methods"]["seqot"] == base["methods"]["seqot"]
    assert adopted["ringpp_replacement"]["ringpp_inputs_sha256"] == previous["ringpp_replacement"]["ringpp_inputs_sha256"]
    report = {"label": LABEL, "seqot_ledgers": len(ledgers), "finite": True,
              "seqot_mean_std_pct": {k: [100 * stats[k]["mean"], 100 * stats[k]["std"]] for k in METRICS},
              "other_methods_unchanged_vs_previous_adopted": sorted(m for m in previous["methods"] if m != "seqot"),
              "outputs_sha256": {str(BASE_OUT.relative_to(REPO)): sha256(BASE_OUT),
                                 str(ADOPTED_OUT.relative_to(REPO)): sha256(ADOPTED_OUT)}}
    (OUT / "adoption_check.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
