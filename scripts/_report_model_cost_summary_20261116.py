#!/usr/bin/env python3
"""보고모델 비용 측정 파일(results/pdv2_report_model_20261116/cost/)을 모아 summary.json을 쓴다. 새 계산 없음.

측정 스크립트(_report_model_cost_20261116.py)는 측정 프로토콜에 자기 해시가 고정되어 있어 요약 코드를 따로 둔다.
docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_report_model_cost_summary_20261116.py
"""
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import _report_model_cost_20261116 as R  # noqa: E402

O, bench, OUT, ROOT, TAG, CHECKPOINT = R.O, R.bench, R.OUT, R.ROOT, R.TAG, R.CHECKPOINT

# 주 결과표의 비교 방법 비용(KB/frame, 갱신, 질의, 전체 ms). 순위 확인용.
OTHERS = {"SC++": (31.43, 14.81, 26.47, 41.28), "FreSCo": (20.29, 10.83, 29.70, 40.54), "RING": (115.72, 7.07, 26.45, 33.52),
          "RING++": (694.33, 60.31, 141.58, 201.88), "DiSCO": (11.42, 3.33, 1.22, 4.56), "DeepRING": (118.89, 9.64, 25.95, 35.59),
          "RING#-L": (278.40, 51.54, 142.56, 194.10), "HeLiOS": (36.63, 326.57, 10.13, 336.70), "SeqOT": (6.70, 6.69, 0.12, 6.81),
          "BEVPlace++": (34.30, 44.99, 8.68, 53.67)}
BEV_TOTAL_MS, BEV_RESIDENT = 53.673, 346.306e6


def summarize() -> None:
    """측정 파일을 모아 summary.json을 쓴다(새 계산 없음)."""
    def row(path: Path) -> dict:
        d = bench.read(path)
        t, r = d["timings_ms"], d["resident_bytes"]
        n = d["n_map"]
        return dict(path=str(path.relative_to(ROOT)), sha256=bench.sha(path), gate=d.get("gate", "previous_model"), n_map=n,
                    encode_ms=t["encode"]["mean"], state_update_ms=t["state_update"]["mean"], update_ms=t["update"]["mean"],
                    query_ms=t["query"]["mean"], processing_ms=t["processing"]["mean"], processing_p95_ms=t["processing"]["p95"],
                    resident_bytes=r["total"], resident_mb=r["total"] / 1e6, kb_per_frame=r["total"] / ((n + 95) * 1000),
                    model_registered_cpu_bytes=r["model_numeric_audit"]["registered_cpu_bytes"],
                    ledger_winners_exact=d["verification"]["final_ledger"]["winners_exact"],
                    ledger_score_max_error=d["verification"]["final_ledger"]["score_max_error"])
    main = {"skipped": row(OUT / "nsd_n10000_window.json"), "computed": row(OUT / "gate_computed/nsd_n10000_window.json")}
    reps = {g: [row(OUT / f"replicate/{g}_r{k}/nsd_n10000_window.json") for k in (1, 2, 3)] for g in ("skipped", "computed")}
    maps = {g: {n: row(OUT / f"{folder}/nsd_n{n}_window.json") for n in (1000, 5000, 10000)}
            for g, folder in (("skipped", "mapsize"), ("computed", "mapsize_gate_computed"))}
    previous = bench.read(O.OUT / "nsd_n10000_window.json")
    cont = row(OUT / "continuity_prev_model/nsd_n10000_window.json")
    ranks = {}
    for g, m in main.items():
        nsd = (m["kb_per_frame"], m["update_ms"], m["query_ms"], m["processing_ms"])
        table = dict(OTHERS, NSD=nsd)
        ranks[g] = {col: sorted(table, key=lambda k: table[k][i])[:3] for i, col in enumerate(("kb_per_frame", "update", "query", "total"))}
    out = dict(
        schema="report-model-cost-summary-v1", model=TAG, checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpoint_sha256=bench.sha(CHECKPOINT),
        recommended="skipped", recommended_reason="보고모델은 eta=-0.5 상수라 게이트 MLP가 없는 구조로 서술된다. 두 변형의 승자·점수는 비트 단위로 같다.",
        main=main, replicate_sessions=reps, mapsize=maps,
        mapsize_update_mean_min_max={g: [sum(v[n]["update_ms"] for n in v) / 3, min(v[n]["update_ms"] for n in v), max(v[n]["update_ms"] for n in v)] for g, v in maps.items()},
        continuity=dict(previous_published=dict(path=str((O.OUT / "nsd_n10000_window.json").relative_to(ROOT)),
                                                processing_ms=previous["timings_ms"]["processing"]["mean"],
                                                update_ms=previous["timings_ms"]["update"]["mean"], query_ms=previous["timings_ms"]["query"]["mean"],
                                                resident_bytes=previous["resident_bytes"]["total"]),
                        remeasured=cont, processing_ratio=cont["processing_ms"] / previous["timings_ms"]["processing"]["mean"]),
        bevplace=dict(total_ms=BEV_TOTAL_MS, resident_bytes=BEV_RESIDENT,
                      speedup={g: BEV_TOTAL_MS / m["processing_ms"] for g, m in main.items()},
                      resident_reduction={g: 1 - m["resident_bytes"] / BEV_RESIDENT for g, m in main.items()}),
        main_table_top3=ranks, others_from="docs/paper/body_v2.4/tables_v3/table2_results.tex",
        files_note="variant=window는 표 생성기 호환을 위한 구현 이름이다. gate 필드가 게이트 처리(skipped/computed)를 구분한다. "
                   "verification.published_measurement는 새 모델에 이전 공개 측정이 없어 새 원장 대조를 다시 적은 호환 필드다. "
                   "protocol_window.json의 checkpoint.checkpoint_sha256은 새 체크포인트 해시다(생성기의 diff_s1 해시 대조는 생성기 쪽에서 바꿔야 한다).")
    bench.write(OUT / "summary.json", out)
    print(json.dumps({k: out[k] for k in ("main_table_top3", "bevplace", "mapsize_update_mean_min_max")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    summarize()
