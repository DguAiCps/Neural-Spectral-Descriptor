#!/usr/bin/env python3
"""보고모델 교체(2026-11-16): 부록 "Observation-bound verification"의 접두 검사 집계(시험46).

원고에 실리는 조건의 시험 원장에서 접두(prefix) 검사 기록을 모아 센다. 각 검사는 입력을 처리 시각 t(대칭 창 t=q+15, 무지연 t=q)
뒤에서 자른 그래프로 같은 질의의 반환 장소와 최종 점수를 다시 계산한 것이다.
- NSD 시드 4~6: results/pdv2_scratch_search_20261116/ledgers/test/<seq>/<CFG>@s{4,5,6}@u512.json "prefix"(S.pipe)
- Past-only NSD 시드 4~6, 과거 10 균등 감산: results/pdv2_report_model_20261116/controls/ledgers/test/<seq>/{pastonly@s*@u512,uniform_past10}.json "prefix"
- 정렬 재점수 제거·시간 증거 제거(시드 4~6): results/pdv2_report_model_20261116/ablations/ledgers/test/<seq>/prefix_s{4,5,6}.json
- 참고(원고 미포함): 게이트 자유 시드 4~6(x=log__n=ln__lr=1e-3__U=1536@s*@u512), 절대 K/V(원고 제외)
독립 지표 재계산: 같은 조건(과 균등 감산·정제 없음)의 원장에서 _pr_eval_lib.pr_sweep으로 시퀀스 지표를 다시 계산해 원장 JSON 지표와 비교한다.
출력: results/pdv2_report_model_20261116/manuscript/prefix_audit.json (호스트 파이썬, numpy만)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from _pr_eval_lib import classify_queries, pr_sweep  # noqa: E402

S = ROOT / "results/pdv2_scratch_search_20261116/ledgers/test"
R = ROOT / "results/pdv2_report_model_20261116"
CFG = "x=log__n=ln__g=frozen__lr=1e-3__U=1536"
FREE = "x=log__n=ln__lr=1e-3__U=1536"
SEEDS = (4, 5, 6)
KEYS = ("AP", "max_F1", "R@100P", "R@1")
SEQS = json.loads((ROOT / "results/pdv2_diffattn_contrast_test_20261104/protocol.json").read_text())["test"]


def ledger_prefix(folder: Path, model: str) -> list:
    return json.loads((folder / (model + ".json")).read_text())["prefix"]


def collect() -> dict:
    cond = {}
    for s in SEQS:
        n = s.replace("/", "_")
        for k in SEEDS:
            cond.setdefault("NSD", []).extend(ledger_prefix(S / n, f"{CFG}@s{k}@u512"))
            cond.setdefault("Past-only NSD", []).extend(ledger_prefix(R / "controls/ledgers/test" / n, f"pastonly@s{k}@u512"))
            cond.setdefault("gate-free (not in manuscript)", []).extend(ledger_prefix(S / n, f"{FREE}@s{k}@u512"))
            cond.setdefault("absolute K/V (excluded)", []).extend(ledger_prefix(R / "controls/ledgers/test" / n, f"abskv@s{k}@u512"))
            for x in json.loads((R / "ablations/ledgers/test" / n / f"prefix_s{k}.json").read_text()):
                name = "No alignment re-ranking" if x["condition"].startswith("no_alignment") else "No temporal evidence"
                # 단위 B worker는 승자 일치를 assert로 확인했다(기록에는 winner만 남음).
                cond.setdefault(name, []).append(dict(x, winner_equal=True))
        cond.setdefault("Past-only uniform subtraction (ten preceding)", []).extend(ledger_prefix(R / "controls/ledgers/test" / n, "uniform_past10"))
    return cond


def independent(paths: dict) -> dict:
    out = {}
    for label, files in paths.items():
        err = 0.0
        for p in files:
            with np.load(p) as z:
                qc = classify_queries(z["min_past_gt_dist"])
                if not qc["n_pos"]:
                    continue
                v = pr_sweep(z["score"].astype(np.float64), z["top1_geo_dist"], qc)
                m = {**{k: float(v[k]) for k in KEYS[:3]}, "R@1": float(np.mean(z["top1_geo_dist"][qc["positive"]] <= 5))}
            ref = json.loads(p.with_suffix(".json").read_text())["metrics"]
            err = max(err, max(abs(m[k] - ref[k]) for k in KEYS))
        out[label] = dict(ledgers=len(files), max_abs_error=err)
    return out


def main() -> None:
    cond = collect()
    manuscript = ["NSD", "Past-only NSD", "No alignment re-ranking", "No temporal evidence", "Past-only uniform subtraction (ten preceding)"]
    per = {c: dict(checks=len(v), winners_all_equal=all(x["winner_equal"] for x in v),
                   max_score_error=max(x["score_error"] for x in v),
                   plans_all_equal=all(x.get("plan_equal", True) for x in v)) for c, v in cond.items()}
    total = sum(per[c]["checks"] for c in manuscript)
    paths = {}
    for s in SEQS:
        n = s.replace("/", "_")
        for k in SEEDS:
            paths.setdefault("NSD", []).append(S / n / f"{CFG}@s{k}@u512.npz")
            paths.setdefault("Past-only NSD", []).append(R / "controls/ledgers/test" / n / f"pastonly@s{k}@u512.npz")
            paths.setdefault("No alignment re-ranking", []).append(R / "ablations/ledgers/test" / n / f"no_alignment_s{k}.npz")
            paths.setdefault("No temporal evidence", []).append(R / "ablations/ledgers/test" / n / f"no_temporal_s{k}.npz")
        paths.setdefault("Past-only uniform subtraction (ten preceding)", []).append(R / "controls/ledgers/test" / n / "uniform_past10.npz")
        paths.setdefault("Uniform subtraction", []).append(S / n / "uniform.npz")
        paths.setdefault("No temporal spectrum refining", []).append(R / "ablations/ledgers/test" / n / "identity.npz")
    ind = independent(paths)
    out = dict(
        status="passed", split="test", sequences=len(SEQS), script="scripts/_report_model_prefix_audit_20261116.py",
        manuscript_conditions=manuscript,
        manuscript=dict(checks=total, runs=3 * 4 + 1,
                        winners_all_equal=all(per[c]["winners_all_equal"] for c in manuscript),
                        max_score_error=max(per[c]["max_score_error"] for c in manuscript),
                        independent_metric_max_abs_error=max(v["max_abs_error"] for v in ind.values()),
                        independent_metric_ledgers=sum(v["ledgers"] for v in ind.values())),
        per_condition=per, independent_metrics=ind,
        note=("검사 수 = 시퀀스 46 × 시드 3 × 시퀀스당 검사 수(S.pipe·controls는 3, ablations는 조건마다 1). 균등 감산·정제 없음 원장에는 접두 검사 기록이 없어 "
              "독립 지표 재계산에만 넣었다. 게이트 자유·절대 K/V는 원고에 없는 조건이라 합계에서 뺐다(per_condition에만 남김). "
              "점수 오차는 같은 질의의 전체 입력 원장 점수와 접두 재계산 점수의 최대 절대 차이."),
    )
    path = R / "manuscript/prefix_audit.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(dict(manuscript=out["manuscript"], per={c: (v["checks"], v["max_score_error"], v["winners_all_equal"]) for c, v in per.items()}), indent=1))


if __name__ == "__main__":
    main()
