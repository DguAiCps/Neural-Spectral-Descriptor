#!/usr/bin/env python3
"""보고모델 교체(2026-11-16) 단위 C 항목 10: 게이트 학습 대조(부록용) 표 자료.

원천(새 학습·추론 없음):
  η 분포   results/pdv2_scratch_search_20261116/server/eta.json (시험46·검증9 전 스캔, 층별)
  시험 지표 results/pdv2_scratch_search_20261116/server/ledgers/test/<seq>/<model>.json (서버 원장, 시드 1~3)
           보고모델 시드 4~6은 로컬 원장(ledgers/test, npz에서 재계산과 같은 json metrics)
짝지은 구간은 시퀀스별 지표(json metrics)로 기존 규약(도시 안 시퀀스 재표집 10,000회, seed 20261104).
균등 감산 기준은 각 원장과 같은 기기의 uniform(서버 uniform = 로컬 uniform, 일지: 재선택 균등 감산은 서버·로컬 비트 동일).
출력: results/pdv2_report_model_20261116/analysis/gate/gate_table.json

단위 G 추가(2026-11-17): 게이트 자유 구성 시드 4~6(서버 학습, 로컬 원장·로컬 eta.json)을 주 수치 행으로 넣는다.
  rows_main  시드 4~6 로컬 원장: 보고모델(게이트 고정), 균등 감산, 게이트 자유 구성(LOCAL_ROWS).
             자유−고정(보고모델), 자유−균등 짝지은 구간, η 분포(results/pdv2_scratch_search_20261116/eta.json),
             q+15 접두 검사 요약, 평가 모듈을 거치지 않은 독립 재계산(_pdv2_scratch_search_verify_20261116.metrics)과의 최대 오차.
  rows       기존 시드 1~3 서버 원장 행(seed_role="구성 탐색 시드"). 내용은 그대로다.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _report_model_common_20261116 as C  # noqa: E402

S = C.SEARCH
ETA = json.loads((S / "server/eta.json").read_text())
ETA_LOCAL = json.loads((S / "eta.json").read_text()) if (S / "eta.json").exists() else {}
LOCAL_ROWS = [  # 시드 4~6 로컬 원장 (표시 이름, 모델 접두, 갱신, 게이트, 같은 설정의 게이트 고정 대응)
    ("log1p + LN, lr 1e-3, 512 (gate learned)", "x=log__n=ln__lr=1e-3__U=1536", 512, "free", C.CFG),
    ("log1p-std + BN-est, lr 3e-4, 1536 (gate learned)", "x=logstd__n=bnest__U=1536", 1536, "free", None),
]
ROWS = [  # (표시 이름, 모델 접두, 갱신, 게이트, 같은 설정의 게이트 고정 대응)
    ("log1p + LN, lr 1e-3, 512 (reported setting)", "x=log__n=ln__g=frozen__lr=1e-3__U=1536", 512, "frozen", None),
    ("log1p + LN, lr 1e-3, 512", "x=log__n=ln__lr=1e-3__U=1536", 512, "free", "x=log__n=ln__g=frozen__lr=1e-3__U=1536"),
    ("log1p + LN, lr 3e-4, 512", "x=log__n=ln__U=1536", 512, "free", "x=log__n=ln__g=frozen__U=1536"),
    ("log1p + LN, lr 1e-3, 512", "x=log__n=ln__g=weak__lr=1e-3__sv=fine", 512, "weaken-only", "x=log__n=ln__g=frozen__lr=1e-3__U=1536"),
    ("log1p-std + BN-est, lr 1e-3, 1536", "x=logstd__n=bnest__lr=1e-3__U=1536", 1536, "free", "x=logstd__n=bnest__g=frozen__lr=1e-3__U=1536"),
    ("log1p-std + BN-est, lr 3e-4, 1536", "x=logstd__n=bnest__U=1536", 1536, "free", None),
    ("standardized raw + BN0, lr 1e-3, 1536", "x=lin__lr=1e-3__U=1536", 1536, "free", None),
]


def seq_metrics(model, folder):
    out = {}
    for s in C.SEQS:
        p = folder / s.replace("/", "_") / (model + ".json")
        m = json.loads(p.read_text())["metrics"]
        out[s] = None if not m["n_pos"] else {k: m[k] for k in C.METRICS}
    return out


def runs(prefix, u, seeds, folder):
    return [seq_metrics(f"{prefix}@s{sd}@u{u}", folder) for sd in seeds]


def eta_row(prefix, u, seeds, src=ETA):
    keys = [f"{prefix}@s{sd}@u{u}" for sd in seeds]
    if not all(k in src for k in keys):
        return None
    t = [src[k]["test"] for k in keys]
    out = dict(mean_by_seed=[v["mean"] for v in t], stronger_than_half_by_seed=[v["frac_stronger_than_half"] for v in t],
               q01_by_seed=[v["quantiles"]["0.01"] for v in t], q50_by_seed=[v["quantiles"]["0.5"] for v in t],
               q99_by_seed=[v["quantiles"]["0.99"] for v in t])
    if src is not ETA:  # 단위 G 행: 약한 쪽 비율과 10·90% 분위, 시퀀스 평균 범위도 싣는다
        out.update(weaker_than_half_by_seed=[v["frac_weaker_than_half"] for v in t],
                   q10_by_seed=[v["quantiles"]["0.1"] for v in t], q90_by_seed=[v["quantiles"]["0.9"] for v in t],
                   sequence_mean_min_by_seed=[v["sequence_mean_min"] for v in t],
                   sequence_mean_max_by_seed=[v["sequence_mean_max"] for v in t],
                   seed_mean={k: [float(np.mean([x[i] for x in out[k + "_by_seed"]])) for i in (0, 1)]
                              for k in ("mean", "stronger_than_half", "q01", "q50", "q99")})
    return out


def val_macro(model):
    """검증9 실제 검색 매크로(데이터셋 묶음 평균, 탐색 스크립트 agg와 같은 정의; 원장 json metrics)."""
    fam, files = {}, sorted((S / "ledgers/val").glob("*/" + model + ".json"))
    assert len(files) == 9, model
    for p in files:
        m = json.loads(p.read_text())["metrics"]
        if m["n_pos"]:
            fam.setdefault(p.parent.name.split("_")[0], []).append(m)
    return {k: float(np.mean([np.mean([x[k] for x in f]) for f in fam.values()])) for k in C.METRICS}


def checks(prefix, u, seeds, folder):
    """q+15 접두 검사 요약과 독립 재계산(npz → 단순 임계값 훑기)과 json metrics의 최대 차."""
    import _pdv2_scratch_search_verify_20261116 as V
    pre, err = [], 0.0
    for sd in seeds:
        for s in C.SEQS:
            base = folder / s.replace("/", "_") / f"{prefix}@s{sd}@u{u}"
            rec = json.loads(base.with_suffix(".json").read_text())
            pre += rec["prefix"]
            mine = V.metrics(dict(np.load(base.with_suffix(".npz"))))
            if mine is not None:
                err = max(err, max(abs(mine[k] - rec["metrics"][k]) for k in C.METRICS))
    return dict(prefix_n=len(pre), prefix_all_winner_equal=all(x["winner_equal"] for x in pre),
                prefix_max_score_error=max(x["score_error"] for x in pre) if pre else None,
                prefix_max_output_error=max(x["output_max_error"] for x in pre) if pre else None,
                independent_max_abs_error=err)


def main():
    server = S / "server/ledgers/test"
    local = S / "ledgers/test"
    uni_server = [seq_metrics("uniform", server)]
    uni_local = [seq_metrics("uniform", local)]
    same = max(abs(uni_server[0][s][k] - uni_local[0][s][k]) for s in C.SEQS if uni_server[0][s] for k in C.METRICS)
    table = []
    for label, prefix, u, gate, frozen in ROWS:
        r = runs(prefix, u, (1, 2, 3), server)
        mac = [C.macro(x)["macro"] for x in r]
        row = dict(label=label, model=f"{prefix}@u{u}", gate=gate, seeds=[1, 2, 3], ledgers="server", seed_role="구성 탐색 시드",
                   test={k: C.mean_sd([m[k] for m in mac]) for k in C.METRICS},
                   minus_uniform=C.paired(r, uni_server), eta=eta_row(prefix, u, (1, 2, 3)))
        if frozen:
            row["minus_frozen_same_setting"] = dict(frozen=f"{frozen}@u{u}", **C.paired(r, runs(frozen, u, (1, 2, 3), server)))
        table.append(row)
    # 보고모델 시드 4~6(로컬)
    r = runs(ROWS[0][1], 512, C.NEW_SEEDS, local)
    mac = [C.macro(x)["macro"] for x in r]
    table.insert(1, dict(label="log1p + LN, lr 1e-3, 512 (reported model, seeds 4-6)", model=f"{ROWS[0][1]}@u512", gate="frozen",
                         seeds=list(C.NEW_SEEDS), ledgers="local", test={k: C.mean_sd([m[k] for m in mac]) for k in C.METRICS},
                         minus_uniform=C.paired(r, uni_local), eta=None,
                         eta_note="일지: 여섯 시드 모두 전 스캔 η=−0.5(게이트 고정)"))
    table[1]["seed_role"] = "주 수치 시드"
    # 단위 G: 시드 4~6 로컬 원장 행
    frozen_r = r
    main_rows = [dict(table[1], label="log1p + LN, lr 1e-3, 512 (reported model, gate fixed)",
                      eta=eta_row(C.CFG, 512, C.NEW_SEEDS, ETA_LOCAL), checks=checks(C.CFG, 512, C.NEW_SEEDS, local),
                      val={k: C.mean_sd([val_macro(f"{C.CFG}@s{sd}@u512")[k] for sd in C.NEW_SEEDS]) for k in C.METRICS})]
    uni_mac = C.macro(uni_local[0])["macro"]
    main_rows.append(dict(label="uniform subtraction", model="uniform", gate="none (eta=-0.5, no network)", seeds=[], ledgers="local",
                          test={k: dict(mean=uni_mac[k], sd=None) for k in C.METRICS},
                          val={k: dict(mean=val_macro("uniform")[k], sd=None) for k in C.METRICS}))
    for label, prefix, u, gate, frozen in LOCAL_ROWS:
        rr = runs(prefix, u, C.NEW_SEEDS, local)
        mac = [C.macro(x)["macro"] for x in rr]
        row = dict(label=label, model=f"{prefix}@u{u}", gate=gate, seeds=list(C.NEW_SEEDS), ledgers="local", seed_role="주 수치 시드",
                   test={k: C.mean_sd([m[k] for m in mac]) for k in C.METRICS}, test_by_seed=mac,
                   minus_uniform=C.paired(rr, uni_local), minus_reported_model=dict(reported=f"{C.CFG}@u512", **C.paired(rr, frozen_r)),
                   eta=eta_row(prefix, u, C.NEW_SEEDS, ETA_LOCAL), checks=checks(prefix, u, C.NEW_SEEDS, local),
                   same_setting_as_reported_except_gate=frozen == C.CFG,
                   val={k: C.mean_sd([val_macro(f"{prefix}@s{sd}@u{u}")[k] for sd in C.NEW_SEEDS]) for k in C.METRICS})
        main_rows.append(row)
    out = dict(rows_main=main_rows, rows=table, uniform_server_vs_local_max_abs_err=same,
               note="test: 시험46 6도시 매크로, 3시드 평균±SD(분율). minus_*: %p, 시퀀스 재표집 95% 구간. eta: 시험46 전 스캔, [1층, 2층]. "
                    "stronger_than_half = η < −0.5(감산이 고정값보다 강함)인 스캔 비율. "
                    "rows_main: 주 수치 시드 4~6, 로컬 원장(서버 학습·로컬 평가), η는 로컬 eta.json. minus_reported_model = 자유 − 보고모델(게이트 고정) 시드 4~6. "
                    "rows: 구성 탐색 시드 1~3, 서버 원장(구성 선택에 쓴 시드라 선택 편향이 있다).")
    C.write(C.OUT / "gate" / "gate_table.json", out)
    for row in main_rows[2:]:
        mu, mf, e, ck = row["minus_uniform"], row["minus_reported_model"], row["eta"], row["checks"]
        print("[s4-6]", row["label"], {k: f"{100*row['test'][k]['mean']:.3f}±{100*row['test'][k]['sd']:.3f}" for k in C.METRICS})
        print("   -uni ", {k: f"{mu['delta_pp'][k]:+.3f}[{mu['conditional_95_pp'][k][0]:+.3f},{mu['conditional_95_pp'][k][1]:+.3f}]" for k in C.METRICS})
        print("   -rep ", {k: f"{mf['delta_pp'][k]:+.3f}[{mf['conditional_95_pp'][k][0]:+.3f},{mf['conditional_95_pp'][k][1]:+.3f}]" for k in C.METRICS})
        if e:
            print("   eta", {k: [round(x, 3) for x in v] for k, v in e["seed_mean"].items()})
        print("   checks", ck)
    print("[s4-6] reported", main_rows[0]["checks"], main_rows[0]["eta"] and main_rows[0]["eta"]["seed_mean"])
    for row in table:
        mu = row["minus_uniform"]
        line = f"{row['label'][:48]:48s} {row['gate']:11s} s{row['seeds']} AP {100*row['test']['AP']['mean']:.3f}±{100*row['test']['AP']['sd']:.3f}"
        line += " | -uni " + " ".join(f"{k} {mu['delta_pp'][k]:+.2f}[{mu['conditional_95_pp'][k][0]:+.2f},{mu['conditional_95_pp'][k][1]:+.2f}]" for k in ("AP", "max_F1", "R@100P"))
        if "minus_frozen_same_setting" in row:
            mf = row["minus_frozen_same_setting"]
            line += " | -frozen " + " ".join(f"{k} {mf['delta_pp'][k]:+.2f}[{mf['conditional_95_pp'][k][0]:+.2f},{mf['conditional_95_pp'][k][1]:+.2f}]" for k in ("AP", "R@100P"))
        print(line)
        e = row["eta"]
        if e:
            f = lambda xs, i: f"{min(x[i] for x in xs):.2f}~{max(x[i] for x in xs):.2f}"
            print(f"      eta mean L1 {f(e['mean_by_seed'],0)} L2 {f(e['mean_by_seed'],1)} | stronger L1 {f(e['stronger_than_half_by_seed'],0)} L2 {f(e['stronger_than_half_by_seed'],1)}"
                  f" | q01 L1 {f(e['q01_by_seed'],0)} | q50 L1 {f(e['q50_by_seed'],0)} L2 {f(e['q50_by_seed'],1)}")
    print("uniform server-local", same)


if __name__ == "__main__":
    main()
