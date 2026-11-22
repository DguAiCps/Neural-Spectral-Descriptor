#!/usr/bin/env python3
"""표2 학습 비교 방법의 검증9 곡선·선택 에폭 재집계(새 학습·추론 없음, 기존 선택 기록만 읽는다).

에폭 표기는 '끝난 학습 에폭 수'로 맞춘다(0 = 학습 전). 원 기록의 표기:
  BEVPlace++·DiSCO: 0부터 센 에폭 번호, 그 에폭 학습 뒤 검증 → +1
  RING#-L: e1..e30 / DeepRING: e00(학습 전)..e20 / HeLiOS: 저장 파일 에폭(10단위 + 공식 best)
  SeqOT: 단계1·단계2 각각 1..20 / ImLPR(표2 아님): at_start_of_epoch − 1
검증 지표는 각 방법 선택 기록의 검증9 계열 평균 5 m R@1이다(RING#-L은 질의 4개마다 1개).
출력: results/pdv2_learning_curves_split_v2_20261118/{summary.json,table.md}
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_learning_curves_split_v2_20261118"
SUMMARY = REPO / ("results/pdv2_deepring_scaleinit_integration_20261110/"
                  "baseline_seeds_summary_seqot_stable_ringpp_i2_fresco_kgf_deepring_scaleinit.json")
DEEPRING = Path("/mnt/d/NSD_local/results/pdv2_overnight_repro_20261108/deepring_scale_3seed_20261110/result.json")
SFX = {"1": "", "2": "_s2", "3": "_s3"}
SRC = {}


def load(p: Path) -> dict:
    p = Path(p)
    SRC[str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)] = \
        hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def run(curve, selected, source, note=""):
    """curve: [(끝난 에폭 수, 검증 R@1)] 정렬. selected: 끝난 에폭 수(표2 체크포인트)."""
    curve = sorted(curve)
    vals = dict(curve)
    trained = [(e, v) for e, v in curve if e > 0]
    first_e, first_v = trained[0]
    best_e, best_v = max(trained, key=lambda x: (x[1], -x[0]))
    last_e, last_v = trained[-1]
    r = {"source": source, "curve": [[e, round(v, 6)] for e, v in curve],
         "n_evaluated": len(trained), "max_epoch": last_e,
         "selected_epoch": selected, "selected_val_r1": vals[selected],
         "best_epoch_in_curve": best_e, "selected_is_curve_best": abs(vals[selected] - best_v) < 1e-12,
         "first_eval_epoch": first_e, "first_eval_val_r1": first_v,
         "gain_best_minus_first_pp": 100 * (best_v - first_v),
         "last_val_r1": last_v, "drop_best_minus_last_pp": 100 * (best_v - last_v),
         "peak_at_first_eval": best_e == first_e,
         "selected_fraction_of_budget": selected / last_e}
    if 0 in vals:
        r["init_val_r1"] = vals[0]
        r["gain_best_minus_init_pp"] = 100 * (best_v - vals[0])
    if note:
        r["note"] = note
    return r


def bevplace_disco(method, folder):
    out = {}
    for s, x in SFX.items():
        p = REPO / f"results/{folder}{x}/train_selection.json"
        d = load(p)
        curve = [(e["epoch"] + 1, e["family_macro_r1"]) for e in d["epochs"]]
        sel = (d["best"]["epoch"] if method == "bevplace" else d[d["chosen"]]["epoch"]) + 1
        out[s] = run(curve, sel, str(p.relative_to(REPO)))
    return out


def ringsharp():
    out = {}
    for s, x in SFX.items():
        p = REPO / f"results/pdv2_ring_official{x}/ringsharp_ckpt_selection.json"
        d = load(p)
        curve = [(int(k[1:]), v["family_macro_r1"]) for k, v in d["epochs"].items()]
        out[s] = run(curve, int(d["selected_epoch"][1:]), str(p.relative_to(REPO)),
                     "every 4th validation query (val_query_stride 4)")
    return out


def deepring():
    d = load(DEEPRING)
    fixed = load(REPO / "results/pdv2_deepring_scaleinit_integration_20261110/deepring/test46/fixed_selection.json")
    out = {}
    for s in ("1", "3", "4"):
        vals = d["per_seed"][s]["values"]
        curve = [(int(k.split("_e")[1]), v["R1_pct"] / 100) for k, v in vals.items()
                 if k.startswith(f"S{s}_e")]
        out[s] = run(curve, fixed["seeds"][s]["epoch"], str(DEEPRING),
                     "values rounded to 0.001 % in the source; e0 = before training (scale-init W0)")
    return out


def helios():
    out = {}
    for s, x in SFX.items():
        p = REPO / f"results/pdv2_helios_official{x}/train_selection.json"
        d = load(p)
        pts = {}
        for c in d["candidates"].values():
            pts.setdefault(c["epoch"], c["family_macro_r1"])
        rec = load(REPO / f"results/pdv2_helios_official{x}/train_record.json")
        ap = max(rec["epochs"], key=lambda e: (e["val_ap"], -e["epoch"]))
        r = run(list(pts.items()), d[d["chosen"]]["epoch"], str(p.relative_to(REPO)),
                "R@1 only at epochs 10,20,..,80 and the official-best epoch (80-epoch budget)")
        r["official_val_ap_best_epoch"] = ap["epoch"]
        out[s] = r
    return out


def seqot():
    out = {}
    for s in SFX:
        p1 = REPO / f"results/pdv2_baseline_revision_20261107/seqot/s{s}/train_selection.json"
        p2 = REPO / f"results/pdv2_seqot_stable_gem_20261108/server_mirror/records/s{s}/train_selection.json"
        d1, d2 = load(p1)["stage1"], load(p2)["stage2"]
        out[s] = {f"stage{k}": run([(int(e), v["family_r1_l2"]) for e, v in d["epochs"].items()],
                                   d["chosen"]["epoch"], str(p.relative_to(REPO)))
                  for k, d, p in ((1, d1, p1), (2, d2, p2))}
    return out


def imlpr():
    out = {}
    for s, x in SFX.items():
        p = REPO / f"results/pdv2_imlpr_official{x}/train_selection.json"
        d = load(p)
        curve = [(e["at_start_of_epoch"] - 1, e["family_macro_r1"]) for e in d["evals"]]
        out[s] = run(curve, d["best"]["at_start_of_epoch"] - 1, str(p.relative_to(REPO)),
                     "not in Table 2 (ImLPR removed from the comparison set)")
    return out


def main():
    summ = load(SUMMARY)["methods"]
    learned = [m for m, v in summ.items() if not v["training_free"]]
    methods = {"bevplace": bevplace_disco("bevplace", "pdv2_bevplace_official"),
               "ringsharp": ringsharp(), "deepring": deepring(), "helios": helios(),
               "seqot": seqot(), "disco": bevplace_disco("disco", "pdv2_disco_official"),
               "imlpr": imlpr()}
    assert sorted(methods) == sorted(learned), (sorted(methods), learned)
    for m, runs in methods.items():   # 시드 목록이 표2 요약과 같은지
        assert sorted(runs) == sorted(summ[m]["seeds"]), (m, sorted(runs), summ[m]["seeds"])
    rows = []
    for m, runs in methods.items():
        for s, r in runs.items():
            for stage, x in ((r.items() if "stage1" in r else [("", r)])):
                rows.append((m + (f" {stage}" if stage else ""), s, x))
    lines = ["| 방법 | 시드 | 평가 에폭 | 선택 에폭 / 최대 | 첫 평가 R@1 | 최고 R@1 | 최고−첫 (%p) | 최고−마지막 (%p) | 첫 평가가 최고 |",
             "|---|---|---|---|---:|---:|---:|---:|---|"]
    for name, s, x in rows:
        init = f" (학습 전 {100 * x['init_val_r1']:.2f})" if "init_val_r1" in x else ""
        lines.append(f"| {name} | {s} | {x['n_evaluated']} | {x['selected_epoch']} / {x['max_epoch']} | "
                     f"{100 * x['first_eval_val_r1']:.2f}{init} | {100 * x['selected_val_r1']:.2f} | "
                     f"{x['gain_best_minus_first_pp']:.2f} | {x['drop_best_minus_last_pp']:.2f} | "
                     f"{'예' if x['peak_at_first_eval'] else '아니오'} |")
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"scope": "표2 학습 비교 방법(요약 json의 training_free=false 전부, ImLPR는 표2 밖)의 검증9 곡선 재집계; 새 실행 없음",
           "epoch_convention": "끝난 학습 에폭 수(0 = 학습 전). 원 표기 변환은 스크립트 머리말",
           "metric": "각 선택 기록의 검증9 계열 평균 5 m R@1(선택에 쓴 값)",
           "methods": methods, "input_sha256": SRC}
    (OUT / "summary.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
