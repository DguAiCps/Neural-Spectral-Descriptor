#!/usr/bin/env python3
"""고정 기준값 결과를 `_pr_eval_lib` 없이 다시 계산해 검산한다.

- 검증 원장만으로 τ 후보를 직접 나열해 세 운영점 τ(동점 규칙 포함)를 비트 단위로 재현한다.
- 시험 TP/FP/FN/TN·도시 평균 재현율·합산 정밀도를 직접 센다.
- freeze 읽기 추적·입력 해시·주표 시험 원장 일치·오수락 목록 개수를 확인한다.
결과는 results/pdv2_review_closure_20261107/threshold/verification.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_closure_20261107/threshold"
PIPE = REPO / "results/pdv2_diffattn_contrast_pipeline_20261104/val"
TEST_DIR = REPO / "results/pdv2_diffattn_contrast_test_20261104"
BEV = REPO / "results/pdv2_review_response_20261104/bev"
OFFICIAL = {1: "results/pdv2_bevplace_official/odom", 2: "results/pdv2_bevplace_official_s2/odom",
            3: "results/pdv2_bevplace_official_s3/odom"}
VAL = ["kitti_09", "kitti360_0004", "kitti360_0006", "helipr_Roundabout01", "helipr_Roundabout02",
       "helipr_Roundabout03", "mulran_Sejong01", "mulran_Sejong02", "mulran_Sejong03"]
FIELD = {"nsd": "score", "bevplace": "score_negl2"}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def path_of(method: str, role: str, seed: int, name: str) -> Path:
    if method == "nsd":
        return (PIPE if role == "val" else TEST_DIR / "test") / name / f"diff_s{seed}.npz"
    return BEV / role / f"s{seed}" / f"{name}.npz"


def labels(d: dict, field: str) -> tuple:
    """점수, 양성, 음성, 정답 반환 여부(직접 정의)."""
    m = d["min_past_gt_dist"]
    good = np.isfinite(d["top1_geo_dist"]) & (d["top1_geo_dist"] <= 5.0)
    return d[field].astype(np.float64), m <= 5.0, m > 25.0, good


def cells(score, pos, neg, good, tau) -> dict:
    acc = np.isfinite(score) & (score >= tau)
    return dict(TP=int((acc & pos & good).sum()), FP=int((acc & (neg | (pos & ~good))).sum()),
                FN=int((pos & ~acc).sum()), TN=int((neg & ~acc).sum()), n_pos=int(pos.sum()))


def choose_independent(score, pos, neg, good) -> dict:
    """후보 τ마다 누적 개수를 searchsorted로 세고 규칙대로 고른다."""
    elig = (pos | neg) & np.isfinite(score)
    s = score[elig]
    tp_s = np.sort(score[elig & pos & good])
    fp_s = np.sort(score[elig & (neg | (pos & ~good))])
    cand = np.r_[np.inf, np.unique(s)[::-1]]
    tp = len(tp_s) - np.searchsorted(tp_s, cand, side="left")
    fp = len(fp_s) - np.searchsorted(fp_s, cand, side="left")
    n_pos = int(pos.sum())
    rec = tp / n_pos
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-300), 0.0)
    out = {}
    for name, ok in (("P99", prec >= .99), ("P100", fp == 0)):
        idx = np.flatnonzero(ok)
        best = rec[idx].max()
        top = idx[rec[idx] == best]
        out[name] = float(cand[top[np.argmax(cand[top])]])   # 같은 재현율 중 가장 높은 τ
    top = np.flatnonzero(f1 == f1.max())
    out["F1"] = float(cand[top[np.argmax(cand[top])]])
    out["ties_at_tau"] = {k: int((s == v).sum()) for k, v in out.items()}
    return out


def main() -> None:
    protocol, freeze = json.loads((OUT / "protocol.json").read_text()), json.loads((OUT / "freeze.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    fa = json.loads((OUT / "false_accepts.json").read_text())
    main_summary = json.loads((TEST_DIR / "summary.json").read_text())
    baseline = json.loads((REPO / "results/pdv2_baseline_seeds/summary.json").read_text())
    city = {}
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from _pdv2_pilot import CITY, TEST
    for c, names in CITY.items():
        for s in names:
            city[s] = c
    checks, details = {}, {}

    # 1. 읽기 추적·순서·해시
    trace = freeze["read_trace"]
    checks["freeze_trace_has_no_test_file"] = not [p for p in trace if "/test/" in p or "_test_" in p] and freeze["test_files_opened"] == 0
    checks["freeze_trace_val_ledgers_complete"] = all(
        str(path_of(m, "val", s, n).relative_to(REPO)) in trace for m in FIELD for s in (1, 2, 3) for n in VAL)
    checks["protocol_hash_in_freeze_and_summary"] = freeze["protocol_sha256"] == sha(OUT / "protocol.json") == summary["protocol_sha256"]
    checks["freeze_hash_in_summary_and_false_accepts"] = summary["freeze_sha256"] == sha(OUT / "freeze.json") == fa["freeze_sha256"]
    checks["freeze_before_summary_mtime"] = freeze["created_unix"] < (OUT / "summary.json").stat().st_mtime
    checks["protocol_before_freeze_mtime"] = (OUT / "protocol.json").stat().st_mtime < freeze["created_unix"]
    checks["validation_ledger_hashes"] = all(sha(REPO / p) == h for p, h in freeze["validation_ledgers_sha256"].items())
    checks["code_hashes_unchanged"] = all(sha(REPO / p) == h for p, h in freeze["code_sha256"].items())
    checks["test_ledger_hashes"] = all(sha(REPO / p) == h for p, h in summary["test_ledgers_sha256"].items())
    checks["checkpoint_hashes"] = all(
        sha(REPO / p) == freeze["checkpoints"]["checkpoint_sha256"][c] for c, p in protocol["methods"]["nsd"]["checkpoints"].items())
    bev_protocol = json.loads((BEV / "protocol.json").read_text())
    checks["bev_checkpoint_hashes"] = all(
        sha(REPO / bev_protocol["checkpoints"][s]["path"]) == freeze["checkpoints"]["bev_checkpoint_sha256"][s] for s in ("1", "2", "3"))
    calib = REPO / "results/pdv2_diffattn_gate_learning_20261104/calibration_recenter.json"
    fit = json.loads(calib.read_text())["spectral"]
    checks["nsd_score_positive_affine"] = fit["coefficients"][0] > 0 and fit["scale"][0] > 0 and sha(calib) == protocol["sources"][str(calib.relative_to(REPO))]

    # 2. 검증 τ 독립 재현
    taus, ties = {}, {}
    ok = True
    for m in FIELD:
        for s in (1, 2, 3):
            parts = [labels(load(path_of(m, "val", s, n)), FIELD[m]) for n in VAL]
            joined = [np.concatenate([p[i] for p in parts]) for i in range(4)]
            got = choose_independent(*joined)
            for point in ("P99", "P100", "F1"):
                frozen = freeze["thresholds"][m][str(s)][point]
                same = got[point] == frozen["tau"] and float.fromhex(frozen["tau_hex"]) == frozen["tau"]
                c = cells(*joined, frozen["tau"])
                same &= (c["TP"], c["FP"]) == (frozen["val_TP"], frozen["val_FP"])
                ok &= bool(same)
                taus[f"{m}_s{s}_{point}"] = got[point]
            ties[f"{m}_s{s}"] = got["ties_at_tau"]
    checks["val_tau_reproduced_bitwise_with_tie_rule"] = ok
    details["eligible_queries_tied_at_tau"] = ties

    # 3. 시험 판정 직접 재계산과 주표 원장 일치
    ok_cells, ok_support, ok_bev_official, ok_main = True, True, True, True
    recount = {}
    for m in FIELD:
        for s in (1, 2, 3):
            seqs = {}
            for sensor, seq in TEST:
                name = f"{sensor}_{seq}"
                d = load(path_of(m, "test", s, name))
                ref = load(TEST_DIR / "test" / name / "identity.npz")
                ok_support &= np.array_equal(d["query_idx"], ref["query_idx"]) and np.array_equal(d["min_past_gt_dist"], ref["min_past_gt_dist"])
                if m == "bevplace":
                    o = load(REPO / OFFICIAL[s] / f"dump_bevplace_{name}.npz")
                    ix = np.searchsorted(o["query_idx"], d["query_idx"])
                    ok_bev_official &= all(np.array_equal(o[k][ix], d[k]) for k in ("query_idx", "top1_geo_dist", "score_negl2"))
                    ok_bev_official &= baseline["input_sha256"].get(f"{OFFICIAL[s]}/dump_bevplace_{name}.npz") == sha(REPO / OFFICIAL[s] / f"dump_bevplace_{name}.npz")
                sc, pos, neg, good = labels(d, FIELD[m])
                key = f"{sensor}/{seq}"
                main_seq = main_summary["per_case"][("diff_s{}" if m == "nsd" else "bev_native_s{}").format(s)]["sequences"]
                if pos.any():
                    ok_main &= main_seq[key]["n_pos"] == int(pos.sum()) and abs(main_seq[key]["R@1"] - float(good[pos].mean())) < 1e-12
                seqs[key] = (sc, pos, neg, good, len(sc))
            assert sum(v[4] for v in seqs.values()) == 195726
            for point in ("P99", "P100", "F1"):
                tau = freeze["thresholds"][m][str(s)][point]["tau"]
                cs = {k: cells(*v[:4], tau) for k, v in seqs.items()}
                tp, fp = sum(c["TP"] for c in cs.values()), sum(c["FP"] for c in cs.values())
                n_pos = sum(c["n_pos"] for c in cs.values())
                bycity = {}
                for k, c in cs.items():
                    if c["n_pos"]:
                        bycity.setdefault(city[k], []).append(c["TP"] / c["n_pos"])
                macro = float(np.mean([np.mean(v) for v in bycity.values()]))
                rec = summary["methods"][m][point]["per_seed"][str(s)]
                same = (rec["pooled"]["TP"], rec["pooled"]["FP"]) == (tp, fp) and abs(rec["city_macro_recall"] - macro) < 1e-12
                same &= abs(rec["pooled"]["precision"] - (tp / (tp + fp) if tp + fp else 1.0)) < 1e-12
                same &= abs(rec["pooled"]["recall"] - tp / n_pos) < 1e-12 and len(bycity) == 6
                same &= all((rec["sequence"][k]["TP"], rec["sequence"][k]["FP"], rec["sequence"][k]["FN"], rec["sequence"][k]["TN"])
                            == (c["TP"], c["FP"], c["FN"], c["TN"]) for k, c in cs.items())
                ok_cells &= bool(same)
                recount[f"{m}_s{s}_{point}"] = dict(TP=tp, FP=fp, city_macro_recall=macro)
    checks["test_cells_recomputed"] = ok_cells
    checks["test_support_matches_main_table_reference"] = ok_support
    checks["bev_test_equals_main_table_official_dumps"] = ok_bev_official
    checks["test_n_pos_and_R1_match_main_summary"] = ok_main
    details["recount"] = recount

    # 4. 오수락 목록
    rows = list(csv.DictReader((OUT / "false_accepts_all.csv").open()))
    ok = sha(OUT / "false_accepts_all.csv") == fa["list_csv"]["all"]["sha256"] and len(rows) == fa["list_csv"]["all"]["rows"]
    for m in FIELD:
        for s in (1, 2, 3):
            for point in ("P99", "P100", "F1"):
                n = sum(r["method"] == m and r["seed"] == str(s) and r[f"in_{point}"] == "True" for r in rows)
                ok &= n == summary["methods"][m][point]["per_seed"][str(s)]["pooled"]["FP"]
    ok &= all(float(r["returned_m"]) > 5 for r in rows)
    ok &= all((float(r["nearest_true_m"]) <= 5) == (r["query_kind"] == "positive_wrong_place") for r in rows)
    ok &= all(float(r["nearest_true_m"]) > 25 for r in rows if r["query_kind"] == "negative")
    checks["false_accept_list_counts_and_labels"] = bool(ok)
    hp = list(csv.DictReader((OUT / "false_accepts_high_precision.csv").open()))
    checks["high_precision_list_is_subset"] = len(hp) == fa["list_csv"]["high_precision"]["rows"] and all(
        r["in_P99"] == "True" or r["in_P100"] == "True" for r in hp)

    # 5. BEVPlace++는 v2.2 연구(다른 경로의 검증 원장 results/pdv2_bevplace_val_odom_*)와 같은 체크포인트라 τ·TP·FP가 같아야 한다
    old = json.loads((REPO / "results/paper_fixed_threshold_20261101/summary.json").read_text())["fixed"]["bevplace"]
    checks["bev_equals_v22_study_from_other_val_ledgers"] = all(
        old[p]["per_seed"][s]["tau"] == summary["methods"]["bevplace"][p]["per_seed"][s]["tau"]
        and old[p]["per_seed"][s]["TP"] == summary["methods"]["bevplace"][p]["per_seed"][s]["pooled"]["TP"]
        and old[p]["per_seed"][s]["FP"] == summary["methods"]["bevplace"][p]["per_seed"][s]["pooled"]["FP"]
        for p in ("P99", "P100", "F1") for s in ("1", "2", "3"))

    passed = all(bool(v) for v in checks.values())
    result = dict(status="passed" if passed else "failed", checks={k: bool(v) for k, v in checks.items()},
                  details=details, verifier=str(Path(__file__).relative_to(REPO)), verifier_sha256=sha(Path(__file__)),
                  independent_of="scripts/_pr_eval_lib.py (직접 마스크·searchsorted 계산)",
                  inputs_sha256={str(p.relative_to(REPO)): sha(p) for p in
                                 (OUT / "protocol.json", OUT / "freeze.json", OUT / "summary.json", OUT / "false_accepts.json")})
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result["status"], {k: v for k, v in result["checks"].items() if not v} or "all checks passed")


if __name__ == "__main__":
    main()
