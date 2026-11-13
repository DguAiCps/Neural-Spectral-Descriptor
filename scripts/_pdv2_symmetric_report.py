#!/usr/bin/env python3
"""대칭 문맥 제거 실험의 무결성 검사·집계·CSV·그림을 만든다."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from _pdv2_no_context_ablation import metrics, sha256, write_json
from _pr_eval_lib import classify_queries, operating_point
from _pdv2_symmetric_ablation import (
    ARMS, LAG, OPS, OUT, PAIR, PREFIX_SEQS, REPO, VAL, protocol,
)

GROUPS = ("same_available", "opposite_only", "other")


def load_ledger(arm: str, op: str, sensor: str, seq: str) -> tuple:
    """공통 구간 질의 기록과 메타데이터를 읽는다."""
    path = OUT / f"val_{arm}_{op}_{sensor}_{seq}.npz"
    with np.load(path) as archive:
        return ({k: archive[k] for k in archive.files if k != "meta"},
                json.loads(str(archive["meta"])))


def statistics(values: np.ndarray) -> dict:
    """유한한 관측값의 범위를 반환한다."""
    assert np.isfinite(values).all()
    return {"min": float(values.min()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def audit() -> dict:
    """전체 입력 해시·질의 정렬·후보 시점·수치·15키프레임 상한을 검사한다."""
    spec = protocol()
    protocol_hash = sha256(OUT / "protocol.json")
    checked = {}
    for name, expected in spec["sha256"].items():
        actual = sha256(REPO / name)
        assert actual == expected, name
        checked[name] = actual
    counts = {"ledgers": 0, "rows": 0, "queries": 0, "positives": 0, "excluded_tail": 0}
    prefix_rows, key_errors = [], []
    for sensor, seq in VAL:
        with np.load(OUT / f"keys_{sensor}_{seq}.npz") as keys:
            emb, scan_ids = keys["symmetric"], keys["scan_ids"]
            meta = json.loads(str(keys["meta"]))
            assert meta["protocol_sha256"] == protocol_hash
            q = np.arange(30, len(emb) - LAG)
            np.testing.assert_array_equal(q, keys["query_idx"])
            assert emb.shape == (len(scan_ids), 416) and emb.dtype == np.float32
            assert np.isfinite(emb).all()
            np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1, atol=2e-7)
            key_errors.append(meta["key_block_max_error"])
        reference = None
        for op in OPS:
            for arm in ARMS:
                arrays, meta = load_ledger(arm, op, sensor, seq)
                assert meta["protocol_sha256"] == protocol_hash
                assert tuple(meta["tuple"]) == tuple(spec["selection"][op]["tuple"])
                assert meta["tau_star"] == spec["selection"][op]["tau_star"]
                np.testing.assert_array_equal(q, arrays["query_idx"])
                for name, values in arrays.items():
                    assert values.shape == (len(q),), name
                    assert np.isfinite(values).all(), name
                assert (arrays["j_star"] <= q - 30).all()
                assert (arrays["j_star"] >= 0).all()
                assert ((arrays["n_candidates"] >= 1) & (arrays["n_candidates"] <= 200)).all()
                if reference is None:
                    reference = arrays
                    counts["queries"] += len(q)
                    counts["positives"] += int((arrays["min_past_gt_dist"] <= 5).sum())
                    counts["excluded_tail"] += LAG
                else:
                    for name in ("query_idx", "min_past_gt_dist", "group"):
                        np.testing.assert_array_equal(reference[name], arrays[name])
                if arm == "no_context":
                    source = REPO / meta["source"]
                    assert sha256(source) == meta["source_sha256"]
                    with np.load(source) as old:
                        for name in arrays:
                            if name != "group":
                                np.testing.assert_array_equal(arrays[name], old[name][:len(q)])
                else:
                    assert meta["keys_sha256"] == sha256(OUT / f"keys_{sensor}_{seq}.npz")
                counts["ledgers"] += 1
                counts["rows"] += len(q)
    for sequence in PREFIX_SEQS:
        p = OUT / f"prefix_{sequence.replace('/', '_')}.json"
        rows = json.loads(p.read_text())
        assert rows["protocol_sha256"] == protocol_hash
        assert rows["checkpoint_sha256"] == checked["checkpoints/pdv2_lag15_s1/best_model.pth"]
        assert rows["classifier_sha256"] == checked["artifacts/edge_classifier_pdv2_lag15.pt"]
        for row in rows["rows"]:
            if row["lookahead"] == LAG:
                assert row["max_error_rows_le_k"] <= spec["prefix_tolerance"]
                prefix_rows.append({"sequence": sequence, **row})
    assert len(prefix_rows) == 9 and counts["ledgers"] == 54
    value = {"passed": True, "counts": counts, "checked_input_sha256": checked,
             "max_key_block_error": max(key_errors), "prefix_rows": prefix_rows,
             "protocol_sha256": protocol_hash,
             "numerical_diagnosis_sha256": sha256(OUT / "tf32_precheck/prefix_numerical_diagnosis.json")}
    write_json(OUT / "audit.json", value)
    return value


def aggregate() -> dict:
    """조건별 전체 지표·방향별 검색률·짝지은 정오답 변화를 집계한다."""
    spec = protocol()
    result = {"scope": spec["scope"], "seed": 1,
              "protocol_sha256": sha256(OUT / "protocol.json"),
              "report_script_sha256": sha256(Path(__file__)), "ops": {},
              "aggregation_note": "공통 질의에서 양성이 0개인 시퀀스는 AP·재현율을 null로 표시하고 세 조건 모두 계열별 평균에서 제외한다. 해당 시퀀스의 비재방문 오수락은 별도 보존한다. 나머지는 시퀀스 평균→3계열 균등 평균이다. 전체 질의 감사에는 모든 9개를 포함한다."}
    for op in OPS:
        arms = {}
        for arm in ARMS:
            per_seq, gate = {}, []
            direction = {g: {"n": 0, "key_correct": 0, "key_hit": 0,
                             "union_hit": 0, "full_correct": 0} for g in GROUPS}
            paired = {other: {g: {"n": 0, "key_recovered": 0, "key_regressed": 0,
                                 "full_recovered": 0, "full_regressed": 0}
                              for g in ("all", *GROUPS)} for other in ARMS if other != arm}
            for sensor, seq in VAL:
                arrays, _ = load_ledger(arm, op, sensor, seq)
                positive = arrays["min_past_gt_dist"] <= 5
                key_ok, full_ok = arrays["key_top1_geo"] <= 5, arrays["top1_geo_dist"] <= 5
                if positive.any():
                    row = metrics(arrays, spec["selection"][op]["tau_star"])
                else:
                    row = {k: None for k in ("AP", "max_F1", "R@100P", "R@99P", "R@95P",
                                             "R@1_revisit", "P_at_tau", "R_at_tau")}
                row.update({"key_R@1": float(key_ok[positive].mean()) if positive.any() else None,
                            "key_R@100": float(arrays["key_hit"][positive].mean()) if positive.any() else None,
                            "union_R@200": float(arrays["union_hit"][positive].mean()) if positive.any() else None,
                            "n_queries": len(positive), "n_positive": int(positive.sum())})
                per_seq[sensor + "/" + seq] = row
                for gi, group in enumerate(GROUPS):
                    mask = positive & (arrays["group"] == gi)
                    values = direction[group]
                    values["n"] += int(mask.sum())
                    for name, ok in (("key_correct", key_ok), ("full_correct", full_ok),
                                     ("key_hit", arrays["key_hit"]), ("union_hit", arrays["union_hit"])):
                        values[name] += int(ok[mask].sum())
                for other in paired:
                    old, _ = load_ledger(other, op, sensor, seq)
                    old_key, old_full = old["key_top1_geo"] <= 5, old["top1_geo_dist"] <= 5
                    for gi, group in enumerate(("all", *GROUPS)):
                        mask = positive if gi == 0 else positive & (arrays["group"] == gi - 1)
                        values = paired[other][group]
                        values["n"] += int(mask.sum())
                        for name, before, after in (("key", old_key, key_ok), ("full", old_full, full_ok)):
                            values[name + "_recovered"] += int((mask & ~before & after).sum())
                            values[name + "_regressed"] += int((mask & before & ~after).sum())
                if op == OPS[0] and arm != "no_context":
                    path = OUT / f"keys_{sensor}_{seq}.npz"
                    with np.load(path) as keys:
                        emb = keys[arm][arrays["query_idx"]]
                        gate.append(np.linalg.norm(emb[:, 288:], axis=1) / np.linalg.norm(emb[:, :288], axis=1))
            names = [k for k in next(iter(per_seq.values())) if not k.startswith("n_")]
            family = {}
            for sensor_family in ("kitti", "helipr", "mulran"):
                rows = [v for k, v in per_seq.items() if v["n_positive"] > 0
                        and ("kitti" if k.startswith("kitti") else k.split("/")[0]) == sensor_family]
                assert rows
                family[sensor_family] = {k: float(np.mean([r[k] for r in rows])) for k in names}
            macro = {k: float(np.mean([v[k] for v in family.values()])) for k in names}
            for values in direction.values():
                for name, numerator in (("key_R@1", "key_correct"), ("full_R@1", "full_correct"),
                                        ("key_R@100", "key_hit"), ("union_R@200", "union_hit")):
                    values[name] = values[numerator] / values["n"] if values["n"] else None
            arms[arm] = {"macro": macro, "families": family, "per_sequence": per_seq,
                         "direction_pooled": direction, "paired_vs": paired}
            arms[arm]["macro_sequences"] = [k for k, v in per_seq.items() if v["n_positive"] > 0]
            arms[arm]["no_positive_sequences"] = {}
            for name, row in per_seq.items():
                if row["n_positive"]:
                    continue
                sensor, seq = name.split("/")
                arrays, _ = load_ledger(arm, op, sensor, seq)
                qc = classify_queries(arrays["min_past_gt_dist"])
                decision = operating_point(arrays["c_spec"], arrays["top1_geo_dist"],
                                           qc, spec["selection"][op]["tau_star"])
                arms[arm]["no_positive_sequences"][name] = {
                    "n_queries": len(arrays["query_idx"]), "n_negative": int(qc["negative"].sum()),
                    "n_ambiguous": int(qc["ambiguous"].sum()),
                    "negative_false_accepts": int(((arrays["c_spec"] >= spec["selection"][op]["tau_star"])
                                                   & qc["negative"]).sum()),
                }
            if gate:
                arms[arm]["gate_alpha"] = statistics(np.concatenate(gate))
        result["ops"][op] = arms
    delay = {"per_sequence": {}}
    times, distances = [], []
    for sensor, seq in VAL:
        with np.load(OUT / f"keys_{sensor}_{seq}.npz") as keys:
            times.append(keys["delay_seconds"])
            distances.append(keys["delay_travel_m"])
            delay["per_sequence"][sensor + "/" + seq] = {
                "seconds": statistics(times[-1]), "odom_travel_m": statistics(distances[-1])}
    delay["pooled"] = {"seconds": statistics(np.concatenate(times)),
                       "odom_travel_m": statistics(np.concatenate(distances))}
    result["delay"] = delay
    write_json(OUT / "aggregation_note.json", {
        "reason": "끝 15키프레임 제외 뒤 KITTI 09의 양성이 0개여서 첫 집계가 NaN 검사에서 중단됐다. 성능 순위에 관계없이 모든 조건에 같은 유효 분모 규칙을 적용한다.",
        "rule": result["aggregation_note"], "protocol_sha256": result["protocol_sha256"],
    })
    write_json(OUT / "summary.json", result)
    return result


def tables_and_figure(result: dict) -> None:
    """분량 작업에 재사용할 수 있는 CSV와 벡터·비트맵 그림을 저장한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = OUT / "UnDotum.ttf"
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False, "pdf.fonttype": 42})
    names = {"no_context": "문맥 없음", "past": "과거 문맥", "symmetric": "대칭 문맥"}
    with (OUT / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        metrics_names = list(result["ops"][OPS[0]][ARMS[0]]["macro"])
        writer.writerow(["op", "arm", *metrics_names])
        for op in OPS:
            for arm in ARMS:
                writer.writerow([op, arm, *[result["ops"][op][arm]["macro"][k] for k in metrics_names]])
    with (OUT / "direction.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["op", "arm", "direction", "n", "key_R@1", "full_R@1", "key_R@100"])
        for op in OPS:
            for arm in ARMS:
                for group, row in result["ops"][op][arm]["direction_pooled"].items():
                    writer.writerow([op, arm, group, *[row[k] for k in ("n", "key_R@1", "full_R@1", "key_R@100")]])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    colors = ("#737373", "#377eb8", "#e68613")
    for ai, arm in enumerate(ARMS):
        x = np.arange(2) + (ai - 1) * .25
        values = [100 * result["ops"][op][arm]["macro"]["AP"] for op in OPS]
        bars = axes[0].bar(x, values, .24, label=names[arm], color=colors[ai])
        axes[0].bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
        values = [100 * result["ops"][OPS[0]][arm]["direction_pooled"][g]["key_R@1"]
                  for g in GROUPS[:2]]
        bars = axes[1].bar(x, values, .24, color=colors[ai])
        axes[1].bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
        values = [100 * result["ops"][op][arm]["direction_pooled"]["opposite_only"]["full_R@1"]
                  for op in OPS]
        bars = axes[2].bar(x, values, .24, color=colors[ai])
        axes[2].bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    for ax in axes:
        ax.set_ylim(0, 105)
        ax.set_ylabel("%")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_xticks([0, 1], ["NSD", "NSD-H"])
    axes[0].set_title("전체 AP · 센서 계열 평균")
    axes[1].set_xticks([0, 1], ["같은 방향 정답 존재", "반대 방향 정답만"])
    axes[1].set_title("키 단독 검색 R@1 · 질의 합산")
    axes[2].set_xticks([0, 1], ["NSD", "NSD-H"])
    axes[2].set_title("반대 방향 최종 R@1 · 질의 합산")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.savefig(OUT / "symmetric_context_ablation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "symmetric_context_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """검사에 통과한 전체 결과만 집계하고 그림을 만든다."""
    check = audit()
    result = aggregate()
    tables_and_figure(result)
    check["summary_sha256"] = sha256(OUT / "summary.json")
    check["report_script_sha256"] = sha256(Path(__file__))
    write_json(OUT / "audit.json", check)
    print(json.dumps(check["counts"]))
    for op in OPS:
        for arm in ARMS:
            row = result["ops"][op][arm]
            print(op, arm, json.dumps(row["macro"]),
                  "opposite", json.dumps(row["direction_pooled"]["opposite_only"]))


if __name__ == "__main__":
    main()
