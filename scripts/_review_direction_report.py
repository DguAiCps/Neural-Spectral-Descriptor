#!/usr/bin/env python3
"""방향 층화와 사전고정 원시 yaw 실험을 검산·집계한다."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import _review_direction_metrics as base

ROOT, OUT, MAIN = base.ROOT, base.OUT, base.MAIN


def stats(values: np.ndarray) -> dict:
    """유한한 측정 배열의 요약을 보존한다."""
    values = np.asarray(values)
    assert np.isfinite(values).all()
    return dict(n=len(values), mean=float(np.mean(values)), min=float(np.min(values)),
                p05=float(np.quantile(values, .05)), p95=float(np.quantile(values, .95)), max=float(np.max(values)))


def unit(values: np.ndarray) -> np.ndarray:
    """행별 단위벡터를 만든다."""
    flat = values.reshape(len(values), -1)
    return flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)


def direction_csv() -> None:
    """방향 지표의 평균·시작점 편차를 사람이 비교하기 쉬운 표로 쓴다."""
    summary = base.read(OUT / "direction_summary.json")
    with (OUT / "direction_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "group", "metric", "mean_percent", "std_pp"])
        writer.writeheader()
        for method, methods in summary["methods"].items():
            for group, metrics in methods["mean_std"].items():
                for key, values in metrics.items():
                    writer.writerow(dict(method=method, group=group, metric=key, mean_percent=100*values["mean"],
                                         std_pp=None if values["std"] is None else 100*values["std"]))
    with (OUT / "direction_city_dataset.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "direction", "level", "name", "metric", "mean_percent", "std_pp", "positive_queries", "sequences"])
        writer.writeheader()
        for method, methods in summary["methods"].items():
            seed_values = list(methods["per_seed"].values())
            for direction in methods["mean_std"]:
                for level in ("cities", "datasets"):
                    for name, reference in seed_values[0][direction][level].items():
                        if reference is None:
                            continue
                        for key in base.KEYS:
                            values = [s[direction][level][name][key] for s in seed_values]
                            writer.writerow(dict(method=method, direction=direction, level=level, name=name, metric=key,
                                                 mean_percent=100*float(np.mean(values)),
                                                 std_pp=100*float(np.std(values, ddof=1)) if len(values)>1 else None,
                                                 positive_queries=reference["n_pos"], sequences=len(reference["sequences"])))


def yaw_summary() -> dict:
    """모든 각도·시작점·시퀀스 결과와 원시 재생 근거를 검산한다."""
    spec = base.read(OUT / "yaw_protocol.json")
    baseline = base.read(OUT / "yaw_baseline_validation.json")
    assert baseline["status"] == "passed"
    sequences = list(spec["queries"])
    city_groups = {c: [s for s in seqs if s in sequences] for c, seqs in base.CITY.items()}
    dataset_groups = {d: [s for s in sequences if s.startswith(d+"/")] for d in sorted({s.split("/")[0] for s in sequences})}
    manifest = base.read(OUT / "yaw_implementation_manifest.json")
    for path, digest in manifest["sha256"].items():
        assert base.digest(ROOT / path) == digest, path
    byangle, source_hashes, principle_samples, per_principle, counts = {}, {}, {}, {}, {}
    coordinate_audit = {}
    for angle in spec["angles"]:
        per_seed = {}
        all_pairs = []
        for tag in spec["cases"]:
            metrics, pairs = {}, {}
            for sequence in sequences:
                folder = OUT / "yaw" / sequence.replace("/", "_")
                path = folder / f"{tag}_{angle:g}.npz"
                record = base.read(path.with_suffix(".json"))
                assert base.digest(path) == record["sha256"]
                source_hashes[str(path.relative_to(ROOT))] = record["sha256"]
                row = dict(np.load(path))
                original = dict(np.load(folder / f"{tag}_0.npz"))
                for key in ("query_idx", "min_past_gt_dist"):
                    np.testing.assert_array_equal(row[key], original[key])
                np.testing.assert_array_equal(row["query_idx"], spec["queries"][sequence])
                assert (row["j_star"] <= row["query_idx"]-30).all()
                assert np.isfinite(row["score"]).all()
                assert 1 <= row["n_candidates"].min() <= row["n_candidates"].max() <= 200
                metrics[sequence] = base.metric(row["score"], row["top1_geo_dist"], row["min_past_gt_dist"])
                for key, value in metrics[sequence].items():
                    np.testing.assert_allclose(value, record["metrics"][key], atol=1e-12, rtol=0)
                qc = base.classify_queries(row["min_past_gt_dist"])
                before = (original["top1_geo_dist"] <= 5) & qc["positive"]
                after = (row["top1_geo_dist"] <= 5) & qc["positive"]
                pair = dict(queries=len(row["query_idx"]), winner_agreement=int(np.sum(row["j_star"] == original["j_star"])),
                            correct_before=int(before.sum()), correct_after=int(after.sum()),
                            recovered=int((~before & after).sum()), regressed=int((before & ~after).sum()),
                            score_absolute_difference=stats(abs(row["score"]-original["score"])),
                            gallery_raw_overlap_count=int(row["gallery_raw_support_overlaps_query"].sum()))
                pairs[sequence] = pair
                all_pairs.append(pair)
                if angle == 0 and tag == "diff_s1":
                    counts[sequence] = dict(queries=len(row["query_idx"]), positive=qc["n_pos"], negative=qc["n_neg"], ambiguous=qc["n_ambiguous"])
            cities = base.aggregate(metrics, city_groups)
            valid = [v for v in cities.values() if v is not None]
            per_seed[tag] = dict(macro={k: float(np.mean([v[k] for v in valid])) for k in base.KEYS},
                                 cities=cities, datasets=base.aggregate(metrics, dataset_groups), sequences=metrics, paired=pairs)
        mean_std = {k: dict(mean=float(np.mean([v["macro"][k] for v in per_seed.values()])),
                           std=float(np.std([v["macro"][k] for v in per_seed.values()], ddof=1))) for k in base.KEYS}
        byangle[str(angle)] = dict(mean_std=mean_std, per_seed=per_seed,
            pooled_query_seed_pairs={k: sum(v[k] for v in all_pairs) for k in ("queries", "winner_agreement", "correct_before", "correct_after", "recovered", "regressed", "gallery_raw_overlap_count")})
        per_principle[str(angle)] = {}
        principle_samples[str(angle)] = {}
        coordinate_audit[str(angle)] = {}
        for sequence in sequences:
            folder = OUT / "yaw" / sequence.replace("/", "_")
            path = folder / f"encoded_{angle:g}.npz"
            rec = base.read(path.with_suffix(".json"))
            orig_rec = base.read(folder / "encoded_0.json")
            assert rec["points_sha256"] == orig_rec["points_sha256"]
            assert base.digest(path) == rec["sha256"]
            source_hashes[str(path.relative_to(ROOT))] = rec["sha256"]
            data = dict(np.load(path))
            original = dict(np.load(folder / "encoded_0.npz"))
            np.testing.assert_array_equal(data["frames"], original["frames"])
            real0, reala = unit(original["raw"]), unit(data["raw"])
            coeff0, coeffa = unit(original["coeff"]), unit(data["coeff"])
            phase = np.exp(1j*np.deg2rad(angle)*np.arange(1, 9))[None, None]
            corrected = unit(data["coeff"]*phase)
            arrays = dict(range_key_cosine=np.sum(real0*reala, axis=1), range_key_l2=np.linalg.norm(real0-reala, axis=1),
                          bev_magnitude_cosine=np.sum(unit(abs(original["coeff"]))*unit(abs(data["coeff"])), axis=1),
                          bev_phase_compensated_l2=np.linalg.norm(coeff0-corrected, axis=1))
            per_principle[str(angle)][sequence] = {k: stats(v) for k, v in arrays.items()}
            for k, v in arrays.items(): principle_samples[str(angle)].setdefault(k, []).append(v)
            with np.load(base.cache_path(sequence)) as cache:
                poses = cache["poses"]
            relative = poses[0, :3, :3].T[None] @ poses[data["frames"], :3, :3]
            rad = np.deg2rad(angle)
            c, s = np.cos(rad), np.sin(rad)
            rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            changed = relative @ rotation.T
            residual = np.arctan2(changed[:, 1, 0], changed[:, 0, 0]) - np.arctan2(relative[:, 1, 0], relative[:, 0, 0]) + rad
            residual = np.angle(np.exp(1j*residual))
            coordinate_audit[str(angle)][sequence] = dict(
                world_point_max_error_m=rec["world_coordinate_max_error"],
                yaw_only_transport_vs_exact_phase_residual_deg=stats(abs(np.rad2deg(residual))))
    for angle, row in byangle.items():
        row["delta_vs_zero_pp"] = {k: 100*(row["mean_std"][k]["mean"]-byangle["0.0"]["mean_std"][k]["mean"]) for k in base.KEYS}
    principle = {angle: {k: stats(np.concatenate(v)) for k, v in arrays.items()} for angle, arrays in principle_samples.items()}
    totals = {k: sum(row[k] for row in counts.values()) for k in next(iter(counts.values()))}
    assert totals["queries"] == 512
    direction_counts = {label: 0 for label in base.LABELS}
    for sequence in sequences:
        with np.load(OUT / "directions" / (sequence.replace("/", "_")+".npz")) as directions:
            ix = np.searchsorted(directions["query_idx"], spec["queries"][sequence])
            selected = directions["group"][ix]
        for code, label in enumerate(base.LABELS):
            direction_counts[label] += int((selected == code).sum())
    result = dict(status="passed", scope=spec["scope"], counts=totals, per_sequence_counts=counts,
                  sampled_positive_direction_counts=direction_counts,
                  final_recognition=byangle, encoding_principle=principle, encoding_per_sequence=per_principle,
                  coordinate_audit=coordinate_audit,
                  coordinate_audit_scope="세계 좌표 오차는 시퀀스별 첫 지원 스캔의 모든 점×7각도(56스캔-각도)에서 실측한다. T'=T Rz(-theta), p'=Rz(theta)p의 대수적 일치는 모든 점에 적용된다. yaw-only 위상 잔차는 모든14274지원프레임에서 측정하며 이 잔차만으로 최종 성능 변화의 원인을 분해하지 않는다.",
                  baseline_validation=dict(queries=baseline["queries"], query_seed_checks=baseline["query_seed_checks"],
                                           max_score_error=baseline["max_score_error"], winner_mismatches=baseline["winner_mismatches"]),
                  protocol_sha256=base.digest(OUT / "yaw_protocol.json"), source_sha256=source_hashes,
                  implementation_manifest_sha256=base.digest(OUT / "yaw_implementation_manifest.json"),
                  gallery_overlap_accounting="gallery_raw_overlap_count는 j+20>=q-15인 보수적 원시 의존 상한의 겹침 건수다. 실제 거리 대응이 택한 각 이웃의 의존을 추적한 정확한 겹침 건수가 아니다. gallery는 이 경우에도 전부 원래 표현을 유지한다.",
                  limitations=["고정512질의의 진단 표본이며 전체195726질의 성능이 아니다.",
                               "원시 점군의 좌표 회전은 새로운 가림·시야 변화·센서 잡음·역주행 이동 경로를 생성하지 않는다.",
                               "gallery는 수정계수와 시간문맥까지 원래 상태로 고정한다. 원시 의존 범위가 겹쳐도 별도 query 표현으로 재인코딩한다.",
                               "학습시작점3개는 공유된 표본·추가학습순서에 조건부이며 독립데이터확증이나 모든yaw각 보장은 아니다."])
    base.write(OUT / "yaw_summary.json", result)
    with (OUT / "yaw_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["angle_deg", *base.KEYS, *[k+"_delta_pp" for k in base.KEYS], "winner_agreement_percent"])
        writer.writeheader()
        for angle, row in byangle.items():
            p = row["pooled_query_seed_pairs"]
            writer.writerow(dict(angle_deg=angle, **{k: 100*v["mean"] for k, v in row["mean_std"].items()},
                                 **{k+"_delta_pp": v for k, v in row["delta_vs_zero_pp"].items()},
                                 winner_agreement_percent=100*p["winner_agreement"]/p["queries"]))
    return result


def main() -> None:
    """완료 자료만 읽어 CSV와 최종 JSON을 만든다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaw", action="store_true")
    args = parser.parse_args()
    direction_csv()
    if args.yaw:
        result = yaw_summary()
        print("PASSED", result["counts"], flush=True)
        for angle, row in result["final_recognition"].items():
            print(angle, {k: round(v["mean"]*100, 4) for k, v in row["mean_std"].items()}, row["delta_vs_zero_pp"], flush=True)


if __name__ == "__main__":
    main()
