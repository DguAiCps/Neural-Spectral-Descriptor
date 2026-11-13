#!/usr/bin/env python3
"""원시 점군을 재인코딩하여 현재 NSD의 질의 yaw 회전 강건성을 잰다."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

import _review_direction_metrics as report
import _pdv2_contrast_pipeline as current
from _pd_build_test_caches import _effective_config, _make_desc_encoder, _make_loader
from _nclt_fov_fix import fixed_config
from _pdv2_input_alignment_lib import make_graph
from _pbev_branch_fusion import BP
from encoding.phase_features import _phase_sketch, _pool_rows
from gnn.spectral_transport import attach_transport, lidar_yaw

ROOT, OUT, MAIN = report.ROOT, report.OUT, report.MAIN
SEQUENCES = ("kitti/00", "kitti360/0000", "helipr/Town01", "helipr/KAIST04",
             "mulran/KAIST01", "nclt/2012-01-08", "mcd/kth_day_06", "mcd/tuhh_day_02")
ANGLES = (0., .5, 1., 6., 7.3, 90., 180.)
TOLERANCE = 8e-6


def freeze() -> dict:
    """결과 전에 표본·각도·모델·관측 범위를 고정한다."""
    path = OUT / "yaw_protocol.json"
    if path.exists():
        spec = report.read(path)
        for name, value in spec["sources"].items():
            assert report.digest(ROOT / name) == value, name
        return spec
    source = report.read(MAIN / "protocol.json")
    samples = {}
    paths = [Path(__file__), MAIN / "protocol.json", ROOT / "src/gnn/spectral_transport.py",
             ROOT / "scripts/_pdv2_input_alignment_lib.py", ROOT / "scripts/_pdv2_contrast_pipeline.py"]
    for sequence in SEQUENCES:
        p = MAIN / "test" / sequence.replace("/", "_") / "diff_s1.npz"
        with np.load(p) as z:
            q = z["query_idx"]
            samples[sequence] = q[np.linspace(0, len(q)-1, 64).astype(int)].tolist()
        paths += [p, report.cache_path(sequence)]
    cases = {f"diff_s{s}": source["cases"][f"diff_s{s}"] for s in (1, 2, 3)}
    paths += [ROOT / c["checkpoint"] for c in cases.values()]
    spec = dict(created_unix=time.time(), angles=ANGLES, queries=samples, cases=cases,
        sample_rule="고정 TEST 순서에서 센서×도시 조합마다 첫 시퀀스. 시퀀스별 공통 질의 시간축의 등간격 인덱스64개. 정답·점수·성능을 표본 선택에 쓰지 않음.",
        scope="질의 회전 강건성. 질의마다 q-15..q+15 원시31스캔을 독립 센서 좌표 회전. 캐시 roll 대체 없음. 전체 연속 궤적·KISS-ICP 재실행이나 실제 역방향 재방문 시점 변화 평가가 아님.",
        coordinate="행벡터 점군 p'=p Rz(theta)^T, local-to-world 자세 T'=T Rz(-theta). T'p'=Tp를 검산한다. 첫 원본 센서 축을 공통 yaw 기준으로 보존. 3D odometry 위치·GT·누적 이동 거리는 그대로. yaw 전용 transport의 roll/pitch 근사는 원래 구현대로다.",
        gallery="원래 전체 입력에서 계산한 gallery 원키·BEV크기·학습복소계수와 후보 시간 문맥을 모두 고정. query q±5 수정계수만 회전된 q±15로 재계산. j<=q-30, 후보 최종 시간 문맥 j±10의 수정계수도 고정이며 그 원시 의존 j±20은 q-15..q-10과 겹칠 수 있다. 겹치는 원시 프레임에도 gallery의 원본 좌표 표현과 query의 회전 표현을 각각 유지하는 독립 질의 실험이다.",
        retrieval="각도마다 재계량288D 원키100+원BEV크기100 후보를 다시 검색하고 수정계수 두 정렬 점수·고정융합·최종시간판정을 재계산. 어떤 각도에도 설정 선택 없음.",
        principle="원키의1도 azimuth 격자와 BEV의6도 격자를 구별. 0.5와7.3은 둘 다 비정수bin. 1도는 원키 정수bin이나 BEV 비정수bin. 6/90/180도는 두 격자의 정수bin. 원시 부동소수 양자화 경계 오차 가능. descriptor·계수 변동과 최종 인식 지표는 분리.",
        baseline="먼저512질의×3시작점 각도0의 승자를 현재 원장과 정확 비교하고 점수오차<=8e-6 검사. 허용치는 기존 전체/접두 평가와 동일하며 실측오차를 별도 기록. 실패하면 회전 본실행 전에 진단.",
        metrics="표본별 원래 양성/음성/모호 규약, 시퀀스→도시6→시작점3 평균 AP/F1/R100/R1. 원래46시퀀스 총계와 다른 표본 통계이며 독립확증이 아님. 절대값보다 각도0의 짝지은 차이와 승자변화도 보고.",
        sources={str(p.relative_to(ROOT)): report.digest(p) for p in paths})
    report.write(path, spec)
    return spec


def rotz(angle: float) -> np.ndarray:
    """도 단위 z축 회전행렬을 만든다."""
    rad = np.deg2rad(angle)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def raw_encode(sequence: str, angles: list, spec: dict) -> dict:
    """필요한 원시 스캔을 실제 회전하여 두 인코딩 경로로 다시 만든다."""
    sensor, seq = sequence.split("/")
    folder = OUT / "yaw" / sequence.replace("/", "_")
    folder.mkdir(parents=True, exist_ok=True)
    frames = np.unique(np.concatenate([np.arange(q-15, q+16) for q in spec["queries"][sequence]]))
    paths = {a: folder / f"encoded_{a:g}.npz" for a in angles}
    missing = [a for a, p in paths.items() if not p.exists()]
    if missing:
        cache = dict(np.load(report.cache_path(sequence)))
        loader = _make_loader(sensor, seq)
        config = fixed_config() if sensor == "nclt" else _effective_config()
        encoder = _make_desc_encoder(config, sensor, "cuda").eval()
        raw = {a: np.empty((len(frames), 288), np.float32) for a in missing}
        coeff = {a: np.empty((len(frames), 16, 8), np.complex64) for a in missing}
        started = time.time()
        raw_hash = __import__("hashlib").sha256()
        max_world_error = {a: 0. for a in missing}
        for k, frame in enumerate(frames):
            points = np.ascontiguousarray(loader[int(cache["scan_ids"][frame])]["points"])
            raw_hash.update(points.tobytes())
            for angle in missing:
                points_rot = points.copy()
                rotation = rotz(angle)
                points_rot[:, :3] = points[:, :3].astype(np.float64) @ rotation.T
                if k == 0:
                    pose = cache["poses"][frame].astype(np.float64)
                    before = points[:, :3].astype(np.float64) @ pose[:3, :3].T + pose[:3, 3]
                    after = points_rot[:, :3].astype(np.float64) @ (pose[:3, :3] @ rotation.T).T + pose[:3, 3]
                    max_world_error[angle] = float(np.max(abs(after-before)))
                    assert max_world_error[angle] < 1e-3
                with torch.no_grad():
                    raw[angle][k] = encoder.encode_points(points_rot).float().cpu().numpy()
                bev, _ = BP.project(points_rot, keep_intensity=False)
                coeff[angle][k] = _phase_sketch(_pool_rows(np.maximum(bev, 0.), 16), 8)
            if (k+1) % 128 == 0:
                report.write(folder / "encoding_status.json", dict(done=k+1, total=len(frames), angles=missing, seconds=time.time()-started))
                print("ENCODE", sequence, k+1, len(frames), missing, round(time.time()-started, 1), flush=True)
        for angle in missing:
            np.savez_compressed(paths[angle], frames=frames, raw=raw[angle], coeff=coeff[angle])
            report.write(paths[angle].with_suffix(".json"), dict(sha256=report.digest(paths[angle]),
                points_sha256=raw_hash.hexdigest(), frame_count=len(frames), seconds_all_angles=time.time()-started,
                world_coordinate_max_error=max_world_error[angle], protocol_sha256=report.digest(OUT / "yaw_protocol.json")))
        del loader, encoder
    output = {}
    for angle, p in paths.items():
        assert report.digest(p) == report.read(p.with_suffix(".json"))["sha256"]
        output[angle] = dict(np.load(p))
        np.testing.assert_array_equal(output[angle]["frames"], frames)
    return output


def original_inputs(sequence: str) -> tuple:
    """저장 원장과 같은 원특징·계수·자세를 읽는다."""
    cache = dict(np.load(report.cache_path(sequence)))
    reference = report.read(MAIN / "test" / sequence.replace("/", "_") / "reference_validation.json")
    sp = ROOT / next(p for p in reference["input_sha256"] if "testspec_" in p)
    z = current.run.white.coefficients(sp)
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(cache["poses"][:, :3, 3], axis=0), axis=1))]
    return cache, z, travel


@torch.no_grad()
def infer(net, raw: np.ndarray, z: np.ndarray, poses: np.ndarray, first_rotation: np.ndarray) -> np.ndarray:
    """공통 원본 첫센서 좌표를 보존한 두 층 문맥을 계산한다."""
    sequence = np.zeros(len(raw), np.int64)
    graph = make_graph(raw, poses, sequence).cuda()
    relative = first_rotation.T[None].astype(np.float64) @ poses[:, :3, :3].astype(np.float64)
    yaw = np.arctan2(relative[:, 1, 0], relative[:, 0, 0])
    attach_transport(graph, z, yaw, sequence)
    return net(graph).cpu().numpy()


def raw_key(raw: np.ndarray) -> np.ndarray:
    """현재 원키 재계량·행별 정규화를 적용한다."""
    scale = np.load(current.run.source.OUT / "remetric_octave288.npy").astype(np.float32)
    value = raw.astype(np.float32) * scale[None]
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)


@torch.no_grad()
def query(q: int, query_raw: np.ndarray, query_z: np.ndarray, query_original_z: np.ndarray,
          keys: np.ndarray, hkeys: np.ndarray, gallery_z: np.ndarray, travel: np.ndarray, weights: np.ndarray) -> tuple:
    """원본 gallery에 회전된 독립 질의 검색·정렬·시간판정을 적용한다."""
    saved_key, saved_hkey = keys[q].copy(), hkeys[q].copy()
    keys[q] = raw_key(query_raw[15:16])[0]
    hkeys[q] = current.run.base.variant_inputs(query_original_z[15:16], 8)[0][0]
    ids = np.unique(np.r_[current.run.base.prefix_topk(keys, q, q-29, 100),
                          current.run.base.prefix_topk(hkeys, q, q-29, 100)])
    both = np.concatenate((gallery_z[ids], query_z[15:16]), 0)
    bank = current.run.base.variant_inputs(both, 8)
    local_ids, local_q = np.arange(len(ids)), len(ids)
    a = current.run.base.align_scores(bank[1], bank[2], local_q, local_ids)
    c = current.run.base.column_align_scores(bank[3], local_q, local_ids)
    value = (weights[0]*current.run.base.minmax(keys[ids] @ keys[q]) +
             weights[1]*current.run.base.minmax(hkeys[ids] @ hkeys[q]) +
             weights[2]*current.run.base.minmax(a) + weights[3]*current.run.base.minmax(c))
    j = int(ids[np.argmax(value)])
    keys[q], hkeys[q] = saved_key, saved_hkey
    plan = current.e.prior.base.prior.plan(travel, np.array([q]), np.array([j]))
    # 후보와 질의는 별도 행으로 옮겨 gallery 원본 표현을 무조건 보존한다.
    cq, cj = np.unique(plan["query"]), np.unique(plan["candidate"])
    assert cq.min() >= q-5 and cq.max() <= q+5
    compact = np.concatenate((query_z[cq-q+15], gallery_z[cj]), 0)
    compact_plan = dict(query=np.searchsorted(cq, plan["query"]),
                        candidate=np.searchsorted(cj, plan["candidate"])+len(cq), weight=plan["weight"])
    score = current.e.prior.score(torch.from_numpy(compact).cuda(),
                                 {k: torch.from_numpy(v).cuda() for k, v in compact_plan.items()})
    return j, float(score.cpu().numpy()[0]), len(ids), int(j+20 >= q-15)


def run_sequence(sequence: str, stage: str, smoke: bool = False) -> None:
    """각도0을 먼저 검산하고 독립 회전 표본의 최종 원장을 저장한다."""
    spec = freeze()
    folder = OUT / "yaw" / sequence.replace("/", "_")
    done = folder / (stage + ("_smoke" if smoke else "") + ".json")
    if done.exists(): return
    if stage == "rotate":
        assert report.read(OUT / "yaw_baseline_validation.json")["status"] == "passed"
    # smoke도 본 표본의 첫 두 질의만 사용하며 사전고정 조건을 바꾸지 않는다.
    if smoke:
        spec = dict(spec, queries={**spec["queries"], sequence: spec["queries"][sequence][:2]})
        # smoke 파일은 본 캐시와 분리한다.
        raise ValueError("smoke는 별도 검산 함수 smoke()를 사용한다")
    angles = [0.] if stage == "baseline" else list(ANGLES[1:])
    encoded = raw_encode(sequence, angles, spec)
    cache, z, travel = original_inputs(sequence)
    keys, hkeys = raw_key(cache["descriptors"]), current.run.base.variant_inputs(z, 8)[0]
    weights = current.run.cal.fusion_weights(report.read(MAIN / "protocol.json")["confidence"])[current.run.cal.FUSIONS.index("aux_half")]
    queries = np.asarray(spec["queries"][sequence], np.int64)
    records = {}
    started = time.time()
    for tag, case in spec["cases"].items():
        net = current.load_model(case)
        gallery_z = infer(net, cache["descriptors"], z, cache["poses"], cache["poses"][0, :3, :3])
        ref = dict(np.load(MAIN / "test" / sequence.replace("/", "_") / (tag+".npz")))
        refix = np.searchsorted(ref["query_idx"], queries)
        for angle in angles:
            raw = encoded[angle]
            values = []
            for q in queries:
                take = np.searchsorted(raw["frames"], np.arange(q-15, q+16))
                poses = cache["poses"][q-15:q+16].copy()
                poses[:, :3, :3] = poses[:, :3, :3] @ rotz(angle).T
                local_z = infer(net, raw["raw"][take], raw["coeff"][take], poses, cache["poses"][0, :3, :3])
                values.append(query(int(q), raw["raw"][take], local_z, raw["coeff"][take],
                                    keys, hkeys, gallery_z, travel, weights))
            j = np.asarray([v[0] for v in values], np.int64)
            score = np.asarray([v[1] for v in values])
            row = dict(query_idx=queries, j_star=j, score=score,
                       min_past_gt_dist=ref["min_past_gt_dist"][refix],
                       top1_geo_dist=np.linalg.norm(cache["poses_gt"][queries, :3, 3]-cache["poses_gt"][j, :3, 3], axis=1),
                       n_candidates=np.asarray([v[2] for v in values]),
                       gallery_raw_support_overlaps_query=np.asarray([v[3] for v in values]))
            path = folder / f"{tag}_{angle:g}.npz"
            np.savez_compressed(path, **row)
            rec = dict(sha256=report.digest(path), angle=angle,
                       metrics=report.metric(score, row["top1_geo_dist"], row["min_past_gt_dist"]))
            if angle == 0:
                error = float(np.max(abs(score-ref["score"][refix])))
                mismatch = int(np.sum(j != ref["j_star"][refix]))
                rec["baseline_check"] = dict(winner_mismatches=mismatch, score_max_error=error,
                    score_tolerance=TOLERANCE, raw_descriptor_max_error=float(np.max(abs(raw["raw"]-cache["descriptors"][raw["frames"]]))),
                    raw_coefficient_max_error=float(np.max(abs(raw["coeff"]-z[raw["frames"]]))))
                report.write(path.with_suffix(".json"), rec)
                assert mismatch == 0 and error <= TOLERANCE, (sequence, tag, rec["baseline_check"])
            report.write(path.with_suffix(".json"), rec)
            records[f"{tag}_{angle:g}"] = rec
            print("EVALUATE", sequence, tag, angle, rec, flush=True)
        del net, gallery_z
    report.write(done, dict(status="passed", records=records, seconds=time.time()-started))


def baseline_validation() -> None:
    """모든 고정 표본의 각도0 검산을 모아 본 회전 실행을 연다."""
    rows = {s: report.read(OUT / "yaw" / s.replace("/", "_") / "baseline.json") for s in SEQUENCES}
    checks = [r["baseline_check"] for v in rows.values() for r in v["records"].values()]
    assert all(r["winner_mismatches"] == 0 and r["score_max_error"] <= TOLERANCE for r in checks)
    report.write(OUT / "yaw_baseline_validation.json", dict(status="passed", queries=512, query_seed_checks=1536,
        winner_mismatches=sum(r["winner_mismatches"] for r in checks),
        max_score_error=max(r["score_max_error"] for r in checks), tolerance=TOLERANCE, sequences=rows))


def main() -> None:
    """사전고정·각도0·회전·완료 검산 단계를 분리 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "baseline", "rotate", "validate"))
    parser.add_argument("--sequence", choices=SEQUENCES)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == "freeze": freeze()
    elif args.stage == "validate": baseline_validation()
    else:
        for sequence in ([args.sequence] if args.sequence else SEQUENCES): run_sequence(sequence, args.stage)


if __name__ == "__main__":
    main()
