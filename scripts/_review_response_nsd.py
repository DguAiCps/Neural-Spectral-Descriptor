#!/usr/bin/env python3
"""현재 원고의 신경망 제거와 시간 판정 제거를 최종 원장으로 대조한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline env \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/_review_response_nsd.py run
"""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import _pdv2_power_adoption_ablation as adoption
from _pdv2_pilot import VAL, TEST, CITY

power, base, cal, white = adoption.power, adoption.base, adoption.cal, adoption.white
ROOT = power.REPO
OUT = ROOT / "results/pdv2_review_response_20261104/nsd"
READ, WRITE, ARR, HASH = power.read, power.write_json, power.arrays, power.sha256
SEEDS = (1, 2, 3)


def prepare() -> dict:
    """대조와 검증 선택 규칙을 결과 생성 전에 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        spec = READ(path)
        for p, h in spec["sources"].items():
            assert HASH(ROOT / p) == h, p
        return spec
    paths = [Path(__file__), Path(adoption.__file__), Path(cal.__file__),
             ROOT / "src/retrieval/sequence_evidence.py", power.OUT / "test_freeze.json",
             ROOT / "results/pdv2_spectral_keys_20261025/remetric_octave288.npy", white.OUT / "statistics.npz"]
    spec = dict(seeds=SEEDS, val=["/".join(s) for s in VAL], test=["/".join(s) for s in TEST],
                created_unix=time.time(), no_context="normalize(omega * d288); 신경망·문맥·게이트를 실행하지 않음. 학습셋으로 맞춘 대각 재계량과 보정 통계는 유지하므로 완전 무학습 방법은 아님.",
                controls="완전한 NSD / 신경망 제거 / 시간 판정 제거 / 둘 다 제거. 원래 100+100 후보·정렬·압축·보정·가중치 그대로. 시드별 전체모델 융합을 고정한 aux_half가 주 대조.",
                retuned="신경망 제거 대조의 보조 결과: 기존 9개 융합 후보 중 검증 3시드·3계열 평균 최종 AP 최대를 공통 선택; 동률 목록순. 파워0.5·동일 채널·75:25·시간 규칙 고정. 시험에서 고르지 않음.",
                fusion_grid=cal.FUSIONS, sources={str(p.relative_to(ROOT)): HASH(p) for p in paths})
    WRITE(path, spec)
    return spec


def center_scores(bank: tuple, corrected: tuple, q: np.ndarray, j: np.ndarray) -> np.ndarray:
    """시간 이웃 없이 같은 두 스펙트럼 채널의 중심 점수만 합친다."""
    channels = {}
    for ch in ("c_spec", "c_col"):
        values = []
        for b in (bank, corrected):
            parts = [adoption.previous.correlation_profiles(b, q[k:k+256], j[k:k+256], ch).max(1)
                     for k in range(0, len(q), 256)]
            value = np.concatenate(parts)
            values.append(value.astype(np.float64) if b is bank else value)
        channels[ch] = .75 * values[0] + .25 * values[1]
    return cal.confidence(channels["c_spec"], channels["c_col"], "equal", "c_col")


def worker(seed: int, sequence: str, role: str) -> None:
    """고정 모델의 최종 승자·점수와 두 제거 대조를 보존한다."""
    spec = prepare()
    if role == "test":
        freeze = READ(OUT / "freeze.json")
        assert freeze["protocol_sha256"] == HASH(OUT / "protocol.json")
        fusions = list(dict.fromkeys(["aux_half", freeze["selected_fusion"]]))
    else:
        fusions = list(cal.FUSIONS)
    name = sequence.replace("/", "_")
    folder = OUT / role / f"s{seed}"
    target = folder / (name + ".json")
    if target.exists():
        assert READ(target)["status"] == "passed"
        return
    folder.mkdir(parents=True, exist_ok=True)
    tick = time.time()
    refpath = (ROOT / "results/paper_fixed_threshold_20261101/nsd_val" / f"s{seed}" / (name + ".npz")
               if role == "val" else power.OUT / "test" / f"s{seed}" / (name + "_p050__aux_half__equal.npz"))
    ref = ARR(refpath)
    _, _, paths, _ = white.prior.BASE.exp.old.base.inputs(role, sequence)
    cache = ARR(paths["cache"])
    omega = np.load(ROOT / "results/pdv2_spectral_keys_20261025/remetric_octave288.npy")
    key = cache["descriptors"].astype(np.float32) * omega.astype(np.float32)
    key /= np.maximum(np.linalg.norm(key, axis=1, keepdims=True), 1e-12)
    assert key.shape[1] == 288 and np.isfinite(key).all()
    original = ARR(base.source_folder(seed, role, READ(power.OUT / "protocol.json")) / ("keys_"+name+".npz"))
    raw_block = original["embedding"][:, :288].copy()
    raw_block /= np.maximum(np.linalg.norm(raw_block, axis=1, keepdims=True), 1e-12)
    np.testing.assert_allclose(key, raw_block, atol=3e-7, rtol=3e-6)
    x = white.coefficients(paths["spec"])
    with np.load(paths["spec"]) as source:
        np.testing.assert_array_equal(source["scan_ids"], cache["scan_ids"])
    wx = white.transform(x, ARR(white.OUT / "statistics.npz")["full"])
    bank = base.variant_inputs(x, 8)
    nb, nc = [base.variant_inputs(power.transform(v, .5), 8) for v in (x, wx)]
    xyz, pos = cache["poses"][:, :3, 3], cache["poses_gt"][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    q = ref["query_idx"]
    conf = READ(power.OUT / "protocol.json")["confidence"][str(seed)]
    weights = cal.fusion_weights(conf)[[cal.FUSIONS.index(f) for f in fusions]]
    ids = np.stack([cal.candidates(key, bank, weights, int(t)) for t in q])
    assert (ids <= q[:, None] - 30).all()
    # 동일 쌍의 중복 계산을 피하되 각 융합의 최종 승자를 그대로 보존한다.
    pair, inv = np.unique((q[:, None]*len(key) + ids).ravel(), return_inverse=True)
    pq, pj = pair//len(key), pair % len(key)
    temporal = adoption.scores(nb, nc, travel, pq, pj)["score"][inv].reshape(ids.shape)
    center = center_scores(nb, nc, pq, pj)[inv].reshape(ids.shape)
    np.testing.assert_array_equal(adoption.scores(nb, nc, travel, q, ref["j_star"])["score"], ref["score"])
    full_center = center_scores(nb, nc, q, ref["j_star"])
    records = {}
    def save(label: str, row: dict, score: np.ndarray) -> None:
        path = folder / f"{name}_{label}.npz"
        np.savez_compressed(path, **{k: row[k] for k in base.FIELDS}, score=score)
        records[label] = dict(path=str(path.relative_to(ROOT)), sha256=HASH(path),
                              metrics=base.independent_metrics(row, score))
    save("main", ref, ref["score"])
    save("no_temporal_score", ref, full_center)
    for i, fusion in enumerate(fusions):
        row = dict(ref, j_star=ids[:, i], top1_geo_dist=np.linalg.norm(pos[q]-pos[ids[:, i]], axis=1))
        save("no_context__" + fusion, row, temporal[:, i])
        save("neither__" + fusion, row, center[:, i])
    for at in (len(q)//3, 2*len(q)//3):
        end = int(q[at]) + 16
        short = [adoption.previous.source.base.exp.truncate_bank(b, end) for b in (nb, nc)]
        np.testing.assert_array_equal(adoption.scores(*short, travel[:end], q[at:at+1], ids[at:at+1, 0])["score"], temporal[at:at+1, 0])
        np.testing.assert_array_equal(cal.candidates(key[:end], adoption.previous.source.base.exp.truncate_bank(bank, end), weights, int(q[at])), ids[at])
    WRITE(target, dict(status="passed", role=role, sequence=sequence, seed=seed, elapsed_s=time.time()-tick,
                       queries=len(q), main_exact=True, truncated_prefix_checks=2, records=records,
                       inputs={str(p.relative_to(ROOT)): HASH(p) for p in [refpath, paths["cache"], paths["spec"]]}))
    print("DONE", role, seed, sequence, round(time.time()-tick, 1), flush=True)


def aggregate(role: str) -> dict:
    """모든 최종 원장에서 검증·시험의 동일 집계를 재계산한다."""
    spec = prepare()
    rows = {}
    for seed in SEEDS:
        for seq in spec[role]:
            record = READ(OUT / role / f"s{seed}" / (seq.replace("/", "_") + ".json"))
            for arm, entry in record["records"].items():
                path = ROOT / entry["path"]
                assert HASH(path) == entry["sha256"]
                data = ARR(path)
                metric = base.independent_metrics(data, data["score"])
                assert metric == entry["metrics"]
                rows.setdefault(arm, {}).setdefault(str(seed), {})[seq] = metric
    result = {}
    for arm, seeds in rows.items():
        per_seed = {s: dict((base.macro(r) if role == "val" else adoption.previous.macro(r, CITY)), sequences=r)
                    for s, r in seeds.items()}
        stats = {m: dict(mean=float(np.mean([v["macro"][m] for v in per_seed.values()])),
                         std=float(np.std([v["macro"][m] for v in per_seed.values()], ddof=1))) for m in base.METRICS}
        result[arm] = dict(per_seed=per_seed, mean_std=stats)
    WRITE(OUT / f"{role}_summary.json", result)
    if role == "val":
        best = max(cal.FUSIONS, key=lambda f: result["no_context__"+f]["mean_std"]["AP"]["mean"])
        freeze = dict(selected_fusion=best, selected_from="validation only, mean final AP", created_unix=time.time(),
                      protocol_sha256=HASH(OUT/"protocol.json"), validation_sha256=HASH(OUT/"val_summary.json"), test_used=False)
        path = OUT/"freeze.json"
        if path.exists():
            assert READ(path)["selected_fusion"] == best
        else:
            WRITE(path, freeze)
    return result


def run(workers: int) -> None:
    """검증→고정→시험 순서를 지키며 시퀀스별 프로세스를 실행한다."""
    spec = prepare()
    log = OUT/"logs"
    log.mkdir(exist_ok=True)
    for role in ("val", "test"):
        jobs = [(seed, seq) for seed in SEEDS for seq in spec[role]]
        def call(job: tuple) -> None:
            seed, seq = job
            with (log / f"{role}_s{seed}_{seq.replace('/', '_')}.log").open("w") as stream:
                subprocess.run([sys.executable, __file__, "worker", "--role", role, "--seed", str(seed), "--sequence", seq],
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for done, future in enumerate(concurrent.futures.as_completed([pool.submit(call, j) for j in jobs]), 1):
                future.result()
                WRITE(OUT/"status.json", dict(phase=role, done=done, total=len(jobs), status="running"))
                print("PROGRESS", role, done, len(jobs), flush=True)
        aggregate(role)
    WRITE(OUT/"status.json", dict(status="complete", test_used_for_selection=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "worker", "run", "aggregate"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sequence", default="kitti/09")
    parser.add_argument("--role", choices=("val", "test"), default="val")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.action == "worker": worker(args.seed, args.sequence, args.role)
    elif args.action == "run": run(args.workers)
    elif args.action == "aggregate": aggregate(args.role)
    else: prepare()
