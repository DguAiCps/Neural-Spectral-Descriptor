#!/usr/bin/env python3
"""BEVPlace++ 자체 L2 점수에 NSD와 동일한 시간 판정을 적용한다.

GPU 작업은 로컬 또는 cps2의 ring_baseline에서 실행한다.
검증·시험 설정은 동일하고 재선택하지 않는다. 기존 후보·승자는 유지한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src"), str(ROOT/"scripts")]
from retrieval.sequence_evidence import neighbor_plans, repeated_candidate_weights, path_means

OUT = ROOT / "results/pdv2_review_response_20261104/bev"
BETA = (2./3.)*(11./10.)


def digest(path: Path) -> str:
    """큰 특징 파일도 스트림으로 해시한다."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    tmp.replace(path)


def temporal(desc: np.ndarray, travel: np.ndarray, q: np.ndarray, j: np.ndarray,
             center: np.ndarray) -> np.ndarray:
    """자체 음의 L2를 거리 대응·중복 가중·정역방향 최대로 모은다."""
    parts = []
    for start in range(0, len(q), 128):
        sl = slice(start, start+128)
        qn, jn, valid = neighbor_plans(travel, q[sl], j[sl])["arc_unique"]
        n = len(desc)
        pairs, inverse = np.unique((qn*n+jn)[valid], return_inverse=True)
        peaks = np.empty(len(pairs), np.float32)
        for k in range(0, len(pairs), 256):
            ids = pairs[k:k+256]
            delta = desc[ids//n]-desc[ids % n]
            peaks[k:k+256] = -np.sqrt(np.sum(delta*delta, axis=1))
        values = np.zeros(qn.shape, np.float32)
        values[valid] = peaks[inverse]
        weights = repeated_candidate_weights(jn, valid)
        means = path_means(center[sl], values, valid, weights)
        parts.append((1-BETA)*center[sl] + BETA*means.max(0))
    return np.concatenate(parts)


def selftest() -> dict:
    """반대 방향·중복 이웃·접두·경계 동작을 독립 반복문으로 검산한다."""
    rng = np.random.default_rng(20261104)
    desc = rng.normal(size=(95, 16)).astype(np.float32)
    path = np.cumsum(rng.uniform(.1, 2, len(desc)))
    path[20:25] = path[19]
    q, j = np.array([30, 70, 79]), np.array([0, 25, 45])
    center = -np.linalg.norm(desc[q]-desc[j], axis=1).astype(np.float64)
    actual = temporal(desc, path, q, j, center)
    qn, jn, valid = neighbor_plans(path, q, j)["arc_unique"]
    expected = []
    for i in range(len(q)):
        directions = []
        for direction in range(2):
            counts = {int(t): int(((jn[direction, i] == t) & valid[direction, i]).sum())
                      for t in jn[direction, i][valid[direction, i]]}
            total, mass = float(center[i]), 1.
            for k in range(10):
                if valid[direction, i, k]:
                    qi, ji = qn[direction, i, k], jn[direction, i, k]
                    w = 1./counts[int(ji)]
                    total += -float(np.linalg.norm(desc[qi]-desc[ji]))*w
                    mass += w
            directions.append(total/mass)
        expected.append((1-BETA)*center[i] + BETA*max(directions))
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
    end = int(q[-1])+16
    np.testing.assert_array_equal(actual, temporal(desc[:end], path[:end], q, j, center))
    # 동일 특징이면 방향이나 중복 수와 무관하게 0 거리다.
    np.testing.assert_array_equal(temporal(np.zeros_like(desc), path, q, j, np.zeros(3)), np.zeros(3))
    return dict(status="passed", independent_loop=True, repeated_neighbors=True, truncated_prefix=True, zero_distance=True)


def prepare() -> dict:
    """원고의 동일 체크포인트·입력·승자와 고정 시간 판정을 선언한다."""
    path = OUT/"protocol.json"
    if path.exists():
        spec = read(path)
        assert spec["script_sha256"] == digest(Path(__file__))
        return spec
    import _pdv2_bevplace_val_odom as bv
    from _pdv2_bevplace_val_odom_seeds import SELECTION
    from _pdv2_pilot import VAL, TEST
    jobs, sources, checkpoints = [], {}, {}
    def track(p: Path) -> str:
        name = str(p.relative_to(ROOT))
        if name not in sources:
            sources[name] = digest(p)
        return name
    for seed in (1, 2, 3):
        best = read(SELECTION[seed])["best"]
        p = ROOT/best["checkpoint"]
        assert digest(p) == best["sha256"]
        checkpoints[str(seed)] = dict(path=track(p), sha256=best["sha256"], epoch=best["epoch"])
        for role, roster in (("val", VAL), ("test", TEST)):
            for sensor, seq in roster:
                name = f"{sensor}_{seq}"
                if role == "val":
                    ref = (bv.OUT/f"dump_bevplace_{name}.npz" if seed == 1 else
                           ROOT/f"results/pdv2_bevplace_val_odom_seeds_20261101/dump_bevplace_s{seed}_{name}.npz")
                    cache = bv.bev.cache_path("val_odom", sensor, seq)
                else:
                    suffix = "" if seed == 1 else f"_s{seed}"
                    ref = ROOT/f"results/pdv2_bevplace_official{suffix}/odom/dump_bevplace_{name}.npz"
                    cache = bv.bev.cache_path("test_odom", sensor, seq)
                jobs.append(dict(seed=seed, role=role, sensor=sensor, sequence=seq, reference=track(ref), cache=track(cache)))
    for p in [ROOT/"src/retrieval/sequence_evidence.py", ROOT/"scripts/_bevplace_official_dump.py",
              ROOT/"scripts/_bevplace_official_bev.py"]:
        track(p)
    spec = dict(created_unix=time.time(), script_sha256=digest(Path(__file__)), checkpoints=checkpoints,
                beta=BETA, query_window=5, candidate_window=10, decision_delay_keyframes=15,
                score="BEVPlace++ 자체 score_negl2; NSD 스펙트럼·학습기·보정 없음",
                selection="재학습·재선택 없음. NSD와 동일한 arc_unique_b100, 중심 보존과 정역방향 최대. 모든 조건 사전 고정.",
                limitation="후보·승자를 바꾸지 않는 판정 대조. 양쪽 동일 관측 범위이나 학습 구조까지 동일하게 만드는 비교는 아님.",
                test_used_for_selection=False, selftest=selftest(), jobs=jobs, sources=sources)
    write(path, spec)
    return spec


def worker(seed: int, role: str, name: str) -> None:
    """공식 특징을 만들고 기존 최종 원장의 중심 점수를 재현한 뒤 시간 판정을 낸다."""
    spec = prepare()
    job = next(j for j in spec["jobs"] if j["seed"] == seed and j["role"] == role and j["sensor"]+"/"+j["sequence"] == name)
    sensor, seq = job["sensor"], job["sequence"]
    folder = OUT/role/f"s{seed}"
    target = folder/f"{sensor}_{seq}.json"
    if target.exists():
        assert read(target)["status"] == "passed"
        return
    for p in (job["cache"], job["reference"], "src/retrieval/sequence_evidence.py",
              "scripts/_bevplace_official_dump.py", "scripts/_bevplace_official_bev.py"):
        assert digest(ROOT/p) == spec["sources"][p], p
    folder.mkdir(parents=True, exist_ok=True)
    tick = time.time()
    with np.load(ROOT/job["cache"]) as data:
        ids = data["scan_ids"]
        xyz = data["poses"][:, :3, 3]
    with np.load(ROOT/job["reference"]) as data:
        ref = {k: data[k] for k in ("query_idx", "j_star", "top1_geo_dist", "min_past_gt_dist", "score_negl2")}
        reference_meta = json.loads(str(data["meta"]))
    infer_batch = int(reference_meta.get("encoder", {}).get("infer_batch", 32))
    keep = ref["query_idx"] < len(ids)-15
    ref = {k: v[keep] for k, v in ref.items()}
    np.testing.assert_array_equal(ref["query_idx"], np.arange(30, len(ids)-15))
    stored = ROOT/f"results/pdv2_bevplace_val_odom_20261018/descriptors_{sensor}_{seq}.npy"
    cache_prefix = "descriptors" if infer_batch == 32 else f"descriptors_batch{infer_batch}"
    cached = folder/f"{cache_prefix}_{sensor}_{seq}.npy"
    if role == "val" and seed == 1 and stored.exists():
        feature_path = stored
        desc = np.load(stored, mmap_mode="r")
    elif cached.exists():
        feature_path = cached
        desc = np.load(cached, mmap_mode="r")
    else:
        import torch
        from _bevplace_official_dump import _official_model
        from _bevplace_official_bev import BEV, seq_name
        best = spec["checkpoints"][str(seed)]
        checkpoint = ROOT/best["path"]
        assert digest(checkpoint) == best["sha256"]
        torch.set_num_threads(1)
        module, dataset_module, _ = _official_model(checkpoint, 4, infer_batch, "cuda")
        ds = dataset_module.InferDataset(seq=seq_name(sensor, seq), dataset_path=str(BEV/(role+"_odom"))+"/")
        assert [Path(p).name for p in ds.imgs_path] == [f"{sid:06d}.png" for sid in ids]
        desc = module.infer(ds).astype(np.float32)
        assert desc.shape == (len(ids), 8192) and np.isfinite(desc).all()
        np.save(cached, desc)
        feature_path = cached
        del module, ds
        torch.cuda.empty_cache()
    encode_s = time.time()-tick
    assert desc.shape == (len(ids), 8192)
    q, j = ref["query_idx"], ref["j_star"]
    assert (j <= q-30).all()
    center = np.concatenate([-np.linalg.norm(desc[q[k:k+128]]-desc[j[k:k+128]], axis=1)
                             for k in range(0, len(q), 128)])
    error = float(np.max(np.abs(center-ref["score_negl2"])))
    # 추론 배치에 따른 FP32 반올림만 허용하고 원래 중심 판정값은 보존한다.
    np.testing.assert_allclose(center, ref["score_negl2"], atol=2e-5, rtol=2e-5)
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    score = temporal(desc, travel, q, j, ref["score_negl2"])
    for at in (len(q)//3, 2*len(q)//3):
        end = int(q[at])+16
        np.testing.assert_array_equal(temporal(desc[:end], travel[:end], q[at:at+1], j[at:at+1], ref["score_negl2"][at:at+1]), score[at:at+1])
    path = folder/f"{sensor}_{seq}.npz"
    np.savez_compressed(path, **ref, score=score)
    write(target, dict(status="passed", seed=seed, role=role, sequence=name, queries=len(q), center_max_error=error,
                       encode_s=encode_s, infer_batch=infer_batch, elapsed_s=time.time()-tick, truncated_prefix_checks=2,
                       checkpoint=spec["checkpoints"][str(seed)], features=str(feature_path.relative_to(ROOT)),
                       ledger=str(path.relative_to(ROOT)), sha256=digest(path), protocol_sha256=digest(OUT/"protocol.json")))
    print("DONE", seed, role, name, round(time.time()-tick, 1), error, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "selftest", "worker"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--role", choices=("val", "test"), default="val")
    parser.add_argument("--sequence", default="kitti/09")
    args = parser.parse_args()
    if args.action == "worker": worker(args.seed, args.role, args.sequence)
    elif args.action == "selftest": print(selftest())
    else: prepare()
