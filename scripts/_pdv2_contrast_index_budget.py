#!/usr/bin/env python3
"""현재 NSD의 색인 선택을 같은 후보 수·상한에서 최종 판정까지 비교한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _pdv2_contrast_pipeline_test as original
from _pdv2_input_alignment_lib import make_graph
from gnn.contrast_spectral_difference import ContrastSpectralDifference
from gnn.spectral_transport import attach_transport, lidar_yaw

pipeline, run = original.pipeline, original.run
MAIN = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
OUT = ROOT / "results/pdv2_contrast_index_budget_20261106"
ARMS = ("key_200", "aux_200", "dual_100_100", "key_matched", "aux_matched")


def read(path: Path) -> dict:
    """JSON을 읽는다."""
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    """파일 해시를 반환한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    """완성된 JSON을 원자적으로 교체한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def topk(scores: np.ndarray, count: int) -> np.ndarray:
    """기존 prefix_topk의 동점·정렬 규칙을 그대로 사용한다."""
    count = min(count, len(scores))
    ids = (np.argpartition(-scores, count - 1)[:count]
           if count < len(scores) else np.arange(len(scores)))
    return ids[np.argsort(-scores[ids], kind="stable")]


def candidates(key_scores: np.ndarray, aux_scores: np.ndarray) -> dict:
    """200개 상한과 실제 합집합 크기를 맞춘 다섯 후보 집합을 만든다."""
    union = np.union1d(topk(key_scores, 100), topk(aux_scores, 100))
    count = len(union)
    return {
        "key_200": np.sort(topk(key_scores, 200)),
        "aux_200": np.sort(topk(aux_scores, 200)),
        "dual_100_100": union,
        "key_matched": np.sort(topk(key_scores, count)),
        "aux_matched": np.sort(topk(aux_scores, count)),
    }


def model(seed: int, device: str) -> torch.nn.Module:
    """현재 표2의 고정 가중치를 같은 구조에 엄격히 불러온다."""
    case = read(MAIN / "protocol.json")["cases"][f"diff_s{seed}"]
    path = ROOT / case["checkpoint"]
    assert digest(path) == case["checkpoint_sha256"]
    net = ContrastSpectralDifference(run.source.factory(288), "diff")
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)
                        ["model_state_dict"], strict=True)
    return net.to(device).eval()


def graph_for(cache: dict, spectrum: np.ndarray, device: str):
    """정답 자세를 사용하지 않는 기존 시간 그래프를 만든다."""
    seq = np.zeros(len(spectrum), np.int64)
    graph = make_graph(cache["descriptors"], cache["poses"], seq).to(device)
    attach_transport(graph, spectrum, lidar_yaw(cache["poses"]), seq)
    return graph


def source_paths() -> list:
    """동결할 구현·학습 상태·통계를 모은다."""
    old = read(MAIN / "protocol.json")
    paths = [Path(__file__), MAIN / "protocol.json", MAIN / "summary.json",
             ROOT / "src/gnn/contrast_spectral_difference.py",
             ROOT / "scripts/_pdv2_input_alignment_lib.py",
             run.white.OUT / "statistics.npz",
             run.source.OUT / "remetric_octave288.npy"]
    paths += [ROOT / old["cases"][f"diff_s{s}"]["checkpoint"] for s in (1, 2, 3)]
    paths += [Path(pipeline.__file__), Path(original.__file__), Path(run.base.__file__)]
    return paths


def freeze(device: str) -> dict:
    """시험 전에 비교군·질의·집계·허용 오차를 고정한다."""
    path = OUT / "protocol.json"
    if path.exists():
        spec = read(path)
        assert spec["device"] == device
        for p, h in spec["sources"].items():
            assert digest(ROOT / p) == h, p
        return spec
    old = read(MAIN / "protocol.json")
    spec = dict(created_unix=time.time(), device=device, seeds=[1, 2, 3],
        arms=list(ARMS), test=old["test"], validation=["kitti360/0004"],
        candidate_policy="원키200·보조200·원키100+보조100(중복미보충), 질의별합집합실제K와같은원키K·보조K. 적격과거 j<=q-30.",
        fixed="현재표2의diff_s1/2/3가중치·두원색인·모든4융합채널·융합계수·정렬·최종시간판정·q+15유지. 검색후보생성만교체. 재학습·검증재선택·시험선택없음.",
        interpretation="기존파이프라인안에서검색색인의조건부기여. 색인삭제후스코어채널까지삭제한최적독립방법대조가아님.",
        aggregation="양성있는45시퀀스->6도시균등->3시작점평균, 표2와같은195726질의·4최종지표. 후보정답포함률은같은양성분모에서별도보고.",
        verification="검증에서원두색인후보목록·승자정확일치, 최종점수최대오차8e-6이하. 시험도모든시드의원두색인승자와원장을재현. 독립지표2구현1e-12이하.",
        score_atol=8e-6, sources={str(p.relative_to(ROOT)): digest(p) for p in source_paths()})
    write(path, spec)
    return spec


@torch.no_grad()
def evaluate(sequence: str, role: str, device: str, sample: int = 0) -> dict:
    """같은 후보 예산의 최종 승자·수락 점수·후보 포함률을 계산한다."""
    started = time.time()
    original.activate()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    name = sequence.replace("/", "_")
    folder = OUT / ("pilot" if sample else role) / name
    folder.mkdir(parents=True, exist_ok=True)
    if not sample and (folder / "summary.json").exists():
        previous = read(folder / "summary.json")
        if "candidate_pools_sha256" in previous:
            assert digest(folder / "candidate_pools.npz") == previous["candidate_pools_sha256"]
            return previous
    if role == "test":
        spec = freeze(device)
        assert sequence in spec["test"]
        assert read(OUT / "preflight.json")["status"] == "passed"
    cache, pos, minimum, travel, bank, _, _, paths = run.inputs(role, sequence)
    raw = pipeline.raw_key(cache["descriptors"])
    z0 = run.white.coefficients(paths[1])
    graph = graph_for(cache, z0, device)
    source_folder = (MAIN if role == "test" else original.PARENT) / role / name
    ref = run.ARR(source_folder / "diff_s1.npz")
    queries = ref["query_idx"]
    indices = (np.unique(np.linspace(0, len(queries)-1, sample).astype(int))
               if sample else np.arange(len(queries)))
    queries = queries[indices]
    old_pool = run.ARR(source_folder / "candidate_pool.npz")
    weights = run.cal.fusion_weights(read(MAIN / "protocol.json")["confidence"])[
        run.cal.FUSIONS.index("aux_half")]
    pools = []
    sizes = {a: [] for a in ARMS}
    hits = {a: [] for a in ARMS}
    for at, q in zip(indices, queries):
        group = candidates(raw[:q-29] @ raw[q], bank[0][:q-29] @ bank[0][q])
        lo, hi = old_pool["offset"][at:at+2]
        np.testing.assert_array_equal(group["dual_100_100"], old_pool["ids"][lo:hi])
        for arm, ids in group.items():
            assert len(ids) <= 200 and (ids <= q-30).all()
            sizes[arm].append(len(ids))
            hits[arm].append(bool((np.linalg.norm(pos[ids]-pos[q], axis=1) <= 5).any()))
        assert len(group["key_matched"]) == len(group["aux_matched"]) == len(group["dual_100_100"])
        pools.append(group)
    stage_time = {"input_and_candidates": time.time()-started}
    saved_pools = {"query_idx": queries}
    for arm in ARMS:
        saved_pools[arm + "_offset"] = np.r_[0, np.cumsum(sizes[arm])].astype(np.int64)
        saved_pools[arm + "_ids"] = np.concatenate([p[arm] for p in pools]).astype(np.int32)
    pool_path = folder / "candidate_pools.npz"
    np.savez_compressed(pool_path, **saved_pools)
    records, checks = {}, []
    for seed in (1, 2, 3):
        seed_started = time.time()
        net = model(seed, device)
        z = net(graph)
        changed = run.base.variant_inputs(z.cpu().numpy(), 8)
        selected = {a: [] for a in ARMS}
        for count, (q, group) in enumerate(zip(queries, pools)):
            all_ids = np.unique(np.concatenate(list(group.values())))
            spectra = run.base.align_scores(changed[1], changed[2], int(q), all_ids)
            columns = run.base.column_align_scores(changed[3], int(q), all_ids)
            for arm, ids in group.items():
                locations = np.searchsorted(all_ids, ids)
                value = (weights[0]*run.base.minmax(raw[ids] @ raw[q])
                         + weights[1]*run.base.minmax(bank[0][ids] @ bank[0][q])
                         + weights[2]*run.base.minmax(spectra[locations])
                         + weights[3]*run.base.minmax(columns[locations]))
                selected[arm].append(int(ids[np.argmax(value)]))
            if (count+1) % 512 == 0:
                write(folder / "progress.json", dict(stage="alignment", seed=seed,
                    completed=count+1, total=len(queries), updated_unix=time.time()))
        reference = run.ARR(source_folder / f"diff_s{seed}.npz")
        for field in ("query_idx", "min_past_gt_dist"):
            np.testing.assert_array_equal(reference[field][indices], ref[field][indices])
        winners = np.asarray([selected[a] for a in ARMS], np.int64).T
        np.testing.assert_array_equal(winners[:, 2], reference["j_star"][indices])
        # 같은 질의·후보 쌍은 한 번만 계산하고 각 조건의 순서로 되돌린다.
        packed, inverse = np.unique((queries[:, None]*len(raw)+winners).ravel(),
                                    return_inverse=True)
        plan = pipeline.e.prior.base.prior.plan(travel, packed//len(raw), packed%len(raw))
        values = []
        for start in range(0, len(packed), 128):
            pp = {k: torch.from_numpy(v[start:start+128]).to(device) for k, v in plan.items()}
            values.append(pipeline.e.prior.score(z, pp).cpu().numpy())
        scores = np.concatenate(values).astype(np.float64)[inverse].reshape(winners.shape)
        err = float(np.max(abs(scores[:, 2]-reference["score"][indices])))
        assert err <= 8e-6, (sequence, seed, err)
        seed_checks = dict(seed=seed, winners_exact=True, max_score_error=err)
        checks.append(seed_checks)
        for column, arm in enumerate(ARMS):
            row = dict(query_idx=queries, j_star=winners[:, column],
                min_past_gt_dist=minimum[queries],
                top1_geo_dist=np.linalg.norm(pos[queries]-pos[winners[:, column]], axis=1),
                group=reference["group"][indices], score=scores[:, column],
                candidate_count=np.asarray(sizes[arm], np.int64),
                candidate_hit=np.asarray(hits[arm], bool))
            assert np.isfinite(row["score"]).all()
            actual = run.base.independent_metrics(row, row["score"])
            other = run.white.prior.BASE.exp.metrics(row["score"], row)
            for metric in run.base.METRICS:
                if actual[metric] is not None:
                    np.testing.assert_allclose(actual[metric], other[metric], atol=1e-12, rtol=0)
            positive = row["min_past_gt_dist"] <= 5
            hit_rate = float(np.mean(row["candidate_hit"][positive])) if positive.any() else None
            path = folder / f"{arm}_s{seed}.npz"
            np.savez_compressed(path, **row)
            records[f"{arm}_s{seed}"] = dict(metrics=actual,
                candidate_recall=hit_rate, candidate_mean=float(np.mean(row["candidate_count"])),
                path=str(path.relative_to(ROOT)), sha256=digest(path))
        stage_time[f"seed{seed}"] = time.time()-seed_started
        print("SEED_DONE", role, sequence, seed, stage_time[f"seed{seed}"], err, flush=True)
        del net, z, changed
        if device == "cuda":
            coefficient_cache = pipeline.e.prior.coefficients.__globals__.get("COEFFICIENT_CACHE")
            if coefficient_cache is not None:
                coefficient_cache[:] = [None, None]
            torch.cuda.empty_cache()
    result = dict(status="passed", role=role, sequence=sequence, device=device,
        queries=len(queries), records=records, reproduction=checks,
        timing_s=stage_time, seconds=time.time()-started,
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths},
        candidate_pools_sha256=digest(pool_path),
        source_protocol_sha256=digest(MAIN / "protocol.json"))
    write(folder / "summary.json", result)
    return result


def campaign(device: str, workers: int) -> None:
    """검증 재현 완료 뒤 동결한 시험 시퀀스를 순서대로 실행한다."""
    OUT.mkdir(exist_ok=True)
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__)), "worker", "--device", device]
    with (logs / "preflight.log").open("a") as log:
        subprocess.run(command + ["--role", "val", "--sequence", "kitti360/0004"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    proof = read(OUT / "val/kitti360_0004/summary.json")
    write(OUT / "preflight.json", dict(status="passed", sequence=proof["sequence"],
          queries=proof["queries"], reproduction=proof["reproduction"],
          implementation_sha256=digest(Path(__file__))))
    spec = freeze(device)
    def launch(sequence: str) -> str:
        with (logs / (sequence.replace("/", "_")+".log")).open("a") as log:
            subprocess.run(command + ["--role", "test", "--sequence", sequence],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        return sequence
    completed = []
    write(OUT / "status.json", dict(status="running", completed=completed,
          total=len(spec["test"]), workers=workers, updated_unix=time.time()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(launch, s): s for s in spec["test"]}
        for job in concurrent.futures.as_completed(jobs):
            completed.append(job.result())
            write(OUT / "status.json", dict(status="running", completed=completed,
                  total=len(spec["test"]), workers=workers, updated_unix=time.time()))
    write(OUT / "status.json", dict(status="measurements_complete", updated_unix=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pilot", "worker", "run"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--role", choices=("val", "test"), default="val")
    parser.add_argument("--sequence", default="kitti360/0004")
    parser.add_argument("--sample", type=int, default=128)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.action == "run":
        campaign(args.device, args.workers)
    else:
        evaluate(args.sequence, args.role, args.device, args.sample if args.action == "pilot" else 0)
