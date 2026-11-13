#!/usr/bin/env python3
"""검증에서 유효했던 국소 후보 보정의 반경1·2·5만 비교한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import _pdv2_minimal_rerank as parent

REPO = parent.REPO
OUT = REPO / "results/pdv2_local_refine_20261101"
LOG = REPO / "logs/local_refine_20261101"
ARMS = ("baseline", "radius1", "radius2", "radius5", "radius1_cap", "radius2_cap", "radius5_cap")
METRICS, FIELDS, white = parent.METRICS, parent.FIELDS, parent.white
read, arrays, sha256, write_json = parent.read, parent.arrays, parent.sha256, parent.write_json
independent_metrics, reference_path = parent.independent_metrics, parent.reference_path
source_folder = parent.source_folder


def prepare() -> dict:
    """첫 검증의 양성 결과를 본 후 추가한 두 반경임을 기록한다."""
    OUT.mkdir(exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        spec = read(path)
        for p, h in spec["sources"].items():
            assert sha256(REPO / p) == h, p
        return spec
    old = parent.prepare()
    summary = read(parent.OUT / "summary.json")
    assert summary["selected"] == "local5"
    assert not (parent.OUT / "test_freeze.json").exists()
    paths = {Path(__file__), Path(parent.__file__), parent.OUT / "protocol.json",
             parent.OUT / "summary.json", parent.OUT / "completion_validation.json"}
    spec = {k: old[k] for k in ("val", "seeds", "models", "confidence", "selection", "test_exposure")}
    spec.update({"created_unix": time.time(), "arms": ARMS, "radii": [1, 2, 5],
        "scope": "검증1차에서±2보정의3시드AP+.4026%p를확인한후추가한반경민감도대조. 시험미사용상태에서±1,±5고정. 아키텍처/가중치/통계/지연불변. GT는평가만사용.",
        "rule": "기존승자j와j±r범위의유효j<=q-30후보중현재75:25시간점수최대. 동률은기존승자우선후작은인덱스. cap은수락점수만기존승자점수이하로제한. 반경2는첫단계의정확재현대조.",
        "campaign_selection": "첫단계와같은승격기준통과조건중AP최대. 첫단계우승보다좋을때만후속선택으로사용. 두단계전체에서최대한설정만시험46x3평가. 반경을시험에서고르지않는다.",
        "sources": {**old["sources"], **{str(p.relative_to(REPO)): sha256(p) for p in sorted(paths)}}})
    write_json(path, spec)
    return spec


def indices(q: np.ndarray, center: np.ndarray, radius: int) -> np.ndarray:
    """기존 승자를 먼저 두고 유효한 주변 프레임만 나열한다."""
    offsets = np.r_[0, np.arange(-radius, 0), np.arange(1, radius + 1)]
    ids = center[:, None] + offsets
    return np.where((ids >= 0) & (ids <= q[:, None] - 30), ids, center[:, None])


def worker(seed: int, sequence: str, role: str = "val") -> None:
    """이웃 탐색만 수행하고 최종 후보·점수·검산을 저장한다."""
    spec = read(OUT / "protocol.json")
    assert sha256(Path(__file__)) == spec["sources"][str(Path(__file__).relative_to(REPO))]
    arms = ARMS if role == "val" else ("baseline", read(OUT / "test_freeze.json")["selected"])
    radius_max = 5 if role == "val" else int(arms[1].split("_")[0].replace("radius", ""))
    folder = OUT / role / f"s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    name = sequence.replace("/", "_")
    target = folder / (name + ".json")
    if target.exists():
        assert read(target)["status"] == "passed"
        return
    ref_path = reference_path(seed, role, name + ".npz")
    ref = arrays(ref_path)
    _, _, paths, _ = white.prior.BASE.exp.old.base.inputs(role, sequence)
    cache = arrays(paths["cache"])
    x = white.coefficients(paths["spec"])
    bank = parent.variant_inputs(x, 8)
    corrected = parent.variant_inputs(white.transform(x, arrays(white.OUT / "statistics.npz")["full"]), 8)
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    positions = cache["poses_gt"][:, :3, 3]
    q = ref["query_idx"]
    ids = indices(q, ref["j_star"], radius_max)
    conf = spec["confidence"][str(seed)]
    tick = time.perf_counter()
    pair, inverse = np.unique((q[:, None] * len(x) + ids).ravel(), return_inverse=True)
    values = parent.scores_for(bank, corrected, travel, pair // len(x), pair % len(x), conf["decision"])
    value = values[inverse].reshape(ids.shape)
    seconds = time.perf_counter() - tick
    np.testing.assert_allclose(value[:, 0], ref["score"], atol=1e-7, rtol=0)
    rows, changes = {}, {}
    offsets = np.r_[0, np.arange(-radius_max, 0), np.arange(1, radius_max + 1)]
    for arm in arms:
        radius = 0 if arm == "baseline" else int(arm.split("_")[0].replace("radius", ""))
        columns = np.flatnonzero(abs(offsets) <= radius)
        chosen = columns[value[:, columns].argmax(1)]
        winner = ids[np.arange(len(q)), chosen]
        score = value[np.arange(len(q)), chosen]
        if arm.endswith("_cap"):
            score = np.minimum(score, value[:, 0])
        row = {k: ref[k] for k in FIELDS}
        row["j_star"] = winner
        row["top1_geo_dist"] = np.linalg.norm(positions[q] - positions[winner], axis=1)
        row["score"] = score
        metrics = independent_metrics(row, score)
        other = white.prior.BASE.exp.metrics(score, row)
        for m in METRICS:
            if metrics["n_pos"] and metrics[m] is not None:
                np.testing.assert_allclose(metrics[m], other[m], atol=1e-12, rtol=0)
        if arm == "baseline" or (role == "val" and arm in ("radius2", "radius2_cap")):
            prior = ref if arm == "baseline" else arrays(parent.OUT / "val" / f"s{seed}" / f"{name}_{'local5_cap' if arm.endswith('_cap') else 'local5'}.npz")
            np.testing.assert_array_equal(winner, prior["j_star"])
            np.testing.assert_allclose(score, prior["score"], atol=1e-7, rtol=0)
            expected = independent_metrics(prior, prior["score"])
            for m in METRICS:
                if metrics[m] is not None:
                    np.testing.assert_allclose(metrics[m], expected[m], atol=1e-10, rtol=0)
        positive = ref["min_past_gt_dist"] <= 5
        a = positive & (ref["top1_geo_dist"] <= 5)
        b = positive & (row["top1_geo_dist"] <= 5)
        changes[arm] = {"winner_changed": int((winner != ref["j_star"]).sum()),
                       "gained_correct": int((b & ~a).sum()), "lost_correct": int((a & ~b).sum())}
        p = folder / f"{name}_{arm}.npz"
        np.savez_compressed(p, **row)
        rows[arm] = {"metrics": metrics, "path": str(p.relative_to(REPO)), "sha256": sha256(p)}
    prefixes = []
    for i in (len(q) // 3, 2 * len(q) // 3):
        end, t = int(q[i]) + 16, int(q[i])
        b = white.prior.BASE.exp.truncate_bank(bank, end)
        c = white.prior.BASE.exp.truncate_bank(corrected, end)
        tiny = parent.scores_for(b, c, travel[:end], np.full(ids.shape[1], t), ids[i], conf["decision"])
        np.testing.assert_array_equal(tiny, value[i])
        prefixes.append({"query": t, "scores_and_choices_equal": True})
    candidate_path = folder / f"{name}_candidates.npz"
    np.savez_compressed(candidate_path, query_idx=q, indices=ids, score=value)
    write_json(target, {"status": "passed", "seed": seed, "sequence": sequence, "role": role,
        "rows": len(q), "arms": rows, "changes": changes, "prefix_checks": prefixes,
        "baseline_score_max_error": float(abs(value[:, 0] - ref["score"]).max()),
        "all_candidates_seconds": seconds, "candidate_sha256": sha256(candidate_path),
        "inputs": {str(p.relative_to(REPO)): sha256(p) for p in (ref_path, *paths.values())}})
    print("LOCAL", seed, sequence, round(seconds, 1), flush=True)


def finish() -> None:
    """첫 단계와 같은 검산·선택식을 재사용한다."""
    old_out, old_arms = parent.OUT, parent.ARMS
    try:
        parent.OUT, parent.ARMS = OUT, ARMS
        parent.finish()
    finally:
        parent.OUT, parent.ARMS = old_out, old_arms


def run() -> None:
    """서로 다른 검증 원장을 세 작업씩 평가한다."""
    spec = prepare()
    LOG.mkdir(parents=True, exist_ok=True)
    def invoke(item: tuple) -> None:
        seed, seq = item
        with (LOG / f"s{seed}_{seq.replace('/', '_')}.log").open("w") as log:
            subprocess.run([sys.executable, str(Path(__file__)), "worker", "--seed", str(seed),
                            "--sequence", seq], stdout=log, stderr=subprocess.STDOUT, check=True)
    tick = time.time()
    jobs = [(s, q) for s in spec["seeds"] for q in spec["val"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(invoke, item): item for item in jobs}
        for n, f in enumerate(concurrent.futures.as_completed(pending), 1):
            f.result()
            write_json(OUT / "status.json", {"status": "running", "done": n, "total": len(jobs)})
            print("PROGRESS", n, len(jobs), pending[f], flush=True)
    finish()
    write_json(OUT / "status.json", {"status": "complete", "elapsed_s": time.time() - tick})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "worker", "finish"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sequence", default="kitti360/0004")
    parser.add_argument("--role", choices=("val", "test"), default="val")
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.seed, args.sequence, args.role)
    else:
        globals()[args.action]()
