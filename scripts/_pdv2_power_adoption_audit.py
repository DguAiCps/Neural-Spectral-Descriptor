#!/usr/bin/env python3
"""새 주모델 표4의 각 제거 조건을 접두 입력에서 따로 재검산한다."""
from __future__ import annotations

import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import _pdv2_power_adoption_ablation as run

OUT = run.OUT / "prefix_audit"


def worker(seed: int, sequence: str) -> None:
    """각 대조의 두 질의를 실제 q+15 접두 뱅크로 재계산한다."""
    power, base, white = run.power, run.base, run.white
    spec = power.read(run.OUT / "protocol.json")
    name = sequence.replace("/", "_")
    target = OUT / f"s{seed}_{name}.json"
    if target.exists():
        assert power.read(target)["status"] == "passed"
        return
    _, _, paths, _ = white.prior.BASE.exp.old.base.inputs("test", sequence)
    cache = power.arrays(paths["cache"])
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    x = white.coefficients(paths["spec"])
    matrix = power.arrays(white.OUT / "statistics.npz")["full"]
    metadata = power.read(run.OUT / f"s{seed}" / (name + ".json"))
    ledgers = {arm: power.arrays(run.REPO / record["path"]) for arm, record in metadata["arms"].items()}
    q = ledgers["NSD"]["query_idx"]
    records = []
    for at in (len(q) // 3, 2 * len(q) // 3):
        end = int(q[at]) + 16
        xx, wx = x[:end], white.transform(x[:end], matrix)
        banks = {p: [base.variant_inputs(power.transform(v, p), 8) for v in (xx, wx)] for p in (1., .5)}
        for arm in run.ARMS:
            row = ledgers[arm]
            power_value = 1. if arm == "without_amplitude_compression" else .5
            result = run.scores(*banks[power_value], travel[:end], q[at:at+1], row["j_star"][at:at+1], arm == "no_alignment")
            value = result["original" if arm == "without_correlation_correction" else "score"]
            np.testing.assert_array_equal(value, row["score"][at:at+1])
            records.append({"arm": arm, "query": int(q[at]), "score_exact": True})
    power.write_json(target, {"status": "passed", "checks": records,
        "source_sha256": power.sha256(Path(__file__)), "protocol_sha256": power.sha256(run.OUT / "protocol.json")})


def main() -> None:
    """모든 대조·시드의 접두 결과를 분리해서 확인한다."""
    OUT.mkdir(exist_ok=True)
    spec = run.prepare()
    jobs = [(s, q) for s in spec["seeds"] for q in spec["test"]]
    def invoke(item: tuple) -> None:
        s, q = item
        deadline = time.monotonic() + 1800
        source = run.OUT / f"s{s}" / (q.replace("/", "_") + ".json")
        while not source.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(str(source))
            time.sleep(1)
        subprocess.run([sys.executable, str(Path(__file__)), str(s), q], check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(invoke, jobs))
    results = [run.power.read(OUT / f"s{s}_{q.replace('/', '_')}.json") for s, q in jobs]
    count = sum(len(r["checks"]) for r in results)
    assert count == 2208
    run.power.write_json(OUT / "completion.json", {"status": "passed", "checks": count, "sequences_and_seeds": len(jobs),
        "scope": "각 제거 조건 자체의 최종 점수를 q+15 접두 입력으로 검산. 공통 p=.5 점수의 대조군 간 재사용 검사와 구분한다."})
    print("PREFIX_AUDIT_PASSED", count, flush=True)


if __name__ == "__main__":
    worker(int(sys.argv[1]), sys.argv[2]) if len(sys.argv) == 3 else main()
