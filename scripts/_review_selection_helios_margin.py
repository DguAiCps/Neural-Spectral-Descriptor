#!/usr/bin/env python3
"""로컬에서 계산한 HeLiOS 선택 상위 후보를 시험 전에 서버에서 재검산한다."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(ROOT / "src")]
from _review_selection_bev import VAL, read, write, sha

OUT = ROOT / "results/pdv2_review_selection_full_20261105"


def needed(summary: dict, assigned: set, threshold: float) -> list:
    """두 목적의 최고값에 가까운 로컬 후보를 중복 없이 반환한다."""
    result = set()
    for seed, rows in summary.items():
        for objective in ("AP", "R@1"):
            best = max(row["macro"][objective] for row in rows.values())
            for name, row in rows.items():
                if (seed, name) in assigned and best - row["macro"][objective] <= threshold:
                    result.add((seed, name))
    return sorted(result)


def ensure_server_reference(summary: dict) -> None:
    """현재 상위 로컬 후보의 검증9가 서버에서 끝나야 선택을 동결한다."""
    plan_path = OUT / "local_helios_plan.json"
    if not plan_path.exists():
        return
    plan = read(plan_path)
    assert sha(OUT / "retained_protocol.json") == plan["protocol_sha256"]
    assert plan["selection_numerical_guard"]["threshold"] == 1e-4
    assigned = {(str(job["seed"]), job["candidate"]) for job in plan["jobs"]}
    target = OUT / "helios_numerical_override.json"
    overrides = read(target).get("records", {}) if target.exists() else {}
    required = needed(summary, assigned, 1e-4)
    pending = [(seed, name) for seed, name in required if name not in overrides.get(seed, {})]
    request = dict(updated_unix=time.time(), required=required, pending=pending,
        plan_sha256=sha(plan_path), threshold=1e-4, test_read=False,
        rule="최고 로컬 후보와 최고값의1e-4 이내 로컬 후보를 같은 서버GPU0로 검증9 재평가")
    write(OUT / "helios_numerical_margin_request.json", request)
    if not pending:
        write(OUT / "helios_numerical_margin_verified.json", dict(request, complete=True))
        return
    seed, name = pending[0]
    home = OUT / "helios_numerical_margin/val/helios" / f"s{seed}" / name
    records = {}
    for sensor, seq in VAL:
        result = home / f"{sensor}_{seq}.json"
        if result.exists():
            record = read(result)
            assert sha(ROOT / record["ledger"]) == record["ledger_sha256"]
            records[f"{sensor}/{seq}"] = str(result.relative_to(ROOT))
            continue
        env = dict(os.environ, REVIEW_SELECTION_EXECUTOR=plan["executor"], SEED=seed,
                   NVIDIA_TF32_OVERRIDE="0")
        code = subprocess.call([sys.executable, "-u", str(Path(__file__)), "worker",
            "--seed", seed, "--candidate", name, "--sequence", f"{sensor}/{seq}"], env=env)
        assert code in (0, 75), code
        raise SystemExit(75)
    overrides.setdefault(seed, {})[name] = records
    write(target, dict(records=overrides, plan_sha256=sha(plan_path), test_read=False,
        updated_unix=time.time(), execution="server GPU0; same official model/input/scorer"))
    # 서버값으로 집계와 필요 후보를 다시 계산한 다음에만 동결한다.
    raise SystemExit(75)


def worker(args) -> None:
    """원래 평가기를 별도 원장 폴더에서 재개한다."""
    import _review_selection_full_eval as evaluator
    evaluator.folder = lambda method, seed, name, scope, role: (
        OUT / "helios_numerical_margin" / role / method / f"s{seed}" / name)
    evaluator.main(argparse.Namespace(method="helios", seed=args.seed, candidate=args.candidate,
        scope="retained", role="val", sequence=args.sequence, query_budget=128,
        query_offset=0, forward_budget=2048))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("worker",))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--sequence", required=True)
    worker(parser.parse_args())
