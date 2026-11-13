#!/usr/bin/env python3
"""주파수 키 대조의 캐시·학습·검증동결·시험·보고를 순서대로 실행한다."""
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_spectral_keys_20261025"
LOG = REPO / "logs/spectral_keys_20261025"
ARMS = ("octave288", "trunc288", "trunc352")
SEEDS = (1, 2, 3)
EPOCHS = (10, 20)


def read(path: Path) -> dict:
    """저장된 작업 상태를 읽는다."""
    return json.loads(path.read_text())


def invoke(script: str, args: list, name: str) -> None:
    """실패 시 기록을 보존하고 뒤 작업을 실행하지 않는다."""
    LOG.mkdir(parents=True, exist_ok=True)
    path = LOG / (name + ".log")
    started = time.time()
    with path.open("a") as stream:
        result = subprocess.run(
            [sys.executable, "-u", "scripts/" + script, *map(str, args)],
            cwd=REPO,
            env=os.environ.copy(),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(name + "\n" + path.read_text()[-6000:])
    print("COMPLETE", name, round(time.time() - started, 1), flush=True)


def run_jobs(jobs: list, phase: str) -> None:
    """메모리와 CPU 경합을 제한해 네 평가 작업을 병행한다."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(invoke, "_pdv2_spectral_keys.py", args, name)
            for args, name in jobs
        ]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            (OUT / "run_status.json").write_text(
                json.dumps(
                    {
                        "phase": phase,
                        "done": count,
                        "total": len(jobs),
                        "updated_unix": time.time(),
                    },
                    indent=2,
                )
                + "\n"
            )


def main() -> None:
    """학습/검증 캐시 완료를 기다려 이미 고정한 전체 대조를 수행한다."""
    spec = read(OUT / "cache_protocol.json")
    cache_root = REPO / "data/pdv2_spectral_keys_20261025"
    expected = [
        cache_root / role / (s.replace("/", "_") + ".json")
        for role in ("train",)
        for s in spec[role]
    ]
    while not all(p.exists() for p in expected):
        print(
            "WAITING_TRAIN_CACHE",
            sum(p.exists() for p in expected),
            len(expected),
            flush=True,
        )
        time.sleep(30)
    invoke("_pdv2_spectral_keys.py", ["prepare"], "prepare")
    for seed in SEEDS:
        for arm in ARMS:
            invoke(
                "_pdv2_spectral_keys.py",
                ["train", "--seed", seed, "--arm", arm],
                f"train_s{seed}_{arm}",
            )
    expected = [
        cache_root / "val" / (s.replace("/", "_") + ".json") for s in spec["val"]
    ]
    while not all(p.exists() for p in expected):
        print(
            "WAITING_VAL_CACHE",
            sum(p.exists() for p in expected),
            len(expected),
            flush=True,
        )
        time.sleep(30)
    jobs = []
    for seed in SEEDS:
        for arm in ARMS:
            for epoch in EPOCHS:
                for sequence in spec["val"]:
                    args = [
                        "eval",
                        "--seed",
                        seed,
                        "--arm",
                        arm,
                        "--epoch",
                        epoch,
                        "--role",
                        "val",
                        "--sequence",
                        sequence,
                    ]
                    jobs.append(
                        (
                            args,
                            f"val_s{seed}_{arm}_e{epoch}_" + sequence.replace("/", "_"),
                        )
                    )
    run_jobs(jobs, "validation")
    invoke("_pdv2_spectral_keys.py", ["select"], "select_and_freeze")
    invoke("_pdv2_spectral_keys_cache.py", ["run", "--role", "test"], "test_cache")
    frozen = read(OUT / "test_freeze.json")
    jobs = []
    for seed in SEEDS:
        for arm in ARMS:
            epoch = frozen["models"][str(seed)][arm]["epoch"]
            for sequence in spec["test"]:
                args = [
                    "eval",
                    "--seed",
                    seed,
                    "--arm",
                    arm,
                    "--epoch",
                    epoch,
                    "--role",
                    "test",
                    "--sequence",
                    sequence,
                ]
                jobs.append((args, f"test_s{seed}_{arm}_" + sequence.replace("/", "_")))
    run_jobs(jobs, "test")
    invoke("_pdv2_spectral_keys_finish.py", [], "final_report")
    print("ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
