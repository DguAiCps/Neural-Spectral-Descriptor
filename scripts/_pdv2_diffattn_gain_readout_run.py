#!/usr/bin/env python3
"""출력층 재학습과 구조 대조를 학습·검증에서만 실행한다."""
import concurrent.futures
import subprocess
import sys

import _pdv2_diffattn_gain_readout as e


def main() -> None:
    """공유 검색을 생성하고 동일 예산 학습과 최종 평가를 완료한다."""
    spec = e.prepare()
    logs = e.OUT / "logs"
    logs.mkdir(exist_ok=True)

    def call(args: list, tag: str) -> None:
        with (logs / (tag + ".log")).open("w") as stream:
            subprocess.run(
                [sys.executable, e.__file__, *args],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def jobs(tasks: list, phase: str) -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(call, args, tag) for args, tag in tasks]
            for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
                f.result()
                e.run.WRITE(
                    e.OUT / "status.json", dict(phase=phase, done=i, total=len(tasks))
                )
                print(phase, i, len(tasks), flush=True)

    jobs(
        [
            (
                ["pool", "--role", role, "--sequence", s],
                f"pool_{role}_{s.replace('/','_')}",
            )
            for role in ("train", "val")
            for s in spec[role]
        ],
        "pools",
    )
    for i, arm in enumerate(e.ARMS, 1):
        call(["train", "--arm", arm], "train_" + arm)
        e.run.WRITE(
            e.OUT / "status.json", dict(phase="training", done=i, total=len(e.ARMS))
        )
        print("TRAINED", i, len(e.ARMS), arm, flush=True)
    jobs(
        [(["val", "--sequence", s], "val_" + s.replace("/", "_")) for s in spec["val"]],
        "validation",
    )
    e.finish()
    e.run.WRITE(e.OUT / "status.json", dict(phase="complete", test_used=False))


if __name__ == "__main__":
    main()
