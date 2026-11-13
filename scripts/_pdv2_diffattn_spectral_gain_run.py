#!/usr/bin/env python3
"""스펙트럼 이득 실험을 중단 가능한 로그와 함께 순서대로 실행한다."""
import concurrent.futures
import subprocess
import sys

import _pdv2_diffattn_spectral_gain as experiment


def main() -> None:
    """두 학습과 독립 시퀀스 평가를 완료한 뒤 결과를 집계한다."""
    e = experiment
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

    for arm in e.ARMS:
        call(["train", "--arm", arm], "train_" + arm)
    jobs = [
        (a, ep, s)
        for a, epochs in [("frozen", (0,))] + [(a, e.EPOCHS) for a in e.ARMS]
        for ep in epochs
        for s in spec["val"]
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                call,
                ["val", "--arm", a, "--epoch", str(ep), "--sequence", s],
                f"val_{a}_{ep}_{s.replace('/', '_')}",
            )
            for a, ep, s in jobs
        ]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            e.run.WRITE(
                e.OUT / "status.json", dict(phase="validation", done=i, total=len(jobs))
            )
            print("VALIDATED", i, len(jobs), flush=True)
    e.finish()
    e.run.WRITE(e.OUT / "status.json", dict(phase="complete", test_used=False))


if __name__ == "__main__":
    main()
