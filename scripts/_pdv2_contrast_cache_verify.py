#!/usr/bin/env python3
"""추론 중복 계산 제거가 첫 시드의 모든 선택 원장을 재현하는지 검사한다."""
from pathlib import Path

import numpy as np
import torch

import _pdv2_diffattn_contrast_seeds as e


def main() -> None:
    """같은 저장 모델을 캐시 실행으로 재평가해 원 점수와 비교한다."""
    spec = e.prepare()
    tags = e.e.run.READ(e.SOURCE / "summary.json")["best"]
    records = []
    with torch.no_grad():
        for sequence in spec["val"]:
            name = sequence.replace("/", "_")
            graph, _, _ = e.e.prior.graph_inputs("val", sequence)
            data = e.e.run.ARR(e.e.prior.PARENT / "pools/val" / name / "pool.npz")
            old = e.e.run.ARR(e.SOURCE / "val" / (name + ".npz"))
            p = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
            for kind, tag in tags.items():
                arm, epoch = tag.rsplit("_", 1)
                net = e.model(kind, e.SOURCE / arm / f"epoch{epoch}.pth")
                z = net(graph)
                values = []
                for start in range(0, len(p["query"]), 128):
                    pp = {
                        k: torch.from_numpy(v[start : start + 128]).cuda()
                        for k, v in p.items()
                    }
                    values.append(e.e.prior.score(z, pp).cpu().numpy())
                value = np.concatenate(values).astype(np.float64)
                error = float(abs(value - old[tag]).max())
                assert error < 8e-6, (sequence, tag, error)
                records.append(
                    dict(
                        sequence=sequence,
                        condition=tag,
                        rows=len(value),
                        max_error=error,
                    )
                )
                del net, z
            del graph
            print("REPRODUCED", sequence, flush=True)
    out = dict(
        status="passed",
        conditions=len(records),
        maximum_score_error=max(r["max_error"] for r in records),
        records=records,
        test_used=False,
        source_sha256=e.e.run.HASH(Path(__file__)),
    )
    e.e.run.WRITE(e.CAMPAIGN / "cache_validation.json", out)
    print("PASSED", out["maximum_score_error"], flush=True)


if __name__ == "__main__":
    e.e.configure()
    main()
