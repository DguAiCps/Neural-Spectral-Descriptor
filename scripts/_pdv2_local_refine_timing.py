#!/usr/bin/env python3
"""검증에서 고정한 국소 보정의 추가 질의 비용을 같은 상태에서 잰다."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import torch

import _pdv2_local_refine as local
import _pdv2_minimal_rerank as experiment
import _pdv2_table3_nsd_v4 as current
from _causal_pr_dump import prefix_topk
from _xs_eval_lib import minmax

bench = current.bench


class Runtime(current.Runtime):
    """기존 CPU 증분 인코딩과 필요한 행만 파생하는 저장 구조를 유지한다."""

    def query_refined(self, q: int, radius: int, cap: bool) -> tuple:
        """현재 검색 후 이웃 중심들을 같은 시간 점수로 비교한다."""
        self.transient = 0
        ids = np.unique(np.r_[prefix_topk(self.keys, q, q - 29, 100),
                              prefix_topk(self.hkey, q, q - 29, 100)])
        a = current.v3.spectral_scores(self.coeff, q, ids)
        c = current.v3.column_scores(self.columns(np.union1d(ids, [q]), ("original",))["original"], q, ids)
        w = self.conf["tuple"]
        fused = (w[0] * minmax(self.keys[ids] @ self.keys[q])
                 + w[1] * minmax(self.hkey[ids] @ self.hkey[q])
                 + w[2] * minmax(a) + w[3] * minmax(c))
        j = int(ids[np.argsort(-fused, kind="stable")[0]])
        centers = local.indices(np.array([q]), np.array([j]), radius)[0]
        span = np.r_[np.arange(q - 5, q + 6),
                     np.concatenate([np.arange(k - 10, k + 11) for k in centers])]
        rows = np.unique(span[(span >= 0) & (span < self.n)])
        columns = self.columns(rows, ("original", "whitened"))
        values = experiment.scores_for((None, None, None, columns["original"]),
            (None, None, None, columns["whitened"]), self.travel[:self.n],
            np.full(len(centers), q), centers, "c_col")
        best = int(values.argmax())
        score = min(values[best], values[0]) if cap else values[best]
        return int(centers[best]), float(score), len(ids)


def main() -> None:
    """초기 후보1만·96도착×3회에서 기본·보정 질의를 짝지어 잰다."""
    frozen = local.read(local.OUT / "test_freeze.json")
    selected = frozen["selected"]
    radius = int(selected.split("_")[0].replace("radius", ""))
    cap = selected.endswith("_cap")
    target = local.OUT / "timing_n10000.json"
    assert not target.exists()
    protocol = {"created_unix": time.time(), "selected": selected, "n_map": 10000,
        "workload": "MulRanSejong01,시드1,기존표3의원시96도착×3회,CPU1,TF32off. 같은증분상태에서기본/추가질의순서를반복마다바꾼짝비교. 추가스캔/인코딩/상주특징없음.",
        "scope": "부분작업추가비용실측,새주표3게시없음. 새후보/점수는검증전수원장과대조한다.",
        "source_sha256": local.sha256(Path(__file__)),
        "test_freeze_sha256": local.sha256(local.OUT / "test_freeze.json")}
    local.write_json(local.OUT / "timing_protocol.json", protocol)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = bench.prepare()
    work = bench.Workload(spec, 10000)
    runtime = Runtime(spec, work)
    old = local.arrays(current.LEDGER)
    new = local.arrays(local.OUT / "val/s1" / f"mulran_Sejong01_{selected}.npz")
    samples = []
    for rep in range(bench.REPEATS):
        runtime.reset()
        gc.collect()
        torch.cuda.empty_cache()
        for i, points in enumerate(work.points):
            t = work.initial + i
            encoded, encode_ms = bench.timed(lambda: runtime.encode(points))
            _, state_ms = bench.timed(lambda: runtime.append(*encoded, work.poses[t]))
            q = t - 15
            order = ("original", "refined") if rep % 2 == 0 else ("refined", "original")
            measures = {}
            for mode in order:
                function = (lambda: runtime.query(q)) if mode == "original" else (lambda: runtime.query_refined(q, radius, cap))
                value, elapsed = bench.timed(function)
                ledger = old if mode == "original" else new
                index = np.searchsorted(ledger["query_idx"], q)
                assert ledger["query_idx"][index] == q
                assert value[0] == ledger["j_star"][index], (q, mode, value[0], ledger["j_star"][index])
                error = float(abs(value[1] - ledger["score"][index]))
                assert error <= 1e-6, (q, mode, error)
                measures[mode] = {"query_ms": elapsed, "score_max_error": error,
                                  "winner": value[0], "transient_bytes": runtime.transient}
            if i >= bench.WARMUP:
                update = encode_ms + state_ms
                samples.append({"repeat": rep, "arrival": t, "query": q,
                    "update_ms": update, **measures,
                    "original_total_ms": update + measures["original"]["query_ms"],
                    "refined_total_ms": update + measures["refined"]["query_ms"]})
    result = {"status": "passed", "selected": selected, "samples": len(samples),
        "raw_samples": samples, "resident": runtime.memory(),
        "mean_query_ms": {m: float(np.mean([s[m]["query_ms"] for s in samples])) for m in ("original", "refined")},
        "mean_total_ms": {m: float(np.mean([s[m + "_total_ms"] for s in samples])) for m in ("original", "refined")},
        "protocol_sha256": local.sha256(local.OUT / "timing_protocol.json"),
        "scope": "첫12개워밍업제외84도착×3회252쌍. 부분작업짝비교이며표3기존값자동교체없음."}
    local.write_json(target, result)
    print("TIMING", result["mean_query_ms"], result["mean_total_ms"], flush=True)


if __name__ == "__main__":
    main()
