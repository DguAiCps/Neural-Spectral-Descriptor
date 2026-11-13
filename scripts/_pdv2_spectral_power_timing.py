#!/usr/bin/env python3
"""선택된 크기 압축 판정의 증분 실행 일치와 추가 비용을 측정한다."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import torch

import _pdv2_spectral_power as study
import _pdv2_table3_nsd_v4 as current

base, cal, bench = study.base, study.cal, current.bench


class Runtime(current.Runtime):
    """현재 저장 상태에서 필요한 행만 변환하고 새 판정을 수행한다."""
    def query_power(self, q: int, selected: str) -> tuple:
        """동결된 융합 승자와 최종 크기 압축 점수를 반환한다."""
        from _causal_pr_dump import prefix_topk
        from _xs_eval_lib import minmax
        name, fusion, channel = selected.split("__")
        power = study.POWERS.get(name, 1.)
        w = cal.fusion_weights(self.conf)[cal.FUSIONS.index(fusion)]
        self.transient = 0
        ids = np.unique(np.r_[prefix_topk(self.keys, q, q - 29, 100),
                              prefix_topk(self.hkey, q, q - 29, 100)])
        a = current.v3.spectral_scores(self.coeff, q, ids)
        c = current.v3.column_scores(self.columns(np.union1d(ids, [q]), ("original",))["original"], q, ids)
        score = (w[0] * minmax(self.keys[ids] @ self.keys[q])
                 + w[1] * minmax(self.hkey[ids] @ self.hkey[q])
                 + w[2] * minmax(a) + w[3] * minmax(c))
        j = int(ids[np.argmax(score)])
        span = np.r_[np.arange(q - 5, q + 6), np.arange(j - 10, j + 11)]
        rows = np.unique(span[(span >= 0) & (span < self.n)])
        banks = []
        for coeff in (self.coeff[rows], self.white.transform(self.coeff[rows], self.matrix)):
            x = study.transform(coeff, power)
            bank = base.variant_inputs(x, 8)
            self.transient += x.nbytes + sum(a.nbytes for a in (*bank[:3], *bank[3]))
            banks.append((*[current.base.Rows(rows, a) for a in bank[:3]],
                          tuple(current.base.Rows(rows, a) for a in bank[3])))
        endpoints = {ch: base.scores_for(*banks, self.travel[:self.n], np.array([q]), np.array([j]), ch)
                     for ch in ("c_spec", "c_col")}
        value = cal.confidence(endpoints["c_spec"], endpoints["c_col"], channel, self.conf["decision"])
        return j, float(value[0]), len(ids)


def main() -> None:
    """표3 초기후보1만 부분 작업에서 기존·새 질의를 짝지어 잰다."""
    frozen = study.read(study.OUT / "test_freeze.json")
    selected = frozen["selected"]
    target = study.OUT / "timing_n10000.json"
    assert not target.exists()
    spec = study.read(study.OUT / "protocol.json")
    name, fusion, channel = selected.split("__")
    fi = cal.FUSIONS.index(fusion)
    source = cal.OUT / "val/s1/mulran_Sejong01.npz"
    src = study.arrays(source)
    value_path = source if name == "p100" else study.OUT / "val/s1/mulran_Sejong01.npz"
    vals = study.arrays(value_path)
    prefix = "" if name == "p100" else name + "_"
    expected = cal.confidence(vals[prefix + "c_spec"][:, fi], vals[prefix + "c_col"][:, fi],
                              channel, spec["confidence"]["1"]["decision"])
    new = {"query_idx": src["query_idx"], "j_star": src["candidates"][:, fi], "score": expected}
    old = study.arrays(current.LEDGER)
    protocol = {"created_unix": time.time(), "selected": selected,
        "scope": "MulRan Sejong01 시드1,초기후보1만,96도착x3,첫12워밍업제외. 같은증분상태에서기존/새질의순서를교대로실행. 두구현의승자와점수를검증원장과대조. 전체데이터평균이나표3자동교체아님.",
        "sources": {str(p.relative_to(study.REPO)): study.sha256(p) for p in
                    (Path(__file__), Path(current.__file__), Path(study.__file__), source, value_path, current.LEDGER)},
        "test_freeze_sha256": study.sha256(study.OUT / "test_freeze.json")}
    study.write_json(study.OUT / "timing_protocol.json", protocol)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    work = bench.Workload(bench.prepare(), 10000)
    runtime = Runtime(bench.prepare(), work)
    samples, checked, worst = [], 0, 0.
    for rep in range(bench.REPEATS):
        runtime.reset()
        gc.collect()
        torch.cuda.empty_cache()
        for i, points in enumerate(work.points):
            t = work.initial + i
            encoded, encode_ms = bench.timed(lambda: runtime.encode(points))
            _, state_ms = bench.timed(lambda: runtime.append(*encoded, work.poses[t]))
            q = t - 15
            order = ("original", "power") if rep % 2 == 0 else ("power", "original")
            measure = {}
            for mode in order:
                function = (lambda: runtime.query(q)) if mode == "original" else (lambda: runtime.query_power(q, selected))
                result, elapsed = bench.timed(function)
                ledger = old if mode == "original" else new
                ix = np.searchsorted(ledger["query_idx"], q)
                assert ledger["query_idx"][ix] == q and ledger["j_star"][ix] == result[0]
                err = float(abs(ledger["score"][ix] - result[1]))
                assert err <= 1e-6, (q, mode, err)
                worst, checked = max(worst, err), checked + 1
                measure[mode] = {"query_ms": elapsed, "score_error": err, "transient_bytes": runtime.transient}
            if i >= bench.WARMUP:
                update = encode_ms + state_ms
                samples.append({"repeat": rep, "query": q, "update_ms": update, **measure,
                                "original_total_ms": update + measure["original"]["query_ms"],
                                "power_total_ms": update + measure["power"]["query_ms"]})
    result = {"status": "passed", "selected": selected, "samples": len(samples), "raw_samples": samples,
              "resident": runtime.memory(), "checks": checked, "score_max_error": worst,
              "mean_query_ms": {m: float(np.mean([r[m]["query_ms"] for r in samples])) for m in ("original", "power")},
              "mean_total_ms": {m: float(np.mean([r[m + "_total_ms"] for r in samples])) for m in ("original", "power")},
              "protocol_sha256": study.sha256(study.OUT / "timing_protocol.json")}
    study.write_json(target, result)
    print("TIMING", result["mean_query_ms"], result["mean_total_ms"], result["resident"]["total"], flush=True)


if __name__ == "__main__":
    main()
