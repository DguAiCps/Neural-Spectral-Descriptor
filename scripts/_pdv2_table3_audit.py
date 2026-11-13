#!/usr/bin/env python3
"""현재 세 시드의 증분 키·판정과 저장 원장을 추가 대조한다."""
import sys
from pathlib import Path
import numpy as np
import torch
import _pdv2_table3_benchmark as bench


def main() -> None:
    """비용 대표 시드 밖 두 채널까지 같은 증분 구현을 확인한다."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = bench.read(bench.OUT / 'protocol.json')
    work = bench.Workload(spec, 1000)
    rows = {}
    for seed in (1, 2, 3):
        runtime = bench.NSD(spec, work, seed)
        checks = bench.verify_nsd(runtime, work)
        original = bench.REPO / f'results/pdv2_spectral_whitening_20261026/blend/quarter/s{seed}/mulran_Sejong01.npz'
        with np.load(original) as ledger:
            measured = []
            for i in range(bench.WARMUP):
                t = work.initial+i
                runtime.append(*runtime.encode(work.points[i]),work.poses[t])
                j, score, _ = runtime.query(t-15)
                ix = int(np.searchsorted(ledger['query_idx'],t-15))
                assert int(ledger['j_star'][ix]) == j
                assert float(ledger['score'][ix]) == score
                measured.append({'query':t-15,'winner':j,'score_error':0.0})
        rows[str(seed)] = {'channel':runtime.conf['decision'], 'fullprefix':checks,
                          'frozen_ledger': measured, 'ledger_sha256':bench.sha(original)}
        del runtime
        torch.cuda.empty_cache()
    bench.write(bench.OUT/'three_seed_audit.json',{'status':'passed','seeds':rows,
        'sources':{'scripts/_pdv2_table3_benchmark.py':bench.sha(Path(bench.__file__)),
                   'scripts/_pdv2_table3_audit.py':bench.sha(Path(__file__))}})

if __name__ == '__main__':
    main()
