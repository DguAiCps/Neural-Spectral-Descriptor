#!/usr/bin/env python3
"""기존 주결과와 같은 시퀀스 단위 조건부 구간을 방향별로 확장한다."""
from pathlib import Path
import time
import numpy as np
import _review_direction_metrics as base


def main() -> None:
    """기존 전체 구간을 재현하고 방향별·도시별 짝지은 차이를 보존한다."""
    out=base.OUT/'direction_learning'
    source=base.read(base.OUT/'direction_summary.json')
    original=base.read(base.MAIN/'integrated_report.json')
    protocol=dict(created_unix=time.time(),replicates=10000,seed=20261104,
        scope='기존 integrated_report와 동일한 도시 내 시퀀스 복원추출. 질의·시작점은 고정. 블록 재표집 구간은 summary.json에 별도 보존하며 둘 중 유리한 결과를 선택하지 않는다.',
        sources={str(p.relative_to(base.ROOT)):base.digest(p) for p in [Path(__file__),base.OUT/'direction_summary.json',base.MAIN/'analysis_plan.json']})
    base.write(out/'sequence_protocol.json',protocol)
    result={}
    for a,b in [('convs_only','fixed'),('diff','fixed'),('diff','identity')]:
        result[a+'-'+b]={}
        for group in ['all','forward_only','reverse_only','mixed']:
            rng=np.random.default_rng(20261104)
            cities,draws={},[]
            for city,seqs in base.CITY.items():
                def value(method,seq,metric):
                    seeds=source['methods'][method]['per_seed'].values()
                    rows=[s[group]['sequences'][seq] for s in seeds]
                    return None if rows[0] is None else np.mean([r[metric] for r in rows])
                available=[s for s in seqs if value(a,s,'AP') is not None]
                if not available:continue
                matrix=np.array([[100*(value(a,s,m)-value(b,s,m)) for m in base.KEYS] for s in available])
                index=rng.integers(0,len(available),size=(10000,len(available)))
                boot=matrix[index].mean(1);draws.append(boot)
                cities[city]={m:dict(delta_pp=float(matrix[:,k].mean()),ci95_pp=np.percentile(boot[:,k],[2.5,97.5]).tolist()) for k,m in enumerate(base.KEYS)}
            boot=np.mean(draws,axis=0)
            overall={m:dict(delta_pp=100*(source['methods'][a]['mean_std'][group][m]['mean']-source['methods'][b]['mean_std'][group][m]['mean']),ci95_pp=np.percentile(boot[:,k],[2.5,97.5]).tolist()) for k,m in enumerate(base.KEYS)}
            if group=='all' and a+'-'+b in original['comparisons']:
                for m in base.KEYS:np.testing.assert_allclose(overall[m]['ci95_pp'],original['comparisons'][a+'-'+b]['conditional_95_pp'][m],atol=1e-10,rtol=0)
            result[a+'-'+b][group]=dict(overall=overall,cities=cities)
    base.write(out/'sequence_summary.json',dict(status='passed',comparisons=result,original_intervals_reproduced=True,protocol_sha256=base.digest(out/'sequence_protocol.json')))
    print('SEQUENCE CI PASSED')


if __name__=='__main__':main()
