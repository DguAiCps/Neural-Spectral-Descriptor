#!/usr/bin/env python3
"""현재 주모델 검증 원장에서 학습 키가 최종 판정에 미치는 경로를 진단한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from _pr_eval_lib import classify_queries, pr_sweep

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/pdv2_review_response_20261104/nsd/val'
OUT = ROOT / 'results/pdv2_learning_plan_20261104'
METRICS = ('AP', 'max_F1', 'R@100P')


def digest(path: Path) -> str:
    """입력 원장의 해시를 기록한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_row(path: Path) -> dict:
    """고정된 최종 원장을 읽는다."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def ap(row: dict, score: np.ndarray) -> float:
    """원 규약의 AP를 계산한다."""
    return float(pr_sweep(score, row['top1_geo_dist'], classify_queries(row['min_past_gt_dist']))['AP'])


def error_budget(row: dict) -> dict:
    """정답 라벨을 쓰는 진단 상한이며 배포 개선량은 아니다."""
    positive = row['min_past_gt_dist'] <= 5
    if not positive.any():
        return {}
    negative = row['min_past_gt_dist'] > 25
    wrong = positive & (row['top1_geo_dist'] > 5)
    values = {}
    score = row['score']
    floor = float(score.min()-max(float(np.ptp(score)), 1.))
    for name, mask in (('actual', np.zeros(len(score), bool)), ('nonrevisit', negative),
                       ('wrong_place', wrong), ('both', negative | wrong)):
        changed = score.copy(); changed[mask] = floor
        values[name] = ap(row, changed)
    recall = float(np.mean(row['top1_geo_dist'][positive] <= 5))
    np.testing.assert_allclose(values['both'], recall, rtol=0, atol=1e-12)
    result = dict(retrieval_miss=1-recall,
                  nonrevisit_false_accept=.5*(values['nonrevisit']-values['actual']+values['both']-values['wrong_place']),
                  wrong_place_false_accept=.5*(values['wrong_place']-values['actual']+values['both']-values['nonrevisit']))
    np.testing.assert_allclose(sum(result.values()),1-values['actual'],rtol=0,atol=1e-12)
    return dict(components=result, oracle_AP=values)


def main() -> None:
    """시험 파일을 읽지 않고 3시드 검증 승자·점수·오류 상한을 집계한다."""
    rows, sources, seed_summaries = {}, {}, {}
    for seed in (1, 2, 3):
        per = {}
        for p in sorted((SOURCE/f's{seed}').glob('*_main.npz')):
            other = p.with_name(p.name.replace('_main.npz', '_no_context__aux_half.npz'))
            record = json.loads(p.with_name(p.name.replace('_main.npz', '.json')).read_text())
            for arm,path in (('main',p),('no_context__aux_half',other)):
                actual=digest(path);assert record['records'][arm]['sha256']==actual
                sources[str(path.relative_to(ROOT))]=actual
            a,b=read_row(p),read_row(other)
            for k in ('query_idx','min_past_gt_dist'):
                np.testing.assert_array_equal(a[k],b[k])
            same=a['j_star']==b['j_star'];positive=a['min_past_gt_dist']<=5;negative=a['min_past_gt_dist']>25
            good_a=positive & (a['top1_geo_dist']<=5);good_b=positive & (b['top1_geo_dist']<=5)
            delta=np.abs(a['score']-b['score'])
            np.testing.assert_allclose(a['score'][same],b['score'][same],rtol=0,atol=1e-12)
            sequence=p.name[:-len('_main.npz')]
            per[sequence]=dict(queries=len(same),unchanged_winner=int(same.sum()),
                unchanged_score=int((delta<=1e-12).sum()),unchanged_winner_score_max_error=float(delta[same].max(initial=0)),
                positive=int(positive.sum()),negative=int(negative.sum()),
                unchanged_positive_winner=int((same & positive).sum()),unchanged_negative_winner=int((same & negative).sum()),
                neural_gained_correct=int((good_a & ~good_b).sum()),neural_lost_correct=int((good_b & ~good_a).sum()),
                main_error_budget=error_budget(a),no_context_error_budget=error_budget(b))
        assert len(per)==9
        counts={k:sum(v[k] for v in per.values()) for k in ('queries','unchanged_winner','unchanged_score','positive','negative','unchanged_positive_winner','unchanged_negative_winner','neural_gained_correct','neural_lost_correct')}
        families={}
        for f in ('kitti','helipr','mulran'):
            values=[v['main_error_budget']['components'] for n,v in per.items() if n.startswith(f) and v['positive']]
            families[f]={k:float(np.mean([v[k] for v in values])) for k in values[0]}
        budget={k:float(np.mean([v[k] for v in families.values()])) for k in families['kitti']}
        seed_summaries[str(seed)]=dict(counts=counts,main_error_budget=budget,by_family=families)
        rows[str(seed)]=per
    result=dict(scope='검증 9시퀀스·3시드만. 원장 고정, 모델·설정 선택 및 시험 사용 없음.',
                error_budget_scope='정답을 알고 오류 점수만 최솟값 아래로 내리는 사후 상한. 온라인 실현 가능한 개선량 아님.',
                per_seed=seed_summaries,per_sequence=rows,input_sha256=sources,
                mean_error_budget={k:float(np.mean([v['main_error_budget'][k] for v in seed_summaries.values()])) for k in budget})
    (OUT/'validation_learning_path.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(per_seed=seed_summaries,mean_error_budget=result['mean_error_budget']),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
