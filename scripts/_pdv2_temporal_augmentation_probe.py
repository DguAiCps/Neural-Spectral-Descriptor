#!/usr/bin/env python3
"""시간 간선 변형에 대한 문맥 변화와 학습 후보 구분력의 안정성을 진단한다."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_temporal_augmentation as run
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash


def diversity(context: np.ndarray) -> dict:
    """정규화 문맥의 시퀀스 내 분산과 공분산 유효 차원을 기록한다."""
    work = context.astype(np.float64)
    work -= work.mean(0)
    covariance = work.T @ work / len(work)
    spectrum = np.linalg.eigvalsh(covariance).clip(0)
    trace = float(spectrum.sum())
    if trace > 1e-15:
        probability = spectrum[spectrum > 0] / trace
        rank = float(np.exp(-(probability*np.log(probability)).sum()))
    else:
        rank = 0.
    return {'variance_trace': trace, 'effective_rank': rank}


def statistics(values: np.ndarray) -> dict:
    """건수·평균·중앙값·95백분위를 빈 집단과 구분한다."""
    return {'n': len(values), 'mean': float(values.mean()) if len(values) else None,
        'median': float(np.median(values)) if len(values) else None,
        'p95': float(np.quantile(values, .95)) if len(values) else None}


def main(stage: str) -> None:
    """학습 전과 선택 후에 같은 두 입력 변형을 적용한다. 가중치는 갱신하지 않는다."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = run.protocol(True)
    chosen = {'initial': 0} if stage == 'initial' else run.base.read(run.OUT / 'selection.json')['selected_epoch']
    target = run.OUT / f'probe_{stage}.json'
    assert not target.exists()
    sources = {**spec['runtime'], **spec['inputs']}
    direction_root = REPO / 'results/pdv2_context_transfer_20261024'
    for role in ('train', 'val'):
        for sequence in spec[role]:
            path = direction_root / role / (sequence.replace('/', '_') + '.npz')
            sources[str(path.relative_to(REPO))] = sha256(path)
    for arm, epoch in chosen.items():
        path = run.prior.INITIAL if arm == 'initial' else run.OUT / arm / f'epoch{epoch}.pth'
        sources[str(path.relative_to(REPO))] = sha256(path)
    write_json(run.OUT / f'probe_{stage}_protocol.json', {'sources': sources, 'selected_epoch': chosen,
        'scope': '진단용 같은시드431·drop/stride. 표현 변화와 고정 학습 후보의 순위 안정성이다. '
                 '최종 성능은 별도 후보 재검색 원장을 따른다. 일치 계수는 이 진단을 보기 전에 학습 기울기로 고정했다.',
        'test_used': False, 'new_updates': 0})
    results, aggregates = {}, {}
    net = run.initial_model()
    for arm, epoch in chosen.items():
        if arm != 'initial':
            state = net.state_dict()
            state.update(torch.load(run.OUT / arm / f'epoch{epoch}.pth', map_location='cpu', weights_only=False)['model_state_dict'])
            net.load_state_dict(state)
        results[arm], aggregates[arm] = {}, {}
        for role in ('train', 'val'):
            results[arm][role] = {}
            pooled = {view: {cohort: [] for cohort in ('all', 'same', 'opposite', 'other')} for view in run.CFG['views']}
            for sequence in spec[role]:
                cache = run.base.exp.arrays(run.factor.paths_for(role, sequence)['cache'])
                key, _ = run.view_keys(net, cache)
                source_key = (run.factor.key_path(role, 'interaction', sequence) if arm == 'initial' else
                    run.OUT / arm / f'epoch{epoch}' / role / ('keys_' + sequence.replace('/', '_') + '.npz'))
                reference = run.base.exp.arrays(source_key)['embedding']
                np.testing.assert_allclose(key, reference, atol=4e-6, rtol=4e-5)
                sources[str(source_key.relative_to(REPO))] = sha256(source_key)
                direction = run.base.exp.arrays(direction_root / role / (sequence.replace('/', '_') + '.npz'))
                queries = np.arange(30, len(key)-15)
                context = run.base.normalize(key[:, 288:])
                record = {'nodes': len(key), 'clean_key_hash': array_hash(key),
                    'clean_diversity': diversity(context[queries]), 'views': {}}
                pool = run.load_pool(sequence, 'cpu') if role == 'train' else None
                clean_tensor = torch.from_numpy(key)
                for view in run.CFG['views']:
                    changed, _ = run.view_keys(net, cache, view)
                    other = run.base.normalize(changed[:, 288:])
                    drift = np.maximum(1 - np.sum(context.astype(np.float64)*other.astype(np.float64), axis=1), 0.)
                    np.testing.assert_allclose(run.base.normalize(key[:, :288]),
                        run.base.normalize(changed[:, :288]), atol=4e-6, rtol=4e-5)
                    cohorts = {'all': queries, **{label: direction['query'][direction['group'] == g]
                               for g, label in enumerate(('same', 'opposite', 'other'))}}
                    item = {'key_hash': array_hash(changed), 'drift': {label: statistics(drift[q]) for label, q in cohorts.items()},
                        'diversity': diversity(other[queries])}
                    for label, q in cohorts.items():
                        pooled[view][label].append(drift[q])
                    if role == 'train' and len(pool['queries']):
                        altered_tensor = torch.from_numpy(changed)
                        before, after, agreements = [], [], []
                        for start in range(0, len(pool['queries']), 512):
                            rows = torch.arange(start, min(start+512, len(pool['queries'])))
                            weights = tuple(spec['confidence']['1']['tuple'])
                            before.append(run.terms_for(clean_tensor, pool, rows, weights)['margin'].numpy())
                            after.append(run.terms_for(altered_tensor, pool, rows, weights)['margin'].numpy())
                            agreements.append((len(rows), float(run.consistency(clean_tensor, altered_tensor, pool, rows))))
                        before, after = np.concatenate(before), np.concatenate(after)
                        good0, good1 = before > 0, after > 0
                        item['fixed_training_pool'] = {'n': len(before), 'clean_correct': int(good0.sum()),
                            'changed_correct': int(good1.sum()), 'gained': int((~good0 & good1).sum()),
                            'lost': int((good0 & ~good1).sum()),
                            'relative_score_mse_scaled': float(sum(n*v for n, v in agreements)/len(before))}
                    record['views'][view] = item
                results[arm][role][sequence] = record
                print('VIEW_PROBE_DONE', stage, arm, role, sequence, flush=True)
            aggregates[arm][role] = {view: {label: statistics(np.concatenate(values)) for label, values in groups.items()}
                                     for view, groups in pooled.items()}
    write_json(target, {'results': results, 'aggregates': aggregates, 'test_used': False, 'new_updates': 0,
        'cohort_weighting': '질의 수로 가중한 기술 통계다. 최종 성능의 계열 매크로와 다르다.'})
    for p in (target, run.OUT / f'probe_{stage}_protocol.json'):
        sources[str(p.relative_to(REPO))] = sha256(p)
    for path, value in sources.items():
        assert sha256(REPO / path) == value, path
    write_json(run.OUT / f'probe_{stage}_validation.json', {'passed': True, 'sources': sources,
        'model_count': len(chosen), 'sequence_count': len(spec['train'])+len(spec['val']), 'test_used': False})
    print('VIEW_PROBE_COMPLETE', stage, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('initial', 'selected'))
    main(parser.parse_args().stage)
