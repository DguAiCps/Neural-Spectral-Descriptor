#!/usr/bin/env python3
"""학습 손실의 2x2 대조를 원천 후보 검산과 함께 읽기 쉬운 결과로 정리한다."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results/pdv2_fusion_guard_20261023'


def read(path: Path) -> dict:
    """저장된 결과 JSON을 읽는다."""
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    """파일의 SHA256을 계산한다."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path: Path, data: dict) -> None:
    """결과를 UTF-8 JSON으로 저장한다."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def independent_pool_audit(spec: dict) -> dict:
    """numpy 배정밀도로 실제 후보의 상한·기준 정답·순위 손실을 검산한다."""
    weights = np.asarray(spec['confidence']['1']['tuple'], dtype=np.float64)
    results = {}
    for sequence in spec['train']:
        folder = OUT / 'pools' / sequence.replace('/', '_')
        recorded = read(folder / 'audit.json')
        with np.load(folder / 'pool.npz') as z:
            arrays = {k: z[k] for k in z.files}
        n = len(arrays['queries'])
        if not n:
            results[sequence] = {'queries': 0, 'impossible': 0, 'protected': 0,
                                  'rank_loss_sum': 0., 'impossible_loss_sum': 0.}
            continue
        valid, positive = arrays['valid'], arrays['correct'] & arrays['valid']
        negative = valid & ~positive
        assert positive.any(1).all() and negative.any(1).all()
        assert np.all(arrays['candidates'][valid] <= np.broadcast_to(arrays['queries'][:, None] - 30, valid.shape)[valid])
        channels = np.concatenate((arrays['teacher_key'][..., None], arrays['static']), axis=-1).astype(np.float64)
        lo = np.min(np.where(valid[..., None], channels, np.inf), axis=1, keepdims=True)
        hi = np.max(np.where(valid[..., None], channels, -np.inf), axis=1, keepdims=True)
        pieces = weights * (channels - lo) / np.maximum(hi - lo, 1e-8)
        score, static = pieces.sum(-1), pieces[..., 1:].sum(-1)
        margin = np.max(np.where(positive, score, -np.inf), 1) - np.max(np.where(negative, score, -np.inf), 1)
        upper = weights[0] + np.max(np.where(positive, static, -np.inf), 1) - np.max(np.where(negative, static, -np.inf), 1)
        impossible, protected = upper <= 0, margin > 0
        assert not np.any(impossible & protected)
        logits = score / weights[0] / spec['config']['temperature']
        maximum = np.max(np.where(valid, logits, -np.inf), 1)
        shifted = np.exp(np.where(valid, logits - maximum[:, None], -np.inf))
        losses = np.log(shifted.sum(1)) - np.log(np.where(positive, shifted, 0).sum(1))
        assert int(impossible.sum()) == recorded['active'] - recorded['admissible']
        assert int(protected.sum()) == recorded['protected']
        np.testing.assert_allclose(losses.sum(), recorded['rank_loss_sum'], rtol=2e-5, atol=.01)
        np.testing.assert_allclose(losses[impossible].sum(), recorded['impossible_loss_sum'], rtol=2e-5, atol=.01)
        results[sequence] = {'queries': n, 'impossible': int(impossible.sum()),
            'protected': int(protected.sum()), 'rank_loss_sum': float(losses.sum()),
            'impossible_loss_sum': float(losses[impossible].sum()),
            'key_width_min': float((hi - lo)[:, 0, 0].min())}
    totals = {k: sum(x[k] for x in results.values()) for k in
              ('queries', 'impossible', 'protected', 'rank_loss_sum', 'impossible_loss_sum')}
    totals['impossible_query_percent'] = 100 * totals['impossible'] / totals['queries']
    totals['impossible_loss_percent'] = 100 * totals['impossible_loss_sum'] / totals['rank_loss_sum']
    totals['impossible_among_initial_errors_percent'] = 100 * totals['impossible'] / (totals['queries'] - totals['protected'])
    return {'by_sequence': results, 'total': totals, 'passed': True,
            'interpretation': '손실값 비중이며 기울기나 매개변수 갱신의 기여율이 아니다.'}


def main() -> None:
    """최종 손익과 고정 학습 후보의 변화가 일치하는 범위를 보여준다."""
    audit = read(OUT / 'completion_validation.json')
    assert audit['passed']
    sources = dict(audit['sources'])
    gradient_audit = read(OUT / 'gradient_probe/completion_validation.json')
    assert gradient_audit['passed']
    for name, sha in gradient_audit['sources'].items():
        assert sources.get(name, sha) == sha
        sources[name] = sha
    case_audit = read(OUT / 'case_validation.json')
    assert case_audit['passed']
    for name, sha in case_audit['sources'].items():
        assert sources.get(name, sha) == sha
        sources[name] = sha
    for name, sha in sources.items():
        assert digest(REPO / name) == sha, name
    summary, spec = read(OUT / 'summary.json'), read(OUT / 'protocol.json')
    pool = independent_pool_audit(spec)
    fixed_pool = {}
    for arm, rows in summary['training_fixed_pool'].items():
        fixed_pool[arm] = {k: sum(r[k] for r in rows.values()) for k in
            ('fixed_pool_correct', 'protected', 'protected_lost', 'fixed_pool_gained')}
        assert fixed_pool[arm]['protected'] == pool['total']['protected']
        assert fixed_pool[arm]['fixed_pool_correct'] == (pool['total']['protected']
            - fixed_pool[arm]['protected_lost'] + fixed_pool[arm]['fixed_pool_gained'])
    report = {'final_results': summary['results'], 'selected_epoch': summary['selected_epoch'],
        'initial_pool': pool, 'fixed_pool_after_training': fixed_pool,
        'fresh_pool_transitions': {a: {role: summary['candidate_transitions'][a][role][str(epoch)]
            for role in ('train', 'val')} for a, epoch in summary['selected_epoch'].items()},
        'same_epoch_validation': summary['all_validation_checkpoints'],
        'promotion_checks': summary['promotion_checks'], 'eligible_candidates': summary['eligible_candidates'],
        'same_epoch_factorial_interaction_pp': summary['same_epoch_factorial_interaction_pp'],
        'initial_gradient_probe': read(OUT / 'gradient_probe/summary.json'),
        'combined_case_diagnosis': read(OUT / 'case_diagnosis.json')['aggregates'],
        'test_used': False, 'new_training_updates': summary['total_new_updates'],
        'scope': '학습 GT8·검증 KISS9, 시드1. 네 조건은 같은252회 업데이트. '
                 '후보·점수식·시간 판정·추론 매개변수 추가 없음. 초기 실제 후보에서 학습하고 '
                 '최종 평가에서는 후보를 다시 검색한다. 시작 모델과 같은 조건의 순위 학습이 대조군이다.'}
    write(OUT / 'integrated_report.json', report)
    plot(report)
    for path in (Path(__file__), OUT / 'integrated_report.json', OUT / 'completion_validation.json',
                 OUT / 'fusion_guard_summary.png', OUT / 'fusion_guard_summary.pdf',
                 OUT / 'gradient_probe/completion_validation.json', OUT / 'case_validation.json'):
        sources[str(path.relative_to(REPO))] = digest(path)
    write(OUT / 'integrated_validation.json', {'passed': True, 'sources': sources,
        'source_count': len(sources), 'pool_queries_checked': pool['total']['queries'],
        'ledgers': audit['ledgers'], 'rows': audit['rows'], 'test_used': False})
    print('FUSION_GUARD_REPORT_PASSED', len(sources), pool['total'], flush=True)


def plot(report: dict) -> None:
    """학습·검증 최종 AP와 정답 손익을 독립 축으로 표시한다."""
    names = ['rank_all', 'rank_filtered', 'rank_guarded', 'rank_filtered_guarded']
    labels = ['Rank all', 'Filter impossible', 'Protect correct', 'Filter + protect']
    colors = ['#70849f', '#c58e54', '#72a48b', '#876f9c']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, role, title in zip(axes[:2], ('train', 'val'), ('Training GT (8)', 'Validation KISS (9)')):
        base_ap = report['final_results'][role]['start']['macro']['AP']
        ap = [100 * report['final_results'][role][a]['macro']['AP'] for a in names]
        delta = np.array(ap) - 100 * base_ap
        bars = axis.barh(np.arange(4), delta, color=colors)
        axis.bar_label(bars, labels=[f'{x:+.3f}' for x in delta], padding=3, fontsize=9)
        axis.axvline(0, color='#999', linewidth=.8)
        axis.set_yticks(np.arange(4), labels)
        axis.invert_yaxis()
        span = max(.03, float(np.max(np.abs(delta))))
        axis.set_xlim(min(0, float(delta.min())) - span * .6, max(0, float(delta.max())) + span * .4)
        axis.set_xlabel('Final AP change (percentage points)')
        axis.set_title(f'{title}\nStarting final AP: {base_ap * 100:.3f}%', fontsize=11)
    gains = [report['fresh_pool_transitions'][a]['val']['gained'] for a in names]
    losses = [-report['fresh_pool_transitions'][a]['val']['lost'] for a in names]
    x = np.arange(4)
    b1 = axes[2].bar(x, gains, color='#72a48b', label='Gained correct queries')
    b2 = axes[2].bar(x, losses, color='#bf7665', label='Lost correct queries')
    axes[2].bar_label(b1, padding=2, fontsize=9)
    axes[2].bar_label(b2, padding=2, fontsize=9)
    axes[2].axhline(0, color='#999', linewidth=.8)
    axes[2].set_xticks(x, ['Rank', 'Filter', 'Protect', 'Both'])
    axes[2].set_ylabel('Validation query count')
    axes[2].set_title('Fresh candidate search\nOne seed; epoch 2/4 selected by final AP', fontsize=11)
    axes[2].legend(loc='best', fontsize=8)
    axes[2].margins(y=.25)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / ('fusion_guard_summary.' + ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
