#!/usr/bin/env python3
"""문맥 전이 진단과 정규화 폭 기울기 대조의 최종 지표를 함께 보존한다."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_fusion_width as run
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """최종 성능·방향별 분모·원시 기울기 벡터를 검산해 그림과 표를 만든다."""
    spec = run.protocol(True)
    audit = run.base.read(run.OUT / 'completion_validation.json')
    summary = run.base.read(run.OUT / 'summary.json')
    transfer_root = REPO / 'results/pdv2_context_transfer_20261024'
    transfer = run.base.read(transfer_root / 'report.json')
    transfer_audit = run.base.read(transfer_root / 'report_validation.json')
    gradient = run.base.read(run.OUT / 'gradient_probe/summary.json')
    gradient_audit = run.base.read(run.OUT / 'gradient_probe/completion_validation.json')
    assert audit['passed'] and transfer_audit['passed'] and gradient_audit['passed']
    sources = {**audit['sources'], **transfer_audit['sources'], **gradient_audit['sources']}
    sources[str(Path(__file__).relative_to(REPO))] = sha256(Path(__file__))
    gradient_checks = 0
    with np.load(run.OUT / 'gradient_probe/vectors.npz') as arrays:
        for arm in arrays.files:
            exact, direct = arrays[arm]
            width = exact - direct
            ne, nd, nw = [np.linalg.norm(x) for x in (exact, direct, width)]
            expected = {'exact_norm': ne, 'direct_norm': nd, 'width_norm': nw,
                'width_to_direct_norm': nw / nd, 'width_direct_cosine': width @ direct / nw / nd,
                'exact_direct_cosine': exact @ direct / ne / nd}
            for k, v in expected.items():
                assert abs(v - gradient['initial_batch_mean'][arm][k]) < 1e-12
                gradient_checks += 1
    arms = ['start', 'rank_all', 'rank_all_stop_width', 'rank_filtered_guarded',
            'rank_filtered_guarded_stop_width']
    selected = {**run.base.read(run.prior.OUT / 'selection.json')['selected_epoch'],
                **summary['selected_epoch'], 'start': 0}
    table = []
    for role in ('train', 'val'):
        for arm in arms:
            metrics = summary['results'][role][arm]['macro']
            table.append({'role': role, 'arm': arm, 'epoch': selected[arm],
                **{k: 100*metrics[k] for k in run.base.exp.METRICS}})
    with (run.OUT / 'final_metrics.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    result = {'final_metrics_percent': table, 'selected_epoch': selected,
        'eligible_candidates': summary['eligible_candidates'],
        'same_epoch_stop_width_minus_exact_pp': summary['same_epoch_stop_width_minus_exact_pp'],
        'candidate_transitions': summary['candidate_transitions'], 'gradient_probe': gradient['initial_batch_mean'],
        'direction': {role: transfer['roles'][role]['direction'] for role in ('train', 'val')},
        'context_transfer': {role: transfer['roles'][role]['all'] for role in ('train', 'val')},
        'reversal_max_key_error': transfer['reversal_max_key_error'], 'test_used': False,
        'new_training_updates': audit['new_training_updates'],
        'scope': '시드1 검증 예비. 표2의 시험 성능과 직접 혼합하지 않는다. 방향 구성·이웃 유사도는 '
                 '관찰 연관, 분모 대리 기울기만 같은예산 학습 개입이다. 역재생은 인코더 정합성 검증이다.'}
    write_json(run.OUT / 'integrated_report.json', result)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    colors = ['#336699', '#c35a32', '#999999']
    for x, role in enumerate(('train', 'val')):
        total = transfer['roles'][role]['all']['positive']
        bottom = 0.
        for g, label in enumerate(('Same available', 'Opposite only', 'Other')):
            rate = 100*transfer['roles'][role]['direction'][str(g)]['positive']/total
            axes[0, 0].bar(x, rate, bottom=bottom, color=colors[g], label=label if x == 0 else None)
            if g == 1:
                axes[0, 0].annotate(f'{rate:.2f}%', (x, bottom+rate/2), ha='center', va='center',
                    xytext=(35, 0), textcoords='offset points', fontsize=9)
            bottom += rate
    axes[0, 0].set(xticks=[0, 1], xticklabels=['Train positives', 'Validation positives'],
        ylabel='Share of revisit queries (%)', title='A. Direction composition', ylim=(0, 105))
    axes[0, 0].legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.08), ncol=3)
    direction = transfer['roles']['val']['direction']
    rates = [100*direction[str(g)]['loss_rate_given_correct'] for g in range(3)]
    axes[0, 1].bar(np.arange(3), rates, color=colors)
    for g, rate in enumerate(rates):
        d = direction[str(g)]
        axes[0, 1].text(g, rate+.025, f"{d['lost']}/{d['old_correct']}", ha='center', fontsize=9)
    axes[0, 1].set(xticks=np.arange(3), xticklabels=['Same available', 'Opposite only', 'Other'],
        ylabel='Lost / initially correct (%)', title='B. Prior guarded-model losses', ylim=(0, 1.15))
    names = ['Start', 'Rank', 'Rank, stop width', 'Filter + guard', 'Filter + guard, stop width']
    vals = [100*summary['results']['val'][a]['macro']['AP'] for a in arms]
    axes[1, 0].scatter(vals, np.arange(5), c=['#777777', '#336699', '#4d9664', '#336699', '#4d9664'], s=55)
    for i, v in enumerate(vals):
        axes[1, 0].annotate(f'{v:.4f}', (v, i), xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 0].set(yticks=np.arange(5), yticklabels=names, xlabel='Final validation AP (%)',
        title='C. Candidate search + final temporal score')
    axes[1, 0].invert_yaxis()
    axes[1, 0].margins(x=.4)
    for a, label, color in zip(arms[1:], names[1:], ['#336699', '#4d9664', '#aa6633', '#993366']):
        delta = [100*(summary['results'][r][a]['macro']['AP'] - summary['results'][r]['start']['macro']['AP'])
                 for r in ('train', 'val')]
        axes[1, 1].plot([0, 1], delta, 'o-', label=label, color=color)
    axes[1, 1].axhline(0., color='#999999', linewidth=.8)
    axes[1, 1].set(xticks=[0, 1], xticklabels=['Train', 'Validation'], ylabel='Final AP change (pp)',
        title='D. Transfer of improvement')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    for ext in ('png', 'pdf'):
        fig.savefig(run.OUT / ('context_transfer_summary.' + ext), dpi=180)
    plt.close(fig)
    for path in (run.OUT / 'integrated_report.json', run.OUT / 'final_metrics.csv',
                 run.OUT / 'context_transfer_summary.png', run.OUT / 'context_transfer_summary.pdf'):
        sources[str(path.relative_to(REPO))] = sha256(path)
    for path, digest in sources.items():
        assert sha256(REPO / path) == digest, path
    write_json(run.OUT / 'integrated_validation.json', {'passed': True, 'sources': sources,
        'final_metric_rows': len(table), 'gradient_scalar_checks': gradient_checks,
        'ledgers': audit['ledgers'], 'ledger_rows': audit['rows'], 'metric_sets': audit['metric_sets'],
        'prefix_checks': audit['prefix_checks'], 'scalar_checks': audit['scalar_checks'],
        'maximum_metric_error': audit['maximum_metric_error'], 'reversal_runs': transfer_audit['reversal_runs'],
        'new_training_updates': audit['new_training_updates'], 'test_used': False})
    print('WIDTH_REPORT_COMPLETE', summary['eligible_candidates'], len(sources), flush=True)


if __name__ == '__main__':
    main()
