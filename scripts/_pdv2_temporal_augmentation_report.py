#!/usr/bin/env python3
"""시간 입력 변형의 최종 성능과 문맥 안정성을 함께 검산하고 보고한다."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_temporal_augmentation as run
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """초기 모델의 변형 추론 원장과 새 학습의 최종 원장을 통합 감사한다."""
    spec = run.protocol(True)
    summary = run.base.read(run.OUT / 'summary.json')
    audit = run.base.read(run.OUT / 'completion_validation.json')
    initial = run.base.read(run.OUT / 'probe_initial.json')
    selected = run.base.read(run.OUT / 'probe_selected.json')
    calibration = run.base.read(run.OUT / 'calibration.json')
    sources = dict(audit['sources'])
    support = run.base.read(run.OUT / 'training_support/summary.json')
    support_audit = run.base.read(run.OUT / 'training_support/completion_validation.json')
    assert support_audit['passed']
    sources.update(support_audit['sources'])
    for stage in ('initial', 'selected'):
        data = run.base.read(run.OUT / f'probe_{stage}_validation.json')
        assert data['passed']
        sources.update(data['sources'])
    sources[str(Path(__file__).relative_to(REPO))] = sha256(Path(__file__))
    stress, counts = {}, {k: audit[k] for k in ('ledgers', 'rows', 'metric_sets', 'prefix_checks', 'scalar_checks')}
    max_metric_error = audit['maximum_metric_error']
    max_score_error = audit['maximum_score_error']
    for view in run.CFG['views']:
        per_sequence, transition = {}, np.zeros((3, 3), np.int64)
        for sequence in spec['val']:
            path = run.OUT / 'stress/initial/epoch0' / view / 'val' / (sequence.replace('/', '_') + '.npz')
            info = run.base.read(path.with_suffix('.json'))
            key_path = path.with_name('keys_' + path.name)
            assert sha256(path) == info['sha256'] and sha256(key_path) == info['keys_sha256']
            row = run.base.exp.arrays(path)
            key = run.base.exp.arrays(key_path)['embedding']
            reference_path = run.factor.OUT / 'val' / sequence.replace('/', '_') / 'new.npz'
            reference = run.base.exp.arrays(reference_path)
            q, j = row['query_idx'], row['j_star']
            np.testing.assert_array_equal(q, reference['query_idx'])
            assert np.all(j <= q-30) and key.shape[1] == 416 and np.isfinite(key).all()
            paths = run.factor.paths_for('val', sequence)
            cache = run.base.exp.arrays(paths['cache'])
            position = cache['poses_gt'][:, :3, 3].astype(np.float64)
            np.testing.assert_allclose(row['top1_geo_dist'], np.linalg.norm(position[q]-position[j], axis=1), atol=1e-10)
            np.testing.assert_allclose(row['min_past_gt_dist'], reference['min_past_gt_dist'], atol=1e-10)
            for method in ('raw', run.METHOD):
                actual = run.base.exp.metrics(row[method], row)
                for metric, value in actual.items():
                    expected = info['metrics'][method][metric]
                    if value is None:
                        assert expected is None
                    else:
                        max_metric_error = max(max_metric_error, abs(value-expected))
                        assert abs(value-expected) < 5e-13
                counts['metric_sets'] += 1
            per_sequence[sequence] = info['metrics'][run.METHOD]
            positive = row['min_past_gt_dist'] <= 5
            old = np.where(reference['top1_geo_dist'] <= 5, 2, np.where(reference['union_hit'], 1, 0))
            new = np.where(row['top1_geo_dist'] <= 5, 2, np.where(row['union_hit'], 1, 0))
            transition += np.bincount(3*old[positive]+new[positive], minlength=9).reshape(3, 3)
            with np.load(paths['spec']) as z:
                bank = run.base.exp.old.base.bank_for(run.base.exp.OP, z)
            travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(cache['poses'][:, :3, 3], axis=0), axis=1))]
            conf = spec['confidence']['1']
            for index in (0, len(q)-1):
                t = int(q[index])
                ids = np.unique(np.r_[run.base.prefix_topk(key, t, t-29, 100), run.base.prefix_topk(bank[0], t, t-29, 100)])
                channels = (key[ids] @ key[t], bank[0][ids] @ bank[0][t],
                    run.base.align_scores(bank[1], bank[2], t, ids), run.base.column_align_scores(bank[3], t, ids))
                fused = sum(float(w)*run.base.minmax(v) for w, v in zip(conf['tuple'], channels))
                winner = int(ids[np.argmax(fused)])
                assert winner == j[index]
                raw = float(channels[3 if conf['decision'] == 'c_col' else 2][np.argmax(fused)])
                direct = run.factor.scalar_score(bank, travel, t, winner, conf['decision'], raw)
                max_score_error = max(max_score_error, abs(direct-row[run.METHOD][index]))
                assert abs(direct-row[run.METHOD][index]) < 1e-12
                counts['scalar_checks'] += 1
            for check in info['prefixes']:
                assert check['winner_equal'] and check['key_max_error'] < 4e-6 and check['score_error'] == 0
                counts['prefix_checks'] += 1
            for p in (path, path.with_suffix('.json'), key_path, reference_path):
                sources[str(p.relative_to(REPO))] = sha256(p)
            counts['ledgers'] += 1
            counts['rows'] += len(q)
        stress[view] = {'final_metrics': run.base.macro(per_sequence), 'matrix': transition.tolist(),
            'gained': int(transition[:2, 2].sum()), 'lost': int(transition[2, :2].sum())}
    with np.load(run.OUT / 'calibration_gradients.npz') as z:
        norms = [np.linalg.norm(z[k]) for k in ('rank', 'consistency')]
        value = float(np.clip(.25*norms[0]/norms[1], .01, 100.))
        assert abs(value-spec['consistency_coefficient']) < 1e-12
    arms = ['start'] + run.CFG['arms']
    table, stability, diversity = [], {}, {}
    for role in ('train', 'val'):
        stability[role], diversity[role] = {}, {}
        for arm in arms:
            result = summary['results'][role][arm]
            epoch = 0 if arm == 'start' else summary['selected_epoch'][arm]
            table.append({'role': role, 'arm': arm, 'epoch': epoch,
                **{m: 100*result['macro'][m] for m in run.base.exp.METRICS}})
            probe, probe_arm = (initial, 'initial') if arm == 'start' else (selected, arm)
            stability[role][arm] = probe['aggregates'][probe_arm][role]
            d = [r['clean_diversity'] for r in probe['results'][probe_arm][role].values()]
            diversity[role][arm] = {k: float(np.mean([r[k] for r in d])) for k in d[0]}
    with (run.OUT / 'final_metrics.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    # 새 정상 대조의 가중치와 이전 같은 손실 실행을 비교하되 작은 GPU 수치 차이를 숨기지 않는다.
    control_reproduction = {}
    for epoch in run.CFG['checkpoint_epochs']:
        paths = [run.OUT / 'clean_rank' / f'epoch{epoch}.pth', run.prior.OUT / 'rank_all' / f'epoch{epoch}.pth']
        a, b = [torch.load(p, map_location='cpu', weights_only=False)['model_state_dict'] for p in paths]
        error = max(float((a[k].double()-b[k].double()).abs().max()) for k in a)
        prior_result = run.base.read(run.prior.OUT / 'selection.json')['results']['rank_all'][str(epoch)]['macro']
        current = summary['all_validation_checkpoints']['clean_rank'][str(epoch)]['macro']
        control_reproduction[str(epoch)] = {'parameter_max_abs_error': error,
            'final_metric_delta_pp': {k: 100*(current[k]-prior_result[k]) for k in run.base.exp.METRICS}}
        for p in paths:
            sources[str(p.relative_to(REPO))] = sha256(p)
    marginal_consistency = {role: {m: 100*(summary['results'][role]['augmented_consistent']['macro'][m]
        - summary['results'][role]['augmented_rank']['macro'][m]) for m in run.base.exp.METRICS} for role in ('train', 'val')}
    same_epoch_consistency = {str(e): {m: 100*(
        summary['all_validation_checkpoints']['augmented_consistent'][str(e)]['macro'][m]
        - summary['all_validation_checkpoints']['augmented_rank'][str(e)]['macro'][m])
        for m in run.base.exp.METRICS} for e in run.CFG['checkpoint_epochs']}
    by_direction = {}
    for role in ('train', 'val'):
        by_direction[role] = {}
        for arm in run.CFG['arms']:
            totals = {str(g): dict(positive=0, old_correct=0, new_correct=0, gained=0, lost=0) for g in range(3)}
            epoch = summary['selected_epoch'][arm]
            for sequence in spec[role]:
                name = sequence.replace('/', '_') + '.npz'
                direction_path = REPO / 'results/pdv2_context_transfer_20261024' / role / name
                group = run.base.exp.arrays(direction_path)
                old = run.base.exp.arrays(run.factor.OUT / role / sequence.replace('/', '_') / 'new.npz')
                new = run.base.exp.arrays(run.OUT / arm / f'epoch{epoch}' / role / name)
                index = np.searchsorted(old['query_idx'], group['query'])
                np.testing.assert_array_equal(old['query_idx'][index], group['query'])
                np.testing.assert_array_equal(new['query_idx'], old['query_idx'])
                first, second = old['top1_geo_dist'][index] <= 5, new['top1_geo_dist'][index] <= 5
                for g in range(3):
                    mask = group['group'] == g
                    row = {'positive': int(mask.sum()), 'old_correct': int((mask & first).sum()),
                        'new_correct': int((mask & second).sum()), 'gained': int((mask & ~first & second).sum()),
                        'lost': int((mask & first & ~second).sum())}
                    for k, value in row.items():
                        totals[str(g)][k] += value
                sources[str(direction_path.relative_to(REPO))] = sha256(direction_path)
            expected = summary['candidate_transitions'][arm][role][str(epoch)]
            for k in ('gained', 'lost'):
                assert sum(r[k] for r in totals.values()) == expected[k]
            by_direction[role][arm] = totals
    integrated = {'final_metrics_percent': table, 'initial_stress': stress, 'stability': stability,
        'clean_context_diversity_sequence_mean': diversity, 'consistency_coefficient': spec['consistency_coefficient'],
        'calibration_gradient_cosine': calibration['cosine'], 'candidate_transitions': summary['candidate_transitions'],
        'eligible_candidates': summary['eligible_candidates'], 'control_reproduction': control_reproduction,
        'consistency_minus_augmentation_pp': marginal_consistency,
        'same_epoch_consistency_minus_augmentation_pp': same_epoch_consistency,
        'direction_transitions_vs_initial': by_direction,
        'training_support': support,
        'scope': '시드1 검증 예비. 정상 입력 최종 판정으로 선택한다. 안정성 지표는 성능 지표가 아니다. '
                 '간선 가림은 실제 반대편 관측의 합성이 아니다.', 'test_used': False}
    write_json(run.OUT / 'integrated_report.json', integrated)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    names = ['Start', 'Clean training', 'Augmented ranking', '+ Relative consistency']
    colors = ['#888888', '#336699', '#4d9664', '#c35a32']
    values = [100*summary['results']['val'][a]['macro']['AP'] for a in arms]
    axes[0, 0].scatter(values, np.arange(4), c=colors, s=60)
    for i, v in enumerate(values):
        axes[0, 0].annotate(f'{v:.4f}', (v, i), xytext=(4, 5), textcoords='offset points', fontsize=9)
    axes[0, 0].set(yticks=np.arange(4), yticklabels=names, xlabel='Final validation AP (%)', title='A. Normal inference')
    axes[0, 0].invert_yaxis()
    axes[0, 0].margins(x=.35)
    for index, view in enumerate(run.CFG['views']):
        axes[0, 1].bar(np.arange(4)+(index-.5)*.34,
            [stability['val'][a][view]['all']['mean'] for a in arms], width=.34, label=view)
    axes[0, 1].set(xticks=np.arange(4), xticklabels=['Start', 'Clean', 'Augment', '+ Consist.'],
        ylabel='Mean 1 - context cosine', title='B. Perturbation drift on validation')
    axes[0, 1].legend()
    for a, label, color in zip(arms[1:], names[1:], colors[1:]):
        delta = [100*(summary['results'][r][a]['macro']['AP'] - summary['results'][r]['start']['macro']['AP'])
                 for r in ('train', 'val')]
        axes[1, 0].plot([0, 1], delta, 'o-', label=label, color=color)
    axes[1, 0].axhline(0., color='#888888', linewidth=.8)
    axes[1, 0].set(xticks=[0, 1], xticklabels=['Train', 'Validation'], ylabel='Final AP change (pp)', title='C. Transfer of improvement')
    axes[1, 0].legend(fontsize=8)
    stress_values = [100*summary['results']['val']['start']['macro']['AP']] + [100*stress[v]['final_metrics']['macro']['AP'] for v in run.CFG['views']]
    axes[1, 1].plot(np.arange(3), stress_values, 'o-', color='#336699')
    for i, v in enumerate(stress_values):
        axes[1, 1].annotate(f'{v:.4f}', (i, v), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=9)
    axes[1, 1].set(xticks=np.arange(3), xticklabels=['Normal', 'Drop edges', 'Stride 2'],
        ylabel='Final validation AP (%)', title='D. Fixed initial model, inference intervention')
    axes[1, 1].margins(y=.3)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    for ext in ('png', 'pdf'):
        fig.savefig(run.OUT / ('temporal_augmentation.' + ext), dpi=180)
    plt.close(fig)
    for name in ('integrated_report.json', 'final_metrics.csv', 'temporal_augmentation.png', 'temporal_augmentation.pdf'):
        p = run.OUT / name
        sources[str(p.relative_to(REPO))] = sha256(p)
    for path, digest in sources.items():
        assert sha256(REPO / path) == digest, path
    write_json(run.OUT / 'integrated_validation.json', {'passed': True, 'sources': sources, **counts,
        'maximum_metric_error': max_metric_error, 'maximum_score_error': max_score_error,
        'new_training_updates': audit['new_training_updates'], 'test_used': False})
    print('AUGMENTATION_REPORT_COMPLETE', summary['eligible_candidates'], len(sources), counts, flush=True)


if __name__ == '__main__':
    main()
