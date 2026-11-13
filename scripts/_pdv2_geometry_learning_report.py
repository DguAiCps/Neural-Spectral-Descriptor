#!/usr/bin/env python3
"""기울기 추적·정밀도·감쇠·고정 추론 개입의 근거를 함께 검산한다."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
from _pdv2_no_context_ablation import sha256, write_json

OUT = REPO / 'results/pdv2_geometry_learning_report_20261023'


def read(path: Path) -> dict:
    """저장된 JSON을 읽는다."""
    return json.loads(path.read_text())


def main() -> None:
    """학습 기울기의 대수 검산과 최종 판정 출처를 묶어 저장한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    root = REPO / 'results'
    trace = root / 'pdv2_geometry_gradient_20261023'
    precision = root / 'pdv2_geometry_precision_20261023'
    nullspace = root / 'pdv2_geometry_nullspace_20261023'
    decay = root / 'pdv2_geometry_decay_20261023'
    usage = root / 'pdv2_geometry_usage_20261023'
    sources = {str(Path(__file__).relative_to(REPO)): sha256(Path(__file__))}
    protocols = [read(trace / 'protocol.json'), read(decay / 'protocol.json'), read(usage / 'protocol.json')]
    for spec in protocols:
        sources.update(spec['runtime'])
        sources.update(spec['inputs'])
    for folder in (decay, usage):
        audit = read(folder / 'completion_validation.json')
        assert audit['passed'] and not audit['test_used']
        sources.update(audit['sources'])
    precision_data, null_data = [read(p / 'diagnosis.json') for p in (precision, nullspace)]
    sources.update(precision_data['sources'])
    sources.update(null_data['sources'])
    traces, diagnostics, initial, order, trace_count = {}, {}, None, None, 0
    max_gradient_identity_error = max_clipping_relative_error = 0.
    for arm in protocols[0]['config']['arms']:
        folder = trace / 's1' / arm
        current_initial = read(folder / 'initial_model.json')
        complete = read(folder / 'training_complete.json')
        if initial is None:
            initial, order = current_initial, complete['orders']
        assert current_initial == initial and complete['orders'] == order and complete['updates'] == 900
        rows = read(folder / 'gradient_trace.json')['records']
        assert [v['step'] for v in rows] == protocols[0]['config']['trace_steps']
        for row in rows:
            factor = min(1., 1. / (row['total_gradient_norm_before_clip'] + 1e-6))
            for group, value in row['after_clip'].items():
                if not value['parameters_with_grad']:
                    continue
                gn, dn = value['loss_gradient_norm'], value['decay_gradient_norm']
                expected = gn ** 2 + dn ** 2 + 2 * gn * dn * (value['loss_decay_cosine'] or 0.)
                error = abs(expected - value['combined_gradient_norm'] ** 2) / max(expected, 1e-30)
                max_gradient_identity_error = max(error, max_gradient_identity_error)
                assert error < 1e-10
                expected_gradient = row['before_clip'][group]['loss_gradient_norm'] * factor
                clipping = abs(gn - expected_gradient) / max(expected_gradient, 1e-30)
                max_clipping_relative_error = max(clipping, max_clipping_relative_error)
                assert clipping < 3e-6
        late = [v for v in rows if v['step'] >= 450]
        diagnostics[arm] = {'late_observations': len(late),
            'late_median_loss_to_decay_ratio': {g: float(np.median([
                v['after_clip'][g]['loss_to_decay_norm_ratio'] for v in late])) for g in
                ('edge_encoder', 'bias0_hidden', 'bias0_last', 'bias1_hidden', 'bias1_last')},
            'last_bias': rows[-1]['bias_after_update'],
            'replay_comparison': read(folder / 'trace_validation.json')['checkpoint_comparison']}
        traces[arm] = rows
        trace_count += len(rows)
        for p in folder.glob('*.json'):
            sources[str(p.relative_to(REPO))] = sha256(p)
        for name, digest in complete['checkpoints'].items():
            p = folder / 'checkpoints' / name
            assert sha256(p) == digest
            sources[str(p.relative_to(REPO))] = digest
    assert trace_count == 84
    precision_summary = {}
    for arm, stages in precision_data['records'].items():
        precision_summary[arm] = {}
        for stage, rows in stages.items():
            assert len(rows) == 3
            precision_summary[arm][stage] = {mode: {
                group: float(np.mean([v[mode]['gradients'][group]['loss_gradient_norm'] for v in rows]))
                for group in ('edge_encoder', 'bias0_hidden', 'bias0_last', 'bias1_hidden', 'bias1_last')}
                for mode in ('fp16', 'fp32')}
    null_summary = {}
    for arm, rows in null_data['records'].items():
        null_summary[arm] = {mode: {layer: {k: float(np.mean([v[mode]['layers'][layer][k] for v in rows]))
            for k in ('common_to_all_gradient_norm_ratio', 'projected_to_raw_weight_gradient_norm_ratio', 'raw_projected_cosine')}
            for layer in ('0', '1')} for mode in ('fp16', 'fp32')}
    decay_summary, usage_summary = read(decay / 'summary.json'), read(usage / 'summary.json')
    bias = read(decay / 'bias_observations.json')['models']
    bias_mean = {arm: {layer: float(np.mean([v[layer]['centered_bias_rms'] for v in rows.values()]))
                      for layer in ('0', '1')} for arm, rows in bias.items()}
    final = {'reference': decay_summary['results']['diff_tanh']['arc_unique_b100'],
             'no_geometry_decay': decay_summary['results']['diff_tanh_nodecay']['arc_unique_b100'],
             'no_geometry_decay_content_only': usage_summary['results']['content']['arc_unique_b100']}
    np.testing.assert_allclose(final['no_geometry_decay']['macro']['AP'],
        usage_summary['results']['full']['arc_unique_b100']['macro']['AP'], atol=5e-13, rtol=0)
    result = {'gradient_trace': diagnostics, 'trace_observations': trace_count,
        'precision': precision_summary, 'null_direction': null_summary,
        'validation_mean_bias_rms': bias_mean, 'final_validation': final,
        'decay_change_pp': decay_summary['nodecay_minus_reference_pp'],
        'remove_geometry_after_nodecay_pp': usage_summary['content_minus_full_pp'],
        'bias_rms_increase_ratio': {k: bias_mean['diff_tanh_nodecay'][k] / bias_mean['diff_tanh'][k] for k in ('0', '1')},
        'test_used': False, 'new_optimizer_updates': 3609,
        'training_scope': '완전학습4개×900갱신과 정밀도진단용3개×3갱신이다. 같은초기값·배치순서는 확인했지만 기존학습의 최종가중치는 재현되지 않았다. 이번기준·개입을 직접최종평가하며 과거모델수치를 대신 쓰지 않는다.',
        'promotion': None}
    write_json(OUT / 'summary.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout='constrained')
    for arm, label in (('diff_linear', 'Diff / linear V'), ('standard_linear', 'Standard / linear V'),
                       ('diff_tanh', 'Diff / tanh V')):
        rows = traces[arm]
        values = np.array([v['after_clip']['edge_encoder']['loss_to_decay_norm_ratio'] for v in rows])
        axes[0].plot([v['step'] for v in rows], np.where(values > 0, values, np.nan), label=label)
    axes[0].axhline(1., color='gray', linestyle='--', linewidth=1)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Optimizer update')
    axes[0].set_ylabel('Loss gradient / L2 gradient norm')
    axes[0].set_title('Edge encoder: actual training')
    axes[0].legend(fontsize=8)
    for offset, layer, label in ((-.17, '0', 'Layer 1'), (.17, '1', 'Layer 2')):
        axes[1].bar(np.arange(2) + offset, [bias_mean[a][layer] for a in ('diff_tanh', 'diff_tanh_nodecay')], width=.32, label=label)
    axes[1].set_xticks([0, 1], ['Reference', 'No geometry decay'])
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Centered geometry bias RMS')
    axes[1].set_title('Geometry branch becomes active')
    axes[1].legend(fontsize=8)
    values = [100 * x['macro']['AP'] for x in final.values()]
    axes[2].plot(range(3), values, 'o-', color='#2166ac')
    axes[2].set_xticks(range(3), ['Reference', 'No geom. decay', 'Then remove bias'], rotation=15)
    axes[2].set_ylabel('Final validation AP (%)')
    axes[2].set_title('Fixed epoch 20, same final decision')
    axes[2].set_ylim(min(values) - .08, max(values) + .08)
    for i, value in enumerate(values):
        axes[2].annotate(f'{value:.3f}', (i, value), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(OUT / 'geometry_learning.png', dpi=180)
    fig.savefig(OUT / 'geometry_learning.pdf')
    plt.close(fig)
    for folder in (precision, nullspace, decay, usage):
        for name in ('diagnosis.json', 'summary.json', 'completion_validation.json', 'protocol.json'):
            p = folder / name
            if p.exists():
                sources[str(p.relative_to(REPO))] = sha256(p)
    for p in (OUT / 'summary.json', OUT / 'geometry_learning.png', OUT / 'geometry_learning.pdf'):
        sources[str(p.relative_to(REPO))] = sha256(p)
    for path, digest in sources.items():
        assert sha256(REPO / path) == digest, path
    write_json(OUT / 'completion_validation.json', {'passed': True, 'sources': sources,
        'trace_observations': trace_count, 'precision_backward_passes': 54, 'null_direction_backward_passes': 18,
        'maximum_gradient_identity_error': max_gradient_identity_error,
        'maximum_clipping_relative_error': max_clipping_relative_error,
        'distinct_final_ledgers': 27, 'distinct_final_rows': 297708,
        'test_used': False, 'new_optimizer_updates': 3609})
    print('GEOMETRY_LEARNING_VALIDATED', len(sources), 'sources', flush=True)
    for name, row in final.items():
        print('FINAL', name, {k: round(100 * v, 6) for k, v in row['macro'].items()}, flush=True)


if __name__ == '__main__':
    main()
