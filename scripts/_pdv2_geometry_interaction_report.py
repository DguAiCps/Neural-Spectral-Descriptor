#!/usr/bin/env python3
"""기하 결합의 최종 결과·고정 개입·KITTI 손실 진단을 한곳에 모은다."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from _pdv2_no_context_ablation import sha256, write_json

ROOT = REPO / 'results/pdv2_geometry_interaction_20261023'
USAGE = REPO / 'results/pdv2_geometry_interaction_usage_20261023'


def read(path: Path) -> dict:
    """JSON을 읽는다."""
    return json.loads(path.read_text())


def main() -> None:
    """검산한 수치를 합치고 공유 가능한 그림을 저장한다."""
    summary, usage = read(ROOT / 'summary.json'), read(USAGE / 'summary.json')
    diagnosis = read(ROOT / 'mechanism_diagnosis.json')
    audit, usage_audit = read(ROOT / 'completion_validation.json'), read(USAGE / 'completion_validation.json')
    assert audit['passed'] and usage_audit['passed'] and diagnosis['passed']
    assert audit['ledgers'] == usage_audit['ledgers'] == 27
    assert audit['rows'] == usage_audit['rows'] == 297708
    sources = {**audit['sources'], **usage_audit['sources'], **diagnosis['sources']}
    for p in (Path(__file__), ROOT / 'summary.json', ROOT / 'mechanism_diagnosis.json',
              ROOT / 'completion_validation.json', USAGE / 'summary.json', USAGE / 'completion_validation.json'):
        sources[str(p.relative_to(REPO))] = sha256(p)
    method = 'arc_unique_b100'
    final = {a: v[method] for a, v in summary['results'].items()}
    for metric, value in final['interaction']['macro'].items():
        np.testing.assert_allclose(value, usage['results']['full'][method]['macro'][metric], atol=5e-13, rtol=0)
    for arm in ('interaction_off', 'geometry_off'):
        final[arm] = usage['results'][arm][method]
    comparisons = dict(diagnosis['final']['comparisons'])
    comparisons['previous_nodecay'] = diagnosis['previous_decay_final']['comparisons']['nodecay']
    loss_counts = {}
    for arm, comparison in comparisons.items():
        sequences = [v for k, v in comparison['sequences'].items() if 'loss_categories' in v]
        loss_counts[arm] = {name: sum(s['loss_categories'][name] for s in sequences)
                           for name in ('missing_union', 'key_prefers_true_fusion_wrong', 'key_also_wrong')}
        loss_counts[arm]['gains'] = sum(s['gains'] for s in sequences)
        loss_counts[arm]['losses'] = sum(s['losses'] for s in sequences)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    names = ['additive', 'interaction', 'temperature', 'interaction_off', 'geometry_off']
    labels = ['Additive', 'Edge interaction', 'Temperature', 'Interaction off*', 'Geometry off*']
    ap = [100 * final[a]['macro']['AP'] for a in names]
    bars = axes[0].barh(labels[::-1], ap[::-1], color=['#4b78a8', '#ce8d43', '#668b66', '#aaa', '#ccc'][::-1])
    axes[0].bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
    axes[0].set_xlim(min(ap) - .08, max(ap) + .12)
    axes[0].set_xlabel('Final validation AP (%)')
    axes[0].set_title('Fixed epoch 20; one seed\n*Inference intervention on interaction model')
    arms = ['previous_nodecay', 'interaction', 'temperature']
    bottom = np.zeros(3)
    for key, label, color in [('missing_union', 'Correct candidate missing', '#c66b5a'),
                              ('key_also_wrong', 'Key also prefers a wrong candidate', '#b7a27a'),
                              ('key_prefers_true_fusion_wrong', 'Key correct, fusion wrong', '#698bb6')]:
        values = np.array([loss_counts[a][key] for a in arms])
        axes[1].bar(arms, values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[1].set_xticks(range(3), ['Prior no decay', 'Edge interaction', 'Temperature'])
    axes[1].set_ylabel('Previously correct KITTI queries lost')
    axes[1].set_title('Losses only; gains are reported separately')
    axes[1].legend(fontsize=8)
    for ext in ('png', 'pdf'):
        p = ROOT / ('geometry_interaction.' + ext)
        figure.savefig(p, dpi=180)
        sources[str(p.relative_to(REPO))] = sha256(p)
    plt.close(figure)
    report = {'final_validation': final, 'kitti_losses': loss_counts,
        'attention_sequence_mean': diagnosis['attention']['validation_sequence_mean'],
        'ap_decomposition': {a: c['ap_arithmetic_decomposition'] for a, c in comparisons.items()},
        'new_training_updates': 2700, 'distinct_new_final_ledgers': 45, 'distinct_new_final_rows': 496180,
        'test_used': False, 'promotion': False, 'scope': '단일시드·검증최종판정. 추가곱제거와기하제거는고정가중치추론개입. 이전감쇠대조는별도학습쌍이며새실험의학습효과와혼합하지않는다.'}
    write_json(ROOT / 'integrated_report.json', report)
    sources[str((ROOT / 'integrated_report.json').relative_to(REPO))] = sha256(ROOT / 'integrated_report.json')
    for p, digest in sources.items():
        assert sha256(REPO / p) == digest, p
    write_json(ROOT / 'integrated_validation.json', {'passed': True, 'sources': sources,
        'primary_audit_ledgers': audit['ledgers'], 'usage_audit_ledgers': usage_audit['ledgers'],
        'distinct_new_final_ledgers': 45, 'distinct_new_final_rows': 496180,
        'test_used': False})
    print('INTEGRATED_REPORT_PASSED', len(sources), flush=True)
    for name in names:
        print(name, final[name]['macro'], flush=True)
    print('KITTI_LOSSES', loss_counts, flush=True)


if __name__ == '__main__':
    main()
