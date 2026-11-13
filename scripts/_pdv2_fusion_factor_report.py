#!/usr/bin/env python3
"""후보·키 점수·정규화 진단을 통합하고 최종 지표 그림을 만든다."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results/pdv2_fusion_factor_20261023'


def read(path: Path) -> dict:
    """JSON 결과를 읽는다."""
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    """파일 전체의 SHA256을 계산한다."""
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path: Path, data: dict) -> None:
    """사람이 읽을 수 있는 JSON으로 저장한다."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main() -> None:
    """표본 선택 없는 전체 집계와 사후 단일 분모 대조를 구분한다."""
    summary = read(OUT / 'summary.json')
    probe = read(OUT / 'key_width_probe/summary.json')
    audits = [read(OUT / 'completion_validation.json'),
              read(OUT / 'key_width_probe/completion_validation.json')]
    sources = {}
    for audit in audits:
        assert audit['passed']
        for name, sha in audit['sources'].items():
            if name in sources:
                assert sources[name] == sha, name
            sources[name] = sha
    for name, sha in read(OUT / 'supplemental_sources.json')['sources'].items():
        assert sources.get(name, sha) == sha
        sources[name] = sha
    for name, sha in sources.items():
        assert digest(REPO / name) == sha, name
    results, errors = {}, {}
    for role in ('train', 'val'):
        results[role] = dict(summary['results'][role])
        results[role]['new_pool_old_key_width'] = probe['results'][role]
        cohort, changed = Counter(), {}
        for item in summary['cohorts'][role].values():
            cohort.update(item)
        for direction in ('loss', 'gain'):
            totals, corrections = Counter(), Counter()
            for sequence, item in summary['changed_correctness'][role].items():
                cases = item[direction]
                totals.update({k: v for k, v in cases.items() if k != 'correct_by_variant'})
                corrections.update(cases['correct_by_variant'])
            corrections['new_pool_old_key_width'] = sum(c['correct']
                for d in probe['diagnostics'][role].values()
                for c in d['parent_changed_cases'] if c['change'] == direction)
            changed[direction] = {**totals, 'correct_by_variant': dict(corrections)}
        transitions = {}
        for variant, matrix in summary['new_to_variant_transitions'][role].items():
            m = np.asarray(matrix)
            transitions[variant] = {'corrected': int(m[:2, 2].sum()), 'damaged': int(m[2, :2].sum())}
        transitions['new_pool_old_key_width'] = {
            'corrected': sum(d['gains_vs_new'] for d in probe['diagnostics'][role].values()),
            'damaged': sum(d['losses_vs_new'] for d in probe['diagnostics'][role].values())}
        assert transitions['old']['corrected'] == changed['loss']['count']
        assert transitions['old']['damaged'] == changed['gain']['count']
        errors[role] = {'cohorts': dict(cohort), 'changed_correctness': changed,
            'new_to_variant': transitions,
            'key_prefers_correct_among_available_errors_percent':
                100 * cohort['key_prefers_correct_over_final_wrong'] / cohort['positive_with_candidate_but_wrong'],
            'queries': sum(d['queries'] for d in probe['diagnostics'][role].values()),
            'positive': sum(d['positive'] for d in probe['diagnostics'][role].values())}
    # 주 진단에서 발견한 대표 사례이며 전체 현상의 빈도 추정 표본이 아니다.
    sequence = 'kitti360_0004'
    cases = read(OUT / 'val' / sequence / 'diagnosis.json')['changed_correctness']
    case = next(x for x in cases if x['q'] == 8425)
    target = next(x for x in probe['diagnostics']['val']['kitti360/0004']['parent_changed_cases'] if x['q'] == 8425)
    assert case['true_in_old'] and case['true_in_new'] and case['false_in_old']
    old_component, new_component = [case['fusion_pair_components'][v] for v in ('old', 'new')]
    new_key_margin = new_component[0] * case['channel_widths']['new'][0]
    replacement_component = new_key_margin / target['old_key_width_on_new_pool']
    example = {'sequence': 'kitti360/0004', 'query': 8425,
        'true': case['true'], 'false': case['false'], 'false_distance': case['false_distance'],
        'old': {'raw_key_pair_margin': case['margins']['old']['total'],
                'key_width': case['channel_widths']['old'][0],
                'normalized_key_pair_margin': old_component[0],
                'other_channels_sum': sum(old_component[1:]), 'fusion_margin': sum(old_component)},
        'new': {'raw_key_pair_margin': case['margins']['new']['total'],
                'key_width': case['channel_widths']['new'][0],
                'normalized_key_pair_margin': new_component[0],
                'other_channels_sum': sum(new_component[1:]), 'fusion_margin': sum(new_component)},
        'key_width_probe': {**target, 'normalized_key_pair_margin': replacement_component,
            'other_channels_sum': sum(new_component[1:]),
            'fusion_margin': replacement_component + sum(new_component[1:])}}
    report = {'final_results': results, 'errors': errors, 'normalization_example': example,
        'ap_arithmetic_decomposition': summary['ap_arithmetic_decomposition'],
        'scope': {'seed': 1, 'epoch': 20, 'new_training_updates': 0, 'test_used': False,
            'main_conditions': 6, 'posthoc_key_width_conditions': 1},
        'validation': {k: sum(a[k] for a in audits) for k in ('ledgers', 'rows')},
    }
    report['scope']['interpretation'] = ('모든 조건에서 기존 최종 시간 판정을 계산한다. '
        '조건부 개입이며 독립 인과 기여가 아니다. 학습 GT와 검증 KISS의 절대 AP 차이를 일반화 차이로 해석하지 않는다.')
    write(OUT / 'integrated_report.json', report)
    plot(report)
    for p in (Path(__file__), OUT / 'integrated_report.json', OUT / 'fusion_factor_summary.png',
              OUT / 'fusion_factor_summary.pdf', OUT / 'completion_validation.json',
              OUT / 'key_width_probe/completion_validation.json', OUT / 'supplemental_sources.json'):
        sources[str(p.relative_to(REPO))] = digest(p)
    write(OUT / 'integrated_validation.json', {'passed': True, 'sources': sources,
        'source_count': len(sources), **report['validation'], 'test_used': False, 'training_updates': 0})
    print('FUSION_FACTOR_INTEGRATED', len(sources), report['validation'])


def plot(report: dict) -> None:
    """최종 AP와 한 실제 오류의 정규화 효과를 그린다."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    names = list(report['final_results']['val'])
    labels = ['Additive baseline', 'Interaction baseline', 'Old pool, new key',
              'New pool, old key', 'Old pool ranges, new key', 'Raw key scoring only',
              'Old key width only (post hoc)']
    current = report['final_results']['val']['new']['macro']['AP']
    delta = [100 * (report['final_results']['val'][v]['macro']['AP'] - current) for v in names]
    colors = ['#637b9f' if v >= 0 else '#bf7665' for v in delta]
    axes[0].barh(np.arange(len(names)), delta, color=colors)
    for i, (name, value) in enumerate(zip(names, delta)):
        ap = 100 * report['final_results']['val'][name]['macro']['AP']
        axes[0].text(.16, i, f'{ap:.4f}%', va='center', ha='right', fontsize=9)
    axes[0].set_yticks(np.arange(len(names)), labels)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color='#888', linewidth=.8)
    axes[0].set_xlim(min(-.95, min(delta) - .1), max(.17, max(delta) + .12))
    axes[0].set_xlabel('Final AP change vs interaction (percentage points)')
    axes[0].set_title('Validation 9; frozen epoch 20; one seed\nFinal temporal decision is retained', fontsize=11)
    example = report['normalization_example']
    x = np.arange(3)
    for offset, key, label, color in [(-.25, 'normalized_key_pair_margin', 'Normalized key', '#637b9f'),
            (0, 'other_channels_sum', 'Other three channels', '#bf7665'),
            (.25, 'fusion_margin', 'Fused margin', '#78a787')]:
        values = [example[name][key] for name in ('old', 'new', 'key_width_probe')]
        bars = axes[1].bar(x + offset, values, .23, label=label, color=color)
        axes[1].bar_label(bars, fmt='%.3f', fontsize=8, padding=2)
    axes[1].axhline(0, color='#888', linewidth=.8)
    axes[1].set_xticks(x, ['Additive', 'Interaction', 'Old key width'])
    axes[1].set_ylabel('True candidate minus false candidate')
    axes[1].set_title('KITTI-360 0004, query 8425\nSame candidate pair; false place is 947.7 m away', fontsize=11)
    axes[1].set_ylim(-.59, .58)
    axes[1].legend(loc='lower center', ncol=3, fontsize=8, framealpha=1)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / ('fusion_factor_summary.' + ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
