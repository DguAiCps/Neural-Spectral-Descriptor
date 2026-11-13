#!/usr/bin/env python3
"""46시퀀스 확대 평가의 원장·지표·출처를 최종 검산한다."""
from pathlib import Path
import time
import numpy as np
import _review_direction_metrics as base

OUT = base.OUT / 'expanded46'


def independent_metrics(data: dict):
    """64질의의 모든 고유 임계값을 직접 수락해 PR 지표를 검산한다."""
    positive = data['min_past_gt_dist'] <= 5
    negative = data['min_past_gt_dist'] > 25
    if not positive.any():
        return None
    correct = positive & (data['top1_geo_dist'] <= 5)
    eligible = (positive | negative) & np.isfinite(data['score'])
    thresholds = sorted(set(data['score'][eligible].tolist()), reverse=True)
    precision, recall = [1.], [0.]
    for threshold in thresholds:
        accepted = eligible & (data['score'] >= threshold)
        tp = int((accepted & correct).sum())
        precision.append(tp / int(accepted.sum()))
        recall.append(tp / int(positive.sum()))
    precision, recall = np.array(precision), np.array(recall)
    envelope = np.array([precision[i:].max() for i in range(len(precision))])
    ap = float(np.sum((recall[1:] - recall[:-1]) * envelope[1:]))
    f1 = float(np.max(2*precision*recall / np.maximum(precision+recall, 1e-300)))
    return dict(AP=ap, max_F1=f1, **{'R@100P':float(recall[precision == 1].max()), 'R@1':float(correct.sum()/positive.sum())})


def main() -> None:
    """완료 파일을 다시 읽고 검산 범위·출처 해시를 명시한다."""
    spec = base.read(OUT/'yaw_protocol.json')
    summary = base.read(OUT/'yaw_summary.json')
    causes = base.read(OUT/'causes/summary.json')
    direction = base.read(base.OUT/'direction_summary.json')
    assert summary['status'] == causes['status'] == 'passed'
    assert direction['original_totals_reproduced'] and direction['legacy_direction_labels_reproduced']
    assert direction['counts']['queries'] == 195726 and direction['counts']['positive'] == 61682
    sources = {}
    for protocol in [OUT/'yaw_protocol.json', OUT/'causes/protocol.json', OUT/'grid_probe/protocol.json', OUT/'roll_check/protocol.json',
                     base.OUT/'direction_learning/protocol.json', base.OUT/'direction_learning/sequence_protocol.json']:
        for path, value in base.read(protocol).get('sources', {}).items():
            assert base.digest(base.ROOT/path) == value, path
            sources[path] = value
    ledger_count = rows = encoding_count = baseline_rows = 0
    independent_error = baseline_error = 0.
    for seq, queries in spec['queries'].items():
        folder = OUT/'yaw'/seq.replace('/', '_')
        expected_frames = np.unique(np.concatenate([np.arange(q-15, q+16) for q in queries]))
        assert len(expected_frames) == spec['support_frames'][seq]
        assert base.read(folder/'baseline.json')['status'] == base.read(folder/'rotate.json')['status'] == 'passed'
        for angle in spec['angles']:
            encoded = folder/f'encoded_{angle:g}.npz'
            assert base.digest(encoded) == base.read(encoded.with_suffix('.json'))['sha256']
            with np.load(encoded) as data:
                np.testing.assert_array_equal(data['frames'], expected_frames)
                assert np.isfinite(data['raw']).all() and np.isfinite(data['coeff']).all()
            encoding_count += 1
            for tag in spec['cases']:
                path = folder/f'{tag}_{angle:g}.npz'
                data = dict(np.load(path))
                assert len(data['query_idx']) == 64
                actual = independent_metrics(data)
                expected = summary['final_recognition'][str(angle)]['per_seed'][tag]['sequences'][seq]
                if actual is None:
                    assert expected is None
                else:
                    for key in base.KEYS:
                        error = abs(actual[key] - expected[key])
                        independent_error = max(error, independent_error)
                        assert error <= 1e-12
                if angle == 0:
                    original = dict(np.load(base.MAIN/'test'/seq.replace('/', '_')/(tag+'.npz')))
                    indices = np.searchsorted(original['query_idx'], data['query_idx'])
                    np.testing.assert_array_equal(data['j_star'], original['j_star'][indices])
                    error = float(np.max(abs(data['score'] - original['score'][indices])))
                    assert error <= 8e-6
                    baseline_error = max(error, baseline_error)
                    baseline_rows += 64
                ledger_count += 1
                rows += 64
    assert (ledger_count, rows, encoding_count, baseline_rows) == (966, 61824, 322, 8832)
    assert causes['rows'] == 39744 and causes['actual_ledger_checks'] == 6624
    roll = base.read(OUT/'roll_check/summary.json')
    assert roll['status'] == 'passed' and roll['rows'] == 552
    assert all(row['interpolation_commutation_max']['max'] == 0 for row in roll['results'].values())
    learning = base.read(base.OUT/'direction_learning/sequence_summary.json')
    assert learning['original_intervals_reproduced']
    synthetic = base.read(base.OUT/'direction_learning/synthetic_validation.json')
    assert synthetic['status'] == 'passed' and synthetic['duplicated_ledger_checks'] == 60
    files = [OUT/'yaw_summary.json', OUT/'yaw_metrics.csv', OUT/'yaw_city_dataset.csv',
             OUT/'causes/summary.json', OUT/'grid_probe/summary.json', OUT/'phase_covariance_proposal.json', OUT/'roll_check/summary.json',
             base.OUT/'direction_summary.json', base.OUT/'direction_city_dataset.csv',
             base.OUT/'direction_learning/summary.json', base.OUT/'direction_learning/sequence_summary.json',
             base.OUT/'direction_learning/fixed_city_block_summary.json', base.OUT/'direction_learning/learning_intervals.csv',
             base.OUT/'direction_learning/synthetic_validation.json',
             base.ROOT/'docs/RAW/실험일지/2026-11-05_46시퀀스-방향학습과-yaw-원인분리.md',
             Path(__file__), base.ROOT/'scripts/_review_direction_expanded_report.py',
             base.ROOT/'scripts/_review_direction_write_report.py']
    result = dict(status='passed', completed_unix=time.time(), sequences=46, queries=2944,
        ledger_files=ledger_count, ledger_rows=rows, encoding_archives=encoding_count,
        raw_support_frames=77595, raw_encode_calls_including_reuse=543165,
        baseline_query_seed_checks=baseline_rows, baseline_winner_mismatches=0, baseline_max_score_error=baseline_error,
        independent_metric_max_error=independent_error, independent_metric_scope='966원장의64질의마다모든고유임계값을직접수락한PR곡선과집계JSON을비교한다.',
        cause_rows=causes['rows'], cause_actual_ledger_checks=causes['actual_ledger_checks'],
        cause_actual_max_score_error=causes['actual_max_score_error'],
        original_direction_totals_reproduced=True, original_sequence_bootstrap_intervals_reproduced=True,
        weighted_pr_synthetic_checks=60,
        exact_grid_roll_cpu_checks=552, exact_grid_interpolation_commutation_passed=True,
        protocol_source_hashes_checked=sources,
        artifacts_sha256={str(p.relative_to(base.ROOT)):base.digest(p) for p in files},
        process_scope='GPU진단두실행의종료와GPU반환은외부프로세스상태로확인해별도기록한다. 이검사는CPU검산이다.')
    base.write(OUT/'completion_validation.json', result)
    print('COMPLETION PASSED', ledger_count, rows, independent_error)


if __name__ == '__main__':
    main()
