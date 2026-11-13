#!/usr/bin/env python3
"""고정 후보에서의 문맥 변화와 학습·검증의 방향·장면 반복성을 진단한다."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_fusion_guard as prior
from _pdv2_fusion_factor_lib import fuse_with_range
from _pdv2_pair_evaluate import direction_groups
from _pdv2_no_context_ablation import sha256, write_json

OUT = REPO / 'results/pdv2_context_transfer_20261024'
ARM = 'rank_filtered_guarded'


def neighborhood(raw: np.ndarray, q: int, j: int) -> float:
    """±5 원 특징의 정·역 순서 대응 중 큰 평균 코사인이다. 중심도 포함한다."""
    delta = np.arange(-5, 6)
    values = []
    for sign in (1, -1):
        a, b = q + delta, j + sign * delta
        valid = (a >= 0) & (a < len(raw)) & (b >= 0) & (b < len(raw))
        values.append(float(np.sum(raw[a[valid]] * raw[b[valid]], axis=1).mean()))
    return max(values)


def aggregate(rows: dict, mask: np.ndarray) -> dict:
    """변경 건수와 해당 분모를 함께 보존한다. 최종 AP를 대신하지 않는다."""
    old, new, eligible = rows['old_correct'], rows['new_correct'], rows['eligible']
    gain, loss = mask & ~old & new, mask & old & ~new
    data = {'positive': int(mask.sum()), 'old_correct': int((mask & old).sum()),
        'old_wrong': int((mask & ~old).sum()), 'gained': int(gain.sum()), 'lost': int(loss.sum()),
        'gain_rate_given_wrong': float(gain.sum() / (mask & ~old).sum()) if (mask & ~old).any() else None,
        'loss_rate_given_correct': float(loss.sum() / (mask & old).sum()) if (mask & old).any() else None,
        'fixed_pool_eligible': int((mask & eligible).sum()),
        'fixed_pool_lost': int((mask & old & ~rows['fixed_correct'] & eligible).sum()),
        'fixed_pool_gained': int((mask & ~old & rows['fixed_correct'] & eligible).sum())}
    for label, chosen in (('all', mask & eligible), ('gained', gain & eligible), ('lost', loss & eligible)):
        data[label] = {}
        for name in ('key_margin_delta', 'context_contribution_delta', 'raw_contribution_delta',
                     'key_width_delta', 'raw_cos_margin', 'neighbor_raw_margin', 'smoothness',
                     'step_m', 'initial_margin'):
            values = rows[name][chosen]
            data[label][name] = {'n': len(values), 'mean': float(values.mean()) if len(values) else None,
                'median': float(np.median(values)) if len(values) else None}
    return data


def main() -> None:
    """기존 두 모델의 모든 양성 질의를 진단하고 학습에서 정한 구간으로 집계한다."""
    spec = prior.protocol(True)
    epoch = prior.base.read(prior.OUT / 'selection.json')['selected_epoch'][ARM]
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'summary.json').exists(), '완료한 진단을 덮어쓰지 않는다'
    sources = {**spec['runtime'], **spec['inputs']}
    for p in (Path(__file__), REPO / 'scripts/_pdv2_pair_evaluate.py',
              REPO / 'scripts/_pdv2_context_diagnosis.py', prior.OUT / 'selection.json'):
        sources[str(p.relative_to(REPO))] = sha256(p)
    write_json(OUT / 'protocol.json', {'sources': sources, 'arm': ARM, 'epoch': epoch,
        'scope': '사후 진단. 모든 양성을 분모로 사용한다. 정답/오답 쌍은 시작 모델의 고정 후보에서 '
                 '융합 점수가 가장 큰 정답/오답이다. 방향은 모든 과거5m정답으로 기존 규칙을 쓴다. '
                 '반복성은 원 특징 코사인과±5정/역대응으로 설명하며 물리적 장면 동일성을 뜻하지 않는다.',
        'bins': '원 특징 정답-오답 차이 ≤0, (0,.01], (.01,.05], >.05; '
                '주변 유사도는 학습 양성의 사분위 경계로만 구간을 정한다.',
        'new_updates': 0, 'test_used': False})
    combined, per_sequence = {}, {}
    maximum_decomposition_error = 0.
    fixed_replay_checks = 0
    cases = prior.base.read(prior.OUT / 'case_diagnosis.json')['cases']
    for role in ('train', 'val'):
        collection, per_sequence[role] = [], {}
        for sequence in spec[role]:
            name = sequence.replace('/', '_') + '.npz'
            paths = prior.factor.paths_for(role, sequence)
            cache = prior.base.exp.arrays(paths['cache'])
            positions = cache['poses' if role == 'train' else 'poses_gt'][:, :3, 3].astype(np.float64)
            old_path = prior.factor.OUT / role / sequence.replace('/', '_') / 'new.npz'
            new_path = prior.OUT / ARM / f'epoch{epoch}' / role / name
            key_paths = [prior.factor.key_path(role, 'interaction', sequence), new_path.with_name('keys_' + name)]
            old, new = [prior.base.exp.arrays(p) for p in (old_path, new_path)]
            keys = [prior.base.exp.arrays(p)['embedding'] for p in key_paths]
            for p in (old_path, new_path, *key_paths, paths['cache'], paths['spec']):
                sources[str(p.relative_to(REPO))] = sha256(p)
            raw = prior.base.normalize(keys[0][:, :288])
            np.testing.assert_allclose(raw, prior.base.normalize(keys[1][:, :288]), atol=4e-6)
            all_min = np.full(len(raw), np.nan)
            all_min[old['query_idx']] = old['min_past_gt_dist']
            group = direction_groups(positions, all_min)
            positive = old['min_past_gt_dist'] <= 5
            q = old['query_idx'][positive]
            n = len(q)
            row = {'query': q, 'group': group[q], 'old_correct': old['top1_geo_dist'][positive] <= 5,
                'new_correct': new['top1_geo_dist'][positive] <= 5,
                'eligible': np.zeros(n, bool), 'fixed_correct': np.zeros(n, bool),
                'old_width_correct': np.zeros(n, bool)}
            for field in ('key_margin_delta', 'context_contribution_delta', 'raw_contribution_delta',
                          'key_width_delta', 'raw_cos_margin', 'neighbor_raw_margin', 'smoothness',
                          'step_m', 'initial_margin', 'width_only_pair_delta', 'current_margin'):
                row[field] = np.full(n, np.nan)
            for i, t in enumerate(q):
                neighbors = t + np.r_[np.arange(-5, 0), np.arange(1, 6)]
                row['smoothness'][i] = np.mean(raw[neighbors] @ raw[t])
                row['step_m'][i] = np.linalg.norm(np.diff(cache['poses'][t-5:t+6, :3, 3], axis=0), axis=1).mean()
            with np.load(paths['spec']) as z:
                bank = prior.base.exp.old.base.bank_for(prior.base.exp.OP, z)
            weights = tuple(spec['confidence']['1']['tuple'])
            training_pool = prior.base.exp.arrays(prior.pool_path(sequence) / 'pool.npz') if role == 'train' else None
            case_map = {c['q']: c for c in cases[role] if c['sequence'] == sequence}
            for i, t0 in enumerate(q):
                t = int(t0)
                if role == 'train':
                    ix = np.searchsorted(training_pool['queries'], t)
                    if ix == len(training_pool['queries']) or training_pool['queries'][ix] != t:
                        continue
                    valid = training_pool['valid'][ix]
                    ids = training_pool['candidates'][ix, valid]
                    static = training_pool['static'][ix, valid]
                else:
                    ids = np.unique(np.r_[prior.base.prefix_topk(keys[0], t, t-29, 100),
                                          prior.base.prefix_topk(bank[0], t, t-29, 100)])
                    static = np.column_stack((bank[0][ids] @ bank[0][t],
                        prior.base.align_scores(bank[1], bank[2], t, ids),
                        prior.base.column_align_scores(bank[3], t, ids)))
                labels = np.linalg.norm(positions[ids] - positions[t], axis=1) <= 5
                if not labels.any() or labels.all():
                    continue
                values = [np.column_stack((k[ids] @ k[t], static)) for k in keys]
                fused = [fuse_with_range(v, weights) for v in values]
                scores, ranges = zip(*fused)
                assert ids[np.argmax(scores[0])] == old['j_star'][np.searchsorted(old['query_idx'], t)]
                pos = int(np.argmax(np.where(labels, scores[0], -np.inf)))
                neg = int(np.argmax(np.where(~labels, scores[0], -np.inf)))
                raw_parts = [float(k[t, :288] @ (k[ids[pos], :288] - k[ids[neg], :288])) for k in keys]
                ctx_parts = [float(k[t, 288:] @ (k[ids[pos], 288:] - k[ids[neg], 288:])) for k in keys]
                margins = [float(v[pos, 0] - v[neg, 0]) for v in values]
                error = max(abs(m - a - b) for m, a, b in zip(margins, raw_parts, ctx_parts))
                maximum_decomposition_error = max(maximum_decomposition_error, error)
                row['eligible'][i] = True
                row['fixed_correct'][i] = labels[np.argmax(scores[1])]
                row['initial_margin'][i] = scores[0][labels].max() - scores[0][~labels].max()
                row['current_margin'][i] = scores[1][labels].max() - scores[1][~labels].max()
                row['key_margin_delta'][i] = margins[1] - margins[0]
                row['raw_contribution_delta'][i] = raw_parts[1] - raw_parts[0]
                row['context_contribution_delta'][i] = ctx_parts[1] - ctx_parts[0]
                row['key_width_delta'][i] = ranges[1][0, 2] - ranges[0][0, 2]
                row['width_only_pair_delta'][i] = weights[0] * margins[1] * (1/ranges[1][0, 2] - 1/ranges[0][0, 2])
                row['raw_cos_margin'][i] = raw[t] @ (raw[ids[pos]] - raw[ids[neg]])
                row['neighbor_raw_margin'][i] = neighborhood(raw, t, int(ids[pos])) - neighborhood(raw, t, int(ids[neg]))
                width_score = scores[1] + weights[0] * (values[1][:, 0] - values[1][:, 0].min()) * (1/ranges[0][0, 2] - 1/ranges[1][0, 2])
                row['old_width_correct'][i] = labels[np.argmax(width_score)]
                if t in case_map:
                    assert int(ids[np.argmax(scores[1])]) == case_map[t]['winner']['fixed_old_pool']
                    fixed_replay_checks += 1
            assert np.isfinite(row['smoothness']).all() and np.isfinite(row['step_m']).all()
            for field, value in row.items():
                assert value.shape == (n,)
                if value.dtype.kind == 'f':
                    assert np.isfinite(value[row['eligible']]).all(), field
            row['sequence_id'] = np.full(n, spec[role].index(sequence), dtype=np.int16)
            collection.append(row)
            per_sequence[role][sequence] = aggregate(row, np.ones(n, bool))
            dest = OUT / role / name
            dest.parent.mkdir(exist_ok=True)
            np.savez_compressed(dest, **row)
            sources[str(dest.relative_to(REPO))] = sha256(dest)
            print('TRANSFER_DONE', role, sequence, n, flush=True)
        combined[role] = {k: np.concatenate([r[k] for r in collection]) for k in collection[0]}
    bounds = np.quantile(combined['train']['smoothness'], [.25, .5, .75])
    result = {'per_sequence': per_sequence, 'roles': {}, 'smoothness_train_quartiles': bounds.tolist()}
    for role, row in combined.items():
        eligible = row['eligible']
        bins = np.digitize(row['raw_cos_margin'], [0., .01, .05], right=True)
        smooth_bins = np.digitize(row['smoothness'], bounds, right=True)
        losses = row['old_correct'] & ~row['new_correct'] & eligible
        result['roles'][role] = {'all': aggregate(row, np.ones(len(row['query']), bool)),
            'directions': {str(g): aggregate(row, row['group'] == g) for g in range(3)},
            'raw_margin_bins': {str(g): aggregate(row, eligible & (bins == g)) for g in range(4)},
            'smoothness_bins': {str(g): aggregate(row, smooth_bins == g) for g in range(4)},
            'changed_losses': {'n': int(losses.sum()),
                'negative_context_delta': int((losses & (row['context_contribution_delta'] < 0)).sum()),
                'negative_raw_margin_delta': int((losses & (row['key_margin_delta'] < 0)).sum()),
                'width_hurts_pair': int((losses & (row['width_only_pair_delta'] < 0)).sum()),
                'rescued_by_old_width_on_old_pool': int((losses & ~row['fixed_correct'] & row['old_width_correct']).sum())}}
    result.update({'test_used': False, 'posthoc': True, 'new_updates': 0,
        'interpretation': '집단별 건수와 비율은 관찰 연관이다. 장소·입력·계열이 함께 다르므로 '
                          '단독 원인이나 효과크기로 해석하지 않는다. 검증 라벨은 학습에 쓰지 않는다.'})
    write_json(OUT / 'summary.json', result)
    assert maximum_decomposition_error < 2e-6
    expected_replays = sum(int((r['eligible'] & (r['old_correct'] != r['new_correct'])).sum())
                           for r in combined.values())
    assert fixed_replay_checks == expected_replays
    for p in (OUT / 'protocol.json', OUT / 'summary.json', prior.OUT / 'case_diagnosis.json'):
        sources[str(p.relative_to(REPO))] = sha256(p)
    for path, digest in sources.items():
        assert sha256(REPO / path) == digest, path
    write_json(OUT / 'completion_validation.json', {'passed': True, 'sources': sources,
        'fixed_replay_checks': fixed_replay_checks,
        'changed_cases_without_initial_positive_candidate': sum(len(v) for v in cases.values()) - expected_replays,
        'maximum_decomposition_error': maximum_decomposition_error,
        'positive_queries': {k: len(v['query']) for k, v in combined.items()}, 'test_used': False})
    print('TRANSFER_COMPLETE', {k: v['changed_losses'] for k, v in result['roles'].items()}, flush=True)


if __name__ == '__main__':
    main()
