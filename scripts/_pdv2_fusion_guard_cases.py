#!/usr/bin/env python3
"""두 제약을 함께 학습한 모델의 모든 정답 손익을 고정 후보와 재검색으로 나눈다."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_fusion_guard as run
from _pdv2_fusion_factor_lib import fuse_with_range
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """학습·검증에서 정오가 바뀐 질의를 빠짐없이 기록한다."""
    spec = run.protocol(True)
    summary = run.base.read(run.OUT / 'summary.json')
    arm = 'rank_filtered_guarded'
    epoch = summary['selected_epoch'][arm]
    sources = {str(Path(__file__).relative_to(REPO)): sha256(Path(__file__)),
        str((run.OUT / 'summary.json').relative_to(REPO)): sha256(run.OUT / 'summary.json')}
    records, aggregates = {}, {}
    for role in ('train', 'val'):
        records[role] = []
        for sequence in spec[role]:
            folder = run.OUT / arm / f'epoch{epoch}' / role
            path = folder / (sequence.replace('/', '_') + '.npz')
            new = run.base.exp.arrays(path)
            old_path = run.factor.OUT / role / sequence.replace('/', '_') / 'new.npz'
            old = run.base.exp.arrays(old_path)
            old_key_path = run.factor.key_path(role, 'interaction', sequence)
            new_key_path = path.with_name('keys_' + path.name)
            k0, k1 = [run.base.exp.arrays(p)['embedding'] for p in (old_key_path, new_key_path)]
            paths = run.factor.paths_for(role, sequence)
            cache = run.base.exp.arrays(paths['cache'])
            position = cache['poses' if role == 'train' else 'poses_gt'][:, :3, 3].astype(np.float64)
            with np.load(paths['spec']) as z:
                bank = run.base.exp.old.base.bank_for(run.base.exp.OP, z)
            positive = old['min_past_gt_dist'] <= 5
            old_correct, new_correct = old['top1_geo_dist'] <= 5, new['top1_geo_dist'] <= 5
            chosen = np.flatnonzero(positive & (old_correct != new_correct))
            for i in chosen:
                t = int(old['query_idx'][i])
                aux = run.base.prefix_topk(bank[0], t, t - 29, 100)
                pools = [np.unique(np.r_[run.base.prefix_topk(k, t, t - 29, 100), aux]) for k in (k0, k1)]
                values, correct, margins, winners = {}, {}, {}, {}
                for label, key, ids in (('old', k0, pools[0]), ('new', k1, pools[1]),
                                         ('fixed_old_pool', k1, pools[0])):
                    channels = np.column_stack((key[ids] @ key[t], bank[0][ids] @ bank[0][t],
                        run.base.align_scores(bank[1], bank[2], t, ids),
                        run.base.column_align_scores(bank[3], t, ids)))
                    score, _ = fuse_with_range(channels, tuple(spec['confidence']['1']['tuple']))
                    winner = int(ids[np.argmax(score)])
                    labels = np.linalg.norm(position[ids] - position[t], axis=1) <= 5
                    winners[label] = winner
                    correct[label] = bool(labels[np.argmax(score)])
                    margins[label] = (float(score[labels].max() - score[~labels].max())
                                     if labels.any() and (~labels).any() else None)
                    values[label] = int(labels.sum())
                assert winners['old'] == old['j_star'][i] and winners['new'] == new['j_star'][i]
                true = int(old['j_star'][i] if old_correct[i] else new['j_star'][i])
                false = int(new['j_star'][i] if old_correct[i] else old['j_star'][i])
                records[role].append({'sequence': sequence, 'q': t,
                    'change': 'loss' if old_correct[i] else 'gain', 'true': true, 'false': false,
                    'false_distance': float(np.linalg.norm(position[t] - position[false])),
                    'old_true_in_new_pool': bool(true in pools[1]),
                    'false_in_old_pool': bool(false in pools[0]), 'winner': winners,
                    'correct': correct, 'positive_candidates': values, 'fused_margin': margins})
            for p in (path, old_path, old_key_path, new_key_path, paths['cache'], paths['spec']):
                sources[str(p.relative_to(REPO))] = sha256(p)
        aggregates[role] = {}
        for kind in ('loss', 'gain'):
            cases = [c for c in records[role] if c['change'] == kind]
            aggregates[role][kind] = {'count': len(cases),
                'correct_on_fixed_old_pool': sum(c['correct']['fixed_old_pool'] for c in cases),
                'no_correct_in_new_pool': sum(c['positive_candidates']['new'] == 0 for c in cases),
                'false_farther_than_25m': sum(c['false_distance'] > 25 for c in cases)}
            expected = summary['candidate_transitions'][arm][role][str(epoch)]['lost' if kind == 'loss' else 'gained']
            assert len(cases) == expected
    path = run.OUT / 'case_diagnosis.json'
    write_json(path, {'arm': arm, 'epoch': epoch, 'aggregates': aggregates, 'cases': records,
        'scope': '선택한 두 제약 결합 모델의 모든 정답 손익에 대한 사후 원인 진단이다. '
                 '고정 후보에서의 정오이며 이 개입의 전체 최종 AP는 계산하지 않았다.',
        'posthoc': True, 'test_used': False, 'new_training_updates': 0})
    sources[str(path.relative_to(REPO))] = sha256(path)
    write_json(run.OUT / 'case_validation.json', {'passed': True, 'sources': sources,
        'case_count': sum(len(c) for c in records.values()), 'test_used': False})
    print('GUARD_CASES_COMPLETE', aggregates, flush=True)


if __name__ == '__main__':
    main()
