#!/usr/bin/env python3
"""반대 방향 학습 관측을 무방향 쌍 열거와 직접 거리 계산으로 교차 검산한다."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_temporal_augmentation as run
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """질의별 검색 대신 모든 무방향 근접 쌍을 열거해 집계를 재현한다."""
    spec = run.protocol(True)
    folder = run.OUT / 'training_support'
    reference = run.base.read(folder / 'summary.json')
    arrays = []
    lengths = []
    for s, sequence in enumerate(spec['train']):
        pose = run.base.exp.arrays(run.factor.paths_for('train', sequence)['cache'])['poses']
        p = pose[:, :3, 3].astype(np.float64)
        index = np.arange(len(p))
        displacement = p-p[np.maximum(0, index-5)]
        norm = np.linalg.norm(displacement, axis=1)
        arrays.append({'p': p, 'h': displacement/np.maximum(norm[:, None], 1e-12),
            'valid': (index >= 5) & (norm >= 1.), 'i': index, 's': np.full(len(p), s),
            'c': np.full(len(p), 100 if sequence.startswith('helipr/') else s),
            'q': (index >= 30) & (index < len(p)-15)})
        lengths.append(len(p))
    a = {k: np.concatenate([v[k] for v in arrays]) for k in arrays[0]}
    pairs = []
    brute_checks = 0
    for coordinate in np.unique(a['c']):
        nodes = np.flatnonzero(a['c'] == coordinate)
        tree = cKDTree(a['p'][nodes])
        local = tree.query_pairs(5., output_type='ndarray')
        pairs.append(nodes[local])
        for index in np.linspace(0, len(nodes)-1, 12, dtype=int):
            direct = nodes[np.linalg.norm(a['p'][nodes]-a['p'][nodes[index]], axis=1) <= 5.]
            got = nodes[np.asarray(tree.query_ball_point(a['p'][nodes[index]], 5.), dtype=np.int64)]
            np.testing.assert_array_equal(np.sort(direct), np.sort(got))
            brute_checks += 1
    undirected = np.concatenate(pairs)
    directed = np.concatenate((undirected, undirected[:, ::-1]))
    directed = directed[a['q'][directed[:, 0]]]
    q, j = directed.T
    same_session = a['s'][q] == a['s'][j]
    same_time = np.abs(a['i'][q]-a['i'][j]) >= 30
    modes = {'same_session_past': same_session & (a['i'][j] <= a['i'][q]-30),
        'same_session_both_time': same_session & same_time,
        'shared_coordinate_all_sessions': ~same_session | same_time}
    dot = np.sum(a['h'][q]*a['h'][j], axis=1)
    valid = a['valid'][q] & a['valid'][j]
    opposite = valid & (dot <= -.5)
    same = valid & (dot >= .5)
    result = {}
    for name, mask in modes.items():
        n = np.bincount(q[mask], minlength=len(a['p']))
        no = np.bincount(q[mask & opposite], minlength=len(a['p']))
        ns = np.bincount(q[mask & same], minlength=len(a['p']))
        row = {'queries': int(a['q'].sum()), 'positive': int((n > 0).sum()),
            'same_available': int((ns > 0).sum()), 'opposite_available': int((no > 0).sum()),
            'both_available': int(((no > 0) & (ns > 0)).sum()),
            'opposite_only': int(((n > 0) & (no == n)).sum()),
            'positive_pairs': int(n.sum()), 'opposite_pairs': int(no.sum())}
        assert row == reference['aggregate'][name], (name, row)
        result[name] = row
    chosen = modes['shared_coordinate_all_sessions'] & opposite
    qp, jp = q[chosen], j[chosen]
    distances = np.linalg.norm(a['p'][qp]-a['p'][jp], axis=1)
    assert len(qp) == 8590 and len(np.unique(qp)) == 1477 and np.all(distances <= 5.+1e-10)
    np.savez_compressed(folder / 'available_opposite_pairs.npz', anchor_global=qp, positive_global=jp,
        anchor_sequence=a['s'][qp], positive_sequence=a['s'][jp], anchor_index=a['i'][qp],
        positive_index=a['i'][jp], direction_cosine=dot[chosen], distance=distances,
        sequence_lengths=np.asarray(lengths), sequence_names=np.asarray(spec['train']))
    sources = dict(run.base.read(folder / 'completion_validation.json')['sources'])
    for p in (Path(__file__), folder / 'available_opposite_pairs.npz'):
        sources[str(p.relative_to(REPO))] = sha256(p)
    for path, value in sources.items():
        assert sha256(REPO / path) == value, path
    write_json(folder / 'independent_validation.json', {'passed': True, 'sources': sources,
        'aggregate': result, 'brute_distance_checks': brute_checks,
        'directed_opposite_pairs': len(qp), 'opposite_anchors': len(np.unique(qp)),
        'new_training_updates': 0, 'test_used': False})
    print('TRAIN_SUPPORT_INDEPENDENT_CHECK_PASSED', brute_checks, len(qp), len(np.unique(qp)), flush=True)


if __name__ == '__main__':
    main()
