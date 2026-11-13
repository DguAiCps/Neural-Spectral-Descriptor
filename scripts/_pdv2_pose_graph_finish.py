#!/usr/bin/env python3
"""포즈 그래프 진단을 검산하고 계열 요약과 비교 그림을 저장한다."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
from _pdv2_no_context_ablation import sha256, write_json

OUT = REPO / 'results/pdv2_pose_graph_audit_20261023'


def main() -> None:
    """모든 입력 해시·집계 개수를 확인하고 해석용 요약을 만든다."""
    data = json.loads((OUT / 'diagnosis.json').read_text())
    sources = dict(data['sources'])
    sources[str(Path(__file__).relative_to(REPO))] = sha256(Path(__file__))
    for p in (OUT / 'protocol.json', OUT / 'diagnosis.json'):
        sources[str(p.relative_to(REPO))] = sha256(p)
    for path, digest in sources.items():
        assert sha256(REPO / path) == digest, path
    assert data['train_graph_exact_edges'] and data['train_graph_attr_max_error'] == 0.
    summary, rows = {}, []
    for split, sequences in data['graphs'].items():
        summary[split] = {}
        for family in ('kitti', 'helipr', 'mulran'):
            selected = [v for k, v in sequences.items() if k.startswith(family)]
            if not selected:
                continue
            value = {'sequences': len(selected),
                'nodes': sum(v['nodes'] for v in selected), 'edges': sum(v['edges'] for v in selected),
                'mean_sequence_window_median_m': float(np.mean([v['full_window_path_m']['p50'] for v in selected])),
                'mean_sequence_symmetric_pair_fraction': float(np.mean([v['past_future_similar_pair_fraction'] for v in selected])),
                'mean_sequence_rotation_below_1deg_fraction': float(np.mean([v['edge_rotation_below_1deg_fraction'] for v in selected]))}
            summary[split][family] = value
            rows.append({'split': split, 'family': family, **value})
    with (OUT / 'family_geometry.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {split: {key: sum(v[key] for v in records.values()) for key in ('nodes', 'edges')}
              for split, records in data['graphs'].items()}
    assert counts == {'train': {'nodes': 72306, 'edges': 722820},
                      'val': {'nodes': 99641, 'edges': 996140}}
    errors = [layer['finite_difference_max_error'] for model in data['branch_responses'].values()
              for seq in model['sequences'].values() for layer in seq['layers'].values()]
    assert len(errors) == 102 and max(errors) < 1e-6
    sensitivity = {}
    for arm, model in data['branch_responses'].items():
        sensitivity[arm] = {}
        for layer in ('0', '1'):
            values = [v['layers'][layer] for k, v in model['sequences'].items() if k.startswith('val/')]
            sensitivity[arm][layer] = {
                'centered_bias_rms': float(np.mean([v['centered_bias_rms'] for v in values])),
                'derivative_per_m_and_degree_rms': np.mean([v['derivative_physical_rms_per_m_and_degree'] for v in values], axis=0).tolist(),
                'geometry_only_offset_mass': np.mean([v['geometry_only_offset_mass'] for v in values], axis=0).tolist()}
    result = {'family_sequence_means': summary, 'validation_sequence_mean_sensitivity': sensitivity,
              'counts': counts, 'branch_layer_records': len(errors),
              'finite_difference_max_error': max(errors), 'test_used': False, 'training_performed': False,
              'scope': '기하 요약은 시퀀스별 통계의 평균이다. AP나 장소 분류 정확도가 아니다.'}
    write_json(OUT / 'summary.json', result)
    # 관측된 검증 GT/KISS 회전 차이의 최댓값은 첫 기준 자세 주변에 국한되는지 확인한다.
    import _pdv2_attention_review as run
    from _pdv2_input_alignment_lib import temporal_edges, pose_attributes
    from _pdv2_pose_graph_lib import ordered_windows
    _, _, paths, _ = run.base.exp.old.base.inputs('val', 'kitti360/0004')
    cache = run.base.exp.arrays(paths['cache'])
    edges = temporal_edges(np.zeros(len(cache['poses']), np.int64))
    odom, gt = [pose_attributes(cache[key], edges) for key in ('poses', 'poses_gt')]
    rotation_error = abs(odom[:, 1] - gt[:, 1]) * 180.
    large = np.flatnonzero(rotation_error > 20)
    assert len(large) == 10 and np.all(np.any(edges[:, large] == 0, axis=0))
    raw_pose_path = Path('/workspace/data/kitti360/2013_05_28_drive_0004_sync/poses.txt')
    with raw_pose_path.open() as handle:
        first = handle.readline().split()
    first_pose = np.asarray(first[1:], dtype=np.float64).reshape(3, 4)
    np.testing.assert_array_equal(first_pose[:, :3], cache['poses_gt'][0, :3, :3])
    anomaly = {'sequence': 'kitti360/0004', 'gt_kiss_rotation_difference_max_deg': float(rotation_error.max()),
        'edges_above_20deg': len(large), 'all_incident_to_node_zero': True,
        'gt_first_rotation_matches_source': True, 'source_pose_file': str(raw_pose_path),
        'source_pose_sha256': sha256(raw_pose_path),
        'interpretation': '시작 노드에 걸린 회전 차이이며 원자료 첫 자세와 캐시가 같다. 현재 검증 간선에는 KISS 자세를 쓰고 KITTI360은 학습에 없으므로 학습 기하 편향의 약함을 설명하지 못한다. GT의 물리적 정확성은 확정하지 않는다.'}
    write_json(OUT / 'startup_pose_diagnosis.json', anomaly)
    # 306 m 떨어진 실제 오답 두 창을 같은 시간 오프셋 축으로 비교한다.
    _, _, paths, _ = run.base.exp.old.base.inputs('val', 'mulran/Sejong02')
    cache = run.base.exp.arrays(paths['cache'])
    _, distance, angle = ordered_windows(cache['poses'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    offsets = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), layout='constrained')
    for node, label in ((12374, 'Query'), (12136, 'False winner')):
        axes[0].plot(offsets, distance[node - 5], 'o-', label=label, markersize=3)
        axes[1].plot(offsets, np.rad2deg(angle[node - 5]), 'o-', label=label, markersize=3)
    axes[0].set_title('Different places: 306.1 m apart')
    axes[0].set_ylabel('Neighbor distance (m)')
    axes[1].set_title('Unsigned local rotation')
    axes[1].set_ylabel('Rotation magnitude (degrees)')
    for arm, label in (('diff_linear', 'Diff / linear V'), ('diff_tanh', 'Diff / tanh V'),
                       ('standard_linear', 'Standard / linear V')):
        mass = sensitivity[arm]['0']['geometry_only_offset_mass']
        axes[2].plot(offsets, np.array(mass) * 100, 'o-', label=label, markersize=3)
    axes[2].set_title('Geometry-only weights, layer 1')
    axes[2].set_ylabel('Neighbor weight (%)')
    for ax in axes:
        ax.set_xlabel('Neighbor keyframe offset')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Pose graph diagnosis | fixed checkpoints | validation only')
    fig.savefig(OUT / 'pose_graph_diagnosis.png', dpi=180)
    fig.savefig(OUT / 'pose_graph_diagnosis.pdf')
    plt.close(fig)
    for p in (OUT / 'summary.json', OUT / 'family_geometry.csv', OUT / 'startup_pose_diagnosis.json',
              OUT / 'pose_graph_diagnosis.png', OUT / 'pose_graph_diagnosis.pdf'):
        sources[str(p.relative_to(REPO))] = sha256(p)
    for path, digest in sources.items():
        assert sha256(REPO / path) == digest
    write_json(OUT / 'completion_validation.json', {'passed': True, 'sources': sources,
        'counts': counts, 'branch_layer_records': len(errors), 'finite_difference_max_error': max(errors),
        'external_source': {str(raw_pose_path): sha256(raw_pose_path)},
        'test_used': False, 'training_performed': False})
    print('POSE_AUDIT_VALIDATED', counts, 'finite_difference', max(errors), flush=True)


if __name__ == '__main__':
    main()
