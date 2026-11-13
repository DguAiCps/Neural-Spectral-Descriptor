#!/usr/bin/env python3
"""학습·검증 시간 포즈 그래프의 생성, 정보 손실, 학습된 편향을 진단한다."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_attention_review as previous
from _pdv2_input_alignment_lib import temporal_edges, pose_attributes
from _pdv2_pose_graph_lib import relative_geometry, quantiles, ordered_windows, window_difference
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_matching_mine import cache_path

OUT = REPO / 'results/pdv2_pose_graph_audit_20261023'
MODELS = ('diff_linear', 'standard_linear', 'diff_tanh')


def audit_sequence(poses: np.ndarray, frame_ids: np.ndarray) -> tuple:
    """전체 간선을 독립 상대 변환과 대조하고 창의 공간 크기를 집계한다."""
    n = len(poses)
    assert poses.shape == (n, 4, 4) and np.isfinite(poses).all()
    assert np.all(np.diff(frame_ids) > 0)
    np.testing.assert_allclose(poses[:, 3], np.tile([0., 0., 0., 1.], (n, 1)), atol=1e-12)
    edges = temporal_edges(np.zeros(n, dtype=np.int64))
    attr = pose_attributes(poses, edges)
    assert len(attr) == 10 * n - 30
    assert len(np.unique(edges[0] * n + edges[1])) == len(attr)
    np.testing.assert_array_equal(attr, pose_attributes(poses, edges[::-1]))
    xyz, angle = relative_geometry(poses, edges)
    independent = np.column_stack((np.log1p(np.linalg.norm(xyz, axis=1)) / 5., angle / np.pi))
    rotation = poses[:, :3, :3]
    orthogonal = rotation.transpose(0, 2, 1) @ rotation - np.eye(3)
    centers, distance, rotation_angle = ordered_windows(poses)
    path = np.r_[0., np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))]
    steps = np.diff(path)
    # 앞뒤 이웃의 같은 |시차|끼리 비교한다. 두 기준은 해석용이며 모델 선택에 쓰지 않는다.
    past, future = distance[:, :5][:, ::-1], distance[:, 5:]
    rel_gap = abs(past - future) / np.maximum((past + future) / 2., 1e-12)
    turn_gap = np.rad2deg(abs(rotation_angle[:, :5][:, ::-1] - rotation_angle[:, 5:]))
    unsigned_turn = np.rad2deg(rotation_angle)
    degree = np.bincount(edges[1], minlength=n)
    record = {
        'nodes': n, 'edges': len(attr), 'full_degree_nodes': int((degree == 10).sum()),
        'scan_ids_increasing': True, 'unique_edges': True, 'reverse_attributes_exact': True,
        'rotation_orthogonality_max': float(abs(orthogonal).max()),
        'rotation_determinant_error_max': float(abs(np.linalg.det(rotation) - 1).max()),
        'independent_normalized_distance_error': quantiles(abs(independent[:, 0] - attr[:, 0])),
        'independent_rotation_error_deg': quantiles(np.rad2deg(abs(independent[:, 1] - attr[:, 1]) * np.pi)),
        'adjacent_distance_m': quantiles(steps),
        'adjacent_stationary_fraction_005m': float((steps < .05).mean()),
        'edge_distance_m': quantiles(distance), 'edge_rotation_deg': quantiles(unsigned_turn),
        'full_window_path_m': quantiles(path[centers + 5] - path[centers - 5]),
        'full_window_chord_to_path': quantiles(np.linalg.norm(
            poses[centers + 5, :3, 3] - poses[centers - 5, :3, 3], axis=1) /
            np.maximum(path[centers + 5] - path[centers - 5], 1e-12)),
        'past_future_distance_relative_gap': quantiles(rel_gap),
        'past_future_rotation_gap_deg': quantiles(turn_gap),
        'past_future_similar_pair_fraction': float(((rel_gap < .1) & (turn_gap < 1.)).mean()),
        'edge_rotation_below_1deg_fraction': float((unsigned_turn < 1).mean()),
    }
    sample = centers[np.linspace(0, len(centers) - 1, min(512, len(centers))).astype(int)]
    offsets = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
    selected_edges = np.stack(((sample[:, None] + offsets).reshape(-1), np.repeat(sample, 10)))
    return record, pose_attributes(poses, selected_edges), (centers, distance, rotation_angle)


def error_geometry(poses: np.ndarray, windows: tuple, row: dict) -> dict:
    """동결 원장의 정답·오답에서 관측되는 창의 기하 유사도를 기술한다."""
    _, distance, angle = windows
    q, j = row['query_idx'], row['j_star']
    usable = (j >= 5) & (j < len(poses) - 5) & (q >= 5) & (q < len(poses) - 5)
    positive = row['min_past_gt_dist'] <= 5
    valid = positive | (row['min_past_gt_dist'] > 25)
    correct = positive & (row['top1_geo_dist'] <= 5)
    false = valid & ~correct
    result = {}
    groups = {'correct': usable & correct, 'false': usable & false,
              'positive_but_wrong': usable & positive & ~correct}
    for name, mask in groups.items():
        value = {'queries': int(mask.sum())}
        if mask.any():
            difference = window_difference(distance, angle, q[mask], j[mask])
            value.update({k: quantiles(v) for k, v in difference.items()})
            value['geometry_similar_fraction'] = float(((difference['relative_distance_rms'] < .1) &
                (difference['rotation_rms_deg'] < 1.)).mean())
        result[name] = value
    highest = np.flatnonzero(false & (row['arc_unique_b100'] == row['arc_unique_b100'][false].max())) if false.any() else []
    result['highest_false'] = []
    for i in highest:
        item = {'query': int(q[i]), 'winner': int(j[i]), 'distance_m': float(row['top1_geo_dist'][i]),
                'score': float(row['arc_unique_b100'][i]), 'full_window': bool(usable[i])}
        if usable[i]:
            item.update({k: float(v[0]) for k, v in window_difference(
                distance, angle, q[i:i+1], j[i:i+1]).items()})
        result['highest_false'].append(item)
    return result


def branch_responses(raw: dict, sources: dict) -> dict:
    """고정 가중치에서 두 물리 입력의 편향 미분과 거리별 가중치를 잰다."""
    output = {}
    for arm in MODELS:
        saved = previous.read(previous.source.OUT / 'summary.json')['per_seed']['1'][arm]['selected']
        checkpoint = previous.source.OUT / 's1' / arm / 'checkpoints' / f"checkpoint_epoch_{saved['epoch']}.pth"
        sources[str(checkpoint.relative_to(REPO))] = sha256(checkpoint)
        net = previous.source.model(arm).cuda().double().eval()
        net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state_dict'])
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        output[arm] = {'epoch': saved['epoch'], 'sequences': {}}
        for seq, attributes in raw.items():
            attr = torch.tensor(attributes, device='cuda', dtype=torch.float64, requires_grad=True)
            types = torch.zeros(len(attr), dtype=torch.long, device='cuda')
            encoded = net.gnn.edge_encoder(attr, types)
            layers = {}
            for index, conv in enumerate(net.gnn.convs):
                bias = conv.bias_mlp(encoded)
                jac = torch.stack([torch.autograd.grad(bias[:, h].sum(), attr,
                    retain_graph=True)[0][:, :2] for h in range(4)], dim=1)
                # log1p(d)/5와 theta/pi의 연쇄 법칙으로 미터·도 단위 미분을 구한다.
                meters = torch.expm1(attr[:, 0].detach() * 5)
                physical = jac.detach().clone()
                physical[:, :, 0] /= (5 * (1 + meters[:, None]))
                physical[:, :, 1] /= 180.
                shaped = bias.detach().reshape(-1, 10, 4)
                centered = shaped - shaped.mean(1, keepdim=True)
                geom_attention = shaped.softmax(1)
                layers[str(index)] = {
                    'centered_bias_rms': float(centered.square().mean().sqrt()),
                    'derivative_normalized_rms_by_input': jac.square().mean((0, 1)).sqrt().cpu().tolist(),
                    'derivative_physical_rms_per_m_and_degree': physical.square().mean((0, 1)).sqrt().cpu().tolist(),
                    'derivative_physical_mean_per_head': physical.mean(0).cpu().tolist(),
                    'geometry_only_offset_mass': geom_attention.mean((0, 2)).cpu().tolist(),
                }
                # 중앙 차분을 다른 연산 경로로 계산하여 미분 계수의 의미를 검산한다.
                check = attr[:20].detach()
                finite_errors = []
                for field in (0, 1):
                    delta = torch.zeros_like(check)
                    delta[:, field] = 1e-6
                    plus = conv.bias_mlp(net.gnn.edge_encoder(check + delta, types[:20]))
                    minus = conv.bias_mlp(net.gnn.edge_encoder(check - delta, types[:20]))
                    error = float(abs((plus - minus) / 2e-6 - jac[:20, :, field]).max())
                    assert error < 1e-6, (arm, seq, index, field, error)
                    finite_errors.append(error)
                layers[str(index)]['finite_difference_max_error'] = max(finite_errors)
            rot = net.gnn.edge_encoder.rot_encoder(attr[:, 1]).detach().reshape(-1, 10, 16)
            rot = (rot - rot.mean(1, keepdim=True)).reshape(-1, 16)
            eigen = torch.linalg.eigvalsh(rot.T @ rot).clamp_min(0)
            output[arm]['sequences'][seq] = {'layers': layers,
                'rotation_encoding_first_component_energy': float(eigen[-1] / eigen.sum())}
        print('BRANCH_AUDITED', arm, flush=True)
    return output


def main() -> None:
    """시험 자료에 접근하지 않고 기존 그래프·선택 모델만 관측한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'diagnosis.json').exists(), '완료 결과를 덮어쓰지 않는다'
    spec = previous.protocol(True)
    sources = {str(p.relative_to(REPO)): sha256(p) for p in (
        Path(__file__), REPO / 'scripts/_pdv2_pose_graph_lib.py',
        REPO / 'tests/keyframe/test_pose_graph_diagnosis.py',
        REPO / 'scripts/_pdv2_input_alignment_lib.py', REPO / 'src/keyframe/graph_manager.py',
        REPO / 'src/gnn/model.py', previous.OUT / 'protocol.json',
        previous.OUT / 'summary.json', previous.source.OUT / 'summary.json',
        previous.base.PREPARED, previous.base.GRAPH)}
    state = torch.load(previous.base.PREPARED, map_location='cpu', weights_only=False)
    original = torch.load(previous.base.GRAPH, map_location='cpu', weights_only=False)
    graph = previous.base.temporal_subgraph(original, np.arange(original.num_nodes))
    del original
    expected_edges = temporal_edges(state['sequences'])
    expected_attr = pose_attributes(state['poses'], expected_edges)
    np.testing.assert_array_equal(graph.edge_index.numpy(), expected_edges)
    np.testing.assert_array_equal(graph.x.numpy(), state['descriptors'])
    attr_error = float(abs(graph.edge_attr.numpy() - expected_attr).max())
    assert attr_error < 2e-7 and bool((graph.edge_type == 0).all())
    assert np.all(state['sequences'][expected_edges[0]] == state['sequences'][expected_edges[1]])
    # 이 실험은 입력 관측과 고정 모델의 국소 미분뿐이다. 갱신·설정 선택은 없다.
    write_json(OUT / 'protocol.json', {'models': MODELS, 'train': spec['train'], 'val': spec['val'],
        'graph_neighbor_offsets': [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
        'similarity_description_thresholds': {'relative_distance': .1, 'angle_degree': 1.},
        'thresholds_used_for_selection': False, 'model_update': False, 'test_used': False,
        'source_protocol_sha256': sha256(previous.OUT / 'protocol.json')})
    records, raw = {'train': {}, 'val': {}}, {}
    for i, seq in enumerate(spec['train']):
        ids = np.flatnonzero(state['sequences'] == i)
        assert len(ids) and np.all(np.diff(ids) == 1)
        path = cache_path(seq)
        cache = previous.base.exp.arrays(path)
        np.testing.assert_array_equal(cache['poses'], state['poses'][ids])
        np.testing.assert_array_equal(cache['descriptors'], state['descriptors'][ids])
        assert np.all(np.diff(cache['timestamps']) >= 0)
        # prepared의 frames는 좌표계 묶음이다. 실제 시각 순서는 원 캐시 scan_ids로 검산한다.
        record, attr, _ = audit_sequence(state['poses'][ids], cache['scan_ids'])
        sources[str(path.relative_to(REPO))] = sha256(path)
        records['train'][seq] = record
        raw['train/' + seq] = attr
        print('GRAPH_AUDITED', 'train', seq, record['nodes'], flush=True)
    for seq in spec['val']:
        _, _, paths, _ = previous.base.exp.old.base.inputs('val', seq)
        cache = previous.base.exp.arrays(paths['cache'])
        assert np.all(np.diff(cache['timestamps']) >= 0)
        record, attr, windows = audit_sequence(cache['poses'], cache['scan_ids'])
        raw['val/' + seq] = attr
        ledger = previous.OUT / 's1/full/epoch20' / (seq.replace('/', '_') + '.npz')
        row = previous.base.exp.arrays(ledger)
        np.testing.assert_array_equal(row['query_idx'], np.arange(30, len(cache['poses']) - 15))
        assert np.all(row['j_star'] <= row['query_idx'] - 30)
        np.testing.assert_allclose(row['top1_geo_dist'], np.linalg.norm(
            cache['poses_gt'][row['query_idx'], :3, 3] - cache['poses_gt'][row['j_star'], :3, 3], axis=1), atol=1e-10)
        record['final_error_geometry'] = error_geometry(cache['poses'], windows, row)
        edges = temporal_edges(np.zeros(len(cache['poses']), dtype=np.int64))
        odom_attr, gt_attr = pose_attributes(cache['poses'], edges), pose_attributes(cache['poses_gt'], edges)
        record['odom_gt_edge_difference'] = {
            'distance_m': quantiles(abs(np.expm1(5 * odom_attr[:, 0]) - np.expm1(5 * gt_attr[:, 0]))),
            'rotation_deg': quantiles(abs(odom_attr[:, 1] - gt_attr[:, 1]) * 180)}
        records['val'][seq] = record
        for p in (paths['cache'], ledger):
            sources[str(Path(p).relative_to(REPO))] = sha256(Path(p))
        print('GRAPH_AUDITED', 'val', seq, record['nodes'], flush=True)
    branches = branch_responses(raw, sources)
    for path, expected in sources.items():
        assert sha256(REPO / path) == expected, path
    write_json(OUT / 'diagnosis.json', {'graphs': records, 'branch_responses': branches,
        'train_graph_exact_edges': True, 'train_graph_attr_max_error': attr_error,
        'sources': sources, 'test_used': False, 'training_performed': False,
        'scope': '학습8·검증9 전체 그래프 감사. 모델당 각 시퀀스512중심의 국소 편향 미분. 미분은 고정 가중치의 입력 민감도이며 학습 손실 기울기가 아니다. 오류별 기하 유사도는 상관 관측이며 새 점수·최종 성능 또는 원인 확증이 아니다.'})
    print('POSE_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
