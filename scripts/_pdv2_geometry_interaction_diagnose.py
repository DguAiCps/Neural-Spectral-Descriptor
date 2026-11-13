#!/usr/bin/env python3
"""기하 결합의 실제 attention과 최종 후보·수락 순위 변화를 진단한다."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_geometry_interaction as run
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_no_context_ablation import sha256, write_json


def observe(net: torch.nn.Module, graph) -> tuple:
    """실제 층 입력에서 점수를 독립 계산하고 메시지와 비교한다."""
    degree = torch.bincount(graph.edge_index[1], minlength=graph.num_nodes)
    eligible = torch.nonzero(degree == 10).flatten()
    selected = eligible[torch.linspace(0, len(eligible) - 1, min(512, len(eligible)),
                                      device=eligible.device).long()]
    handles, stats, pending = [], {}, {}
    for number, conv in enumerate(net.gnn.convs):
        def before(layer, args, kwargs, index=number):
            h, edge = args[:2]
            assert kwargs.get('edge_logit') is None and kwargs.get('edge_value_gate') is None
            ids = torch.nonzero(torch.isin(edge[1], selected)).flatten()
            src, tgt = edge[:, ids]
            order = torch.argsort(tgt * len(h) + src)
            ids, src, tgt = ids[order], src[order], tgt[order]
            n, heads, width = len(selected), layer.heads, layer.head_dim
            assert len(src) == n * 10
            rel = h[src] - h[tgt]
            query = layer.W_q(h[tgt]).reshape(n, 10, heads, width)
            key = layer.W_k(rel).tanh().reshape(n, 10, heads, width)
            c = (query * key).sum(-1) / layer.scale
            b = layer.bias_mlp(kwargs['edge_attr'][ids]).reshape(n, 10, heads)
            cc, bb = c - c.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
            if layer.geometry_mode == 'interaction':
                modulation = bb
            elif layer.geometry_mode == 'temperature':
                modulation = b.mean(1, keepdim=True)
            else:
                modulation = torch.zeros_like(b)
            extra = cc * modulation.tanh()
            weights = (c + b + extra).softmax(1)
            additive, content = (c + b).softmax(1), c.softmax(1)
            value = layer.W_v(rel).tanh().reshape(n, 10, heads, width)
            pending[index] = (weights[..., None] * value).sum(1).flatten(1)
            entropy = -(weights * weights.clamp_min(1e-30).log()).sum(1)
            stats[index] = {
                'centers': n, 'content_centered_rms': float(cc.square().mean().sqrt()),
                'geometry_centered_rms': float(bb.square().mean().sqrt()),
                'interaction_centered_rms': float((extra - extra.mean(1, keepdim=True)).square().mean().sqrt()),
                'tv_vs_additive': float((.5 * (weights - additive).abs().sum(1)).mean()),
                'tv_vs_content': float((.5 * (weights - content).abs().sum(1)).mean()),
                'effective_neighbors': float(entropy.exp().mean()),
                'modulation_gain_mean': float((1 + modulation.tanh()).mean()),
                'modulation_saturated_fraction': float((modulation.tanh().abs() > .95).float().mean()),
            }
        def after(layer, args, output, index=number):
            expected = pending.pop(index)
            torch.testing.assert_close(output[selected], expected, atol=3e-5, rtol=1e-4)
            stats[index]['message_max_error'] = float((output[selected] - expected).abs().max())
        handles.append(conv.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(conv.register_forward_hook(after))
    try:
        with torch.no_grad():
            output = net(graph).float().cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    return output, stats


def attention(spec: dict, sources: dict) -> dict:
    """검증9의 표본 attention을 관측하고 기존 전체 키와 재현성을 확인한다."""
    records = {}
    for arm in run.CFG['arms']:
        net = run.model(arm).cuda().eval()
        cp = run.base.root(1, arm) / 'checkpoints/checkpoint_epoch_20.pth'
        net.load_state_dict(torch.load(cp, map_location='cpu', weights_only=False)['model_state_dict'])
        net.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(np.load(run.base.exp.old.base.RVEC)).float().cuda(), requires_grad=False)
        legacy = None
        if arm == 'additive':
            legacy = run.source.model('diff_tanh').cuda().eval()
            legacy.gnn.key_scale = torch.nn.Parameter(net.gnn.key_scale.detach().clone(), requires_grad=False)
            legacy.load_state_dict(net.state_dict(), strict=True)
        records[arm] = {}
        sources[str(cp.relative_to(REPO))] = sha256(cp)
        for seq in spec['val']:
            _, _, paths, _ = run.base.exp.old.base.inputs('val', seq)
            cache = run.base.exp.arrays(paths['cache'])
            graph = run.base.make_graph(cache['descriptors'].astype(np.float32), cache['poses'],
                                       np.zeros(len(cache['poses']), dtype=np.int64)).cuda()
            output, values = observe(net, graph)
            kp = run.base.root(1, arm) / 'epoch20' / ('keys_' + seq.replace('/', '_') + '.npz')
            key = run.base.normalize(output)
            expected = run.base.exp.arrays(kp)['embedding']
            np.testing.assert_allclose(key, expected, atol=4e-6, rtol=4e-5)
            records[arm][seq] = {'layers': values, 'key_max_error': float(abs(key - expected).max())}
            if legacy is not None:
                with torch.no_grad():
                    legacy_key = run.base.normalize(legacy(graph).float().cpu().numpy())
                np.testing.assert_allclose(legacy_key, expected, atol=4e-6, rtol=4e-5)
                records[arm][seq]['legacy_operator_key_max_error'] = float(abs(legacy_key - expected).max())
            sources[str(kp.relative_to(REPO))] = sha256(kp)
            del graph
        del net, legacy
        print('ATTENTION_DONE', arm, flush=True)
    names = ('content_centered_rms', 'geometry_centered_rms', 'interaction_centered_rms',
             'tv_vs_additive', 'tv_vs_content', 'effective_neighbors',
             'modulation_gain_mean', 'modulation_saturated_fraction')
    means = {a: {str(i): {name: float(np.mean([v['layers'][i][name] for v in seqs.values()]))
                         for name in names} for i in (0, 1)} for a, seqs in records.items()}
    return {'sequences': records, 'validation_sequence_mean': means,
            'scope': '검증시퀀스별512중심의10이웃·두층. 전체메시지와저장키독립재현. 국소효과는최종기여와구분.'}


def final_diagnosis(spec: dict, sources: dict, folders: dict = None) -> dict:
    """공통 질의의 정오 판정과 수락 점수 순위를 나누어 진단한다."""
    if folders is None:
        folders = {a: run.base.root(1, a) / 'epoch20' for a in run.CFG['arms']}
    arms = tuple(folders)
    rows, keys = {}, {}
    for arm in arms:
        rows[arm], keys[arm] = {}, {}
        for seq in spec['val']:
            folder = folders[arm]
            p = folder / (seq.replace('/', '_') + '.npz')
            rows[arm][seq] = run.base.exp.arrays(p)
            keys[arm][seq] = folder / ('keys_' + p.name)
            sources[str(p.relative_to(REPO))] = sha256(p)
            sources[str(keys[arm][seq].relative_to(REPO))] = sha256(keys[arm][seq])
    comparisons = {}
    for arm in arms[1:]:
        mixed_metrics = {k: {} for k in ('old_old', 'new_old', 'old_new', 'new_new')}
        details = {}
        for seq in spec['val']:
            a, b = rows['additive'][seq], rows[arm][seq]
            for k in ('query_idx', 'min_past_gt_dist', 'group'):
                np.testing.assert_array_equal(a[k], b[k])
            for name, labels, scores in (('old_old', a, a), ('new_old', b, a),
                                        ('old_new', a, b), ('new_new', b, b)):
                mixed_metrics[name][seq] = independent_metrics(labels, scores[run.METHOD])
            positive = a['min_past_gt_dist'] <= 5
            old_correct = a['top1_geo_dist'] <= 5
            new_correct = b['top1_geo_dist'] <= 5
            gains = np.flatnonzero(positive & ~old_correct & new_correct)
            losses = np.flatnonzero(positive & old_correct & ~new_correct)
            details[seq] = {'gains': len(gains), 'losses': len(losses),
                'final_ap_change_pp': 100 * (mixed_metrics['new_new'][seq]['AP'] - mixed_metrics['old_old'][seq]['AP'])}
            if not seq.startswith(('kitti/', 'kitti360/')):
                continue
            _, _, paths, _ = run.base.exp.old.base.inputs('val', seq)
            cache = run.base.exp.arrays(paths['cache'])
            key = run.base.exp.arrays(keys[arm][seq])['embedding']
            with np.load(paths['spec']) as z:
                bank = run.base.exp.old.base.bank_for(run.base.exp.OP, z)
            conf = spec['confidence']['1']
            losses_diagnosis = []
            for ix in losses:
                q = int(b['query_idx'][ix])
                ids = np.unique(np.r_[run.base.prefix_topk(key, q, q - 29, 100),
                                      run.base.prefix_topk(bank[0], q, q - 29, 100)])
                distance = np.linalg.norm(cache['poses_gt'][ids, :3, 3] - cache['poses_gt'][q, :3, 3], axis=1)
                channels = (key[ids] @ key[q], bank[0][ids] @ bank[0][q],
                    run.base.align_scores(bank[1], bank[2], q, ids),
                    run.base.column_align_scores(bank[3], q, ids))
                fused = sum(float(w) * run.base.minmax(v) for w, v in zip(conf['tuple'], channels))
                winner = int(np.argsort(-fused, kind='stable')[0])
                assert int(ids[winner]) == int(b['j_star'][ix])
                key_winner = int(np.argsort(-channels[0], kind='stable')[0])
                correct_ids = np.flatnonzero(distance <= 5)
                category = ('missing_union' if not len(correct_ids) else
                            'key_prefers_true_fusion_wrong' if distance[key_winner] <= 5 else 'key_also_wrong')
                losses_diagnosis.append({'query': q, 'old_winner': int(a['j_star'][ix]),
                    'new_winner': int(b['j_star'][ix]), 'new_winner_distance_m': float(distance[winner]),
                    'old_winner_in_new_union': bool(a['j_star'][ix] in ids),
                    'category': category, 'final_score_before': float(a[run.METHOD][ix]),
                    'final_score_after': float(b[run.METHOD][ix]),
                    'true_to_wrong_fused_margin': float(fused[correct_ids].max() - fused[winner]) if len(correct_ids) else None})
            details[seq]['lost_correct_queries'] = losses_diagnosis
            details[seq]['loss_categories'] = {name: sum(x['category'] == name for x in losses_diagnosis)
                for name in ('missing_union', 'key_prefers_true_fusion_wrong', 'key_also_wrong')}
        aggregates = {k: run.base.macro(v) for k, v in mixed_metrics.items()}
        decomposition = {}
        for name in ('macro', 'kitti', 'helipr', 'mulran'):
            values = {k: (v['macro'] if name == 'macro' else v['families'][name])['AP'] for k, v in aggregates.items()}
            winner = .5 * ((values['new_old'] - values['old_old']) + (values['new_new'] - values['old_new']))
            order = .5 * ((values['old_new'] - values['old_old']) + (values['new_new'] - values['new_old']))
            assert abs(winner + order - (values['new_new'] - values['old_old'])) < 1e-12
            decomposition[name] = {'winner_correctness_pp': 100 * winner, 'acceptance_score_order_pp': 100 * order,
                                   'total_pp': 100 * (values['new_new'] - values['old_old'])}
        comparisons[arm] = {'sequences': details, 'mixed_metrics': aggregates, 'ap_arithmetic_decomposition': decomposition}
    return {'comparisons': comparisons, 'scope': '같은질의에서승자정오와최종수락점수만교차하여AP변화를두교환순서평균으로산술분해. 혼합점수는실행가능한새방법이아니며독립인과효과나가산모듈기여가아니다. KITTI손실질의는새후보부터재구성.'}


def main() -> None:
    """동결한 최종 결과에서만 진단하고 별도 해시를 보존한다."""
    spec = run.protocol(True)
    audit = run.base.read(run.OUT / 'completion_validation.json')
    assert audit['passed']
    sources = {str(Path(__file__).relative_to(REPO)): sha256(Path(__file__)),
               str((run.OUT / 'protocol.json').relative_to(REPO)): sha256(run.OUT / 'protocol.json'),
               str((run.OUT / 'completion_validation.json').relative_to(REPO)): sha256(run.OUT / 'completion_validation.json')}
    stats = attention(spec, sources)
    final = final_diagnosis(spec, sources)
    decay = REPO / 'results/pdv2_geometry_decay_20261023'
    for name in ('protocol.json', 'completion_validation.json'):
        sources[str((decay / name).relative_to(REPO))] = sha256(decay / name)
    assert run.base.read(decay / 'completion_validation.json')['passed']
    decay_final = final_diagnosis(run.base.read(decay / 'protocol.json'), sources,
        {'additive': decay / 's1/diff_tanh/epoch20', 'nodecay': decay / 's1/diff_tanh_nodecay/epoch20'})
    for p, digest in sources.items():
        assert sha256(REPO / p) == digest, p
    write_json(run.OUT / 'mechanism_diagnosis.json', {'attention': stats, 'final': final, 'previous_decay_final': decay_final,
        'sources': sources, 'test_used': False, 'passed': True})
    print('MECHANISM_DIAGNOSIS_DONE', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
