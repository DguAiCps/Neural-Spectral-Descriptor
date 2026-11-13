#!/usr/bin/env python3
"""실제 층 입력에서 attention의 특징 점수·기하 편향·집중도를 분해한다."""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_attention_review as run
from _pdv2_no_context_ablation import sha256, write_json


def summarize(content: torch.Tensor, bias: torch.Tensor, value: torch.Tensor,
              offset: torch.Tensor) -> dict:
    """중심×이웃10×head의 점수와 같은 값 전달에서 국소 차이를 계산한다."""
    weights = {'full': (content + bias).softmax(1), 'content': content.softmax(1),
               'geometry': bias.softmax(1), 'uniform': torch.full_like(content, .1)}
    full = weights['full']
    entropy = -(full * full.clamp_min(1e-30).log()).sum(1)
    centered_content, centered_bias = content - content.mean(1, keepdim=True), bias - bias.mean(1, keepdim=True)
    message = {name: (w[..., None] * value).sum(1).flatten(1) for name, w in weights.items()}
    full_norm = message['full'].norm(dim=1).clamp_min(1e-12)
    result = {
        'normalized_entropy_mean': float((entropy / np.log(10)).mean()),
        'effective_neighbors_mean': float(entropy.exp().mean()),
        'max_weight_mean': float(full.max(1).values.mean()),
        'content_centered_rms': float(centered_content.square().mean().sqrt()),
        'geometry_centered_rms': float(centered_bias.square().mean().sqrt()),
        'head_pair_tv_mean': float(torch.stack([
            .5 * (full[..., a] - full[..., b]).abs().sum(1)
            for a, b in itertools.combinations(range(full.shape[-1]), 2)]).mean()),
        'past_mass_mean': float((full * (offset < 0)[..., None]).sum(1).mean()),
        'offset_mass_mean': {str(int(k)): float((full * (offset == k)[..., None]).sum(1).mean())
                             for k in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)},
    }
    for mode in ('content', 'geometry', 'uniform'):
        result['tv_full_vs_' + mode] = float((.5 * (full - weights[mode]).abs().sum(1)).mean())
        relative = (message[mode] - message['full']).norm(dim=1) / full_norm
        result['local_message_relative_change_' + mode] = torch.quantile(relative,
            torch.tensor([.1, .5, .9], device=relative.device)).cpu().tolist()
    return result


def measure(net: torch.nn.Module, graph, nonlinear_key: bool) -> tuple:
    """실제 forward를 바꾸지 않는 hook에서 표본 모든 이웃을 독립 합산한다."""
    stats, pending, handles = {}, {}, []
    degree = torch.bincount(graph.edge_index[1], minlength=graph.num_nodes)
    eligible = torch.flatnonzero(degree == 10) if hasattr(torch, 'flatnonzero') else torch.nonzero(degree == 10).flatten()
    selected = eligible[torch.linspace(0, len(eligible) - 1,
        min(len(eligible), run.CFG['attention_centers_per_graph']), device='cuda').long()]
    for index, conv in enumerate(net.gnn.convs):
        def before(layer, args, kwargs, number=index):
            h, edge = args[:2]
            assert kwargs.get('edge_logit') is None and kwargs.get('edge_value_gate') is None
            indices = torch.nonzero(torch.isin(edge[1], selected)).flatten()
            src, tgt = edge[:, indices]
            ordering = torch.argsort(tgt * len(h) + src)
            indices, src, tgt = indices[ordering], src[ordering], tgt[ordering]
            count, heads, width = len(selected), layer.heads, layer.head_dim
            assert len(src) == count * 10
            offset = (src - tgt).reshape(count, 10)
            torch.testing.assert_close(offset, torch.tensor([-5,-4,-3,-2,-1,1,2,3,4,5],
                device='cuda').expand(count, -1), atol=0, rtol=0)
            center, neighbor = h[tgt], h[src]
            relation = neighbor - center if layer.value_source == 'diff' else neighbor
            q = layer.W_q(center).reshape(count, 10, heads, width)
            k = layer.W_k(relation).reshape(count, 10, heads, width)
            content = (q * (k.tanh() if nonlinear_key else k)).sum(-1) / layer.scale
            bias = layer.bias_mlp(kwargs['edge_attr'][indices]).reshape(count, 10, heads)
            projected = layer.W_v(relation).reshape(count, 10, heads, width)
            value = layer.transform_value(projected) if hasattr(layer, 'transform_value') else projected
            weights = (content + bias).softmax(1)
            expected = (weights[..., None] * value).sum(1).flatten(1)
            q64 = F.linear(center.double(), layer.W_q.weight.double()).reshape(count,10,heads,width)
            kd = F.linear((neighbor-center).double(), layer.W_k.weight.double()).reshape(count,10,heads,width)
            ks = F.linear(neighbor.double(), layer.W_k.weight.double()).reshape(count,10,heads,width)
            ad = ((q64*kd).sum(-1)/layer.scale + bias.double()).softmax(1)
            ast = ((q64*ks).sum(-1)/layer.scale + bias.double()).softmax(1)
            error = float((ad-ast).abs().max())
            # 차이를 float32로 먼저 구하지 않고 float64 안에서 빼야 수학적 항등식을 검산한다.
            kd_exact = F.linear(neighbor.double()-center.double(), layer.W_k.weight.double()).reshape(count,10,heads,width)
            ad_exact = ((q64*kd_exact).sum(-1)/layer.scale + bias.double()).softmax(1)
            exact_error = float((ad_exact-ast).abs().max())
            assert exact_error < 1e-11
            stats[number] = {'centers': count, 'edges': len(src), 'degree': 10,
                'linear_identity_max_error_float64': exact_error,
                'linear_identity_with_fp32_subtraction_error': error,
                **summarize(content, bias, value, offset)}
            pending[number] = expected
        def after(layer, args, output, number=index):
            expected = pending.pop(number)
            actual = output[selected]
            torch.testing.assert_close(expected, actual, atol=3e-5, rtol=1e-4)
            stats[number]['independent_message_max_error'] = float((expected-actual).abs().max())
        handles.append(conv.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(conv.register_forward_hook(after))
    try:
        with torch.no_grad():
            output = net(graph).float().cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    return output, stats


def main() -> None:
    """원래 차이 및 최근 선택 모델의 학습·검증 attention을 관측한다."""
    spec = run.protocol(True)
    old = run.source.previous.previous
    original = torch.load(run.base.GRAPH, map_location='cpu', weights_only=False)
    train_graph = run.base.temporal_subgraph(original, np.arange(original.num_nodes)).cuda()
    del original
    records, sources = {}, {str(Path(__file__).relative_to(REPO)): sha256(Path(__file__))}
    for arm in run.CFG['diagnostic_models']:
        original_arm = arm == 'original_diff'
        parent, name = (old, 'diff') if original_arm else (run.source, arm)
        saved = run.read(parent.OUT / 'summary.json')['per_seed']['1'][name]['selected']
        checkpoint = parent.OUT / 's1' / name / 'checkpoints' / f"checkpoint_epoch_{saved['epoch']}.pth"
        sources[str(checkpoint.relative_to(REPO))] = sha256(checkpoint)
        net = parent.model(name).cuda().eval()
        net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state_dict'], strict=True)
        net.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(np.load(run.base.exp.old.base.RVEC)).float().cuda(), requires_grad=False)
        _, training = measure(net, train_graph, not original_arm)
        records[arm] = {'epoch': saved['epoch'], 'training': training, 'validation': {}}
        for seq in spec['val']:
            _, _, paths, _ = run.base.exp.old.base.inputs('val', seq)
            cache = run.base.exp.arrays(paths['cache'])
            graph = run.base.make_graph(cache['descriptors'].astype(np.float32), cache['poses'],
                np.zeros(len(cache['descriptors']), dtype=np.int64)).cuda()
            key, stats = measure(net, graph, not original_arm)
            path = parent.OUT / 's1' / name / f"epoch{saved['epoch']}" / ('keys_' + seq.replace('/', '_') + '.npz')
            expected = run.base.exp.arrays(path)['embedding']
            key = run.base.normalize(key)
            np.testing.assert_allclose(key, expected, atol=4e-6, rtol=4e-5)
            sources[str(path.relative_to(REPO))] = sha256(path)
            records[arm]['validation'][seq] = {'layers': stats, 'key_max_error': float(abs(key-expected).max())}
            del graph
        print('ATTENTION_OBSERVED', arm, flush=True)
        del net
    macro = {}
    metrics = ('normalized_entropy_mean','effective_neighbors_mean','max_weight_mean',
        'content_centered_rms','geometry_centered_rms','head_pair_tv_mean','past_mass_mean',
        'tv_full_vs_content','tv_full_vs_geometry','tv_full_vs_uniform')
    for arm, model in records.items():
        macro[arm] = {str(layer): {m: float(np.mean([r['layers'][layer][m] for r in model['validation'].values()]))
                      for m in metrics} for layer in (0,1)}
    for p,h in sources.items(): assert sha256(REPO/p)==h
    write_json(run.OUT / 'attention_statistics.json', {'models':records,'validation_sequence_mean':macro,
        'sources':sources,'test_used':False,'training_performed':False,
        'scope':'선택모델5개·추론모드·실제 학습전체와검증9. 그래프마다 이웃10개인512중심의 모든 이웃 관측. 검증통계는9시퀀스동일비중이며 AP집계와구분한다. 학습전체표본은계열균형표본이아니다. 국소점수삭제의메시지변화는최종성능이나재학습의인과효과가아니다.'})

if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
