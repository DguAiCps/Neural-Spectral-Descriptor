#!/usr/bin/env python3
"""소프트맥스에서 상쇄되어야 하는 공통 점수 방향의 수치 기울기를 측정한다."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_geometry_decay as run
from gnn.trainer import InfoNCELoss
from _pdv2_no_context_ablation import sha256, write_json

OUT = REPO / 'results/pdv2_geometry_nullspace_20261023'


def main() -> None:
    """같은 중심의 점수 기울기 합과 정보 없는 방향의 영향을 대조한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'diagnosis.json').exists(), '완료 결과는 덮어쓰지 않는다'
    run.protocol(True)
    graph = torch.load(run.base.GRAPH, map_location='cpu', weights_only=False)
    graph = run.base.temporal_subgraph(graph, np.arange(graph.num_nodes)).cuda()
    target = graph.edge_index[1]
    degree = torch.bincount(target, minlength=graph.num_nodes).double().clamp_min(1)
    triplets = run.base.triplets(1).copy()
    np.random.RandomState(100001).shuffle(triplets)
    batches = [torch.as_tensor(triplets[k * 4096:k * 4096 + 1024], device='cuda') for k in (0, 15, 30)]
    models = [(a, run.base.root(1, a) / 'checkpoints/checkpoint_epoch_20.pth') for a in run.CFG['arms']]
    models.append(('standard_linear', run.source.OUT / 's1/standard_linear/checkpoints/checkpoint_epoch_10.pth'))
    sources = {str(p.relative_to(REPO)): sha256(p) for p in (
        Path(__file__), run.OUT / 'protocol.json', run.base.GRAPH)}
    records = {}
    for arm, checkpoint in models:
        sources[str(checkpoint.relative_to(REPO))] = sha256(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state_dict']
        net = run.source.model('standard_linear' if arm == 'standard_linear' else 'diff_tanh').cuda().train()
        net.gnn.dropout = 0.
        for module in net.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        records[arm] = []
        for ids in batches:
            pair = {}
            for precision in ('fp16', 'fp32'):
                net.load_state_dict(state, strict=True)
                net.zero_grad(set_to_none=True)
                measured, handles = {}, []
                for layer, conv in enumerate(net.gnn.convs):
                    def before_backward(module, args, output, number=layer):
                        if not output.requires_grad:
                            return
                        with torch.no_grad():
                            hidden = module[0](args[0]).relu().detach()

                        def measure(gradient):
                            assert number not in measured, '실제 역전파 관측은 층당 한 번이다'
                            g = gradient.detach().double() / 128.
                            h = hidden.double()
                            total = torch.zeros(graph.num_nodes, 4, device='cuda', dtype=torch.float64)
                            total.index_add_(0, target, g)
                            common = total[target] / degree[target, None]
                            projected = g - common
                            raw_weight, informative_weight = g.T @ h, projected.T @ h
                            common_weight = raw_weight - informative_weight
                            rn, pn, cn = [float(t.norm()) for t in (raw_weight, informative_weight, common_weight)]
                            measured[number] = {
                                'edge_gradient_norm': float(g.norm()),
                                'common_direction_gradient_norm': float(common.norm()),
                                'common_to_all_gradient_norm_ratio': float(common.norm() / g.norm().clamp_min(1e-30)),
                                'last_weight_gradient_raw_norm': rn,
                                'last_weight_gradient_projected_norm': pn,
                                'last_weight_gradient_common_norm': cn,
                                'projected_to_raw_weight_gradient_norm_ratio': pn / max(rn, 1e-30),
                                'raw_projected_cosine': float((raw_weight * informative_weight).sum()) / max(rn * pn, 1e-30),
                                'last_bias_gradient_from_edges': g.sum(0).cpu().tolist(),
                                'gradient_additivity_error': float(abs(raw_weight - informative_weight - common_weight).max()),
                            }
                            return gradient
                        output.register_hook(measure)
                    handles.append(conv.bias_mlp.register_forward_hook(before_backward))
                try:
                    with torch.autocast('cuda', dtype=torch.float16, enabled=precision == 'fp16'):
                        embedding = net(graph)
                        loss = InfoNCELoss(.1)(embedding[ids[:, 0]], embedding[ids[:, 1]], embedding[ids[:, 2]])
                    (loss * 128.).backward()
                finally:
                    for handle in handles:
                        handle.remove()
                assert set(measured) == {0, 1}
                for layer, conv in enumerate(net.gnn.convs):
                    measured[layer]['actual_last_bias_gradient'] = (conv.bias_mlp[-1].bias.grad / 128.).double().cpu().tolist()
                pair[precision] = {'layers': measured, 'loss': float(loss.detach())}
                del embedding, loss
            records[arm].append(pair)
        print('NULL_DIRECTION_OBSERVED', arm, flush=True)
        del net
        torch.cuda.empty_cache()
    for path, digest in sources.items():
        assert sha256(REPO / path) == digest, path
    write_json(OUT / 'diagnosis.json', {'records': records, 'sources': sources, 'test_used': False,
        'scope': '동결가중치3종·학습 BN/dropout0·배치3개·FP16/FP32의18역전파,36층 관측. 이웃별 점수에 같은 상수를 더하는 방향은 소프트맥스 불변성이므로 정확한 산술에서 기울기가0이다. 관측 기울기를 중심별 평균 제거해 분해할 뿐 모델을 바꾸지 않는다. 학습 기여율이나 인과 확증이 아니다.'})


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
