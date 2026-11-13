#!/usr/bin/env python3
"""초기 실제 세 업데이트와 고정 가중치에서 FP16/FP32 손실 기울기를 비교한다."""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_value_relation as source
import gnn.trainer as training
from _pdv2_geometry_gradient_lib import gradient_statistics, observe_bias
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash

OUT = REPO / 'results/pdv2_geometry_precision_20261023'
ARMS = ('diff_linear', 'standard_linear', 'diff_tanh')
OriginalTrainer = training.GNNTrainer


def prefix(arm: str) -> Path:
    """원래 학습의 세 번째 갱신 뒤를 저장하고 계획대로 중단한다."""
    target = OUT / 'prefix' / arm / 'step3.pth'
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)

    class FinishedPrefix(Exception):
        pass

    class PrefixTrainer(OriginalTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prefix_steps = 0

            def after(optimizer, args, kwargs):
                self.prefix_steps += 1
                if self.prefix_steps == 3:
                    torch.save({'model_state_dict': self.model.state_dict(), 'steps': 3}, target)
                    raise FinishedPrefix()
            self.optimizer.register_step_post_hook(after)

    base = source.base
    base.OUT = OUT / 'prefix_training'
    base.model, base.protocol = source.model, source.protocol
    training.GNNTrainer = PrefixTrainer
    try:
        base.train(1, arm)
    except FinishedPrefix:
        pass
    finally:
        training.GNNTrainer = OriginalTrainer
    assert target.exists()
    return target


def main() -> None:
    """드롭아웃을 끈 동일 배치에서 수치 정밀도를 대조한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'diagnosis.json').exists(), '완료 결과는 덮어쓰지 않는다'
    spec = source.protocol(True)
    sources = {str(p.relative_to(REPO)): sha256(p) for p in (
        Path(__file__), REPO / 'scripts/_pdv2_geometry_gradient_lib.py',
        source.OUT / 'protocol.json', source.OUT / 'summary.json', source.base.GRAPH)}
    graph = torch.load(source.base.GRAPH, map_location='cpu', weights_only=False)
    graph = source.base.temporal_subgraph(graph, np.arange(graph.num_nodes)).cuda()
    values = source.base.triplets(1).copy()
    rng = np.random.RandomState(100001)
    rng.shuffle(values)
    batches = [torch.as_tensor(values[i * 4096:i * 4096 + 1024], device='cuda') for i in (0, 15, 30)]
    criterion = training.InfoNCELoss(.1)
    records = {}
    for arm in ARMS:
        early = prefix(arm)
        selected = json.loads((source.OUT / 'summary.json').read_text())['per_seed']['1'][arm]['selected']['epoch']
        checkpoint = source.OUT / 's1' / arm / 'checkpoints' / f'checkpoint_epoch_{selected}.pth'
        sources[str(early.relative_to(REPO))] = sha256(early)
        sources[str(checkpoint.relative_to(REPO))] = sha256(checkpoint)
        torch.manual_seed(1)
        initial = source.model(arm).state_dict()
        snapshots = {'initial': initial,
            'step3': torch.load(early, map_location='cpu', weights_only=False)['model_state_dict'],
            'selected': torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state_dict']}
        records[arm] = {}
        for stage, state in snapshots.items():
            net = source.model(arm).cuda().train()
            # BN은 학습 배치 통계, dropout은 0으로 고정하여 난수 차이를 제거한다.
            net.gnn.dropout = 0.
            for module in net.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.
            records[arm][stage] = []
            for batch_index, ids in enumerate(batches):
                pair = {}
                for precision in ('fp16', 'fp32'):
                    net.load_state_dict(state, strict=True)
                    net.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', enabled=precision == 'fp16', dtype=torch.float16):
                        embedding = net(graph)
                        loss = criterion(embedding[ids[:, 0]], embedding[ids[:, 1]], embedding[ids[:, 2]])
                    (loss * 128.).backward()
                    for parameter in net.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(128.)
                            assert torch.isfinite(parameter.grad).all()
                    pair[precision] = {'loss': float(loss.detach()),
                        'gradients': gradient_statistics(list(net.named_parameters()), 1e-5)}
                    del embedding, loss
                records[arm][stage].append(pair)
            print('PRECISION_OBSERVED', arm, stage, flush=True)
            del net
            torch.cuda.empty_cache()
    for path, expected in sources.items():
        assert sha256(REPO / path) == expected, path
    write_json(OUT / 'diagnosis.json', {'records': records, 'sources': sources,
        'batch_sha256': [array_hash(b.cpu().numpy()) for b in batches],
        'test_used': False, 'prefix_updates_per_arm': 3,
        'scope': '모델3종×초기/실제3업데이트/기존선택가중치×고정학습배치3개×FP16/FP32=54역전파. 학습 BN·dropout0·loss scale128. 역사적 확률적 학습 평균이나 최종 성능이 아니다.'})


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.WARNING)
    main()
