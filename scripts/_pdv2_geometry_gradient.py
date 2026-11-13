#!/usr/bin/env python3
"""동일한 실제 학습을 재실행하면서 기하 분기의 기울기·갱신을 기록한다."""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_value_relation as source
import gnn.trainer as trainer_module
from _pdv2_geometry_gradient_lib import gradient_statistics, observe_bias, parameter_group
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash

base = source.base
OUT = REPO / 'results/pdv2_geometry_gradient_20261023'
CONFIG = REPO / 'configs/pdv2_geometry_gradient.yaml'
CFG = yaml.safe_load(CONFIG.read_text())
OriginalTrainer = trainer_module.GNNTrainer


def protocol(check_inputs: bool = False) -> dict:
    """소스·학습 입력의 동결 해시를 확인한다."""
    value = base.read(OUT / 'protocol.json')
    for field in (('runtime', 'inputs') if check_inputs else ('runtime',)):
        for path, expected in value[field].items():
            assert sha256(REPO / path) == expected, path
    return value


def prepare() -> None:
    """관측 지점과 원래 학습 입력을 실행 전에 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        protocol(True)
        return
    old = source.protocol(True)
    runtime = dict(old['runtime'])
    for path in (CONFIG, Path(__file__), REPO / 'scripts/_pdv2_geometry_gradient_lib.py',
                 REPO / 'tests/gnn/test_geometry_gradient_diagnosis.py'):
        runtime[str(path.relative_to(REPO))] = sha256(path)
    inputs = dict(old['inputs'])
    for arm in CFG['arms']:
        folder = source.OUT / 's1' / arm
        for path in (folder / 'initial_model.json', folder / 'training_complete.json',
                     folder / 'checkpoints/checkpoint_epoch_10.pth', folder / 'checkpoints/checkpoint_epoch_20.pth'):
            inputs[str(path.relative_to(REPO))] = sha256(path)
    write_json(OUT / 'protocol.json', {'config': CFG, 'runtime': runtime, 'inputs': inputs,
        **{key: old[key] for key in ('graph_hashes', 'train', 'val', 'confidence')},
        'test_used': False, 'purpose': '학습 기울기 원인 진단. 저장 모델의 성능 향상이나 역사적 학습 궤적의 비트 동일성을 전제하지 않는다.'})


def train(arm: str) -> None:
    """optimizer 직전·직후와 clipping 전후를 관측하며 원래 train 함수를 쓴다."""
    spec = protocol(True)
    folder = base.root(1, arm)
    assert not (folder / 'training_complete.json').exists(), '완료 학습을 덮어쓰지 않는다'
    original = torch.load(base.GRAPH, map_location='cpu', weights_only=False)
    edge = original.edge_index[:, original.edge_type == 0]
    attr = original.edge_attr[original.edge_type == 0]
    degree = torch.bincount(edge[1], minlength=original.num_nodes)
    centers = torch.nonzero(degree == 10).flatten()
    centers = centers[torch.linspace(0, len(centers) - 1, 512).long()]
    ids = torch.nonzero(torch.isin(edge[1], centers)).flatten()
    ids = ids[torch.argsort(edge[1, ids] * original.num_nodes + edge[0, ids])]
    attributes = attr[ids].cuda()
    assert len(attributes) == 5120
    del original, edge, attr
    active, records, selected = {}, [], set(CFG['trace_steps'])
    original_clip = torch.nn.utils.clip_grad_norm_

    class TracedTrainer(OriginalTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            active['trainer'] = self
            active['steps'] = 0
            active['named'] = list(self.model.named_parameters())

            def after_step(optimizer, args, kwargs):
                active['steps'] += 1
                pending = active.pop('pending', None)
                if pending is None:
                    return
                before = active.pop('before')
                delta = {}
                for name, parameter in active['named']:
                    group = parameter_group(name)
                    change = (parameter.detach() - before[name]).double().square().sum()
                    delta[group] = delta.get(group, 0.) + float(change)
                pending['actual_update_norm'] = {k: float(np.sqrt(v)) for k, v in delta.items()}
                pending['bias_after_update'] = observe_bias(self.model, attributes)
                records.append(pending)
                write_json(folder / 'gradient_trace.json', {'records': records, 'test_used': False})
            self.optimizer.register_step_post_hook(after_step)

    def observed_clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        step = active['steps'] + 1
        measure = step in selected
        if measure:
            trainer = active['trainer']
            active['pending'] = {'step': step, 'epoch': trainer.epoch + 1,
                'learning_rate': trainer.optimizer.param_groups[0]['lr'],
                'amp_scale': trainer.scaler.get_scale(),
                'before_clip': gradient_statistics(active['named'], CFG['weight_decay']),
                'bias_before_update': observe_bias(trainer.model, attributes)}
            active['before'] = {n: p.detach().clone() for n, p in active['named']}
        norm = original_clip(parameters, *args, **kwargs)
        if measure:
            active['pending']['total_gradient_norm_before_clip'] = float(norm)
            active['pending']['after_clip'] = gradient_statistics(active['named'], CFG['weight_decay'])
        return norm

    trainer_module.GNNTrainer = TracedTrainer
    torch.nn.utils.clip_grad_norm_ = observed_clip
    try:
        base.train(1, arm)
    finally:
        trainer_module.GNNTrainer = OriginalTrainer
        torch.nn.utils.clip_grad_norm_ = original_clip
    assert active['steps'] == 900 and [v['step'] for v in records] == CFG['trace_steps']
    before = base.read(source.OUT / 's1' / arm / 'initial_model.json')
    assert before == base.read(folder / 'initial_model.json')
    expected_orders = base.read(source.OUT / 's1' / arm / 'training_complete.json')['orders']
    assert expected_orders == base.read(folder / 'training_complete.json')['orders']
    comparisons = {}
    for epoch in (10, 20):
        name = f'checkpoint_epoch_{epoch}.pth'
        old = torch.load(source.OUT / 's1' / arm / 'checkpoints' / name, map_location='cpu', weights_only=False)
        new = torch.load(folder / 'checkpoints' / name, map_location='cpu', weights_only=False)
        differences = {k: float(abs(new['model_state_dict'][k] - v).max())
                       for k, v in old['model_state_dict'].items()}
        comparisons[str(epoch)] = {'maximum_tensor_error': max(differences.values()),
            'exact_tensors': sum(e == 0 for e in differences.values()), 'tensors': len(differences),
            'parameter_errors': differences}
    write_json(folder / 'trace_validation.json', {'initial_exact': True, 'orders_exact': True,
        'steps': 900, 'observations': len(records), 'checkpoint_comparison': comparisons,
        'sources': {str(p.relative_to(REPO)): sha256(p) for p in (
            folder / 'gradient_trace.json', folder / 'training_complete.json')}, 'test_used': False})
    print('TRACE_COMPLETE', arm, {k: v['maximum_tensor_error'] for k, v in comparisons.items()}, flush=True)


base.OUT, base.CONFIG, base.CFG = OUT, CONFIG, CFG
base.model, base.protocol = source.model, protocol

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'train'))
    parser.add_argument('--arm', choices=CFG['arms'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.WARNING)
    prepare() if args.action == 'prepare' else train(args.arm)
