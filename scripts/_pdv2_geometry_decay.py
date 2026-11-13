#!/usr/bin/env python3
"""차이 K/V-tanh의 기하 분기 감쇠만 제거하고 같은 최종 판정을 비교한다."""
import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_value_relation as source
import gnn.trainer as training
from _pdv2_geometry_gradient_lib import optimizer_partition, observe_bias
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash

base = source.base
OUT = REPO / 'results/pdv2_geometry_decay_20261023'
TRACE = REPO / 'results/pdv2_geometry_gradient_20261023'
CONFIG = REPO / 'configs/pdv2_geometry_decay.yaml'
CFG = yaml.safe_load(CONFIG.read_text())
METHOD = CFG['final_score']


def model(arm: str) -> torch.nn.Module:
    """두 조건에서 구조·초기값이 같은 차이 K/V-tanh 모델을 만든다."""
    assert arm in CFG['arms']
    return source.model('diff_tanh')


def protocol(inputs: bool = False) -> dict:
    """실행 전에 고정한 코드와 입력을 확인한다."""
    result = base.read(OUT / 'protocol.json')
    for field in (('runtime', 'inputs') if inputs else ('runtime',)):
        for p, h in result[field].items():
            assert sha256(REPO / p) == h, p
    return result


def prepare() -> None:
    """이번 추적에서 다시 학습한 기준과 감쇠 제거의 한 가지 개입을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        protocol(True)
        return
    old = source.protocol(True)
    reference = TRACE / 's1/diff_tanh'
    complete = base.read(reference / 'training_complete.json')
    inputs = dict(old['inputs'])
    for p in (TRACE / 'protocol.json', reference / 'trace_validation.json',
              reference / 'gradient_trace.json', reference / 'initial_model.json',
              reference / 'training_complete.json'):
        inputs[str(p.relative_to(REPO))] = sha256(p)
    target = base.root(1, 'diff_tanh')
    (target / 'checkpoints').mkdir(parents=True, exist_ok=True)
    for name, digest in complete['checkpoints'].items():
        p = reference / 'checkpoints' / name
        assert sha256(p) == digest
        inputs[str(p.relative_to(REPO))] = digest
        shutil.copy2(p, target / 'checkpoints' / name)
    for name in ('initial_model.json', 'training_complete.json', 'training_progress.json'):
        shutil.copy2(reference / name, target / name)
    runtime = dict(old['runtime'])
    for p in (CONFIG, Path(__file__), REPO / 'scripts/_pdv2_geometry_gradient_lib.py',
              REPO / 'scripts/_pdv2_geometry_decay_run.py', REPO / 'scripts/_pdv2_geometry_decay_finish.py',
              REPO / 'tests/gnn/test_geometry_decay_partition.py'):
        runtime[str(p.relative_to(REPO))] = sha256(p)
    write_json(OUT / 'protocol.json', {'config': CFG, 'runtime': runtime, 'inputs': inputs,
        **{k: old[k] for k in ('graph_hashes', 'train', 'val', 'confidence')},
        'test_used': False, 'checkpoint_selection': '양쪽 20에포크900갱신 고정. 검증으로 에포크·감쇠값을 재선택하지 않는다.',
        'scope': '기하 인코더와 두 bias MLP의 Adam coupled weight decay만1e-5에서0으로 바꾼다. 나머지 학습·모듈·후보·최종 판정 유지. 기준은 이번 기울기 추적 재학습이므로 역사적 가중치와 동일하다고 전제하지 않는다.'})


def train() -> None:
    """기하 분기만 감쇠0인 매개변수 그룹으로 원래 900갱신을 수행한다."""
    protocol(True)
    original = training.GNNTrainer
    partition_record = {}

    class GeometryDecayTrainer(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            groups = optimizer_partition(self.model, CFG['geometry_weight_decay'], CFG['other_weight_decay'])
            names = {id(p): n for n, p in self.model.named_parameters()}
            partition_record.update({str(i): {'weight_decay': g['weight_decay'],
                'parameters': [names[id(p)] for p in g['params']]} for i, g in enumerate(groups)})
            self.optimizer = torch.optim.Adam(groups, lr=kwargs['learning_rate'])

    training.GNNTrainer = GeometryDecayTrainer
    try:
        base.train(1, 'diff_tanh_nodecay')
    finally:
        training.GNNTrainer = original
    folder = base.root(1, 'diff_tanh_nodecay')
    reference = base.root(1, 'diff_tanh')
    assert base.read(folder / 'initial_model.json') == base.read(reference / 'initial_model.json')
    assert base.read(folder / 'training_complete.json')['orders'] == base.read(reference / 'training_complete.json')['orders']
    write_json(folder / 'optimizer_partition.json', {'groups': partition_record,
        'initial_exact': True, 'triplet_orders_exact': True, 'updates': 900})


def report() -> None:
    """고정한 최종 판정의 성능과 기하 분기의 사용 정도를 집계한다."""
    spec = protocol(True)
    result = {}
    for arm in CFG['arms']:
        folder = base.root(1, arm) / 'epoch20'
        rows = {s: base.read(folder / (s.replace('/', '_') + '.json')) for s in spec['val']}
        result[arm] = {m: base.macro({s: v['metrics'][m] for s, v in rows.items()}) for m in ('raw', METHOD)}
        print('FINAL', arm, result[arm][METHOD]['macro'], flush=True)
    change = {k: 100 * (result['diff_tanh_nodecay'][METHOD]['macro'][k] - result['diff_tanh'][METHOD]['macro'][k])
              for k in base.exp.METRICS}
    write_json(OUT / 'summary.json', {'results': result, 'nodecay_minus_reference_pp': change,
        'epoch': 20, 'seed': 1, 'test_used': False, 'protocol_sha256': sha256(OUT / 'protocol.json')})
    observations = {}
    for arm in CFG['arms']:
        net = model(arm).cuda().eval()
        p = base.root(1, arm) / 'checkpoints/checkpoint_epoch_20.pth'
        net.load_state_dict(torch.load(p, map_location='cpu', weights_only=False)['model_state_dict'])
        observations[arm] = {}
        for seq in spec['val']:
            _, _, paths, _ = base.exp.old.base.inputs('val', seq)
            cache = base.exp.arrays(paths['cache'])
            from _pdv2_input_alignment_lib import pose_attributes
            centers = np.linspace(5, len(cache['poses']) - 6, 512).astype(int)
            offsets = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
            edges = np.stack(((centers[:, None] + offsets).reshape(-1), np.repeat(centers, 10)))
            attr = torch.tensor(pose_attributes(cache['poses'], edges), device='cuda')
            observations[arm][seq] = observe_bias(net, attr)
    write_json(OUT / 'bias_observations.json', {'models': observations, 'test_used': False})


base.OUT, base.CONFIG, base.CFG = OUT, CONFIG, CFG
base.model, base.protocol = model, protocol

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'train', 'val', 'report'))
    parser.add_argument('--arm', choices=CFG['arms'])
    parser.add_argument('--sequence')
    args = parser.parse_args()
    torch.set_num_threads(4 if args.action == 'train' else 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.WARNING)
    if args.action == 'prepare':
        prepare()
    elif args.action == 'train':
        train()
    elif args.action == 'val':
        source.validation(1, args.arm, 20, args.sequence)
    else:
        report()
