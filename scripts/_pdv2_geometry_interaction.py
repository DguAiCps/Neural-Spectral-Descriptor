#!/usr/bin/env python3
"""같은 차이 K/V-tanh에서 기하와 특징 점수의 결합만 대조한다."""
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
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash
from gnn.geometry_interaction import attach_geometry_interaction

base = source.base
OUT = REPO / 'results/pdv2_geometry_interaction_20261023'
CONFIG = REPO / 'configs/pdv2_geometry_interaction.yaml'
CFG = yaml.safe_load(CONFIG.read_text())
METHOD = CFG['final_score']


def model(arm: str) -> torch.nn.Module:
    """공통 초기 텐서와 매개변수 수를 유지한다."""
    assert arm in CFG['arms']
    net = source.previous.BASE_MODEL('diff')
    attach_geometry_interaction(net, arm)
    return net


def protocol(inputs: bool = False) -> dict:
    """고정한 실행 코드와 자료의 해시를 검사한다."""
    spec = base.read(OUT / 'protocol.json')
    for field in (('runtime', 'inputs') if inputs else ('runtime',)):
        for p, digest in spec[field].items():
            assert sha256(REPO / p) == digest, p
    return spec


def prepare() -> None:
    """학습 전에 가설·세 조건·동일 초기값·고정 에포크를 기록한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        protocol(True)
        return
    old = source.protocol(True)
    initial, counts = {}, {}
    for arm in CFG['arms']:
        torch.manual_seed(1)
        net = model(arm)
        initial[arm] = {k: array_hash(v.numpy()) for k, v in net.state_dict().items()}
        counts[arm] = sum(p.numel() for p in net.parameters())
    assert all(x == initial['additive'] for x in initial.values())
    assert set(counts.values()) == {167561}
    original = torch.load(base.GRAPH, map_location='cpu', weights_only=False)
    graph = base.temporal_subgraph(original, np.arange(original.num_nodes)).cuda()
    del original
    smoke = {}
    for arm in CFG['arms']:
        torch.manual_seed(1)
        net = model(arm).cuda().train()
        with torch.autocast('cuda', dtype=torch.float16):
            output = net(graph)
            loss = output.float().square().mean()
        loss.backward()
        assert output.shape == (72306, 416) and torch.isfinite(output).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        smoke[arm] = {'shape': list(output.shape), 'finite': True}
        del net, output, loss
    del graph
    torch.cuda.empty_cache()
    write_json(OUT / 'amp_smoke.json', smoke)
    runtime = dict(old['runtime'])
    for p in (CONFIG, Path(__file__), REPO / 'src/gnn/geometry_interaction.py',
              REPO / 'tests/gnn/test_geometry_interaction.py',
              REPO / 'scripts/_pdv2_geometry_interaction_run.py',
              REPO / 'scripts/_pdv2_geometry_interaction_finish.py'):
        runtime[str(p.relative_to(REPO))] = sha256(p)
    write_json(OUT / 'protocol.json', {'config': CFG, 'runtime': runtime,
        'inputs': old['inputs'], **{k: old[k] for k in ('graph_hashes', 'train', 'val', 'confidence')},
        'parameters': counts, 'initial_equal': True, 'initial': initial['additive'],
        'test_used': False,
        'scores': {'additive': 'c+b', 'interaction': 'c+b+(c-mean_in(c))*tanh(b-mean_in(b))',
                   'temperature': 'c+b+(c-mean_in(c))*tanh(mean_in(b))'},
        'hypothesis': '기하만의 독립 가산은 특징 차이에 따라 이웃의 기하를 다르게 사용할 수 없다. 이웃별 두 점수의 중심화 곱과 중심별 온도 대조를 구분한다. 계수1 고정이며 결과를 보고 조정하지 않는다.',
        'training': '모두 새 학습. 같은 초기텐서·GT학습8·183249삼중쌍·순서·20에포크900갱신·Adam5e-4·감쇠1e-5·AMP. no_interdiff288·gate초기.0625. 차이K/V-tanh·시간간선±5·2층·4head. 새학습매개변수0.',
        'evaluation': 'KISS검증9·99236질의·q+15·j<=q-30·기존재계량/융합/arc_unique_b100고정. 세모델모두후보부터재검색.20에포크고정이며선택없음. 시험/원고/추가시드승격없음.',
        'planned_usage': 'interaction20에포크에서추가곱만끔/기하전체끔을별도고정추론개입으로검사한다. 새학습과구분한다.'})


def report() -> None:
    """최종 판정 지표를 집계하고 학습 시작·표본 순서를 대조한다."""
    spec = protocol(True)
    results, reference_order = {}, None
    for arm in CFG['arms']:
        folder = base.root(1, arm)
        assert base.read(folder / 'initial_model.json') == spec['initial']
        complete = base.read(folder / 'training_complete.json')
        assert complete['updates'] == 900
        if reference_order is None:
            reference_order = complete['orders']
        assert complete['orders'] == reference_order
        rows = {s: base.read(folder / 'epoch20' / (s.replace('/', '_') + '.json')) for s in spec['val']}
        results[arm] = {m: base.macro({s: r['metrics'][m] for s, r in rows.items()})
                        for m in ('raw', METHOD)}
        print('FINAL', arm, results[arm][METHOD]['macro'], flush=True)
    write_json(OUT / 'summary.json', {'results': results,
        'minus_reference_pp': {a: {m: 100 * (results[a][METHOD]['macro'][m] -
            results['additive'][METHOD]['macro'][m]) for m in base.exp.METRICS}
            for a in CFG['arms'] if a != 'additive'},
        'initial_and_orders_equal': True, 'new_updates': 2700, 'test_used': False,
        'fixed_epoch': 20, 'protocol_sha256': sha256(OUT / 'protocol.json')})


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
        base.train(1, args.arm)
    elif args.action == 'val':
        source.validation(1, args.arm, 20, args.sequence)
    else:
        report()
