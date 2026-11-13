#!/usr/bin/env python3
"""선택된 차이+tanh 값 모델의 attention 성분을 동결 상태로 분리한다."""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_mixed_value as source
from _pdv2_no_context_ablation import sha256, write_json
from gnn.attention_probe import attach_attention_probe

base = source.base
OUT = REPO / 'results/pdv2_attention_review_20261023'
CONFIG = REPO / 'configs/pdv2_attention_review.yaml'
CFG = yaml.safe_load(CONFIG.read_text())
ARMS = tuple(CFG['arms'])
METHOD = CFG['final_score']
read = source.read


def model(arm: str) -> torch.nn.Module:
    """동일한 값 전달과 매개변수에서 attention 점수 성분만 지운다."""
    net = source.model(CFG['source_arm'])
    attach_attention_probe(net, arm)
    return net


def protocol(inputs: bool = False) -> dict:
    """진단 소스·선택 모델·입력을 확인한다."""
    spec = read(OUT / 'protocol.json')
    for field in (('runtime', 'inputs') if inputs else ('runtime',)):
        for path, expected in spec[field].items():
            assert sha256(REPO / path) == expected, path
    return spec


def prepare() -> None:
    """학습 없이 선택 가중치와 네 추론 진단을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        protocol(True)
        return
    old = source.protocol(True)
    previous = read(source.OUT / 'summary.json')
    chosen = previous['per_seed']['1'][CFG['source_arm']]['selected']['epoch']
    assert chosen == CFG['source_epoch'] == 20
    folder = source.OUT / 's1' / CFG['source_arm']
    checkpoint = folder / 'checkpoints' / f'checkpoint_epoch_{chosen}.pth'
    sources = dict(old['inputs'])
    for p in (source.OUT / 'summary.json', source.OUT / 'protocol.json',
              source.OUT / 'completion_validation.json', checkpoint, folder / 'training_complete.json'):
        sources[str(p.relative_to(REPO))] = sha256(p)
    for p in sorted((folder / f'epoch{chosen}').glob('*')):
        if p.is_file():
            sources[str(p.relative_to(REPO))] = sha256(p)
    for arm in ARMS:
        target = base.root(1, arm)
        (target / 'checkpoints').mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, target / 'checkpoints' / checkpoint.name)
        write_json(target / 'training_complete.json', {
            'checkpoints': {checkpoint.name: sha256(checkpoint)}, 'trained_in_this_experiment': False,
            'source_updates': 900, 'source': str(folder.relative_to(REPO))})
    runtime = dict(old['runtime'])
    for p in (CONFIG, Path(__file__), REPO / 'scripts/_pdv2_attention_review_run.py',
              REPO / 'scripts/_pdv2_attention_review_stats.py', REPO / 'src/gnn/attention_probe.py',
              REPO / 'tests/gnn/test_attention_probe.py'):
        runtime[str(p.relative_to(REPO))] = sha256(p)
    write_json(OUT / 'protocol.json', {
        'created_unix': time.time(), 'config': CFG, 'runtime': runtime, 'inputs': sources,
        'val': old['val'], 'train': old['train'], 'confidence': old['confidence'],
        'source_checkpoint': str(checkpoint.relative_to(REPO)), 'checkpoint_sha256': sha256(checkpoint),
        'test_used': False, 'training_performed': False,
        'scope': '시드1·검증AP로 이미 선택한 차이+tanh값20에포크 고정. 두층의 QK점수/간선편향만 분리. 값·잔차·BN·문맥게이트·후보예산·최종판정 유지. 추론 개입은 재학습 ablation이나 새로운 모델 채택 근거가 아니다. 시험과 재선택 없음.'})


def report() -> dict:
    """재학습 없이 고정된 네 조건의 최종 판정 성능을 비교한다."""
    spec = protocol()
    results = {}
    for arm in ARMS:
        rows = {seq: read(base.root(1, arm) / 'epoch20' / (seq.replace('/', '_') + '.json'))
                for seq in spec['val']}
        results[arm] = {method: base.macro({seq: row['metrics'][method] for seq, row in rows.items()})
                        for method in ('raw', METHOD)}
    deltas = {arm: {m: 100 * (results[arm][METHOD]['macro'][m] - results['full'][METHOD]['macro'][m])
                    for m in base.exp.METRICS} for arm in ARMS}
    summary = {'results': results, 'minus_full_pp': deltas, 'epoch': 20, 'seed': 1,
               'protocol_sha256': sha256(OUT / 'protocol.json'), 'training_performed': False,
               'test_used': False, 'promotion': None}
    write_json(OUT / 'summary.json', summary)
    for arm in ARMS:
        print('FINAL', arm, results[arm][METHOD]['macro'], flush=True)
    return summary


base.OUT, base.CONFIG, base.CFG, base.ARMS = OUT, CONFIG, CFG, ARMS
base.model, base.protocol = model, protocol

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'val', 'report'))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--sequence')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.action == 'prepare':
        prepare()
    elif args.action == 'val':
        source.validation(1, args.arm, 20, args.sequence)
    else:
        report()
