#!/usr/bin/env python3
"""학습 자료에서 고정한 문맥 세기로 DiffAttnConv 방향 학습을 대조한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import _pdv2_diffattn_alignment_teacher as exp

ROOT = exp.ROOT
PARENT = exp.OUT
OUT = ROOT / 'results/pdv2_diffattn_fixed_gate_teacher_20261104'
ARMS = {'teacher_lr5': ('teacher', 1e-5), 'teacher_lr4': ('teacher', 1e-4)}


def initialize() -> None:
    """초기 가중치·상수 세기·점수 척도를 학습8에서만 정한다."""
    if (OUT / 'protocol.json').exists():
        prepare()
        return
    spec = exp.activate().copy()
    OUT.mkdir(exist_ok=True)
    net = exp.model()
    means, sources = [], {}
    for sequence in spec['train']:
        path = exp.PARENT / 'pools' / sequence.replace('/', '_') / 'graph.pt'
        graph = torch.load(path, map_location='cuda', weights_only=False)
        with torch.no_grad():
            net(graph)
        means.append(float(net.gnn._last_alpha.mean()))
        sources[str(path.relative_to(ROOT))] = exp.run.HASH(path)
        del graph
    alpha = float(np.mean(means))
    with torch.no_grad():
        net.gnn.gate[-1].weight.zero_()
        net.gnn.gate[-1].bias.fill_(math.log(alpha / (1 - alpha)))
    initial = OUT / 'frozen/epoch0.pth'
    initial.parent.mkdir(exist_ok=True)
    torch.save(dict(model_state_dict=net.state_dict(), epoch=0, updates=0, alpha=alpha), initial)
    values, weights = [], []
    for sequence in spec['train']:
        folder = exp.PARENT / 'pools' / sequence.replace('/', '_')
        graph = torch.load(folder / 'graph.pt', map_location='cuda', weights_only=False)
        arrays = exp.run.ARR(folder / 'pool.npz')
        plan = {k: torch.from_numpy(arrays['plan_' + k]).cuda() for k in ('query', 'candidate', 'weight')}
        with torch.no_grad():
            key = F.normalize(net(graph), dim=1)
            torch.testing.assert_close(net.gnn._last_alpha, torch.full_like(net.gnn._last_alpha, alpha), atol=1e-6, rtol=0)
            context = torch.cat([exp.gated_context_score(key, {k: v[i:i+256] for k, v in plan.items()})
                                 for i in range(0, len(plan['query']), 256)]).cpu().numpy().astype(np.float64)
        values.append(context)
        weights.append(np.full(len(context), 1. / len(context)))
        sources[str((folder / 'pool.npz').relative_to(ROOT))] = exp.run.HASH(folder / 'pool.npz')
        del graph, key, plan
    c, w = np.concatenate(values), np.concatenate(weights)
    w /= w.sum()
    mean = float(w @ c)
    scale = max(float(np.sqrt(w @ ((c - mean) ** 2))), .05)
    fit = exp.run.READ(PARENT / 'calibration.json')
    fit.update(context_mean=mean, context_scale=scale, sources=sources, test_used=False)
    exp.run.WRITE(OUT / 'calibration.json', fit)
    sources = dict(spec['sources'], **sources)
    for path in (Path(__file__), PARENT / 'summary.json', initial, OUT / 'calibration.json'):
        sources[str(path.relative_to(ROOT))] = exp.run.HASH(path)
    spec.update(created_unix=time.time(), initial=str(initial.relative_to(ROOT)), arms=ARMS,
        alpha=alpha, alpha_sequence_means=dict(zip(spec['train'], means)),
        architecture='기존 두 DiffAttnConv 층·차원 유지. 게이트 마지막 가중치0·절편logit(학습8 시퀀스 균등 평균alpha)로 노드별 세기를 상수화. 같은 상수의 동결 대조. 추론 추가 모듈 없음. 다른 가중치 고정.',
        selection='같은 상수 동결 대비 AP+.1%p·키R1+.2%p·정렬전융합R1손실<=.1%p·F1손실<=.3%p·R100손실<=1%p. 동적게이트 동결 대비 AP손실<=.3%p·키R1손실<=.1%p·R100손실<=1%p. BEV동일시간검증대비AP/F1열세2%p·R100/R1열세3%p이내. 통과중AP최대,작은lr,이른epoch. 단일시드탐색·3시드확인전시험없음.',
        calibration='학습8 원래고정승자에서 새 상수동결의 문맥평균/척도만 적합. 스펙트럼 affine,문맥비중.5 유지. 모든학습조건이같은척도사용.',
        limitation='동적게이트 학습의 실패를 본 뒤 정한 후속 가설. 세기 고정 자체와 이후 학습의 효과를 구분. 추가학습전도학습된 모델. 시험·원고·그림 유지.',
        sources=sources, test_used=False)
    exp.run.WRITE(OUT / 'protocol.json', spec)
    print('INITIAL', alpha, mean, scale, flush=True)


def prepare() -> dict:
    """고정 기록의 소스와 시작 모델 무결성을 확인한다."""
    spec = exp.run.READ(OUT / 'protocol.json')
    for name, digest in spec['sources'].items():
        assert exp.run.HASH(ROOT / name) == digest, name
    return spec


def install() -> dict:
    """같은 검색·학습·판정 엔진에 독립 실험 경로를 공급한다."""
    exp.OUT, exp.ARMS, exp.prepare = OUT, ARMS, prepare
    return exp.activate()


def collect(arm: str, epoch: int, spec: dict) -> dict:
    """9시퀀스의 최종 지표와 중간 회수율을 계열 균등 평균한다."""
    rows, groups, spectral = {}, {}, {}
    for sequence in spec['val']:
        path = OUT / arm / f'epoch{epoch}/val' / (sequence.replace('/', '_') + '.json')
        record = exp.run.READ(path)
        assert record['status'] == 'passed' and len(record['prefix_checks']) == 2
        assert exp.run.HASH(path.with_suffix('.npz')) == record['ledger_sha256']
        rows[sequence] = record['metrics']['score']
        spectral[sequence] = record['metrics']['spectral']
        if rows[sequence]['n_pos']:
            g = 'kitti' if sequence.startswith('kitti') else sequence.split('/')[0]
            groups.setdefault(g, []).append(record['intermediate'])
    grouped = {g: {m: float(np.mean([r[m] for r in values])) for m in values[0]}
               for g, values in groups.items()}
    mid = {m: float(np.mean([r[m] for r in grouped.values()])) for m in next(iter(grouped.values()))}
    return dict(arm=arm, epoch=epoch, **exp.run.base.macro(rows), sequences=rows,
                intermediate=mid, intermediate_groups=grouped, spectral=exp.run.base.macro(spectral))


def finish() -> None:
    """동결 세기 대조와 원래 동적 게이트를 함께 보존 기준으로 확인한다."""
    spec = install()
    original = exp.run.READ(PARENT / 'summary.json')
    baseline = collect('frozen', 0, spec)
    results, eligible = {}, []
    sample_orders = []
    for arm in ARMS:
        train = exp.run.READ(OUT / arm / 'training_complete.json')
        assert train['updates'] == 448 and train['status'] == 'passed'
        assert all(n.startswith('gnn.convs.') for n in train['changed_parameters'])
        sample_orders.append([r['batch_hashes'] for r in train['records']])
        for epoch in exp.EPOCHS:
            net = exp.model(OUT / arm / f'epoch{epoch}.pth')
            assert torch.count_nonzero(net.gnn.gate[-1].weight) == 0
            assert abs(float(net.gnn.gate[-1].bias.sigmoid()) - spec['alpha']) < 1e-6
            result = collect(arm, epoch, spec)
            delta = {m: 100 * (result['macro'][m] - baseline['macro'][m]) for m in exp.run.base.METRICS}
            mid = result['intermediate']
            checks = dict(AP_gain=delta['AP'] >= .1,
                key_gain=mid['key_R1'] >= baseline['intermediate']['key_R1'] + .002,
                pre_preserved=mid['pre_R1'] >= baseline['intermediate']['pre_R1'] - .001,
                F1_preserved=delta['max_F1'] >= -.3, R100_preserved=delta['R@100P'] >= -1.,
                original_AP=result['macro']['AP'] >= original['frozen']['AP'] - .003,
                original_key=mid['key_R1'] >= original['frozen_intermediate']['R1'] - .001,
                original_R100=result['macro']['R@100P'] >= original['frozen']['R@100P'] - .01)
            checks.update({f'bev_{m}': result['macro'][m] >= original['bev_temporal'][m] - tolerance
                           for m, tolerance in {'AP': .02, 'max_F1': .02, 'R@100P': .03, 'R@1': .03}.items()})
            result.update(delta_pp=delta, advancement_checks=checks)
            results[f'{arm}_{epoch}'] = result
            if all(checks.values()):
                eligible.append(result)
    assert sample_orders[0] == sample_orders[1]
    selected = min(eligible, key=lambda r: (-round(r['macro']['AP'], 12), ARMS[r['arm']][1], r['epoch'])) if eligible else None
    exp.run.WRITE(OUT / 'summary.json', dict(results=results, frozen=baseline, dynamic_frozen=original['frozen'],
        bev_temporal=original['bev_temporal'], selected=selected, test_used=False, protocol_sha256=exp.run.HASH(OUT / 'protocol.json')))
    exp.run.WRITE(OUT / 'completion_validation.json', dict(status='passed', new_ledgers=45, prefix_checks=90,
        trained_models=2, updates_per_model=448, equal_training_queries=True, fixed_gate_verified=True, test_used=False))
    exp.run.WRITE(OUT / 'status.json', dict(status='validation_complete', selected=selected, test_used=False))
    print('FINISHED', selected, flush=True)


def campaign() -> None:
    """상수 세기의 두 학습률과 동결 대조를 끝까지 계산한다."""
    initialize()
    spec = install()
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=True)
    def call(args: list, name: str) -> None:
        with (logs / (name + '.log')).open('w') as stream:
            subprocess.run([sys.executable, __file__, *args], stdout=stream, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(call, ['mine', '--sequence', s], 'mine_' + s.replace('/', '_')) for s in spec['train']]
        for f in concurrent.futures.as_completed(tasks):
            f.result()
    for arm in ARMS:
        exp.run.WRITE(OUT / 'status.json', dict(status='running', phase='train', arm=arm))
        call(['train', '--arm', arm], 'train_' + arm)
    jobs = [('frozen', 0, s) for s in spec['val']]
    jobs += [(a, e, s) for a in ARMS for e in exp.EPOCHS for s in spec['val']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(call, ['val', '--arm', a, '--epoch', str(e), '--sequence', s],
                            f'val_{a}_{e}_{s.replace("/", "_")}') for a, e, s in jobs]
        for done, f in enumerate(concurrent.futures.as_completed(tasks), 1):
            f.result()
            exp.run.WRITE(OUT / 'status.json', dict(status='running', phase='validation', done=done, total=len(jobs)))
            print('PROGRESS', done, len(jobs), flush=True)
    finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'mine', 'train', 'val', 'finish'))
    parser.add_argument('--arm', default='teacher_lr5')
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('--sequence', default='kitti/06')
    args = parser.parse_args()
    exp.run.configure()
    if args.action == 'run':
        campaign()
    else:
        install()
        if args.action == 'mine': exp.mine(args.sequence, OUT / 'pools' / args.sequence.replace('/', '_'))
        elif args.action == 'train': exp.train(args.arm)
        elif args.action == 'val': exp.validate(args.arm, args.epoch, args.sequence)
        else: finish()
