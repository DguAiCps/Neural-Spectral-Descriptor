#!/usr/bin/env python3
"""고정 후보·새 키에서 정규화 분모만 기존 키의 점수 폭으로 바꾸는 사후 진단."""
import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_fusion_factor as base_run
from _pdv2_fusion_factor_lib import fuse_with_range
from _pdv2_no_context_ablation import sha256, write_json

base = base_run.base
OUT = base_run.OUT / 'key_width_probe'
LOG = REPO / 'logs/fusion_key_width_20261023'


def protocol() -> dict:
    """사후 진단 실행 전에 고정한 코드와 입력을 확인한다."""
    data = base.read(OUT / 'protocol.json')
    for path, digest in data['sources'].items():
        assert sha256(REPO / path) == digest, path
    return data


def prepare() -> None:
    """주 진단 결과를 본 뒤 정한 단일 개입임을 기록한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        protocol()
        return
    source = base_run.protocol(True)
    audit = base.read(base_run.OUT / 'completion_validation.json')
    assert audit['passed']
    hashes = dict(audit['sources'])
    supplement = base.read(base_run.OUT / 'supplemental_sources.json')
    hashes.update(supplement['sources'])
    for path in (Path(__file__), base_run.OUT / 'completion_validation.json',
                 base_run.OUT / 'supplemental_sources.json'):
        hashes[str(path.relative_to(REPO))] = sha256(path)
    for path, digest in hashes.items():
        assert sha256(REPO / path) == digest, path
    write_json(OUT / 'protocol.json', {
        'sources': hashes, 'train': source['train'], 'val': source['val'],
        'posthoc': True, 'test_used': False, 'training_updates': 0,
        'intervention': 'P1 후보와 K1 점수를 유지한다. 키 정규화의 최솟값은 K1, '
                        '분모만 같은 P1에서 K0의 max-min으로 바꾼다. 다른 세 채널은 그대로다. '
                        '값을 [0,1]로 자르지 않는다. 융합 후 기존 최종 시간 판정을 계산한다.',
        'reason': 'KITTI360/0004 q8425는 키 쌍 점수 차이가 커져도 키 점수 폭의 증가로 '
                  '융합 순위가 뒤집힌다. 후보 목록 효과와 학습된 점수 폭의 효과를 구분한다.',
        'scope': '두 모델을 함께 쓰는 조건부 원인 진단이다. 새 방법·선택·독립 인과 기여가 아니다.',
    })


def replay(role: str, sequence: str) -> None:
    """한 시퀀스의 분모 교체와 최종 시간 판정을 검산한다."""
    spec = protocol()
    assert sequence in spec[role]
    folder = OUT / role / sequence.replace('/', '_')
    folder.mkdir(parents=True, exist_ok=True)
    parent = base_run.OUT / role / sequence.replace('/', '_')
    reference = base.exp.arrays(parent / 'new.npz')
    row = {k: v.copy() for k, v in reference.items()}
    paths = base_run.paths_for(role, sequence)
    cache = base.exp.arrays(paths['cache'])
    positions = cache['poses' if role == 'train' else 'poses_gt'][:, :3, 3].astype(np.float64)
    xyz = cache['poses'][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    keys = [base.exp.arrays(base_run.key_path(role, arm, sequence))['embedding']
            for arm in ('additive', 'interaction')]
    with np.load(paths['spec']) as z:
        bank = base.exp.old.base.bank_for(base.exp.OP, z)
    conf = base.read(base_run.OUT / 'protocol.json')['confidence']['1']
    weights = tuple(conf['tuple'])
    q = row['query_idx']
    n = len(q)
    width_old, width_new = np.empty(n), np.empty(n)
    scalar_checks, prefix_checks, cases = [], [], []
    parent_cases = {c['q']: c for c in base.read(parent / 'diagnosis.json')['changed_correctness']}
    for i, query in enumerate(q):
        t = int(query)
        pool = np.unique(np.r_[base.prefix_topk(keys[1], t, t - 29, 100),
                               base.prefix_topk(bank[0], t, t - 29, 100)])
        channels = np.column_stack((keys[1][pool] @ keys[1][t], bank[0][pool] @ bank[0][t],
            base.align_scores(bank[1], bank[2], t, pool), base.column_align_scores(bank[3], t, pool)))
        scores, ranges = fuse_with_range(channels, weights)
        assert int(pool[np.argmax(scores)]) == reference['j_star'][i], (role, sequence, t)
        old_scores = keys[0][pool] @ keys[0][t]
        old_width = max(float(old_scores.max()) - float(old_scores.min()), 1e-8)
        width_old[i], width_new[i] = old_width, ranges[0, 2]
        pieces = [float(w) * ((channels[:, c] - float(ranges[c, 0])) /
                  (old_width if c == 0 else float(ranges[c, 2]))) for c, w in enumerate(weights)]
        score = sum(pieces)
        index = int(np.argmax(score))
        j = int(pool[index])
        row['j_star'][i] = j
        row['top1_geo_dist'][i] = np.linalg.norm(positions[t] - positions[j])
        row['raw'][i] = channels[index, 3 if conf['decision'] == 'c_col' else 2]
        if t in parent_cases:
            case = parent_cases[t]
            cases.append({'q': t, 'change': case['change'],
                'correct': bool(row['top1_geo_dist'][i] <= 5), 'winner': j,
                'old_key_width_on_new_pool': old_width, 'new_key_width_on_new_pool': width_new[i]})
        if i in (0, n // 3, 2 * n // 3, n - 1):
            direct = np.array([sum(float(w) * (float(x[c]) - float(ranges[c, 0])) /
                (old_width if c == 0 else float(ranges[c, 2])) for c, w in enumerate(weights))
                for x in channels])
            error = float(abs(direct - score).max())
            assert error < 2e-6 and int(pool[np.argmax(direct)]) == j
            scalar_checks.append({'q': t, 'maximum_error': error})
    changed = np.flatnonzero(row['j_star'] != reference['j_star'])
    for start in range(0, len(changed), 256):
        ids = changed[start:start + 256]
        row[base_run.METHOD][ids] = base.fixed.frozen_scores(bank, travel, q[ids], row['j_star'][ids],
            conf['decision'], row['raw'][ids])[base_run.METHOD]
    check_ids = np.unique(np.r_[n // 3, 2 * n // 3, changed[:1], changed[-1:]])
    for i in check_ids:
        t, j = int(q[i]), int(row['j_star'][i])
        direct = base_run.scalar_score(bank, travel, t, j, conf['decision'], float(row['raw'][i]))
        short = base.fixed.frozen_scores(base.exp.truncate_bank(bank, t + 16), travel[:t + 16],
            q[i:i + 1], row['j_star'][i:i + 1], conf['decision'], row['raw'][i:i + 1])[base_run.METHOD][0]
        error = max(abs(short - row[base_run.METHOD][i]), abs(direct - row[base_run.METHOD][i]))
        assert error < 1e-7
        prefix_checks.append({'q': t, 'maximum_error': float(error)})
    row['old_key_width_on_new_pool'] = width_old
    row['new_key_width_on_new_pool'] = width_new
    path = folder / 'ledger.npz'
    np.savez_compressed(path, **row)
    positive = row['min_past_gt_dist'] <= 5
    before, after = reference['top1_geo_dist'] <= 5, row['top1_geo_dist'] <= 5
    write_json(folder / 'diagnosis.json', {'role': role, 'sequence': sequence,
        'queries': n, 'positive': int(positive.sum()), 'changed_winners': len(changed),
        'gains_vs_new': int((positive & ~before & after).sum()),
        'losses_vs_new': int((positive & before & ~after).sum()),
        'parent_changed_cases': cases, 'scalar_checks': scalar_checks, 'prefix_checks': prefix_checks,
        'metrics': base_run.independent_metrics(row, row[base_run.METHOD]), 'sha256': sha256(path),
        'test_used': False, 'posthoc': True})


def finish() -> None:
    """독립 지표 계산과 기존 계산을 대조하고 계열별로 집계한다."""
    spec = protocol()
    sources = dict(spec['sources'])
    results, diagnostics = {}, {}
    maximum_error, count, rows = 0., 0, 0
    for role in ('train', 'val'):
        per_sequence, diagnostics[role] = {}, {}
        for sequence in spec[role]:
            folder = OUT / role / sequence.replace('/', '_')
            report = base.read(folder / 'diagnosis.json')
            path = folder / 'ledger.npz'
            assert sha256(path) == report['sha256']
            row = base.exp.arrays(path)
            assert np.all(row['j_star'] <= row['query_idx'] - 30)
            assert all(np.isfinite(v).all() for v in row.values())
            actual = base.exp.metrics(row[base_run.METHOD], row)
            for k in base.exp.METRICS:
                if actual[k] is not None:
                    error = abs(actual[k] - report['metrics'][k])
                    maximum_error = max(maximum_error, error)
                    assert error < 5e-13
                else:
                    assert report['metrics'][k] is None
            per_sequence[sequence] = report['metrics']
            diagnostics[role][sequence] = report
            count += 1
            rows += len(row['query_idx'])
            for p in (path, folder / 'diagnosis.json'):
                sources[str(p.relative_to(REPO))] = sha256(p)
        results[role] = base.macro(per_sequence)
    write_json(OUT / 'summary.json', {'results': results, 'diagnostics': diagnostics,
        'posthoc': True, 'test_used': False, 'training_updates': 0})
    for path in (OUT / 'protocol.json', OUT / 'summary.json'):
        sources[str(path.relative_to(REPO))] = sha256(path)
    write_json(OUT / 'completion_validation.json', {'passed': True, 'sources': sources,
        'ledgers': count, 'rows': rows, 'maximum_metric_error': maximum_error})


def main() -> None:
    """고정 후 독립 시퀀스를 두 개씩 처리한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=('train', 'val'))
    parser.add_argument('--sequence')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.role:
        replay(args.role, args.sequence)
        return
    prepare()
    LOG.mkdir(parents=True, exist_ok=True)
    spec = protocol()
    started = time.time()
    def worker(job):
        role, sequence = job
        path = LOG / (role + '_' + sequence.replace('/', '_') + '.log')
        with path.open('w') as log:
            done = subprocess.run([sys.executable, '-u', str(Path(__file__)), '--role', role,
                '--sequence', sequence], cwd=REPO, env=os.environ, stdout=log, stderr=subprocess.STDOUT)
        if done.returncode:
            raise RuntimeError(path.read_text()[-4000:])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [(role, sequence) for role in ('train', 'val') for sequence in spec[role]]
        futures = [pool.submit(worker, job) for job in jobs]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            print('KEY_WIDTH_DONE', i, len(jobs), flush=True)
    finish()
    print('KEY_WIDTH_COMPLETE', round(time.time() - started, 1), flush=True)


if __name__ == '__main__':
    main()
