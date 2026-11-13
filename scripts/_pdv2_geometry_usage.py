#!/usr/bin/env python3
"""감쇠 제거로 커진 기하 분기의 실제 추론 기여를 고정 모델에서 검사한다."""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_geometry_decay as previous
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_temporal_test_control_finish import scalar_score
from gnn.attention_probe import attach_attention_probe

base = previous.base
OUT = REPO / 'results/pdv2_geometry_usage_20261023'
ARMS = ('full', 'content')


def protocol(inputs: bool = False) -> dict:
    """동결한 모델과 평가 입력을 확인한다."""
    spec = base.read(OUT / 'protocol.json')
    for field in (('runtime', 'inputs') if inputs else ('runtime',)):
        for p, h in spec[field].items():
            assert sha256(REPO / p) == h, p
    return spec


def model(arm: str) -> torch.nn.Module:
    """같은 학습된 값 전달·잔차·정규화에서 기하 편향만 제거한다."""
    net = previous.source.model('diff_tanh')
    attach_attention_probe(net, arm)
    return net


def prepare() -> None:
    """새 모델 선택 없이 감쇠 제거 모델의20에포크를 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        protocol(True)
        return
    old = previous.protocol(True)
    reference = previous.OUT / 's1/diff_tanh_nodecay'
    checkpoint = reference / 'checkpoints/checkpoint_epoch_20.pth'
    runtime, inputs = dict(old['runtime']), dict(old['inputs'])
    for p in (Path(__file__), REPO / 'src/gnn/attention_probe.py', REPO / 'tests/gnn/test_attention_probe.py'):
        runtime[str(p.relative_to(REPO))] = sha256(p)
    for p in (previous.OUT / 'summary.json', previous.OUT / 'completion_validation.json',
              checkpoint, reference / 'training_complete.json'):
        inputs[str(p.relative_to(REPO))] = sha256(p)
    for arm in ARMS:
        target = base.root(1, arm)
        (target / 'checkpoints').mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, target / 'checkpoints' / checkpoint.name)
        write_json(target / 'training_complete.json', {'updates': 900,
            'checkpoints': {checkpoint.name: sha256(checkpoint)}, 'training_performed': False})
    (base.root(1, 'full') / 'epoch20').mkdir(exist_ok=True)
    for p in sorted((reference / 'epoch20').glob('*')):
        if p.is_file():
            inputs[str(p.relative_to(REPO))] = sha256(p)
            shutil.copy2(p, base.root(1, 'full') / 'epoch20' / p.name)
    write_json(OUT / 'protocol.json', {'runtime': runtime, 'inputs': inputs,
        **{k: old[k] for k in ('val', 'train', 'confidence')}, 'test_used': False,
        'checkpoint_sha256': sha256(checkpoint), 'training_performed': False,
        'scope': '동결된 시드1·감쇠제거20에포크의 두층 기하편향만0으로 만든 추론 개입. full은 검산한 원장을 그대로 복사하고 content만 재검색한다. 재학습 ablation 또는 원고 채택 근거가 아니다.'})


def report() -> None:
    """최종 지표와 후보 판정을 독립 계산하고 두 조건을 비교한다."""
    spec = protocol(True)
    sources = {**spec['runtime'], **spec['inputs']}
    results, states = {}, {}
    errors = {'metric': 0., 'score': 0., 'prefix_key': 0.}
    ledgers = total_rows = metric_sets = scalar_checks = prefixes = 0
    for arm in ARMS:
        measured, states[arm] = {m: {} for m in ('raw', previous.METHOD)}, {}
        folder = base.root(1, arm)
        assert sha256(folder / 'checkpoints/checkpoint_epoch_20.pth') == spec['checkpoint_sha256']
        for seq in spec['val']:
            p = folder / 'epoch20' / (seq.replace('/', '_') + '.npz')
            info = base.read(p.with_suffix('.json'))
            kp = p.with_name('keys_' + p.name)
            cp = p.with_name('coverage_' + p.name)
            for path, digest in ((p, info['sha256']), (kp, info['keys_sha256']),
                (cp, base.read(cp.with_suffix('.json'))['sha256'])):
                assert sha256(path) == digest
                sources[str(path.relative_to(REPO))] = digest
            row, keys, coverage = [base.exp.arrays(path) for path in (p, kp, cp)]
            q, j, key = row['query_idx'], row['j_star'], keys['embedding']
            np.testing.assert_array_equal(q, np.arange(30, len(key) - 15))
            np.testing.assert_array_equal(j, coverage['j_star'])
            assert np.all(j <= q - 30)
            _, _, paths, _ = base.exp.old.base.inputs('val', seq)
            cache = base.exp.arrays(paths['cache'])
            np.testing.assert_allclose(row['top1_geo_dist'], np.linalg.norm(
                cache['poses_gt'][q, :3, 3] - cache['poses_gt'][j, :3, 3], axis=1), atol=1e-10)
            for method in measured:
                values = independent_metrics(row, row[method])
                measured[method][seq] = values
                for metric, value in values.items():
                    expected = info['metrics'][method][metric]
                    if value is None:
                        assert expected is None
                    else:
                        errors['metric'] = max(errors['metric'], abs(value - expected))
                        assert abs(value - expected) < 5e-13
                metric_sets += 1
            with np.load(paths['spec']) as z:
                bank = base.exp.old.base.bank_for(base.exp.OP, z)
            xyz = cache['poses'][:, :3, 3]
            travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
            conf = spec['confidence']['1']
            for ix in (0, len(q) // 3, 2 * len(q) // 3, len(q) - 1):
                t = int(q[ix])
                ids = np.unique(np.r_[base.prefix_topk(key, t, t - 29, 100), base.prefix_topk(bank[0], t, t - 29, 100)])
                channels = (key[ids] @ key[t], bank[0][ids] @ bank[0][t],
                    base.align_scores(bank[1], bank[2], t, ids), base.column_align_scores(bank[3], t, ids))
                fused = sum(float(w) * base.minmax(v) for w, v in zip(conf['tuple'], channels))
                winner = int(ids[np.argsort(-fused, kind='stable')[0]])
                assert winner == j[ix]
                raw = float(channels[2 if conf['decision'] == 'c_spec' else 3][np.flatnonzero(ids == winner)[0]])
                assert abs(raw - row['raw'][ix]) < 1e-12
                value = scalar_score(bank, travel, t, winner, conf['decision'], raw)
                errors['score'] = max(errors['score'], abs(value - row[previous.METHOD][ix]))
                assert errors['score'] < 1e-12
                scalar_checks += 1
            for prefix in info['prefixes']:
                assert prefix['winner_equal'] and prefix['score_error'] == 0.
                errors['prefix_key'] = max(errors['prefix_key'], prefix['key_max_error'])
                assert prefix['key_max_error'] < 4e-6
                prefixes += 1
            positive = row['min_past_gt_dist'] <= 5
            state = np.where(row['top1_geo_dist'] <= 5, 2, np.where(coverage['union_hit'], 1, 0))
            states[arm][seq] = {'q': q, 'positive': positive, 'state': state}
            ledgers += 1
            total_rows += len(q)
        results[arm] = {m: base.macro(rows) for m, rows in measured.items()}
    transition = np.zeros((3, 3), np.int64)
    for seq in spec['val']:
        a, b = states['full'][seq], states['content'][seq]
        np.testing.assert_array_equal(a['q'], b['q'])
        np.testing.assert_array_equal(a['positive'], b['positive'])
        mask = a['positive']
        transition += np.bincount(3 * a['state'][mask] + b['state'][mask], minlength=9).reshape(3, 3)
    write_json(OUT / 'summary.json', {'results': results,
        'content_minus_full_pp': {k: 100 * (results['content'][previous.METHOD]['macro'][k] -
            results['full'][previous.METHOD]['macro'][k]) for k in base.exp.METRICS},
        'transition_full_to_content': transition.tolist(), 'test_used': False, 'training_performed': False})
    for p in (OUT / 'summary.json', OUT / 'protocol.json'):
        sources[str(p.relative_to(REPO))] = sha256(p)
    for p, h in sources.items():
        assert sha256(REPO / p) == h, p
    write_json(OUT / 'completion_validation.json', {'passed': True, 'sources': sources,
        'ledgers': ledgers, 'rows': total_rows, 'metric_sets': metric_sets, 'scalar_checks': scalar_checks,
        'prefixes': prefixes, 'maximum_errors': errors, 'test_used': False})
    for arm, result in results.items():
        print('FINAL_USAGE', arm, result[previous.METHOD]['macro'], flush=True)


base.OUT, base.model, base.protocol = OUT, model, protocol

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'val', 'report'))
    parser.add_argument('--sequence')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.action == 'prepare':
        prepare()
    elif args.action == 'val':
        previous.source.validation(1, 'content', 20, args.sequence)
    else:
        report()
