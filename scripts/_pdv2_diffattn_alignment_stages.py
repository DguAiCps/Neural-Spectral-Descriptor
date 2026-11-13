#!/usr/bin/env python3
"""고정된 추가 학습 모델의 중간 검색과 순환 정렬 의존도를 측정한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts')]
import _pdv2_diffattn_gate_seeds as seeds
import _pdv2_diffattn_comparable_test as engine
import _pdv2_alignment_budget as budget
from retrieval.spectral_evidence import correlation_profiles
from retrieval.sequence_evidence import neighbor_plans, path_means, repeated_candidate_weights
from retrieval.gated_context import gated_context_score

run = seeds.run
OUT = ROOT / 'results/pdv2_diffattn_alignment_stages_20261104'
MODELS = ('frozen', 'convs', 'whole')
ANGLES = (60, 12, 4, 1)
CONDITIONS = {'full': (60, 200), 'angle12': (12, 200),
              'angle4': (4, 200), 'angle1': (1, 200),
              'budget20': (60, 20), 'budget5': (60, 5), 'budget1': (60, 1)}


def prepare() -> dict:
    """결과 전에 모델·열화 수준·집계와 시험 범위를 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        spec = run.READ(path)
        sources = dict(spec['sources'])
        amendment = OUT / 'runtime_amendments.json'
        if amendment.exists():
            for change in run.READ(amendment)['changes']:
                assert sources[change['path']] == change['before']
                sources[change['path']] = change['after']
        for name, digest in sources.items():
            assert run.HASH(ROOT / name) == digest, name
        return spec
    common = run.READ(seeds.OUT / 'protocol.json')
    source = [Path(__file__), Path(run.__file__), Path(engine.__file__),
              ROOT / 'src/retrieval/spectral_evidence.py',
              ROOT / 'src/retrieval/sequence_evidence.py',
              ROOT / 'src/retrieval/gated_context.py',
              seeds.OUT / 'protocol.json', seeds.OUT / 'summary.json']
    models = {}
    for seed in (1, 2, 3):
        parent = seeds.PARENT if seed == 1 else seeds.OUT / f's{seed}'
        whole = seeds.BALANCE if seed == 1 else parent
        models[str(seed)] = dict(root=str(parent.relative_to(ROOT)),
            frozen=str((parent / 'recenter_frozen/epoch0.pth').relative_to(ROOT)),
            convs=str((parent / 'recenter_convs/epoch8.pth').relative_to(ROOT)),
            whole=str((whole / 'recenter_balanced/epoch4.pth').relative_to(ROOT)))
        source += [ROOT / models[str(seed)][m] for m in MODELS]
        source += [parent / 'calibration_recenter.json', parent / 'protocol.json']
    spec = dict(created_unix=time.time(), models=models, seeds=[1, 2, 3],
        val=run.READ(seeds.PARENT / 'protocol.json')['val'],
        test=['/'.join(s) for s in run.current.TEST], confidence=common['confidence'],
        conditions=CONDITIONS, test_conditions=['full', 'angle12'],
        angle_rule='60개 6도 회전 격자에서 등간격 12개(30도),4개(90도),1개(0도)만 허용. 후보 순위와 원본/보정·중심/시간 이웃의 모든 스펙트럼에 동일 적용.',
        budget_rule='100+100 합집합의 키/보조 minmax 가중합에서 상위20/5/1개만 정렬. 같은 목록 내 네 점수 minmax를 다시 계산. 마지막 시간 정렬은60각도 유지.',
        stages='키 코사인 검색 AP/R1/R5/R20/R100, 합집합 정답 포함률, 정렬 전 키+보조 승자, 정렬 후 단일 중심 스펙트럼, 시간 스펙트럼, 최종 신경 점수 가산. 정렬 완전 배제는 정렬 전 승자+전체 키 시간 코사인.',
        fixed='체크포인트·재계량·기존 시드별 aux_half 융합·학습8 점수 척도·문맥계수0.5 고정. 재학습/검증 재선택 없음. 각 열화에서 세 모델 동일 조건. 각도 축소는 실제 계산시간 절감 측정 아님.',
        limitation='열화된 점수의 척도/융합을 재적합하지 않은 고정 모델 민감도 진단이다. 스펙트럼 단독과 최종 점수를 함께 보고 척도 교란을 구분한다. 학습 전도 이미 학습된 모델이며 추가 학습 효과만 비교한다. 검증/시험은 기존 개발 노출 자료다.',
        scope='검증9·3시드에서 모든 조건. 시험46·3시드는 사전 고정 full/angle12와 중간 단계만 확인하며 검증 결과로 수준을 고르지 않는다. 원고/그림 변경 없음.',
        sources={str(p.relative_to(ROOT)): run.HASH(p) for p in source})
    run.WRITE(path, spec)
    return spec


def activate(seed: int, spec: dict) -> tuple:
    """시드별 고정 입력·점수 척도를 공급한다."""
    run.OUT = ROOT / spec['models'][str(seed)]['root']
    run.context_score = gated_context_score
    fit = run.READ(run.OUT / 'calibration_recenter.json')
    weights = run.cal.fusion_weights(spec['confidence'][str(seed)])[run.cal.FUSIONS.index('aux_half')]
    return fit, weights


def query_search(key: np.ndarray, bank: tuple, positions: np.ndarray, q: int,
                 weights: np.ndarray, conditions: dict) -> tuple:
    """같은 후보에서 각도·후보 예산만 바꿔 승자를 구한다."""
    ka = run.base.prefix_topk(key, q, q - 29, 100)
    aa = run.base.prefix_topk(bank[0], q, q - 29, 100)
    ids = np.unique(np.r_[ka, aa])
    kv, av = key[ids] @ key[q], bank[0][ids] @ bank[0][q]
    pre = weights[0] * run.base.minmax(kv) + weights[1] * run.base.minmax(av)
    order = np.lexsort((ids, -pre))
    qp = np.full(len(ids), q, np.int64)
    profiles = [correlation_profiles(bank, qp, ids, ch) for ch in ('c_spec', 'c_col')]
    winners = {'pre': int(ids[np.argmax(pre)])}
    ranking = ka[np.lexsort((ka, -(key[ka] @ key[q])))]
    winners['key'] = int(ranking[0])
    distances = np.linalg.norm(positions[ranking] - positions[q], axis=1)
    hits = [bool(np.any(distances[:k] <= 5)) for k in (1, 5, 20, 100)]
    hits.append(bool(np.any(np.linalg.norm(positions[ids] - positions[q], axis=1) <= 5)))
    for name, (angle, limit) in conditions.items():
        ix = np.arange(len(ids)) if limit == 200 else np.sort(order[:limit])
        values = [kv[ix], av[ix], *(p[ix, ::60 // angle].max(1) for p in profiles)]
        score = sum(weights[i] * run.base.minmax(v) for i, v in enumerate(values))
        winners[name] = int(ids[ix[np.argmax(score)]])
    return winners, hits, float(key[winners['key']] @ key[q]), float(pre.max())


def spectral_stages(banks: tuple, travel: np.ndarray, q: np.ndarray,
                    j: np.ndarray, angles: tuple) -> dict:
    """같은 회전 제한을 단일 중심과 시간 판정 모두에 적용한다."""
    output = {a: {'center': [], 'temporal': []} for a in angles}
    n = len(travel)
    beta = (2. / 3.) * (11. / 10.)
    for start in range(0, len(q), 128):
        qq, jj = q[start:start + 128], j[start:start + 128]
        qn, jn, valid = neighbor_plans(travel, qq, jj)['arc_unique']
        unique, inverse = np.unique((qn * n + jn)[valid], return_inverse=True)
        repeated = repeated_candidate_weights(jn, valid)
        channels = {a: {} for a in angles}
        for channel in ('c_spec', 'c_col'):
            for bi, bank in enumerate(banks):
                cp = correlation_profiles(bank, qq, jj, channel)
                npf = correlation_profiles(bank, unique // n, unique % n, channel)
                for a in angles:
                    center = cp[:, ::60 // a].max(1)
                    peaks = npf[:, ::60 // a].max(1)
                    values = np.zeros(qn.shape, dtype=peaks.dtype)
                    values[valid] = peaks[inverse]
                    means = path_means(center, values, valid, repeated)
                    raw = center.astype(np.float64) if bi == 0 else center
                    channels[a][channel, bi] = (raw, (1 - beta) * raw + beta * means.max(0))
        for a in angles:
            for si, stage in enumerate(('center', 'temporal')):
                c = channels[a]
                mixed = run.cal.confidence(.75 * c['c_spec', 0][si] + .25 * c['c_spec', 1][si],
                    .75 * c['c_col', 0][si] + .25 * c['c_col', 1][si], 'equal', 'c_col')
                output[a][stage].append(mixed)
    return {a: {s: np.concatenate(v) for s, v in stages.items()} for a, stages in output.items()}


def previous_ledger(seed: int, role: str, sequence: str, model: str, spec: dict) -> dict:
    """원래60각도 조건을 검산할 보존 원장을 읽는다."""
    name = sequence.replace('/', '_')
    if role == 'val':
        checkpoint = ROOT / spec['models'][str(seed)][model]
        path = checkpoint.with_suffix('') / 'val' / (name + '.npz')
    else:
        folder = 'convs_fixed_test' if model == 'convs' else 'fixed_test'
        label = 'frozen' if model == 'frozen' else 'learned'
        path = seeds.PARENT / folder / f's{seed}/test' / (name + '_' + label + '.npz')
    return run.ARR(path)


def worker(seed: int, role: str, sequence: str) -> None:
    """세 모델의 실제 검색·최종 점수를 같은 질의에서 다시 계산한다."""
    spec = prepare()
    assert sequence in spec[role]
    fit, weights = activate(seed, spec)
    folder = OUT / role / f's{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    name = sequence.replace('/', '_')
    target = folder / (name + '.json')
    if target.exists():
        assert run.READ(target)['status'] == 'passed'
        return
    started = time.time()
    cache, pos, minimum, travel, bank, nb, nc, paths = engine.inputs(role, sequence)
    q = np.arange(30, len(pos) - 15, dtype=np.int64)
    conditions = {k: v for k, v in CONDITIONS.items() if role == 'val' or k in spec['test_conditions']}
    angles = tuple(sorted({v[0] for v in conditions.values()}, reverse=True))
    records, checks = {}, []
    for model in MODELS:
        net = seeds.experiment.MODEL(ROOT / spec['models'][str(seed)][model])
        key = run.source.make_keys(net, cache['descriptors'], cache['poses'])
        winner, hits, key_score, pre_score = [], [], [], []
        for t in q:
            w, h, ks, ps = query_search(key, bank, pos, int(t), weights, conditions)
            winner.append(w); hits.append(h); key_score.append(ks); pre_score.append(ps)
        names = ['key', 'pre', *conditions]
        ids = {c: np.asarray([w[c] for w in winner], np.int64) for c in names}
        hits = np.asarray(hits, bool)
        union, inverse = np.unique(np.concatenate([q * len(pos) + ids[c] for c in conditions]), return_inverse=True)
        spectral = spectral_stages((nb, nc), travel, union // len(pos), union % len(pos), angles)
        context = run.neural_scores(key, travel, union // len(pos), union % len(pos))
        ref = previous_ledger(seed, role, sequence, model, spec)
        changed = np.flatnonzero(ids['full'] != ref['j_star'])
        assert len(changed) <= max(2, int(np.ceil(len(q) * .001)))
        replay = []
        # GPU 간선 합산의 FP32 미세 차이로 동점 부근 승자가 달라질 수 있다.
        # 바뀐 질의는 같은 새 키를 원래 검색 식에 넣어 별도로 검산한다.
        for at in changed:
            t = int(q[at])
            candidates = np.unique(np.r_[run.base.prefix_topk(key, t, t - 29, 100),
                                        run.base.prefix_topk(bank[0], t, t - 29, 100)])
            a = run.base.align_scores(bank[1], bank[2], t, candidates)
            c = run.base.column_align_scores(bank[3], t, candidates)
            original = (weights[0] * run.base.minmax(key[candidates] @ key[t]) +
                        weights[1] * run.base.minmax(bank[0][candidates] @ bank[0][t]) +
                        weights[2] * run.base.minmax(a) + weights[3] * run.base.minmax(c))
            assert candidates[np.argmax(original)] == ids['full'][at]
            old_at = np.flatnonzero(candidates == ref['j_star'][at])
            gap = float(original.max() - original[old_at[0]]) if len(old_at) else None
            cut_gaps = {}
            for label, feature in (('key', key), ('aux', bank[0])):
                top = run.base.prefix_topk(feature, t, t - 29, 100)
                cutoff = float((feature[top] @ feature[t]).min())
                cut_gaps[label] = {label2: float(feature[jj] @ feature[t]) - cutoff
                    for label2, jj in (('old', ref['j_star'][at]), ('new', ids['full'][at]))}
            # 상위100 경계에서 후보가 드나들면 최종 융합 점수 차이는 클 수 있다.
            boundary = min(abs(v) for g in cut_gaps.values() for v in g.values())
            assert (gap is not None and 0 <= gap <= 2e-6) or boundary <= 2e-6, (t, gap, cut_gaps)
            replay.append(dict(query=t, old=int(ref['j_star'][at]), new=int(ids['full'][at]),
                               score_gap=gap, candidate_cutoff_gaps=cut_gaps, original_search_same=True))
        np.testing.assert_array_equal(q, ref['query_idx'])
        np.testing.assert_array_equal(minimum[q], ref['min_past_gt_dist'])
        values = {'key': {'score': np.asarray(key_score)}, 'pre': {'score': np.asarray(pre_score)},
                  'no_alignment': {'score': budget.key_temporal(key, travel, q, ids['pre'])}}
        ids['no_alignment'] = ids['pre']
        for i, (condition, (angle, _)) in enumerate(conditions.items()):
            ix = inverse[i * len(q):(i + 1) * len(q)]
            sp, ct = spectral[angle]['temporal'][ix], context[ix]
            values[condition] = dict(center=spectral[angle]['center'][ix], spectral=sp, context=ct,
                score=run.predict_score(sp[:, None], fit['spectral']) + .5 * (ct - fit['context_mean']) / fit['context_scale'])
        same = ids['full'] == ref['j_star']
        np.testing.assert_array_equal(values['full']['spectral'][same], ref['spectral'][same])
        np.testing.assert_allclose(values['full']['score'][same], ref['lambda_0.5' if role == 'val' else 'score'][same], atol=2e-6, rtol=0)
        if len(changed):
            np.testing.assert_array_equal(values['full']['spectral'][changed],
                run.current.adoption.scores(nb, nc, travel, q[changed], ids['full'][changed])['score'])
        record = {}
        for condition, scores in values.items():
            row = dict(query_idx=q, j_star=ids[condition], min_past_gt_dist=minimum[q],
                top1_geo_dist=np.linalg.norm(pos[q] - pos[ids[condition]], axis=1), group=ref['group'])
            assert (row['j_star'] <= q - 30).all()
            metrics = {}
            for field in ('score', 'center', 'spectral'):
                if field not in scores:
                    continue
                assert np.isfinite(scores[field]).all()
                metrics[field] = run.base.independent_metrics(row, scores[field])
                independent = run.white.prior.BASE.exp.metrics(scores[field], row)
                for metric in run.base.METRICS:
                    if metrics[field][metric] is not None:
                        np.testing.assert_allclose(metrics[field][metric], independent[metric], atol=1e-12, rtol=0)
            path = folder / f'{name}_{model}_{condition}.npz'
            extras = {'hits': hits} if condition == 'key' else {}
            np.savez_compressed(path, **row, **scores, **extras)
            record[condition] = dict(path=str(path.relative_to(ROOT)), sha256=run.HASH(path), metrics=metrics)
        positive = minimum[q] <= 5
        recall = {label: float(hits[positive, i].mean()) if positive.any() else None
                  for i, label in enumerate(('R1', 'R5', 'R20', 'R100', 'union'))}
        for at in (len(q) // 3, 2 * len(q) // 3):
            t, end = int(q[at]), int(q[at]) + 16
            short = run.source.make_keys(net, cache['descriptors'], cache['poses'], end)
            np.testing.assert_allclose(short[:t + 6], key[:t + 6], atol=4e-6, rtol=4e-5)
            truncate = run.current.adoption.previous.source.base.exp.truncate_bank
            w = query_search(short, truncate(bank, end), pos[:end], t, weights, conditions)[0]
            assert w == winner[at]
            qq = np.full(len(conditions), t, np.int64)
            jj = np.asarray([ids[c][at] for c in conditions], np.int64)
            small = spectral_stages((truncate(nb, end), truncate(nc, end)), travel[:end], qq, jj, angles)
            partial_context = run.neural_scores(short, travel[:end], qq, jj)
            for i, (c, (a, _)) in enumerate(conditions.items()):
                np.testing.assert_array_equal(small[a]['temporal'][i], values[c]['spectral'][at])
                np.testing.assert_allclose(partial_context[i], values[c]['context'][at], atol=2e-6, rtol=0)
            np.testing.assert_allclose(budget.key_temporal(short, travel[:end], q[at:at + 1], ids['pre'][at:at + 1]), values['no_alignment']['score'][at:at + 1], atol=2e-6, rtol=0)
            checks.append(dict(model=model, query=t, conditions=list(conditions), prefix_passed=True))
        records[model] = dict(conditions=record, recall=recall, n_pos=int(positive.sum()),
                             numerical_replay_changes=replay)
        del net, key
        torch.cuda.empty_cache()
        print('MODEL', role, seed, sequence, model, round(time.time() - started, 1), flush=True)
    run.WRITE(target, dict(status='passed', role=role, seed=seed, sequence=sequence, queries=len(q),
        records=records, checks=checks, full_replay=True, seconds=time.time() - started,
        inputs={str(p.relative_to(ROOT)): run.HASH(p) for p in paths}, protocol_sha256=run.HASH(OUT / 'protocol.json')))


def finish(role: str) -> dict:
    """기존 표2 집계와 같은 평균을 쓰며 중간 회수율은 따로 집계한다."""
    spec = prepare()
    metrics, recalls = {}, {}
    for seed in spec['seeds']:
        for sequence in spec[role]:
            meta = run.READ(OUT / role / f's{seed}' / (sequence.replace('/', '_') + '.json'))
            assert meta['status'] == 'passed' and len(meta['checks']) == 6 and meta['full_replay']
            for model, record in meta['records'].items():
                recalls.setdefault(model, {}).setdefault(str(seed), {})[sequence] = dict(record['recall'], n_pos=record['n_pos'])
                for condition, entry in record['conditions'].items():
                    assert run.HASH(ROOT / entry['path']) == entry['sha256']
                    for stage, metric in entry['metrics'].items():
                        label = model + '__' + condition + '__' + stage
                        metrics.setdefault(label, {}).setdefault(str(seed), {})[sequence] = metric
    def aggregate(rows: dict, fields: tuple) -> dict:
        groups = {}
        for sequence, value in rows.items():
            if not value['n_pos']:
                continue
            group = (next(city for city, names in run.current.CITY.items() if sequence in names)
                     if role == 'test' else 'kitti' if sequence.startswith('kitti') else sequence.split('/')[0])
            groups.setdefault(group, []).append(value)
        group_means = {g: {m: float(np.mean([v[m] for v in vs])) for m in fields} for g, vs in groups.items()}
        return dict(groups=group_means, macro={m: float(np.mean([g[m] for g in group_means.values()])) for m in fields})
    results = {}
    for label, rows in metrics.items():
        per = {s: dict(**aggregate(v, run.base.METRICS), sequences=v) for s, v in rows.items()}
        results[label] = dict(per_seed=per, mean={m: float(np.mean([v['macro'][m] for v in per.values()])) for m in run.base.METRICS})
    recall_results = {}
    for model, rows in recalls.items():
        fields = ('R1', 'R5', 'R20', 'R100', 'union')
        per = {s: aggregate(v, fields) for s, v in rows.items()}
        recall_results[model] = dict(per_seed=per, mean={m: float(np.mean([v['macro'][m] for v in per.values()])) for m in fields})
    summary = dict(role=role, results=results, recall=recall_results, test_used_for_selection=False,
        protocol_sha256=run.HASH(OUT / 'protocol.json'), no_retraining=True)
    run.WRITE(OUT / role / 'summary.json', summary)
    run.WRITE(OUT / role / 'status.json', dict(status='complete', jobs=3 * len(spec[role])))
    print('COMPLETE', role, {m: r['mean'] for m, r in recall_results.items()}, flush=True)
    return summary


def campaign(role: str, workers: int) -> None:
    """독립 시퀀스를 컨테이너 프로세스로 실행한다."""
    spec = prepare()
    logs = OUT / role / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    jobs = [(s, q) for s in spec['seeds'] for q in spec[role]]
    def invoke(job: tuple) -> None:
        s, q = job
        with (logs / f's{s}_{q.replace("/", "_")}.log').open('w') as log:
            subprocess.run([sys.executable, __file__, 'worker', '--role', role, '--seed', str(s), '--sequence', q], stdout=log, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, future in enumerate(concurrent.futures.as_completed([pool.submit(invoke, j) for j in jobs]), 1):
            future.result()
            run.WRITE(OUT / role / 'status.json', dict(status='running', done=i, total=len(jobs)))
            print('PROGRESS', role, i, len(jobs), flush=True)
    finish(role)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'worker', 'run', 'finish'))
    parser.add_argument('--role', choices=('val', 'test'), default='val')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sequence', default='kitti360/0004')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    run.configure()
    if args.action == 'worker':
        worker(args.seed, args.role, args.sequence)
    elif args.action == 'run':
        campaign(args.role, args.workers)
    elif args.action == 'finish':
        finish(args.role)
    else:
        prepare()
