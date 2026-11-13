#!/usr/bin/env python3
"""실제 과거 후보의 위치 정답·정렬 순서를 DiffAttnConv 두 층에 학습한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts')]
import _pdv2_diffattn_gate_learning as gate
import _pdv2_diffattn_alignment_stages as stages
from retrieval.alignment_distillation import alignment_ranking_loss
from retrieval.gated_context import gated_context_score

run = gate.run
PARENT = ROOT / 'results/pdv2_diffattn_gate_learning_20261104'
BASELINE = stages.OUT
OUT = ROOT / 'results/pdv2_diffattn_alignment_teacher_20261104'
ARMS = {f'{mode}_{tag}': (mode, rate) for mode in ('key', 'joint', 'teacher')
        for tag, rate in (('lr5', 1e-5), ('lr4', 1e-4))}
EPOCHS = (4, 8)


def prepare() -> dict:
    """학습 범위·교사 신뢰 규칙·선택 기준을 검증 전에 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        spec = run.READ(path)
        for name, digest in spec['sources'].items():
            assert run.HASH(ROOT / name) == digest, name
        return spec
    prior = run.READ(PARENT / 'protocol.json')
    initial = PARENT / 'recenter_frozen/epoch0.pth'
    paths = [Path(__file__), Path(stages.__file__), Path(run.__file__),
             ROOT / 'src/retrieval/alignment_distillation.py',
             ROOT / 'tests/retrieval/test_alignment_distillation.py',
             ROOT / 'src/retrieval/fixed_fusion_learning.py',
             ROOT / 'src/retrieval/candidate_learning.py',
             ROOT / 'src/gnn/geometry_interaction.py', ROOT / 'src/gnn/model.py',
             Path(run.source.__file__), initial, PARENT / 'calibration_recenter.json',
             BASELINE / 'val/summary.json', ROOT / 'results/pdv2_learning_plan_20261104/training_proposal.json']
    for seq in prior['train']:
        paths.append(PARENT / 'pools' / seq.replace('/', '_') / 'graph.pt')
    spec = dict(created_unix=time.time(), seed=1, train=prior['train'], val=prior['val'],
        initial=str(initial.relative_to(ROOT)), confidence=prior['confidence'], arms=ARMS,
        epochs=EPOCHS, steps_per_sequence=8, batch_queries=128, roster_limit=1024,
        weight_decay=1e-5, clip=1., temperature=.1, coefficient=.5,
        architecture='기존288+128·시간±5·2층 차이K/V 및 기하 상호작용. 초기 포화 완화 모델 공통. gnn.convs.*만 갱신, 게이트·투영·BN통계·재계량 고정. 새 추론 모듈·차원 없음.',
        pool='학습8만 GT 입력. 시퀀스별 재방문 질의 최대1024개를 고정 난수로 선정. 실제 과거100키+100보조 합집합. 정답이 없으면 과거 최근접<=5m 한 개를 학습에만 주입하며 따로 기록. 5~25m 후보는minmax에 포함하되손실라벨 제외. 양성 없는KITTI01은순위갱신없음.',
        teacher='원본/25%보정·크기제곱근 두채널의 중심60각도 순환 정렬 점수. 시간 판정 없는 교사로 계산을 제한. 교사의 전체실제후보1위가위치정답일때만KL. 위치정답을대신하지않음. 추론교사추가실행없음.',
        loss='key: 정규화 키 순위의 정답확률합 손실. joint: 키와 정렬전 키/보조 융합 손실 각1/2. teacher: joint+같은두학생의 정렬분포KL평균, 신뢰가능행만활성. 모든온도.1,KL계수1. 회색후보는분포에서제외. 최종 수락BCE는 이번 순위 대조에 넣지 않음.',
        training='세손실×학습률1e-5/1e-4. 동일초기·질의·난수·8에포크.4/8체크포인트.1~4초기풀공통,5시작각학습모델로같은질의의과거후보갱신. 후보갱신으로표본수가바뀌지않음. 정답/오답둘다있는행만손실.',
        selection='고정문맥비중.5·기존정렬/융합의검증최종AP최대. 확장조건: 같은동결대비AP+.1%p·키R1+.2%p,정렬전융합R1손실<=.1%p,F1손실<=.3%p,R100손실<=1%p. BEV동일시간검증대비AP/F1열세2%p·R100/R1열세3%p이내. 동률작은lr→이른epoch→이름.통과후설정고정3시드확인.시험은재선택에사용하지않음.',
        limitation='단일초기시드와반복검증의탐색. GT학습/KISS검증차이는유지. 교사의위상구분력이입력에복원가능하다는보장없음. 추가학습전도학습된모델. 표준attention대비고유우위아님. 원고·그림유지.',
        sources={str(p.relative_to(ROOT)): run.HASH(p) for p in paths}, test_used=False)
    run.WRITE(path, spec)
    run.WRITE(OUT / 'calibration.json', run.READ(PARENT / 'calibration_recenter.json'))
    return spec


def activate() -> dict:
    """기존 입력·모델 로더에 이 실험의 고정 경로를 공급한다."""
    spec = prepare()
    run.OUT = OUT
    run.context_score = gated_context_score
    return spec


def model(checkpoint: Path = None) -> torch.nn.Module:
    """두 차이 주의 층만 갱신 가능한 현재 모델을 불러온다."""
    spec = activate()
    net = gate.MODEL(ROOT / spec['initial'] if checkpoint is None else checkpoint)
    for name, parameter in net.named_parameters():
        parameter.requires_grad_(name.startswith('gnn.convs.'))
    return net


def mine(sequence: str, folder: Path, checkpoint: Path = None) -> dict:
    """학습 위치 정답을 유지하며 현재 검색 후보와 정렬 교사를 저장한다."""
    spec = activate()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'audit.json'
    if target.exists():
        meta = run.READ(target)
        assert run.HASH(folder / 'pool.npz') == meta['sha256']
        return meta
    tick = time.time()
    cache, pos, minimum, travel, bank, nb, nc, paths = run.inputs('train', sequence)
    net = model(checkpoint)
    graph_path = PARENT / 'pools' / sequence.replace('/', '_') / 'graph.pt'
    graph = torch.load(graph_path, map_location='cuda', weights_only=False)
    with torch.no_grad():
        key = F.normalize(net(graph).float(), dim=1).cpu().numpy()
    query = np.arange(30, len(key) - 15, dtype=np.int64)
    query = query[minimum[query] <= 5]
    rng = np.random.default_rng(24000 + spec['train'].index(sequence))
    if len(query) > spec['roster_limit']:
        query = np.sort(rng.choice(query, spec['roster_limit'], replace=False))
    width = 201
    candidates = np.full((len(query), width), -1, np.int64)
    aux = np.zeros((len(query), width), np.float32)
    teacher = np.zeros_like(aux)
    correct = np.zeros_like(candidates, bool)
    eligible = np.zeros_like(correct)
    injected, trusted, usable = 0, 0, 0
    for at, q in enumerate(query):
        ids = np.unique(np.r_[run.base.prefix_topk(key, q, q - 29, 100),
                              run.base.prefix_topk(bank[0], q, q - 29, 100)])
        distances = np.linalg.norm(pos[ids] - pos[q], axis=1)
        if not np.any(distances <= 5):
            nearest = int(np.argmin(np.linalg.norm(pos[:q - 29] - pos[q], axis=1)))
            ids = np.unique(np.r_[ids, nearest])
            distances = np.linalg.norm(pos[ids] - pos[q], axis=1)
            injected += 1
        channels = []
        for b in (nb, nc):
            a = run.base.align_scores(b[1], b[2], int(q), ids)
            c = run.base.column_align_scores(b[3], int(q), ids)
            channels.append((a, c))
        value = run.cal.confidence(.75 * channels[0][0] + .25 * channels[1][0],
                                   .75 * channels[0][1] + .25 * channels[1][1], 'equal', 'c_col')
        size = len(ids)
        candidates[at, :size] = ids
        aux[at, :size] = bank[0][ids] @ bank[0][q]
        teacher[at, :size] = value
        correct[at, :size] = distances <= 5
        eligible[at, :size] = (distances <= 5) | (distances > 25)
        trusted += int(distances[np.argmax(value)] <= 5)
        usable += int((distances <= 5).any() and (distances > 25).any())
    present = candidates >= 0
    assert np.all(candidates[present] <= np.broadcast_to(query[:, None] - 30, candidates.shape)[present])
    assert correct.any(1).all() and np.isfinite(teacher).all()
    path = folder / 'pool.npz'
    np.savez_compressed(path, query=query, candidates=candidates, auxiliary=aux,
                        teacher=teacher, correct=correct, present=present, eligible=eligible)
    meta = dict(sequence=sequence, nodes=len(key), queries=len(query), injected_positive=injected,
                teacher_correct=trusted, usable=usable, seconds=time.time() - tick,
                sha256=run.HASH(path), graph=str(graph_path.relative_to(ROOT)),
                checkpoint_sha256=run.HASH(ROOT / spec['initial'] if checkpoint is None else checkpoint),
                inputs={str(p.relative_to(ROOT)): run.HASH(p) for p in paths}, test_used=False)
    run.WRITE(target, meta)
    print('MINE', sequence, len(query), trusted, injected, round(meta['seconds'], 1), flush=True)
    return meta


def train(arm: str) -> None:
    """같은 질의 순서로 순위 손실만 바꿔 두 층을 학습한다."""
    spec = activate()
    folder = OUT / arm
    folder.mkdir(exist_ok=True)
    if (folder / 'training_complete.json').exists():
        return
    assert not list(folder.glob('epoch*.pth')), '부분 학습을 덮어쓰지 않는다'
    mode, lr = ARMS[arm]
    torch.manual_seed(1)
    net = model()
    initial = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=spec['weight_decay'])
    weights = run.cal.fusion_weights(spec['confidence'])[run.cal.FUSIONS.index('aux_half')]
    logs, hashes, updates = [], {}, 0
    tick = time.time()
    for epoch in range(1, max(EPOCHS) + 1):
        for ordinal, seq in enumerate(spec['train']):
            name = seq.replace('/', '_')
            pooldir = OUT / 'pools' / name if epoch <= 4 else folder / 'refreshed' / name
            if epoch == 5:
                mine(seq, pooldir, folder / 'epoch4.pth')
            meta = run.READ(pooldir / 'audit.json')
            arrays = run.ARR(pooldir / 'pool.npz')
            original = run.ARR(OUT / 'pools' / name / 'pool.npz')
            np.testing.assert_array_equal(arrays['query'], original['query'])
            if not len(arrays['query']):
                continue
            graph = torch.load(ROOT / meta['graph'], map_location='cuda', weights_only=False)
            pool = {k: torch.from_numpy(v).cuda() for k, v in arrays.items()}
            rng = np.random.default_rng(1000 * epoch + ordinal)
            values, order = [], []
            for step in range(spec['steps_per_sequence']):
                chosen = rng.choice(len(arrays['query']), spec['batch_queries'], replace=len(arrays['query']) < spec['batch_queries'])
                order.append(run.array_hash(chosen))
                rows = torch.from_numpy(chosen).cuda()
                optimizer.zero_grad(set_to_none=True)
                embedding = F.normalize(net(graph), dim=1)
                ids = pool['candidates'][rows]
                key = (embedding[pool['query'][rows], None] * embedding[ids.clamp_min(0)]).sum(-1)
                loss, terms = alignment_ranking_loss(key, pool['auxiliary'][rows], pool['teacher'][rows],
                    pool['correct'][rows], pool['present'][rows], pool['eligible'][rows], weights, mode, spec['temperature'])
                assert torch.isfinite(loss)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], spec['clip'])
                assert torch.isfinite(grad)
                optimizer.step()
                updates += 1
                values.append([float(loss.detach()), float(terms['key'].detach()), float(terms['joint'].detach()),
                               float(terms['distill'].detach()), float(terms['trusted']), float(grad)])
            record = dict(epoch=epoch, sequence=seq, updates=updates,
                loss_key_joint_distill_trusted_grad=np.mean(values, axis=0).tolist(), batch_hashes=order,
                pool_sha256=meta['sha256'])
            logs.append(record)
            run.WRITE(folder / 'progress.json', record)
            del graph, pool, embedding, loss, key, terms
            torch.cuda.empty_cache()
        if epoch in EPOCHS:
            path = folder / f'epoch{epoch}.pth'
            torch.save(dict(model_state_dict=net.state_dict(), epoch=epoch, updates=updates,
                            protocol_sha256=run.HASH(OUT / 'protocol.json')), path)
            hashes[path.name] = run.HASH(path)
        print('TRAIN', arm, epoch, updates, record['loss_key_joint_distill_trusted_grad'], flush=True)
    changed = []
    for name, tensor in net.state_dict().items():
        assert torch.isfinite(tensor).all(), name
        if not torch.equal(tensor.detach().cpu(), initial[name]):
            assert name.startswith('gnn.convs.'), name
            changed.append(name)
    assert changed and updates == 448
    run.WRITE(folder / 'training_complete.json', dict(status='passed', arm=arm, updates=updates,
        changed_parameters=changed, fixed_buffers_verified=True, checkpoints=hashes, records=logs,
        seconds=time.time() - tick, test_used=False))


def validate(arm: str, epoch: int, sequence: str) -> None:
    """실제 검색·융합·최종 판정과 중간 회수율을 함께 확인한다."""
    spec = activate()
    folder = OUT / arm / f'epoch{epoch}/val'
    folder.mkdir(parents=True, exist_ok=True)
    name = sequence.replace('/', '_')
    target = folder / (name + '.json')
    if target.exists():
        assert run.READ(target)['status'] == 'passed'
        return
    tick = time.time()
    net = model(OUT / arm / f'epoch{epoch}.pth')
    cache, pos, minimum, travel, bank, nb, nc, paths = run.inputs('val', sequence)
    key = run.source.make_keys(net, cache['descriptors'], cache['poses'])
    weights = run.cal.fusion_weights(spec['confidence'])[run.cal.FUSIONS.index('aux_half')]
    q = np.arange(30, len(key) - 15, dtype=np.int64)
    winners, hits, key_scores = [], [], []
    for t in q:
        w, h, ks, _ = stages.query_search(key, bank, pos, int(t), weights, {'full': (60, 200)})
        winners.append(w); hits.append(h); key_scores.append(ks)
    ids = {k: np.asarray([w[k] for w in winners], np.int64) for k in ('key', 'pre', 'full')}
    hits = np.asarray(hits, bool)
    reference = run.ARR(BASELINE / 'val/s1' / (name + '_frozen_full.npz'))
    np.testing.assert_array_equal(q, reference['query_idx'])
    np.testing.assert_array_equal(minimum[q], reference['min_past_gt_dist'])
    row = dict(query_idx=q, j_star=ids['full'], min_past_gt_dist=minimum[q],
               top1_geo_dist=np.linalg.norm(pos[q] - pos[ids['full']], axis=1), group=reference['group'])
    spectral = run.current.adoption.scores(nb, nc, travel, q, ids['full'])['score']
    context = run.neural_scores(key, travel, q, ids['full'])
    fit = run.READ(OUT / 'calibration.json')
    score = run.predict_score(spectral[:, None], fit['spectral']) + spec['coefficient'] * (context - fit['context_mean']) / fit['context_scale']
    metrics = {field: run.base.independent_metrics(row, value) for field, value in (('score', score), ('spectral', spectral))}
    for field, value in (('score', score), ('spectral', spectral)):
        check = run.white.prior.BASE.exp.metrics(value, row)
        for m in run.base.METRICS:
            if metrics[field][m] is not None:
                np.testing.assert_allclose(metrics[field][m], check[m], atol=1e-12, rtol=0)
    positive = minimum[q] <= 5
    mid = {k: float(hits[positive, i].mean()) if positive.any() else None
           for i, k in enumerate(('key_R1', 'key_R5', 'key_R20', 'key_R100', 'union'))}
    mid['pre_R1'] = float((np.linalg.norm(pos[q[positive]] - pos[ids['pre'][positive]], axis=1) <= 5).mean()) if positive.any() else None
    checks = []
    for at in (len(q) // 3, 2 * len(q) // 3):
        t, end = int(q[at]), int(q[at]) + 16
        short = run.source.make_keys(net, cache['descriptors'], cache['poses'], end)
        np.testing.assert_allclose(short[:t + 6], key[:t + 6], atol=4e-6, rtol=4e-5)
        trunc = run.current.adoption.previous.source.base.exp.truncate_bank
        assert stages.query_search(short, trunc(bank, end), pos[:end], t, weights, {'full': (60, 200)})[0] == winners[at]
        small = run.current.adoption.scores(trunc(nb, end), trunc(nc, end), travel[:end], q[at:at + 1], ids['full'][at:at + 1])['score']
        np.testing.assert_array_equal(small, spectral[at:at + 1])
        np.testing.assert_allclose(run.neural_scores(short, travel[:end], q[at:at + 1], ids['full'][at:at + 1]), context[at:at + 1], atol=2e-6, rtol=0)
        checks.append(t)
    ledger = target.with_suffix('.npz')
    np.savez_compressed(ledger, **row, spectral=spectral, context=context, score=score,
                        key_winner=ids['key'], pre_winner=ids['pre'], key_score=key_scores, hits=hits)
    run.WRITE(target, dict(status='passed', arm=arm, epoch=epoch, sequence=sequence, metrics=metrics,
        intermediate=mid, prefix_checks=checks, ledger_sha256=run.HASH(ledger),
        checkpoint_sha256=run.HASH(OUT / arm / f'epoch{epoch}.pth'), seconds=time.time() - tick,
        input_sha256={str(p.relative_to(ROOT)): run.HASH(p) for p in paths}, test_used=False))
    print('VAL', arm, epoch, sequence, round(time.time() - tick, 1), flush=True)


def finish() -> dict:
    """동결 대조와 같은 비중에서 키 순위·최종 성능이 함께 좋아지는지 고른다."""
    spec = activate()
    prior = run.READ(BASELINE / 'val/summary.json')
    frozen = prior['results']['frozen__full__score']['per_seed']['1']['macro']
    frozen_mid = prior['recall']['frozen']['per_seed']['1']['macro']
    pre_r1 = prior['results']['frozen__pre__score']['per_seed']['1']['macro']['R@1']
    bev = run.READ(gate.previous.OUT / 'bev_validation.json')['temporal']['macro']
    results, eligible = {}, []
    for arm in ARMS:
        training = run.READ(OUT / arm / 'training_complete.json')
        assert training['status'] == 'passed' and training['updates'] == 448
        assert all(k.startswith('gnn.convs.') for k in training['changed_parameters'])
        for epoch in EPOCHS:
            rows, mids, spectral = {}, {}, {}
            for seq in spec['val']:
                path = OUT / arm / f'epoch{epoch}/val' / (seq.replace('/', '_') + '.json')
                record = run.READ(path)
                assert record['status'] == 'passed' and len(record['prefix_checks']) == 2
                assert run.HASH(path.with_suffix('.npz')) == record['ledger_sha256']
                rows[seq], spectral[seq], mids[seq] = record['metrics']['score'], record['metrics']['spectral'], record['intermediate']
            macro = run.base.macro(rows)
            groups = {}
            for seq, value in mids.items():
                if not rows[seq]['n_pos']:
                    continue
                group = 'kitti' if seq.startswith('kitti') else seq.split('/')[0]
                groups.setdefault(group, []).append(value)
            grouped = {g: {m: float(np.mean([v[m] for v in values])) for m in values[0]} for g, values in groups.items()}
            mid = {m: float(np.mean([v[m] for v in grouped.values()])) for m in next(iter(grouped.values()))}
            delta = {m: 100 * (macro['macro'][m] - frozen[m]) for m in run.base.METRICS}
            checks = dict(AP_gain=delta['AP'] >= .1, key_gain=mid['key_R1'] >= frozen_mid['R1'] + .002,
                pre_preserved=mid['pre_R1'] >= pre_r1 - .001,
                F1_preserved=delta['max_F1'] >= -.3, R100_preserved=delta['R@100P'] >= -1.)
            checks.update({f'bev_{m}': macro['macro'][m] >= bev[m] - tolerance
                           for m, tolerance in {'AP': .02, 'max_F1': .02, 'R@100P': .03, 'R@1': .03}.items()})
            result = dict(arm=arm, epoch=epoch, coefficient=.5, **macro, sequences=rows,
                intermediate=mid, intermediate_groups=grouped, spectral=run.base.macro(spectral),
                delta_pp=delta, advancement_checks=checks)
            results[f'{arm}_{epoch}'] = result
            if all(checks.values()):
                eligible.append(result)
    orders = [[r['batch_hashes'] for r in run.READ(OUT / a / 'training_complete.json')['records']] for a in ARMS]
    assert all(o == orders[0] for o in orders)
    selected = min(eligible, key=lambda r: (-round(r['macro']['AP'], 12), ARMS[r['arm']][1], r['epoch'], r['arm'])) if eligible else None
    summary = dict(results=results, frozen=frozen, frozen_intermediate=frozen_mid, frozen_pre_R1=pre_r1,
        bev_temporal=bev, selected=selected, equal_training_samples=True, test_used=False,
        protocol_sha256=run.HASH(OUT / 'protocol.json'))
    run.WRITE(OUT / 'summary.json', summary)
    run.WRITE(OUT / 'status.json', dict(status='validation_complete', selected=None if selected is None else
                                      {k: selected[k] for k in ('arm', 'epoch', 'macro')}, test_used=False))
    run.WRITE(OUT / 'completion_validation.json', dict(status='passed', new_ledgers=108, prefix_checks=216,
        checkpoints=12, parameter_scope='gnn.convs.*', equal_queries=True, equal_updates=448, test_used=False))
    print('FINISHED', None if selected is None else {k: selected[k] for k in ('arm', 'epoch', 'macro')}, flush=True)
    return summary


def campaign(workers: int) -> None:
    """초기 후보 생성·학습·검증을 끝내고 설정을 고정한다."""
    spec = prepare()
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=True)
    def call(args: list, name: str) -> None:
        with (logs / (name + '.log')).open('w') as log:
            subprocess.run([sys.executable, __file__, *args], stdout=log, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(call, ['mine', '--sequence', s], 'mine_' + s.replace('/', '_')) for s in spec['train']]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    for arm in ARMS:
        run.WRITE(OUT / 'status.json', dict(status='running', phase='training', arm=arm))
        call(['train', '--arm', arm], 'train_' + arm)
    jobs = [(a, e, s) for a in ARMS for e in EPOCHS for s in spec['val']]
    def evaluate(job: tuple) -> None:
        a, e, s = job
        call(['val', '--arm', a, '--epoch', str(e), '--sequence', s], f'val_{a}_{e}_{s.replace("/", "_")}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, f in enumerate(concurrent.futures.as_completed([pool.submit(evaluate, j) for j in jobs]), 1):
            f.result()
            run.WRITE(OUT / 'status.json', dict(status='running', phase='validation', done=i, total=len(jobs)))
            print('PROGRESS', i, len(jobs), flush=True)
    finish()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('prepare', 'mine', 'train', 'val', 'finish', 'run'))
    p.add_argument('--arm', choices=ARMS, default='teacher_lr5')
    p.add_argument('--epoch', type=int, default=4)
    p.add_argument('--sequence', default='kitti/06')
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    run.configure()
    if args.action == 'prepare': prepare()
    elif args.action == 'mine': mine(args.sequence, OUT / 'pools' / args.sequence.replace('/', '_'))
    elif args.action == 'train': train(args.arm)
    elif args.action == 'val': validate(args.arm, args.epoch, args.sequence)
    elif args.action == 'finish': finish()
    else: campaign(args.workers)
