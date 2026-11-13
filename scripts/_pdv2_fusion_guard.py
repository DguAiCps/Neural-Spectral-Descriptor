#!/usr/bin/env python3
"""시간 전용 차이 모델의 교정 가능성·기존 정답 보호 손실을 2x2 비교한다."""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_fusion_factor as factor
from _pdv2_fusion_factor_lib import fuse_with_range
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash
from retrieval.fusion_guard import fusion_guard_terms, fusion_guard_loss

base = factor.base
OUT = REPO / 'results/pdv2_fusion_guard_20261023'
CONFIG = REPO / 'configs/pdv2_fusion_guard.yaml'
CFG = yaml.safe_load(CONFIG.read_text())
METHOD = CFG['final_score']
INITIAL = factor.prior.OUT / 's1/interaction/checkpoints/checkpoint_epoch_20.pth'


def model() -> torch.nn.Module:
    """기하 결합을 포함한 차이 K/V-tanh의 기존 구조를 그대로 만든다."""
    return factor.prior.model('interaction')


def initial_model() -> torch.nn.Module:
    """동일 체크포인트·재계량으로 시작하고 추론 상태에서 미분한다."""
    net = model().cuda().eval()
    net.load_state_dict(torch.load(INITIAL, map_location='cpu', weights_only=False)['model_state_dict'])
    net.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(np.load(base.exp.old.base.RVEC)).float().cuda(),
                                          requires_grad=False)
    return net


def protocol(inputs: bool = False) -> dict:
    """학습 전에 고정한 실행 코드와 입력을 확인한다."""
    data = base.read(OUT / 'protocol.json')
    for field in (('runtime', 'inputs') if inputs else ('runtime',)):
        for path, digest in data[field].items():
            assert sha256(REPO / path) == digest, path
    return data


def pool_path(sequence: str) -> Path:
    """초기 모델의 실제 후보 목록 위치를 반환한다."""
    return OUT / 'pools' / sequence.replace('/', '_')


def load_pool(sequence: str, device: str = 'cuda') -> dict:
    """패딩을 포함한 후보·라벨·고정 점수를 읽는다."""
    with np.load(pool_path(sequence) / 'pool.npz') as z:
        return {k: torch.from_numpy(z[k]).to(device) for k in z.files}


def terms_for(embedding: torch.Tensor, pool: dict, rows: torch.Tensor,
              weights: tuple) -> dict:
    """선택 질의의 현재 키와 고정 보조 점수로 손실 성분을 만든다."""
    candidates = pool['candidates'][rows]
    key = (embedding[pool['queries'][rows], None] * embedding[candidates.clamp_min(0)]).sum(-1)
    return fusion_guard_terms(key, pool['teacher_key'][rows], pool['static'][rows],
        pool['valid'][rows], pool['correct'][rows], weights, CFG['temperature'], CFG['margin_cap'])


def snapshot(sequence: str, ordinal: int, graph, groups: np.ndarray, conf: dict) -> dict:
    """학습 장소의 모든 초기 정답 포함 후보 목록을 고정한다."""
    folder = pool_path(sequence)
    folder.mkdir(parents=True, exist_ok=True)
    node_ids = np.flatnonzero(groups == ordinal)
    local = base.temporal_subgraph(graph, node_ids)
    paths = factor.paths_for('train', sequence)
    cache = base.exp.arrays(paths['cache'])
    np.testing.assert_array_equal(local.x.numpy(), cache['descriptors'])
    torch.save(local, folder / 'graph.pt')
    key_data = base.exp.arrays(factor.key_path('train', 'interaction', sequence))
    np.testing.assert_array_equal(key_data['scan_ids'], cache['scan_ids'])
    key = key_data['embedding']
    baseline = base.exp.arrays(factor.OUT / 'train' / sequence.replace('/', '_') / 'new.npz')
    eligible = (baseline['min_past_gt_dist'] <= 5) & baseline['union_hit']
    queries = baseline['query_idx'][eligible]
    positions = cache['poses'][:, :3, 3].astype(np.float64)
    with np.load(paths['spec']) as z:
        bank = base.exp.old.base.bank_for(base.exp.OP, z)
    n = len(queries)
    arrays = {'queries': queries, 'candidates': np.full((n, 200), -1, np.int64),
        'teacher_key': np.zeros((n, 200), np.float32), 'static': np.zeros((n, 200, 3), np.float32),
        'correct': np.zeros((n, 200), bool), 'valid': np.zeros((n, 200), bool)}
    for i, query in enumerate(queries):
        t = int(query)
        ids = np.unique(np.r_[base.prefix_topk(key, t, t - 29, 100),
                              base.prefix_topk(bank[0], t, t - 29, 100)])
        values = np.column_stack((key[ids] @ key[t], bank[0][ids] @ bank[0][t],
            base.align_scores(bank[1], bank[2], t, ids), base.column_align_scores(bank[3], t, ids)))
        score, _ = fuse_with_range(values, tuple(conf['tuple']))
        assert ids[np.argmax(score)] == baseline['j_star'][t - 30]
        size = len(ids)
        arrays['candidates'][i, :size] = ids
        arrays['teacher_key'][i, :size] = values[:, 0]
        arrays['static'][i, :size] = values[:, 1:]
        arrays['correct'][i, :size] = np.linalg.norm(positions[ids] - positions[t], axis=1) <= 5
        arrays['valid'][i, :size] = True
    active_rows = arrays['correct'].any(1) & (arrays['valid'] & ~arrays['correct']).any(1)
    no_competition = int((~active_rows).sum())
    arrays = {k: v[active_rows] for k, v in arrays.items()}
    n = len(arrays['queries'])
    np.savez_compressed(folder / 'pool.npz', **arrays)
    counts = {'queries': n, 'all_queries': len(baseline['query_idx']),
              'positive': int((baseline['min_past_gt_dist'] <= 5).sum()),
              'missing_candidate': int(((baseline['min_past_gt_dist'] <= 5) & ~baseline['union_hit']).sum()),
              'nodes': len(node_ids), 'no_competing_negative': no_competition,
              'steps_per_epoch': math.ceil(n / CFG['batch_queries'])}
    if n:
        pool = {k: torch.from_numpy(v) for k, v in arrays.items()}
        terms = fusion_guard_terms(pool['teacher_key'], pool['teacher_key'], pool['static'],
            pool['valid'], pool['correct'], tuple(conf['tuple']), CFG['temperature'], CFG['margin_cap'])
        counts.update({'active': int(terms['active'].sum()), 'admissible': int(terms['admissible'].sum()),
            'protected': int(terms['protected'].sum()), 'rank_loss_sum': float(terms['rank'].sum()),
            'impossible_loss_sum': float(terms['rank'][terms['active'] & ~terms['admissible']].sum()),
            'initial_guard_sum': float(terms['guard'].sum())})
        assert counts['active'] == n and counts['initial_guard_sum'] == 0
    write_json(folder / 'audit.json', counts)
    print('POOL_DONE', sequence, counts, flush=True)
    return counts


def prepare() -> None:
    """동일 초기 모델·후보·학습 예산·선택 규칙을 학습 전에 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        protocol(True)
        return
    parent = factor.protocol(True)
    sources = {**parent['inputs'], **base.read(factor.OUT / 'training_keys.json')['outputs']}
    runtime = {**parent['runtime'], **base.read(factor.OUT / 'supplemental_sources.json')['sources']}
    for path in (Path(__file__), CONFIG, REPO / 'scripts/_pdv2_fusion_guard_run.py',
                 REPO / 'scripts/_pdv2_fusion_guard_finish.py', REPO / 'src/retrieval/fusion_guard.py',
                 REPO / 'src/retrieval/fixed_fusion_learning.py', REPO / 'src/retrieval/candidate_learning.py',
                 REPO / 'tests/retrieval/test_fusion_guard.py', REPO / 'tests/test_fixed_fusion_learning.py'):
        runtime[str(path.relative_to(REPO))] = sha256(path)
    for sequence in parent['train']:
        path = factor.OUT / 'train' / sequence.replace('/', '_') / 'new.npz'
        sources[str(path.relative_to(REPO))] = sha256(path)
    original = torch.load(base.GRAPH, map_location='cpu', weights_only=False)
    graph = base.temporal_subgraph(original, np.arange(original.num_nodes))
    del original
    assert {k: array_hash(getattr(graph, k).numpy()) for k in parent['graph_hashes']} == parent['graph_hashes']
    prepared = torch.load(base.PREPARED, map_location='cpu', weights_only=False)
    groups = np.asarray(prepared['sequences'])
    counts = {}
    for ordinal, sequence in enumerate(parent['train']):
        counts[sequence] = snapshot(sequence, ordinal, graph, groups, parent['confidence']['1'])
        for name in ('graph.pt', 'pool.npz', 'audit.json'):
            path = pool_path(sequence) / name
            sources[str(path.relative_to(REPO))] = sha256(path)
    del graph, prepared
    sequence = max(counts, key=lambda s: counts[s]['nodes'])
    net = initial_model()
    local = torch.load(pool_path(sequence) / 'graph.pt', map_location='cuda', weights_only=False)
    pool = load_pool(sequence)
    torch.cuda.reset_peak_memory_stats()
    embedding = F.normalize(net(local), dim=1)
    reference = base.exp.arrays(factor.key_path('train', 'interaction', sequence))['embedding']
    key_error = float(np.abs(embedding.detach().cpu().numpy() - reference).max())
    np.testing.assert_allclose(embedding.detach().cpu().numpy(), reference, atol=4e-6, rtol=4e-5)
    rows = torch.arange(min(CFG['batch_queries'], len(pool['queries'])), device='cuda')
    terms = terms_for(embedding, pool, rows, tuple(parent['confidence']['1']['tuple']))
    loss = fusion_guard_loss(terms, True, True)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    assert sum(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters()) > 0
    preflight = {'sequence': sequence, 'initial_key_max_error': key_error, 'loss': float(loss.detach()),
        'initial_guard_max': float(terms['guard'].max().detach()),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'finite_backward': True}
    initial = {k: array_hash(v.detach().cpu().numpy()) for k, v in net.state_dict().items()}
    parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert parameters == 167561
    del net, local, pool, embedding, loss, terms
    torch.cuda.empty_cache()
    for path, digest in {**sources, **runtime}.items():
        assert sha256(REPO / path) == digest, path
    write_json(OUT / 'protocol.json', {'config': CFG, 'runtime': runtime, 'inputs': sources,
        'train': parent['train'], 'val': parent['val'], 'confidence': parent['confidence'],
        'initial': initial, 'trainable_parameters': parameters, 'pool_counts': counts,
        'updates_per_epoch': sum(c['steps_per_epoch'] for c in counts.values()), 'preflight': preflight,
        'test_used': False, 'baseline': 'geometry_interaction seed1 epoch20',
        'scope': '네 조건 모두 같은 초기 가중치·고정 실제 후보·표본 순서·업데이트. '
                 '추론 모듈과 매개변수 추가 없음. BN 통계·재계량 고정, dropout 없이 FP32 미분.',
        'loss': '기존 다중 정답 융합 순위 손실을 공통 사용한다. 2x2는 키만으로도 순위 교정이 '
                '불가능한 행의 순위 손실 제외와 기준 정답의 정규화 후 차이 min(기준차이,.1) '
                '보호이다. 보호는 온도.1로 나눈 부족분 제곱, 계수1. 동일 배치 분모를 유지한다.',
        'limitations': '고정 초기 후보에서의 필요조건 검사이며 교정 가능성의 충분조건이 아니다. '
                       '후보 누락과 정답 없는 질의는 직접 학습하지 않는다. 검증에서는 전부 새로 '
                       '검색하여 후보 변화·오수락까지 최종 AP에 반영한다. 시간 수락 점수는 고정이다.',
        'selection': '각 조건 epoch2/4의 검증 최종 AP 최대, 1e-12 동률 이른 epoch. '
                     '2x2 같은 epoch 결과도 함께 보고한다. 사전 개선 기준은 시작 모델과 rank_all '
                     '대비 AP+.1%p, F1/R100 손실≤.1%p, 계열 AP 손실≤.25%p다. '
                     '통과 여부와 무관하게 시험·원고는 바꾸지 않는다.'})
    print('GUARD_PREPARED', parameters, preflight, flush=True)


def train(arm: str) -> None:
    """같은 표본 순서로 기존 모델만 추가 학습한다."""
    spec = protocol(True)
    folder = OUT / arm
    folder.mkdir(exist_ok=True)
    if (folder / 'training_complete.json').exists():
        for p, h in base.read(folder / 'training_complete.json')['checkpoints'].items():
            assert sha256(folder / p) == h
        return
    assert not list(folder.glob('epoch*.pth')), '부분 학습을 덮어쓰지 않는다'
    torch.manual_seed(1)
    net = initial_model()
    initial = {k: array_hash(v.detach().cpu().numpy()) for k, v in net.state_dict().items()}
    assert initial == spec['initial']
    fixed = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()
             if 'running_' in k or 'num_batches_tracked' in k or k.endswith('key_scale')}
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
        lr=CFG['learning_rate'], weight_decay=CFG['weight_decay'])
    filter_impossible, protect = 'filtered' in arm, 'guarded' in arm
    updates, records, orders = 0, [], []
    started = time.time()
    for epoch in range(1, CFG['epochs'] + 1):
        loss_sum = rank_sum = guard_sum = grad_max = 0.
        used = protected_violations = impossible_seen = protected_seen = 0
        tick = time.time()
        for ordinal, sequence in enumerate(spec['train']):
            n = spec['pool_counts'][sequence]['queries']
            if not n:
                continue
            graph = torch.load(pool_path(sequence) / 'graph.pt', map_location='cuda', weights_only=False)
            pool = load_pool(sequence)
            order = np.random.default_rng(1000 * epoch + ordinal).permutation(n)
            orders.append({'epoch': epoch, 'sequence': sequence, 'sha256': array_hash(order)})
            for start in range(0, n, CFG['batch_queries']):
                rows = torch.as_tensor(order[start:start + CFG['batch_queries']], device='cuda')
                optimizer.zero_grad(set_to_none=True)
                embedding = F.normalize(net(graph), dim=1)
                terms = terms_for(embedding, pool, rows, tuple(spec['confidence']['1']['tuple']))
                loss = fusion_guard_loss(terms, filter_impossible, protect)
                assert torch.isfinite(loss)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), CFG['gradient_clip'])
                assert torch.isfinite(norm)
                optimizer.step()
                updates += 1
                used += len(rows)
                loss_sum += float(loss.detach()) * len(rows)
                rank_sum += float(terms['rank'].detach().sum())
                guard_sum += float(terms['guard'].detach().sum())
                grad_max = max(grad_max, float(norm))
                impossible_seen += int((terms['active'] & ~terms['admissible']).sum())
                protected_seen += int(terms['protected'].sum())
                protected_violations += int((terms['protected'] & (terms['margin'] < 0)).sum())
                del embedding, terms, loss
            del pool, graph
        assert updates == epoch * spec['updates_per_epoch']
        assert all(torch.isfinite(v).all() for v in net.state_dict().values())
        record = {'epoch': epoch, 'updates': updates, 'queries': used, 'loss': loss_sum / used,
            'all_rank_loss': rank_sum / used, 'guard_loss': guard_sum / used, 'gradient_norm_max': grad_max,
            'impossible_seen': impossible_seen, 'protected_seen': protected_seen,
            'protected_violations_seen': protected_violations, 'seconds': time.time() - tick}
        records.append(record)
        write_json(folder / 'training_progress.json', record)
        print('EPOCH', arm, record, flush=True)
        if epoch in CFG['checkpoint_epochs']:
            state = {k: v.detach().cpu() for k, v in net.state_dict().items() if not k.endswith('key_scale')}
            torch.save({'model_state_dict': state, 'epoch': epoch, 'updates': updates,
                'protocol_sha256': sha256(OUT / 'protocol.json')}, folder / f'epoch{epoch}.pth')
    for k, v in fixed.items():
        torch.testing.assert_close(net.state_dict()[k].cpu(), v, atol=0, rtol=0)
    write_json(folder / 'training_complete.json', {'initial': initial, 'orders': orders,
        'updates': updates, 'history': records, 'seconds': time.time() - started,
        'checkpoints': {p.name: sha256(p) for p in folder.glob('epoch*.pth')},
        'bn_and_remetric_frozen': True, 'trainable_parameters': spec['trainable_parameters'],
        'test_used': False})


def evaluate(arm: str, epoch: int, role: str, sequence: str) -> None:
    """전체 질의에서 후보를 다시 검색하고 기존 최종 시간 판정을 계산한다."""
    spec = protocol()
    assert sequence in spec[role]
    checkpoint = OUT / arm / f'epoch{epoch}.pth'
    assert sha256(checkpoint) == base.read(OUT / arm / 'training_complete.json')['checkpoints'][checkpoint.name]
    folder = OUT / arm / f'epoch{epoch}' / role
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (sequence.replace('/', '_') + '.npz')
    if target.exists() and target.with_suffix('.json').exists():
        assert sha256(target) == base.read(target.with_suffix('.json'))['sha256']
        return
    tick = time.time()
    net = model().cuda().eval()
    net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state_dict'])
    net.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(np.load(base.exp.old.base.RVEC)).float().cuda(),
                                          requires_grad=False)
    paths = factor.paths_for(role, sequence)
    cache = base.exp.arrays(paths['cache'])
    key, _ = base.make_keys(net, cache, 'diff')
    positions = cache['poses' if role == 'train' else 'poses_gt'][:, :3, 3].astype(np.float64)
    min_past = factor.prefix_distance(positions)
    with np.load(paths['spec']) as z:
        bank = base.exp.old.base.bank_for(base.exp.OP, z)
    conf = spec['confidence']['1']
    ledger = base.evaluate_keys(key, bank, positions, min_past, {1: tuple(conf['tuple'])})[1]
    keep = ledger['query_idx'] < len(key) - 15
    ledger = {k: v[keep] for k, v in ledger.items()}
    reference = base.exp.arrays(factor.OUT / role / sequence.replace('/', '_') / 'new.npz')
    q, j = ledger['query_idx'], ledger['j_star']
    np.testing.assert_array_equal(q, reference['query_idx'])
    np.testing.assert_allclose(ledger['min_past_gt_dist'], reference['min_past_gt_dist'], atol=1e-10, rtol=0)
    assert np.all(j <= q - 30)
    xyz = cache['poses'][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    raw = ledger[conf['decision']]
    chunks = [base.fixed.frozen_scores(bank, travel, q[i:i + 256], j[i:i + 256],
        conf['decision'], raw[i:i + 256]) for i in range(0, len(q), 256)]
    ledger.update({name: np.concatenate([c[name] for c in chunks]) for name in ('raw', METHOD)})
    prefixes = []
    for ix in (len(q) // 3, 2 * len(q) // 3):
        t, end = int(q[ix]), int(q[ix] + 16)
        small, _ = base.make_keys(net, cache, 'diff', end)
        np.testing.assert_allclose(small[:t + 1], key[:t + 1], atol=4e-6, rtol=4e-5)
        ids = np.unique(np.r_[base.prefix_topk(small, t, t - 29, 100),
                              base.prefix_topk(bank[0], t, t - 29, 100)])
        values = np.column_stack((small[ids] @ small[t], bank[0][ids] @ bank[0][t],
            base.align_scores(bank[1], bank[2], t, ids), base.column_align_scores(bank[3], t, ids)))
        score, _ = fuse_with_range(values, tuple(conf['tuple']))
        assert ids[np.argmax(score)] == j[ix]
        direct = factor.scalar_score(bank, travel, t, int(j[ix]), conf['decision'], float(raw[ix]))
        short = base.fixed.frozen_scores(base.exp.truncate_bank(bank, end), travel[:end],
            q[ix:ix + 1], j[ix:ix + 1], conf['decision'], raw[ix:ix + 1])[METHOD][0]
        assert abs(direct - ledger[METHOD][ix]) < 1e-12 and abs(short - direct) < 1e-12
        prefixes.append({'q': t, 'key_max_error': float(abs(small[:t + 1] - key[:t + 1]).max()),
                         'winner_equal': True, 'score_error': 0.})
    np.savez_compressed(target, **ledger)
    key_path = target.with_name('keys_' + target.name)
    np.savez_compressed(key_path, embedding=key, scan_ids=cache['scan_ids'])
    diagnostics = None
    if role == 'train':
        pool = load_pool(sequence, 'cpu')
        if len(pool['queries']):
            stored_key = torch.from_numpy(key)
            all_terms = []
            for start in range(0, len(pool['queries']), 512):
                rows = torch.arange(start, min(start + 512, len(pool['queries'])))
                all_terms.append(terms_for(stored_key, pool, rows, tuple(conf['tuple'])))
            terms = {k: torch.cat([x[k] for x in all_terms]) for k in all_terms[0]}
            protected = terms['protected']
            diagnostics = {'fixed_pool_correct': int((terms['margin'] > 0).sum()),
                'protected': int(protected.sum()), 'protected_lost': int((protected & (terms['margin'] <= 0)).sum()),
                'fixed_pool_gained': int((~protected & (terms['margin'] > 0)).sum()),
                'ranking_loss': float(terms['rank'].mean()), 'guard_loss': float(terms['guard'].mean())}
    write_json(target.with_suffix('.json'), {'sha256': sha256(target), 'keys_sha256': sha256(key_path),
        'role': role, 'sequence': sequence, 'arm': arm, 'epoch': epoch, 'queries': len(q),
        'metrics': {m: factor.independent_metrics(ledger, ledger[m]) for m in ('raw', METHOD)},
        'prefixes': prefixes, 'training_fixed_pool': diagnostics, 'seconds': time.time() - tick,
        'test_used': False})
    print('EVAL_DONE', role, arm, epoch, sequence, round(time.time() - tick, 1), flush=True)


def select() -> None:
    """최종 검증 AP로 정해 둔 두 체크포인트만 비교한다."""
    spec = protocol(True)
    results, selections = {}, {}
    for arm in CFG['arms']:
        rows = {}
        for epoch in CFG['checkpoint_epochs']:
            per_sequence = {s: base.read(OUT / arm / f'epoch{epoch}/val' /
                (s.replace('/', '_') + '.json'))['metrics'][METHOD] for s in spec['val']}
            rows[str(epoch)] = base.macro(per_sequence)
        epoch = max(CFG['checkpoint_epochs'], key=lambda e: (round(rows[str(e)]['macro']['AP'], 12), -e))
        results[arm], selections[arm] = rows, epoch
        print('SELECTED', arm, epoch, rows[str(epoch)]['macro'], flush=True)
    write_json(OUT / 'selection.json', {'results': results, 'selected_epoch': selections,
        'test_used': False, 'protocol_sha256': sha256(OUT / 'protocol.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'train', 'evaluate', 'select'))
    parser.add_argument('--arm', choices=CFG['arms'])
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--role', choices=('train', 'val'))
    parser.add_argument('--sequence')
    args = parser.parse_args()
    torch.set_num_threads(4 if args.action == 'train' else 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.action == 'prepare':
        prepare()
    elif args.action == 'train':
        train(args.arm)
    elif args.action == 'evaluate':
        evaluate(args.arm, args.epoch, args.role, args.sequence)
    else:
        select()
