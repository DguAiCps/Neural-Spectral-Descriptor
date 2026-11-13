#!/usr/bin/env python3
"""기존 순전파를 유지하며 키 정규화 폭의 학습 기울기만 대조한다."""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_fusion_guard as prior
from _pdv2_fusion_factor_lib import fuse_with_range
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash
from retrieval.fusion_guard import fusion_guard_loss
from retrieval.fusion_gradient import fixed_width_terms

base, factor = prior.base, prior.factor
OUT = REPO / 'results/pdv2_fusion_width_20261024'
CONFIG = REPO / 'configs/pdv2_fusion_width.yaml'
CFG = yaml.safe_load(CONFIG.read_text())
METHOD = CFG['final_score']
model, initial_model = prior.model, prior.initial_model
pool_path, load_pool = prior.pool_path, prior.load_pool


def protocol(inputs: bool = False) -> dict:
    """고정 소스와 입력의 해시를 확인한다."""
    data = base.read(OUT / 'protocol.json')
    for field in (('runtime', 'inputs') if inputs else ('runtime',)):
        for path, digest in data[field].items():
            assert sha256(REPO / path) == digest, path
    return data


def terms_for(embedding: torch.Tensor, pool: dict, rows: torch.Tensor, weights: tuple) -> dict:
    """같은 후보·보조 점수에서 분모의 미분만 차단한다."""
    candidates = pool['candidates'][rows]
    key = (embedding[pool['queries'][rows], None] * embedding[candidates.clamp_min(0)]).sum(-1)
    return fixed_width_terms(key, pool['teacher_key'][rows], pool['static'][rows],
        pool['valid'][rows], pool['correct'][rows], weights, CFG['temperature'], CFG['margin_cap'])


def prepare() -> None:
    """대조 가중치·후보·순서·예산과 선택 규칙을 결과 관측 전에 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        protocol(True)
        return
    parent = prior.protocol(True)
    assert base.read(prior.OUT / 'completion_validation.json')['passed']
    runtime, inputs = dict(parent['runtime']), dict(parent['inputs'])
    for p in (Path(__file__), CONFIG, REPO / 'scripts/_pdv2_fusion_width_run.py',
              REPO / 'scripts/_pdv2_fusion_width_finish.py', REPO / 'scripts/_pdv2_fusion_width_probe.py',
              REPO / 'src/retrieval/fusion_gradient.py', REPO / 'tests/retrieval/test_fusion_gradient.py'):
        runtime[str(p.relative_to(REPO))] = sha256(p)
    for p in (prior.OUT / 'summary.json', prior.OUT / 'selection.json',
              REPO / 'results/pdv2_context_transfer_20261024/summary.json'):
        inputs[str(p.relative_to(REPO))] = sha256(p)
    for arm in ('rank_all', 'rank_filtered_guarded'):
        folder = prior.OUT / arm
        for p in [folder / 'training_complete.json', *folder.glob('epoch*.pth')]:
            inputs[str(p.relative_to(REPO))] = sha256(p)
    net = initial_model()
    initial = {k: array_hash(v.detach().cpu().numpy()) for k, v in net.state_dict().items()}
    assert initial == parent['initial']
    seq = max(parent['pool_counts'], key=lambda s: parent['pool_counts'][s]['nodes'])
    graph = torch.load(pool_path(seq) / 'graph.pt', map_location='cuda', weights_only=False)
    pool = load_pool(seq)
    embedding = F.normalize(net(graph), dim=1)
    rows = torch.arange(min(512, len(pool['queries'])), device='cuda')
    exact = prior.terms_for(embedding, pool, rows, tuple(parent['confidence']['1']['tuple']))
    surrogate = terms_for(embedding, pool, rows, tuple(parent['confidence']['1']['tuple']))
    for key in exact:
        torch.testing.assert_close(exact[key], surrogate[key], rtol=0, atol=0)
    loss = fusion_guard_loss(surrogate, True, True)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert parameters == 167561
    write_json(OUT / 'protocol.json', {**{k: parent[k] for k in ('train', 'val', 'confidence', 'initial',
        'pool_counts', 'updates_per_epoch', 'trainable_parameters')}, 'runtime': runtime, 'inputs': inputs,
        'config': CFG, 'test_used': False, 'preflight': {'forward_terms_bitwise_equal': True,
        'sequence': seq, 'finite_backward': True},
        'control': '이전 rank_all 및 rank_filtered_guarded의 동일 초기값·표본 순서·252업데이트 결과를 재사용한다.',
        'change': '매 순전파 현재 키의 min-max 폭을 다시 계산한다. 분모만 detach하고 '
                  '최솟값·정답/오답 점수 및 기존 손실을 유지한다. 실제 min-max의 미분이 아닌 대리 기울기다. '
                  '추론은 완전히 같은 코드·모듈·167561개 학습 매개변수다.',
        'selection': '각 조건 추가2/4에포크 중 검증 최종AP 최대. 동률 이른에포크. 같은에포크 차이도 보고. '
                     '시작 및 대응 대조 대비 AP+.1%p, F1/R100 손실≤.1%p, 계열AP 손실≤.25%p. '
                     '시험과 원고 교체 없음.',
        'limitations': '시드1·동일 학습자료·고정 후보의 미세조정이다. 방향 자료 부족은 해결하지 않는다.'})
    print('WIDTH_PREPARED', parameters, flush=True)


# 아래 공통 학습·평가 루프는 동결된 fusion_guard 구현과 같은 순서/연산을 사용한다.
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
