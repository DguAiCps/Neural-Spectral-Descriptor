#!/usr/bin/env python3
"""시간 간선 입력 변형과 문맥 상대 점수 일치 학습을 비교한다."""
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
from _pdv2_fusion_guard_gradient import vector_gradient
from _pdv2_fusion_factor_lib import fuse_with_range
from _pdv2_input_alignment_lib import make_graph
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash
from gnn.temporal_views import temporal_view
from retrieval.context_consistency import context_pair_scores, relational_consistency

base, factor = prior.base, prior.factor
OUT = REPO / 'results/pdv2_temporal_augmentation_20261024'
CONFIG = REPO / 'configs/pdv2_temporal_augmentation.yaml'
CFG = yaml.safe_load(CONFIG.read_text())
METHOD = CFG['final_score']
model, initial_model, terms_for = prior.model, prior.initial_model, prior.terms_for
pool_path, load_pool = prior.pool_path, prior.load_pool


def protocol(inputs: bool = False) -> dict:
    """학습 전에 고정한 소스·입력을 검산한다."""
    spec = base.read(OUT / 'protocol.json')
    for field in (('runtime', 'inputs') if inputs else ('runtime',)):
        for path, digest in spec[field].items():
            assert sha256(REPO / path) == digest, path
    return spec


def view_setting(epoch: int, ordinal: int, batch: int) -> tuple:
    """조건 간 동일한 간선 변형과 시드를 반환한다."""
    return CFG['views'][(epoch + batch) % 2], 100000 * epoch + 1000 * ordinal + batch


def consistency(clean: torch.Tensor, changed: torch.Tensor, pool: dict,
                rows: torch.Tensor) -> torch.Tensor:
    """원래 입력의 현재 상대 문맥 점수를 변형 입력의 목표로 쓴다."""
    q, candidates, valid = pool['queries'][rows], pool['candidates'][rows], pool['valid'][rows]
    return relational_consistency(context_pair_scores(clean, q, candidates),
        context_pair_scores(changed, q, candidates), valid, CFG['consistency_temperature']).mean()


def prepare() -> None:
    """학습 자료의 초기 기울기만으로 일치 손실 계수를 정하고 실험을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        protocol(True)
        return
    parent = prior.protocol(True)
    runtime, inputs = dict(parent['runtime']), dict(parent['inputs'])
    for name in ('scripts/_pdv2_temporal_augmentation.py', 'scripts/_pdv2_temporal_augmentation_run.py',
                 'scripts/_pdv2_temporal_augmentation_finish.py', 'scripts/_pdv2_temporal_augmentation_probe.py',
                 'scripts/_pdv2_fusion_guard_gradient.py', 'src/gnn/temporal_views.py',
                 'src/retrieval/context_consistency.py', 'tests/gnn/test_temporal_views.py',
                 'tests/retrieval/test_context_consistency.py', 'configs/pdv2_temporal_augmentation.yaml'):
        runtime[name] = sha256(REPO / name)
    for path in (prior.OUT / 'summary.json', prior.OUT / 'rank_all/training_complete.json'):
        inputs[str(path.relative_to(REPO))] = sha256(path)
    write_json(OUT / 'plan.json', {'runtime': runtime, 'inputs': inputs, 'config': CFG,
        'coefficient_selection': '학습 첫에포크63배치를 초기 가중치에 고정한다. 정상/변형 평균 순위 '
        '손실 기울기 평균과 일치 손실 기울기 평균의 노름 비×.25를 [.01,100]으로 자른다. '
        '검증 결과로 계수를 조정하지 않는다.', 'test_used': False})
    net = initial_model()
    initial = {k: array_hash(v.detach().cpu().numpy()) for k, v in net.state_dict().items()}
    assert initial == parent['initial']
    params = [p for p in net.parameters() if p.requires_grad]
    dimension = sum(p.numel() for p in params)
    assert dimension == 167561
    gradients = np.zeros((2, dimension), np.float64)
    records = []
    for ordinal, sequence in enumerate(parent['train']):
        n = parent['pool_counts'][sequence]['queries']
        if not n:
            continue
        graph = torch.load(pool_path(sequence) / 'graph.pt', map_location='cuda', weights_only=False)
        pool = load_pool(sequence)
        order = np.random.default_rng(1000 + ordinal).permutation(n)
        for batch, start in enumerate(range(0, n, CFG['batch_queries'])):
            rows = torch.as_tensor(order[start:start+CFG['batch_queries']], device='cuda')
            mode, seed = view_setting(1, ordinal, batch)
            view = temporal_view(graph, mode, seed)
            clean, changed = F.normalize(net(graph), dim=1), F.normalize(net(view), dim=1)
            if start == 0:
                reference = base.exp.arrays(factor.key_path('train', 'interaction', sequence))['embedding']
                np.testing.assert_allclose(clean.detach().cpu().numpy(), reference, atol=4e-6, rtol=4e-5)
            rank = .5 * (terms_for(clean, pool, rows, tuple(parent['confidence']['1']['tuple']))['rank'].mean()
                + terms_for(changed, pool, rows, tuple(parent['confidence']['1']['tuple']))['rank'].mean())
            agreement = consistency(clean, changed, pool, rows)
            gr = vector_gradient(rank, params, True)
            gc = vector_gradient(agreement, params, False)
            gradients += torch.stack((gr, gc)).detach().double().cpu().numpy()
            records.append({'sequence': sequence, 'batch': batch, 'view': mode, 'seed': seed,
                'rank': float(rank.detach()), 'consistency': float(agreement.detach()),
                'rank_gradient_norm': float(gr.norm()), 'consistency_gradient_norm': float(gc.norm()),
                'edges': graph.num_edges, 'view_edges': view.num_edges})
            del clean, changed, view, rank, agreement, gr, gc
        del graph, pool
        print('CALIBRATION_DONE', sequence, flush=True)
    assert len(records) == parent['updates_per_epoch'] == 63
    gradients /= len(records)
    norms = np.linalg.norm(gradients, axis=1)
    assert norms[1] > 1e-12
    raw_coefficient = CFG['consistency_gradient_ratio'] * norms[0] / norms[1]
    coefficient = float(np.clip(raw_coefficient, *CFG['consistency_coefficient_clip']))
    calibration = {'coefficient': coefficient, 'unclipped_coefficient': float(raw_coefficient),
        'rank_gradient_norm': float(norms[0]), 'consistency_gradient_norm': float(norms[1]),
        'cosine': float(gradients[0] @ gradients[1] / norms[0] / norms[1]), 'batches': records,
        'parameter_dimension': dimension, 'new_updates': 0, 'test_used': False}
    write_json(OUT / 'calibration.json', calibration)
    np.savez_compressed(OUT / 'calibration_gradients.npz', rank=gradients[0], consistency=gradients[1])
    for path in (OUT / 'plan.json', OUT / 'calibration.json', OUT / 'calibration_gradients.npz'):
        inputs[str(path.relative_to(REPO))] = sha256(path)
    write_json(OUT / 'protocol.json', {**{k: parent[k] for k in ('train', 'val', 'confidence', 'initial',
        'pool_counts', 'updates_per_epoch', 'trainable_parameters')}, 'runtime': runtime, 'inputs': inputs,
        'config': CFG, 'consistency_coefficient': coefficient, 'test_used': False,
        'initial_checkpoint': str(prior.INITIAL.relative_to(REPO)), 'initial_sha256': sha256(prior.INITIAL),
        'arms': {'clean_rank': '정상 입력 순위 손실', 'augmented_rank': '정상/변형 순위 손실의 절반씩 합',
            'augmented_consistent': '정상/변형 순위 평균+고정 계수×상대 문맥 점수 일치'},
        'mask': 'drop은±1보존·나머지무방향쌍약절반, stride는±2/±4. 순서·특징·자세는 유지. '
                '새 간선·새 관측·추론 모듈 없음. 스캔 삭제나 실제 반대편 관측 합성으로 부르지 않는다.',
        'budget': '세 조건 모두 같은 초기값·질의·순서·252업데이트·두 번 순전파. 대조군은 변형 출력에 '
                  '학습 손실을 주지 않는다. 변형 조건은 추가 역전파 계산을 사용하며 같은 FLOP 대조는 아니다.',
        'selection': '추가2/4에포크의 정상 입력 최종 검증AP 최대, 동률 이른 에포크. '
                     '시작과 clean_rank 각각 대비 AP+.1%p, F1/R100 손실≤.1%p, 계열AP 손실≤.25%p. '
                     '일치 항의 독립효과는 augmented_rank와도 비교한다. 시험·원고 교체 없음.',
        'limitations': '후보와보조점수는시작모델로고정. 정답없는질의·실제반대편관측은추가하지않음. '
                       '고정 초기 기울기 비는 실제 Adam 갱신 중 기여율을 보장하지 않음.'})
    protocol(True)
    print('AUGMENTATION_PREPARED', coefficient, norms.tolist(), flush=True)


def train(arm: str) -> None:
    """기존 모델을 같은 예산으로 추가 학습하고 실제 변형 순서를 보존한다."""
    spec = protocol(True)
    folder = OUT / arm
    folder.mkdir(exist_ok=True)
    assert not list(folder.glob('epoch*.pth')), '부분 학습 결과를 덮어쓰지 않는다'
    torch.manual_seed(1)
    net = initial_model()
    initial = {k: array_hash(v.detach().cpu().numpy()) for k, v in net.state_dict().items()}
    assert initial == spec['initial']
    fixed = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()
             if 'running_' in k or 'num_batches_tracked' in k or k.endswith('key_scale')}
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                                  lr=CFG['learning_rate'], weight_decay=CFG['weight_decay'])
    updates, history, orders, view_orders = 0, [], [], []
    started = time.time()
    for epoch in range(1, CFG['epochs']+1):
        used = 0
        sums = np.zeros(4)
        grad_max = 0.
        tick = time.time()
        for ordinal, sequence in enumerate(spec['train']):
            n = spec['pool_counts'][sequence]['queries']
            if not n:
                continue
            graph = torch.load(pool_path(sequence) / 'graph.pt', map_location='cuda', weights_only=False)
            pool = load_pool(sequence)
            order = np.random.default_rng(1000*epoch+ordinal).permutation(n)
            orders.append({'epoch': epoch, 'sequence': sequence, 'sha256': array_hash(order)})
            for batch, start in enumerate(range(0, n, CFG['batch_queries'])):
                rows = torch.as_tensor(order[start:start+CFG['batch_queries']], device='cuda')
                mode, seed = view_setting(epoch, ordinal, batch)
                view = temporal_view(graph, mode, seed)
                view_orders.append({'epoch': epoch, 'sequence': sequence, 'batch': batch, 'mode': mode,
                    'seed': seed, 'edges_sha256': array_hash(view.edge_index.detach().cpu().numpy())})
                optimizer.zero_grad(set_to_none=True)
                clean = F.normalize(net(graph), dim=1)
                # 정상 대조에서도 같은 변형 순전파를 실행하되 손실에는 넣지 않는다.
                with torch.set_grad_enabled(arm != 'clean_rank'):
                    changed = F.normalize(net(view), dim=1)
                    augmented = terms_for(changed, pool, rows, tuple(spec['confidence']['1']['tuple']))['rank'].mean()
                ordinary = terms_for(clean, pool, rows, tuple(spec['confidence']['1']['tuple']))['rank'].mean()
                agreement = consistency(clean, changed, pool, rows)
                loss = ordinary if arm == 'clean_rank' else .5*(ordinary+augmented)
                if arm == 'augmented_consistent':
                    loss = loss + spec['consistency_coefficient']*agreement
                assert torch.isfinite(loss)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), CFG['gradient_clip'])
                assert torch.isfinite(norm)
                optimizer.step()
                updates += 1
                used += len(rows)
                sums += np.array([float(x.detach()) for x in (loss, ordinary, augmented, agreement)])*len(rows)
                grad_max = max(grad_max, float(norm))
                del clean, changed, view, ordinary, augmented, agreement, loss
            del graph, pool
        assert updates == epoch*spec['updates_per_epoch']
        record = {'epoch': epoch, 'updates': updates, 'queries': used, 'loss': sums[0]/used,
            'clean_rank_loss': sums[1]/used, 'augmented_rank_loss': sums[2]/used,
            'consistency_loss': sums[3]/used, 'gradient_norm_max': grad_max, 'seconds': time.time()-tick}
        history.append(record)
        write_json(folder / 'training_progress.json', record)
        print('EPOCH', arm, record, flush=True)
        if epoch in CFG['checkpoint_epochs']:
            state = {k: v.detach().cpu() for k, v in net.state_dict().items() if not k.endswith('key_scale')}
            assert all(torch.isfinite(v).all() for v in state.values())
            torch.save({'model_state_dict': state, 'epoch': epoch, 'updates': updates,
                'protocol_sha256': sha256(OUT / 'protocol.json')}, folder / f'epoch{epoch}.pth')
    for k, v in fixed.items():
        torch.testing.assert_close(net.state_dict()[k].cpu(), v, atol=0, rtol=0)
    write_json(folder / 'training_complete.json', {'initial': initial, 'orders': orders, 'view_orders': view_orders,
        'updates': updates, 'history': history, 'seconds': time.time()-started,
        'checkpoints': {p.name: sha256(p) for p in folder.glob('epoch*.pth')},
        'bn_and_remetric_frozen': True, 'trainable_parameters': spec['trainable_parameters'], 'test_used': False})


def view_keys(net, cache: dict, view: str = 'clean', end: int = None) -> tuple:
    """q+15 범위 안의 기존 시간 그래프에만 간선 변형을 적용한다."""
    if view == 'clean':
        return base.make_keys(net, cache, 'diff', end)
    raw, pose = cache['descriptors'][:end], cache['poses'][:end]
    graph = temporal_view(make_graph(raw, pose, np.zeros(len(raw), np.int64)), view, CFG['diagnostic_view_seed'])
    with torch.no_grad():
        key = base.normalize(net(graph.cuda()).float().cpu().numpy())
    return key, None


def evaluate(arm: str, epoch: int, role: str, sequence: str, view: str = 'clean') -> None:
    """전체 질의에서 후보를 다시 검색하고 기존 최종 시간 판정을 계산한다."""
    spec = protocol()
    assert sequence in spec[role]
    checkpoint = prior.INITIAL if arm == 'initial' else OUT / arm / f'epoch{epoch}.pth'
    if arm != 'initial':
        assert sha256(checkpoint) == base.read(OUT / arm / 'training_complete.json')['checkpoints'][checkpoint.name]
    folder = (OUT / arm / f'epoch{epoch}' / role if view == 'clean' else
              OUT / 'stress' / arm / f'epoch{epoch}' / view / role)
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
    key, _ = view_keys(net, cache, view)
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
        small, _ = view_keys(net, cache, view, end)
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
        'role': role, 'sequence': sequence, 'arm': arm, 'epoch': epoch, 'view': view, 'queries': len(q),
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
    parser.add_argument('--arm', choices=['initial'] + CFG['arms'])
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--role', choices=('train', 'val'))
    parser.add_argument('--sequence')
    parser.add_argument('--view', choices=['clean'] + CFG['views'], default='clean')
    args = parser.parse_args()
    torch.set_num_threads(4 if args.action in ('train', 'prepare') else 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.action == 'prepare':
        prepare()
    elif args.action == 'train':
        train(args.arm)
    elif args.action == 'evaluate':
        evaluate(args.arm, args.epoch, args.role, args.sequence, args.view)
    else:
        select()
