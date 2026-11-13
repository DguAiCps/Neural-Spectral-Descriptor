#!/usr/bin/env python3
"""학습8·검증9에서 후보·문맥 점수·정규화 효과를 교차해 최종 판정한다."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_geometry_interaction as prior
from _pdv2_matching_mine import cache_path, spec_path
from _pdv2_training_coverage import prefix_distance
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_fusion_factor_lib import fuse_with_range, pair_components
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_temporal_test_control_finish import scalar_score
from _train_attention_neighbors import array_hash

base = prior.base
OUT = REPO / 'results/pdv2_fusion_factor_20261023'
CONFIG = REPO / 'configs/pdv2_fusion_factor.yaml'
CFG = yaml.safe_load(CONFIG.read_text())
VARIANTS = tuple(CFG['variants'])
METHOD = CFG['final_score']


def protocol(inputs: bool = False) -> dict:
    """실험 전에 고정한 코드·입력 해시를 확인한다."""
    spec = base.read(OUT / 'protocol.json')
    for field in (('runtime', 'inputs') if inputs else ('runtime',)):
        for p, digest in spec[field].items():
            assert sha256(REPO / p) == digest, p
    return spec


def paths_for(role: str, sequence: str) -> dict:
    """학습 GT와 검증 KISS의 기존 캐시를 구분한다."""
    if role == 'train':
        return {'cache': cache_path(sequence), 'spec': spec_path(sequence)}
    return base.exp.old.base.inputs('val', sequence)[2]


def key_path(role: str, arm: str, sequence: str) -> Path:
    """고정 체크포인트의 검색 키 위치를 반환한다."""
    name = sequence.replace('/', '_') + '.npz'
    return (OUT / 'keys' / arm / name if role == 'train' else
            prior.OUT / 's1' / arm / 'epoch20' / ('keys_' + name))


def prepare() -> None:
    """두 모델과 여섯 진단 개입을 사전에 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        protocol(True)
        return
    old = prior.protocol(True)
    runtime, inputs = dict(old['runtime']), dict(old['inputs'])
    for p in (Path(__file__), CONFIG, REPO / 'scripts/_pdv2_fusion_factor_lib.py',
              REPO / 'scripts/_pdv2_fusion_factor_run.py', REPO / 'scripts/_pdv2_fusion_factor_finish.py',
              REPO / 'tests/retrieval/test_fusion_factor_diagnosis.py'):
        runtime[str(p.relative_to(REPO))] = sha256(p)
    for role in ('train', 'val'):
        for seq in old[role]:
            for p in paths_for(role, seq).values():
                inputs[str(p.relative_to(REPO))] = sha256(p)
            if role == 'train':
                m = base.read(spec_path(seq).with_suffix('.json'))
                assert m['sha256'] == sha256(spec_path(seq))
                assert m['cache_sha256'] == sha256(cache_path(seq))
            else:
                for arm in CFG['models']:
                    folder = prior.OUT / 's1' / arm / 'epoch20'
                    for name in (seq.replace('/', '_') + '.npz', 'keys_' + seq.replace('/', '_') + '.npz'):
                        p = folder / name
                        inputs[str(p.relative_to(REPO))] = sha256(p)
    for arm in CFG['models']:
        p = prior.OUT / 's1' / arm / 'checkpoints/checkpoint_epoch_20.pth'
        inputs[str(p.relative_to(REPO))] = sha256(p)
    for name in ('protocol.json', 'completion_validation.json', 'selection_details.json'):
        p = prior.OUT / name
        inputs[str(p.relative_to(REPO))] = sha256(p)
    write_json(OUT / 'protocol.json', {'runtime': runtime, 'inputs': inputs, 'config': CFG,
        **{k: old[k] for k in ('train', 'val', 'confidence', 'graph_hashes')}, 'test_used': False,
        'scope': '학습GT8·검증KISS9. 같은고정모델두개·같은q+15와j<=q-30. 새학습·선택·시험없음.',
        'variants': {'old': 'P0,K0,range(P0,K0)', 'new': 'P1,K1,range(P1,K1)',
            'old_pool_new_key': 'P0,K1,range(P0,K1)', 'new_pool_old_key': 'P1,K0,range(P1,K0)',
            'new_pool_old_ranges': 'P1,K1,range(P0,K1): 모든채널범위는이전후보목록. [0,1]밖값을자르지않음.',
            'new_pool_raw_key': 'P1,normalize(K1앞288),range(P1,raw): 후보는고정하고점수에서만문맥제거'},
        'interpretation': 'P0/1은두모델각각의100키+100보조후보. 전체점수교차진단은최종시간판정까지재계산하나새방법이나문맥제거재학습이아니다. 정답라벨은진단·집계에만사용. 학습진단은학습된장소의출력이며일반화주장안함.',
        'hypothesis': '문맥점수의변화와후보구성의변화가혼재한다. min-max채널폭은다른후보에의존하므로고정두후보의융합순위도바뀔수있다. 교차대조로정답확보·표현·정규화효과를분리한다.'})


def training_keys() -> None:
    """학습 때의 시간 그래프에서 두 고정 모델의 키만 추출한다."""
    spec = protocol(True)
    graph = torch.load(base.GRAPH, map_location='cpu', weights_only=False)
    graph = base.temporal_subgraph(graph, np.arange(graph.num_nodes))
    assert {k: array_hash(getattr(graph, k).numpy()) for k in spec['graph_hashes']} == spec['graph_hashes']
    state = torch.load(base.PREPARED, map_location='cpu', weights_only=False)
    np.testing.assert_array_equal(graph.x.numpy(), state['descriptors'])
    groups = np.asarray(state['sequences'])
    checks, outputs = {}, {}
    for arm in CFG['models']:
        net = prior.model(arm).cuda().eval()
        cp = prior.OUT / 's1' / arm / 'checkpoints/checkpoint_epoch_20.pth'
        net.load_state_dict(torch.load(cp, map_location='cpu', weights_only=False)['model_state_dict'])
        net.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(np.load(base.exp.old.base.RVEC)).float().cuda(), requires_grad=False)
        with torch.no_grad():
            key = base.normalize(net(graph.cuda()).float().cpu().numpy())
        assert key.shape == (72306, 416) and np.isfinite(key).all()
        checks[arm] = []
        for ordinal, seq in enumerate(spec['train']):
            ids = np.flatnonzero(groups == ordinal)
            assert (np.diff(ids) == 1).all()
            cache = base.exp.arrays(cache_path(seq))
            np.testing.assert_array_equal(cache['descriptors'], state['descriptors'][ids])
            np.testing.assert_array_equal(cache['poses'], state['poses'][ids])
            for q in (len(ids) // 3, 2 * len(ids) // 3):
                tiny, _ = base.make_keys(net, cache, arm, q + 16)
                error = float(abs(tiny[:q + 1] - key[ids[:q + 1]]).max())
                np.testing.assert_allclose(tiny[:q + 1], key[ids[:q + 1]], atol=4e-6, rtol=4e-5)
                checks[arm].append({'sequence': seq, 'q': q, 'key_max_error': error})
            path = key_path('train', arm, seq)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, embedding=key[ids], scan_ids=cache['scan_ids'])
            outputs[str(path.relative_to(REPO))] = sha256(path)
        del net
    write_json(OUT / 'training_keys.json', {'outputs': outputs, 'prefix_checks': checks, 'updates': 0, 'test_used': False})


def replay(role: str, sequence: str) -> None:
    """동일 질의에서 여섯 후보·점수 교차와 최종 판정을 계산한다."""
    spec = protocol()
    assert sequence in spec[role]
    started = time.time()
    folder = OUT / role / sequence.replace('/', '_')
    folder.mkdir(parents=True, exist_ok=True)
    paths = paths_for(role, sequence)
    cache = base.exp.arrays(paths['cache'])
    positions = cache['poses' if role == 'train' else 'poses_gt'][:, :3, 3].astype(np.float64)
    xyz = cache['poses'][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    keydata = [base.exp.arrays(key_path(role, arm, sequence)) for arm in CFG['models']]
    keys = [x['embedding'] for x in keydata]
    for x in keydata:
        np.testing.assert_array_equal(x['scan_ids'], cache['scan_ids'])
    raw_key = base.normalize(keys[1][:, :288])
    with np.load(paths['spec']) as z:
        np.testing.assert_array_equal(z['scan_ids'], cache['scan_ids'])
        bank = base.exp.old.base.bank_for(base.exp.OP, z)
    conf = spec['confidence']['1']
    weights = tuple(conf['tuple'])
    q = np.arange(30, len(keys[0]) - 15, dtype=np.int64)
    min_past = prefix_distance(positions)[q]
    n = len(q)
    ledgers = {v: {'query_idx': q, 'min_past_gt_dist': min_past,
        'j_star': np.empty(n, np.int64), 'top1_geo_dist': np.empty(n),
        'raw': np.empty(n), 'union_hit': np.empty(n, bool)} for v in VARIANTS}
    records, prefixes, scalar_checks = [], [], []
    expected = {}
    if role == 'val':
        for mode, arm in (('old', 'additive'), ('new', 'interaction')):
            expected[mode] = base.exp.arrays(prior.OUT / 's1' / arm / 'epoch20' / (sequence.replace('/', '_') + '.npz'))
            np.testing.assert_array_equal(q, expected[mode]['query_idx'])
            np.testing.assert_allclose(min_past, expected[mode]['min_past_gt_dist'], atol=1e-10, rtol=0)
    cohort = {'positive_with_candidate_but_wrong': 0, 'key_prefers_correct_over_final_wrong': 0,
              'raw_cosine_prefers_correct_but_full_key_not': 0, 'changed_winner': 0}
    for i, query in enumerate(q):
        t = int(query)
        aux = base.prefix_topk(bank[0], t, t - 29, 100)
        pools = [np.unique(np.r_[base.prefix_topk(key, t, t - 29, 100), aux]) for key in keys]
        union = np.union1d(*pools)
        assert union.max() <= t - 30
        indexes = [np.searchsorted(union, p) for p in pools]
        common = (bank[0][union] @ bank[0][t], base.align_scores(bank[1], bank[2], t, union),
                  base.column_align_scores(bank[3], t, union))
        channels = [np.column_stack((k[union] @ k[t], *common)) for k in keys]
        channels.append(np.column_stack((raw_key[union] @ raw_key[t], *common)))
        variant_specs = {'old': (0, 0, 0), 'new': (1, 1, 1), 'old_pool_new_key': (0, 1, 0),
            'new_pool_old_key': (1, 0, 1), 'new_pool_old_ranges': (1, 1, 0), 'new_pool_raw_key': (1, 2, 1)}
        scores, ranges = {}, {}
        distance = np.linalg.norm(positions[union] - positions[t], axis=1)
        for variant, (pool, key_id, range_pool) in variant_specs.items():
            ids = indexes[pool]
            score, span = fuse_with_range(channels[key_id][ids], weights, channels[key_id][indexes[range_pool]])
            winner = int(ids[np.argsort(-score, kind='stable')[0]])
            row = ledgers[variant]
            row['j_star'][i] = union[winner]
            row['top1_geo_dist'][i] = distance[winner]
            row['raw'][i] = channels[key_id][winner, 3 if conf['decision'] == 'c_col' else 2]
            row['union_hit'][i] = bool((distance[ids] <= 5).any())
            scores[variant], ranges[variant] = score, span
            if variant in expected:
                assert row['j_star'][i] == expected[variant]['j_star'][i], (sequence, variant, t)
                assert abs(row['raw'][i] - expected[variant]['raw'][i]) < 1e-7
        old_j, new_j = [int(ledgers[v]['j_star'][i]) for v in ('old', 'new')]
        old_ok, new_ok = [bool(ledgers[v]['top1_geo_dist'][i] <= 5) for v in ('old', 'new')]
        cohort['changed_winner'] += int(old_j != new_j)
        new_pool = indexes[1]
        true_ids = new_pool[distance[new_pool] <= 5]
        if min_past[i] <= 5 and len(true_ids) and not new_ok:
            cohort['positive_with_candidate_but_wrong'] += 1
            winner = int(np.searchsorted(union, new_j))
            full_margin = float(channels[1][true_ids, 0].max() - channels[1][winner, 0])
            raw_margin = float(channels[2][true_ids, 0].max() - channels[2][winner, 0])
            cohort['key_prefers_correct_over_final_wrong'] += int(full_margin > 0)
            cohort['raw_cosine_prefers_correct_but_full_key_not'] += int(raw_margin > 0 and full_margin <= 0)
        if old_ok != new_ok:
            true, false = (old_j, new_j) if old_ok else (new_j, old_j)
            pi, ni = np.searchsorted(union, [true, false])
            comp = {v: pair_components(channels[k][pi], channels[k][ni], ranges[v], weights).tolist()
                    for v, (_, k, _) in variant_specs.items()}
            margins = {}
            for label, key in zip(('old', 'new'), keys):
                x, p, neg = [key[j].astype(np.float64) for j in (t, true, false)]
                raw = float(x[:288] @ (p[:288] - neg[:288]))
                ctx = float(x[288:] @ (p[288:] - neg[288:]))
                total = float(x @ (p - neg))
                assert abs(raw + ctx - total) < 1e-12
                margins[label] = {'raw': raw, 'context': ctx, 'total': total}
            records.append({'q': t, 'change': 'loss' if old_ok else 'gain', 'true': true, 'false': false,
                'false_distance': float(distance[ni]), 'true_in_old': bool(true in pools[0]),
                'true_in_new': bool(true in pools[1]), 'false_in_old': bool(false in pools[0]),
                'margins': margins, 'fusion_pair_components': comp,
                'channel_widths': {v: r[:, 2].tolist() for v, r in ranges.items()},
                'winner_by_variant': {v: int(r['j_star'][i]) for v, r in ledgers.items()},
                'correct_by_variant': {v: bool(r['top1_geo_dist'][i] <= 5) for v, r in ledgers.items()}})
        if i in (0, n // 3, 2 * n // 3, n - 1):
            for v, (pool, key_id, range_pool) in variant_specs.items():
                # Scalar formula on the exact candidate list is independent of the helper.
                values = channels[key_id][indexes[pool]]
                reference = channels[key_id][indexes[range_pool]]
                scalar = np.array([sum(float(w) * (float(value[c]) - float(reference[:, c].min())) /
                    max(float(reference[:, c].max()) - float(reference[:, c].min()), 1e-8)
                    for c, w in enumerate(weights)) for value in values])
                assert np.max(abs(scalar - scores[v])) < 2e-6
                assert pools[pool][np.argmax(scalar)] == ledgers[v]['j_star'][i]
                scalar_checks.append({'q': t, 'variant': v, 'maximum_error': float(abs(scalar - scores[v]).max())})
    # 같은 q,j를 공유하는 개입의 시간 판정은 한 번만 계산한다.
    pair_arrays = np.concatenate([np.column_stack((q, row['j_star'])) for row in ledgers.values()])
    pairs, unique_indices, inverse = np.unique(pair_arrays, axis=0, return_index=True, return_inverse=True)
    raw_values = np.concatenate([row['raw'] for row in ledgers.values()])[unique_indices]
    final = np.concatenate([base.fixed.frozen_scores(bank, travel, pairs[s:s + 256, 0], pairs[s:s + 256, 1],
            conf['decision'], raw_values[s:s + 256])[METHOD] for s in range(0, len(pairs), 256)])
    for ordinal, (variant, row) in enumerate(ledgers.items()):
        row[METHOD] = final[inverse[ordinal * n:(ordinal + 1) * n]]
        if variant in expected:
            np.testing.assert_allclose(row[METHOD], expected[variant][METHOD], atol=1e-7, rtol=0)
        for i in (n // 3, 2 * n // 3):
            t, j = int(q[i]), int(row['j_star'][i])
            small = base.fixed.frozen_scores(base.exp.truncate_bank(bank, t + 16), travel[:t + 16],
                q[i:i + 1], row['j_star'][i:i + 1], conf['decision'], row['raw'][i:i + 1])[METHOD][0]
            direct = scalar_score(bank, travel, t, j, conf['decision'], float(row['raw'][i]))
            assert abs(small - row[METHOD][i]) < 1e-12 and abs(direct - row[METHOD][i]) < 1e-12
            prefixes.append({'variant': variant, 'q': t, 'score_error': float(abs(small - row[METHOD][i]))})
        np.savez_compressed(folder / (variant + '.npz'), **row)
    for case in records:
        i = int(case['q'] - 30)
        case['final_score_by_variant'] = {v: float(row[METHOD][i]) for v, row in ledgers.items()}
    outputs = {v + '.npz': sha256(folder / (v + '.npz')) for v in VARIANTS}
    write_json(folder / 'diagnosis.json', {'sequence': sequence, 'role': role, 'cohorts': cohort,
        'changed_correctness': records, 'prefix_checks': prefixes, 'scalar_checks': scalar_checks,
        'queries': n, 'positive': int((min_past <= 5).sum()), 'unique_final_pairs': len(pairs),
        'metrics': {v: independent_metrics(row, row[METHOD]) for v, row in ledgers.items()},
        'outputs': outputs, 'seconds': time.time() - started, 'test_used': False})
    print('REPLAY_DONE', role, sequence, n, round(time.time() - started, 1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'keys', 'replay'))
    parser.add_argument('--role', choices=('train', 'val'))
    parser.add_argument('--sequence')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.action == 'prepare':
        prepare()
    elif args.action == 'keys':
        training_keys()
    else:
        replay(args.role, args.sequence)
