#!/usr/bin/env python3
"""고정 시험 스캔 쌍의 여섯 방법 매칭 점수를 재생성한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('SEQOT_EVAL_WINDOW', 'centered')
os.environ.setdefault('SEED', '1')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'src'), str(REPO)]
OUT = REPO / 'results/pdv2_pair_visualization_20261030'
SEQUENCES = ('kitti/08', 'mulran/DCC01', 'nclt/2012-01-08')
METHODS = ('scpp', 'ringsharp', 'helios', 'seqot', 'bevplace', 'nsd')
LEDGERS = {
    'scpp': ('pdv2_scpp_official_hsens/apc/odom', 'dump_scpp', 'score_scpp'),
    'ringsharp': ('pdv2_ring_official/odom', 'dump_ringsharp', 'score_trans'),
    'helios': ('pdv2_helios_official/odom', 'dump_helios', 'score_cos'),
    'seqot': ('pdv2_seqot_official_centered/odom', 'dump_seqotl2', 'score_l2'),
    'bevplace': ('pdv2_bevplace_official/odom', 'dump_bevplace', 'score_negl2'),
    'nsd': ('pdv2_spectral_whitening_20261026/fixed_test/s1', '', 'score'),
}
POSITIVE, RANDOM, HARD, VERIFY, MATRIX = 1, 2, 4, 8, 16


def read(path: Path) -> dict:
    """JSON을 읽는다."""
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    """분석 기록을 원자적으로 보존한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path: Path) -> str:
    """파일의 SHA256을 구한다."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def folder(sequence: str) -> Path:
    """시퀀스별 진단 폴더다."""
    return OUT / sequence.replace('/', '_')


def ledger(method: str, sequence: str) -> tuple:
    """표 2의 시드 1 원장과 점수 이름을 읽는다."""
    root, prefix, score = LEDGERS[method]
    name = sequence.replace('/', '_')
    path = REPO / 'results' / root / ((prefix + '_' if prefix else '') + name + '.npz')
    with np.load(path) as data:
        values = {k: data[k] for k in data.files}
    return values, score, path


def prepare() -> None:
    """방법 점수를 새로 계산하기 전에 동일 쌍과 그림 위치를 고정한다."""
    if (OUT / 'protocol.json').exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'created_unix': time.time(), 'seed': 1, 'sequences': list(SEQUENCES),
        'methods': list(METHODS), 'query_count_per_sequence': 64, 'random_negatives_per_query': 8,
        'selection': '각 시퀀스 재방문 질의의 시간 순서 균등 64개. 정답은 과거 5m 안 최근접 두 개. 어려운 오답은 여섯 고정 방법의 실제 최종 승자 중 25m 초과의 합집합. 무작위 오답은 과거 25m 초과에서 RNG 20261030으로 8개.',
        'matrix_selection': 'KITTI08의 q<2000 재방문 중 점수와 무관하게 GT 30m 간격으로 처음 다섯 장소. 장소별 질의 두 개·참조 두 개. 모든 참조가 모든 질의보다 30키프레임 이상 과거.',
        'pair_scope': '후보 생성 없이 고정된 동일 쌍에 각 방법의 매칭 점수를 적용하는 진단. 표2의 최종 검색 성능이나 전체 시험셋 평가를 대체하지 않음.',
        'scpp_pair': '주어진 참조의 세 A-PC 중 retrieval-key 거리가 가장 작은 augmentation 선택 후 공식 sector-key 정렬 및 search_ratio=0 거리. 전역 K=1 후보 검색은 생략.',
        'ringsharp_pair': '각 쌍의 공식 yaw 추정과 rotate-then-trans_cnn 두 yaw의 최소 translation error. top1000 후보 제한은 생략.',
        'nsd_pair': '주어진 후보에 현재 c_col 시간 판정과 75% 원점수+25% 상관보정 적용. 후보 생성·승자 선택은 생략. 원점수도 별도 보존.',
        'matrix_colour': '각 질의의 표시된 참조 안 점수 순위 백분율. 동점 평균 순위. 모든 방법 동일 0..1. 확률이나 원점수 크기가 아님.',
        'distribution': '원래 점수 축. 정답·무작위 오답·여섯 방법 합집합 어려운 오답을 구분. 질의별 각 집단 총 가중치 동일. 표본 범위·시드1 명시.',
        'models_changed': False, 'test_used_for_model_selection': False,
        'sources': {}, 'sequences_info': {},
    }
    for sequence in SEQUENCES:
        target = folder(sequence)
        target.mkdir(exist_ok=True)
        ls = {m: ledger(m, sequence) for m in METHODS}
        meta = json.loads(str(ls['bevplace'][0]['meta']))
        cp = REPO / meta['caches']['operating_cache']
        with np.load(cp) as data:
            xyz = data['poses_gt'][:, :3, 3]
            scan_ids = data['scan_ids']
        common = ls['nsd'][0]
        available = common['query_idx'][common['min_past_gt_dist'] <= 5]
        qs = available[np.linspace(0, len(available)-1, 64).round().astype(int)]
        assert len(np.unique(qs)) == 64
        rng = np.random.default_rng(20261030)
        pairs = {}
        reference = {m: [] for m in METHODS}

        def add(q: int, j: int, flag: int) -> None:
            assert 0 <= j <= q-30 and q+15 < len(xyz), (sequence, q, j)
            pairs[(int(q), int(j))] = pairs.get((int(q), int(j)), 0) | flag

        for q in qs:
            distances = np.linalg.norm(xyz[:q-29]-xyz[q], axis=1)
            positives = np.flatnonzero(distances <= 5)
            for j in positives[np.argsort(distances[positives], kind='stable')[:2]]:
                add(q, j, POSITIVE)
            negatives = np.flatnonzero(distances > 25)
            for j in rng.choice(negatives, 8, replace=False):
                add(q, j, RANDOM)
            for method, (values, score, path) in ls.items():
                ix = np.searchsorted(values['query_idx'], q)
                assert values['query_idx'][ix] == q
                j = int(values['j_star'][ix])
                if j >= 0 and np.isfinite(values[score][ix]):
                    add(q, j, VERIFY)
                    reference[method].append([int(q), j, float(values[score][ix])])
                    if distances[j] > 25:
                        add(q, j, HARD)
        matrix = None
        if sequence == 'kitti/08':
            centres, mq, mj = [], [], []
            for q in available[available < 2000]:
                if centres and np.min(np.linalg.norm(xyz[centres]-xyz[q], axis=1)) < 30:
                    continue
                all_past_end = (mq[0] if mq else q)-29
                dd = np.linalg.norm(xyz[:all_past_end]-xyz[q], axis=1)
                js = np.flatnonzero(dd <= 3)
                if len(js) < 2:
                    continue
                js = js[np.argsort(dd[js], kind='stable')[:2]]
                if np.max(np.linalg.norm(xyz[js]-xyz[q+1], axis=1)) > 5:
                    continue
                centres.append(int(q)); mq.extend([int(q), int(q+1)]); mj.extend(map(int, js))
                if len(centres) == 5:
                    break
            assert len(centres) == 5, centres
            for q in mq:
                for j in mj:
                    add(q, j, MATRIX)
            matrix = {'centres': centres, 'query': mq, 'reference': mj,
                      'gt_distance': np.linalg.norm(xyz[mq, None]-xyz[np.array(mj)[None]], axis=2).tolist()}
            write(target / 'matrix.json', matrix)
        keys = sorted(pairs)
        q, j = np.array(keys, dtype=np.int64).T
        flags = np.array([pairs[k] for k in keys], dtype=np.uint8)
        distances = np.linalg.norm(xyz[q]-xyz[j], axis=1)
        indices = np.unique(np.r_[q, j])
        np.savez_compressed(target / 'pairs.npz', query=q, reference=j, flags=flags,
                            distance=distances, indices=indices, scan_ids=scan_ids[indices], selected_queries=qs)
        write(target / 'ledger_replay.json', reference)
        protocol['sources'][str(cp.relative_to(REPO))] = sha(cp)
        for values, score, path in ls.values():
            protocol['sources'][str(path.relative_to(REPO))] = sha(path)
        protocol['sources'][str((target/'pairs.npz').relative_to(REPO))] = sha(target/'pairs.npz')
        protocol['sequences_info'][sequence] = {
            'cache': str(cp.relative_to(REPO)), 'pairs': len(q), 'frames': len(indices),
            'counts': {name: int(np.sum(flags & bit != 0)) for name, bit in
                       [('positive', POSITIVE), ('random', RANDOM), ('hard', HARD), ('matrix', MATRIX)]},
            'artifacts': {m: json.loads(str(v[0]['meta'])).get('artifacts', {})
                          for m, v in ls.items() if 'meta' in v[0]},
        }
        print('FIXED', sequence, protocol['sequences_info'][sequence]['counts'], 'frames', len(indices), flush=True)
    write(OUT / 'protocol.json', protocol)


def inputs(sequence: str) -> tuple:
    """고정한 스캔 쌍과 프레임 순서를 읽는다."""
    target = folder(sequence)
    with np.load(target / 'pairs.npz') as data:
        pairs = {k: data[k] for k in data.files}
    lookup = {int(i): k for k, i in enumerate(pairs['indices'])}
    return target, pairs, lookup


def configure() -> None:
    """공식 FP32 추론의 재현 설정을 적용한다."""
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def save_scores(sequence: str, method: str, scores: np.ndarray, extra: dict = None,
                tolerance: float = 2e-4) -> None:
    """실제 최종 승자 점수를 원장과 재대조한 뒤 저장한다."""
    target, pairs, _ = inputs(sequence)
    assert scores.shape == pairs['query'].shape and np.isfinite(scores).all()
    lookup = {(int(q), int(j)): i for i, (q, j) in enumerate(zip(pairs['query'], pairs['reference']))}
    checks = read(target / 'ledger_replay.json')[method]
    errors = [abs(float(scores[lookup[(q, j)]])-v) for q, j, v in checks]
    maximum = max(errors)
    assert maximum <= tolerance, (sequence, method, maximum, tolerance)
    np.savez_compressed(target / f'scores_{method}.npz', score=scores)
    write(target / f'scores_{method}.json', {
        'status': 'passed', 'pairs_sha256': sha(target / 'pairs.npz'),
        'score_sha256': sha(target / f'scores_{method}.npz'),
        'ledger_replays': len(checks), 'max_score_error': maximum,
        'tolerance': tolerance, 'source_sha256': sha(Path(__file__)), **(extra or {})})
    print('SCORES', sequence, method, len(scores), 'replay_max_error', maximum, flush=True)


def raw(sequence: str) -> None:
    """고정 프레임만 공식 데이터 로더에서 읽어 MATLAB과 그림 입력을 만든다."""
    from _pd_build_test_caches import _make_loader
    target, pairs, _ = inputs(sequence)
    if (target/'raw_complete.json').exists():
        return
    sensor, seq = sequence.split('/')
    loader = _make_loader(sensor, seq)
    counts = []
    thumbnails = {}
    matrix = read(target/'matrix.json') if (target/'matrix.json').exists() else None
    show = set(matrix['query'] + matrix['reference']) if matrix else set()
    with (target/'chunk_000000.xyz').open('wb') as xyz:
        for count, (kf, sid) in enumerate(zip(pairs['indices'], pairs['scan_ids'])):
            points = np.asarray(loader[int(sid)]['points'], dtype=np.float32)
            np.ascontiguousarray(points[:, :3]).tofile(xyz)
            counts.append(len(points))
            if int(kf) in show:
                thumbnails[str(kf)] = points[:, :3]
            if count % 200 == 0:
                print('RAW', sequence, count, len(pairs['indices']), flush=True)
    np.array(counts, dtype=np.int64).tofile(target/'chunk_000000.cnt')
    if thumbnails:
        np.savez_compressed(target/'thumbnail_points.npz', **thumbnails)
    write(target/'raw_complete.json', {'frames': len(counts), 'scan_ids': pairs['scan_ids'].tolist(),
                                      'points': int(sum(counts)), 'source': 'original loader xyz float32'})


def nsd(sequence: str) -> None:
    """학습 통계와 시간 판정을 그대로 고정 쌍에 적용한다."""
    import _pdv2_spectral_whitening as source
    target, pairs, _ = inputs(sequence)
    q, j = pairs['query'], pairs['reference']
    _, _, paths, _ = source.prior.BASE.exp.old.base.inputs('test', sequence)
    cache = source.prior.BASE.exp.arrays(paths['cache'])
    pos = cache['poses'][:, :3, 3]
    travel = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
    x = source.coefficients(paths['spec'])
    with np.load(source.OUT/'statistics.npz') as data:
        matrix = data['full']
    spec = read(source.OUT/'fixed_test/freeze.json')
    channel = spec['confidence']['1']['decision']
    scores = []
    for features in (x, source.transform(x, matrix)):
        bank = source.variant_inputs(features, 8)
        chunks = []
        for start in range(0, len(q), 256):
            qq, jj = q[start:start+256], j[start:start+256]
            peak = source.prior.BASE.fixed.correlation_profiles(bank, qq, jj, channel).max(1)
            value = source.prior.BASE.fixed.frozen_scores(bank, travel, qq, jj, channel, peak)[source.METHOD]
            chunks.append(value)
        scores.append(np.concatenate(chunks))
    np.savez_compressed(target/'nsd_components.npz', original=scores[0], corrected=scores[1])
    save_scores(sequence, 'nsd', .75*scores[0]+.25*scores[1], {'channel': channel}, 2e-6)


def vector_method(sequence: str, method: str) -> None:
    """공식 모델과 전처리로 선택된 스캔의 전역 특징을 계산한다."""
    import torch
    from _pd_build_test_caches import _make_loader
    configure()
    target, pairs, lookup = inputs(sequence)
    sensor, seq = sequence.split('/')
    checkpoint_info = read(OUT/'protocol.json')['sequences_info'][sequence]['artifacts'][method]
    feature_path = target/f'features_{method}.npy'
    if feature_path.exists():
        descriptors = np.load(feature_path)
    elif method == 'seqot' and (OUT/'canonical_seqot'/f'{sensor}_{seq}.npz').exists():
        canonical = OUT/'canonical_seqot'/f'{sensor}_{seq}.npz'
        with np.load(canonical) as data:
            assert str(data['stage1_sha256']) == checkpoint_info['stage1']['sha256']
            assert str(data['stage2_sha256']) == checkpoint_info['stage2']['sha256']
            np.testing.assert_array_equal(data['kf_scan_ids'][pairs['indices']], pairs['scan_ids'])
            descriptors = data['final'][pairs['indices']]
        write(target/'canonical_seqot.json', {'path': str(canonical.relative_to(REPO)),
              'sha256': sha(canonical), 'source': 'cps2 selected seed1 test descriptor archive; both checkpoint hashes and scan ids checked'})
    elif method == 'bevplace':
        # 공식 dump는 NVIDIA 컨테이너의 기본 TF32 matmul/conv를 사용했다.
        # 이를 끄면 affine_grid 등에서 원장 점수와 약 0.002 차이가 난다.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        import cv2
        import _bevplace_official_dump as dump
        weight = REPO/checkpoint_info['checkpoint']
        assert sha(weight) == checkpoint_info['checkpoint_sha256']
        _, _, model = dump._official_model(weight, 1, 1, 'cuda')
        model.eval()
        meta = json.loads(str(ledger(method, sequence)[0]['meta']))
        root = OUT/'canonical_bev'/f'{sensor}_{seq}'/'bev_imgs'
        assert all((root/f'{sid:06d}.png').exists() for sid in pairs['scan_ids'])
        write(target/'canonical_bev.json', {'source': 'cps2 original table2 PNG inputs',
              'files': {str(p.relative_to(REPO)): sha(p) for p in sorted(root.glob('*.png'))}})
        import _bevplace_official_bev as bev
        loader = _make_loader(sensor, seq)
        get_bev = bev.official_getBEV()
        descriptors = []
        for start in range(0, len(pairs['scan_ids']), 16):
            ids = pairs['scan_ids'][start:start+16]
            images = []
            for sid in ids:
                image_path = root/f'{sid:06d}.png'
                if image_path.exists():
                    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                else:
                    image = bev.bev_image(loader[int(sid)]['points'], get_bev)
                    ok, encoded = cv2.imencode('.png', image)
                    assert ok
                    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
                assert image is not None
                images.append(image)
            array = np.stack(images).astype(np.float32)[:, None].repeat(3, axis=1)/256
            with torch.no_grad():
                descriptors.append(model(torch.from_numpy(array).cuda())[2].cpu().numpy())
        descriptors = np.concatenate(descriptors)
    elif method == 'helios':
        import _pdv2_table3_global_v2 as global_encoder
        import _helios_official_prep as prep
        raw_path = checkpoint_info['checkpoint'].split('Neural-Spectral-Codec/')[-1]
        weight = REPO/raw_path
        assert sha(weight) == checkpoint_info['checkpoint_sha256']
        encoder = global_encoder.Encoder('helios', {'weight': str(weight)})
        loader = _make_loader(sensor, seq)
        descriptors = []
        for n, sid in enumerate(pairs['scan_ids']):
            points = loader[int(sid)]['points']
            value, _ = prep.process_scan(encoder.toolbox, points, prep.fill_rng(sensor, seq, int(sid)))
            descriptors.append(encoder.forward(np.ascontiguousarray(value[:, :3])))
            if n % 100 == 0:
                print('ENCODE', method, sequence, n, len(pairs['indices']), flush=True)
        descriptors = np.stack(descriptors)
    else:
        import _seqot_official_train as official
        from _seqot_official_prep import range_image
        p1, a1 = official._chosen('stage1'); p2, a2 = official._chosen('stage2')
        assert a1['sha256'] == checkpoint_info['stage1']['sha256']
        assert a2['sha256'] == checkpoint_info['stage2']['sha256']
        fe = official._load_fe(p1, 'cuda'); gem = official._load_gem(p2, 'cuda')
        loader = _make_loader(sensor, seq)

        class RawSource:
            """공식 창 규칙을 그대로 쓰는 선택 원시 스캔 투영 입력이다."""
            n_raw = len(loader)

            def exists(self, r):
                return 0 <= r < self.n_raw

            def rows(self, raws):
                return np.stack([images[int(i)] for i in raws])

        src = RawSource()
        need = official.centered_need(src, pairs['scan_ids'])
        windows = official.subdesc_windows(need, src.exists, 3, 'centered')
        raw_ids = np.unique(windows)
        images = {}
        for n, sid in enumerate(raw_ids):
            images[int(sid)] = range_image(sensor, loader[int(sid)]['points'])
            if n % 500 == 0:
                print('RANGE', sequence, n, len(raw_ids), flush=True)
        descriptors, _, _ = official.final_descriptors(fe, gem, src, pairs['scan_ids'], 'cuda', batch=32)
    np.save(feature_path, descriptors.astype(np.float32))
    a = descriptors[[lookup[int(i)] for i in pairs['query']]]
    b = descriptors[[lookup[int(i)] for i in pairs['reference']]]
    if method == 'helios':
        score = np.sum(a*b, axis=1)
    else:
        score = -np.linalg.norm(a-b, axis=1)
    save_scores(sequence, method, score, {'dimension': descriptors.shape[1], 'feature_sha256': sha(feature_path),
                'tf32_for_bevplace': method == 'bevplace',
                'numeric_note': 'BEVPlace++는 원래 dump의 NVIDIA 컨테이너 TF32 기본 설정을 재현한다. 다른 재추론은 FP32.'})


def ringsharp(sequence: str) -> None:
    """공식 회전·이동 매칭을 고정 쌍 전체에 실행한다."""
    import torch
    import _ringsharp_official as official
    from _pd_build_test_caches import _make_loader
    from _pdv2_table3_ringsharp_bounded import translation_branch
    configure()
    target, pairs, lookup = inputs(sequence)
    sensor, seq = sequence.split('/')
    info = read(OUT/'protocol.json')['sequences_info'][sequence]['artifacts']['ringsharp']
    weight = REPO/info['weight'].split('Neural-Spectral-Codec/')[-1]
    assert sha(weight) == info['weight_sha256']
    base = official._base()
    model, trans = base.load_model(str(weight), 'cuda')
    model.eval(); trans.eval()
    feature_path = target/'features_ringsharp.npz'
    if feature_path.exists():
        with np.load(feature_path) as data:
            specs = torch.from_numpy(data['spec']).cuda()
            bts = torch.from_numpy(data['bt']).cuda()
            packed = data['packed']
    else:
        loader = _make_loader(sensor, seq)
        # 공식 출력 연산은 그대로 두고 이미 표3에서 검증한 배치1로
        # WSL GPU 메모리 압박을 줄인다. 최종 점수는 원장과 재대조한다.
        official.FEAT_BATCH = 1
        stream = official.bev_batches_from_loader(base, loader, pairs['scan_ids'], official.rol.ringsharp_bounds(sensor))
        def progress_stream():
            for number, batch in enumerate(stream):
                if number % 64 == 0:
                    print('ENCODE', sequence, 'ringsharp', number, len(pairs['indices']), flush=True)
                yield batch
        packed_list = []
        _, specs, bts = official.features(base, model, trans, official.tee_packed(progress_stream(), packed_list), len(pairs['indices']), 'cuda')
        packed = np.concatenate(packed_list)
        np.savez(feature_path, spec=specs.cpu().numpy(), bt=bts.cpu().numpy(), packed=packed)
        specs, bts = specs.cuda(), bts.cuda()
    scores = np.empty(len(pairs['query']), np.float32)
    checks = []
    for n, q in enumerate(np.unique(pairs['query'])):
        rows = np.flatnonzero(pairs['query'] == q)
        candidates = np.array([lookup[int(j)] for j in pairs['reference'][rows]])
        qi = lookup[int(q)]
        _, angles = official.rotation_branch(base, specs, specs[qi:qi+1], candidates)
        qfeat = official.query_features(base, model, packed[qi:qi+1], 'cuda')
        scores[rows] = -translation_branch(base, trans, bts, qfeat, candidates, angles)
        if n == 0:
            ref = official.translation_branch(base, trans, bts, qfeat, candidates[:4], angles[:4], True, dedup=False)
            error = float(np.max(abs(-scores[rows[:4]]-ref)))
            assert error < 1e-5
            checks.append(error)
        if n % 16 == 0:
            print('PAIR_SCORE', sequence, 'ringsharp', n, flush=True)
    save_scores(sequence, 'ringsharp', scores, {'official_translation_checks': checks,
                                             'feature_sha256': sha(feature_path)})


def collect_scpp(sequence: str) -> None:
    """공식 MATLAB의 쌍 점수를 동일 원장과 대조한다."""
    target, pairs, _ = inputs(sequence)
    scores = np.loadtxt(target/'scpp_scores.csv', delimiter=',', skiprows=1)
    np.testing.assert_array_equal(scores[:, 0].astype(int), pairs['query'])
    np.testing.assert_array_equal(scores[:, 1].astype(int), pairs['reference'])
    save_scores(sequence, 'scpp', scores[:, 2], {'implementation': 'official MATLAB A-PC functions'}, 1e-10)


def main() -> None:
    """독립 단계별 실행으로 캐시를 보존하며 실패 지점부터 재개한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'raw', 'nsd', 'bevplace', 'helios', 'seqot', 'ringsharp', 'collect_scpp'])
    parser.add_argument('--sequence', choices=SEQUENCES)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(); return
    sequence = args.sequence
    assert sequence
    if args.action in METHODS and (folder(sequence)/f'scores_{args.action}.json').exists():
        print('VERIFIED_CACHE', sequence, args.action); return
    if args.action in ('bevplace', 'helios', 'seqot'):
        vector_method(sequence, args.action)
    else:
        globals()[args.action](sequence)


if __name__ == '__main__':
    main()
