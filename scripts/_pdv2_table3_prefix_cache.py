#!/usr/bin/env python3
"""HeLiOS/ImLPR의 원본 5k 캐시를 감사하고 같은 10k 접두를 재사용한다.

5k: 동결 global_v2.cache를 그대로 호출하고 전후 입력 manifest를 남긴다.
10k: 감사가 완료된 5k의 5,140행을 복사하고 원본 Encoder로 5,000행만 추가한다.
원본 캐시·측정 소스·설정은 수정하지 않는다. GPU 실행은 부모의 슬롯에서만 한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results/pdv2_table3_20261029'
SAMPLES = 96


def read(path: Path) -> dict:
    """JSON 기록을 읽는다."""
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    """완료 JSON은 임시 파일에서 원자적으로 옮긴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix('.tmp.json')
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    part.replace(path)


def sha(path: Path) -> str:
    """파일 바이트의 SHA256이다."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    """dtype·shape와 배열 바이트를 함께 고정한다."""
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.dtype.str, value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def base(method: str, n_map: int) -> Path:
    """동결 어댑터가 이미 읽는 캐시 이름이다."""
    return OUT/f'{method}_batch1_v2_n{n_map}'


def preparation(method: str, n_map: int) -> Path:
    """기존 캐시 옆의 독립 준비 근거 폴더다."""
    return OUT/f'{method}_batch1_v2_n{n_map}_preparation'


def stat_record(path: Path) -> dict:
    """읽기만으로 변하지 않는 파일 식별·변경 상태다."""
    value = path.stat()
    return {'size': value.st_size, 'mtime_ns': value.st_mtime_ns,
            'ctime_ns': value.st_ctime_ns, 'device': value.st_dev, 'inode': value.st_ino}


def context(method: str, n_map: int) -> dict:
    """GPU 추론 없이 현재 동결 설정·키프레임·실제 로더 연결을 확인한다."""
    import _pdv2_table3_benchmark as common
    import _pdv2_table3_global_v2 as frozen
    from _pd_build_test_caches import _make_loader
    import inspect

    spec = common.read(OUT/'protocol.json')
    assert spec['sequence'] == 'mulran/Sejong01' and spec['samples'] == SAMPLES
    assert n_map in (5000, 10000)
    config = frozen.settings(method)
    assert config['common_protocol_sha256'] == sha(OUT/'protocol.json')
    inputs = {}
    for name in ('cache', 'spec'):
        path = REPO/spec['input_paths'][name]
        actual = sha(path)
        assert actual == spec['sources'][str(path.relative_to(REPO))], '공통 입력 캐시가 변경됐다'
        inputs[name] = {'path': str(path), 'sha256': actual}
    ending = n_map+44+SAMPLES
    with np.load(REPO/spec['input_paths']['cache']) as z:
        ids = z['scan_ids'][:ending].copy()
        poses = z['poses'][:ending].copy()
    with np.load(REPO/spec['input_paths']['spec']) as z:
        np.testing.assert_array_equal(z['scan_ids'][:ending], ids)
    assert len(ids) == ending and len(poses) == ending and np.all(np.diff(ids) > 0)
    loader = _make_loader('mulran', 'Sejong01')
    assert type(loader).__name__ == 'MulRanLoader'
    assert str(loader.sequence_path) == '/workspace/data/mulran/Sejong01', '동결 실행과 다른 raw 경로'
    loader_files = [REPO/'scripts/_pd_build_test_caches.py',
                    REPO/'scripts/_pd_build_val_layouts.py', Path(inspect.getfile(type(loader)))]
    return {'method': method, 'n_map': n_map, 'end': ending, 'ids': ids, 'poses': poses,
            'config': config, 'loader': loader, 'frozen': frozen,
            'invariants': {'method': method, 'sequence': spec['sequence'],
                           'common_protocol_sha256': sha(OUT/'protocol.json'),
                           'config': config, 'input_artifacts': inputs,
                           'raw_sequence_path': str(loader.sequence_path),
                           'resolved_raw_sequence_path': str(loader.sequence_path.resolve()),
                           'loader_sources': {str(p.relative_to(REPO)): sha(p) for p in loader_files},
                           'loader_class': type(loader).__module__+'.'+type(loader).__name__,
                           'loader_pose_catalog': {'path': str(loader.pose_file.resolve()),
                                                   'sha256': sha(loader.pose_file),
                                                   'stat': stat_record(loader.pose_file)},
                           'pose_time_tolerance_ns': loader.pose_time_tolerance_ns,
                           'raw_format': 'unchanged MulRanLoader float32 XYZI; reshape only; no per-sequence accumulation'}}


def manifest(ctx: dict) -> dict:
    """전체 접두 raw 바이트와 매핑을 읽기 감사한다. 시간은 준비 IO로만 기록한다."""
    started = time.perf_counter()
    records = []
    for row, sid in enumerate(ctx['ids']):
        path = ctx['loader'].scan_files[int(sid)]
        before = stat_record(path)
        digest = sha(path)
        assert stat_record(path) == before, f'읽는 중 raw 파일이 변경됐다: {path}'
        assert before['size'] % 16 == 0
        records.append({'row': row, 'scan_id': int(sid), 'path': str(path),
                        'resolved_path': str(path.resolve()), 'sha256': digest,
                        'timestamp_ns': int(ctx['loader'].scan_timestamps_ns[int(sid)]),
                        'stat': before})
        if row % 1000 == 0:
            print('PREFIX_RAW_AUDIT', ctx['method'], ctx['n_map'], row, ctx['end'], flush=True)
    return {'schema': 'pdv2-cache-input-manifest-v1',
            'identity': {'invariants': ctx['invariants'], 'frames': ctx['end'],
                         'scan_ids_sha256': array_sha(ctx['ids']),
                         'kiss_pose_prefix_sha256': array_sha(ctx['poses'])},
            'raw_files': records, 'audit_io_seconds': time.perf_counter()-started,
            'raw_bytes_hashed': sum(r['stat']['size'] for r in records)}


def compare_manifests(before: dict, after: dict) -> None:
    """원본 캐시 준비 전후 입력·설정·파일 바이트가 같아야 한다."""
    assert before['identity'] == after['identity'], '준비 중 입력/설정 연결이 변경됐다'
    assert before['raw_files'] == after['raw_files'], '준비 중 원시 점군 또는 파일 매핑이 변경됐다'


def assert_source_prefix(source_manifest: dict, target: dict, target_manifest: dict) -> None:
    """5k의 5,140개 원시 입력·KISS pose가 10k의 동일 접두인지 확인한다."""
    count = source_manifest['identity']['frames']
    assert count == 5140 and target['end'] == 10140
    assert source_manifest['identity']['invariants'] == target_manifest['identity']['invariants']
    assert array_sha(target['ids'][:count]) == source_manifest['identity']['scan_ids_sha256']
    assert array_sha(target['poses'][:count]) == source_manifest['identity']['kiss_pose_prefix_sha256']
    assert target_manifest['raw_files'][:count] == source_manifest['raw_files']


def validate_array(path: Path, frames: int, dim: int) -> None:
    """할당만 된 행이나 NaN/Inf가 있는 파일은 완료할 수 없다."""
    bank = np.load(path, mmap_mode='r')
    assert bank.shape == (frames, dim) and bank.dtype == np.float32
    for start in range(0, frames, 256):
        assert np.isfinite(bank[start:start+256]).all(), f'미생성/비정상 행: {start}'


def checked_cache(method: str, n_map: int, config: dict) -> tuple:
    """완료 JSON과 실제 NPY 형식·전체 해시를 확인한다."""
    path = base(method, n_map)
    meta = read(path.with_suffix('.json'))
    assert meta['config'] == config and meta['frames'] == n_map+140 and meta['batch'] == 1
    assert sha(path.with_suffix('.npy')) == meta['feature_sha256']
    validate_array(path.with_suffix('.npy'), meta['frames'], meta['dim'])
    return path, meta


def attested_source(method: str, config: dict) -> tuple:
    """감사 완료된 원본 5k 캐시만 재사용한다."""
    folder = preparation(method, 5000)
    proof = read(folder/'provenance.json')
    assert proof['status'] == 'passed' and proof['mode'] == 'unchanged_frozen_5k_cache'
    assert proof['preparer_source_sha256'] == sha(Path(__file__)), '감사 준비기 원천이 변경됐다'
    assert proof['config'] == config
    assert sha(folder/'before.json') == proof['before_manifest_sha256']
    assert sha(folder/'after.json') == proof['after_manifest_sha256']
    before, after = read(folder/'before.json'), read(folder/'after.json')
    compare_manifests(before, after)
    assert before['identity']['invariants']['config'] == config
    path, meta = checked_cache(method, 5000, config)
    assert sha(path.with_suffix('.json')) == proof['cache_metadata_sha256']
    assert meta['feature_sha256'] == proof['cache_feature_sha256']
    return path, meta, proof, after


def points_for(ctx: dict, row: int, recorded: dict) -> np.ndarray:
    """인코더에 전달하는 배열도 raw manifest와 바이트 단위로 대조한다."""
    points = np.ascontiguousarray(ctx['loader'][int(ctx['ids'][row])]['points'])
    assert points.dtype == np.float32 and points.ndim == 2 and points.shape[1] == 4
    assert hashlib.sha256(points.tobytes()).hexdigest() == recorded['sha256']
    return points


def prepare_5000(method: str, folder: Path, started: float) -> None:
    """5k 원본 cache 함수를 변경 없이 실행하고 별도 전후 근거만 남긴다."""
    ctx = context(method, 5000)
    before = manifest(ctx)
    write(folder/'before.json', before)
    tick = time.perf_counter()
    ctx['frozen'].cache(method, 5000)
    original_wall = time.perf_counter()-tick
    path, meta = checked_cache(method, 5000, ctx['config'])
    after = manifest(context(method, 5000))
    compare_manifests(before, after)
    write(folder/'after.json', after)
    proof = {'schema': 'pdv2-prefix-cache-provenance-v1', 'status': 'passed',
             'mode': 'unchanged_frozen_5k_cache', 'method': method, 'n_map': 5000,
             'preparer_source_sha256': sha(Path(__file__)),
             'before_manifest_sha256': sha(folder/'before.json'),
             'after_manifest_sha256': sha(folder/'after.json'),
             'cache_metadata_sha256': sha(path.with_suffix('.json')),
             'cache_feature_sha256': meta['feature_sha256'],
             'frames': meta['frames'], 'config': ctx['config'],
             'timing_seconds': {'original_cache_invocation_wall': original_wall,
                                'original_cache_reported_build': meta['seconds'],
                                'extra_audit_io_before': before['audit_io_seconds'],
                                'extra_audit_io_after': after['audit_io_seconds'],
                                'preparation_total_wall': time.perf_counter()-started},
             'scope': 'Original frozen global_v2.cache was called unchanged; original cache JSON/NPY were not edited. Raw byte hashes, mapping, KISS pose prefix, loader source, config and weights match before and after the original cache invocation.'}
    write(folder/'provenance.json', proof)


def prepare_10000(method: str, folder: Path, started: float) -> None:
    """감사된 접두를 복사하고 기존 Encoder로 나머지만 생성한다."""
    ctx = context(method, 10000)
    source_path, source_meta, source_proof, source_manifest = attested_source(method, ctx['config'])
    source_proof_path = preparation(method, 5000)/'provenance.json'
    source_proof_sha = sha(source_proof_path)
    before = manifest(ctx)
    assert_source_prefix(source_manifest, ctx, before)
    write(folder/'before.json', before)
    dim, count = source_meta['dim'], source_meta['frames']
    source = np.load(source_path.with_suffix('.npy'), mmap_mode='r')
    temporary = folder/'features.partial.npy'
    assert not temporary.exists(), '기존 부분 파일을 덮어쓰지 않는다'
    bank = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(ctx['end'], dim))
    bank[:] = np.nan
    written = np.zeros(ctx['end'], dtype=np.bool_)
    tick = time.perf_counter()
    bank[:count] = source
    bank.flush()
    np.testing.assert_array_equal(bank[:count], source)
    written[:count] = True
    copied_seconds = time.perf_counter()-tick
    frozen = ctx['frozen']
    frozen.configure()
    tick = time.perf_counter()
    enc = frozen.Encoder(method, ctx['config'])
    assert not any(part.training for part in enc.model.modules())
    model_init = time.perf_counter()-tick
    tick = time.perf_counter()
    first_new_arrival = ctx['n_map']+44
    points = points_for(ctx, first_new_arrival, before['raw_files'][first_new_arrival])
    preprocessing = enc.verify_preprocessing(points, int(ctx['ids'][first_new_arrival]))
    checks = []
    for row in np.unique(np.linspace(0, count-1, 12).round().astype(int)):
        points = points_for(ctx, int(row), before['raw_files'][int(row)])
        result = enc.encode(points, int(ctx['ids'][row]))
        np.testing.assert_allclose(result, source[row], atol=1e-6, rtol=1e-6)
        checks.append({'row': int(row), 'scan_id': int(ctx['ids'][row]),
                       'max_feature_error': float(np.max(np.abs(result-source[row])))})
    replay_seconds = time.perf_counter()-tick
    tick = time.perf_counter()
    for row in range(count, ctx['end']):
        points = points_for(ctx, row, before['raw_files'][row])
        result = enc.encode(points, int(ctx['ids'][row]))
        assert result.shape == (dim,) and np.isfinite(result).all()
        bank[row] = result
        written[row] = True
        if row % 100 == 0:
            write(folder/'progress.json', {'status': 'running', 'reused_rows': count,
                                          'computed_rows': row-count+1, 'required_new_rows': ctx['end']-count})
            print('PREFIX_SUFFIX_ENCODE', method, row, ctx['end'], round(time.perf_counter()-tick, 1), flush=True)
    suffix_seconds = time.perf_counter()-tick
    assert written.all() and written.sum() == ctx['end']
    np.testing.assert_array_equal(bank[:count], source)
    bank.flush()
    del bank
    validate_array(temporary, ctx['end'], dim)
    after = manifest(context(method, 10000))
    compare_manifests(before, after)
    assert_source_prefix(source_manifest, ctx, after)
    write(folder/'after.json', after)
    # 복사 후 원본 캐시와 원본 감사 근거도 변경되지 않았어야 한다.
    checked_cache(method, 5000, ctx['config'])
    assert sha(source_path.with_suffix('.json')) == source_proof['cache_metadata_sha256']
    assert sha(source_proof_path) == source_proof_sha
    final = base(method, 10000)
    assert not final.with_suffix('.npy').exists() and not final.with_suffix('.json').exists()
    feature_sha = sha(temporary)
    proof = {'schema': 'pdv2-prefix-cache-provenance-v1', 'status': 'passed',
             'mode': 'verified_prefix_reuse', 'method': method, 'n_map': 10000,
             'preparer_source_sha256': sha(Path(__file__)), 'config': ctx['config'],
             'source_cache_path': str(source_path.with_suffix('.npy').relative_to(REPO)),
             'source_cache_sha256': source_meta['feature_sha256'],
             'source_cache_metadata_sha256': source_proof['cache_metadata_sha256'],
             'source_provenance_path': str(source_proof_path.relative_to(REPO)),
             'source_provenance_sha256': source_proof_sha,
             'source_build_timing_seconds': source_proof['timing_seconds'],
             'before_manifest_sha256': sha(folder/'before.json'),
             'after_manifest_sha256': sha(folder/'after.json'),
             'feature_sha256': feature_sha, 'reused_rows': count, 'computed_rows': ctx['end']-count,
             'all_rows_written_and_finite': True, 'prefix_copy_exact': True,
             'prefix_raw_replay_checks': checks,
             'timing_seconds': {'prefix_copy': copied_seconds, 'encoder_initialization': model_init,
                                'prefix_replay_and_preprocessing_check': replay_seconds,
                                'new_suffix_raw_load_encode_store': suffix_seconds,
                                'extra_audit_io_before': before['audit_io_seconds'],
                                'extra_audit_io_after': after['audit_io_seconds'],
                                'preparation_total_wall': time.perf_counter()-started},
             'scope': 'Only initial cache construction reuses verified features. Same frozen Encoder, batch1, weights, raw loading and preprocessing. No measured update/query is skipped. Reused prefix is byte-identical; independent GPU reruns can have the already-diagnosed HeLiOS FP32 variation.'}
    write(folder/'provenance.json', proof)
    temporary.replace(final.with_suffix('.npy'))
    # 완료 JSON은 모든 행·입력 불변·원본 SHA 검사가 끝난 뒤 가장 마지막에 생성한다.
    write(final.with_suffix('.json'), {'config': ctx['config'], 'frames': ctx['end'],
          'feature_sha256': feature_sha, 'seconds': time.perf_counter()-started,
          'seconds_scope': 'This preparation invocation, including audit IO; original 5k build cost recorded separately.',
          'preprocessing_check': preprocessing, 'dim': dim, 'batch': 1,
          'cache_preparation': {'mode': 'verified_prefix_reuse',
                                'provenance_path': str((folder/'provenance.json').relative_to(REPO)),
                                'provenance_sha256': sha(folder/'provenance.json')}})


def build(method: str, n_map: int) -> None:
    """완료·기존 산출물을 보호하면서 한 크기의 준비를 수행한다."""
    started = time.perf_counter()
    path, folder = base(method, n_map), preparation(method, n_map)
    if (folder/'provenance.json').exists():
        proof = read(folder/'provenance.json')
        assert proof['status'] == 'passed'
        assert proof['preparer_source_sha256'] == sha(Path(__file__))
        ctx = context(method, n_map)
        _, meta = checked_cache(method, n_map, ctx['config'])
        assert sha(folder/'before.json') == proof['before_manifest_sha256']
        assert sha(folder/'after.json') == proof['after_manifest_sha256']
        compare_manifests(read(folder/'before.json'), read(folder/'after.json'))
        if n_map == 5000:
            attested_source(method, ctx['config'])
        else:
            assert proof['mode'] == 'verified_prefix_reuse'
            assert proof['feature_sha256'] == meta['feature_sha256']
            assert meta['cache_preparation']['provenance_sha256'] == sha(folder/'provenance.json')
            assert sha(REPO/proof['source_provenance_path']) == proof['source_provenance_sha256']
        print('PREFIX_CACHE_ALREADY_COMPLETE', method, n_map, flush=True)
        return
    assert not path.with_suffix('.json').exists(), '기존 완료 캐시에 사후 provenance를 붙이지 않는다'
    assert not path.with_suffix('.npy').exists(), '기존/부분 캐시를 덮어쓰지 않는다'
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder/'running.lock'
    with lock.open('x') as handle:
        handle.write(json.dumps({'pid': os.getpid(), 'method': method, 'n_map': n_map})+'\n')
    try:
        (prepare_5000 if n_map == 5000 else prepare_10000)(method, folder, started)
    except BaseException as error:
        write(folder/'failure.json', {'status': 'failed', 'error': repr(error),
                                     'completed_metadata_exists': path.with_suffix('.json').exists(),
                                     'note': 'Existing source artifacts are preserved; no missing row is marked completed.'})
        raise
    finally:
        lock.unlink(missing_ok=True)
    print('PREFIX_CACHE_COMPLETE', method, n_map, round(time.perf_counter()-started, 1), flush=True)


def selftest() -> None:
    """CPU 합성 캐시로 접두 복사·미생성 행·잘못된 shape/dtype를 확인한다."""
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)/'features.npy'
        source = np.arange(20, dtype=np.float32).reshape(5, 4)
        bank = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(9, 4))
        bank[:] = np.nan
        bank[:5] = source
        bank.flush()
        try:
            validate_array(path, 9, 4)
        except AssertionError:
            pass
        else:
            raise AssertionError('미생성 행을 완료로 잘못 처리했다')
        for i in range(5, 9):
            bank[i] = np.arange(4, dtype=np.float32)+i*4
        bank.flush()
        validate_array(path, 9, 4)
        np.testing.assert_array_equal(np.load(path)[:5], source)
        try:
            validate_array(path, 8, 4)
        except AssertionError:
            pass
        else:
            raise AssertionError('잘못된 행 수를 놓쳤다')
        wrong_dtype = Path(temp)/'wrong_dtype.npy'
        np.save(wrong_dtype, np.zeros((9, 4), np.float64))
        try:
            validate_array(wrong_dtype, 9, 4)
        except AssertionError:
            pass
        else:
            raise AssertionError('잘못된 dtype을 놓쳤다')
        before = {'identity': {'value': 1}, 'raw_files': [{'sha256': 'abc'}]}
        compare_manifests(before, dict(before))
        for bad in ({'identity': {'value': 2}, 'raw_files': before['raw_files']},
                    {'identity': before['identity'], 'raw_files': [{'sha256': 'changed'}]}):
            try:
                compare_manifests(before, bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('변경된 원천을 놓쳤다')
        target = {'end': 10140, 'ids': np.arange(10140, dtype=np.int64),
                  'poses': np.arange(10140, dtype=np.float32)[:, None]}
        source_manifest = {'identity': {'frames': 5140, 'invariants': {'config': 'same'},
                                       'scan_ids_sha256': array_sha(target['ids'][:5140]),
                                       'kiss_pose_prefix_sha256': array_sha(target['poses'][:5140])},
                           'raw_files': list(range(5140))}
        target_manifest = {'identity': {'invariants': {'config': 'same'}},
                           'raw_files': list(range(10140))}
        assert_source_prefix(source_manifest, target, target_manifest)
        for key in ('ids', 'poses'):
            bad_target = dict(target)
            bad_target[key] = target[key].copy()
            bad_target[key][0] += 1
            try:
                assert_source_prefix(source_manifest, bad_target, target_manifest)
            except AssertionError:
                pass
            else:
                raise AssertionError('다른 프레임/포즈 접두를 놓쳤다')
        target_manifest['raw_files'][0] = 'other raw file'
        try:
            assert_source_prefix(source_manifest, target, target_manifest)
        except AssertionError:
            pass
        else:
            raise AssertionError('다른 원시 파일 접두를 놓쳤다')
    print('PREFIX_CACHE_SELFTEST_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('helios', 'imlpr'))
    parser.add_argument('--n-map', type=int, choices=(5000, 10000))
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        if args.method is None or args.n_map is None:
            parser.error('--method와 --n-map을 지정한다')
        build(args.method, args.n_map)
