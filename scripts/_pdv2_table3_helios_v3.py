#!/usr/bin/env python3
"""HeLiOS 비용은 그대로 재고 같은 실행의 검색과 재추론 오차를 구분한다.

공식 추론과 v2 초기 캐시를 유지한다. 서로 다른 GPU 추론의 점수를
동일 비트로 요구하지 않고, 실제 실행 bank의 검색은 엄격하게 검증한다.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import resource
import time

import _pdv2_table3_benchmark as common
import _pdv2_table3_global_v2 as frozen
import numpy as np
import torch


def protocol(n_map: int) -> dict:
    """수정하지 않은 v2·캐시와 별도의 검증 개정을 고정한다."""
    diagnostic = common.OUT / 'helios_diagnostic_n1000' / 'complete.json'
    report = common.read(diagnostic)
    assert report['status'] == 'complete' and all(report['checks'].values())
    assert report['frozen_winner_changes'] == 0
    assert report['frozen_adapter_sha256'] == common.sha(Path(frozen.__file__))
    assert report['frozen_protocol_sha256'] == common.sha(common.OUT/'helios_protocol_v2.json')
    assert report['diagnostic_source_sha256'] == common.sha(common.REPO/'scripts/_pdv2_table3_helios_diagnostic.py')
    config = frozen.settings('helios')
    cache = frozen.target('helios', n_map)
    meta = common.read(cache.with_suffix('.json'))
    assert meta['config'] == config
    assert common.sha(cache.with_suffix('.npy')) == meta['feature_sha256']
    if n_map == 1000:
        assert meta['feature_sha256'] == report['frozen_feature_cache_sha256']
    names = [Path(__file__), Path(frozen.__file__), Path(common.__file__),
             common.REPO/'scripts/_pdv2_table3_model_memory.py',
             common.REPO/'scripts/_pdv2_table3_helios_diagnostic.py']
    evidence = [diagnostic, *sorted(diagnostic.parent.glob('replay_*.json')),
                diagnostic.parent/'fixed_batch.json',
                diagnostic.parent/'raw_replay_features.npy',
                diagnostic.parent/'fixed_batch_features.npy',
                common.OUT/'helios_batch1_v2_n1000.json']
    value = {'schema': 'pdv2-table3-helios-protocol-v3', 'revision': 3,
             'parent_protocol_sha256': common.sha(common.OUT/'protocol.json'),
             'v2_protocol_sha256': common.sha(common.OUT/'helios_protocol_v2.json'),
             'v2_config': config, 'weight_sha256': config['weight_sha256'],
             'sources': {str(p.relative_to(common.REPO)): common.sha(p) for p in names},
             'diagnostic_artifacts': {str(p.relative_to(common.REPO)): common.sha(p) for p in evidence},
             'diagnostic_sha256': common.sha(diagnostic),
             'diagnostic_reference_cache_sha256': report['frozen_feature_cache_sha256'],
             'execution': 'Unmodified v2 Runtime and official model/input/FP32 computation; unchanged v2 reference cache.',
             'verification': 'After each timed repeat, all96 actual feature rows retain original atol=rtol=1e-6 checks; all96 recorded winners/dot scores must exactly equal same-execution full FP32 search and same-execution KDTree top1. Frozen-cache winners must match; double-precision score drift must fit the measured feature L2 perturbation bound, and every frozen top1/2 margin must exceed twice that bound. Cross-inference confidence drift is reported, not asserted <1e-7.',
             'timing': 'Same raw encode, state append and query stages as v2; independent validation occurs only after each full96 timed arrivals and memory snapshot.'}
    target = common.OUT/'helios_protocol_v3.json'
    if target.exists():
        assert common.read(target) == value, '이미 동결한 v3 원천이 달라졌다'
    else:
        common.write(target, value)
    return value


def validate_query(bank: np.ndarray, reference: np.ndarray, q: int, n: int,
                   answer: tuple, *, bank64=None, source64=None,
                   errors=None, source_norm=None) -> dict:
    """같은 실행 검색의 정확성과 독립 추론 간 차이를 각각 검증한다."""
    if bank64 is None:
        bank64 = bank.astype(np.float64)
        source64 = reference.astype(np.float64)
        errors = np.linalg.norm(bank64-source64, axis=1)
        source_norm = np.linalg.norm(source64, axis=1)
    actual32 = bank[:n] @ bank[q]
    j = int(np.argmax(actual32))
    # 같은 배열에서 같은 dot를 검산하므로 허용오차를 넓히지 않는다.
    assert j == int(answer[0]), ('same-execution winner', q, j, answer[0])
    exact = float(bank[j] @ bank[q])
    assert float(answer[1]) == exact, ('same-execution score', q, answer[1], exact)
    frozen32 = reference[:n] @ reference[q]
    frozen_j = int(np.argmax(frozen32))
    assert j == frozen_j, ('frozen winner changed', q, j, frozen_j)
    actual64 = bank64[:n] @ bank64[q]
    frozen64 = source64[:n] @ source64[q]
    order = np.argsort(-frozen64, kind='stable')
    assert j == int(np.argmax(actual64)) == int(order[0]), ('precision changes winner', q)
    margin = float(frozen64[order[0]]-frozen64[order[1]])
    bound = errors[:n]*source_norm[q]+source_norm[:n]*errors[q]+errors[:n]*errors[q]
    drift = np.abs(actual64-frozen64)
    # 1e-12는 float64 검산의 덧셈 반올림 여유다. FP32 판정 완화에 쓰지 않는다.
    assert float(np.max(drift-bound)) < 1e-12, ('perturbation identity', q)
    assert margin > 2*float(np.max(bound)), ('winner margin not certified', q, margin, float(np.max(bound)))
    from sklearn.neighbors import KDTree
    _, kd_j = KDTree(bank[:n]).query(bank[q:q+1], k=1)
    assert j == int(kd_j[0, 0]), ('same-execution KDTree', q, j, kd_j)
    return {'query': q, 'winner': j, 'same_execution_score_error': 0.0,
            'same_execution_KDTree_match': True, 'frozen_winner_match': True,
            'frozen_score_difference': abs(exact-float(reference[frozen_j] @ reference[q])),
            'max_candidate_score_difference32': float(np.max(np.abs(actual32-frozen32))),
            'max_candidate_score_difference64': float(np.max(drift)),
            'max_score_perturbation_bound64': float(np.max(bound)),
            'frozen_top12_margin64': margin, 'margin_certified': True}


def validate_execution(runtime, rows: list) -> dict:
    """최종 bank의 과거 prefix로 방금 기록한 모든 도착을 독립 검산한다."""
    work = runtime.work
    reference = np.asarray(runtime.source)
    np.testing.assert_allclose(runtime.bank[:runtime.n], reference[:runtime.n], atol=1e-6, rtol=1e-6)
    source64 = reference.astype(np.float64)
    bank64 = runtime.bank.astype(np.float64)
    errors = np.linalg.norm(bank64-source64, axis=1)
    norms = np.linalg.norm(source64, axis=1)
    checks = [validate_query(runtime.bank, reference, row['query'], row['n_candidates'],
                             (row['winner'], row['confidence']), bank64=bank64,
                             source64=source64, errors=errors, source_norm=norms)
              for row in rows]
    return {'status': 'passed', 'queries': checks, 'raw_feature_checks': work.end-work.initial,
            'raw_feature_max_abs': float(np.max(np.abs(runtime.bank[work.initial:]-reference[work.initial:]))),
            'raw_feature_max_l2': float(np.max(errors[work.initial:])),
            'verification_fixture_bytes': int(bank64.nbytes+source64.nbytes+errors.nbytes+norms.nbytes),
            'fixture_note': 'CPU double-precision validation arrays are allocated after the timed repeat and its memory snapshot; excluded from method resident storage; process peak RSS can include prior validation.'}


def measure(n_map: int = 1000) -> None:
    """공통 96도착×3회 전체 비용을 재고 독립 검증을 마친 뒤 게시한다."""
    frozen.configure()
    config = protocol(n_map)
    work = common.Workload(common.read(common.OUT/'protocol.json'), n_map)
    started = time.perf_counter()
    runtime = frozen.Runtime('helios', work)
    common.sync()
    initial_s = time.perf_counter()-started
    for i, points in enumerate(work.points[:common.WARMUP]):
        value = runtime.encode(points)
        np.testing.assert_allclose(value, runtime.source[work.initial+i], atol=1e-6, rtol=1e-6)
        runtime.append(value)
        runtime.query(work.initial+i-15)
    rows, memories, validations = [], [], []
    for repeat in range(common.REPEATS):
        runtime.reset()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = common.rss()
        current = []
        for i, points in enumerate(work.points):
            t = work.initial+i
            encoded, enc = common.timed(lambda: runtime.encode(points))
            _, state = common.timed(lambda: runtime.append(encoded))
            answer, query = common.timed(lambda: runtime.query(t-15))
            current.append({'repeat': repeat, 'arrival': t, 'query': t-15, 'n_candidates': t-44,
                            'encode': enc, 'state_update': state, 'update': enc+state,
                            'query_ms': query, 'processing': enc+state+query,
                            'winner': answer[0], 'confidence': answer[1]})
        mem = runtime.memory()
        mem.update({'process_rss_before': before, 'process_rss_after': common.rss(),
                    'process_peak_rss': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    'cuda_allocated': torch.cuda.memory_allocated(), 'cuda_reserved': torch.cuda.memory_reserved(),
                    'cuda_peak_allocated': torch.cuda.max_memory_allocated(),
                    'cuda_peak_reserved': torch.cuda.max_memory_reserved()})
        memories.append(mem)
        validations.append(validate_execution(runtime, current))
        rows.extend(current)
        common.write(common.OUT/'verification'/f'helios_v3_n{n_map}_repeat{repeat}.json', validations[-1])
        print('HELIOS_V3_MEASURE_VERIFIED', n_map, repeat, flush=True)
    repeat_drift = []
    for i in range(common.SAMPLES):
        sample = [rows[r*common.SAMPLES+i] for r in range(common.REPEATS)]
        assert len({row['winner'] for row in sample}) == 1
        repeat_drift.append(float(np.ptp([row['confidence'] for row in sample])))
    verification = {'status': 'passed', 'revision': 3,
                    'raw_batch1_feature_checks': common.SAMPLES*common.REPEATS,
                    'same_execution_queries': common.SAMPLES*common.REPEATS,
                    'official_KDTree_checks': common.SAMPLES*common.REPEATS,
                    'frozen_winners_changed': 0, 'all_margins_certified': True,
                    'max_frozen_score_difference': max(c['frozen_score_difference'] for v in validations for c in v['queries']),
                    'max_repeat_confidence_range': max(repeat_drift),
                    'diagnostic_sha256': config['diagnostic_sha256'],
                    'preprocessing': runtime.meta['preprocessing_check'],
                    'validation_repeats': validations}
    result = {'schema': 'pdv2-table3-measurement-v1', 'method': 'helios', 'n_map': n_map,
              'status': 'passed', 'publish_ready': True, 'verification': verification,
              'protocol_sha256': common.sha(common.OUT/'protocol.json'),
              'source_sha256': common.sha(Path(__file__)), 'method_config': config,
              'method_protocol_sha256': common.sha(common.OUT/'helios_protocol_v3.json'),
              'sequence': common.SEQUENCE, 'seed': 1, 'candidate_range': [n_map, n_map+common.SAMPLES-1],
              'mean_candidates': n_map+(common.SAMPLES-1)/2, 'samples': common.SAMPLES,
              'repeats': common.REPEATS, 'warmup': common.WARMUP,
              'timings_ms': {key: common.stats([r[field] for r in rows]) for key, field in
                             (('encode', 'encode'), ('state_update', 'state_update'), ('update', 'update'),
                              ('query', 'query_ms'), ('processing', 'processing'))},
              'resident_bytes': memories[-1], 'memory_repeats': memories, 'raw_samples': rows,
              'raw_input_fixture_bytes': work.input_bytes, 'raw_input_sha256': work.input_hash,
              'reference_cache': {'path': str(frozen.target('helios', n_map).with_suffix('.npy').relative_to(common.REPO)),
                                  'sha256': runtime.meta['feature_sha256'],
                                  'meta_sha256': common.sha(frozen.target('helios', n_map).with_suffix('.json'))},
              'initial_cached_state_build_s': initial_s, 'completed_unix': time.time(),
              'environment': {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                              'threads': 1, 'tf32': False},
              'scope': 'Unchanged frozen official HeLiOS model and raw preprocessing; original v2 cache; exact same-execution retrieval; independently inferred cached confidence drift audited through feature error and winner margin.'}
    common.write(common.OUT/'verification'/f'helios_n{n_map}.json', verification)
    common.write(common.OUT/'methods'/f'helios_n{n_map}.json', result)
    common.summarize()
    print('HELIOS_V3_COMPLETE', result['timings_ms'], flush=True)


def selftest() -> None:
    """작은 재추론 차이는 분리하고 실제 잘못된 승자·점수는 거부한다."""
    reference = np.array([[1., 0., 0.], [0., 1., 0.], [.8, .6, 0.]], np.float32)
    actual = reference.copy()
    actual[2, 0] = np.nextafter(actual[2, 0], np.float32(1.))
    valid = (0, float(actual[0] @ actual[2]))
    good = validate_query(actual, reference, 2, 2, valid)
    assert good['frozen_score_difference'] > 0
    for bad in ((1, valid[1]), (0, valid[1]+1e-4)):
        try:
            validate_query(actual, reference, 2, 2, bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('잘못된 실제 실행의 승자/점수를 놓쳤다')
    unsafe = reference.copy()
    unsafe[2, 0] = .95
    try:
        validate_query(unsafe, reference, 2, 2, (0, float(unsafe[0] @ unsafe[2])))
    except AssertionError as error:
        assert 'winner margin not certified' in str(error)
    else:
        raise AssertionError('승자는 같지만 너무 큰 특징 변동을 놓쳤다')
    print('HELIOS_V3_SELFTEST_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-map', type=int, default=1000)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    selftest() if args.selftest else measure(args.n_map)
