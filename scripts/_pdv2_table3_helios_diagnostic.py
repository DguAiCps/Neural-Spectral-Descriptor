#!/usr/bin/env python3
"""동결 HeLiOS의 입력·추론 재현성과 검색 오류를 분리해서 진단한다.

정확 시간 계측이 아니다. 공식 모델/양자화/파라미터와 v2 소스는 바꾸지
않고, 96개 실제 도착을 3회 재생한 뒤 같은 GPU batch를 3회씩 재사용한다.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import os
from pathlib import Path
import time

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['SEED'] = '1'

import numpy as np
import torch
import _pdv2_table3_benchmark as common
import _pdv2_table3_global_v2 as frozen


def array_hash(value: np.ndarray) -> str:
    """배열의 dtype·shape·원시 바이트를 함께 확인한다."""
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.dtype.str, value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def state_hash(model) -> str:
    """진단 전후 등록 모델 상태가 바뀌지 않았는지 확인한다."""
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def distance(actual: np.ndarray, reference: np.ndarray) -> dict:
    """특징의 최대 절대차와 L2 차이를 float64로 계산한다."""
    delta = actual.astype(np.float64)-reference.astype(np.float64)
    return {'max_abs': float(np.max(np.abs(delta))),
            'l2': float(np.linalg.norm(delta)),
            'exact': bool(np.array_equal(actual, reference))}


def ranked(scores: np.ndarray) -> tuple:
    """진단용 완전 정렬의 승자·차순위와 간격이다."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    return int(order[0]), int(order[1]), float(scores[order[0]]-scores[order[1]])


def diagnose(n_map: int = 1000) -> None:
    """입력 반복, 실제 검색, 같은 batch 반복을 독립 기록한다."""
    frozen.configure()
    out = common.OUT / f'helios_diagnostic_n{n_map}'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'complete.json').exists(), '기존 완료 진단을 덮어쓰지 않는다'
    work = common.Workload(common.read(common.OUT / 'protocol.json'), n_map)
    runtime = frozen.Runtime('helios', work)
    model = runtime.encoder.model
    assert not any(part.training for part in model.modules()), '모든 모듈이 eval이어야 한다'
    before_state = state_hash(model)
    source = np.asarray(runtime.source)
    source64 = source.astype(np.float64)
    source_norm = np.linalg.norm(source64, axis=1)
    references = []
    for i in range(common.SAMPLES):
        q, n = work.initial+i-15, work.n_map+i
        scores32 = source[:n] @ source[q]
        scores64 = source64[:n] @ source64[q]
        j32, j2, margin32 = ranked(scores32)
        j64, _, margin64 = ranked(scores64)
        references.append({'winner': j32, 'runner_up': j2,
                           'score': float(source[j32] @ source[q]),
                           'margin32': margin32, 'winner64': j64, 'margin64': margin64,
                           'scores32': scores32, 'scores64': scores64})
    captures = {}

    def capture_batch(unused_model, arguments):
        batch = arguments[0]
        captures.clear()
        for key in ('coords', 'features'):
            captures[key] = batch[key].detach().cpu().numpy().copy()

    handle = model.register_forward_pre_hook(capture_batch)
    # 동일 입력으로 warmup하되 v2와 똑같이 다음 replay에서 초기 상태를 복원한다.
    for i in range(common.WARMUP):
        runtime.append(runtime.encode(work.points[i]))
        runtime.query(work.initial+i-15)
    inputs, batches = [], []
    features = np.empty((common.REPEATS, common.SAMPLES, source.shape[1]), np.float32)
    input_checks, query_rows = [], []
    tick = time.time()
    from sklearn.neighbors import KDTree
    import MinkowskiEngine as ME
    for repeat in range(common.REPEATS):
        runtime.reset()
        gc.collect()
        torch.cuda.empty_cache()
        bank64 = source64.copy()
        errors = np.zeros(work.end, np.float64)
        for i, points in enumerate(work.points):
            t, q, n = work.initial+i, work.initial+i-15, work.n_map+i
            sid = int(work.ids[t])
            value = runtime.encoder.input(points, sid)
            output = runtime.encoder.forward(value)
            features[repeat, i] = output
            if repeat == 0:
                inputs.append(value.copy())
                batches.append({k: v.copy() for k, v in captures.items()})
            # 원본 compute_embedding이 받은 batch를 hook으로 관찰하며 수정하지 않는다.
            coords, indices = runtime.encoder.params.model_params.quantizer(torch.tensor(value))
            independently_batched = ME.utils.batched_coordinates([coords]).numpy()
            check = {'repeat': repeat, 'arrival': t, 'scan_id': sid,
                     'input_sha256': array_hash(value),
                     'input_equal_first': bool(np.array_equal(value, inputs[i])),
                     'coords_sha256': array_hash(captures['coords']),
                     'coords_equal_first': bool(np.array_equal(captures['coords'], batches[i]['coords'])),
                     'coords_equal_independent_quantizer': bool(np.array_equal(captures['coords'], independently_batched)),
                     'quantizer_index_sha256': array_hash(indices.numpy()),
                     'features_equal_first': bool(np.array_equal(captures['features'], batches[i]['features'])),
                     'feature_vs_cache': distance(output, source[t]),
                     'feature_vs_first_replay': distance(output, features[0, i]),
                     'output_norm': float(np.linalg.norm(output.astype(np.float64)))}
            input_checks.append(check)
            runtime.append(output)
            answer = runtime.query(q)
            bank64[t] = output.astype(np.float64)
            errors[t] = check['feature_vs_cache']['l2']
            # 같은 실행에서 얻은 bank로 독립 완전 검색과 공식 Euclidean KDTree를 수행한다.
            live32 = runtime.bank[:n] @ runtime.bank[q]
            live64 = bank64[:n] @ bank64[q]
            j32, runner_up, margin32 = ranked(live32)
            j64, _, margin64 = ranked(live64)
            _, kd_indices = KDTree(runtime.bank[:n]).query(runtime.bank[q:q+1], k=1)
            ref = references[i]
            # 특징 차이만으로 허용되는 각 내적의 Cauchy-Schwarz 상한이다.
            error_bound = (errors[:n]*source_norm[q]+source_norm[:n]*errors[q]
                           + errors[:n]*errors[q])
            score_drift64 = np.abs(live64-ref['scores64'])
            row = {'repeat': repeat, 'arrival': t, 'query': q, 'n_candidates': n,
                   'winner': int(answer[0]), 'score': float(answer[1]),
                   'same_execution_winner32': j32, 'same_execution_winner64': j64,
                   'same_execution_kdtree_winner': int(kd_indices[0, 0]),
                   'same_execution_exact_dot_error': abs(float(answer[1])-float(runtime.bank[answer[0]] @ runtime.bank[q])),
                   'same_execution_runner_up': runner_up, 'same_execution_margin32': margin32,
                   'same_execution_margin64': margin64,
                   'frozen_winner': ref['winner'], 'frozen_score': ref['score'],
                   'frozen_margin32': ref['margin32'], 'frozen_margin64': ref['margin64'],
                   'winner_matches_frozen': bool(answer[0] == ref['winner']),
                   'winner_score_difference': abs(float(answer[1])-ref['score']),
                   'all_candidate_max_score_difference32': float(np.max(np.abs(live32-ref['scores32']))),
                   'all_candidate_max_score_difference64': float(np.max(score_drift64)),
                   'max_score_change_bound64': float(np.max(error_bound)),
                   'max_bound_violation64': float(np.max(score_drift64-error_bound)),
                   'frozen_margin_exceeds_twice_bound': bool(ref['margin64'] > 2*np.max(error_bound)),
                   'query_feature_error_l2': float(errors[q]),
                   'max_candidate_feature_error_l2': float(np.max(errors[:n]))}
            query_rows.append(row)
        common.write(out / f'replay_{repeat}.json', {'inputs': input_checks[-common.SAMPLES:],
                                                    'queries': query_rows[-common.SAMPLES:]})
        print('HELIOS_DIAGNOSTIC_REPLAY', repeat, round(time.time()-tick, 1), flush=True)
    handle.remove()
    np.save(out / 'raw_replay_features.npy', features)
    # compute_embedding의 batch를 그대로 한 번 GPU에 올려 재사용한다.
    # 전처리/양자화/새 batch 생성과 독립된 모델 자체의 변동만 관찰한다.
    fixed_features = np.empty_like(features)
    fixed_rows = []
    with torch.no_grad():
        for i, cpu_batch in enumerate(batches):
            batch = {k: torch.from_numpy(v).cuda() for k, v in cpu_batch.items()}
            pointers = {k: v.data_ptr() for k, v in batch.items()}
            for repeat in range(common.REPEATS):
                output = model(batch)['global'][0].detach().cpu().numpy().astype(np.float32)
                fixed_features[repeat, i] = output
                assert all(batch[k].data_ptr() == p for k, p in pointers.items())
                assert all(np.array_equal(v.detach().cpu().numpy(), cpu_batch[k]) for k, v in batch.items())
                fixed_rows.append({'arrival': work.initial+i, 'repeat': repeat,
                                   'same_gpu_batch_pointer': True, 'batch_content_unchanged': True,
                                   'feature_vs_fixed_first': distance(output, fixed_features[0, i]),
                                   'feature_vs_raw_replay': distance(output, features[repeat, i]),
                                   'feature_vs_cache': distance(output, source[work.initial+i])})
            if i % 24 == 0:
                print('HELIOS_DIAGNOSTIC_FIXED_BATCH', i, round(time.time()-tick, 1), flush=True)
    np.save(out / 'fixed_batch_features.npy', fixed_features)
    common.write(out / 'fixed_batch.json', {'rows': fixed_rows})
    after_state = state_hash(model)
    assert before_state == after_state, '공식 모델의 등록 상태가 변했다'
    checks = {
        'all_input_equal': all(r['input_equal_first'] for r in input_checks),
        'all_coords_equal': all(r['coords_equal_first'] and r['coords_equal_independent_quantizer'] for r in input_checks),
        'all_batch_features_equal': all(r['features_equal_first'] for r in input_checks),
        'all_same_execution_winner32_match': all(r['winner'] == r['same_execution_winner32'] for r in query_rows),
        'all_same_execution_winner64_match': all(r['winner'] == r['same_execution_winner64'] for r in query_rows),
        'all_same_execution_kdtree_match': all(r['winner'] == r['same_execution_kdtree_winner'] for r in query_rows),
        'all_same_execution_dot_exact': all(r['same_execution_exact_dot_error'] == 0 for r in query_rows),
        'all_winners_match_frozen': all(r['winner_matches_frozen'] for r in query_rows),
        'all_frozen_margins_survive_bound': all(r['frozen_margin_exceeds_twice_bound'] for r in query_rows),
        'model_state_unchanged': before_state == after_state}
    summary = {
        'schema': 'pdv2-helios-inference-diagnostic-v1', 'status': 'complete',
        'n_map': n_map, 'arrivals': common.SAMPLES, 'repeats': common.REPEATS,
        'raw_input_sha256': work.input_hash, 'checks': checks,
        'raw_feature_vs_cache_max_abs': max(r['feature_vs_cache']['max_abs'] for r in input_checks),
        'raw_feature_vs_cache_max_l2': max(r['feature_vs_cache']['l2'] for r in input_checks),
        'raw_replay_pair_max_abs': float(np.max(np.ptp(features, axis=0))),
        'fixed_same_batch_pair_max_abs': float(np.max(np.ptp(fixed_features, axis=0))),
        'fixed_same_batch_nonidentical_frames': int(np.any(np.ptp(fixed_features, axis=0) != 0, axis=1).sum()),
        'frozen_score_max_difference': max(r['winner_score_difference'] for r in query_rows),
        'frozen_score_assert_failures_1e7': sum(r['winner_score_difference'] >= 1e-7 for r in query_rows),
        'frozen_winner_changes': sum(not r['winner_matches_frozen'] for r in query_rows),
        'min_frozen_margin64': min(r['frozen_margin64'] for r in query_rows),
        'max_all_candidate_score_difference64': max(r['all_candidate_max_score_difference64'] for r in query_rows),
        'max_score_change_bound64': max(r['max_score_change_bound64'] for r in query_rows),
        'max_bound_violation64': max(r['max_bound_violation64'] for r in query_rows),
        'first_failed_row_1e7': next((r for r in query_rows if r['winner_score_difference'] >= 1e-7 or not r['winner_matches_frozen']), None),
        'model_state_sha256': before_state, 'elapsed_seconds': time.time()-tick,
        'diagnostic_source_sha256': common.sha(Path(__file__)),
        'frozen_adapter_sha256': common.sha(Path(frozen.__file__)),
        'frozen_protocol_sha256': common.sha(common.OUT/'helios_protocol_v2.json'),
        'frozen_feature_cache_sha256': common.sha(frozen.target('helios', n_map).with_suffix('.npy')),
        'scope': 'No timings, model/parameter/precision changes, tolerance relaxation, or source replacement. Same-execution retrieval is separated from independent cached inference. All96 raw arrivals x3, plus all96 identical GPU batches x3.'}
    common.write(out / 'complete.json', summary)
    print('HELIOS_DIAGNOSTIC_COMPLETE', summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-map', type=int, default=1000)
    args = parser.parse_args()
    diagnose(args.n_map)
