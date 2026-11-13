#!/usr/bin/env python3
"""DiSCO·HeLiOS·ImLPR의 고정 공식 추론과 실제 증분 검색 비용을 잰다."""
from __future__ import annotations
import argparse
import gc
import os
from pathlib import Path
import resource
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['SEED'] = '1'
import numpy as np
import torch
import _pdv2_table3_benchmark as common


def configure() -> None:
    """공통 배치 1·단일 CPU 스레드·FP32 조건이다."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    import cv2
    cv2.setNumThreads(1)


def settings(method: str) -> dict:
    """선택 기록과 어댑터를 방법별로 동결한다."""
    selected_path = common.REPO / f'results/pdv2_{method}_official/train_selection.json'
    selection = common.read(selected_path)
    chosen = selection['best' if method == 'imlpr' else 'val_best']
    weight = common.REPO / chosen.get('checkpoint', chosen.get('path')).split('Neural-Spectral-Codec/')[-1]
    assert common.sha(weight) == chosen['sha256']
    source_names = [f'_pdv2_table3_global.py', '_pdv2_table3_benchmark.py',
                    f'_{method}_official_dump.py', f'_{method}_official_train.py',
                    '_imlpr_official_riv.py' if method == 'imlpr' else f'_{method}_official_prep.py',
                    '_causal_pr_dump.py']
    config = {'schema': 'pdv2-table3-global-protocol-v1', 'method': method,
              'common_protocol_sha256': common.sha(common.OUT / 'protocol.json'),
              'weight': str(weight), 'weight_sha256': chosen['sha256'],
              'selection_sha256': common.sha(selected_path), 'seed': 1,
              'sources': {f'scripts/{p}': common.sha(common.REPO/'scripts'/p) for p in source_names},
              'retrieval': ('frozen table2 float64 exact prefix Euclidean shortlist8 and exact rescoring; official KDTree winner check'
                            if method == 'disco' else 'frozen table2 float32 exhaustive prefix cosine and final winner dot; official KDTree top1 check'),
              'input': 'same Sejong01 KISS keyframe raw points; frozen official sensor preprocessing; no labels; batch1',
              'memory': 'full preallocated original descriptor bank incl45support; DiSCO float64 bank plus squared norms; unique CPU arrays plus actual torch CUDA allocations, model not added twice'}
    target = common.OUT/f'{method}_protocol.json'
    if target.exists():
        assert common.read(target) == config, f'{method}: source changed; preserve a new revision'
    else:
        common.write(target, config)
    return config


class Encoder:
    """현재 표 2와 같은 모델 및 원시 전처리다."""
    def __init__(self, method: str, config: dict) -> None:
        self.method = method
        weight = Path(config['weight'])
        if method == 'disco':
            import _disco_official_dump as dump
            import _disco_official_prep as prep
            self.cfg, _, self.evaluation, self.model = dump.official_model(weight)
            import gputransform
            self.prep, self.transform = prep, gputransform
            self.model.eval()
        elif method == 'imlpr':
            import _imlpr_official_dump as dump
            import _imlpr_official_riv as prep
            import _imlpr_official_train as train
            self.model, self.params = dump._official_model(weight, 'cuda')
            self.prep, self.train = prep, train
        else:
            import _helios_official_dump as dump
            self.evaluation, self.params = dump._official()
            self.model = dump.official_model(self.evaluation, self.params, weight, 'cuda').eval()
            self.prep, self.toolbox = dump.P, dump.P.Toolbox()
        configure()

    def input(self, points: np.ndarray, sid: int) -> np.ndarray:
        """원시 점군부터 신경망 직전 입력까지 실제 전처리한다."""
        if self.method == 'disco':
            hits = self.prep.official_hits('mulran', points)
            return self.prep.official_polar(hits, self.transform, self.cfg)
        if self.method == 'imlpr':
            xyz, refl, ring, _ = self.prep.frame_points('mulran', 'val', points, None, sid)
            return self.train.eval_input(self.prep.fast_frame(xyz, refl, ring, 'mulran'))
        value, _ = self.prep.process_scan(self.toolbox, points,
                                         self.prep.fill_rng('mulran', 'Sejong01', sid))
        return np.ascontiguousarray(value[:, :3])

    def forward(self, value: np.ndarray) -> np.ndarray:
        """공식 배치 1 출력만 반환한다."""
        with torch.no_grad():
            if self.method == 'disco':
                result = self.model(torch.from_numpy(value[None]).cuda())[0]
            elif self.method == 'imlpr':
                result = self.model(torch.from_numpy(value[None]).cuda())[1]
            else:
                return self.evaluation.compute_embedding(self.model, torch.tensor(value),
                                                          'cuda', self.params)[0].astype(np.float32)
        return result[0].cpu().numpy().astype(np.float32)

    def encode(self, points: np.ndarray, sid: int) -> np.ndarray:
        """실제 신규 스캔 전처리와 모델 추론이다."""
        return self.forward(self.input(points, sid))

    def verify_preprocessing(self, points: np.ndarray, sid: int) -> dict:
        """공식 전처리의 독립 경로와 수치가 같은지 검사한다."""
        if self.method == 'disco':
            hits = self.prep.official_hits('mulran', points)
            actual = self.input(points, sid)
            ref = self.prep.official_polar(hits, self.prep.fake_gputransform_module(), self.cfg)
            ok, mismatch = self.prep.polar_agrees(actual, ref)
            assert ok, mismatch
            return {'kind': 'official CUDA polar vs CPU transcription', 'mismatch': int(mismatch)}
        if self.method == 'imlpr':
            xyz, refl, ring, _ = self.prep.frame_points('mulran', 'val', points, None, sid)
            fast = self.prep.fast_frame(xyz, refl, ring, 'mulran')
            ref = self.prep.official_frame(xyz, refl, ring, 'mulran')
            np.testing.assert_allclose(fast, ref, atol=2e-4, rtol=1e-5)
            return {'kind': 'full official normal loop and image functions',
                    'max_error': float(abs(fast-ref).max())}
        # The deterministic random fill is part of the frozen evaluation recipe.
        actual = self.input(points, sid)
        cached = self.prep.seq_dir('val', 'mulran', 'Sejong01')/'LiDAR'/f'{sid:06d}.bin'
        if cached.exists():
            ref = np.fromfile(cached, np.float32).reshape(-1, 4)[:, :3]
            np.testing.assert_array_equal(actual, ref)
            return {'kind': 'official prepared input file', 'max_error': 0.0}
        return {'kind': 'frozen official toolbox wrapper; deterministic raw replay',
                'note': 'no local precomputed val scan; wrapper source hashed; descriptor replay checked'}


def target(method: str, n_map: int) -> Path:
    """별도 초기 지도 캐시의 기본 이름이다."""
    return common.OUT/f'{method}_batch1_n{n_map}'


def cache(method: str, n_map: int) -> None:
    """공식 원시 전처리부터 배치 1 참조 캐시를 만든다."""
    from _pd_build_test_caches import _make_loader
    configure()
    config = settings(method)
    path = target(method, n_map)
    if path.with_suffix('.json').exists():
        assert common.read(path.with_suffix('.json'))['config'] == config
        return
    work = common.Workload(common.read(common.OUT/'protocol.json'), n_map)
    enc = Encoder(method, config)
    loader = _make_loader('mulran', 'Sejong01')
    checked = enc.verify_preprocessing(work.points[0], int(work.ids[work.initial]))
    first = enc.encode(loader[int(work.ids[0])]['points'], int(work.ids[0]))
    bank = np.lib.format.open_memmap(path.with_suffix('.npy'), mode='w+', dtype=np.float32,
                                    shape=(work.end, len(first)))
    tick = time.perf_counter()
    for k, sid in enumerate(work.ids):
        bank[k] = first if k == 0 else enc.encode(loader[int(sid)]['points'], int(sid))
        assert np.isfinite(bank[k]).all()
        if k % 100 == 0:
            print('CACHE', method, n_map, k, work.end, round(time.perf_counter()-tick, 1), flush=True)
    bank.flush()
    common.write(path.with_suffix('.json'), {'config': config, 'frames': work.end,
                 'feature_sha256': common.sha(path.with_suffix('.npy')),
                 'seconds': time.perf_counter()-tick, 'preprocessing_check': checked,
                 'dim': len(first), 'batch': 1})


class Runtime:
    """원본 검색 정밀도를 보존한 증분 특징 뱅크다."""
    def __init__(self, method: str, work) -> None:
        self.method, self.work = method, work
        self.config = settings(method)
        path = target(method, work.n_map)
        self.meta = common.read(path.with_suffix('.json'))
        assert self.meta['config'] == self.config
        assert common.sha(path.with_suffix('.npy')) == self.meta['feature_sha256']
        self.source = np.load(path.with_suffix('.npy'), mmap_mode='r')
        self.encoder = Encoder(method, self.config)
        self.bank = np.empty(self.source.shape, np.float64 if method == 'disco' else np.float32)
        self.sq = np.empty(len(self.bank), np.float64) if method == 'disco' else None
        self.reset()

    def reset(self) -> None:
        self.n = self.work.initial
        self.bank[:self.n] = self.source[:self.n]
        if self.sq is not None:
            self.sq[:self.n] = np.einsum('ij,ij->i', self.bank[:self.n], self.bank[:self.n])

    def encode(self, points: np.ndarray) -> np.ndarray:
        return self.encoder.encode(points, int(self.work.ids[self.n]))

    def append(self, desc: np.ndarray) -> None:
        self.bank[self.n] = desc
        if self.sq is not None:
            self.sq[self.n] = np.einsum('i,i->', self.bank[self.n], self.bank[self.n])
        self.n += 1

    def query(self, q: int) -> tuple:
        n = q-29
        if self.method == 'disco':
            from _disco_official_dump import l2_top1_prefix
            j, distance = l2_top1_prefix(self.bank, self.sq, q, n)
            return j, -distance, n
        from _causal_pr_dump import prefix_topk
        j = int(prefix_topk(self.bank, q, n, 1)[0])
        return j, float(self.bank[j] @ self.bank[q]), n

    def memory(self) -> dict:
        model = common.tensor_bytes(self.encoder.model)
        cpu = self.bank.nbytes + (self.sq.nbytes if self.sq is not None else 0) + model['cpu']
        gpu = torch.cuda.memory_allocated()
        return {'cpu': cpu, 'gpu': gpu, 'total': cpu+gpu, 'model_gpu': model['gpu'],
                'breakdown': {'descriptor_bank_capacity': self.bank.nbytes,
                              'squared_norm_capacity': self.sq.nbytes if self.sq is not None else 0,
                              'model_cpu': model['cpu']},
                'logical_feature_payload_bytes_per_frame': self.source.shape[1]*4,
                'timepoint': 'after final arrival; full capacity includes45support; GPU actual allocator bytes'}


def measure(method: str, n_map: int) -> None:
    """같은 96개 도착을 3번 측정해 검증한 결과를 저장한다."""
    configure()
    work = common.Workload(common.read(common.OUT/'protocol.json'), n_map)
    tick = time.perf_counter()
    runtime = Runtime(method, work)
    common.sync()
    initial_s = time.perf_counter()-tick
    checks = []
    from sklearn.neighbors import KDTree
    for i, points in enumerate(work.points[:common.WARMUP]):
        t = work.initial+i
        value = runtime.encode(points)
        np.testing.assert_allclose(value, runtime.source[t], atol=1e-6, rtol=1e-6)
        runtime.append(value)
        j, score, n = runtime.query(t-15)
        distances, indices = KDTree(runtime.bank[:n]).query(runtime.bank[t-15:t-14], k=1)
        assert j == int(indices[0, 0]), (method, t, j, indices)
        reference = -float(distances[0, 0]) if method == 'disco' else float(runtime.bank[j] @ runtime.bank[t-15])
        assert abs(score-reference) < 1e-7
        checks.append({'query': t-15, 'winner': j, 'score_error': abs(score-reference),
                       'feature_max_error': float(abs(value-runtime.source[t]).max())})
    verification = {'status': 'passed', 'raw_batch1_feature_checks': common.WARMUP,
                    'official_KDTree_checks': checks, 'preprocessing': runtime.meta['preprocessing_check']}
    common.write(common.OUT/'verification'/f'{method}_n{n_map}.json', verification)
    rows, memories = [], []
    for repeat in range(common.REPEATS):
        runtime.reset(); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        before = common.rss()
        for i, points in enumerate(work.points):
            t = work.initial+i
            encoded, enc = common.timed(lambda: runtime.encode(points))
            _, state = common.timed(lambda: runtime.append(encoded))
            answer, query = common.timed(lambda: runtime.query(t-15))
            rows.append({'repeat': repeat, 'arrival': t, 'query': t-15, 'n_candidates': t-44,
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
        print('MEASURE', method, n_map, repeat, flush=True)
    for i in range(common.SAMPLES):
        values = [rows[k*common.SAMPLES+i] for k in range(common.REPEATS)]
        assert len({v['winner'] for v in values}) == 1
        assert np.ptp([v['confidence'] for v in values]) < 1e-7
    result = {'schema': 'pdv2-table3-measurement-v1', 'method': method, 'n_map': n_map,
              'status': 'passed', 'publish_ready': True, 'verification': verification,
              'protocol_sha256': common.sha(common.OUT/'protocol.json'),
              'source_sha256': common.sha(Path(__file__)), 'method_config': runtime.config,
              'sequence': common.SEQUENCE, 'seed': 1, 'candidate_range': [n_map, n_map+common.SAMPLES-1],
              'mean_candidates': n_map+(common.SAMPLES-1)/2, 'samples': common.SAMPLES,
              'repeats': common.REPEATS, 'warmup': common.WARMUP,
              'timings_ms': {k: common.stats([r[f] for r in rows]) for k, f in
                             (('encode','encode'),('state_update','state_update'),('update','update'),
                              ('query','query_ms'),('processing','processing'))},
              'resident_bytes': memories[-1], 'memory_repeats': memories, 'raw_samples': rows,
              'raw_input_fixture_bytes': work.input_bytes, 'raw_input_sha256': work.input_hash,
              'initial_cached_state_build_s': initial_s, 'completed_unix': time.time(),
              'environment': {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                              'threads': 1, 'tf32': False},
              'scope': 'frozen official model and preprocessing; MulRan single raw scan; exact prefix search and final confidence; no labels or tuning'}
    common.write(common.OUT/'methods'/f'{method}_n{n_map}.json', result)
    common.summarize()
    print('COMPLETE', method, n_map, result['timings_ms'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('cache', 'measure'))
    parser.add_argument('--method', choices=('disco', 'imlpr', 'helios'), required=True)
    parser.add_argument('--n-map', type=int, default=1000)
    args = parser.parse_args()
    globals()[args.action](args.method, args.n_map)
