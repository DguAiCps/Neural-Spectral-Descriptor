#!/usr/bin/env python3
"""개별 계측에서 요약·위키·표·그림·Dropbox까지 순서대로 반영한다."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def file_sha(path: Path) -> str:
    """큰 캐시도 전체를 메모리에 올리지 않고 확인한다."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check_cache_provenance() -> None:
    """초기 캐시 재사용 근거를 완료 실측과 연결하고 원본을 보존한다."""
    out = ROOT / 'results/pdv2_table3_20261029'
    records, sources = [], {}
    for folder in sorted(out.glob('*_batch1_v2_n*_preparation')):
        proof_path = folder/'provenance.json'
        if not proof_path.exists():
            continue
        proof = json.loads(proof_path.read_text())
        row_path = out/'methods'/f"{proof['method']}_n{proof['n_map']}.json"
        if not row_path.exists():
            continue
        assert proof['status'] == 'passed'
        preparer = ROOT/'scripts/_pdv2_table3_prefix_cache.py'
        assert file_sha(preparer) == proof['preparer_source_sha256']
        checked = {preparer: proof['preparer_source_sha256'], proof_path: file_sha(proof_path)}
        for name in ('before', 'after'):
            path = folder/(name+'.json')
            assert file_sha(path) == proof[name+'_manifest_sha256']
            checked[path] = file_sha(path)
        cache = out/f"{proof['method']}_batch1_v2_n{proof['n_map']}"
        meta_path = cache.with_suffix('.json')
        meta = json.loads(meta_path.read_text())
        expected = proof.get('cache_feature_sha256', proof.get('feature_sha256'))
        assert file_sha(cache.with_suffix('.npy')) == meta['feature_sha256'] == expected
        if proof['n_map'] == 5000:
            assert file_sha(meta_path) == proof['cache_metadata_sha256']
        else:
            assert proof['reused_rows'] == 5140 and proof['computed_rows'] == 5000
            assert proof['prefix_copy_exact'] and proof['all_rows_written_and_finite']
            assert meta['cache_preparation']['provenance_sha256'] == file_sha(proof_path)
            assert file_sha(ROOT/proof['source_provenance_path']) == proof['source_provenance_sha256']
        checked[meta_path] = file_sha(meta_path)
        sources.update({str(p.relative_to(ROOT)): digest for p, digest in checked.items()})
        records.append({'method': proof['method'], 'n_map': proof['n_map'], 'mode': proof['mode'],
                        'measurement_path': str(row_path.relative_to(ROOT)), 'measurement_sha256': file_sha(row_path),
                        'cache_sha256': expected, 'provenance_path': str(proof_path.relative_to(ROOT)),
                        'provenance_sha256': file_sha(proof_path)})
    (out/'cache_provenance_manifest.json').write_text(json.dumps({
        'status': 'passed', 'records': records, 'sources': sources,
        'scope': 'Only initial cached-map preparation reused verified prefixes; all measured arrivals and searches execute unchanged.'}, indent=2)+'\n')


def check_gpu_capacity() -> None:
    """물리 용량을 넘는 예약량은 원인을 감사하기 전 자동 게시하지 않는다."""
    out = ROOT / 'results/pdv2_table3_20261029'
    environment = json.loads((out/'environment.json').read_text())
    match = re.search(r'(\d+) MiB', environment['gpu'])
    assert match, environment['gpu']
    physical = int(match.group(1)) * 1024**2
    issues = []
    for path in sorted((out/'methods').glob('*.json')):
        value = json.loads(path.read_text())
        reserved = max(m.get('cuda_peak_reserved', 0) for m in value['memory_repeats'])
        if reserved > physical:
            issues.append({'method': value['method'], 'n_map': value['n_map'],
                           'peak_reserved': reserved, 'physical_bytes': physical})
    (out/'publication_capacity_check.json').write_text(json.dumps({
        'status': 'requires_workspace_audit' if issues else 'passed',
        'physical_gpu_bytes': physical, 'issues': issues,
        'scope': 'A capacity check guards against unexplained device-memory pressure; it is not a full device peak measurement.'}, indent=2)+'\n')
    assert not issues, ('GPU 예약량이 물리 용량을 넘는다. 원인 감사 후 게시한다.', issues)


def main() -> None:
    """GPU 정확 계측이 없는 구간에서만 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reason', required=True)
    args = parser.parse_args()
    check_gpu_capacity()
    check_cache_provenance()
    commands = [
        ['docker', 'exec', 'ring_baseline', 'chown', '-R', '1000:1000', '/workspace/Neural-Spectral-Codec/results/pdv2_table3_20261029'],
        ['python3', 'scripts/_pdv2_table3_apply_memory_audit.py'],
        ['docker', 'exec', '-w', '/workspace/Neural-Spectral-Codec', 'ring_baseline',
         'python3', 'scripts/_pdv2_table3_benchmark.py', 'summarize'],
        ['python3', 'scripts/_pdv2_table3_report.py'],
        ['docker', 'exec', '-w', '/workspace/Neural-Spectral-Codec', 'ring_baseline', 'python3', 'scripts/_pdv2_table3_archive.py'],
        ['python3', 'scripts/_paper_publish_v22.py', '--reason', args.reason,
         '--refresh-figure', '--figure-python', '/tmp/nsd_figure4_env/bin/python'],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
