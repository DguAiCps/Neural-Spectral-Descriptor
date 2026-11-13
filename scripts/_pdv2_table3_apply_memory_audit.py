#!/usr/bin/env python3
"""기존 계측을 보존하고 모델의 누락된 CPU 수치 상태만 추가한다."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/pdv2_table3_20261029'


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """독립 주소 감사로 확인한 상수 모델 버퍼만 전 크기에 반영한다."""
    original = OUT/'before_model_numeric_audit'
    original.mkdir(exist_ok=True)
    for probe_path in sorted((OUT/'model_memory').glob('*.json')):
        probe = read(probe_path)
        if 'method' not in probe or 'audit' not in probe:
            continue
        assert probe['status'] == 'passed'
        audit = probe['audit']
        assert audit['status'] == 'passed' and not audit['unknown_capacity_paths']
        assert probe['cold_cpu_unique_bytes'] == probe['warm_cpu_unique_bytes'] == audit['cpu_unique_bytes'], 'Nonconstant model state requires per-size audit'
        delta = audit['extra_cpu_bytes']
        for path in sorted((OUT/'methods').glob(probe['method']+'_n*.json')):
            row = read(path)
            if 'model_numeric_memory_correction' in row:
                assert row['model_numeric_memory_correction']['audit_sha256'] == sha(probe_path)
                continue
            before = sha(path)
            backup = original/path.name
            if backup.exists() and sha(backup) != before:
                backup = original/(path.stem + '_' + before[:16] + path.suffix)
            if backup.exists():
                assert sha(backup) == before
            else:
                backup.write_bytes(path.read_bytes())
            record = {'audit_path': str(probe_path.relative_to(ROOT)), 'audit_sha256': sha(probe_path),
                      'original_measurement_sha256': before, 'original_measurement': str(backup.relative_to(ROOT)),
                      'cpu_delta_bytes': delta, 'registered_cpu_bytes': audit['registered_cpu_bytes'],
                      'full_cpu_model_bytes': audit['cpu_unique_bytes'],
                      'scope': 'Only previously unregistered CPU model numerical buffers added; timings and torch GPU allocations unchanged. Constant across cold/12/96 arrivals, fixed model/input dimensions; same model correction at each N.'}
            for memory in [row['resident_bytes']]+row['memory_repeats']:
                memory['cpu'] += delta
                memory['total'] += delta
                memory['breakdown']['model_extra_cpu_numeric'] = delta
                memory['model_numeric_audit'] = {'cpu_unique_bytes': audit['cpu_unique_bytes'],
                                                'registered_cpu_bytes': audit['registered_cpu_bytes'],
                                                'extra_cpu_bytes': delta,
                                                'source_path': str(probe_path.relative_to(ROOT))}
            row['model_numeric_memory_correction'] = record
            temporary = path.with_suffix('.tmp.json')
            temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
            temporary.replace(path)
            print('memory audited', path.name, '+', delta, 'B')


if __name__ == '__main__':
    main()
