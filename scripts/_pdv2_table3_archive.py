#!/usr/bin/env python3
"""효율 원천에 기록된 작은 코드·설정 파일을 해시별로 보존한다."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/pdv2_table3_20261029'


def main() -> None:
    """원천 해시와 같은 파일만 복제하며 큰 데이터·가중치는 복제하지 않는다."""
    archive = OUT/'source_archive'
    archive.mkdir(exist_ok=True)
    found = {}
    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and len(value) == 64 and '/' in key:
                    path = Path(key)
                    path = path if path.is_absolute() else ROOT/path
                    if path.exists() and path.suffix in ('.py', '.m', '.c', '.cpp', '.h', '.hh', '.hpp', '.cu', '.cuh', '.pyx', '.pxd', '.sh', '.json', '.yaml', '.txt') and path.stat().st_size < 5_000_000:
                        digest = hashlib.sha256(path.read_bytes()).hexdigest()
                        assert digest == value, (str(path), digest, value)
                        copied = archive/(digest+path.suffix)
                        if not copied.exists():
                            copied.write_bytes(path.read_bytes())
                        source_name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
                        found[source_name] = {'sha256': digest, 'snapshot': str(copied.relative_to(ROOT))}
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)
    paths = [OUT/'protocol.json']+list((OUT/'methods').glob('*.json'))+list(OUT.glob('*protocol.json'))
    if (OUT/'cache_provenance_manifest.json').exists():
        paths.append(OUT/'cache_provenance_manifest.json')
    for path in paths:
        walk(json.loads(path.read_text()))
    (archive/'manifest.json').write_text(json.dumps({'sources': found, 'source_records': [str(p.relative_to(ROOT)) for p in paths]}, indent=2)+'\n')
    print('archived', len(found), 'sources')


if __name__ == '__main__':
    main()
