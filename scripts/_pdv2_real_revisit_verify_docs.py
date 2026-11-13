#!/usr/bin/env python3
"""실제 재방문 학습 실험의 문서 표와 원천 결과를 독립 대조한다."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results/pdv2_real_revisit_20261025'


def digest(path: Path) -> str:
    """블록 단위로 파일 해시를 계산한다."""
    value = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    """위키의 지표·에포크·흔들림 표28셀과 결과 해시를 검산한다."""
    audit = json.loads((OUT / 'integrated_validation.json').read_text())
    report = json.loads((OUT / 'integrated_report.json').read_text())
    summary = json.loads((OUT / 'summary.json').read_text())
    sources = dict(audit['sources'])
    for path, expected in sources.items():
        assert digest(REPO / path) == expected, path
    verified_hashes = len(sources)
    text = (REPO / 'docs/학습과 선택.md').read_text().split('## 실제 반대 방향 재방문과 최종 순위 학습의 결합')[1]
    cells = 0

    def compare(label: str, expected: list, tolerance: float = 5.01e-7) -> int:
        match = re.search(r'^\| ' + re.escape(label) + r' \| (.+)$', text, re.M)
        assert match, label
        values = [float(v.strip().replace(',', '')) for v in match.group(1).strip('|').split('|')]
        assert len(values) == len(expected)
        assert all(abs(a-b) < tolerance for a, b in zip(values, expected)), (label, values, expected)
        return len(values)

    metrics = ['AP', 'max_F1', 'R@100P', 'R@1']
    labels = {'시작 모델': 'start', '정상 입력 학습': 'clean_rank',
        '전체 키 재방문 제약': 'pair_key', '문맥 재방문 제약': 'pair_context'}
    for label, arm in labels.items():
        epoch = 0 if arm == 'start' else summary['selected_epoch'][arm]
        cells += compare(label, [epoch]+[100*summary['results']['val'][arm]['macro'][m] for m in metrics])
    for label, arm in labels.items():
        cells += compare('학습 쌍·'+label, [report['auxiliary_pair_metrics'][arm][mode]['margin'] for mode in ('key', 'context')])
    assert cells == 28
    wiki = subprocess.run(['python3', 'docs/도구/check_wiki.py', 'docs'], cwd=REPO,
        text=True, capture_output=True, check=True)
    paths = sorted((REPO/'docs').glob('*.md')) + sorted((REPO/'docs/방법').glob('*.md')) + sorted((REPO/'docs/평가').glob('*.md'))
    mermaid = subprocess.run(['node', '/tmp/nsd_wiki_mmd/check_mermaid.mjs', *map(str, paths)],
        cwd=REPO, text=True, capture_output=True, check=True)
    for name in ('학습과 선택.md', '방법/궤적 문맥.md', '개요.md', '결정과 근거.md', '할 일.md', '용어.md',
                 'RAW/실험일지/2026-10-25_실제-반대방향-재방문-학습.md'):
        path = REPO / 'docs' / name
        sources[str(path.relative_to(REPO))] = digest(path)
    sources[str(Path(__file__).relative_to(REPO))] = digest(Path(__file__))
    print(json.dumps({'passed': True, 'document_table_cells': cells, 'verified_result_hashes': verified_hashes,
        'wiki': wiki.stdout.strip(), 'mermaid': mermaid.stdout.strip(), 'sources': sources,
        'eligible_candidates': summary['eligible_candidates'], 'test_used': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
