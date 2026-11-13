#!/usr/bin/env python3
"""최종 문서 표·결과 해시·위키 문법을 검산해 종료 기록을 남긴다."""
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results/pdv2_fusion_width_20261024'


def digest(path: Path) -> str:
    """파일을 블록 단위로 읽어 SHA256을 구한다."""
    import hashlib
    value = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    """문서 표55셀을 원천 집계와 대조하고 기존 실행 결과를 보존한다."""
    summary = json.loads((OUT / 'summary.json').read_text())
    integrated = json.loads((OUT / 'integrated_report.json').read_text())
    audit = json.loads((OUT / 'integrated_validation.json').read_text())
    sources = dict(audit['sources'])
    for path, value in sources.items():
        assert digest(REPO / path) == value, path
    page = (REPO / 'docs/학습과 선택.md').read_text().split('## 문맥 전이의 방향 구성·')[1]
    metrics = ['AP', 'max_F1', 'R@100P', 'R@1']
    labels = {'시작 모델': 'start', '일반 순위·정확 미분': 'rank_all',
        '일반 순위·폭 미분 차단': 'rank_all_stop_width', '제외+보호·정확 미분': 'rank_filtered_guarded',
        '제외+보호·폭 미분 차단': 'rank_filtered_guarded_stop_width'}
    cells = 0
    for label, arm in labels.items():
        match = re.search(r'^\| ' + re.escape(label) + r' \| (.+)$', page, re.M)
        assert match, label
        values = [float(x.strip()) for x in match.group(1).strip('|').split('|')]
        assert values[0] == integrated['selected_epoch'][arm]
        for actual, name in zip(values[1:], metrics):
            assert abs(actual - 100*summary['results']['val'][arm]['macro'][name]) < 5.01e-7, (arm, name)
        cells += len(values)
    for role, prefix in (('train', '학습'), ('val', '검증')):
        for g, label in enumerate(('같은 방향 정답 있음', '반대 방향 정답만 있음', '기타')):
            match = re.search(r'^\| ' + prefix + '·' + label + r' \| (.+)$', page, re.M)
            assert match
            values = [float(x.strip().replace(',', '')) for x in match.group(1).strip('|').split('|')]
            row = integrated['direction'][role][str(g)]
            expected = [row[k] for k in ('positive', 'old_correct', 'gained', 'lost')]
            expected.append(100*row['loss_rate_given_correct'])
            assert all(abs(a-b) < 5.01e-5 for a, b in zip(values, expected))
            cells += len(values)
    assert cells == 55
    wiki = subprocess.run(['python3', 'docs/도구/check_wiki.py', 'docs'], cwd=REPO,
                          text=True, capture_output=True, check=True)
    # Mermaid 의존성은 호스트에 설치되어 있어 호스트 실행을 원문 그대로 보존한다.
    paths = sorted((REPO / 'docs').glob('*.md')) + sorted((REPO / 'docs/방법').glob('*.md')) + sorted((REPO / 'docs/평가').glob('*.md'))
    mermaid = subprocess.run(['node', '/tmp/nsd_wiki_mmd/check_mermaid.mjs', *map(str, paths)],
                             cwd=REPO, text=True, capture_output=True, check=True)
    documents = ['학습과 선택.md', '방법/궤적 문맥.md', '개요.md', '결정과 근거.md', '할 일.md',
                 'RAW/실험일지/2026-10-24_문맥전이-방향-역재생-분모기울기.md']
    for name in documents:
        p = REPO / 'docs' / name
        sources[str(p.relative_to(REPO))] = digest(p)
    sources[str(Path(__file__).relative_to(REPO))] = digest(Path(__file__))
    result = {'passed': True, 'document_table_cells': cells, 'verified_result_source_hashes': len(audit['sources']),
        'wiki': wiki.stdout.strip(), 'mermaid': mermaid.stdout.strip(), 'sources': sources,
        'test_used': False, 'eligible_candidates': summary['eligible_candidates']}
    # 결과 디렉터리는 Docker 소유이므로 stdout을 컨테이너에서 파일로 저장한다.
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
