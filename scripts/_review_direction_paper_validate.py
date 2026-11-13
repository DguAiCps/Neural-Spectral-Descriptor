#!/usr/bin/env python3
"""원고 삽입안의 출처·기존 연구 파일 보존·별도 빌드를 CPU에서 검산한다."""
from pathlib import Path
import re
import subprocess
import time
import _review_direction_metrics as base

OUT = base.OUT / 'expanded46/paper_proposal'


def main() -> None:
    """원고 원본에 쓰지 않고 제안 PDF와 위키를 검사한다."""
    manifest = base.read(OUT/'source_manifest.json')
    for path, digest in manifest['sources'].items():
        assert base.digest(base.ROOT/path) == digest, path
    for name, digest in manifest['proposal_sha256'].items():
        assert base.digest(OUT/name) == digest, name
    for path in manifest['protected_research_files']:
        assert base.digest(base.ROOT/path) == manifest['initial_manuscript_files'][path], path
    existing = '\n'.join(p.read_text() for p in (base.ROOT/'docs/paper/body_v2.3').rglob('*.tex'))
    proposal = '\n'.join(p.read_text() for p in OUT.glob('*.tex'))
    pattern = r'\\label\{([^}]+)\}'
    old_labels = set(re.findall(pattern, existing))
    new_labels = re.findall(pattern, proposal)
    assert len(set(new_labels)) == len(new_labels)
    assert not old_labels.intersection(new_labels)
    assert not set(re.findall(r'\\ref\{([^}]+)\}', proposal)).difference(new_labels)
    row_counts = {}
    for name, expected in [('direction', 6), ('yaw', 7), ('causes', 6), ('sensitivity', 9)]:
        text = (OUT/f'table_{name}.tex').read_text()
        body = text.split(r'\midrule', 1)[1].split(r'\bottomrule', 1)[0]
        count = sum(line.rstrip().endswith(r'\\') for line in body.splitlines())
        assert count == expected
        row_counts[name] = count
    for pass_number in (1, 2):
        run = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'preview.tex'],
                             cwd=OUT, capture_output=True, text=True)
        (OUT/f'build_pass{pass_number}.out').write_text(run.stdout+run.stderr)
        assert run.returncode == 0
    log = (OUT/'preview.log').read_text()
    assert 'Overfull' not in log and 'undefined' not in log.lower()
    assert 'multiply defined' not in log.lower()
    subprocess.run(['pdftotext', '-layout', 'preview.pdf', 'preview.txt'], cwd=OUT, check=True)
    subprocess.run(['pdftoppm', '-f', '3', '-l', '3', '-r', '110', '-png', '-singlefile',
                    'preview.pdf', 'preview_page3'], cwd=OUT, check=True)
    wiki = subprocess.run(['python3', 'docs/도구/check_wiki.py', 'docs'],
                          cwd=base.ROOT, capture_output=True, text=True)
    assert wiki.returncode == 0
    assert 'errors: 0, warnings: 0' in wiki.stdout
    info = subprocess.run(['pdfinfo', 'preview.pdf'], cwd=OUT, check=True, capture_output=True, text=True)
    pages = int(re.search(r'^Pages:\s+(\d+)', info.stdout, re.MULTILINE).group(1))
    base.write(OUT/'validation.json', dict(status='passed', checked_unix=time.time(),
        source_hashes_checked=len(manifest['sources']), proposal_source_hashes_checked=len(manifest['proposal_sha256']),
        preserved_research_files=len(manifest['protected_research_files']), protected_research_files_unchanged=True,
        parent_confirmed_format_and_disclosure_changes=manifest['concurrent_format_and_disclosure_changes'],
        unique_noncolliding_labels=new_labels, table_rows=row_counts, pdflatex_passes=2, preview_pages=pages,
        overfull_boxes=0, undefined_references=0, underfull_boxes=log.count('Underfull'),
        wiki_check=wiki.stdout.strip(), cpu_only=True,
        artifacts_sha256={p.name:base.digest(p) for p in [OUT/'preview.pdf', OUT/'preview.txt', OUT/'source_manifest.json', Path(__file__)]},
        wiki_sha256={p:base.digest(base.ROOT/p) for p in ['docs/평가/인과 평가.md','docs/리뷰.md','docs/할 일.md']}))
    print('PROPOSAL VALIDATION PASSED', pages, 'pages;', wiki.stdout.strip())


if __name__ == '__main__':
    main()
