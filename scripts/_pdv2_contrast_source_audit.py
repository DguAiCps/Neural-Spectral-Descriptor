#!/usr/bin/env python3
"""시험 동결 명세가 참조한 이전 실험 명세의 해시도 따라가 검사한다."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pdv2_diffattn_contrast_test_20261104"


def main() -> None:
    """sources 사전의 파일과 그 안의 sources 사전을 재귀 검산한다."""
    visited = set()
    checked = {}
    skipped = []
    pending = [OUT / "protocol.json"]
    while pending:
        document = pending.pop()
        if document in visited:
            continue
        visited.add(document)
        value = json.loads(document.read_text())
        if not isinstance(value, dict):
            continue
        for name, expected in value.get("sources", {}).items():
            if not isinstance(expected, str) or not re.fullmatch("[a-f0-9]{64}", expected):
                skipped.append(dict(document=str(document.relative_to(ROOT)), entry=name))
                continue
            path = Path(name)
            if not path.is_absolute():
                path = ROOT / path
            if path in checked:
                actual = checked[path]
            else:
                h = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(chunk)
                actual = h.hexdigest()
                checked[path] = actual
            assert actual == expected, (str(document), name, expected, actual)
            if path.suffix == ".json":
                pending.append(path)
    result = dict(status="passed", checked_files=len(checked), json_documents=len(visited),
                  skipped_entries=skipped,
                  scope="루트시험명세에서sources사전으로연결된파일의재귀해시검사. 별도데이터캐시출처는각시퀀스input_audit와reference_validation에서검사한다.")
    (OUT / "recursive_source_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    main()
