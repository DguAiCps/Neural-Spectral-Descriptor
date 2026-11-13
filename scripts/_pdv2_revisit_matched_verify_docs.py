#!/usr/bin/env python3
"""재방문 추가 진단의 위키 표와 원천 지표·입력 해시를 독립 검산한다."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    """파일의 SHA256을 계산한다."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(folder: str, name: str) -> dict:
    """결과 JSON을 읽는다."""
    return json.loads((REPO / "results" / folder / name).read_text())


def main() -> None:
    """표80셀·전체 근거 해시·위키·차트 문법을 확인한다."""
    sources = {}
    studies = {
        "pdv2_revisit_matched_20261025": "integrated_validation.json",
        "pdv2_revisit_observation_20261025": "completion_validation.json",
        "pdv2_revisit_fullbank_20261025": "completion_validation.json",
        "pdv2_revisit_support_20261025/completed": "completion_validation.json",
        "pdv2_temporal_mean_score_20261025": "completion_validation.json",
    }
    for folder, name in studies.items():
        audit = read(folder, name)
        assert audit["passed"] and not audit["test_used"]
        for path, value in audit["sources"].items():
            if path in sources:
                assert sources[path] == value, path
            sources[path] = value
        path = REPO / "results" / folder / name
        sources[str(path.relative_to(REPO))] = digest(path)
    for path, value in sources.items():
        assert digest(REPO / path) == value, path
    hashes = len(sources)
    text = (REPO / "docs/학습과 선택.md").read_text()
    text = text.split("## 실제 반대 방향 재방문과 최종 순위 학습의 결합")[1]

    def compare(label: str, expected: list) -> int:
        match = re.search(r"^\| " + re.escape(label) + r" \| (.+)$", text, re.M)
        assert match, label
        actual = [
            float(x.strip().replace(",", "")) for x in match[1].strip("|").split("|")
        ]
        assert len(actual) == len(expected), label
        assert all(abs(x - y) < 5.01e-7 for x, y in zip(actual, expected)), (
            label,
            actual,
            expected,
        )
        return len(expected)

    metrics = ["AP", "max_F1", "R@100P", "R@1"]
    parent = read("pdv2_real_revisit_20261025", "summary.json")
    report = read("pdv2_real_revisit_20261025", "integrated_report.json")
    matched = read("pdv2_revisit_matched_20261025", "summary.json")
    cells = 0
    for label, arm in {
        "시작 모델": "start",
        "정상 입력 학습": "clean_rank",
        "전체 키 재방문 제약": "pair_key",
        "문맥 재방문 제약": "pair_context",
    }.items():
        epoch = 0 if arm == "start" else parent["selected_epoch"][arm]
        cells += compare(
            label,
            [epoch]
            + [100 * parent["results"]["val"][arm]["macro"][m] for m in metrics],
        )
        cells += compare(
            "학습 쌍·" + label,
            [
                report["auxiliary_pair_metrics"][arm][mode]["margin"]
                for mode in ("key", "context")
            ],
        )
    for label, arm, source in [
        ("시작 모델", "start", parent),
        ("정상 학습", "clean_rank", parent),
        ("전체 키 제약", "pair_key", parent),
        ("문맥 계수0.01", "pair_context", parent),
        ("문맥 계수0.005132463", "pair_context", matched),
    ]:
        epoch = 0 if arm == "start" else source["selected_epoch"][arm]
        cells += compare(
            "강도 대조·" + label,
            [epoch]
            + [100 * source["results"]["val"][arm]["macro"][m] for m in metrics],
        )
    full = read("pdv2_revisit_fullbank_20261025", "summary.json")["results"]
    for label, arm in {
        "시작 모델": "start",
        "정상 학습": "clean_rank",
        "전체 키 제약": "pair_key",
        "문맥 계수0.01": "context_clipped",
        "문맥 계수0.005132463": "context_matched",
    }.items():
        cells += compare(
            "전체 오답·" + label,
            [full[arm][name]["any_positive_win"] for name in ("raw", "context", "key")],
        )
    support = read("pdv2_revisit_support_20261025/completed", "summary.json")
    for radius in (5, 10):
        row = support["temporal_averages"][str(radius)]
        cells += compare(
            "시간 평균·앞뒤" + str(radius),
            [row["any_positive_win"], row["all_positive_win"]],
        )
    score = read("pdv2_temporal_mean_score_20261025", "summary.json")["results"]["val"]
    for label, arm in (("기존 문맥", "context"), ("시간 평균", "mean5")):
        cells += compare(
            "점수 대조·" + label, [100 * score[arm]["macro"][m] for m in metrics]
        )
    assert cells == 80, cells
    wiki = subprocess.run(
        ["python3", "docs/도구/check_wiki.py", "docs"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    paths = (
        sorted((REPO / "docs").glob("*.md"))
        + sorted((REPO / "docs/방법").glob("*.md"))
        + sorted((REPO / "docs/평가").glob("*.md"))
    )
    mermaid = subprocess.run(
        ["node", "/tmp/nsd_wiki_mmd/check_mermaid.mjs", *map(str, paths)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    for name in (
        "학습과 선택.md",
        "방법/궤적 문맥.md",
        "개요.md",
        "결정과 근거.md",
        "할 일.md",
        "RAW/실험일지/2026-10-25_재방문-강도정합과-관측진단.md",
    ):
        path = REPO / "docs" / name
        sources[str(path.relative_to(REPO))] = digest(path)
    sources[str(Path(__file__).relative_to(REPO))] = digest(Path(__file__))
    print(
        json.dumps(
            {
                "passed": True,
                "document_table_cells": cells,
                "verified_result_hashes": hashes,
                "wiki": wiki.stdout.strip(),
                "mermaid": mermaid.stdout.strip(),
                "sources": sources,
                "test_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
