#!/usr/bin/env python3
"""검증된 최종 판정 비교 그림을 원고의 그림 5로 설치한다."""
import json
import shutil
from pathlib import Path

import numpy as np

from _paper_baseline_figure_review import OUT, REPO, digest, read, write


DEST = REPO / "docs/paper/body_v2.2/figs/pair_analysis"
STEM = "fig_precision_recall"
CAPTION = ("Final-detection recall at required precision levels. "
           "The boxed region in (a) is enlarged in (b); shading denotes one seed SD.")


def install(methods: int = 11) -> None:
    """채택한 PDF를 그대로 복사하고 캡션·수치·검토 화면을 갱신한다. methods는 그림의 방법 수다."""
    validation = read(OUT / "validation.json")
    artifact = read(OUT / "artifact_validation.json")
    assert validation["checks_passed"] and validation["table2_max_absolute_error"] < 1e-12
    data = read(OUT / "curves.json")
    assert len(data["methods"]) == methods
    expected_nsd = read(REPO / "results/pdv2_spectral_whitening_20261026/fixed_test/table2_summary.json")
    expected_base = read(REPO / "results/pdv2_baseline_seeds/summary.json")
    for method, row in data["methods"].items():
        expected = (expected_nsd["mean_std"]["whitening25"]["R@100P"]["mean"]
                    if method == "nsd" else expected_base["methods"][method]["mean_std"]["R@100P"]["mean"])
        np.testing.assert_allclose(row["mean"][-1], expected, atol=1e-12, rtol=0)
    for suffix in ("pdf", "png", "svg"):
        source = OUT / ("baseline_precision_candidate." + suffix)
        assert digest(source) == artifact["artifact_sha256"][source.name]
        shutil.copy2(source, DEST / (STEM + "." + suffix))
    (DEST / (STEM + ".tex")).write_text(
        "% scripts/_paper_precision_figure.py가 표 2의 검증된 최종 원장에서 설치한다.\n"
        "\\begin{figure}[t]\n\\centering\n"
        f"\\includegraphics[width=0.95\\columnwidth]{{figs/pair_analysis/{STEM}.pdf}}\n"
        f"\\caption{{\\cmedit{{{CAPTION}}}}}\n"
        "\\label{fig:precision_recall}\n\\end{figure}\n")
    captions = read(DEST / "captions.json")
    captions.pop("fig_pair_distributions", None)
    captions[STEM] = CAPTION
    write(DEST / "captions.json", captions)
    compact = dict(precision_grid=data["precision_grid"], methods={
        method: dict(label=row["label"], mean=row["mean"], std=row["std"],
                     per_seed={seed: run["recall"] for seed, run in row["per_seed"].items()})
        for method, row in data["methods"].items()})
    write(DEST / "data/final_precision_curves.json", compact)
    for name, target in (("protocol.json", "final_precision_protocol.json"),
                         ("comparison.csv", "final_precision_comparison.csv"),
                         ("validation.json", "final_precision_validation.json")):
        shutil.copy2(OUT / name, DEST / "data" / target)
    html = (DEST / "index.html").read_text()
    matrix_marker = '<img src="fig_pair_matrix.png"'
    end = html.index(">", html.index(matrix_marker)) + 1
    # 행렬 설명은 해당 생성기가 관리한다. 뒤의 그림 5 설명만 교체한다.
    html = html[:end].replace("NSD 공통 쌍 진단", "NSD 매칭과 최종 판정 비교")
    rows = "".join(
        f"<tr><td>{row['label']}</td>" + "".join(
            f"<td>{100 * row['mean'][data['precision_grid'].index(p)]:.2f}</td>"
            for p in (.95, .99, 1.)) + "</tr>" for row in data["methods"].values())
    html += (
        '<h2>Fig. 5: 정밀도 요구 수준별 최종 재현율</h2>'
        f'<img src="fig_precision_recall.png" alt="{methods}개 방법의 최종 판정 비교와 고정밀 구간 확대">'
        '<p>가로축은 요구 정밀도(%), 세로축은 그 조건을 만족하는 최대 재현율(%)이다. '
        '표 2와 같은 최종 원장·공통 질의·6도시 평균·학습 방법 3시드를 사용한다. '
        '위 패널은 전체 범위, 아래는 표시 상자의 확대다. 띠는 시드 표준편차다. '
        '100% 지점은 표 2의 R@100P와 같다. 시퀀스별 임계값 전체를 쓸어 계산한 '
        '통계이며 배포용 단일 임계값의 성능은 아니다.</p>'
        '<p>정밀도 99%에서 NSD의 재현율은 65.69%, BEVPlace++는 64.60%다. '
        '100%에서는 62.51%와 57.94%, 95%에서는 73.47%와 76.76%다. '
        '오탐을 엄격히 제한할 때의 장점을 보여주며 전체 우위로 해석하지 않는다.</p>'
        '<table><tr><th>방법</th><th>R@95P (%)</th><th>R@99P (%)</th>'
        '<th>R@100P (%)</th></tr>' + rows + '</table>'
        '<p><a href="pair_figures_editable.pptx">그림 4·5 편집용 PPT</a> · '
        '<a href="data/ambiguous_matrix_results.json">행렬 후보·원점수·순위</a> · '
        '<a href="data/final_precision_curves.json">최종 판정 곡선</a> · '
        '<a href="data/final_precision_protocol.json">집계 규칙</a> · '
        '<a href="data/final_precision_comparison.csv">비교 수치</a></p></html>')
    (DEST / "index.html").write_text(html)
    write(DEST / "final_precision_manifest.json", dict(
        source=str(OUT.relative_to(REPO)), source_curves_sha256=digest(OUT / "curves.json"),
        artifact_sha256={suffix: digest(DEST / (STEM + "." + suffix)) for suffix in ("pdf", "png", "svg")},
        label="fig:precision_recall", methods=methods, table2_endpoint_reproduced=True))


if __name__ == "__main__":
    install()
