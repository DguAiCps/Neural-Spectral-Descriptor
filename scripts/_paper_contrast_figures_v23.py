#!/usr/bin/env python3
"""현재 DiffAttnConv 최종 결과로 v2.3 그림4·5와 편집용 PPT를 갱신한다."""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from _paper_power_figures import read, write, sha

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
OUT = ROOT / "results/paper_v23_measured_assets_20261105/figures"
# 저자 결정(2026-11-07): ImLPR는 현재 비교군에서 뺀다(results/pdv2_baseline_revision_20261107/decision.json).
# 그림5 곡선과 편집용 PPT는 기존 결과 폴더를 덮지 않고 새 폴더에 다시 집계한다. 그림4 행렬 원천(OUT/matrix)은 그대로 읽는다.
EXCLUDED = ("imlpr",)
# 저자 결정(2026-11-07): 그림5의 RING++ 곡선은 표2와 같은 정확max 46원장에서 다시 집계한다.
# 해시로 고정한 RING++ 대체 요약을 기대값으로 쓰고, 이전 ImLPR 제외본(manuscript/figures)은 그대로 둔다.
# 저자 결정(2026-11-08): 표2의 SeqOT는 논문 학습법 + 공식 GeM 수치 안정화 3시드다. 그림5 곡선은 그 138원장으로 다시 집계하고,
# 그림4는 질의·후보·배치를 그대로 두고 SeqOT 점수만 채택 시드1 모델의 저장 descriptor로 바꾼다.
# 이전 결과 폴더(ringpp_integration/figures, OUT/matrix)는 덮지 않고 아래 새 폴더에 쓴다.
from _paper_tables_v23 import RINGPP_SUMMARY  # noqa: E402
INTEGRATION = ROOT / "results/pdv2_seqot_stable_gem_20261108/manuscript_integration_20261108"
CURVES_OUT = INTEGRATION / "figures"
SEQOT_DESC = INTEGRATION / "figures/seqot_desc_s1_test_odom"
# 그림4 행렬 거울. 새 폴더는 리포의 results/ 심볼릭 링크 너머(D)에 있으므로 링크는 절대 경로로 만든다.
MATRIX_MIRROR = CURVES_OUT / "pair_matrix"
# 사용자·root 승인(2026-11-10): 그림5의 RING++ 곡선만 표2와 같은 정확max + I2 46원장으로 다시 집계한다.
# 배치·색·글꼴·방법 목록과 다른 9곡선의 원장은 그대로다. 이전 곡선 폴더(CURVES_OUT/precision)는 덮지 않고 아래 새 폴더에 쓴다.
# 그림4(행렬·SeqOT 점수)는 CURVES_OUT을 그대로 읽는다.
from _paper_tables_v23 import RINGPP_I2_SUMMARY  # noqa: E402
# 사용자 승인(2026-11-10, "교체합시다"): 그림5의 DeepRING 곡선만 표2와 같은 식 9 W0 초기배율 138원장(시드 1·3·4)으로 다시 집계한다.
# 배치·색·글꼴·방법 목록과 다른 10곡선의 원장은 그대로다. 이전 곡선 폴더(results/pdv2_ringpp_i2_figure5_20261110)는 덮지 않는다.
from _paper_tables_v23 import DEEPRING_SUMMARY  # noqa: E402
PRECISION_OUT = ROOT / "results/pdv2_deepring_scaleinit_integration_20261110/figure5"
DEST = ROOT / "docs/paper/body_v2.3/figs/pair_analysis"
# 2026-11-06 그림5 압축: 글자 크기·데이터는 그대로, 범례 4열·두 패널 높이·사이 공백만 줄인다.
# 같은 날 확대 범위를 NSD가 앞서기 시작하는 약 79%를 포함하도록 80~100%로 넓혔다.
# 끝점 라벨은 기존처럼 지시선을 쓰고, 곡선과 겹치지 않게 NSD는 위쪽, BEVPlace++는 왼쪽 위에 둔다.
_H = 3.2
PRECISION_LAYOUT = dict(figsize=(3.35, _H), overview=[.155, 2.05/_H, .81, .95/_H], detail=[.155, .67/_H, .81, .90/_H],
                        legend=dict(loc="upper center", bbox_to_anchor=(.53, .38/_H), ncol=4, columnspacing=.8),
                        zoom_x=(80, 100), zoom_y=(55, 90), zoom_xticks=[80, 85, 90, 95, 100], zoom_yticks=[60, 70, 80, 90],
                        endpoint_offsets={"nsd": (-5, 30), "bevplace": (-20, 6)},
                        endpoint_relpos={"nsd": (1, 0), "bevplace": (1, .5)})
# 2026-11-06 가독성 보완: 인쇄 글자를 약 7~8pt로 키우고, 같은 선 모양끼리 색각 이상에서도 구분되게 색을 고른다.
# NSD 초록·BEVPlace++ 보라는 유지한다. 같은 선 모양 안 최소 색차(OKLab×100) 정상 15.1·색각 이상 7.2,
# 흰 배경 대비 3:1 이상(이전 RING#-L–BEVPlace++ 색각 이상 0.6, RING·DiSCO 대비 3:1 미만).
PRECISION_LAYOUT.update(
    fonts=dict(tick=7.4, legend=7.4, label=7.9, title=8.3, endpoint=7.6, title_pad=4),
    lw_base=1.0, handlelength=1.7,
    colors=dict(ringsharp="#2098DB", ringpp="#1F3A68", helios="#B07A12", scpp="#C0392B",
                deepring="#2A8FA0", seqot="#5E5E5E", ring="#6F8FB3", disco="#D9773A"))
# 그림 4: 바깥 축에만 눈금 숫자를 두고(가로는 1·5·10), 최고점 ×와 정답 테두리의 대비를 높인다.
MATRIX_LAYOUT = dict(
    thumb_font=6.8, title_font=7.9, tick_font=6.7, axis_font=7.3, note_font=7.0,
    cbar_tick_font=7.0, cbar_label_font=7.3,
    grid=dict(left=.085, right=.99, top=.80, bottom=.215, hspace=.24, wspace=.10),
    thumbs=dict(left=.085, right=.99, top=.968, bottom=.864, wspace=.10),
    outer_labels_only=True, xtick_positions=[0, 4, 9],
    cross_color="#0E2841", cross_size=13, cross_lw=.95, outline_halo=(1.6, "#0E2841"),
    query_label="Query", note="White outline: same place (≤5 m)   ×: highest score",
    text_y=dict(reference=.143, note=.100), cax=[.23, .060, .61, .015])


def curves() -> None:
    """현재 최종 원장으로 공통 질의·도시·시드 평균을 검산한다."""
    import _paper_baseline_figure_review as review
    import _paper_baseline_figure_preview as preview
    import _paper_precision_figure as install
    summary = read(MAIN / "summary.json")
    frozen = read(MAIN / "protocol.json")
    view = PRECISION_OUT / "main_view"
    write(view / "freeze.json", frozen)
    adapted = dict(selected="diff/matching_only", per_seed={str(s): summary["per_case"][f"diff_s{s}"] for s in (1,2,3)},
                   mean_std={"whitening25": {m: dict(mean=v, std=summary["std"]["diff"][m]) for m,v in summary["mean"]["diff"].items()}},
                   source=str((MAIN / "summary.json").relative_to(ROOT)), source_sha256=sha(MAIN / "summary.json"))
    write(view / "table2_summary.json", adapted)
    for seed in (1,2,3):
        for sequence in frozen["test"]:
            name = sequence.replace("/", "_")
            src = MAIN / "test" / name / f"diff_s{seed}.npz"
            assert sha(src) == read(src.with_suffix(".json"))["sha256"]
            dst = view / f"s{seed}" / (name+".npz")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.symlink_to(src)   # PRECISION_OUT은 results/ 심볼릭 링크 너머(D)에 있어 절대 경로로 링크한다
            write(dst.with_suffix(".json"), dict(sha256=sha(src), source=str(src.relative_to(ROOT))))
    curve_root = PRECISION_OUT / "precision"
    curve_root.mkdir(parents=True, exist_ok=True)
    review.MAIN, review.OUT = view, curve_root
    from _paper_tables_v23 import (ADOPTED_DEEPRING, ADOPTED_RINGPP, ADOPTED_RINGPP_I2, FRESCO_SUMMARY,
                                   deepring_adoption, ringpp_i2_adoption)
    # 표 생성기와 같은 검산(해시·46원장·인식과 비용의 같은 구현)을 거친 I2 채택 요약에, 같은 검산을 거친 DeepRING 초기배율 사본을
    # 기대값으로 쓴다. DeepRING 외 방법은 I2 채택 요약과 같다(deepring_adoption 안의 검사).
    i2, _, _ = ringpp_i2_adoption(read(FRESCO_SUMMARY))
    assert sha(RINGPP_I2_SUMMARY) == ADOPTED_RINGPP_I2["summary_sha256"]
    ringpp, _ = deepring_adoption(i2)
    assert sha(DEEPRING_SUMMARY) == ADOPTED_DEEPRING["summary_sha256"]
    replacement = ringpp["ringpp_i2_replacement"]
    overrides = {"ringpp": dict(folder=replacement["source"], input_sha256=replacement["ringpp_inputs_sha256"])}
    # I2 원장 파일 이름 접두어(dump_ringpp_i2)만 바꾼다. 점수 필드(score_corr)와 다른 방법의 원장 정의는 그대로다.
    review.METHODS = [(m[0], m[1], replacement["ledger_prefix"], *m[3:]) if m[0] == "ringpp" else m
                      for m in review.METHODS]
    # SeqOT 시드 폴더는 채택 요약의 원장 폴더다. 원장 해시는 채택 요약의 input_sha256으로 확인한다.
    # DeepRING 시드 폴더도 채택 요약의 원장 폴더다(접두어 dump_deepring·점수 score_corr는 같다). 원장 해시는 요약의 input_sha256으로 확인한다.
    review.SEED_FOLDERS = {**review.SEED_FOLDERS, **{m: [(int(seed), run["source"]) for seed, run
                                                         in ringpp["methods"][m]["per_seed"].items()] for m in ("seqot", "deepring")}}
    review.analyze(exclude=EXCLUDED, base_path=DEEPRING_SUMMARY, overrides=overrides)
    protocol = read(curve_root / "protocol.json")
    from _paper_tables_v23 import ADOPTED_SEQOT
    protocol.update(date_kst="2026-11-10", selected_nsd="diff/matching_only", model_protocol=str((MAIN / "protocol.json").relative_to(ROOT)),
                    excluded_methods=list(EXCLUDED), exclusion_decision="results/pdv2_baseline_revision_20261107/decision.json",
                    ringpp_adoption=ADOPTED_RINGPP_I2, ringpp_official_i1_retained=ADOPTED_RINGPP, seqot_adoption=ADOPTED_SEQOT,
                    deepring_adoption=ADOPTED_DEEPRING)
    write(curve_root / "protocol.json", protocol)
    preview.OUT = curve_root
    preview.render(layout=PRECISION_LAYOUT)
    write(curve_root / "artifact_validation.json", dict(artifact_sha256={p.name: sha(p) for p in curve_root.glob("baseline_precision_candidate.*")},
          source_curves_sha256=sha(curve_root / "curves.json")))
    original_read = install.read
    old_summary = ROOT / "results/pdv2_spectral_whitening_20261026/fixed_test/table2_summary.json"
    old_base = ROOT / "results/pdv2_baseline_seeds/summary.json"
    install.read = lambda path: (adapted if path == old_summary else ringpp if path == old_base
                                 else original_read(path))
    install.OUT, install.DEST = curve_root, DEST
    install.install(methods=len(read(curve_root / "curves.json")["methods"]))
    data = read(curve_root / "curves.json")
    points = {m: {str(p): 100*data["methods"][m]["mean"][data["precision_grid"].index(p)] for p in (.95,.99,1.)} for m in ("nsd","bevplace")}
    note = ("Final acceptance scores are swept separately per sequence, then averaged within city, over six cities, and over seeds. "
            "At 100% required precision the curve equals Table 2 R@100P. "
            + " ".join(f"At {100*p:g}% precision, NSD recall is {points['nsd'][str(p)]:.2f}% and BEVPlace++ recall is {points['bevplace'][str(p)]:.2f}%." for p in (.95,.99,1.)))
    (curve_root / "body_note.txt").write_text(note+"\n")
    html = (DEST / "index.html").read_text()
    for old,new in (("65.69",points["nsd"]["0.99"]),("62.51",points["nsd"]["1.0"]),("73.47",points["nsd"]["0.95"])):
        html = html.replace(old,f"{new:.2f}")
    (DEST / "index.html").write_text(html)
    write(PRECISION_OUT / "precision_points.json", points)
    print("CURRENT_PRECISION", points, flush=True)


def matrix() -> None:
    """예시·비교 방법은 고정하고 현재 모델의 동일100쌍 점수를 실측한다."""
    import torch
    import _paper_pair_matrix_alternatives as mat
    import _pdv2_pair_visualization_finish as finish
    import _pdv2_contrast_pipeline as pipeline
    import _pdv2_contrast_pipeline_test as test
    pipeline.e.configure()
    test.activate()
    previous_root = mat.ROOT
    new_root = OUT / "matrix"
    new_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(previous_root / "freeze.json", new_root / "freeze.json")
    name = mat.ADOPTED
    source = previous_root / name
    for p in source.rglob("*"):
        if not p.is_file() or p.name in ("scores_nsd.json","scores_nsd.npz","matrix_results.json"):
            continue
        dst = new_root / name / p.relative_to(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(os.path.relpath(p,dst.parent))
    mat.ROOT = new_root
    sequence, target = mat.config(name)
    pairs = dict(np.load(target / "pairs.npz"))
    case = read(MAIN / "protocol.json")["cases"]["diff_s1"]
    assert sha(ROOT / case["checkpoint"]) == case["checkpoint_sha256"]
    graph, travel, audit = pipeline.e.prior.graph_inputs("test",sequence)
    net = pipeline.load_model(case)
    with torch.no_grad():
        z = net(graph)
        def score(q,j):
            plan = pipeline.e.prior.base.prior.plan(travel,q,j)
            parts = []
            for start in range(0,len(q),64):
                pp = {k:torch.from_numpy(v[start:start+64]).cuda() for k,v in plan.items()}
                parts.append(pipeline.e.prior.score(z,pp).cpu().numpy())
            return np.concatenate(parts)
        value = score(pairs["query"],pairs["reference"])
        ledger_path = MAIN / "test" / sequence.replace("/","_") / "diff_s1.npz"
        ledger = dict(np.load(ledger_path))
        q = np.array(read(target / "matrix.json")["query"])
        ix = np.searchsorted(ledger["query_idx"],q)
        np.testing.assert_array_equal(q,ledger["query_idx"][ix])
        checked = score(q,ledger["j_star"][ix])
        error = float(abs(checked-ledger["score"][ix]).max())
        assert error < 8e-6, error
    np.savez_compressed(target / "scores_nsd.npz",score=value)
    write(target / "scores_nsd.json",dict(status="passed",pairs_sha256=sha(target/"pairs.npz"),
        score_sha256=sha(target/"scores_nsd.npz"),ledger_replays=len(q),max_score_error=error,
        source_sha256=sha(Path(__file__)),ledger_source=str(ledger_path.relative_to(ROOT)),ledger_sha256=sha(ledger_path),
        checkpoint=case,inputs=audit,selection="이전 그림의 질의·후보100쌍 유지. 현재 결과로 재선택하지 않음."))
    finish.DEST = DEST
    mat.install(style=MATRIX_LAYOUT)
    html = (DEST / "index.html").read_text().replace("세 시퀀스에서 NSD가 성공하고", "이전 구성에서 고른 예시를 유지했다. 당시 세 시퀀스에서 NSD가 성공하고")
    (DEST / "index.html").write_text(html)
    write(new_root / "validation.json",dict(status="passed",pairs_unchanged=sha(target/"pairs.npz")==sha(source/sequence.replace("/","_")/"pairs.npz"),
          final_ledger_replays=len(q),score_max_error=error,current_checkpoint_sha256=case["checkpoint_sha256"]))


def matrix_figure() -> None:
    """점수를 다시 계산하지 않고, 검산된 현재 점수로 그림 4만 MATRIX_LAYOUT 배치로 다시 그린다."""
    import _paper_pair_matrix_alternatives as mat
    import _pdv2_pair_visualization_finish as finish
    mat.ROOT = OUT / "matrix"
    finish.DEST = DEST
    mat.install(style=MATRIX_LAYOUT)
    html = (DEST / "index.html").read_text().replace("세 시퀀스에서 NSD가 성공하고", "이전 구성에서 고른 예시를 유지했다. 당시 세 시퀀스에서 NSD가 성공하고")
    (DEST / "index.html").write_text(html)


def matrix_seqot() -> None:
    """그림4의 질의·참조100쌍·배치·색은 그대로 두고, SeqOT 점수만 채택 시드1 모델의 저장 시험 descriptor로 바꾼다.

    점수는 공식 검색 거리의 음수(−L2)다. 같은 descriptor로 시드1 원장 전체의 승자와 점수를 다시 계산해 대조한다.
    새 추론·예시 선택은 없다. 기존 검산 폴더(OUT/matrix)는 덮지 않고 CURVES_OUT/matrix에 링크와 새 점수만 쓴다.
    """
    import _paper_pair_matrix_alternatives as mat
    import _pdv2_pair_visualization_finish as finish
    from _paper_tables_v23 import ADOPTED_SEQOT
    adopted = read(RINGPP_SUMMARY)["methods"]["seqot"]
    source, mirror = OUT / "matrix", MATRIX_MIRROR
    skip = ("matrix_results.json", "scores_seqot.npz", "scores_seqot.json", "canonical_seqot.json",
            "features_seqot.npy", "canonical_seqot")
    for p in source.rglob("*"):
        if not p.is_file() or set(p.relative_to(source).parts) & set(skip):
            continue
        dst = mirror / p.relative_to(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_symlink() and not dst.exists():
            dst.symlink_to(p.resolve())
    mat.ROOT = mirror
    sequence, target = mat.config(mat.ADOPTED)
    name = sequence.replace("/", "_")
    selection = read((ROOT / adopted["per_seed"]["1"]["source"]).parent / "train_selection.json")
    descriptor = SEQOT_DESC / f"{name}.npz"
    assert sha(descriptor) == read(SEQOT_DESC / "provenance.json")["server_sha256"]
    pairs = dict(np.load(target / "pairs.npz"))
    with np.load(descriptor) as z:
        assert str(z["stage1_sha256"]) == selection["stage1"]["chosen"]["sha256"]
        assert str(z["stage2_sha256"]) == selection["stage2"]["chosen"]["sha256"]
        np.testing.assert_array_equal(z["kf_scan_ids"][pairs["indices"]], pairs["scan_ids"])
        final = z["final"]
    assert final.dtype == np.float32 and np.isfinite(final).all()
    ledger_path = ROOT / adopted["per_seed"]["1"]["source"] / f"dump_seqotl2_{name}.npz"
    assert sha(ledger_path) == read(RINGPP_SUMMARY)["input_sha256"][str(ledger_path.relative_to(ROOT))]
    with np.load(ledger_path) as z:
        ledger = {k: z[k] for k in ("query_idx", "j_star", "score_l2")}
    # 원장 전체 재현: 과거 후보 j <= q-30의 전수 L2 최근접과 그 점수
    winners, errors = 0, []
    for q, j, s in zip(ledger["query_idx"], ledger["j_star"], ledger["score_l2"]):
        d = np.linalg.norm(final[:q - 29] - final[q], axis=1)
        winners += int(d[j] == d.min())
        errors.append(abs(float(-d[j]) - float(s)))
    assert winners == len(ledger["query_idx"]) and max(errors) < 1e-5, (winners, max(errors))
    score = -np.linalg.norm(final[pairs["query"]] - final[pairs["reference"]], axis=1)
    q = np.array(read(target / "matrix.json")["query"])
    ix = np.searchsorted(ledger["query_idx"], q)
    np.testing.assert_array_equal(ledger["query_idx"][ix], q)
    replay = np.abs(-np.linalg.norm(final[q] - final[ledger["j_star"][ix]], axis=1) - ledger["score_l2"][ix])
    np.savez_compressed(target / "scores_seqot.npz", score=score)
    write(target / "scores_seqot.json", dict(
        status="passed", pairs_sha256=sha(target / "pairs.npz"), score_sha256=sha(target / "scores_seqot.npz"),
        ledger_replays=len(q), max_score_error=float(replay.max()), full_ledger_queries=len(errors),
        full_ledger_winners_equal=winners, full_ledger_max_score_error=max(errors), dimension=int(final.shape[1]),
        descriptor=str(descriptor.relative_to(ROOT)), descriptor_sha256=sha(descriptor),
        ledger=str(ledger_path.relative_to(ROOT)), ledger_sha256=sha(ledger_path), source_sha256=sha(Path(__file__)),
        model=ADOPTED_SEQOT["label"], stage1_sha256=selection["stage1"]["chosen"]["sha256"],
        stage2_sha256=selection["stage2"]["chosen"]["sha256"],
        selection="이전 그림의 질의·참조100쌍 유지. SeqOT 점수만 채택 시드1 모델의 저장 descriptor(−L2)로 교체"))
    previous = read(source / mat.ADOPTED / "matrix_results.json")
    finish.DEST = DEST
    mat.install(style=MATRIX_LAYOUT)
    html = (DEST / "index.html").read_text().replace("세 시퀀스에서 NSD가 성공하고", "이전 구성에서 고른 예시를 유지했다. 당시 세 시퀀스에서 NSD가 성공하고")
    (DEST / "index.html").write_text(html)
    result = read(mirror / mat.ADOPTED / "matrix_results.json")
    assert result["selection"] == previous["selection"]
    assert all(result["methods"][m] == previous["methods"][m] for m in previous["methods"] if m != "seqot")
    write(mirror / "validation_seqot.json", dict(
        status="passed", pairs_unchanged=sha(target / "pairs.npz") == sha(source / mat.ADOPTED / name / "pairs.npz"),
        selection_unchanged=True, other_methods_unchanged=sorted(m for m in previous["methods"] if m != "seqot"),
        seqot_correct_top_rank={"previous": previous["methods"]["seqot"]["correct_top_rank"],
                                "adopted": result["methods"]["seqot"]["correct_top_rank"]},
        full_ledger_winners_equal=winners, full_ledger_max_score_error=max(errors), ledger_replays=len(q)))


def ppt() -> None:
    """현재 행렬과 곡선을 같은 디자인의 편집용 PPT에 담는다."""
    import _paper_pair_matrix_alternatives as mat
    # 행렬 슬라이드는 collect()가 matrix_results.json을 다시 쓴다. 기존 결과 폴더를 덮지 않도록
    # matrix_seqot가 만든 거울(원천 링크 + SeqOT 채택 점수)만 읽고 그 파일만 새로 쓴다.
    mirror = MATRIX_MIRROR
    assert read(mirror / "validation_seqot.json")["status"] == "passed"
    mat.ROOT = mirror
    import _paper_pair_figures_ppt as generate
    generate.DEST, generate.MATRIX_OUT = DEST, mat.ROOT / mat.ADOPTED
    original_text_box = generate.text_box
    def text_box(slide, name, left, top, *args, **kwargs):
        if name == "endpoint_nsd":
            top = 3.28
        return original_text_box(slide, name, left, top, *args, **kwargs)
    generate.text_box = text_box
    # 2026-11-08: 거울은 matrix_seqot가 만든 SeqOT 채택 점수를 담는다. 다시 만든 행렬 결과가 그 검산 결과와 같아야 한다.
    expected = read(mirror / mat.ADOPTED / "matrix_results.json")
    generate.main()
    assert read(mirror / mat.ADOPTED / "matrix_results.json") == expected
    files = [p for p in DEST.rglob("*") if p.is_file()]
    write(CURVES_OUT / "manifest.json",dict(model="diff/matching_only",source_protocol_sha256=sha(MAIN/"protocol.json"),
          artifacts={str(p.relative_to(DEST)):sha(p) for p in files},generator_sha256=sha(Path(__file__))))


# 2026-11-10: 그림5 DeepRING 곡선을 초기배율 원장으로 바꾼 새 사본. 이전 사본(seqot_stable_20261108, ringpp_i2_20261110)은 덮지 않고 보존한다.
PPT_COPY = "pair_figures_editable_deepring_scaleinit_20261110.pptx"
PPT_COPY_PREVIOUS = "pair_figures_editable_ringpp_i2_20261110.pptx"


def ppt_copy(out: Path) -> None:
    """현재 그림4·5와 같은 데이터·배치(MATRIX_LAYOUT·PRECISION_LAYOUT)의 편집용 PPT 새 사본을 out 폴더에 쓴다.

    기존 pair_figures_editable.pptx·ppt_manifest.json과 검산된 행렬 거울은 덮지 않는다. 행렬 결과는 out 아래 링크 거울에서
    다시 모으고, 거울의 검산 결과와 그림4 데이터 사본(data/ambiguous_matrix_results.json)과 같은지 확인한다.
    """
    import _paper_pair_matrix_alternatives as mat
    import _paper_pair_figures_ppt as generate
    from pptx import Presentation
    from pptx.util import Inches
    assert read(MATRIX_MIRROR / "validation_seqot.json")["status"] == "passed"
    stage = out / "matrix_mirror"
    for p in MATRIX_MIRROR.rglob("*"):
        # collect()가 다시 쓰는 결과 파일은 링크하지 않아 거울 원본에 쓰지 않는다.
        if p.is_file() and p.name != "matrix_results.json":
            dst = stage / p.relative_to(MATRIX_MIRROR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.is_symlink() and not dst.exists():
                dst.symlink_to(p.resolve())
    mat.ROOT = stage
    generate.DEST, generate.MATRIX_OUT = DEST, stage / mat.ADOPTED
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(3.35), Inches(4.9)
    generate.matrix_slide(prs, MATRIX_LAYOUT)
    generate.precision_slide_layout(prs, PRECISION_LAYOUT)
    collected = read(stage / mat.ADOPTED / "matrix_results.json")
    assert collected == read(MATRIX_MIRROR / mat.ADOPTED / "matrix_results.json")
    assert collected == read(DEST / "data/ambiguous_matrix_results.json")
    path = out / PPT_COPY
    prs.save(path)
    inputs = [MATRIX_MIRROR / mat.ADOPTED / "matrix_results.json", DEST / "data/ambiguous_matrix_results.json",
              DEST / "data/final_precision_curves.json", DEST / "final_precision_manifest.json"]
    inputs += [DEST / f"query_{k:02d}.png" for k in range(1, 11)]
    write(out / "ppt_copy_manifest.json", dict(
        file=PPT_COPY, sha256=sha(path), slides=len(prs.slides), slide_size_inches=[3.35, 4.9],
        matrix_layout=MATRIX_LAYOUT, precision_layout=PRECISION_LAYOUT,
        inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        generator_sha256={p.name: sha(p) for p in (Path(__file__), Path(generate.__file__))},
        preserved_not_written=["pair_figures_editable.pptx", "pair_figures_editable_seqot_stable_20261108.pptx", PPT_COPY_PREVIOUS, "ppt_manifest.json",
                               str(MATRIX_MIRROR.relative_to(ROOT))]))
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("matrix","matrix_figure","matrix_seqot","curves","ppt","ppt_copy"))
    parser.add_argument("--out", type=Path, help="ppt_copy 출력 폴더")
    parser.add_argument("--dest", type=Path, help="curves가 그림5를 설치할 폴더(기본: 원고 figs/pair_analysis). 검산용 사본에 먼저 쓸 때 준다")
    args = parser.parse_args()
    if args.dest:
        assert args.action == "curves", "--dest는 curves에만 쓴다"
        DEST = args.dest
    if args.action == "ppt_copy":
        ppt_copy(args.out)
    else:
        globals()[args.action]()
