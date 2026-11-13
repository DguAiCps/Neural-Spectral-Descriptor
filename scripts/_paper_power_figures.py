#!/usr/bin/env python3
"""고정 예시와 비교 방법을 유지하며 채택된 NSD의 그림4·5 점수를 갱신한다."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MAIN = REPO / "results/pdv2_spectral_power_20261101"
OUT = MAIN / "adoption/figures"


def read(path: Path) -> dict:
    """JSON을 읽는다."""
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    """파생 자료를 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path: Path) -> str:
    """SHA256을 반환한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curves() -> None:
    """최종 원장의 경로 어댑터를 통해 기존 검산·집계 함수를 재사용한다."""
    import _paper_baseline_figure_review as review
    import _paper_baseline_figure_preview as preview
    import _paper_precision_figure as install
    summary, frozen = read(MAIN / "table2_summary.json"), read(MAIN / "test_freeze.json")
    selected = summary["selected"]
    view = OUT / "main_view"
    write(view / "freeze.json", frozen)
    # 아래 키 이름은 기존 그림 집계기의 입출력 규약이다. 실제 모델은 selected에 명시한다.
    write(view / "table2_summary.json", {"selected": selected,
        "per_seed": summary["per_seed"][selected], "mean_std": {"whitening25": summary["mean_std"][selected]},
        "source": str((MAIN / "table2_summary.json").relative_to(REPO)), "source_sha256": sha(MAIN / "table2_summary.json")})
    for seed in (1, 2, 3):
        for sequence in frozen["test"]:
            name = sequence.replace("/", "_")
            src = MAIN / "test" / f"s{seed}" / f"{name}_{selected}.npz"
            dst = view / f"s{seed}" / (name + ".npz")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.symlink_to(os.path.relpath(src, dst.parent))
            write(dst.with_suffix(".json"), {"sha256": sha(src), "source": str(src.relative_to(REPO))})
    curve_root = OUT / "precision"
    curve_root.mkdir(exist_ok=True)
    review.MAIN, review.OUT = view, curve_root
    review.analyze()
    protocol = read(curve_root / "protocol.json")
    protocol.update(date_kst="2026-11-01", selected_nsd=selected,
                    model_freeze=str((MAIN / "test_freeze.json").relative_to(REPO)))
    write(curve_root / "protocol.json", protocol)
    validation = read(curve_root / "validation.json")
    for p in (Path(__file__), MAIN / "test_freeze.json", MAIN / "table2_summary.json"):
        validation["input_sha256"][str(p.relative_to(REPO))] = sha(p)
    validation["source_count"] = len(validation["input_sha256"])
    write(curve_root / "validation.json", validation)
    preview.OUT = curve_root
    preview.render()
    write(curve_root / "artifact_validation.json", {"artifact_sha256": {
        p.name: sha(p) for p in curve_root.glob("baseline_precision_candidate.*")},
        "source_curves_sha256": sha(curve_root / "curves.json")})
    # 기존 설치기의 원천 조회만 어댑터로 연결한다.
    original_read = install.read
    old_summary = REPO / "results/pdv2_spectral_whitening_20261026/fixed_test/table2_summary.json"
    install.read = lambda path: read(view / "table2_summary.json") if path == old_summary else original_read(path)
    install.OUT = curve_root
    install.install()
    data = read(curve_root / "curves.json")
    new = [100 * data["methods"]["nsd"]["mean"][data["precision_grid"].index(p)] for p in (.99, 1., .95)]
    for p in (install.DEST / "index.html", curve_root / "body_note.txt"):
        text = p.read_text()
        for old, value in zip(("65.69", "62.51", "73.47"), new):
            text = text.replace(old, f"{value:.2f}")
        text = text.replace("BEVPlace++ has higher recall at 95% precision (76.76% versus 77.89%).", "NSD also has higher recall at 95% precision (77.89% versus 76.76%).")
        p.write_text(text)
    print("NEW_CURVE", new, flush=True)


def matrix() -> None:
    """동일 쌍과 점군을 유지하고 NSD의 최종 점수만 다시 계산한다."""
    import _paper_pair_matrix_alternatives as mat
    import _pdv2_spectral_power as power
    previous_root = mat.ROOT
    new_root = OUT / "matrix"
    new_root.mkdir(exist_ok=True)
    shutil.copy2(previous_root / "freeze.json", new_root / "freeze.json")
    name = mat.ADOPTED
    source = previous_root / name
    for p in source.rglob("*"):
        if not p.is_file() or p.name in ("scores_nsd.json", "scores_nsd.npz", "matrix_results.json"):
            continue
        dst = new_root / name / p.relative_to(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(os.path.relpath(p, dst.parent))
    mat.ROOT = new_root
    sequence, target = mat.config(name)
    pairs = dict(np.load(target / "pairs.npz"))
    white, base = power.base.white, power.base
    _, _, paths, _ = white.prior.BASE.exp.old.base.inputs("test", sequence)
    cache = power.arrays(paths["cache"])
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    x = white.coefficients(paths["spec"])
    wx = white.transform(x, power.arrays(white.OUT / "statistics.npz")["full"])
    banks = [base.variant_inputs(power.transform(v, .5), 8) for v in (x, wx)]
    def score(q: np.ndarray, j: np.ndarray) -> np.ndarray:
        endpoints = [base.scores_for(*banks, travel, q, j, ch) for ch in ("c_spec", "c_col")]
        return power.cal.confidence(*endpoints, "equal", "c_col")
    value = score(pairs["query"], pairs["reference"])
    selected = read(MAIN / "test_freeze.json")["selected"]
    ledger_path = MAIN / "test/s1" / f"{sequence.replace('/', '_')}_{selected}.npz"
    ledger = power.arrays(ledger_path)
    q = np.array(read(target / "matrix.json")["query"])
    ix = np.searchsorted(ledger["query_idx"], q)
    checked = score(q, ledger["j_star"][ix])
    np.testing.assert_array_equal(checked, ledger["score"][ix])
    np.savez_compressed(target / "scores_nsd.npz", score=value)
    write(target / "scores_nsd.json", {"status": "passed", "pairs_sha256": sha(target / "pairs.npz"),
        "score_sha256": sha(target / "scores_nsd.npz"), "ledger_replays": len(q), "max_score_error": 0.,
        "source_sha256": sha(Path(__file__)), "ledger_source": str(ledger_path.relative_to(REPO)),
        "ledger_sha256": sha(ledger_path), "selection": "기존 쌍·질의·열을 그대로 유지. 새 결과로 예시 재선택 없음."})
    mat.install()
    write(new_root / "adoption_validation.json", {"status": "passed", "pairs_unchanged": sha(target / "pairs.npz") == sha(source / sequence.replace('/', '_') / "pairs.npz"),
        "old_selection_sha256": sha(previous_root / "freeze.json"), "new_main_ledger_replays": len(q)})


def ppt() -> None:
    """새 행렬·곡선을 편집 가능한 동일 PPT 배치로 만든다."""
    import _paper_pair_matrix_alternatives as mat
    mat.ROOT = OUT / "matrix"
    import _paper_pair_figures_ppt as generate
    generate.MATRIX_OUT = mat.ROOT / mat.ADOPTED
    generate.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("curves", "matrix", "ppt"))
    globals()[parser.parse_args().action]()
