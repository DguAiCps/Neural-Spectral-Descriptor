"""최종 판정 재현, 후보 예산, 정렬 정보 제거를 합성 입력으로 확인한다."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _pdv2_table4 as table4
from _pdv2_no_context_ablation import evaluate_keys
from _xs_eval_lib import hspec_inputs, normalize


def fixture():
    rng = np.random.default_rng(18)
    emb = normalize(rng.normal(size=(270, 416)).astype(np.float32))
    bank = hspec_inputs(rng.uniform(0, 5, (270, 20, 120)).astype(np.float32), 12)
    positions = rng.normal(size=(270, 3)).astype(np.float32) * 20
    min_past = np.full(270, 100.)
    min_past[30:] = 4.
    return emb, bank, positions, min_past, (2., 1., 2., 1.)


def test_reference_reproduces_original_final_match_and_scores():
    emb, bank, pos, distances, w = fixture()
    expected = evaluate_keys(emb, bank, pos, distances, {1: w})[1]
    actual = table4.evaluate(emb, bank, pos, distances, w, ("full_reference",))["full_reference"][0]
    for key, value in actual.items():
        np.testing.assert_array_equal(value, expected[key][:-15])


def test_candidate_caps_and_same_union_for_alignment_control():
    emb, bank, pos, distances, w = fixture()
    variants = ("key_only", "aux_only", "no_alignment", "full_reference")
    result = table4.evaluate(emb, bank, pos, distances, w, variants)
    for variant, (row, _) in result.items():
        assert np.all(row["j_star"] <= row["query_idx"] - 30)
        assert np.all(row["n_candidates"] <= 200)
        assert all(np.isfinite(v).all() for v in row.values())
        if variant in ("key_only", "aux_only"):
            np.testing.assert_array_equal(row["n_candidates"], np.minimum(200, row["query_idx"] - 29))
    assert result["no_alignment"][1] == result["full_reference"][1]


def test_alignment_removed_from_ranking_and_confidence(monkeypatch):
    emb, bank, pos, distances, w = fixture()
    def forbidden(*args):
        raise AssertionError("정렬을 제거한 조건에서 정렬 함수를 호출했다")
    monkeypatch.setattr(table4, "align_scores", forbidden)
    monkeypatch.setattr(table4, "column_align_scores", forbidden)
    row, _ = table4.evaluate(emb, bank, pos, distances, w, ("no_alignment",))["no_alignment"]
    assert "c_spec" not in row and "c_col" not in row
    # 입력의 정렬 표현을 없애도 최종 판정 점수와 후보는 같아야 한다.
    changed = (bank[0], None, None, None)
    same, _ = table4.evaluate(emb, changed, pos, distances, w, ("no_alignment",))["no_alignment"]
    for key in row:
        np.testing.assert_array_equal(row[key], same[key])
