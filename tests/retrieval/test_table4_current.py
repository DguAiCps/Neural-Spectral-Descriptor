"""표4의 회전 탐색 제거와 원래 최종 판정 재현을 합성 입력으로 확인한다."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _pdv2_table4_current as current
from _pdv2_sequence_fixed_test import frozen_scores
from _xs_eval_lib import variant_inputs
from retrieval.spectral_evidence import correlation_profiles


def fixture():
    rng = np.random.default_rng(307)
    spectrum = (rng.normal(size=(100, 16, 8)) + 1j * rng.normal(size=(100, 16, 8))).astype(np.complex64)
    travel = np.cumsum(rng.uniform(.2, 1.8, 100))
    return spectrum, travel, np.array([30, 57, 81]), np.array([0, 13, 25])


def test_current_temporal_formula_exactly_reproduces_main():
    spectrum, travel, q, j = fixture()
    bank = variant_inputs(spectrum, 8)
    for channel in ("c_spec", "c_col"):
        raw = correlation_profiles(bank, q, j, channel).max(1).astype(np.float64)
        expected = frozen_scores(bank, travel, q, j, channel, raw)[current.source.METHOD]
        actual = current.temporal_score(bank, travel, q, j, channel, raw)
        np.testing.assert_array_equal(actual, expected)


def test_zero_shift_never_selects_rotation_maximum(monkeypatch):
    spectrum, travel, q, j = fixture()
    angle = 2 * np.pi * 17 / 60
    spectrum[j] = spectrum[q] * np.exp(1j * angle * np.arange(1, 9))[None, None]
    bank = variant_inputs(spectrum, 8)
    original = current.correlation_profiles

    class NoRotationMax(np.ndarray):
        def max(self, *args, **kwargs):
            raise AssertionError("회전 최대값을 사용했다")

    def profiles(*args):
        return original(*args).view(NoRotationMax)

    for channel in ("c_spec", "c_col"):
        p = original(bank, q, j, channel)
        assert np.all(p.max(1) > p[:, 0] + .01)
        monkeypatch.setattr(current, "correlation_profiles", profiles)
        score = current.temporal_score(bank, travel, q, j, channel, p[:, 0], True)
        assert score.shape == q.shape and np.isfinite(score).all()


def test_zero_shift_keeps_candidate_ids_and_matches_direct_lag_zero():
    spectrum, _, _, _ = fixture()
    rng = np.random.default_rng(309)
    key = rng.normal(size=(100, 416)).astype(np.float32)
    key /= np.linalg.norm(key, axis=1, keepdims=True)
    bank = variant_inputs(spectrum, 8)
    q = np.array([40, 60, 80])
    result = current.retrieve(key, bank, rng.normal(size=(100, 3)), np.zeros(100),
                              [1., 1., 2., 1.], q, ("full_reference", "no_alignment"))
    assert result["full_reference"][1] == result["no_alignment"][1]
    row = result["no_alignment"][0]
    for channel in ("c_spec", "c_col"):
        expected = correlation_profiles(bank, q, row["j_star"], channel)[:, 0]
        np.testing.assert_array_equal(row[channel], expected)


def test_comparator_initialization_and_graph_free_independence():
    from torch_geometric.data import Data
    torch.set_num_threads(1)
    for seed in (1, 2, 3):
        standard = current.model("standard", seed)
        primary = current.source.model("octave288", seed)
        assert sum(p.numel() for p in standard.parameters()) == 167561
        for key, value in primary.state_dict().items():
            torch.testing.assert_close(standard.state_dict()[key], value, rtol=0, atol=0)
    point = current.model("graph_free", 1).eval()
    assert sum(p.numel() for p in point.parameters()) == 167705
    torch.manual_seed(310)
    x = torch.randn(32, 288)
    edges = torch.stack((torch.arange(31), torch.arange(1, 32)))
    graph = Data(x=x, edge_index=edges, edge_attr=torch.zeros(31, 2), edge_type=torch.zeros(31, dtype=torch.long))
    with torch.no_grad():
        full = point(graph)
        single = point(Data(x=x[:1], edge_index=torch.empty((2, 0), dtype=torch.long)))
    torch.testing.assert_close(full[:1], single, atol=3e-6, rtol=3e-5)
