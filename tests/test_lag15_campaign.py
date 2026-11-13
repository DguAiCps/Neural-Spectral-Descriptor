"""지연 평가의 격자·정답 경계·시험 동결을 합성 입력으로 검증한다."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _pdv2_lag15_campaign as campaign
from _pdv2_no_context_ablation import evaluate_keys, no_context_key, sha256
from _xs_eval_lib import variant_inputs


def test_grid_matches_full_ledger_including_exact_five_metre_boundary():
    rng = np.random.default_rng(31)
    n = 80
    raw = rng.random((n, 288), dtype=np.float32)
    key = no_context_key(raw, np.ones(288, np.float32))
    spec = (rng.normal(size=(n, 16, 8)) + 1j * rng.normal(size=(n, 16, 8))).astype(np.complex64)
    bank = variant_inputs(spec, 8)
    positions = np.zeros((n, 3))
    positions[:, 0] = np.arange(n) * 20
    positions[30] = (5, 0, 0)
    minimum = np.full(n, np.inf)
    for q in range(30, n):
        minimum[q] = np.linalg.norm(positions[:q - 29] - positions[q], axis=1).min()
    weights = [(1., 1., 2., 2.), (.5, 1., 2., 1.), (0., 0., 0., 0.)]
    queries = np.arange(30, n - campaign.LAG)
    grid = campaign.tuple_grid(positions, key, bank, queries, weights)
    ledgers = evaluate_keys(key, bank, positions, minimum, dict(enumerate(weights)))
    for i, w in enumerate(weights):
        result = ledgers[i]
        assert result["top1_geo_dist"][0] == 5
        assert grid[str(w)] == float((result["top1_geo_dist"][:len(queries)] <= 5).mean())
        assert np.isfinite(result["score_abs"]).all()
    assert campaign.tuple_grid(positions, key, bank, np.array([], int), weights) == {str(w): 0. for w in weights}


def test_test_guard_requires_freeze_and_rejects_changed_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "OUT", tmp_path)
    monkeypatch.setattr(campaign, "REPO", tmp_path)
    with pytest.raises(AssertionError, match="먼저"):
        campaign.frozen_guard()
    (tmp_path / "protocol.json").write_text("{}")
    chosen = tmp_path / "selected.json"
    chosen.write_text('{"tuple": [1, 1, 2, 2]}')
    stat = chosen.stat()
    frozen = {"protocol_sha256": sha256(tmp_path / "protocol.json"),
              "sha256": {chosen.name: sha256(chosen)}, "runtime_verify": [chosen.name],
              "file_identity": {chosen.name: [stat.st_size, stat.st_mtime_ns]}}
    (tmp_path / "freeze.json").write_text(json.dumps(frozen))
    assert campaign.frozen_guard(full=True) == frozen
    chosen.write_text('{"tuple": [2, 1, 2, 2]}')
    with pytest.raises(AssertionError):
        campaign.frozen_guard()
