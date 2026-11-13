#!/usr/bin/env python3
"""방향PR 재표집의 가중계산을 결정적 합성 원장 복제로 검산한다."""
from pathlib import Path
import numpy as np
import _review_direction_metrics as base
from _review_direction_bootstrap import prepare, weighted


def main() -> None:
    """동점·누락점수·중복질의·다른방향제외를 직접 복제 평가와 비교한다."""
    rng = np.random.default_rng(20261106)
    checks = 0
    for size in (1, 10, 120):
        minimum = rng.choice([2., 12., 50.], size)
        scores = rng.integers(-3, 4, size).astype(float)
        scores[::17] = np.nan
        geo = np.where(minimum > 25, 50., rng.choice([2., 8.], size))
        geo[~np.isfinite(scores)] = np.nan
        mask = rng.random(size) > .3
        prepared = prepare(scores, geo, minimum, mask)
        for _ in range(20):
            indices = rng.integers(0, size, size=size)
            counts = np.bincount(indices, minlength=size)
            actual = weighted(prepared, counts)
            expected = base.metric(scores[indices], geo[indices], minimum[indices], mask[indices])
            if expected is None:
                assert actual is None
            else:
                assert actual.shape == (4,) and np.isfinite(actual).all()
                np.testing.assert_allclose(actual, [expected[k] for k in base.KEYS], atol=1e-12, rtol=0)
            checks += 1
    base.write(base.OUT / 'direction_learning' / 'synthetic_validation.json', dict(
        status='passed', duplicated_ledger_checks=checks, max_tolerance=1e-12,
        source_sha256={str(p.relative_to(base.ROOT)): base.digest(p) for p in
                       [Path(__file__), Path('scripts/_review_direction_bootstrap.py').resolve()]}))
    print('SYNTHETIC PASSED', checks)


if __name__ == '__main__':
    main()
