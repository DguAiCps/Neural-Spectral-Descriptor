#!/usr/bin/env python3
"""독립 CPU 분할 RIV가 보존한 원래 입력과 같은지 검산한다."""
from pathlib import Path
import numpy as np


def main():
    from _review_selection_full_inputs import OUT, cache, configure
    from _review_selection_full_riv_parallel import generate
    from _review_selection_bev import write, sha
    sensor, seq = "helipr", "Roundabout01"
    ids = np.load(cache(sensor, seq))["scan_ids"]
    positions = [99, 100, 101]
    home = OUT / "riv_parallel_audit"
    home.mkdir(exist_ok=True)
    generate(sensor, seq, home, ids[positions], processes=3)
    p = configure("imlpr")
    cases = []
    for i, original in enumerate(positions):
        actual = home / f"{i:06d}.npy"
        reference = p.role_dir("test_odom", sensor, seq) / f"{original:06d}.npy"
        np.testing.assert_array_equal(np.load(actual), np.load(reference))
        cases.append(dict(position=original, raw_id=int(ids[original]), actual_sha256=sha(actual),
                          reference_sha256=sha(reference), all_arrays_exact=True))
    write(home / "validated.json", dict(all_arrays_exact=True, cases=cases,
        implementation_sha256=sha(Path(__file__).with_name("_review_selection_full_riv_parallel.py"))))


if __name__ == "__main__":
    main()
