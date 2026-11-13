#!/usr/bin/env python3
"""복소 전달 예비와 사전 기준 통과 시 3시드 대조를 실행한다."""
from pathlib import Path

import _pdv2_temporal_derivatives_run as runner

REPO = Path(__file__).resolve().parents[1]
runner.OUT = REPO / "results/pdv2_spectral_transport_20261020"
runner.LOG = REPO / "logs/spectral_transport_20261020"
runner.SCRIPT = REPO / "scripts/_pdv2_spectral_transport.py"

if __name__ == "__main__":
    runner.main()
