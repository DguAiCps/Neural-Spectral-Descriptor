#!/usr/bin/env python3
"""스펙트럼 전용 학습의 동일 예산 대조를 실행한다."""
import _pdv2_diffattn_spectral_only as experiment
import _pdv2_diffattn_spectral_gain_run as campaign

if __name__ == "__main__":
    experiment.base.__file__ = experiment.__file__
    campaign.main()
