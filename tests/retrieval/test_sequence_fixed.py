"""선택 조건만 계산하는 경로와 전체 검증 대조의 일치를 확인한다."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
from _xs_eval_lib import variant_inputs
from _pdv2_sequence_fixed_test import frozen_scores, SCORES
from retrieval.sequence_evidence import sequence_scores


def test_frozen_selected_scores_equal_validation_path_at_boundaries():
    rng=np.random.default_rng(23)
    sk=(rng.normal(size=(90,16,8))+1j*rng.normal(size=(90,16,8))).astype(np.complex64)
    bank=variant_inputs(sk,8)
    travel=np.cumsum(rng.uniform(.2,2.,90))
    q,j=np.array([30,50,70]),np.array([0,3,25])
    for ch in ("c_spec","c_col"):
        all_scores,_=sequence_scores(bank,travel,q,j,ch)
        selected=frozen_scores(bank,travel,q,j,ch,all_scores["raw"])
        for k in SCORES:
            np.testing.assert_array_equal(selected[k],all_scores[k])
