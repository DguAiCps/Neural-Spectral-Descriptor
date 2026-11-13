#!/usr/bin/env python3
"""새 입력 복구 경로의 첫 ImLPR 시험 시퀀스를 독립 검산한다."""
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT),str(ROOT/"src")]
from _review_selection_bev import read,write,sha,METRICS
from _review_selection_full_eval import OUT,query_base,positions
from _review_selection_report import independent


def main() -> None:
    """첫 네 고유모델의 질의·기하·지표·입력 해시를 확인한다."""
    frozen=OUT/"retained/selection_imlpr.json"
    selection=read(frozen)
    assert not selection["test_used_for_selection"]
    base=query_base("kitti","00","test")
    pos=positions("kitti","00","test")[1]
    nsd=ROOT/"results/pdv2_diffattn_contrast_test_20261104/test/kitti_00/diff_s1.npz"
    with np.load(nsd) as z:
        for key in ("query_idx","min_past_gt_dist"):
            np.testing.assert_array_equal(base[key],z[key])
    rows=[];errors=[]
    for path in sorted((OUT/"retained/test/imlpr").glob("*/*/kitti_00.json")):
        record=read(path);ledger=ROOT/record["ledger"]
        assert sha(ledger)==record["ledger_sha256"]
        with np.load(ledger) as z:arrays={k:z[k] for k in z.files if k!="meta"}
        for key,value in base.items():np.testing.assert_array_equal(arrays[key],value)
        q,j=arrays["query_idx"],arrays["j_star"]
        assert ((j>=0)&(j<=q-30)).all()
        np.testing.assert_allclose(arrays["top1_geo_dist"],np.linalg.norm(pos[j]-pos[q],axis=1),rtol=0,atol=1e-10)
        computed=independent(dict(arrays,score_negl2=arrays["score_cos"]))
        for key in METRICS:errors.append(abs(computed[key]-record["metrics"][key]))
        proof=record.get("audit",{}).get("input_prevalidation")
        if proof:
            source=ROOT/proof["record"]
            assert sha(source)==proof["sha256"]
            evidence=read(source)
            assert evidence["selection_sha256"]==sha(frozen)
            assert evidence["downloaded_RIV_all_files_SHA_exact"]
        rows.append(dict(ledger=record["ledger"],sha256=record["ledger_sha256"],
                         input_prevalidation=proof,reused=record.get("audit",{}).get("reused_ledger")))
    assert len(rows)==4,len(rows)
    assert sum(r["input_prevalidation"] is not None for r in rows)==2
    assert max(errors)<1e-12
    write(OUT/"first_test_independent_verification_imlpr.json",dict(
        status="first_sequence_verified_full_test_still_running",completed_unix=time.time(),
        selection_sha256=sha(frozen),sequence="kitti/00",ledgers=rows,
        maximum_metric_error=max(errors),same_current_NSD_queries=True,
        all_top1_geometries_verified=True,model_selection_changed=False))
    print("FIRST_TEST_VERIFIED",len(rows),max(errors),flush=True)


if __name__=="__main__":
    main()
