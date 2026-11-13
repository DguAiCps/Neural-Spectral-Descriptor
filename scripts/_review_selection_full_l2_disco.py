#!/usr/bin/env python3
"""DiSCO 실제 서술자로 동일 float64 L2 가속을 검산한다."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"src")]


def verify() -> None:
    """학습 초반·말기 모델의 긴 검증 시퀀스 모든 행을 원경로와 대조한다."""
    os.environ["NVIDIA_TF32_OVERRIDE"]="0"
    import torch
    import _pd_build_test_caches as loaders
    from _review_selection_bev import read,write,sha
    from _review_selection_full_eval import OUT,RelayLoader,ready,positions,descriptors,folder,query_base
    from _review_selection_full_l2 import ledger_exact
    from _disco_official_dump import ledger
    torch.set_num_threads(4)
    loaders._make_loader=lambda s,q: RelayLoader(s,q)
    home=OUT/"l2_acceleration_audit";home.mkdir(parents=True,exist_ok=True)
    spec=read(OUT/"retained_protocol.json")
    checks=[]
    for at,sequence in [(0,"mulran/Sejong01"),(-1,"mulran/Sejong03")]:
        row=spec["methods"]["disco"]["1"]["candidates"][at]
        args=argparse.Namespace(method="disco",seed=1,candidate=row["id"],sequence=sequence,role="val",scope="retained")
        assert ready(args)
        sensor,seq=sequence.split("/")
        ids,pos=positions(sensor,seq,"val")
        desc,_=descriptors(args,row,ids,folder("disco",1,row["id"],"retained","val"))
        tick=time.time();original=ledger(desc,pos);original_s=time.time()-tick
        base=query_base(sensor,seq,"val")
        tick=time.time();actual,audit=ledger_exact(desc,pos,base);accelerated_s=time.time()-tick
        take=np.searchsorted(original["query_idx"],base["query_idx"])
        mismatches={k:int(np.count_nonzero(v!=original[k][take])) for k,v in actual.items() if not np.array_equal(v,original[k][take])}
        record=dict(candidate=row["id"],sequence=sequence,n_queries=len(take),original_s=original_s,
            accelerated_s=accelerated_s,mismatches=mismatches,audit=audit)
        checks.append(record);write(home/"disco_progress.json",checks)
        print("DISCO_L2_CHECK",record,flush=True)
        assert not mismatches,record
    write(home/"disco_validated.json",dict(completed_unix=time.time(),all_array_fields_exact=True,checks=checks,
        n_queries=sum(r["n_queries"] for r in checks),implementation_sha256=sha(ROOT/"scripts/_review_selection_full_l2.py"),
        original_sha256=sha(ROOT/"scripts/_disco_official_dump.py")))


if __name__=="__main__":
    verify()
