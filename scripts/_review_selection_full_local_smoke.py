#!/usr/bin/env python3
"""로컬 GPU의 같은 모델·입력 원장을 서버 참조와 대조한다."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"src")]


def main(method: str, selected_sequence: str = None, audit_only: bool = False) -> None:
    os.environ["NVIDIA_TF32_OVERRIDE"]="0"
    import torch
    from _review_selection_bev import read,write,sha,metrics
    from _review_selection_full_eval import OUT,descriptors,positions,query_base
    from _review_selection_full_l2 import ledger_exact
    torch.set_num_threads(4)
    home=OUT/"local_gpu_smoke";home.mkdir(parents=True,exist_ok=True)
    row=read(OUT/"retained_protocol.json")["methods"][method]["1"]["candidates"][0]
    assert sha(ROOT/row["path"])==row["sha256"]
    sequence=selected_sequence or {"seqot":"mulran/Sejong01","imlpr":"kitti/09","disco":"kitti360/0004"}[method]
    if selected_sequence:assert method=="disco" and sequence in ("kitti/09","kitti360/0004","kitti360/0006")
    sensor,seq=sequence.split("/")
    args=argparse.Namespace(method=method,seed=1,candidate=row["id"],scope="retained",role="val",sequence=sequence)
    ids,pos=positions(sensor,seq,"val")
    tick=time.time();desc,_=descriptors(args,row,ids,home);encode_s=time.time()-tick
    tick=time.time()
    if method in ("seqot","disco"):actual,audit=ledger_exact(desc,pos,query_base(sensor,seq,"val"))
    else:
        from _causal_bevplace_pr import ledger
        actual=ledger(desc,pos);audit={}
        ix=np.searchsorted(actual["query_idx"],query_base(sensor,seq,"val")["query_idx"])
        actual={k:v[ix] for k,v in actual.items()}
    search_s=time.time()-tick
    refpath=home/{"seqot":"reference_seqot_Sejong01.npz","imlpr":"reference_imlpr_KITTI09.npz",
                  "disco":"reference_disco_K3600004.npz"}[method]
    if method=="disco" and sequence=="kitti/09":refpath=home/"reference_disco_KITTI09.npz"
    if method=="disco" and sequence=="kitti360/0006":refpath=home/"reference_disco_K3600006.npz"
    with np.load(refpath) as z:reference={k:z[k] for k in z.files if k!="meta"}
    take=np.searchsorted(reference["query_idx"],actual["query_idx"])
    differences={k:dict(n_mismatch=int(np.count_nonzero(v!=reference[k][take])),
                        max_abs=float(np.max(np.abs(v-reference[k][take]))))
                 for k,v in actual.items() if not np.array_equal(v,reference[k][take])}
    score="score_disco" if method in ("seqot","disco") else "score_cos"
    result=dict(completed_unix=time.time(),method=method,gpu=torch.cuda.get_device_name(),candidate=row,
        checkpoint_sha256=row["sha256"],sequence=sequence,n_frames=len(ids),n_queries=len(actual["query_idx"]),
        encode_s=encode_s,search_s=search_s,cuda_peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
        all_arrays_exact=not differences,differences=differences,reference_sha256=sha(refpath),
        metrics=metrics(dict(actual,score_negl2=actual[score])),audit=audit,
        precision="원래FP32모델,원래batch·점수; SeqOT/DiSCO최종L2는원래float64")
    if not audit_only:write(home/f"{method}.json",result)
    write(home/f"{method}_{sensor}_{seq}.json",result)
    output=(f"{method}_{sensor}_{seq}_local.npz" if audit_only else f"{method}_local.npz")
    np.savez_compressed(home/output,**actual)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--method",choices=("seqot","imlpr","disco"),required=True)
    p.add_argument("--sequence")
    p.add_argument("--audit-only",action="store_true")
    args=p.parse_args();main(args.method,args.sequence,args.audit_only)
