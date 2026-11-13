#!/usr/bin/env python3
"""SeqOT의 별도 stage1 선택 대조 후보를 추론 전에 동결한다."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"src")]
from _review_selection_bev import read,write,sha
from _review_selection_full_inputs import OUT


def catalog() -> None:
    """기존 stage1 선택 후보 전체를 실제 텐서로 중복 제거해 기록한다."""
    import torch
    from _review_selection_full_catalog import tensor_hash
    import _seqot_official_train as original
    torch.set_num_threads(2)
    target=OUT/"seqot_stage1_protocol.json"
    if target.exists(): return
    retained=read(OUT/"retained_protocol.json")
    gem=original._load_gem(None,"cpu")
    gem_hash,_,_=tensor_hash(gem.state_dict())
    seeds={}
    for seed,entry in retained["methods"]["seqot"].items():
        selection=ROOT/entry["selection_source"]
        assert sha(selection)==entry["selection_sha256"]
        source=read(selection)
        groups={}
        for epoch,info in sorted(source["stage1"]["epochs"].items(),key=lambda x:int(x[0])):
            path=ROOT/info["checkpoint"]
            assert path.exists() and sha(path)==info["sha256"]
            state=torch.load(path,map_location="cpu",weights_only=False)["state_dict"]
            digest,count,elements=tensor_hash(state)
            row=dict(epoch=int(epoch),path=str(path.relative_to(ROOT)),sha256=info["sha256"],
                model_tensor_sha256=digest,tensors=count,elements=elements)
            groups.setdefault(digest,[]).append(row)
        candidates=[]
        for digest,aliases in groups.items():
            aliases.sort(key=lambda r:r["epoch"])
            row=dict(aliases[0],aliases=aliases,id=f"e{aliases[0]['epoch']:03d}_{digest[:12]}")
            candidates.append(row)
        candidates.sort(key=lambda r:(r["epoch"],r["id"]))
        seeds[seed]=dict(candidates=candidates,original_file_count=len(source["stage1"]["epochs"]),
            selection_source=entry["selection_source"],selection_sha256=entry["selection_sha256"],
            original_choice=source["stage1"]["chosen"])
        write(OUT/"seqot_stage1_catalog_progress.json",dict(seed=seed,unique=len(candidates),updated_unix=time.time()))
        print("SEQOT_STAGE1_CATALOG",seed,len(candidates),flush=True)
    write(target,dict(created_unix=time.time(),seeds=seeds,
        fixed_stage2=dict(identity="공식 학습 전 GeM 초기화 p=3; 원래 stage1 선택 경로",model_tensor_sha256=gem_hash),
        protocol_from="retained_protocol.json과 동일 KISS검증9·과거후보·공통99,236질의·계열평균",
        selection="시드별 FE후보 AP최대/R1최대; 동률은앞완료epoch·후보ID",
        next_stage="동결된 FE선택이 원래선택과 다르면 해당FE로 원래조건의 stage2 GeM을 별도재학습",
        test_policy="FE→stage2 학습→stage2 검증선택까지 고정한 이후에만 시험46 적용",
        status="candidates_frozen_evaluation_not_started",
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
            "scripts/_seqot_official_train.py":sha(ROOT/"scripts/_seqot_official_train.py"),
            "_external/SeqOT/modules/gem.py":sha(ROOT/"_external/SeqOT/modules/gem.py")}))
    jobs=[dict(seed=int(s),candidate=r["id"],sequence=seq) for s,e in seeds.items()
          for r in e["candidates"] for seq in retained["val"]]
    write(OUT/"seqot_stage1_queue.json",dict(created_unix=time.time(),protocol_sha256=sha(target),jobs=jobs,
        future_finalization="모든 FE검증 뒤 두 선택을 고정; 필요stage2재학습계획을 추가동결"))
    write(OUT/"seqot_stage1_status.json",dict(status="held_submission_priority",completed_jobs=0,total_jobs=len(jobs),
        reason="제출우선 보존최종모델 검증545후보가 먼저; GPU추가배정/후속우선순위 후 시작",updated_unix=time.time()))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("catalog",))
    catalog()
