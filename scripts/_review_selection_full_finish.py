#!/usr/bin/env python3
"""전후보 검증이 끝난 방법의 AP/R@1을 동결하고 시험46을 순차 평가한다."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from _review_selection_bev import read,write,sha,VAL,TEST,CITY,METRICS
from _review_selection_full_eval import OUT,folder,SCORES,query_base,runtime_control_needed,runtime_policy_path


def original_choice(method: str, entry: dict) -> dict:
    """원래 GT 검증에서 주표용으로 선택한 모델을 추론 텐서 그룹에 연결한다."""
    source=read(ROOT/entry["selection_source"])
    if method in ("bevplace","imlpr","deepring"): chosen=source["best"]
    elif method in ("helios","disco"): chosen=source[source["chosen"]]
    elif method=="seqot": chosen=source["stage2"]["chosen"]
    else: chosen=dict(epoch=source["selected_epoch"],sha256=source["weight_sha256"])
    matches=[r for r in entry["candidates"] if chosen["sha256"] in {a["sha256"] for a in r["aliases"]}]
    assert len(matches)==1,(method,chosen)
    return dict(candidate=matches[0],original_record=chosen,selection_source=entry["selection_source"],
        original_protocol="GT 키프레임·양방향 |j-q|>30·계열 내 질의 pooled R@1",
        val_query_stride=source.get("val_query_stride",1),
        stage1=entry.get("fixed_stage1"))


def freeze(method: str, scope: str) -> dict:
    """검증 원장만 읽어 모든 후보를 집계하고 두 선택을 동결한다."""
    target=OUT/scope/f"selection_{method}.json"
    if target.exists(): return read(target)
    spec=read(OUT/"retained_protocol.json")
    seeds=spec["methods"][method]
    if scope=="rerun":
        from _review_selection_full_eval import candidate
        seeds={str(s):dict(candidates=[candidate(method,s,f"epoch_{e:03d}",scope) for e in range(40)]) for s in (1,2,3)}
    summary,selected,hashes,originals={},{},{},{}
    override_path=OUT/f"{method}_numerical_override.json"
    override=(read(override_path).get("records",{}) if method in ("imlpr","helios") and override_path.exists() else {})
    for seed,entry in seeds.items():
        summary[seed]={}
        for row in entry["candidates"]:
            home=folder(method,int(seed),row["id"],scope,"val")
            per={}
            for sensor,seq in VAL:
                record_path=(ROOT/override[seed][row["id"]][sensor+"/"+seq] if row["id"] in override.get(seed,{})
                             else home/f"{sensor}_{seq}.json")
                record=read(record_path)
                ledger=ROOT/record["ledger"]
                assert sha(ledger)==record["ledger_sha256"]
                hashes[record["ledger"]]=record["ledger_sha256"]
                per[sensor+"/"+seq]=record["metrics"]
            assert sum(r["n_queries"] for r in per.values())==99236
            families={f:{k:float(np.mean([r[k] for n,r in per.items() if n.startswith(f) and r["n_pos"]])) for k in METRICS}
                      for f in ("kitti","helipr","mulran")}
            macro={k:float(np.mean([f[k] for f in families.values()])) for k in METRICS}
            summary[seed][row["id"]]=dict(candidate=row,macro=macro,families=families,per_sequence=per)
        # 후보사전순은 동률에서 오름차순이다. 시험값은 여기서 읽지 않는다.
        ordered=sorted(entry["candidates"],key=lambda r:(r["epoch"],r["id"]))
        selected[seed]={objective:max(ordered,key=lambda r:summary[seed][r["id"]]["macro"][objective]) for objective in ("AP","R@1")}
        if scope=="retained":
            originals[seed]=original_choice(method,entry)
            originals[seed]["common_validation_metrics"]=summary[seed][originals[seed]["candidate"]["id"]]["macro"]
            originals[seed]["same_candidate"]={o:r["id"]==originals[seed]["candidate"]["id"] for o,r in selected[seed].items()}
    if method=="imlpr" and scope=="retained" and (OUT/"local_gpu_smoke/imlpr.json").exists():
        proof=read(OUT/"local_gpu_smoke/imlpr.json")
        if not proof.get("all_arrays_exact",False):
            close=[]
            for seed,rows in summary.items():
                if seed in override:continue
                for objective in ("AP","R@1"):
                    values=sorted((r["macro"][objective],name) for name,r in rows.items())
                    if len(values)>1 and values[-1][0]-values[-2][0]<=1e-6:
                        close.append(dict(seed=seed,objective=objective,gap=values[-1][0]-values[-2][0],candidates=[name for _,name in values]))
            if close:
                write(OUT/"imlpr_numerical_margin_request.json",dict(created_unix=time.time(),status="requires_server_validation_before_freeze",
                    threshold=1e-6,close=close,test_not_started=True,reason="교차GPU FP32 차이의 선택 영향 분리; 후보 집합과 지표는 그대로"))
                raise SystemExit(76)
    if method=="helios" and scope=="retained":
        from _review_selection_helios_margin import ensure_server_reference
        ensure_server_reference(summary)
    summ=OUT/scope/f"validation_{method}.json"; write(summ,summary)
    frozen=dict(created_unix=time.time(),method=method,scope=scope,selected=selected,original_choices=originals,
        test_used_for_selection=False,validation_sha256=sha(summ),validation_ledger_sha256=hashes,
        protocol_sha256=sha(OUT/("retained_protocol.json" if scope=="retained" else "bev_rerun_protocol.json")))
    if override:frozen["numerical_margin_override_sha256"]=sha(override_path)
    if method=="helios" and (OUT/"local_helios_plan.json").exists():
        frozen["validation_execution_partition_sha256"]=sha(OUT/"local_helios_plan.json")
        frozen["validation_numerical_guard_sha256"]=sha(OUT/"helios_numerical_margin_verified.json")
    if method in ("seqot","imlpr") or method in ("disco","helios") and runtime_policy_path(method).exists():
        frozen["same_runtime_original_policy_sha256"]=sha(runtime_policy_path(method))
    write(target,frozen)
    for completed in (summ,target):
        with completed.open("rb") as handle:os.fsync(handle.fileno())
    directory=os.open(target.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
    return frozen


def original_test(method: str, info: dict) -> dict:
    """선택을 동결한 뒤 원래 주표 모델의 동일 시험질의를 별도 열로 집계한다."""
    from _review_selection_bev import metrics
    from _review_selection_report import independent
    parent=(ROOT/info["selection_source"]).parent
    prefix="seqotl2" if method=="seqot" else method
    per,hashes={},{}
    for sensor,seq in TEST:
        path=parent/"odom"/f"dump_{prefix}_{sensor}_{seq}.npz"
        with np.load(path) as z:
            import json
            meta=json.loads(str(z["meta"])); artifacts=meta.get("artifacts",{})
            digest=(artifacts.get("stage2",{}).get("sha256") if method=="seqot" else
                    artifacts.get("checkpoint_sha256",artifacts.get("weight_sha256")))
            assert digest in {r["sha256"] for r in info["candidate"]["aliases"]}
            if method=="seqot": assert artifacts["stage1"]["sha256"]==info["stage1"]["sha256"]
            base=query_base(sensor,seq,"test"); ix=np.searchsorted(z["query_idx"],base["query_idx"])
            arrays={k:z[k][ix] for k in z.files if k!="meta"}
        for k,v in base.items(): np.testing.assert_array_equal(arrays[k],v)
        score="score_l2" if method=="seqot" else SCORES[method]
        arrays["score_negl2"]=arrays[score]
        met,check=metrics(arrays),independent(arrays)
        assert all(abs(met[k]-check[k])<1e-12 for k in METRICS if met[k] is not None)
        per[sensor+"/"+seq]=met; hashes[str(path.relative_to(ROOT))]=sha(path)
    groups={g:{k:float(np.mean([per[n][k] for n in names if per[n]["n_pos"]])) for k in METRICS} for g,names in CITY.items()}
    return dict(candidate=info["candidate"],per_sequence=per,groups=groups,ledger_sha256=hashes,
                macro={k:float(np.mean([v[k] for v in groups.values()])) for k in METRICS})


def evaluation_choices(method: str, seed: str, scope: str, frozen: dict) -> dict:
    """고정된 선택본과 필요한 동일 실행 경로의 원래 모델 대조만 돌려준다."""
    choices=dict(frozen["selected"][seed])
    if runtime_control_needed(method,seed,scope,frozen):
        assert frozen["same_runtime_original_policy_sha256"]==sha(runtime_policy_path(method))
        choices["Original_runtime"]=frozen["original_choices"][seed]["candidate"]
    return choices


def finish(method: str, scope: str) -> None:
    """선택된 고유모델·시퀀스 하나를 처리하고 다음 순환에서 이어간다."""
    frozen=freeze(method,scope)
    report_path=OUT/scope/f"report_{method}.json"
    if report_path.exists(): return
    if scope=="retained" and method in ("disco","helios"):
        validation_proof=OUT/f"validation_independent_verification_{method}.json"
        if not validation_proof.exists():
            from _review_selection_full_verify import verify
            verify(method,validation_only=True)
        proof=read(validation_proof)
        assert proof["status"]=="complete_for_validation"
        assert proof["selection_sha256"]==sha(OUT/scope/f"selection_{method}.json")
    external_plan={}
    if method=="helios" and (OUT/"helios_test_partition_policy.json").exists():
        plan_path=OUT/"local_helios_test_plan.json"
        if not plan_path.exists():
            write(OUT/"helios_test_partition_waiting.json",dict(updated_unix=time.time(),
                selection_sha256=sha(OUT/scope/"selection_helios.json"),test_not_started=True))
            raise SystemExit(76)
        external_plan=read(plan_path)
        assert external_plan["selection_sha256"]==sha(OUT/scope/"selection_helios.json")
        assert external_plan["policy_sha256"]==sha(OUT/"helios_test_partition_policy.json")
    pending_external=False
    from _review_selection_full_test_relay import request,release
    if method in ("seqot","imlpr","ringsharp"):
        changed=any(r["id"]!=frozen["original_choices"][seed]["candidate"]["id"]
                    for seed,choices in frozen["selected"].items() for r in choices.values())
        if changed:
            for sensor,seq in TEST: request(method,sensor,seq,OUT/scope/f"selection_{method}.json")
    # 같은 입력을 모든 선택모델이 소비한 뒤 정리해 디스크 사용을 제한한다.
    for sensor,seq in TEST:
        for seed,choices in frozen["selected"].items():
            choices=evaluation_choices(method,seed,scope,frozen)
            for name in dict.fromkeys(r["id"] for r in choices.values()):
                home=folder(method,int(seed),name,scope,"test")
                path=home/f"{sensor}_{seq}.json"
                if path.exists():
                    complete=read(path)
                    assert sha(ROOT/complete["ledger"])==complete["ledger_sha256"]
                    continue
                if any(str(job["seed"])==seed and job["candidate"]==name and job["sequence"]==sensor+"/"+seq
                       for job in external_plan.get("jobs",[])):
                    pending_external=True
                    continue
                control=read(OUT/"queue_control.json")
                budget=control.get("query_budget",{}).get(method,128)
                code=subprocess.call([sys.executable,"-u",str(ROOT/"scripts/_review_selection_full_eval.py"),
                    "--method",method,"--scope",scope,"--seed",seed,"--candidate",name,
                    "--sequence",sensor+"/"+seq,"--role","test","--query-budget",str(budget)])
                if code not in (0,75,76): raise RuntimeError(f"test {method}/{seed}/{name}/{sensor}/{seq}: {code}")
                if code==76: raise SystemExit(76)
                raise SystemExit(75)
        release(method,sensor,seq)
    if pending_external:raise SystemExit(76)
    summary,identity={},{ }
    from _review_selection_report import independent
    from _review_selection_bev import metrics
    errors=[]
    for seed,choices in frozen["selected"].items():
        choices=evaluation_choices(method,seed,scope,frozen)
        summary[seed]={}; identity[seed]={}
        for objective,row in choices.items():
            home=folder(method,int(seed),row["id"],scope,"test")
            per,hashes={},{ }
            for sensor,seq in TEST:
                record=read(home/f"{sensor}_{seq}.json")
                path=ROOT/record["ledger"]
                assert sha(path)==record["ledger_sha256"]
                with np.load(path) as z: arrays={k:z[k] for k in z.files if k!="meta"}
                arrays["score_negl2"]=arrays[SCORES[method]]
                actual,other=metrics(arrays),independent(arrays)
                errors.extend(abs(actual[k]-other[k]) for k in METRICS if actual[k] is not None)
                per[sensor+"/"+seq]=actual; hashes[sensor+"/"+seq]=sha(path)
            assert sum(r["n_queries"] for r in per.values())==195726
            groups={g:{k:float(np.mean([per[n][k] for n in names if per[n]["n_pos"]])) for k in METRICS} for g,names in CITY.items()}
            macro={k:float(np.mean([g[k] for g in groups.values()])) for k in METRICS}
            summary[seed][objective]=dict(candidate=row,macro=macro,groups=groups,per_sequence=per)
            identity[seed][objective]=hashes
        if choices["AP"]["id"]==choices["R@1"]["id"]:
            assert identity[seed]["AP"]==identity[seed]["R@1"]
        for first,a in choices.items():
            for second,b in choices.items():
                if a["id"]==b["id"]:assert identity[seed][first]==identity[seed][second]
    assert max(errors,default=0)<1e-12
    mean={o:{k:float(np.mean([r[o]["macro"][k] for r in summary.values()])) for k in METRICS} for o in ("AP","R@1")}
    original_results={s:original_test(method,info) for s,info in frozen.get("original_choices",{}).items()}
    controlled_original={s:summary[s].get("Original_runtime",r) for s,r in original_results.items()}
    protocol_effect={s:{k:summary[s]["R@1"]["macro"][k]-r["macro"][k] for k in METRICS} for s,r in controlled_original.items()}
    runtime_drift={s:{k:controlled_original[s]["macro"][k]-r["macro"][k] for k in METRICS} for s,r in original_results.items()}
    write(report_path,dict(completed_unix=time.time(),method=method,scope=scope,selection=frozen,
        test=summary,mean=mean,AP_selection_minus_R1_selection={k:mean["AP"][k]-mean["R@1"][k] for k in METRICS},
        original_selection_test=original_results,common_R1_minus_original_protocol_selection_test=protocol_effect,
        same_runtime_original_selection_test=controlled_original,
        current_runtime_original_minus_historical_original=runtime_drift,
        test_execution_partition=(dict(path=str((OUT/"local_helios_test_plan.json").relative_to(ROOT)),
            sha256=sha(OUT/"local_helios_test_plan.json"),same_seed_choices_use_same_device=True) if external_plan else None),
        effect_labels=dict(AP_minus_R1="동일후보·동일KISS검증에서 선택 목적만의 차이",
            common_R1_minus_original="원래GT/양방향/질의집계에서 공통KISS/과거/시퀀스집계로 바뀐 규약의 결합 영향"),
        test_ledger_hashes=identity,independent_metric_max_error=max(errors,default=0),
        limitation="보존후보 대조는 원래 전체epoch후보 복원을 의미하지 않는다; 별도 재학습은 완료 전까지 미완료"))


if __name__=="__main__":
    from _review_selection_local_reservation import require_background_available
    require_background_available()
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--method",required=True)
    p.add_argument("--scope",choices=("retained","rerun"),required=True)
    a=p.parse_args(); finish(a.method,a.scope)
