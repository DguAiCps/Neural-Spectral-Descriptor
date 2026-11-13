#!/usr/bin/env python3
"""동결된 후보를 공통 검증질의와 기존 방법별 최종 점수로 평가한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Optional

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"src")]
from _review_selection_bev import read,write,sha,VAL,TEST,metrics,old_val
from _review_selection_full_inputs import OUT,INPUT,cache,configure,SeqSource

SCORES={"bevplace":"score_negl2","seqot":"score_disco","disco":"score_disco",
        "helios":"score_cos","imlpr":"score_cos","ringsharp":"score_trans","deepring":"score_corr"}


def imlpr_input_proof(sensor: str, seq: str, role: str) -> Optional[Path]:
    """전수 바이트 검산이 끝난 불변 입력만 반복 존재 검사에서 제외한다."""
    if role=="val":
        path=(OUT/"delivered_riv_verified"/f"{sensor}_{seq}.json" if sensor=="helipr"
              else OUT/"restart_nonheli_inputs.json")
        if not path.exists():return None
        proof=read(path)
        valid=(proof.get("all_files_exact",False) if sensor=="helipr" else
               proof.get("status")=="complete" and proof.get("server_all_file_checksum_comparison",False))
        return path if valid else None
    path=OUT/"test_input_provenance"/f"imlpr_{sensor}_{seq}.json"
    home=INPUT/"test_imlpr/data/test_odom"/f"{sensor}_{seq}"
    if not path.exists() or not (home/"transfer_complete.json").exists():return None
    proof=read(path)
    if not (proof.get("downloaded_RIV_all_files_SHA_exact",False) or
            proof.get("generated_RIV_all_files_SHA_exact",False)):return None
    assert proof["done"]["cache_sha256"]==sha(cache(sensor,seq,"test"))
    assert read(home/"transfer_complete.json")["done_sha256"]==sha(home/"done.json")
    # 재부팅 전 완료 마커만 남는 경우를 신뢰하지 않는다. 이 부팅에서 최초
    # 추론 전에 완성 입력을 다시 대조·flush하고 이후에는 같은 불변 해시를 쓴다.
    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    checked=home/"current_boot_verified.json"
    identity=dict(boot_id=boot_id,provenance_sha256=sha(path),
                  done_sha256=sha(home/"done.json"))
    previous=read(checked) if checked.exists() else {}
    if any(previous.get(k)!=v for k,v in identity.items()):
        for filename,digest in proof["generated_RIV_file_sha256"].items():
            frame=home/filename
            assert sha(frame)==digest,(sensor,seq,filename,"post-reboot input SHA mismatch")
            with frame.open("rb") as stream:os.fsync(stream.fileno())
        write(checked,dict(identity,all_files_SHA_exact=True,data_fsynced=True,
                           frames=len(proof["generated_RIV_file_sha256"]),checked_unix=time.time()))
    return path


def runtime_policy_path(method: str) -> Path:
    """기존에 동결한 정책을 수정하지 않고 방법별 추가 실행 정책을 찾는다."""
    return OUT/(f"same_runtime_original_policy_{method}.json" if method in ("disco", "helios") else "same_runtime_original_policy.json")


def runtime_control_needed(method: str, seed: str, scope: str, frozen: dict) -> bool:
    """선택이 바뀐 방법은 별도 정책에 따라 원래 모델도 같은 현재 경로로 평가한다."""
    if method not in ("seqot", "imlpr", "disco", "helios") or scope != "retained": return False
    policy=runtime_policy_path(method)
    if method in ("disco", "helios") and not policy.exists():return False
    assert policy.exists() and not read(policy)["test_used_for_policy"]
    original=frozen["original_choices"][str(seed)]["candidate"]["id"]
    return any(r["id"]!=original for r in frozen["selected"][str(seed)].values())


def candidate(method: str, seed: int, name: str, scope: str) -> dict:
    """고정 목록에 있는 후보만 읽는다."""
    if scope=="rerun":
        info=read(OUT/f"bev_rerun/s{seed}/epochs/{name}.pth.json")
        return dict(id=name,path=info["checkpoint"],sha256=info["sha256"],epoch=info["epoch"]+1)
    entry=read(OUT/"retained_protocol.json")["methods"][method][str(seed)]
    return next(r for r in entry["candidates"] if r["id"]==name)


def folder(method: str, seed: int, name: str, scope: str, role: str) -> Path:
    return OUT/scope/role/method/f"s{seed}"/name


def positions(sensor: str, seq: str, role: str):
    z=np.load(cache(sensor,seq,role))
    return z["scan_ids"],z["poses_gt"][:,:3,3].astype(np.float64)


def query_base(sensor: str, seq: str, role: str) -> dict:
    """모든 방법에 같은 질의와 과거 GT를 연결한다."""
    home=OUT/"query_base"/role
    path=home/f"{sensor}_{seq}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k:z[k] for k in z.files}
    ids,pos=positions(sensor,seq,role)
    qs=np.arange(30,len(ids)-15,dtype=np.int64)
    mind=np.array([np.linalg.norm(pos[:q-29]-pos[q],axis=1).min() for q in qs])
    arrays=dict(query_idx=qs,n_candidates=qs-29,min_past_gt_dist=mind)
    home.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)
    return arrays


def save_result(args, row: dict, arrays: dict, timing: dict, audit=None) -> None:
    """완성된 원장만 지표·해시와 함께 원자적으로 확정한다."""
    sensor,seq=args.sequence.split("/")
    dest=folder(args.method,args.seed,args.candidate,args.scope,args.role)
    dest.mkdir(parents=True,exist_ok=True)
    base=query_base(sensor,seq,args.role)
    ix=np.searchsorted(arrays["query_idx"],base["query_idx"])
    arrays={k:v[ix] for k,v in arrays.items()}
    for k,v in base.items():
        np.testing.assert_array_equal(arrays[k],v)
    assert np.all(arrays["j_star"]>=0) and np.all(arrays["j_star"]<=arrays["query_idx"]-30)
    score=SCORES[args.method]
    assert np.isfinite(arrays[score]).all()
    meta=dict(schema="pr-dump-v1",method=args.method,seed=args.seed,scope=args.scope,role=args.role,
        candidate=row,primary_score=score,cache_sha256=sha(cache(sensor,seq,args.role)),
        protocol_sha256=sha(OUT/("bev_rerun_protocol.json" if args.scope=="rerun" else "retained_protocol.json")))
    path=dest/f"dump_{sensor}_{seq}.npz"
    temp=path.with_suffix(".tmp.npz")
    np.savez_compressed(temp,meta=json.dumps(meta),**arrays); temp.replace(path)
    met=metrics(dict(arrays,score_negl2=arrays[score]))
    write(dest/f"{sensor}_{seq}.json",dict(sequence=args.sequence,candidate=row,metrics=met,
        timing=timing,audit=audit,ledger=str(path.relative_to(ROOT)),ledger_sha256=sha(path)))
    # 재부팅 뒤 완료 표시만 남고 파일 내용이 사라지는 일을 막는다.
    for completed in (path,dest/f"{sensor}_{seq}.json"):
        with completed.open("rb") as handle: os.fsync(handle.fileno())
    directory_fd=os.open(dest,os.O_RDONLY)
    try: os.fsync(directory_fd)
    finally: os.close(directory_fd)
    print("COMPLETE",args.method,args.seed,args.candidate,args.sequence,met,timing,flush=True)


def reuse(args, row: dict) -> bool:
    """같은 가중치와 규약인 기존 원장을 해시로 검산해 재사용한다."""
    sensor,seq=args.sequence.split("/")
    if args.role=="test" and args.method in ("seqot","imlpr","disco","helios") and args.scope=="retained":
        frozen=read(OUT/args.scope/f"selection_{args.method}.json")
        if runtime_control_needed(args.method,str(args.seed),args.scope,frozen): return False
    path=None
    if args.method=="bevplace" and args.scope=="retained" and args.role=="val":
        old=ROOT/"results/pdv2_review_selection_20261105"
        spec=read(old/"protocol.json")
        matches=[r for r in spec["candidates"][str(args.seed)] if r["sha256"]==row["sha256"]]
        assert len(matches)==1
        if matches[0]["id"]=="best":
            path=old_val(args.seed,sensor,seq)
            assert sha(path)==spec["source_sha256"][str(path.relative_to(ROOT))]
        else:
            record=read(old/f"val/s{args.seed}_{matches[0]['id']}/{sensor}_{seq}.json")
            path=ROOT/record["ledger"]
            assert sha(path)==record["ledger_sha256"]
    elif args.role=="test" and args.scope=="retained":
        entry=read(OUT/"retained_protocol.json")["methods"][args.method][str(args.seed)]
        parent=(ROOT/entry["selection_source"]).parent
        prefix="seqotl2" if args.method=="seqot" else args.method
        possible=[parent/"odom"/f"dump_{prefix}_{sensor}_{seq}.npz"]
        for p in possible:
            if not p.exists(): continue
            meta=json.loads(str(np.load(p)["meta"]))
            artifacts=meta.get("artifacts",{})
            hashes={artifacts.get("checkpoint_sha256"),artifacts.get("weight_sha256")}
            if args.method=="seqot":
                assert artifacts["stage1"]["sha256"]==entry["fixed_stage1"]["sha256"]
                hashes.add(artifacts["stage2"]["sha256"])
            if hashes & {a["sha256"] for a in row.get("aliases",[row])}:
                path=p; break
    if path is None:
        return False
    with np.load(path) as z:
        arrays={k:z[k] for k in z.files if k!="meta"}
    if args.method=="seqot" and "score_l2" in arrays:
        arrays["score_disco"]=arrays.pop("score_l2")
    save_result(args,row,arrays,dict(reuse_s=0),dict(reused_ledger=str(path.relative_to(ROOT)),sha256=sha(path)))
    return True


def descriptors(args, row: dict, ids: np.ndarray, home: Path) -> tuple:
    """원래 공식 모델·입력 코드를 통해 서술자를 만든다."""
    method=args.method; sensor,seq=args.sequence.split("/"); checkpoint=ROOT/row["path"]
    if method=="bevplace":
        import _bevplace_official_bev as p
        from _bevplace_official_dump import _official_model
        p.ROLES["val_odom"]=(VAL,p.ODOM,"val_","poses_gt")
        model,ds,_=_official_model(checkpoint,2,32,"cuda")
        role="val_odom" if args.role=="val" else "test_odom"
        dataset=ds.InferDataset(seq=p.seq_name(sensor,seq),dataset_path=str(p.BEV/role)+"/")
        assert [Path(p).name for p in dataset.imgs_path]==[f"{i:06d}.png" for i in ids]
        return model.infer(dataset).astype(np.float32),{}
    if method=="seqot":
        seqot_tick=time.time()
        import _seqot_official_train as t
        entry=read(OUT/"retained_protocol.json")["methods"][method][str(args.seed)]
        fixed=entry["fixed_stage1"]
        shared=OUT/"seqot_subdesc"/f"s{args.seed}"/args.role
        path=shared/f"{sensor}_{seq}.npz"
        if args.role=="val" and path.exists():
            from types import SimpleNamespace
            info=next(r for r in read(OUT/"input_relay_plan.json")["rows"] if r["sensor"]==sensor and r["seq"]==seq)
            assert info["cache_sha256"]==sha(cache(sensor,seq))
            src=SimpleNamespace(kf=ids,n_raw=info["seqot_nraw"])
        elif args.role=="val": src=SeqSource(sensor,seq)
        else:
            from _review_selection_full_test_relay import TestSeqSource,input_home
            src=(TestSeqSource(sensor,seq) if (input_home("seqot",sensor,seq)/"transfer_complete.json").exists()
                 else t.StoreSource("test_odom",sensor,seq,ext=True))
        assert np.array_equal(src.kf,ids)
        shared.mkdir(parents=True,exist_ok=True)
        if path.exists():
            z=np.load(path); need,sub=z["need"],z["sub"]
            assert str(z["fe_sha256"])==fixed["sha256"]
        else:
            assert sha(ROOT/fixed["checkpoint"])==fixed["sha256"]
            need=t.centered_need(src,ids)
            old=t.SUBDESC/"val"/f"{sensor}_{seq}.npz"
            missing=np.ones(len(need),bool)
            sub=np.empty((len(need),256),np.float32)
            reused=0
            if args.role=="val" and old.exists():
                previous=np.load(old)
                assert str(previous["stage1_sha256"])==fixed["sha256"]
                index={int(r):i for i,r in enumerate(previous["raw"])}
                previous_sub=previous["sub"]
                for i,raw in enumerate(need):
                    if int(raw) in index:
                        sub[i]=previous_sub[index[int(raw)]]; missing[i]=False; reused+=1
            if missing.any():
                fe=t._load_fe(ROOT/fixed["checkpoint"],"cuda")
                sub[missing]=t.sub_descriptors(fe,src,need[missing],"cuda",64,"centered")
            np.savez(path,need=need,sub=sub,fe_sha256=fixed["sha256"])
            write(path.with_suffix(".json"),dict(shared_fe_sha256=fixed["sha256"],
                reused_rows=reused,new_rows=int(missing.sum()),old_cache_sha256=sha(old) if reused else None,
                reuse_rule="동일 FE가중치·raw ID·공식 centered 3scan 창의 보존 서술자"))
        shared_s=time.time()-seqot_tick
        tick=time.time();gem=t._load_gem(checkpoint,"cuda");model_s=time.time()-tick
        tick=time.time();result=t.centered_gem(gem,sub,need,src.n_raw,ids,"cuda");gem_s=time.time()-tick
        return result,dict(shared_fe_sha256=fixed["sha256"],sub_rows=len(need),
                           seqot_stage_seconds=dict(shared_s=shared_s,model_s=model_s,gem_s=gem_s))
    if method=="helios":
        p=configure(method) if args.role=="val" else __import__("_helios_official_prep")
        import _helios_official_dump as d
        config=OUT/"configs"/f"helios_s{args.seed}_{args.role}_{sensor}_{seq}.txt"
        config.parent.mkdir(parents=True,exist_ok=True)
        temporary=home/f"descriptors_{sensor}_{seq}.npy"
        progress=home/f"descriptor_progress_{sensor}_{seq}.json"
        state=read(progress) if progress.exists() else dict(completed=0,seconds=0,checkpoint_sha256=row["sha256"])
        assert state["checkpoint_sha256"]==row["sha256"]
        if state["completed"]==len(ids):
            return np.load(temporary,mmap_mode="r"),dict(descriptor_inference_cumulative_s=state["seconds"])
        E,params=d._official(config)
        model=d.official_model(E,params,checkpoint,"cuda")
        actual=np.load(p.seq_dir("test_odom",sensor,seq)/"index.npz")
        np.testing.assert_array_equal(actual["scan_ids"][actual["generate"]],ids)
        desc=np.load(temporary,mmap_mode="r+") if temporary.exists() else None
        end=min(len(ids),state["completed"]+args.forward_budget)
        for start in range(state["completed"],end,128):
            chunk=ids[start:start+128]
            dataset={i:{"query":p.rel_scan_path("test_odom",sensor,seq,int(sid))} for i,sid in enumerate(chunk)}
            tick=time.time()
            values=E.get_latent_vectors(model,dataset,"cuda",params).astype(np.float32)
            if desc is None:
                desc=np.lib.format.open_memmap(temporary,mode="w+",dtype=values.dtype,shape=(len(ids),values.shape[1]))
            desc[start:start+len(chunk)]=values; desc.flush()
            with temporary.open("rb") as stream:os.fsync(stream.fileno())
            state.update(completed=start+len(chunk),seconds=state["seconds"]+time.time()-tick,total=len(ids),updated_unix=time.time(),data_fsync_before_cursor=True)
            write(progress,state)
            with progress.open("rb") as stream:os.fsync(stream.fileno())
            print("HELIOS_FORWARD",args.sequence,state["completed"],len(ids),round(state["seconds"],1),flush=True)
        if state["completed"]<len(ids):
            raise SystemExit(75)
        return desc,dict(descriptor_inference_cumulative_s=state["seconds"],checkpoint_interval_scans=128)
    if method=="imlpr":
        if args.role=="val": p=configure(method)
        else:
            from _review_selection_full_test_relay import input_home,imlpr_module
            p=(imlpr_module() if (input_home("imlpr",sensor,seq)/"transfer_complete.json").exists()
               else __import__("_imlpr_official_riv"))
        import _imlpr_official_dump as d
        from _imlpr_official_train import encode
        model,mp=d._official_model(checkpoint,"cuda")
        target=p.role_dir("test_odom",sensor,seq)
        paths=[str(target/f"{i:06d}.npy") for i in range(len(ids))]
        proof=imlpr_input_proof(sensor,seq,args.role)
        if proof is None:assert all(Path(p).exists() for p in paths),target
        return encode(model,"cuda",paths,16,2),dict(input_prevalidation=(dict(
            record=str(proof.relative_to(ROOT)),sha256=sha(proof),
            rule="전체 파일 checksum 검산 뒤 중복 exists만 생략; 원래 encode·파일열기·순서·batch16·worker2 유지") if proof else None))
    if method=="disco":
        p=configure(method) if args.role=="val" else __import__("_disco_official_prep")
        if args.role=="test" and (OUT/"disco_local_handoff.json").exists():
            p.WORK=INPUT/"test_disco";p.POLAR=p.WORK/"polar"
        if args.role=="val":
            p.cmd_gen("test_odom",args.sequence)
        import _disco_official_dump as d
        import _disco_official_train as t
        cfg,lp,ev,model=d.official_model(checkpoint)
        bits,index,_=p.load_store("test_odom",sensor,seq)
        np.testing.assert_array_equal(index["scan_ids"],ids)
        assert len(bits)==len(ids)
        bank=t.PolarBank(); bank.add("test_odom",sensor,seq,bits=bits)
        lp.load_pc_file=bank.load
        desc=t.descriptors(ev,model,bank,"test_odom",sensor,seq,len(ids))
        d.assert_descriptors_distinct(desc,args.sequence)
        return desc,{}
    raise ValueError(method)


def ready(args) -> bool:
    """검증 입력 생성·전송이 끝나기 전에는 작업을 미완료로 돌려준다."""
    sensor,seq=args.sequence.split("/")
    if args.role=="test":
        if args.method=="imlpr" and imlpr_input_proof(sensor,seq,"test") is not None:return True
        if args.method=="disco" and (OUT/"disco_local_handoff.json").exists():
            marker=OUT/"disco_test_input_restore_state.json"
            return marker.exists() and read(marker).get("status")=="complete"
        if args.method not in ("seqot","imlpr","ringsharp"): return True
        if args.method=="seqot":
            import _seqot_official_prep as p
            if p.range_complete("test_odom",sensor,seq,"centered"): return True
        if args.method=="imlpr":
            import _imlpr_official_riv as p
            home=p.role_dir("test_odom",sensor,seq)
            if all((home/f"{i:06d}.npy").exists() for i in range(len(np.load(cache(sensor,seq,"test"))["scan_ids"]))): return True
        from _review_selection_full_test_relay import request
        return request(args.method,sensor,seq,OUT/args.scope/f"selection_{args.method}.json")
    if args.method=="bevplace": return True
    if args.method=="seqot": return (INPUT/"seqot"/f"{sensor}_{seq}/done.json").exists()
    if args.method in ("helios","imlpr"):
        if args.method=="imlpr" and sensor=="helipr" and (OUT/"restart_storage_plan.json").exists():
            verified=OUT/"delivered_riv_verified"/f"{sensor}_{seq}.json"
            if not verified.exists() or not read(verified).get("all_files_exact"):return False
        if args.method=="imlpr" and sensor!="helipr" and (OUT/"restart_storage_plan.json").exists():
            verified=OUT/"restart_nonheli_inputs.json"
            if not verified.exists() or read(verified).get("status")!="complete":return False
        p=configure(args.method)
        if args.method=="helios":
            home=p.seq_dir("test_odom",sensor,seq)
            if sensor!="helipr" and (home/"index.npz").exists():
                old=ROOT/f"_external/helios_official_pdv2/data/val/{sensor}_{seq}/LiDAR"
                for sid in np.load(cache(sensor,seq))["scan_ids"]:
                    source=old/f"{sid:06d}.bin"; target=home/"LiDAR"/source.name
                    if source.exists() and not target.exists(): os.link(source,target)
            return (home/"index.npz").exists() and all((home/"LiDAR"/f"{i:06d}.bin").exists() for i in np.load(cache(sensor,seq))["scan_ids"])
        home=p.role_dir("test_odom",sensor,seq)
        if args.method=="imlpr" and imlpr_input_proof(sensor,seq,"val") is not None:
            return (home/"index.npz").exists()
        return (home/"index.npz").exists() and all((home/f"{i:06d}.npy").exists() for i in range(len(np.load(cache(sensor,seq))["scan_ids"])))
    if args.method=="ringsharp": return (INPUT/"ringsharp"/f"{sensor}_{seq}/relay_missing.npz").exists()
    if args.method=="disco":
        p=configure("disco"); home=p.store_dir("test_odom",sensor,seq)
        if read(home/"meta.json").get("complete"): return True
        raw_home=INPUT/"raw"
        if (INPUT/"disco_raw").exists():
            marker=OUT/"disco_raw_restore_state.json"
            if not marker.exists() or not read(marker).get("all_files_exact",False):return False
            raw_home=INPUT/"disco_raw"
        ids=np.load(cache(sensor,seq))["scan_ids"]
        todo=np.flatnonzero(~np.load(home/"done.npy"))
        return all((raw_home/f"{sensor}_{seq}"/f"{int(ids[i]):06d}.npz").exists() for i in todo)
    row=next(r for r in read(OUT/"input_relay_plan.json")["rows"] if r["sensor"]==sensor and r["seq"]==seq)
    return all((INPUT/"raw"/f"{sensor}_{seq}"/f"{row['scan_ids'][p]:06d}.npz").exists() for p in row["shared_missing_positions"])


class RelayLoader:
    """로컬 원래 로더의 points/pose를 무손실 NPZ로 읽는다."""
    def __init__(self,sensor,seq,raw_home=None): self.home=(raw_home or INPUT/"raw")/f"{sensor}_{seq}"
    def __getitem__(self,sid):
        z=np.load(self.home/f"{sid:06d}.npz")
        return {k:z[k] for k in z.files}


def heavy(args, row, ids, pos, home):
    """원래 RING#·DeepRING 점수를 질의 경계에서 저장·재개한다."""
    import torch
    sensor,seq=args.sequence.split("/"); n=len(ids)
    base=query_base(sensor,seq,args.role); qs=base["query_idx"]
    partial=home/f"partial_{sensor}_{seq}.npz"
    arrays={k:v.copy() for k,v in base.items()}
    arrays["j_star"]=np.full(len(qs),-1,np.int64)
    arrays[SCORES[args.method]]=np.full(len(qs),np.nan)
    arrays["top1_geo_dist"]=np.full(len(qs),np.nan)
    if partial.exists():
        with np.load(partial) as z: arrays={k:z[k] for k in z.files}
    todo=np.flatnonzero(arrays["j_star"]<0)
    if not len(todo): return arrays,{}
    if args.query_offset:
        todo=np.concatenate((todo[todo>=args.query_offset],todo[todo<args.query_offset]))
    start=time.time()
    feat=home/f"features_{sensor}_{seq}.pt"
    if args.method=="ringsharp":
        import _ringsharp_official as r
        model,trans=r._base().load_model(str(ROOT/row["path"]),"cuda")
        official=r._base()
        packed_path=INPUT/"ringsharp"/f"{sensor}_{seq}/packed.npy" if args.role=="val" else INPUT/"test_ringsharp"/f"{sensor}_{seq}/packed.npy"
        if args.role=="val":
            old=np.load(r.VAL_BEV/f"{sensor}_{seq}.npz")
            lookup={int(s):i for i,s in enumerate(old["scan_ids"])}
            missing=np.load(INPUT/"ringsharp"/f"{sensor}_{seq}/relay_missing.npz")
            if not (packed_path.parent/"assembled.json").exists():
                packed=np.lib.format.open_memmap(packed_path,mode="w+",dtype=np.uint8,shape=(n,old["packed"].shape[1]))
                source=old["packed"]
                for i,sid in enumerate(ids):
                    if int(sid) in lookup: packed[i]=source[lookup[int(sid)]]
                packed[missing["positions"]]=missing["packed"]; packed.flush()
                write(packed_path.parent/"assembled.json",dict(cache_sha256=sha(cache(sensor,seq))))
            packed=np.load(packed_path,mmap_mode="r")
        else:
            packed=np.load(packed_path,mmap_mode="r")
        if feat.exists():
            saved=torch.load(feat,map_location="cpu",weights_only=False); specs,bts=saved["specs"],saved["bts"]
        else:
            _,specs,bts=r.features(official,model,trans,r.bev_batches_from_packed(official,packed),n,"cuda")
            torch.save(dict(specs=specs,bts=bts),feat)
        specs,bts=specs.cuda(),bts.cuda()
        from _review_selection_full_ring_cache import CachedRerank
        verification=read(OUT/"ring_cache_audit/validated.json")
        assert verification["implementation_sha256"]==sha(ROOT/"scripts/_review_selection_full_ring_cache.py")
        assert verification["original_sha256"]==sha(ROOT/"scripts/_ringsharp_official.py")
        cached=CachedRerank(official,trans,specs,bts)
        check_path=home/f"cache_verification_{sensor}_{seq}.json"
        checks=read(check_path) if check_path.exists() else {}
        check_queries={n//4,n//2,n-16}
        def top(q):
            end=int(q-29); cand=np.arange(end)
            dist,ang=cached.rotation_branch(int(q),cand)
            order=np.argsort(dist); chosen=order[:r._depth(end)]
            qfeat=r.query_features(official,model,packed[q:q+1],"cuda")
            errs=cached.translation_branch(qfeat,cand[chosen],ang[chosen])
            if int(q) in check_queries and str(q) not in checks:
                original_dist,original_ang=r.rotation_branch(official,specs,specs[q:q+1],cand)
                np.testing.assert_array_equal(dist,original_dist)
                np.testing.assert_array_equal(ang,original_ang)
                np.testing.assert_array_equal(order,np.argsort(original_dist))
                original_errs=r.translation_branch(official,trans,bts,qfeat,cand[chosen],ang[chosen],False)
                np.testing.assert_array_equal(errs,original_errs)
                checks[str(q)]=dict(rotation_candidates=end,translation_candidates=len(chosen),all_arrays_exact=True)
                write(check_path,checks)
            idx=int(np.argmin(errs)); return int(chosen[idx]),-float(errs[idx])
    else:
        import _deepring_reimpl as d
        from _deepring_reimpl_model import SpecBank
        d.scorer_fp32_guard("cuda")
        if args.role=="val":
            packed_path=INPUT/"deepring"/f"{sensor}_{seq}.npy"
            packed_path.parent.mkdir(parents=True,exist_ok=True)
            if not packed_path.exists():
                old,_=d.load_bev("val",sensor,seq)
                oldids=d.load_cache("val",sensor,seq)[0]; lookup={int(s):i for i,s in enumerate(oldids)}
                packed=np.empty((n,*old.shape[1:]),np.uint8); core=d.rol.import_ring_core()
                loader=RelayLoader(sensor,seq)
                for i,sid in enumerate(ids):
                    if int(sid) in lookup: packed[i]=old[lookup[int(sid)]]
                    else:
                        occ,_=d.occupancy_bev(core,sensor,loader[int(sid)]["points"],"gpu")
                        packed[i]=np.packbits(occ,axis=-1)
                np.save(packed_path,packed)
            packed=np.load(packed_path,mmap_mode="r")
        else: packed,_=d.load_bev("test_odom",sensor,seq)
        model=d.load_model(ROOT/row["path"],"cuda"); W,b=model.wb()
        bank=SpecBank(n,"cuda")
        if feat.exists(): bank.bank.copy_(torch.load(feat,map_location="cuda",weights_only=False))
        else:
            d.embed_into(model,packed,d.Sino("gpu"),"cuda",bank)
            torch.save(bank.bank.cpu(),feat)
        scorer=d.BlockScorer(bank,W,b,n,32)
        def top(q):
            scores=scorer(int(q),int(q-29)); j=int(np.argmax(scores)); return j,float(scores[j]-1.0)
    encode=time.time()-start; tick=time.time()
    progress=home/f"progress_{sensor}_{seq}.json"
    previous=read(progress) if progress.exists() else {}
    def checkpoint(completed_this_chunk):
        tmp=partial.with_suffix(".tmp.npz"); np.savez_compressed(tmp,**arrays); tmp.replace(partial)
        timing=dict(feature_load_s=encode,queries_this_chunk=completed_this_chunk,search_s=time.time()-tick,
            completed_queries=int((arrays["j_star"]>=0).sum()),total_queries=len(qs))
        timing["cumulative_feature_load_s"]=previous.get("cumulative_feature_load_s",previous.get("feature_load_s",0))+encode
        timing["cumulative_search_s"]=previous.get("cumulative_search_s",previous.get("search_s",0))+timing["search_s"]
        timing["chunks"]=previous.get("chunks",int(bool(previous)))+1
        timing["cuda_peak_allocated_gb"]=torch.cuda.max_memory_allocated()/2**30
        write(progress,dict(updated_unix=time.time(),checkpoint_interval_queries=128,**timing))
        return timing
    for processed,i in enumerate(todo[:args.query_budget],1):
        q=int(qs[i]); j,score=top(q)
        arrays["j_star"][i]=j; arrays[SCORES[args.method]][i]=score
        arrays["top1_geo_dist"][i]=np.linalg.norm(pos[j]-pos[q])
        if processed%128==0: checkpoint(processed)
    timing=checkpoint(min(args.query_budget,len(todo)))
    print("CHUNK",args.method,args.sequence,timing,flush=True)
    if np.any(arrays["j_star"]<0): raise SystemExit(75)
    feat.unlink(missing_ok=True); partial.unlink(missing_ok=True)
    return arrays,timing


def main(args) -> None:
    from _review_selection_local_reservation import require_available
    require_available()
    assert args.sequence in [s+"/"+q for s,q in (VAL if args.role=="val" else TEST)]
    row=candidate(args.method,args.seed,args.candidate,args.scope)
    home=folder(args.method,args.seed,args.candidate,args.scope,args.role); home.mkdir(parents=True,exist_ok=True)
    sensor,seq=args.sequence.split("/")
    result=home/f"{sensor}_{seq}.json"
    if result.exists():
        report=read(result); assert sha(ROOT/report["ledger"])==report["ledger_sha256"]; return
    for assignment in (OUT/"aux_gpu_plan.json",OUT/"local_helios_plan.json",OUT/"local_helios_test_plan.json"):
        if args.scope=="retained" and assignment.exists():
            plan=read(assignment)
            if plan.get("role","val")!=args.role:continue
            assert plan["protocol_sha256"]==sha(OUT/"retained_protocol.json")
            owned=any(job["method"]==args.method and job["seed"]==args.seed and
                      job["candidate"]==args.candidate and job["sequence"]==args.sequence for job in plan["jobs"])
            if owned and os.environ.get("REVIEW_SELECTION_EXECUTOR")!=plan["executor"]:
                print("WAITING_ASSIGNED_EXECUTOR",plan["executor"],args.method,args.sequence,flush=True)
                raise SystemExit(76)
    assert sha(ROOT/row["path"])==row["sha256"]
    if args.role=="test":
        freeze=read(OUT/args.scope/f"selection_{args.method}.json")
        allowed=[r["id"] for r in freeze["selected"][str(args.seed)].values()]
        if runtime_control_needed(args.method,str(args.seed),args.scope,freeze):
            allowed.append(freeze["original_choices"][str(args.seed)]["candidate"]["id"])
        assert args.candidate in allowed
    if reuse(args,row): return
    if not ready(args):
        print("WAITING_INPUT",args.method,args.sequence,flush=True); raise SystemExit(76)
    # 서버에 없는 원시 입력은 로컬 원래 로더의 무손실 기록으로만 보충한다.
    if args.role=="val":
        import _pd_build_test_caches as loaders
        raw_home=INPUT/"raw"
        if args.method=="disco" and (INPUT/"disco_raw").exists():
            marker=read(OUT/"disco_raw_restore_state.json")
            assert marker["all_files_exact"] and marker["source_samples_exact"]
            raw_home=INPUT/"disco_raw"
        loaders._make_loader=lambda s,q: RelayLoader(s,q,raw_home)
    import torch
    torch.set_num_threads(4)
    os.environ["NVIDIA_TF32_OVERRIDE"]="0"
    ids,pos=positions(sensor,seq,args.role)
    if args.method in ("ringsharp","deepring"):
        arrays,timing=heavy(args,row,ids,pos,home); audit={}
    else:
        tick=time.time(); desc,audit=descriptors(args,row,ids,home); encode=time.time()-tick
        assert len(desc)==len(ids) and np.isfinite(desc).all()
        source_file=("_review_selection_prefix_batch.py" if args.method=="bevplace" else
                     "_disco_official_dump.py" if args.method in ("seqot","disco") else "_causal_bevplace_pr.py")
        digest=hashlib.sha256()
        digest.update(json.dumps([list(desc.shape),str(desc.dtype),sha(cache(sensor,seq,args.role)),
                                  sha(ROOT/"scripts"/source_file)]).encode())
        digest.update(np.ascontiguousarray(desc).tobytes())
        descriptor_hash=digest.hexdigest()
        index=OUT/"prediction_index"/args.method/args.role/f"{sensor}_{seq}"/f"{descriptor_hash}.json"
        audit["descriptor_sha256"]=descriptor_hash
        if index.exists():
            original=read(index); source=ROOT/original["ledger"]
            assert sha(source)==original["ledger_sha256"]
            with np.load(source) as z: arrays={k:z[k] for k in z.files if k!="meta"}
            audit.update(identical_descriptor_ledger=original,
                         reuse_rule="전체 서술자 바이트·운영캐시·점수 구현 해시가 같은 경우에만 원장 재사용")
            save_result(args,row,arrays,dict(encode_s=encode,search_s=0.0),audit)
            if args.method=="helios": (home/f"descriptors_{sensor}_{seq}.npy").unlink(missing_ok=True)
            return
        tick=time.time()
        if args.method=="bevplace":
            from _review_selection_prefix_batch import ledger_exact
            arrays,search_audit=ledger_exact(desc,pos); audit.update(search_audit)
        elif args.method in ("seqot","disco"):
            from _review_selection_full_l2 import ledger_exact
            validation_path=OUT/"l2_acceleration_audit"/("validated.json" if args.method=="seqot" else "disco_validated.json")
            validation=read(validation_path)
            assert validation["implementation_sha256"]==sha(ROOT/"scripts/_review_selection_full_l2.py")
            assert validation["original_sha256"]==sha(ROOT/"scripts/_disco_official_dump.py")
            arrays,search_audit=ledger_exact(desc,pos,query_base(sensor,seq,args.role))
            audit.update(search_audit,search_validation_sha256=sha(validation_path))
        else:
            enabled=OUT/"cosine_acceleration_audit"/f"enabled_{args.method}.json"
            parallel_proof=OUT/"helios_query_parallel/enabled.json"
            if args.method=="helios" and parallel_proof.exists():
                validation=read(parallel_proof)
                assert validation["all_array_fields_exact"]
                assert validation["BLAS_threads"]==1 and validation["query_workers"]==4
                assert validation["implementation_sha256"]==sha(ROOT/"scripts/_review_selection_helios_query_parallel.py")
                assert validation["original_sha256"]==sha(ROOT/"scripts/_causal_bevplace_pr.py")
                assert validation["original_prefix_sha256"]==sha(ROOT/"scripts/_causal_pr_dump.py")
                from _review_selection_helios_query_parallel import ledger_parallel
                arrays=ledger_parallel(desc,pos)
                audit.update(search_validation_sha256=sha(parallel_proof),
                             search_execution="원래 BLAS1 prefix_topk·점수·기하 계산의 독립 질의4병렬")
            elif enabled.exists():
                validation=read(enabled)
                assert validation["all_array_fields_exact"]
                assert validation["implementation_sha256"]==sha(ROOT/"scripts/_review_selection_full_cosine.py")
                assert validation["original_sha256"]==sha(ROOT/"scripts/_causal_bevplace_pr.py")
                assert validation["original_prefix_sha256"]==sha(ROOT/"scripts/_causal_pr_dump.py")
                from _review_selection_full_cosine import ledger_exact
                arrays,search_audit=ledger_exact(desc,pos,query_base(sensor,seq,args.role))
                audit.update(search_audit,search_validation_sha256=sha(enabled))
            else:
                from _causal_bevplace_pr import ledger
                arrays=ledger(desc,pos)
        timing=dict(encode_s=encode,search_s=time.time()-tick)
    save_result(args,row,arrays,timing,audit)
    if args.method not in ("ringsharp","deepring"):
        result=read(home/f"{sensor}_{seq}.json")
        write(index,dict(ledger=result["ledger"],ledger_sha256=result["ledger_sha256"],descriptor_sha256=descriptor_hash))
        if args.method=="helios": (home/f"descriptors_{sensor}_{seq}.npy").unlink(missing_ok=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method",required=True,choices=list(SCORES))
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--candidate",required=True)
    parser.add_argument("--sequence",required=True)
    parser.add_argument("--scope",choices=("retained","rerun"),default="retained")
    parser.add_argument("--role",choices=("val","test"),default="val")
    parser.add_argument("--query-budget",type=int,default=128)
    parser.add_argument("--query-offset",type=int,default=0)
    parser.add_argument("--forward-budget",type=int,default=2048)
    main(parser.parse_args())
