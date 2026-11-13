#!/usr/bin/env python3
"""동결된 공통 검증 선택만 같은 native 특징에 적용해 별도 평가한다."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

from _review_native_seqot_cpu_gem import BASE, OUT as CONTROL, REPO, sha, write

OUT = BASE / "common_selection_native"


def main() -> None:
    """정식 선택 명세가 있을 때만 CPU 검증 경로로 GeM·검색을 실행한다."""
    import torch
    import faiss
    torch.set_num_threads(2)
    faiss.omp_set_num_threads(2)
    manifest = json.loads((OUT / "selection_manifest.json").read_text())
    assert manifest["selection_frozen_before_native_evaluation"] is True
    native = json.loads((BASE / "protocol.json").read_text())
    controls = {f"s{s}": json.loads((CONTROL / f"v2_s{s}/report.json").read_text()) for s in [1,2,3]}
    assert all(r["passed"] for r in controls.values())
    protocol = {"selection_manifest_sha256":sha(OUT/"selection_manifest.json"),"native_protocol_sha256":sha(BASE/"protocol.json"),"label_correction_sha256":sha(BASE/"protocol_label_correction.json"),"cpu_control_sha256":{k:sha(CONTROL/f"v2_{k}/report.json") for k in controls},"implementation_sha256":sha(Path(__file__)),"execution":"검산된 CPU float32 batch1 공식GeM; native batch1 FE 고정. 공통 시험batch64 FE와혼용안함", "scope":"검증AP/R@1에서이미선택된GeM만평가. native성능으로다시선택하지않음"}
    if (OUT / "protocol.json").exists():
        assert json.loads((OUT / "protocol.json").read_text()) == protocol
    else:
        write(OUT / "protocol.json", protocol)
    spec=importlib.util.spec_from_file_location("official_gem_selected",REPO/"_external/SeqOT/modules/gem.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recspec=importlib.util.spec_from_file_location("official_seqot_recall_selected",REPO/"_external/SeqOT/test/test_seqot.py")
    recall_module=importlib.util.module_from_spec(recspec)
    recspec.loader.exec_module(recall_module)
    gtpath=REPO/native["ground_truth"]
    truth=np.load(gtpath,allow_pickle=True)
    positive=np.asarray([np.asarray(x).any() for x in truth])
    reports=[]
    for seed in [1,2,3]:
        model=f"v2_s{seed}"
        subarrays={}
        for role in ["database","query"]:
            directory=BASE/model/"sub"/role
            n=json.loads((directory/"done.json").read_text())["total"]
            sub=np.zeros((n+1,256),np.float32)
            seen=np.zeros(n,bool)
            for p in sorted(directory.glob("shard_*.npz")):
                assert sha(p)==controls[f"s{seed}"]["FE_shard_sha256"][str(p.relative_to(BASE))]
                z=np.load(p)
                sub[z["indices"]]=z["descriptors"]
                seen[z["indices"]]=True
            assert seen.all()
            subarrays[role]=sub
        for candidate in [c for c in manifest["candidates"] if c["seed"]==seed]:
            assert candidate["stage1_sha256"]==native["models"][model]["sha256"]["stage1"]
            path=REPO/candidate["stage2_path"]
            assert sha(path)==candidate["stage2_sha256"]
            root=OUT/candidate["name"]
            root.mkdir(exist_ok=True)
            if (root/"report.json").exists():
                row=json.loads((root/"report.json").read_text())
                assert row["protocol_sha256"]==sha(OUT/"protocol.json")
                reports.append(row)
                continue
            started=time.time()
            gem=module.GeM().eval()
            gem.load_state_dict(torch.load(path,map_location="cpu",weights_only=False)["state_dict"],strict=True)
            finals={}
            with torch.inference_mode():
                for role,stride in [("database",10),("query",5)]:
                    sub=subarrays[role]
                    ids=np.arange(0,len(sub),stride)
                    desc=np.zeros((len(sub)//stride+1,256),np.float32) if role=="database" else np.zeros((len(ids),256),np.float32)
                    for i,j in enumerate(ids):
                        window=np.arange(j-10,j+10)
                        if (window<0).any() or (window>=len(sub)).any():
                            window=np.full(20,j)
                        desc[i]=gem(torch.from_numpy(sub[window][None])).squeeze(1)[0].numpy()
                    assert np.isfinite(desc).all()
                    finals[role]=desc
            db,q=finals["database"],finals["query"]
            index=faiss.IndexIVFFlat(faiss.IndexFlatL2(256),256,1,faiss.METRIC_L2)
            index.train(db)
            index.add(db)
            distance,indices=index.search(q,22)
            query_ids=np.arange(0,len(subarrays["query"]),5)
            prediction=np.stack([np.broadcast_to(query_ids[:,None],indices.shape),indices*10,distance],axis=-1)
            file=root/"predicted_L2_dis.npz"
            np.savez_compressed(file,prediction.reshape(-1,1,3))
            hits={str(k):sum(bool(set(indices[i,:k]*10)&set(truth[i])) for i in range(len(truth)) if positive[i]) for k in [1,5,20]}
            official={str(k):float(recall_module.cal_topN(str(file),str(gtpath),k)) for k in [1,5,20]}
            assert official=={k:v/int(positive.sum()) for k,v in hits.items()}
            direct=[]
            for probe in [0,len(q)//2,len(q)-1]:
                d=((db.astype(np.float64)-q[probe].astype(np.float64))**2).sum(axis=1)
                assert np.argmin(d)==indices[probe,0]
                direct.append({"row":probe,"top1_equal":True})
            row={"candidate":candidate,"model":model,"positive_queries":int(positive.sum()),"total_queries":len(truth),"hits":hits,"official_recall":official,"independent_recall":{k:v/int(positive.sum()) for k,v in hits.items()},"direct_l2_checks":direct,"prediction_sha256":sha(file),"protocol_sha256":sha(OUT/"protocol.json"),"elapsed_s":time.time()-started,"scope":"선택동결후의추가native진단;원래선택결과를덮어쓰지않음"}
            write(root/"report.json",row)
            reports.append(row)
            write(OUT/"progress.json",{"completed":len(reports),"total":len(manifest["candidates"]),"last":candidate["name"]})
            print(json.dumps(row,ensure_ascii=False),flush=True)
    write(OUT/"report.json",{"complete":len(reports)==len(manifest["candidates"]),"rows":reports,"protocol_sha256":sha(OUT/"protocol.json")})


if __name__=="__main__":
    main()
