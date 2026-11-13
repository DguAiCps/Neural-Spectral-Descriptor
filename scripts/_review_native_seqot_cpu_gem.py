#!/usr/bin/env python3
"""같은 native FE를 보존하고 공식 CPU GeM의 전체 검색 동치를 검산한다."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/pdv2_review_native_20261105/seqot"
OUT = BASE / "cpu_gem_validation"


def sha(path: Path) -> str:
    """입력과 출력 파일 전체의 SHA256을 계산한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    """완성한 JSON만 노출한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    temp.replace(path)


def prepare() -> None:
    """허용 기준을 기존 대조 및 새 선택 결과를 보기 전에 고정한다."""
    p = {"purpose": "새 common-validation GeM 선택의 native 재평가를 위한 실행 동치 검산", "unchanged": "기존 native batch1 FE 전체, 같은 N+1 배열,20개 raw 중심 창,경계 반복,공식GeM,FAISS nlist1/k22,DB/query stride10/5,공개GT", "execution_change": "GeM만 CPU float32·batch1; 창은 공식 read_one_need_descriptor_from_seq_ft와 같은 인덱스", "acceptance": "세 시드 모두 전체DB/query descriptor max error<1e-5,모든질의top1 ID 동일,R@1/5/20 정수hit 동일. 실패하면 새 선택에CPU경로를사용하지않음", "model_selection": "기존 native 가중치는 대조만. 새가중치는 별도동결common검증선택을 그대로 읽음; native결과로선택안함", "wrapper_sha256": sha(Path(__file__)), "official_gem_sha256": sha(REPO / "_external/SeqOT/modules/gem.py"), "official_window_sha256": sha(REPO / "_external/SeqOT/tools/read_samples.py"), "native_protocol_sha256": sha(BASE / "protocol.json")}
    target = OUT / "protocol.json"
    if target.exists():
        assert json.loads(target.read_text()) == p
    else:
        write(target, p)


def compute(seed: int) -> None:
    """원래 고정3모델의 전체 native 출력을 새 CPU 실행과 비교한다."""
    import torch
    import faiss
    torch.set_num_threads(2)
    faiss.omp_set_num_threads(2)
    prepare()
    model = f"v2_s{seed}"
    root = OUT / model
    root.mkdir(exist_ok=True)
    source = json.loads((BASE / "protocol.json").read_text())
    ckpt = source["models"][model]
    path = REPO / ckpt["stage2"]
    assert sha(path) == ckpt["sha256"]["stage2"]
    spec = importlib.util.spec_from_file_location("official_gem_cpu_diagnostic", REPO / "_external/SeqOT/modules/gem.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gem = module.GeM().eval()
    gem.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    finals, checks, hashes = {}, {}, {}
    started = time.time()
    with torch.inference_mode():
        for role, stride in [("database",10),("query",5)]:
            directory = BASE / model / "sub" / role
            n = json.loads((directory / "done.json").read_text())["total"]
            sub = np.zeros((n+1,256),np.float32)
            seen = np.zeros(n,bool)
            for shard in sorted(directory.glob("shard_*.npz")):
                z = np.load(shard)
                sub[z["indices"]] = z["descriptors"]
                seen[z["indices"]] = True
                hashes[str(shard.relative_to(BASE))] = sha(shard)
            assert seen.all()
            ids = np.arange(0,len(sub),stride)
            desc = np.zeros((len(sub)//stride+1,256),np.float32) if role=="database" else np.zeros((len(ids),256),np.float32)
            for i, j in enumerate(ids):
                window = np.arange(j-10,j+10)
                if (window<0).any() or (window>=len(sub)).any():
                    window = np.full(20,j)
                desc[i] = gem(torch.from_numpy(sub[window][None])).squeeze(1)[0].numpy()
            reference = np.load(BASE / model / "native" / (role+"_descriptors.npy"))
            assert desc.shape == reference.shape and np.isfinite(desc).all()
            checks[role] = {"rows":len(desc),"max_abs_error":float(np.max(np.abs(desc-reference)))}
            np.save(root / (role+"_descriptors.npy"),desc)
            finals[role] = desc
    db,query = finals["database"],finals["query"]
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(256),256,1,faiss.METRIC_L2)
    index.train(db)
    index.add(db)
    distances, ids = index.search(query,22)
    prior = np.load(BASE / model / "native/predicted_L2_dis.npz")["arr_0"].reshape(-1,22,3)
    true = np.load(REPO / source["ground_truth"],allow_pickle=True)
    positive = np.asarray([np.asarray(t).any() for t in true])
    hits = {str(k):sum(bool(set(ids[i,:k]*10)&set(true[i])) for i in range(len(true)) if positive[i]) for k in [1,5,20]}
    old = json.loads((BASE / model / "native/report.json").read_text())
    top1_changes = int(np.sum(ids[:,0]*10 != prior[:,0,1]))
    passed = all(v["max_abs_error"]<1e-5 for v in checks.values()) and top1_changes==0 and hits==old["hits"]
    report = {"passed":passed,"model":model,"descriptors":checks,"top1_changes":top1_changes,"top22_changed_cells":int(np.sum(ids*10 != prior[:,:,1])),"hits":hits,"original_hits":old["hits"],"positive":int(positive.sum()),"checkpoint_sha256":sha(path),"FE_shard_sha256":hashes,"elapsed_s":time.time()-started,"protocol_sha256":sha(OUT/"protocol.json")}
    np.savez_compressed(root / "predictions.npz",indices=ids,distances=distances)
    write(root / "report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!='FE_shard_sha256'}),flush=True)
    assert passed, "CPU 전체 동치 기준 실패; 새선택평가에사용하지않는다"


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,choices=[1,2,3])
    args=parser.parse_args()
    if args.seed is None:
        prepare()
    else:
        compute(args.seed)
