#!/usr/bin/env python3
"""검증에서 고정한 NSD 조건 하나를 공통 시험 46에 적용한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/_pdv2_sequence_fixed_test.py run
"""
from __future__ import annotations
import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
import _pdv2_spectral_evidence as exp
from _pdv2_no_context_ablation import evaluate_keys, sha256, write_json
from _pdv2_pilot import TEST, CITY
from _pr_eval_lib import classify_queries, pr_sweep, operating_point
from retrieval.spectral_evidence import correlation_profiles
from retrieval.sequence_evidence import neighbor_plans, repeated_candidate_weights, path_means

OUT = REPO/"results/pdv2_sequence_fixed_test_20261018"
VAL = REPO/"results/pdv2_sequence_evidence_20261018"
METHOD = "arc_unique_b100"
SCORES = ("raw", "arc_sparse_b25", METHOD)


def frozen_scores(bank: tuple, travel: np.ndarray, q: np.ndarray,
                  j: np.ndarray, channel: str, raw: np.ndarray) -> dict:
    """선택된 조건과 미리 정한 두 기준 점수만 계산한다."""
    plans = neighbor_plans(travel, q, j)
    center = correlation_profiles(bank, q, j, channel).max(1)
    result = {"raw": raw}
    for name, beta, key in (("arc_sparse", .25, "arc_sparse_b25"), ("arc_unique", (2./3.)*(11./10.), METHOD)):
        qn, jn, valid = plans[name]
        n = len(travel)
        unique, inverse = np.unique((qn*n+jn)[valid], return_inverse=True)
        peaks = correlation_profiles(bank, unique//n, unique % n, channel).max(1)
        values = np.zeros(qn.shape, dtype=peaks.dtype)
        values[valid] = peaks[inverse]
        weights = repeated_candidate_weights(jn, valid) if name == "arc_unique" else None
        means = path_means(center, values, valid, weights)
        result[key] = (1-beta)*raw+beta*means.max(0)
    return result


def choose_tau(seed: int) -> dict:
    """검증 계열 평균 F1을 최대화하는 공통 수락 기준을 고정한다."""
    curves, candidates = {}, []
    for sensor, seq in exp.old.base.VAL:
        a = exp.arrays(VAL/f"s{seed}/standard_{sensor}_{seq}.npz")
        qc = classify_queries(a["min_past_gt_dist"])
        if not qc["n_pos"]:
            continue
        c = pr_sweep(a[METHOD], a["top1_geo_dist"], qc)
        curves[sensor+"/"+seq] = c
        candidates.append(c["thresholds"])
    grid = np.unique(np.concatenate(candidates))[::-1]
    groups = []
    for family in ("kitti", "helipr", "mulran"):
        values = []
        for seq, c in curves.items():
            if seq.startswith(family):
                p, r = c["precision"], c["recall"]
                f1 = 2*p*r/np.maximum(p+r, 1e-300)
                index = np.searchsorted(-c["thresholds"], -grid, side="right")-1
                values.append(f1[index])
        groups.append(np.mean(values, axis=0))
    macro = np.mean(groups, axis=0)
    best = int(np.argmax(macro))
    assert np.isfinite(grid[best])
    return {"tau":float(grid[best]), "validation_macro_F1":float(macro[best]),
            "rule":"양성있는검증시퀀스→3계열F1최대.공통임계값,동률높은값.시험변경없음."}


def check_freeze() -> dict:
    """선택 이후 코드·가중치·검증 결과 변경을 차단한다."""
    value = exp.read(OUT/"freeze.json")
    for p, h in value["runtime_sha256"].items():
        assert sha256(REPO/p)==h, p
    assert sha256(VAL/"summary.json")==value["validation_summary_sha256"]
    for s, row in value["seeds"].items():
        assert sha256(REPO/row["checkpoint"])==row["checkpoint_sha256"]
        assert sha256(exp.old.base.BASE/f"s{s}/confidence_selection.json")==row["confidence_sha256"]
    return value


def prepare() -> dict:
    """모델·점수·수락 기준·비교 범위를 시험 전에 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/"freeze.json").exists():
        return check_freeze()
    spec = exp.protocol(True)
    selected = exp.read(VAL/"summary.json")["selected"]
    assert selected["model"]=="standard" and selected["method"]==METHOD
    assert exp.read(VAL/"completion_validation.json")["status"]=="passed"
    runtime = dict(spec["runtime_sha256"])
    for f in (Path(__file__), REPO/"src/retrieval/sequence_evidence.py",
              REPO/"scripts/_pdv2_sequence_evidence.py", REPO/"tests/retrieval/test_sequence_fixed.py"):
        runtime[str(f.relative_to(REPO))] = sha256(f)
    seeds = {}
    for seed in (1, 2, 3):
        epoch = spec["selected"]["standard"][str(seed)]["epoch"]
        ck = exp.old.checkpoint_dir(seed,"standard_original")/f"checkpoint_epoch_{epoch}.pth"
        conf = exp.old.base.BASE/f"s{seed}/confidence_selection.json"
        seeds[str(seed)] = {"epoch":epoch,"checkpoint":str(ck.relative_to(REPO)), "checkpoint_sha256":sha256(ck),
            "confidence_sha256":sha256(conf),"confidence":exp.read(conf)["ops"][exp.OP],"new_tau":choose_tau(seed)}
    value = {"created_unix":time.time(),"selected":selected,"seeds":seeds,"runtime_sha256":runtime,
        "validation_summary_sha256":sha256(VAL/"summary.json"),"test_roster":[s+"/"+q for s,q in TEST],
        "score_conditions":SCORES,
        "scope":"검증에서고정한표준시간모델+거리대응중복가중치한조건.같은키의단일판정·희소25%는사전정한기준만보고.시험결과로설정변경·후보추가·선택안함.원고주표자동갱신없음.",
        "comparison":"BEVPlace++ 공식시드1 저장원장과195726질의동일,45양성시퀀스→6도시→NSD3시드.개발노출이있는시험이며독립새자료주장안함."}
    write_json(OUT/"freeze.json", value)
    return value


def worker(sequence: str, role: str = "test") -> None:
    """고정 모델로 새 키·최종 승자·선택된 판정만 생성한다."""
    import torch
    from _pdv2_pair_evaluate import ArrayCache
    frozen = check_freeze()
    target = OUT/role/(sequence.replace("/","_")+".json")
    if target.exists():
        return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    sensor, seq, paths, min_past = exp.old.base.inputs(role,sequence)
    cache = ArrayCache(exp.arrays(paths["cache"]))
    raw = cache["descriptors"].astype(np.float32)
    xyz = cache["poses"][:,:3,3]
    travel = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
    with np.load(paths["spec"]) as z:
        np.testing.assert_array_equal(z["scan_ids"],cache["scan_ids"])
        bank = exp.old.base.bank_for(exp.OP,z)
    results = {}
    for seed in (1,2,3):
        f = frozen["seeds"][str(seed)]
        cfg, net = exp.old.load_model(seed,"standard_original",f["epoch"])
        tick = time.perf_counter()
        emb, graph = exp.old.old.embedding(cache,sensor,raw,cfg,net,"temporal")
        key_time = time.perf_counter()-tick
        conf = f["confidence"]
        tick = time.perf_counter()
        a = evaluate_keys(emb,bank,cache["poses_gt"][:,:3,3],min_past,
                          {seed:tuple(float(w) for w in conf["tuple"])})[seed]
        keep = a["query_idx"] < len(raw)-15
        a = {k:v[keep] for k,v in a.items()}
        match_time = time.perf_counter()-tick
        ref = exp.arrays(exp.old.base.BASE/f"s{seed}/{role}_selected_symmetric_{exp.OP}_{sensor}_{seq}.npz")
        for k in ("query_idx","min_past_gt_dist"):
            np.testing.assert_array_equal(a[k],ref[k])
        a["group"] = ref["group"]
        q,j = a["query_idx"],a["j_star"]
        assert (j<=q-30).all()
        tick = time.perf_counter()
        chunks = [frozen_scores(bank,travel,q[start:start+256],j[start:start+256],conf["decision"],
                                a[conf["decision"]][start:start+256]) for start in range(0,len(q),256)]
        scores = {k:np.concatenate([r[k] for r in chunks]) for k in SCORES}
        evidence_time = time.perf_counter()-tick
        if role=="val":
            old = exp.arrays(VAL/f"s{seed}/standard_{sensor}_{seq}.npz")
            for k in ("query_idx","j_star","top1_geo_dist"):
                np.testing.assert_array_equal(a[k],old[k])
            for k in SCORES:
                np.testing.assert_array_equal(scores[k],old[k])
        prefix = []
        for i in (len(q)//3,2*len(q)//3):
            end = int(q[i]+16)
            tiny = frozen_scores(exp.truncate_bank(bank,end),travel[:end],q[i:i+1],j[i:i+1],
                                  conf["decision"],a[conf["decision"]][i:i+1])
            for k in SCORES:
                np.testing.assert_array_equal(tiny[k],scores[k][i:i+1])
            prefix.append({"q":int(q[i]),"max_error":0.})
        path = OUT/role/f"s{seed}/{sensor}_{seq}.npz"
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**{k:a[k] for k in ("query_idx","j_star","top1_geo_dist","min_past_gt_dist","group")},**scores)
        qc = classify_queries(a["min_past_gt_dist"])
        op = operating_point(scores[METHOD],a["top1_geo_dist"],qc,f["new_tau"]["tau"])
        results[str(seed)] = {"metrics":{k:exp.metrics(scores[k],a) for k in SCORES},
            "ledger":str(path.relative_to(REPO)),"sha256":sha256(path),"prefix_checks":prefix,"graph":graph,
            "fixed_tau":op,"timing_s":{"keys":key_time,"retrieval_matching":match_time,"three_scores":evidence_time}}
        del net,emb
        torch.cuda.empty_cache()
    target.parent.mkdir(parents=True,exist_ok=True)
    write_json(target,{"sequence":sequence,"role":role,"results":results,
                       "input_sha256":{str(p.relative_to(REPO)):sha256(p) for p in paths.values()}})


def report() -> None:
    """고정 시험의 도시 평균과 BEVPlace++ 차이를 함께 보고한다."""
    check_freeze()
    records = {s+"/"+q:exp.read(OUT/"test"/f"{s}_{q}.json")["results"] for s,q in TEST}
    per_seed, counts, prefix = {},0,0
    for seed in ("1","2","3"):
        per_seed[seed] = {}
        for sequence, record in records.items():
            row = record[seed]
            path = REPO/row["ledger"]
            assert sha256(path)==row["sha256"]
            a = exp.arrays(path)
            for k in SCORES:
                assert np.isfinite(a[k]).all() and exp.metrics(a[k],a)==row["metrics"][k]
            counts += len(a["query_idx"])
            prefix += len(row["prefix_checks"])
        for method in SCORES:
            cities = {}
            for city, members in CITY.items():
                chosen = [records[seq][seed]["metrics"][method] for seq in members
                          if records[seq][seed]["metrics"][method]["n_pos"]]
                assert chosen,city
                cities[city] = {m:float(np.mean([r[m] for r in chosen])) for m in exp.METRICS}
            per_seed[seed][method] = {"cities":cities,"macro":{m:float(np.mean([r[m] for r in cities.values()])) for m in exp.METRICS}}
    means = {method:{m:{"mean":float(np.mean([per_seed[s][method]["macro"][m] for s in per_seed])),
                        "std":float(np.std([per_seed[s][method]["macro"][m] for s in per_seed],ddof=1))} for m in exp.METRICS} for method in SCORES}
    bp = REPO/"results/pdv2_main_official_baselines/baselines_common.json"
    baseline = exp.read(bp)["methods"]["bevplace"]
    for sensor,seq in TEST:
        a = exp.arrays(OUT/"test"/f"s1/{sensor}_{seq}.npz")
        b = exp.arrays(REPO/f"results/pdv2_bevplace_official/odom/dump_bevplace_{sensor}_{seq}.npz")
        ix = np.searchsorted(b["query_idx"],a["query_idx"])
        for field in ("query_idx","min_past_gt_dist"):
            np.testing.assert_array_equal(b[field][ix],a[field])
    delta = {m:100*(means[METHOD][m]["mean"]-baseline["macro"][m]) for m in exp.METRICS}
    write_json(OUT/"summary.json",{"per_seed":per_seed,"mean_std":means,"bevplace_seed1":baseline["macro"],
        "delta_vs_bevplace_pp":delta,"baseline_summary_sha256":sha256(bp),"test_used":True,
        "scope":"고정한한조건의시험.시드1BEVPlace++와NSD3시드평균.시험으로선택안함."})
    write_json(OUT/"completion_validation.json",{"status":"passed","ledgers":138,"rows":counts,
        "prefix_checks":prefix,"prefix_max_error":0.,"same_queries_as_bevplace":True,"test_used":True})
    print("TEST_SUMMARY",means,"DELTA_VS_BEVPLACE_PP",delta,flush=True)


def run() -> None:
    """검증 재현을 통과한 뒤 시퀀스별 새 프로세스에서 시험한다."""
    prepare()
    tick = time.time()
    logs = REPO/"logs/sequence_fixed_test_20261018"
    logs.mkdir(parents=True,exist_ok=True)
    with (logs/"val_replay.log").open("w") as log:
        subprocess.run([sys.executable,str(Path(__file__)),"worker","--sequence","kitti360/0004","--role","val"],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    write_json(OUT/"validation_replay.json",{"status":"passed","sequence":"kitti360/0004","seeds":[1,2,3],"exact_queries_winners_scores":True})
    def invoke(sequence: str) -> None:
        with (logs/(sequence.replace("/","_")+".log")).open("w") as log:
            subprocess.run([sys.executable,str(Path(__file__)),"worker","--sequence",sequence],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(invoke,s+"/"+q):s+"/"+q for s,q in TEST}
        for done,future in enumerate(concurrent.futures.as_completed(pending),1):
            future.result()
            write_json(OUT/"status.json",{"status":"running","done":done,"total":46,"test_used":True})
            print("TEST",done,46,pending[future],flush=True)
    report()
    write_json(OUT/"status.json",{"status":"complete","runtime_s":time.time()-tick,"test_used":True})


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","worker","report","run"))
    parser.add_argument("--sequence")
    parser.add_argument("--role",default="test",choices=("test","val"))
    args = parser.parse_args()
    if args.action=="worker":
        worker(args.sequence,args.role)
    else:
        globals()[args.action]()
