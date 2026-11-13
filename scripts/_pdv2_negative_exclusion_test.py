#!/usr/bin/env python3
"""검증에서 선택한 차이 모델·25m 오답 제외를 표 2의 시험 규약으로 평가한다."""
import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
import _pdv2_negative_exclusion as training
from _pdv2_no_context_ablation import evaluate_keys, sha256, write_json

base = training.base
fixed = base.fixed
OUT = REPO/"results/pdv2_negative_exclusion_test_20261021"
LOG = REPO/"logs/negative_exclusion_test_20261021"
ARMS = ("diff_baseline", "diff_mask25")
REPLAY = ("kitti360/0004", "helipr/Roundabout02", "mulran/Sejong02")
METHOD = training.METHOD
SCORES = ("raw", METHOD)
read = training.read


def freeze(inputs: bool = False) -> dict:
    """새 시험의 코드·가중치·고정 입력을 확인한다."""
    spec = read(OUT/"freeze.json")
    for group in (("runtime", "inputs") if inputs else ("runtime",)):
        for path, expected in spec[group].items():
            assert sha256(REPO/path) == expected, path
    for row in spec["models"].values():
        assert sha256(REPO/row["checkpoint"]) == row["checkpoint_sha256"]
    return spec


def prepare() -> None:
    """시험 원장 생성 전에 두 조건의 선택·평가·비교를 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/"freeze.json").exists():
        freeze(True)
        return
    original = training.protocol(True)
    reference = fixed.check_freeze()
    summary = read(training.OUT/"summary.json")
    assert read(training.OUT/"completion_validation.json")["status"] == "passed"
    assert original["confidence"]["1"] == reference["seeds"]["1"]["confidence"]
    models = {}
    for arm in ARMS:
        folder = (training.reference_folder("center_zero") if arm == "diff_baseline"
                  else training.OUT/"s1/diff_mask25")
        selected = (summary["references"]["center_zero"] if arm == "diff_baseline"
                    else summary["per_seed"]["1"][arm])["selected"]
        epoch = selected["epoch"]
        assert epoch == 10
        ck = folder/f"checkpoints/checkpoint_epoch_{epoch}.pth"
        expected = read(folder/"training_complete.json")["checkpoints"][ck.name]
        assert sha256(ck) == expected
        models[arm] = {"seed": 1, "epoch": epoch, "checkpoint": str(ck.relative_to(REPO)),
            "checkpoint_sha256": expected, "validation_folder": str((folder/f"epoch{epoch}").relative_to(REPO)),
            "validation_selected_metrics": selected[METHOD], "parameters": 167561}
    runtime = {**reference["runtime_sha256"], **original["runtime"]}
    runtime[str(Path(__file__).relative_to(REPO))] = sha256(Path(__file__))
    sources = dict(original["inputs"])
    for p in (training.OUT/"protocol.json", training.OUT/"summary.json",
              training.OUT/"completion_validation.json", fixed.OUT/"freeze.json", fixed.OUT/"summary.json",
              REPO/"results/pdv2_main_official_baselines/baselines_common.json"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    for row in models.values():
        for p in (REPO/row["validation_folder"]).glob("*"):
            if p.is_file():
                sources[str(p.relative_to(REPO))] = sha256(p)
        p = REPO/row["checkpoint"]
        sources[str(p.relative_to(REPO))] = row["checkpoint_sha256"]
    for sensor, seq in fixed.TEST:
        sequence = sensor+"/"+seq
        _, _, paths, _ = base.exp.old.base.inputs("test", sequence)
        for p in paths.values():
            sources[str(p.relative_to(REPO))] = sha256(p)
        for p in (fixed.OUT/"test"/f"{sensor}_{seq}.json", fixed.OUT/"test/s1"/f"{sensor}_{seq}.npz",
                  REPO/f"results/pdv2_bevplace_official/odom/dump_bevplace_{sensor}_{seq}.npz"):
            sources[str(p.relative_to(REPO))] = sha256(p)
    write_json(OUT/"freeze.json", {"created_unix": time.time(), "models": models,
        "arms": list(ARMS), "primary": "diff_mask25", "seed": 1, "method": METHOD,
        "scores": list(SCORES), "confidence": original["confidence"]["1"],
        "test_roster": [s+"/"+q for s, q in fixed.TEST], "cities": fixed.CITY,
        "replay_roster": list(REPLAY), "runtime": runtime, "inputs": sources,
        "test_used_for_selection": False, "paper_changed": False,
        "scope": "사용자의 시험 평가 요청. 이미 검증에서 선택된 diff_mask25와 같은구조 무제외 기준을 각시드1·epoch10으로고정. 개발노출이 있는 기존시험이며새독립자료확증이 아님. 시험을 보고 모델/체크포인트/제외반경/융합/판정선택을 변경하지않음.",
        "comparison": "표2와같은46시퀀스195726질의·KISS·q+15·j<=q-30·100+100. KITTI07은양성없어AP/재현율평균제외,45시퀀스→6도시. 3시드평균대신시드1만명시. 동일시드기존표준시간모델과BEVPlace++보존원장비교. raw는부가,선택한최종arc_unique_b100이주지표.",
        "training_change": "두새시험모델의구조는차이+tanh·시간간선만·동일초기값과표본·900학습업데이트. 오답제외만다름. 평가에서25m후보제외없음. 기준자세는정답판정에만사용.",
        "checks": "세검증시퀀스에서원장/선택/점수재현후시험. 각시험시퀀스2지점의q+15키/검색/최종점수접두재계산. 독립PR집계/3D거리/후보융합/스칼라최종점수검산."})
    print("FROZEN", models, "sources", len(sources), flush=True)


def model(arm: str) -> torch.nn.Module:
    """새 학습을 수행하지 않고 지정한 기존 구조만 구성한다."""
    return (training.previous.model("center_zero") if arm == "diff_baseline"
            else training.model("diff_mask25"))


def worker(arm: str, sequence: str, role: str) -> None:
    """고정된 모델의 후보 검색부터 최종 판정까지 다시 실행한다."""
    spec = freeze()
    folder = OUT/role/arm
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/(sequence.replace("/", "_")+".npz")
    if path.with_suffix(".json").exists():
        info = read(path.with_suffix(".json"))
        assert sha256(path) == info["sha256"]
        assert sha256(path.with_name("keys_"+path.name)) == info["keys_sha256"]
        assert sha256(path.with_name("coverage_"+path.name)) == info["coverage_sha256"]
        return
    chosen = spec["models"][arm]
    net = model(arm).cuda().eval()
    net.load_state_dict(torch.load(REPO/chosen["checkpoint"], map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    net.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(np.load(base.exp.old.base.RVEC)).float().cuda(), requires_grad=False)
    sensor, seq, paths, min_past = base.exp.old.base.inputs(role, sequence)
    cache = base.exp.arrays(paths["cache"])
    started = time.time()
    key, _ = base.make_keys(net, cache, arm)
    with np.load(paths["spec"]) as z:
        np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
        bank = base.exp.old.base.bank_for(base.exp.OP, z)
    conf = spec["confidence"]
    ledger = evaluate_keys(key, bank, cache["poses_gt"][:, :3, 3], min_past, {1: tuple(conf["tuple"])})[1]
    keep = ledger["query_idx"] < len(key)-15
    ledger = {k: v[keep] for k, v in ledger.items()}
    refpath = (fixed.OUT/"test/s1"/path.name if role == "test"
               else REPO/chosen["validation_folder"]/path.name)
    reference = base.exp.arrays(refpath)
    for name in ("query_idx", "min_past_gt_dist"):
        np.testing.assert_array_equal(ledger[name], reference[name])
    ledger["group"] = reference["group"]
    q, j = ledger["query_idx"], ledger["j_star"]
    assert ((j >= 0) & (j <= q-30)).all()
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    pieces = [fixed.frozen_scores(bank, travel, q[i:i+256], j[i:i+256], conf["decision"],
              ledger[conf["decision"]][i:i+256]) for i in range(0, len(q), 256)]
    scores = {name: np.concatenate([p[name] for p in pieces]) for name in SCORES}
    replay_error = None
    if role == "val":
        for name in (*base.FIELDS, *SCORES):
            np.testing.assert_array_equal(scores[name] if name in scores else ledger[name], reference[name])
        oldkey = base.exp.arrays(refpath.with_name("keys_"+refpath.name))["embedding"]
        replay_error = float(abs(oldkey-key).max())
        np.testing.assert_allclose(key, oldkey, atol=4e-6, rtol=4e-5)
    prefixes = []
    for index in (len(q)//3, 2*len(q)//3):
        t, end = int(q[index]), int(q[index]+16)
        small, _ = base.make_keys(net, cache, arm, end)
        np.testing.assert_allclose(small[:t+1], key[:t+1], atol=4e-6, rtol=4e-5)
        ids = np.unique(np.r_[base.prefix_topk(small, t, t-29, 100), base.prefix_topk(bank[0], t, t-29, 100)])
        channels = (small[ids]@small[t], bank[0][ids]@bank[0][t],
                    base.align_scores(bank[1], bank[2], t, ids), base.column_align_scores(bank[3], t, ids))
        fused = sum(float(w)*base.minmax(v) for w, v in zip(conf["tuple"], channels))
        winner = int(ids[np.argsort(-fused, kind="stable")[0]])
        assert winner == int(j[index])
        tiny = fixed.frozen_scores(base.exp.truncate_bank(bank, end), travel[:end], q[index:index+1],
                                  j[index:index+1], conf["decision"], ledger[conf["decision"]][index:index+1])
        for name in SCORES:
            np.testing.assert_array_equal(tiny[name], scores[name][index:index+1])
        prefixes.append({"q": t, "key_max_error": float(abs(small[:t+1]-key[:t+1]).max()),
            "winner_equal": True, "score_error": 0.})
    np.savez_compressed(path, **{name: ledger[name] for name in base.FIELDS}, **scores)
    kp, cp = path.with_name("keys_"+path.name), path.with_name("coverage_"+path.name)
    np.savez_compressed(kp, embedding=key, scan_ids=cache["scan_ids"])
    np.savez_compressed(cp, **ledger)
    write_json(path.with_suffix(".json"), {"arm": arm, "role": role, "sequence": sequence,
        "sha256": sha256(path), "keys_sha256": sha256(kp), "coverage_sha256": sha256(cp),
        "freeze_sha256": sha256(OUT/"freeze.json"), "checkpoint_sha256": chosen["checkpoint_sha256"],
        "metrics": {name: base.exp.metrics(value, ledger) for name, value in scores.items()},
        "prefixes": prefixes, "validation_key_max_error": replay_error,
        "validation_ledger_exact": role == "val", "seconds": time.time()-started,
        "test_used": role == "test", "test_used_for_selection": False})
    print("EVALUATED", role, arm, sequence, round(time.time()-started, 2), flush=True)


def macro(rows: dict, cities: dict) -> dict:
    """표 2의 시퀀스→6도시 평균을 계산한다."""
    values = {}
    for city, members in cities.items():
        chosen = [rows[s] for s in members if rows[s]["n_pos"]]
        assert chosen
        values[city] = {m: float(np.mean([r[m] for r in chosen])) for m in base.exp.METRICS}
    return {"cities": values, "macro": {m: float(np.mean([r[m] for r in values.values()])) for m in base.exp.METRICS}}


def report() -> None:
    """동결한 두 조건과 같은 시드 기존 모델의 최종 수치를 보고한다."""
    spec = freeze(True)
    records = {arm: {seq: read(OUT/"test"/arm/(seq.replace("/", "_")+".json"))["metrics"]
                     for seq in spec["test_roster"]} for arm in ARMS}
    aggregate = {arm: {name: macro({s: r[name] for s, r in rows.items()}, spec["cities"])
                       for name in SCORES} for arm, rows in records.items()}
    for arm in ARMS:
        print("FINAL_TEST", arm, aggregate[arm][METHOD]["macro"], flush=True)
    old = read(fixed.OUT/"summary.json")
    references = {"standard_seed1": old["per_seed"]["1"][METHOD],
                  "standard_3seed": old["mean_std"][METHOD], "bevplace_seed1": old["bevplace_seed1"]}
    delta = {name: {m: 100*(aggregate["diff_mask25"][METHOD]["macro"][m]-ref[m]) for m in base.exp.METRICS}
             for name, ref in (("diff_baseline", aggregate["diff_baseline"][METHOD]["macro"]),
                               ("standard_seed1", references["standard_seed1"]["macro"]),
                               ("bevplace_seed1", references["bevplace_seed1"]))}
    city_delta = {city: {m: 100*(aggregate["diff_mask25"][METHOD]["cities"][city][m]
                                -aggregate["diff_baseline"][METHOD]["cities"][city][m]) for m in base.exp.METRICS}
                  for city in spec["cities"]}
    write_json(OUT/"summary.json", {"per_sequence": records, "aggregate": aggregate,
        "references": references, "diff_mask25_minus_reference_pp": delta,
        "city_mask25_minus_baseline_pp": city_delta, "seed": 1,
        "freeze_sha256": sha256(OUT/"freeze.json"), "test_used": True,
        "test_used_for_selection": False, "scope": spec["scope"]})


def run() -> None:
    """검증 재현 후 같은 고정 모델 두 개의 시험 92원장을 생성한다."""
    LOG.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        prepare()
        def invoke(job: tuple) -> None:
            arm, seq, role = job
            path = LOG/f"{role}_{arm}_{seq.replace('/', '_')}.log"
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            with path.open("a") as handle:
                p = subprocess.run([sys.executable, "-u", str(Path(__file__)), "worker", "--arm", arm,
                                    "--sequence", seq, "--role", role], cwd=REPO, env=env,
                                   stdout=handle, stderr=subprocess.STDOUT)
            if p.returncode:
                raise RuntimeError(str(path)+"\n"+path.read_text()[-5000:])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(invoke, [(a, s, "val") for a in ARMS for s in REPLAY]))
        write_json(OUT/"validation_replay.json", {"status": "passed", "arms": list(ARMS),
            "sequences": list(REPLAY), "queries_winners_scores_exact": True})
        spec = freeze()
        jobs = [(a, s, "test") for s in spec["test_roster"] for a in ARMS]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending = {pool.submit(invoke, job): job for job in jobs}
            for done, future in enumerate(concurrent.futures.as_completed(pending), 1):
                future.result()
                write_json(OUT/"status.json", {"status": "running", "stage": "test", "done": done,
                    "total": len(jobs), "elapsed_s": time.time()-started, "test_used": True})
                print("TEST", done, len(jobs), pending[future], flush=True)
        report()
        write_json(OUT/"status.json", {"status": "evaluated_pending_audit", "elapsed_s": time.time()-started, "test_used": True})
    except Exception as error:
        write_json(OUT/"status.json", {"status": "failed", "reason": str(error), "test_used": True})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "worker", "report", "run"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--sequence")
    parser.add_argument("--role", choices=("val", "test"), default="test")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.action == "worker":
        worker(args.arm, args.sequence, args.role)
    else:
        globals()[args.action]()
