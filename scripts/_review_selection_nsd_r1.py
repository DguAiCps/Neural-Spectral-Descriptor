#!/usr/bin/env python3
"""NSD 네 보존 설정을 현재 최종 반환 장소의 공통 R@1로 직접 선택한다."""
from __future__ import annotations

import argparse
import concurrent.futures
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]
OUT = ROOT / "results/pdv2_review_nsd_r1_20261106"
MAIN = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
VALID = ROOT / "results/pdv2_diffattn_contrast_pipeline_20261104"
SEEDS = ROOT / "results/pdv2_diffattn_contrast_seeds_20261104"
TAGS = ("diff_lr5_4", "diff_lr4_4", "diff_lr5_8", "diff_lr4_8")
METRICS = ("AP", "max_F1", "R@100P", "R@1")


def read(path: Path) -> dict:
    """JSON을 읽는다."""
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    """파일 SHA256을 반환한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    """JSON을 원자적으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    temp.replace(path)


def checkpoint(seed: int, tag: str) -> Path:
    """같은 학습률·완료 epoch를 각 초기 시드에 적용한다."""
    arm, epoch = tag.rsplit("_", 1)
    return SEEDS / f"s{seed}" / arm / f"epoch{epoch}.pth"


def freeze() -> dict:
    """실제 네 후보의 최종 검증값을 계산하기 전에 선택 규칙을 고정한다."""
    path = OUT / "protocol.json"
    if path.exists():
        return read(path)
    old = read(MAIN / "protocol.json")
    candidates = {}
    for tag in TAGS:
        candidates[tag] = {str(seed): dict(path=str(checkpoint(seed, tag).relative_to(ROOT)),
            exists=checkpoint(seed, tag).exists(),
            sha256=sha(checkpoint(seed, tag)) if checkpoint(seed, tag).exists() else None)
            for seed in (1, 2, 3)}
        assert candidates[tag]["1"]["exists"]
    sources = [Path(__file__), MAIN / "protocol.json", SEEDS / "freeze.json",
        ROOT / "scripts/_pdv2_contrast_pipeline.py", ROOT / "scripts/_pdv2_contrast_pipeline_test.py",
        ROOT / "scripts/_pdv2_c3.py", ROOT / "scripts/_bevplace_official_train.py",
        ROOT / "baselines/eval_utils.py", ROOT / "scripts/_pdv2_input_alignment_lib.py",
        ROOT / "src/gnn/contrast_spectral_difference.py"]
    pools = {s: sha(VALID / "val" / s.replace("/", "_") / "candidate_pool.npz") for s in old["val"]}
    spec = dict(created_unix=time.time(), candidates=candidates, validation=old["val"],
        test=old["test"], selection_seed=1, seeds=[1, 2, 3], original_AP_choice="diff_lr4_8",
        primary_objective="현재 matching_only 최종 반환 j_star의 5m 재방문 R@1",
        primary_definition="분모 min_past_gt_dist<=5 양성질의; 분자 반환 top1_geo_dist<5. baseline scorer의 엄격한 <5를 일치시킴",
        primary_aggregation="KITTI+kitti360/HeLiPR/MulRan 각 계열 내 hit수/양성질의수 pooling, 양성이 있는 세 계열 균등 평균",
        tie_order=list(TAGS), tie_rule="정확한 hit/n_positive 분수 평균 동률이면 완료 epoch4 우선, 이어 원래 학습률 실행순서 lr5→lr4",
        seed_propagation="s1에서 고른 학습률/epoch를 s2/3에 그대로 적용. 새 시드에서 재선택 금지. 없는 lr5는 동일 원학습 조건 복원이 필요하며 가용후보로 대체하지 않음",
        auxiliary_AP="현재 최종 AP의 시퀀스→3계열 균등 평균 최고를 별도 보조 선택으로만 기록. 원래 고정승자 판정단계 AP 선정과 다름",
        fixed="기존 KISS 검증9, q=30..N-16, j<=q-30, 원키100+보조100 후보합집합, 두 코사인/두 정렬 융합·계수, 시간판정·q+15·가중치 모두 고정; scorer 추가 없음",
        baseline_protocol_difference="baseline 원래 선택의 GT 키프레임/양방향 검색은 변경하지 않은 NSD 인과 평가와 다르다. 이번에는 목적/5m hit정의/계열 pooling만 일치시키며 전체 검색규약 동일성이라 주장하지 않음",
        CPU_verification="기존 AP선정본의 검증9 전체 승자 정확일치, 점수최대오차8e-6 이하; 지표 두 구현1e-12 이하",
        score_atol=8e-6, device="cpu", candidate_pool_sha256=pools,
        test_policy="공통검증 선택 파일을 동결한 뒤만 시험을 연다. R@1 선택이 달라지면 새 설정과 원래 설정의3시드×46 평가. 같으면 체크포인트SHA 일치와 기존 동결 시험 원장으로 동일성 검산",
        historical_test_exposure=True, test_values_used_for_selection=False,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources})
    write(path, spec)
    return spec


def family(sequence: str) -> str:
    """baseline의 KITTI/KITTI360 계열 통합을 적용한다."""
    sensor = sequence.split("/")[0]
    return "kitti" if sensor.startswith("kitti") else sensor


def pooled(rows: dict) -> tuple:
    """정수 hit를 계열별로 모으고 정확 분수로 선택 동률을 계산한다."""
    groups = {}
    for sequence, row in rows.items():
        g = groups.setdefault(family(sequence), dict(hits=0, positives=0))
        g["hits"] += row["baseline_R1_counts"]["hits"]
        g["positives"] += row["baseline_R1_counts"]["positives"]
    fractions = [Fraction(g["hits"], g["positives"]) for g in groups.values() if g["positives"]]
    return sum(fractions, Fraction()) / len(fractions), groups


def evaluate(role: str, sequence: str) -> dict:
    """원래 후보에서 새 정렬 승자와 최종 점수를 CPU로 실제 계산한다."""
    import numpy as np
    import torch
    import _pdv2_contrast_pipeline_test as original
    from _pdv2_input_alignment_lib import make_graph
    from gnn.contrast_spectral_difference import ContrastSpectralDifference
    from gnn.spectral_transport import attach_transport, lidar_yaw
    from _review_selection_report import independent
    spec = freeze()
    assert sequence in spec["validation" if role == "val" else "test"]
    selection = None if role == "val" else read(OUT / "selection.json")
    if selection:
        assert selection["protocol_sha256"] == sha(OUT / "protocol.json")
        assert selection["R1_selected"] != spec["original_AP_choice"]
    original.activate()
    pipeline, run = original.pipeline, original.run
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    name = sequence.replace("/", "_")
    folder = OUT / role / name
    folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cache, pos, minimum, travel, _, _, _, paths = run.inputs(role, sequence)
    z0 = run.white.coefficients(paths[1])
    seq = np.zeros(len(z0), np.int64)
    graph = make_graph(cache["descriptors"], cache["poses"], seq)
    attach_transport(graph, z0, lidar_yaw(cache["poses"]), seq)
    source = (VALID if role == "val" else MAIN) / role / name
    poolpath = source / "candidate_pool.npz"
    pool = run.ARR(poolpath)
    if role == "val":
        assert sha(poolpath) == spec["candidate_pool_sha256"][sequence]
    else:
        assert sha(poolpath) == read(poolpath.with_suffix(".json"))["sha256"]
    queries = pool["query"]
    np.testing.assert_array_equal(queries, np.arange(30, len(pos)-15))
    weights = run.cal.fusion_weights(read(MAIN / "protocol.json")["confidence"])[run.cal.FUSIONS.index("aux_half")]
    tags = list(TAGS) if role == "val" else list(dict.fromkeys([spec["original_AP_choice"], selection["R1_selected"]]))
    seeds = [1] if role == "val" else [1, 2, 3]
    records = {}
    with torch.no_grad():
        for seed in seeds:
            for tag in tags:
                label = f"{tag}_s{seed}"
                record_path = folder / f"{label}.json"
                if record_path.exists():
                    record = read(record_path)
                    assert sha(ROOT / record["ledger"]) == record["ledger_sha256"]
                    records[label] = record
                    continue
                item = spec["candidates"][tag][str(seed)]
                path = ROOT / item["path"]
                assert path.exists() and sha(path) == item["sha256"], path
                tick = time.time()
                net = ContrastSpectralDifference(run.source.factory(288), "diff")
                net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
                net.eval()
                z = net(graph)
                changed = run.base.variant_inputs(z.numpy(), 8)
                winners = pipeline.winners(pool, changed, weights)
                assert (winners <= queries-30).all()
                temporal = pipeline.e.prior.base.prior.plan(travel, queries, winners)
                scores = []
                for start in range(0, len(queries), 128):
                    pp = {k: torch.from_numpy(v[start:start+128]) for k, v in temporal.items()}
                    scores.append(pipeline.e.prior.score(z, pp).numpy())
                reference = run.ARR(source / f"diff_s{seed}.npz")
                row = dict(query_idx=queries, j_star=winners, min_past_gt_dist=minimum[queries],
                    top1_geo_dist=np.linalg.norm(pos[queries]-pos[winners], axis=1),
                    score=np.concatenate(scores).astype(np.float64), group=reference["group"])
                for field in ("query_idx", "min_past_gt_dist"):
                    np.testing.assert_array_equal(row[field], reference[field])
                actual = run.base.independent_metrics(row, row["score"])
                separate = independent(dict(row, score_negl2=row["score"]))
                if not (row["min_past_gt_dist"] <= 5).any():
                    # 원 함수의 빈 양성 AP=0 표기를 보고용 null로 통일한다.
                    # 사전 동결한 양성 없는 시퀀스 제외와 분모0 규칙은 같다.
                    actual = {key: None for key in METRICS}
                errors = [abs(actual[k]-separate[k]) for k in METRICS if actual[k] is not None]
                assert max(errors, default=0) < 1e-12
                reproduction = None
                if tag == spec["original_AP_choice"]:
                    np.testing.assert_array_equal(winners, reference["j_star"])
                    error = float(np.max(abs(row["score"]-reference["score"])))
                    assert error <= spec["score_atol"], (sequence, seed, error)
                    reproduction = dict(winners_exact=True, max_score_error=error)
                positive = row["min_past_gt_dist"] <= 5
                counts = dict(hits=int(((row["top1_geo_dist"] < 5) & positive).sum()),
                    positives=int(positive.sum()), boundary_equal_5=int((row["top1_geo_dist"][positive] == 5).sum()))
                target = folder / f"{label}.npz"
                np.savez_compressed(target, **row)
                record = dict(tag=tag, seed=seed, sequence=sequence, metrics=actual,
                    baseline_R1_counts=counts, checkpoint_sha256=item["sha256"],
                    ledger=str(target.relative_to(ROOT)), ledger_sha256=sha(target),
                    candidate_pool_sha256=sha(poolpath), query_count=len(queries),
                    independent_metric_max_error=max(errors, default=0), original_reproduction=reproduction,
                    input_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths}, seconds=time.time()-tick)
                write(record_path, record)
                records[label] = record
                print("DONE", role, sequence, label, len(queries), record["seconds"], flush=True)
                write(folder / "progress.json", dict(updated_unix=time.time(), completed=list(records)))
                del net, z, changed
    result = dict(status="complete", records=records, sequence=sequence, seconds=time.time()-started)
    write(folder / "summary.json", result)
    return result


def finish_validation() -> dict:
    """시험을 읽지 않고 R@1 선택과 보조 최종 AP 선택을 확정한다."""
    import numpy as np
    from _pdv2_c3 import _family_pooled
    spec = freeze()
    rows, values, hashes = {}, {}, {}
    for tag in TAGS:
        per = {}
        for sequence in spec["validation"]:
            record_path = OUT / "val" / sequence.replace("/", "_") / f"{tag}_s1.json"
            record = read(record_path)
            assert sha(ROOT / record["ledger"]) == record["ledger_sha256"]
            per[sequence] = record
            hashes[str(record_path.relative_to(ROOT))] = sha(record_path)
        assert sum(r["query_count"] for r in per.values()) == 99236
        exact, groups = pooled(per)
        check = _family_pooled({s: dict(r1=r["baseline_R1_counts"]["hits"]/r["baseline_R1_counts"]["positives"]
            if r["baseline_R1_counts"]["positives"] else 0, n_queries=r["baseline_R1_counts"]["positives"])
            for s, r in per.items()})
        assert abs(float(exact)-check) < 1e-12
        families = {f: {m: float(np.mean([r["metrics"][m] for s, r in per.items()
            if family(s)==f and r["baseline_R1_counts"]["positives"]])) for m in METRICS}
            for f in ("kitti", "helipr", "mulran")}
        macro = {m: float(np.mean([r[m] for r in families.values()])) for m in METRICS}
        rows[tag] = dict(baseline_family_pooled_R1=float(exact), exact_fraction=str(exact),
            groups=groups, current_final_family_sequence_macro=macro, per_sequence=per)
        values[tag] = exact
    chosen = max(TAGS, key=lambda tag: values[tag])
    auxiliary = max(TAGS, key=lambda tag: rows[tag]["current_final_family_sequence_macro"]["AP"])
    write(OUT / "validation_summary.json", rows)
    frozen = dict(created_unix=time.time(), R1_selected=chosen, original_AP_selected=spec["original_AP_choice"],
        auxiliary_current_final_AP_selected=auxiliary, same_as_original=chosen==spec["original_AP_choice"],
        protocol_sha256=sha(OUT / "protocol.json"), validation_summary_sha256=sha(OUT / "validation_summary.json"),
        validation_record_sha256=hashes, test_used_for_selection=False,
        seed_propagation=spec["seed_propagation"], selected_availability=spec["candidates"][chosen])
    write(OUT / "selection.json", frozen)
    return frozen


def campaign(workers: int) -> None:
    """중복 없는 CPU 작업열을 로그와 상태에 남기며 검증을 끝낸다."""
    import fcntl
    spec = freeze()
    lock = (OUT / "campaign.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    done = []
    def launch(sequence: str) -> str:
        with (logs / (sequence.replace("/", "_")+".log")).open("a") as log:
            subprocess.run([sys.executable, "-u", str(Path(__file__)), "worker", "--role", "val",
                "--sequence", sequence], stdout=log, stderr=subprocess.STDOUT, check=True)
        return sequence
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(launch, sequence) for sequence in spec["validation"]]
        write(OUT / "status.json", dict(status="validation_running", pid=os.getpid(), workers=workers,
            completed=done, total=9, updated_unix=time.time()))
        for future in concurrent.futures.as_completed(futures):
            done.append(future.result())
            write(OUT / "status.json", dict(status="validation_running", pid=os.getpid(), workers=workers,
                completed=done, total=9, updated_unix=time.time()))
    frozen = finish_validation()
    write(OUT / "status.json", dict(status="validation_frozen", pid=os.getpid(), completed=done,
        selection_sha256=sha(OUT / "selection.json"), selected=frozen["R1_selected"], updated_unix=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "worker", "validation", "finish_validation"))
    parser.add_argument("--role", choices=("val", "test"), default="val")
    parser.add_argument("--sequence", default="kitti/09")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.action == "freeze": freeze()
    elif args.action == "worker": evaluate(args.role, args.sequence)
    elif args.action == "finish_validation": finish_validation()
    else: campaign(args.workers)
