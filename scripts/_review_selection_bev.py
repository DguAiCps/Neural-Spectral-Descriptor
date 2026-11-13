#!/usr/bin/env python3
"""남아 있는 BEVPlace++ 가중치에서 검증 AP/R@1 선택 목적만 대조한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]
from _pdv2_pilot import VAL, TEST, CITY
from _pr_eval_lib import classify_queries, pr_sweep

OUT = ROOT / "results/pdv2_review_selection_20261105"
METRICS = ("AP", "max_F1", "R@100P", "R@1")


def sha(path: Path) -> str:
    """파일의 SHA256을 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    """JSON 기록을 읽는다."""
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    """기록을 원자적으로 쓴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def old_val(seed: int, sensor: str, seq: str) -> Path:
    """기존 best의 KISS 검증 원장을 찾는다."""
    if seed == 1:
        return ROOT / f"results/pdv2_bevplace_val_odom_20261018/dump_bevplace_{sensor}_{seq}.npz"
    return ROOT / f"results/pdv2_bevplace_val_odom_seeds_20261101/dump_bevplace_s{seed}_{sensor}_{seq}.npz"


def freeze() -> None:
    """평가 전에 남은 두 후보와 규칙을 고정한다."""
    path = OUT / "protocol.json"
    assert not path.exists(), "이미 동결된 명세를 덮어쓰지 않는다."
    candidates, sources = {}, {}
    for seed in (1, 2, 3):
        suffix = "" if seed == 1 else f"_s{seed}"
        selection_path = ROOT / f"results/pdv2_bevplace_official{suffix}/train_selection.json"
        selection = read(selection_path)
        best = selection["best"]
        last = max(selection["epochs"], key=lambda row: row["epoch"])
        sources[str(selection_path.relative_to(ROOT))] = sha(selection_path)
        candidates[str(seed)] = []
        for tag, epoch in (("best", best["epoch"]), ("final", last["epoch"])):
            original = ROOT / best["checkpoint"]
            if tag == "final":
                original = OUT / "checkpoints" / original.parent.relative_to(ROOT) / "checkpoint.pth.tar"
            fixed = OUT / "checkpoints" / f"s{seed}_{tag}_epoch{epoch}.pth.tar"
            shutil.copy2(original, fixed)
            if tag == "best":
                assert sha(fixed) == best["sha256"]
            row = next(r for r in selection["epochs"] if r["epoch"] == epoch)
            candidates[str(seed)].append(dict(id=tag, epoch_zero_based=epoch,
                completed_epochs=epoch + 1, checkpoint=str(fixed.relative_to(ROOT)),
                sha256=sha(fixed), original_rule_r1=row["family_macro_r1"]))
        for sensor, seq in VAL:
            p = old_val(seed, sensor, seq)
            sources[str(p.relative_to(ROOT))] = sha(p)
    for name in ("_review_selection_bev.py", "_review_selection_prefix_exact.py",
                 "_bevplace_official_dump.py", "_bevplace_official_bev.py", "_pr_eval_lib.py", "_pdv2_pilot.py"):
        p = ROOT / "scripts" / name
        sources[str(p.relative_to(ROOT))] = sha(p)
    write(path, dict(created_unix=time.time(), scope="보존된 best/final 두 후보의 목적 민감도; 전체40에포크 AP 최적 선택 아님",
        candidates=candidates, val=[s + "/" + q for s, q in VAL],
        queries="KISS-ICP 키프레임, 30 <= q < N-15, 99,236질의",
        candidates_rule="j <= q-30", score="기존 공식 8192D 서술자의 -L2; scorer 추가 없음",
        ground_truth="과거 5m 이내 양성, 25m 초과 음성, 그 사이 PR 제외; 잘못된 장소 수락은 FP",
        aggregation="양성 시퀀스 평균 → KITTI/KITTI360, HeLiPR, MulRan 세 계열 균등 평균",
        selection="시드마다 같은 두 후보에서 AP 최대 / 재방문 R@1 최대를 각각 선택",
        tie_break="정확히 같은 값이면 앞 epoch_zero_based, 이어서 id 사전순",
        test_policy="두 목적의 선택을 동결한 뒤 선택된 후보만 시험. 같은 후보이면 동일 원장 해시를 검산하여 기존 시험 재사용",
        limitations=["중간38에포크 가중치와 비재방문 점수가 없다", "기존 best 선택은 GT 키프레임·양방향 |j-q|>30·계열 내 질의 pooled R@1",
                     "같은 두 후보의 KISS AP/R@1 대조만 선택 목적 효과; 기존 선택과의 차이는 규약 변경도 포함",
                     "기존 검증과 시험 결과가 알려진 사후 민감도 분석; 새로운 독립 시험이 아님"],
        source_sha256=sources))


def metrics(arrays: dict) -> dict:
    """공통 지표를 계산한다."""
    cls = classify_queries(arrays["min_past_gt_dist"])
    curve = pr_sweep(arrays["score_negl2"], arrays["top1_geo_dist"], cls)
    pos = cls["positive"]
    result = {key: curve[key] if cls["n_pos"] else None for key in METRICS[:-1]}
    result["R@1"] = float(np.mean(arrays["top1_geo_dist"][pos] <= 5)) if pos.any() else None
    return dict(result, n_pos=cls["n_pos"], n_queries=len(pos))


def evaluate(seed: int, tag: str, sequence: str, role: str) -> None:
    """고정 후보를 공식 추론·공식 CPU L2 점수로 평가한다."""
    import torch
    import _bevplace_official_bev as bev
    from _bevplace_official_dump import _official_model

    spec = read(OUT / "protocol.json")
    candidate = next(r for r in spec["candidates"][str(seed)] if r["id"] == tag)
    checkpoint = ROOT / candidate["checkpoint"]
    assert sha(checkpoint) == candidate["sha256"]
    assert torch.load(checkpoint, map_location="cpu", weights_only=False)["epoch"] == candidate["epoch_zero_based"]
    if role == "test":
        selected = read(OUT / "selection_freeze.json")
        assert tag in [selected["selected"][str(seed)][m]["id"] for m in ("AP", "R@1")]
    sensor, seq = sequence.split("/")
    folder = OUT / role / f"s{seed}_{tag}"
    target = folder / f"{sensor}_{seq}.json"
    if target.exists():
        record = read(target)
        assert sha(ROOT / record["ledger"]) == record["ledger_sha256"]
        return
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    bev_role = "val_odom" if role == "val" else "test_odom"
    bev.ROLES["val_odom"] = (VAL, bev.ODOM, "val_", "poses_gt")
    module, datasets, _ = _official_model(checkpoint, 2, 32, "cuda")
    dataset = datasets.InferDataset(seq=bev.seq_name(sensor, seq), dataset_path=str(bev.BEV / bev_role) + "/")
    cache = bev.cache_path(bev_role, sensor, seq)
    with np.load(cache) as z:
        ids, positions = z["scan_ids"], z["poses_gt"][:, :3, 3].astype(np.float64)
    assert [Path(p).name for p in dataset.imgs_path] == [f"{i:06d}.png" for i in ids]
    tick = time.perf_counter()
    desc = module.infer(dataset).astype(np.float32)
    encode_s = time.perf_counter() - tick
    assert desc.shape == (len(ids), 8192) and np.isfinite(desc).all()
    desc_path, pos_path = folder / "temporary_descriptors.npy", folder / "temporary_positions.npy"
    np.save(desc_path, desc)
    np.save(pos_path, positions)
    search_path = folder / f"search_{sensor}_{seq}.npz"
    env = dict(os.environ, NVIDIA_TF32_OVERRIDE="0")
    tick = time.perf_counter()
    subprocess.run([sys.executable, str(ROOT / "scripts/_review_selection_prefix_exact.py"),
        "--descriptors", str(desc_path), "--positions", str(pos_path), "--out", str(search_path)], env=env, check=True)
    search_s = time.perf_counter() - tick
    with np.load(search_path) as z:
        keep = z["query_idx"] < len(ids) - 15
        arrays = {k: z[k][keep] for k in z.files}
    reference = old_val(1, sensor, seq) if role == "val" else ROOT / f"results/pdv2_bevplace_official/odom/dump_bevplace_{sensor}_{seq}.npz"
    with np.load(reference) as z:
        ix = np.searchsorted(z["query_idx"], arrays["query_idx"])
        for key in ("query_idx", "min_past_gt_dist", "n_candidates"):
            np.testing.assert_array_equal(arrays[key], z[key][ix])
    # 실제 서술자의 첫/중간/끝 질의를 전체 CPU FAISS와 직접 대조한다.
    import faiss
    faiss.omp_set_num_threads(1)
    audit = []
    for row in (0, len(arrays["query_idx"]) // 2, len(arrays["query_idx"]) - 1):
        q = int(arrays["query_idx"][row])
        index = faiss.IndexFlatL2(8192)
        index.add(desc[:q-29])
        distance, winner = index.search(desc[q:q+1], 1)
        score = -float(np.sqrt(max(float(distance[0, 0]), 0.0)))
        assert int(winner[0, 0]) == arrays["j_star"][row]
        assert score == arrays["score_negl2"][row]
        audit.append(dict(query=q, winner=int(winner[0, 0]), score_error=0.0))
    ledger = folder / f"dump_bevplace_{sensor}_{seq}.npz"
    meta = dict(method="bevplace", seed=seed, candidate=candidate, role=role,
                primary_score="score_negl2", protocol_sha256=sha(OUT / "protocol.json"))
    np.savez_compressed(ledger, meta=json.dumps(meta), **arrays)
    replay = None
    if role == "val" and tag == "best":
        with np.load(old_val(seed, sensor, seq)) as z:
            replay = dict(same_winners=bool(np.array_equal(arrays["j_star"], z["j_star"])),
                max_score_error=float(np.max(np.abs(arrays["score_negl2"] - z["score_negl2"]))))
    write(target, dict(sequence=sequence, candidate=candidate, metrics=metrics(arrays),
        encode_s=encode_s, search_s=search_s, torch_peak_gb=torch.cuda.max_memory_allocated()/2**30,
        cache=str(cache.relative_to(ROOT)), cache_sha256=sha(cache), direct_cpu_audit=audit, replay=replay,
        ledger=str(ledger.relative_to(ROOT)), ledger_sha256=sha(ledger)))
    desc_path.unlink()
    pos_path.unlink()
    print("COMPLETE", seed, tag, sequence, encode_s, search_s, flush=True)


def select() -> None:
    """동일 검증 원장에서 두 목적의 선택을 동결한다."""
    assert not (OUT / "selection_freeze.json").exists()
    spec = read(OUT / "protocol.json")
    summary, selected, hashes = {}, {}, {}
    for seed in (1, 2, 3):
        summary[str(seed)] = {}
        for candidate in spec["candidates"][str(seed)]:
            tag, per = candidate["id"], {}
            for sensor, seq in VAL:
                p = old_val(seed, sensor, seq) if tag == "best" else OUT / "val" / f"s{seed}_{tag}" / f"dump_bevplace_{sensor}_{seq}.npz"
                hashes[str(p.relative_to(ROOT))] = sha(p)
                with np.load(p) as z:
                    per[sensor + "/" + seq] = metrics({k: z[k] for k in z.files if k != "meta"})
                with np.load(old_val(1, sensor, seq)) as ref, np.load(p) as z:
                    for k in ("query_idx", "min_past_gt_dist", "n_candidates"):
                        np.testing.assert_array_equal(ref[k], z[k])
            families = {f: {k: float(np.mean([r[k] for n, r in per.items() if n.startswith(f) and r["n_pos"]])) for k in METRICS} for f in ("kitti", "helipr", "mulran")}
            macro = {k: float(np.mean([f[k] for f in families.values()])) for k in METRICS}
            assert sum(r["n_queries"] for r in per.values()) == 99236
            summary[str(seed)][tag] = dict(candidate=candidate, per_sequence=per, families=families, macro=macro)
        selected[str(seed)] = {m: max(spec["candidates"][str(seed)], key=lambda c: (
            summary[str(seed)][c["id"]]["macro"][m], -c["epoch_zero_based"], c["id"])) for m in ("AP", "R@1")}
    write(OUT / "validation_summary.json", summary)
    write(OUT / "selection_freeze.json", dict(created_unix=time.time(), selected=selected,
        protocol_sha256=sha(OUT / "protocol.json"), validation_summary_sha256=sha(OUT / "validation_summary.json"),
        validation_ledger_sha256=hashes, test_used_for_selection=False))
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "evaluate", "select"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--candidate", choices=("best", "final"), default="final")
    parser.add_argument("--sequence")
    parser.add_argument("--role", choices=("val", "test"), default="val")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "select":
        select()
    else:
        evaluate(args.seed, args.candidate, args.sequence, args.role)
