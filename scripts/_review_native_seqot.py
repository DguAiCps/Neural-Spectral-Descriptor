#!/usr/bin/env python3
"""공개 SeqOT NCLT 규약을 저자·분할v2 가중치로 별도 재현한다."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105/seqot"
OFFICIAL = REPO / "_external/SeqOT"
SEQUENCES = {"database": "2012-01-08", "query": "2012-02-05"}
sys.path[:0] = [str(OFFICIAL), str(OFFICIAL / "tools")]
# 다른 설치 패키지의 tools가 공식 코드의 namespace directory를 가리지 않게 한다.
official_tools = types.ModuleType("tools")
official_tools.__path__ = [str(OFFICIAL / "tools")]
sys.modules["tools"] = official_tools


def sha(path: Path) -> str:
    """실제 파일의 SHA256을 반환한다."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            for attempt in range(5):
                try:
                    block = f.read(1024 * 1024)
                    break
                except OSError as exc:
                    if exc.errno != 12 or attempt == 4:
                        raise
                    time.sleep(1)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write(path: Path, value: dict) -> None:
    """진행 기록을 원자적으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def protocol() -> dict:
    """공식 공개 평가 쌍·가중치·소스를 결과 전에 고정한다."""
    sources = ["modules/seqTransformerCat.py", "modules/netvlad.py", "modules/gem.py",
               "tools/read_samples.py", "tools/utils/utils.py", "tools/utils/gen_depth_data.py",
               "train/gen_sub_descriptors.py", "test/test_gem_prepare.py", "test/test_seqot.py"]
    author = REPO / "_external/seqot_official_eval"
    models = {"author": {"stage1": str((author / "pretrained_seqot.pth.tar").relative_to(REPO)),
                          "stage2": str((author / "pretrained_gem.pth.tar").relative_to(REPO))}}
    for seed in (1, 2, 3):
        src = REPO / f"results/pdv2_seqot_official_centered{'' if seed == 1 else '_s' + str(seed)}/train_selection.json"
        d = json.loads(src.read_text())
        models[f"v2_s{seed}"] = {s: d[s]["chosen"]["checkpoint"] for s in ("stage1", "stage2")}
        models[f"v2_s{seed}"]["sha256"] = {s: d[s]["chosen"]["sha256"] for s in ("stage1", "stage2")}
    for values in models.values():
        if "sha256" not in values:
            values["sha256"] = {s: sha(REPO / values[s]) for s in ("stage1", "stage2")}
        for stage in ("stage1", "stage2"):
            if (REPO / values[stage]).exists():
                assert sha(REPO / values[stage]) == values["sha256"][stage]
    value = {"official_repo": "https://github.com/BIT-MJY/SeqOT",
             "official_commit": "6e10697a4e6a423b5581e6a6ed6df9cfb386658b",
             "paper": "https://arxiv.org/abs/2209.07951",
             "published_table_II_2012_02_05": {"r1": 0.917, "r5": 0.947, "r20": 0.968},
             "database": SEQUENCES["database"], "query": SEQUENCES["query"],
             "database_stride_raw": 10, "query_stride_raw": 5,
             "sub_window": "[s-1,s+1], incomplete window repeats centre; N+1 row array with final zero",
             "gem_window": "[s-10,s+9], incomplete window repeats centre descriptor",
             "distance": "official FAISS IndexIVFFlat(nlist=1), L2, k=22, no final normalization",
             "ground_truth": str((author / "gt_120108_120205.npy").relative_to(REPO)),
             "ground_truth_sha256": sha(author / "gt_120108_120205.npy"),
             "ground_truth_rule": "저자 공개 정답 배열 그대로; released generator는3D거리<15m(논문서술overlap>0.3와다름); 공식 np.any 조건으로 빈 질의 제외",
             "source_sha256": {s: sha(OFFICIAL / s) for s in sources}, "models": models,
             "selection": "기존 검증 선택본 고정; native 시험 결과로 가중치/설정 선택 안 함",
             "contamination": "저자 모델은 NCLT 2012-01-08 학습. 같은 Ann Arbor 장소 native 진단이며 장소분리 주표에 편입 안 함"}
    p = OUT / "protocol.json"
    if p.exists():
        existing = json.loads(p.read_text())
        if existing["ground_truth_rule"].startswith("저자 공개 overlap>0.3"):
            # 기존 동결 파일은 보존한다. 실행 조건은 같고 설명 오류만 정정했다.
            assert (OUT / "protocol_label_correction.json").exists()
            existing["ground_truth_rule"] = value["ground_truth_rule"]
        assert existing == value, "고정 프로토콜이 변경됨"
    else:
        write(p, value)
    return value


def check() -> dict:
    """입력 및 소스 고정을 확인한다."""
    d = json.loads((OUT / "protocol.json").read_text())
    for p, h in d["source_sha256"].items():
        assert sha(OFFICIAL / p) == h, p
    assert sha(REPO / d["ground_truth"]) == d["ground_truth_sha256"]
    return d


def prepare(role: str) -> None:
    """보유 원규약 거리영상 전체를 손실 없이 묶고 원시3개를 재투영한다."""
    check()
    seq = SEQUENCES[role]
    data = Path("/workspace/data/nclt")
    images = data / "_seqot_official" / seq / "range_image/depth"
    paths = sorted(images.glob("*.npy"))
    assert len(paths) == {"database": 28127, "query": 28239}[role]
    directory = OUT / "inputs"
    directory.mkdir(exist_ok=True)
    done = directory / f"{role}.json"
    if done.exists():
        print(f"입력 완료 재사용: {role}", flush=True)
        return
    target = directory / f"{role}.npy"
    tmp = directory / f"{role}.tmp.npy"
    hashes = []
    start = time.time()
    reused = target.exists()
    if not reused:
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(paths), 32, 900))
        for i, p in enumerate(paths):
            assert p.name == f"{i:06d}.npy"
            a = np.load(p)
            assert a.dtype == np.float32 and a.shape == (32, 900) and np.isfinite(a).all()
            arr[i] = a
            hashes.append(sha(p))
            if i % 1000 == 0:
                print(role, i, len(paths), round(time.time() - start, 1), flush=True)
        arr.flush()
        del arr
        tmp.replace(target)
    packed = np.load(target, mmap_mode="r")
    assert packed.dtype == np.float32 and packed.shape == (len(paths), 32, 900)
    for first in range(0, len(paths), 512):
        assert np.isfinite(packed[first:first + 512]).all()
    spec = importlib.util.spec_from_file_location("official_depth", OFFICIAL / "tools/utils/gen_depth_data.py")
    depth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(depth)
    from tools.utils.utils import load_files, range_projection
    depth.load_files = load_files
    depth.range_projection = range_projection
    raw_root = (data / "_seqot_dl" / seq / "velodyne_sync"
                if role == "query" else data / seq / "velodyne_sync")
    raw = depth.load_files(str(raw_root))
    assert len(raw) == len(paths), (raw_root, len(raw), len(paths))
    checks = []
    for i in [0, len(paths) // 2, len(paths) - 1]:
        points = depth.get_velo(raw[i])[0]
        rebuilt, _, _ = depth.range_projection(points, fov_up=30.67, fov_down=-10.67,
            proj_H=32, proj_W=900, max_range=80, cut_z=False, low=0.1, high=6)
        old = np.load(paths[i])
        assert np.array_equal(old, packed[i]), (role, i, "packed")
        assert np.array_equal(rebuilt, old), (role, i, np.max(np.abs(rebuilt - old)))
        checks.append({"index": i, "raw": raw[i], "raw_sha256": sha(Path(raw[i])),
                       "range_image_sha256": sha(paths[i]), "reprojection_bitwise_equal": True})
    write(done, {"role": role, "sequence": seq, "n_frames": len(paths), "dtype": "float32",
                 "shape": [len(paths), 32, 900], "packed_sha256": sha(target),
                 "individual_sha256": hashes, "raw_reprojection_checks": checks,
                 "reused_atomically_completed_pack": reused,
                 "elapsed_s": time.time() - start})


def load_model(model_name: str, stage: str):
    """고정 모델을 공식 클래스에 엄격하게 읽는다."""
    import torch
    from modules.seqTransformerCat import featureExtracter
    from modules.gem import GeM
    p = check()["models"][model_name]
    path = REPO / p[stage]
    assert sha(path) == p["sha256"][stage]
    model = featureExtracter(seqL=3) if stage == "stage1" else GeM()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.eval().cuda()


def infer(model_name: str, role: str, batch: int, smoke: bool) -> None:
    """공식 가운데 창으로 전 프레임 특징을 재개 가능하게 추론한다."""
    import torch
    torch.set_num_threads(2)
    model = load_model(model_name, "stage1")
    packed = OUT / "inputs" / f"{role}.npy"
    manifest = json.loads((OUT / "inputs" / f"{role}.json").read_text())
    assert sha(packed) == manifest["packed_sha256"], packed
    images = np.load(packed, mmap_mode="r")
    n = len(images)
    root = OUT / model_name / ("smoke" if smoke else "sub") / role
    root.mkdir(parents=True, exist_ok=True)
    if (root / "done.json").exists():
        return
    start = time.time()
    indices = np.unique(np.linspace(0, n - 1, 12, dtype=int)) if smoke else np.arange(n)
    def windows(ix):
        groups = [([j] * 3 if j == 0 or j == n - 1 else [j - 1, j, j + 1]) for j in ix]
        return torch.from_numpy(np.asarray(images[np.asarray(groups)], dtype=np.float32)).cuda()
    with torch.inference_mode():
        # 배치가 계산 내용을 바꾸지 않는지 고정3입력으로 확인한다.
        probes = np.asarray([0, n // 2, n - 1])
        single = torch.cat([model(windows([j])) for j in probes])
        grouped = model(windows(probes))
        error = float((single - grouped).abs().max().item())
        if batch != 1:
            assert error < 1e-5, error
        write(root / "batch_check.json", {"indices": probes.tolist(), "max_abs_error": error,
                                         "batch": batch, "limit": 1e-5,
                                         "policy": "batch1은 저자 원코드; 묶음추론은 수치일치검사 통과할 때만 허용"})
        for first in range(0, len(indices), 512):
            chunk = indices[first:first + 512]
            target = root / f"shard_{first:06d}.npz"
            if target.exists():
                continue
            values = []
            for k in range(0, len(chunk), batch):
                values.append(model(windows(chunk[k:k + batch])).cpu().numpy())
            values = np.concatenate(values)
            assert values.shape == (len(chunk), 256) and np.isfinite(values).all()
            temp = target.with_name(target.stem + ".tmp.npz")
            np.savez_compressed(temp, indices=chunk, descriptors=values)
            temp.replace(target)
            status = {"model": model_name, "role": role, "frames": first + len(chunk),
                      "total": len(indices), "elapsed_s": time.time() - start,
                      "gpu": torch.cuda.get_device_name(), "pid": os.getpid(), "batch": batch}
            write(root / "status.json", status)
            print(json.dumps(status), flush=True)
    write(root / "done.json", status)


def score(model_name: str, use_previous: bool = False) -> None:
    """공식 GeM·FAISS·Recall 코드를 실행하고 별도 정수 집계로 검산한다."""
    import torch
    import faiss
    from tools.read_samples import read_one_need_descriptor_from_seq_ft
    spec = importlib.util.spec_from_file_location("official_recall", OFFICIAL / "test/test_seqot.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    cal_topN = evaluator.cal_topN
    faiss.omp_set_num_threads(2)
    gem = load_model(model_name, "stage2")
    root = OUT / model_name / ("cached_reference_replay" if use_previous else "native")
    root.mkdir(parents=True, exist_ok=True)
    finals = {}
    with torch.inference_mode():
        for role, stride in [("database", 10), ("query", 5)]:
            if use_previous:
                assert model_name == "author"
                sub = np.load(OFFICIAL / f"train/des_list_{role}.npy")
            else:
                directory = OUT / model_name / "sub" / role
                done = json.loads((directory / "done.json").read_text())
                n = done["total"]
                sub = np.zeros((n + 1, 256), np.float32)
                seen = np.zeros(n, bool)
                for p in sorted(directory.glob("shard_*.npz")):
                    z = np.load(p)
                    sub[z["indices"]] = z["descriptors"]
                    seen[z["indices"]] = True
                assert seen.all()
            # 공식 평가 코드의 N+1 배열 및 DB 추가행 초기화를 그대로 유지한다.
            query_ids = np.arange(0, len(sub), stride)
            desc = np.zeros((len(sub) // stride + 1, 256), np.float32) if role == "database" else np.zeros((len(query_ids), 256), np.float32)
            for k, j in enumerate(query_ids):
                inp, _ = read_one_need_descriptor_from_seq_ft(str(j).zfill(6), sub, seq_len=20)
                desc[k] = gem(inp).squeeze(1)[0].cpu().numpy()
            np.save(root / f"{role}_descriptors.npy", desc)
            finals[role] = (desc, query_ids)
        db, _ = finals["database"]
        query, query_ids = finals["query"]
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(256), 256, 1, faiss.METRIC_L2)
        index.train(db)
        index.add(db)
        dist, ids = index.search(query, 22)
    pred = np.stack([np.broadcast_to(query_ids[:, None], ids.shape), ids * 10, dist], axis=-1)
    predfile = root / "predicted_L2_dis.npz"
    np.savez_compressed(predfile, pred.reshape(-1, 1, 3))
    gtfile = REPO / check()["ground_truth"]
    official = {str(k): float(cal_topN(str(predfile), str(gtfile), k)) for k in (1, 5, 20)}
    gt = np.load(gtfile, allow_pickle=True)
    hits = {k: 0 for k in (1, 5, 20)}
    valid = 0
    for j, q in enumerate(range(0, 28239, 5)):
        true = np.asarray(gt[j])
        if not true.any():
            continue
        valid += 1
        for k in hits:
            hits[k] += int(np.isin(ids[j, :k] * 10, true).any())
    independent = {str(k): hits[k] / valid for k in hits}
    assert official == independent, (official, independent)
    # FAISS의 1위가 직접 float64 거리 최소와 일치하는지 고정3질의를 검증한다.
    checks = []
    for q in [0, len(query) // 2, len(query) - 1]:
        d = ((db.astype(np.float64) - query[q].astype(np.float64)) ** 2).sum(axis=1)
        best = int(np.argmin(d))
        assert best == int(ids[q, 0]), (q, best, ids[q, 0])
        checks.append({"query_row": q, "top1_equal": True, "distance_error": float(abs(d[best] - dist[q, 0]))})
    report = {"model": model_name, "scope": "보유 공식서술자 재집계" if use_previous else "새 전체 추론 native 평가",
              "total_queries": len(gt), "positive_queries": valid, "hits": hits,
              "official_recall": official, "independent_recall": independent,
              "direct_l2_checks": checks, "prediction_sha256": sha(predfile),
              "protocol_sha256": sha(OUT / "protocol.json"),
              "published": check()["published_table_II_2012_02_05"]}
    write(root / "report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["protocol", "prepare", "infer", "score"])
    p.add_argument("--role", choices=list(SEQUENCES))
    p.add_argument("--model", default="author", choices=["author", "v2_s1", "v2_s2", "v2_s3"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--use-previous", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "protocol":
        protocol()
    elif a.stage == "prepare":
        prepare(a.role)
    elif a.stage == "infer":
        infer(a.model, a.role, a.batch, a.smoke)
    else:
        score(a.model, a.use_previous)
