#!/usr/bin/env python3
"""ImLPR 저자 가공 입력으로 원 논문 표7(MulRan DCC01-03)을 공식 평가 함수로 재현·비교한다.

서버에서 실행한다. `chain`은 서버 호스트 python3(표준 라이브러리만)로 돌며 나머지 단계를
ring_baseline 컨테이너에서 차례로 부르고, GPU 추론 구간만 호스트·컨테이너의
/tmp/nsd_native_reproduction_gpu0.lock을 flock EX로 잡는다.

    python3 scripts/_review_imlpr_author_eval.py protocol          # 자료 확인 전 조건 고정
    python3 scripts/_review_imlpr_author_eval.py chain --attempt 01
    docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
        python3 scripts/_review_imlpr_author_eval.py {inspect,prepare,infer,score,compare,report} --attempt 01

결과: results/pdv2_reproduction_orchestration_20261106/imlpr/run/attempt_<N>/
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REL_OUT = "results/pdv2_reproduction_orchestration_20261106/imlpr"
OUT = REPO / REL_OUT
RUN = OUT / "run"
OFFICIAL = REPO / "_external/ImLPR"
NATIVE = REPO / "results/pdv2_review_native_20261105/imlpr"
CONTAINER_REPO = "/workspace/Neural-Spectral-Codec"
LOCK = "/tmp/nsd_native_reproduction_gpu0.lock"
SEQUENCES = ["DCC01", "DCC02", "DCC03"]
SEQ_RE = re.compile(r"(?i)dcc[_-]?0?([1-3])(?![0-9])")
PUBLISHED = {"DCC01": {"r1": .909, "f1": .965}, "DCC02": {"r1": .861, "f1": .945}, "DCC03": {"r1": .825, "f1": .918}}
PROTOCOL = {
    "purpose": "저자 공개 가공 입력(data.tar.gz)의 MulRan DCC01-03이 있으면 저자 공개 가중치와 공식 평가 함수로 표7 ImLPR(D)를 재현한다",
    "frozen_before_content_inspection": True,
    "paper": {"source": "https://proceedings.mlr.press/v305/jung25c.html", "table": "Table7 ImLPR (D)",
              "published": PUBLISHED, "published_average": {"r1": .865, "f1": .943}, "total_scans_D2": 4328,
              "normal_k_appendixA": 25, "resize": "1022x64 → 1022x126 (E.1)"},
    "archive": "https://drive.google.com/file/d/1ipKCykBp0MU6O15QmEkBwbbsLDz-jsfD/view",
    "model": {"path": "_external/ImLPR/weights/ImLPR_default.pth",
              "sha256": "1f70df7774f0cc5fef5261371abe266b84757db9e29c900f65c534cac7528df5"},
    "official_eval_sha256": {"_external/ImLPR/eval/evaluate.py": "6076431330eff727138ca4f2d9020995c76f1991262d84c515522722ee55b5c4"},
    "input_source_priority": [
        "A: 저자 평가 pickle의 집합이 한 DCC 시퀀스만 참조하고 그 영상이 모두 있으면 그대로 사용(질의 pickle의 양성, db pickle의 시각)",
        "B: A가 없고 같은 폴더 poses.txt 행 수와 000000부터 연속인 npy 수가 같으면 공식 generate_test_sets_img의 함수로 집합 생성",
        "없음: 영상 또는 정답이 없으면 평가하지 않고 포함 목록과 근거만 남긴다",
        "A와 B가 모두 가능하면 A를 주 평가로, B는 정답 일치 검사로만 쓴다"],
    "inference": "공식 eval.evaluate._get_latent_vectors, RESIZE_TO_HELIPR=True(공식 주석의 MulRan 설정), 배치1, GPU0",
    "metric": ("공식 _process_query 같은 세션(m=n): 시작90초 제외, 60초 이전 후보만, 상위200 뒤 과거 필터, "
               "R@1·R@5·최대F1(공식 epsilon, 임계값 linspace(0,1,1000))·OPR·AUC; 3시퀀스 산술 평균"),
    "cross_checks": ["공식 _get_recall_for_pair(n_jobs=1) 출력과 대조", "주 조건은 float64 전수 L2 numpy 독립 재계산",
                     "로컬 감시기에서 저장 예측으로 numpy 재집계"],
    "comparison_to_regenerated": {
        "regenerated": "results/pdv2_review_native_20261105/imlpr (k15 공개코드, k25 논문App.A; 4275장)",
        "frame_matching": "시각 상호 최근접, |dt|<=50ms",
        "image": "공식 모델 입력(28열 원통 덧댐+126행 크기변경) 기준 채널별 차이, 같은 크기면 원 배열도 비교; 법선 채널이 k15/k25 중 어느 쪽에 가까운지 집계(진단, 선택 아님)",
        "labels": "평가 시 60초 필터 뒤 유효 양성 집합을 공통 영상 공간에서 비교",
        "decomposition": {"C1": "저자 영상+저자 정답 전체(주 평가)", "C2": "C1을 공통 영상으로 제한",
                          "C3_k": "재생성 영상(k)+저자 정답, 공통 영상", "C4_k": "재생성 영상(k)+재생성 정답, 공통 영상",
                          "C5_k": "재생성 전체(기존 결과)", "effects": "표본 C1-C2, 영상 C2-C3, 정답 C3-C4, 재생성 표본 C4-C5"}},
    "selection": "시험 결과로 조건·가중치·표본·정답을 고르지 않는다. 사전 지정 조건을 모두 보고한다",
    "scope": "원 논문 대표 평가의 공개 입력 재현 진단. 분할 v2 주표와 우리 재학습 결과에 섞지 않는다. 저자 학습장소(HeLiPR DCC04-06)와 MulRan DCC는 장소가 겹친다",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    tmp.replace(path)


def read(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def dirs(attempt: str) -> tuple[Path, Path]:
    return OUT / "author_data" / f"attempt_{attempt}", RUN / f"attempt_{attempt}"


# ---------------------------------------------------------------- protocol
def stage_protocol() -> dict:
    p = RUN / "protocol.json"
    if p.exists():
        assert read(p)["protocol"] == PROTOCOL, "protocol changed after freezing"
        return read(p)
    assert sha(REPO / PROTOCOL["model"]["path"]) == PROTOCOL["model"]["sha256"]
    for rel, h in PROTOCOL["official_eval_sha256"].items():
        assert sha(REPO / rel) == h, rel
    value = {"protocol": PROTOCOL, "frozen_unix": time.time(), "frozen_by_script_sha256": sha(Path(__file__))}
    write(p, value)
    return value


# ---------------------------------------------------------------- safe pickle
SAFE_GLOBALS = {("builtins", n) for n in ("set", "frozenset", "list", "dict", "tuple", "int", "float", "str",
                                            "bytes", "bytearray", "bool", "complex", "slice", "range")}
SAFE_GLOBALS |= {("collections", "OrderedDict"), ("collections", "defaultdict"), ("_codecs", "encode"),
                 ("numpy", "ndarray"), ("numpy", "dtype")}
SAFE_GLOBALS |= {(m, n) for m in ("numpy.core.multiarray", "numpy._core.multiarray") for n in ("_reconstruct", "scalar")}
SAFE_GLOBALS |= {(m, "_frombuffer") for m in ("numpy.core.numeric", "numpy._core.numeric")}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in SAFE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def safe_load(path: Path):
    with Path(path).open("rb") as f:
        return SafeUnpickler(f).load()


def is_set(obj) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    first = next(iter(obj.values()))
    return isinstance(first, dict) and "query" in first


def iter_sets(obj):
    if is_set(obj):
        yield None, obj
    elif isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            if is_set(x):
                yield i, x


def seq_of(rel: str) -> str | None:
    found = {f"DCC0{m.group(1)}" for part in str(rel).split("/") for m in [SEQ_RE.search(part)] if m}
    return found.pop() if len(found) == 1 else None


def describe_set(s: dict) -> dict:
    vals = list(s.values())
    seqs = {}
    for v in vals:
        k = seq_of(v["query"])
        seqs[str(k)] = seqs.get(str(k), 0) + 1
    ts = [float(v["timestamp"]) for v in vals if "timestamp" in v]
    int_keys = sorted({k for v in vals for k in v if isinstance(k, int)})
    return {"n": len(vals), "sequences": seqs, "first_query": vals[0]["query"], "last_query": vals[-1]["query"],
            "keys_sample": [str(k) for k in vals[0]][:12], "positive_keys": int_keys[:10],
            "timestamp_min": min(ts) if ts else None, "timestamp_max": max(ts) if ts else None}


# ---------------------------------------------------------------- inspect
def stage_inspect(attempt: str) -> dict:
    auth, ar = dirs(attempt)
    res = read(auth / "extract_result.json")
    assert res and res["passed"], "extraction not passed"
    rows = [json.loads(x) for x in (auth / "manifest.jsonl").open()]
    bad = []
    for r in rows:
        if r.get("saved"):
            p = auth / r["saved"]
            if not p.is_file() or p.stat().st_size != r["size"] or sha(p) != r["sha256"]:
                bad.append(r["name"])
    tree = {}
    for r in rows:
        if r["type"] != "reg" or r.get("blocked"):
            continue
        parts = r["name"].split("/")
        key = "/".join(parts[:3]) if len(parts) > 3 else "/".join(parts[:-1])
        t = tree.setdefault(key, {"files": 0, "bytes": 0, "suffixes": {}, "npy": {}})
        t["files"] += 1
        t["bytes"] += r["size"]
        suf = os.path.splitext(r["name"])[1].lower()
        t["suffixes"][suf] = t["suffixes"].get(suf, 0) + 1
        if r.get("npy") and "shape" in r["npy"]:
            k = f"{r['npy']['descr']}|{r['npy']['shape']}"
            t["npy"][k] = t["npy"].get(k, 0) + 1
    mulran = [r for r in rows if r["type"] == "reg" and r.get("class") == "mulran_data"]
    seq_dirs = {}
    for r in mulran:
        s = seq_of(os.path.dirname(r["name"])) or seq_of(r["name"])
        d = seq_dirs.setdefault(str(s), {})
        dd = d.setdefault(os.path.dirname(r["name"]), {"n": 0, "saved": 0, "npy": {}})
        dd["n"] += 1
        dd["saved"] += bool(r.get("saved"))
        if r.get("npy") and "shape" in r["npy"]:
            k = f"{r['npy']['descr']}|{r['npy']['shape']}"
            dd["npy"][k] = dd["npy"].get(k, 0) + 1
    pickles = []
    for r in rows:
        if r.get("saved") and r["name"].lower().endswith((".pickle", ".pkl")):
            item = {"name": r["name"], "size": r["size"], "sha256": r["sha256"]}
            try:
                obj = safe_load(auth / r["saved"])
                item["type"] = type(obj).__name__
                item["len"] = len(obj) if hasattr(obj, "__len__") else None
                item["sets"] = [{"index": i, **describe_set(s)} for i, s in iter_sets(obj)]
            except Exception as exc:
                item["load_error"] = repr(exc)[:300]
            pickles.append(item)
    poses = [{"name": r["name"], "size": r["size"], "sha256": r["sha256"], "sequence": seq_of(r["name"])}
             for r in rows if r.get("saved") and os.path.basename(r["name"]).lower() == "poses.txt"]
    validation_dirs = sorted({k for k in tree if "validation" in k.split("/")})
    mulran_named = sorted({r["name"] for r in rows if "mulran" in r["name"].lower()})[:200]
    inv = {"attempt": attempt, "extract_result_sha256": sha(auth / "extract_result.json"),
           "manifest_sha256": sha(auth / "manifest.jsonl"), "n_manifest_rows": len(rows),
           "saved_file_integrity_failures": bad, "tree": tree, "validation_dirs": validation_dirs,
           "mulran_sequence_dirs": seq_dirs, "mulran_named_entries": mulran_named, "pickles": pickles,
           "poses_files": poses, "blocked": res.get("n_blocked"),
           "mulran_inputs_present": bool(mulran) and all(s in seq_dirs for s in SEQUENCES),
           "any_mulran_data": bool(mulran),
           "mulran_labels_present": any(any(st["sequences"].get(s, 0) for s in SEQUENCES for st in p.get("sets", []))
                                        for p in pickles) or any(p["sequence"] for p in poses)}
    assert not bad, f"saved files changed: {bad[:5]}"
    write(ar / "inventory.json", inv)
    return inv


# ---------------------------------------------------------------- prepare
def saved_index(attempt: str) -> dict:
    auth, _ = dirs(attempt)
    out = {}
    for line in (auth / "manifest.jsonl").open():
        r = json.loads(line)
        if r.get("saved"):
            out[r["name"]] = auth / r["saved"]
    return out


def resolve_folder(rels: list[str], saved: dict) -> tuple[Path | None, int]:
    """평가 집합의 상대경로가 모두 같은 기준 폴더 아래 있는지 찾는다."""
    for name, p in saved.items():
        if name.endswith("/" + rels[0]) or name == rels[0]:
            base = str(p)[: -len(rels[0])].rstrip("/")
            missing = sum(not os.path.isfile(os.path.join(base, r)) for r in rels)
            return Path(base), missing
    return None, len(rels)


def stage_prepare(attempt: str) -> dict:
    auth, ar = dirs(attempt)
    inv = read(ar / "inventory.json")
    saved = saved_index(attempt)
    plan = {}
    cand = {s: [] for s in SEQUENCES}
    for p in inv["pickles"]:
        if "sets" not in p:
            continue
        obj = safe_load(saved[p["name"]])
        for idx, st in iter_sets(obj):
            seqs = {seq_of(v["query"]) for v in st.values()}
            if len(seqs) == 1 and (s := seqs.pop()) in cand:
                kind = "query" if "query" in os.path.basename(p["name"]).lower() else (
                    "db" if re.search(r"(^|[^a-z])(db|database)", os.path.basename(p["name"]).lower()) else "unknown")
                cand[s].append({"pickle": p["name"], "index": idx, "kind": kind, "set": st})
    gen = None
    poses_by_seq = {pp["sequence"]: pp for pp in inv["poses_files"] if pp["sequence"] in SEQUENCES}
    if poses_by_seq:
        gen = official_generator(attempt, poses_by_seq, saved)
    for s in SEQUENCES:
        entry = {"sequence": s}
        qs = [c for c in cand[s] if c["kind"] == "query"] or [c for c in cand[s] if c["kind"] == "unknown"] or cand[s]
        ds = [c for c in cand[s] if c["kind"] == "db"] or qs
        source = None
        if qs:
            q, d = qs[0], ds[0]
            rels = [v["query"] for v in d["set"].values()]
            base, missing = resolve_folder(rels, saved)
            entry["A"] = {"query_pickle": q["pickle"], "query_index": q["index"], "db_pickle": d["pickle"],
                          "db_index": d["index"], "n": len(rels), "image_base": str(base) if base else None,
                          "missing_images": missing, "same_query_order":
                          [v["query"] for v in q["set"].values()] == rels, "alternatives": len(cand[s])}
            if base is not None and missing == 0:
                source = "A"
                m = q["index"] if q["index"] is not None else 0
                payload = {"db": d["set"], "query": q["set"], "m": m, "image_base": str(base), "source": "A"}
        if gen and s in gen["sets"]:
            entry["B"] = {k: v for k, v in gen["sets"][s].items() if k not in ("db", "query")}
            if source is None and gen["sets"][s]["missing_images"] == 0:
                source = "B"
                g = gen["sets"][s]
                payload = {"db": g["db"], "query": g["query"], "m": g["m"], "image_base": g["image_base"], "source": "B"}
        entry["source"] = source
        if source:
            (ar / "sets").mkdir(parents=True, exist_ok=True)
            with (ar / "sets" / f"{s}.pickle").open("wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            entry["n_frames"] = len(payload["db"])
            if source == "A" and "B" in entry:
                entry["A_vs_B_labels"] = compare_label_sets(payload, gen["sets"][s])
        plan[s] = entry
    out = {"plan": plan, "generator": {k: v for k, v in (gen or {}).items() if k != "sets"},
           "ready": all(plan[s]["source"] for s in SEQUENCES)}
    write(ar / "prepare.json", out)
    return out


def official_generator(attempt: str, poses_by_seq: dict, saved: dict) -> dict:
    """공식 construct_query_and_database_sets를 그대로 불러 저자 poses.txt에서 집합을 만든다."""
    _, ar = dirs(attempt)
    folders, bases, rows = [], {}, {}
    for s in SEQUENCES:
        if s not in poses_by_seq:
            continue
        pp = saved[poses_by_seq[s]["name"]]
        n_rows = sum(1 for line in pp.open() if line.strip())
        npys = sorted(x.name for x in pp.parent.glob("*.npy"))
        contiguous = npys == [f"{i:06d}.npy" for i in range(len(npys))]
        rows[s] = {"poses": poses_by_seq[s]["name"], "rows": n_rows, "npy": len(npys), "contiguous": contiguous}
        if contiguous and n_rows == len(npys):
            folders.append(pp.parent.name)
            bases[s] = pp.parent.parent
    result = {"inputs": rows, "folders": folders, "sets": {}}
    if not folders or len({str(b) for b in bases.values()}) != 1:
        result["status"] = "not_applicable"
        return result
    base = next(iter(bases.values()))
    work = ar / "official_gen" / "work"
    work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL / "preprocess"))
    import generate_test_sets_img as G
    cwd = os.getcwd()
    os.chdir(work)
    try:
        G.construct_query_and_database_sets(str(base) + "/", "", folders, "poses.txt")
    finally:
        os.chdir(cwd)
    db = pickle.load((ar / "official_gen/data/db_img.pickle").open("rb"))
    q = pickle.load((ar / "official_gen/data/query_img.pickle").open("rb"))
    result["status"] = "generated"
    result["generator_sha256"] = sha(OFFICIAL / "preprocess/generate_test_sets_img.py")
    for i, folder in enumerate(folders):
        s = seq_of(folder)
        rels = [v["query"] for v in db[i].values()]
        missing = sum(not (base / r).is_file() for r in rels)
        result["sets"][s] = {"db": db[i], "query": q[i], "m": i, "image_base": str(base), "n": len(rels),
                             "missing_images": missing, "folder": folder}
    return result


def compare_label_sets(a: dict, b: dict) -> dict:
    """평가 함수가 실제로 쓰는 60초 필터 뒤 양성 목록을 비교한다."""
    qa, qb = list(a["query"].values()), list(b["query"].values())
    if [v["query"] for v in qa] != [v["query"] for v in qb]:
        return {"same_order": False, "na": len(qa), "nb": len(qb)}

    def eff(q, db, m):
        ts = [float(d["timestamp"]) / 1e9 for d in db.values()]
        return [sorted(j for j in x.get(m, []) if ts[j] < float(x["timestamp"]) / 1e9 - 60) for x in q]

    ea, eb = eff(qa, a["db"], a["m"]), eff(qb, b["db"], b["m"])
    raw = sum(sorted(x.get(a["m"], [])) == sorted(y.get(b["m"], [])) for x, y in zip(qa, qb))
    return {"same_order": True, "n": len(qa), "equal_effective_positive_lists": sum(x == y for x, y in zip(ea, eb)),
            "equal_raw_positive_lists": raw}


# ---------------------------------------------------------------- infer
def load_payload(ar: Path, s: str) -> dict:
    with (ar / "sets" / f"{s}.pickle").open("rb") as f:
        return pickle.load(f)


def dataset_folder(ar: Path, image_base: str) -> Path:
    """공식 로더의 dataset_folder/validation/<rel> 규약을 위해 기준 폴더를 링크한다."""
    root = ar / "dataset" / hashlib.sha256(image_base.encode()).hexdigest()[:12]
    link = root / "validation"
    root.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(image_base, target_is_directory=True)
    assert os.path.realpath(link) == os.path.realpath(image_base)
    return root


def stage_infer(attempt: str) -> dict:
    import numpy as np
    import torch
    _, ar = dirs(attempt)
    stage_protocol()
    model_path = REPO / PROTOCOL["model"]["path"]
    assert sha(model_path) == PROTOCOL["model"]["sha256"]
    assert torch.cuda.device_count() == 1, "CUDA_VISIBLE_DEVICES=0 expected"
    torch.hub.set_dir(str(REPO / "_external/imlpr_official_pdv2/torch_hub/hub"))
    torch.set_num_threads(2)
    sys.path[:0] = [str(REPO / "scripts"), str(REPO), str(REPO / "src")]
    from _imlpr_desc import load_model
    model = load_model(model_path, "cuda")
    import eval.evaluate as E
    E.RESIZE_TO_HELIPR = True
    prep = read(ar / "prepare.json")
    summary = {"gpu": torch.cuda.get_device_name(0), "pid": os.getpid(), "lock": LOCK, "sequences": {}}
    for s in SEQUENCES:
        if not prep["plan"][s]["source"]:
            continue
        pay = load_payload(ar, s)
        folder = dataset_folder(ar, pay["image_base"])
        keys = list(pay["db"].keys())
        for k in keys:
            assert (folder / "validation" / pay["db"][k]["query"]).is_file()
        root = ar / "desc" / s
        root.mkdir(parents=True, exist_ok=True)
        start = time.time()
        if not (root / "batch_check.json").exists():
            probe = [keys[i] for i in sorted({0, len(keys) // 2, len(keys) - 1})]
            sub = {k: pay["db"][k] for k in probe}
            one = E._get_latent_vectors(model, sub, "cuda", str(folder), batch_size=1)
            three = E._get_latent_vectors(model, sub, "cuda", str(folder), batch_size=3)
            write(root / "batch_check.json", {"probe": [str(k) for k in probe], "max_abs_error": float(np.abs(one - three).max()),
                                              "used_inference_batch": 1, "policy": "진단만. 평가는 배치1"})
        for first in range(0, len(keys), 256):
            target = root / f"shard_{first:06d}.npz"
            if target.exists():
                continue
            sub = {k: pay["db"][k] for k in keys[first:first + 256]}
            desc = E._get_latent_vectors(model, sub, "cuda", str(folder), batch_size=1)
            assert desc.shape == (len(sub), 8448) and np.isfinite(desc).all()
            tmp = target.with_name(target.stem + ".tmp.npz")
            np.savez_compressed(tmp, positions=np.arange(first, first + len(sub)), desc=desc)
            tmp.replace(target)
            write(root / "status.json", {"frames": first + len(sub), "total": len(keys), "elapsed_s": time.time() - start})
        write(root / "done.json", {"total": len(keys), "elapsed_s": time.time() - start})
        summary["sequences"][s] = {"frames": len(keys), "elapsed_s": time.time() - start}
    write(ar / "infer.json", summary)
    return summary


# ---------------------------------------------------------------- score
def load_desc(root: Path, n: int, key: str = "positions"):
    import numpy as np
    desc = np.zeros((n, 8448), np.float32)
    seen = np.zeros(n, bool)
    for p in sorted(root.glob("shard_*.npz")):
        z = np.load(p)
        desc[z[key]] = z["desc"]
        seen[z[key]] = True
    assert seen.all() and np.isfinite(desc).all(), root
    return desc


def official_score(desc_db, desc_q, db_meta: list, q_meta: list, m: int, independent: bool = False,
                   official_pair: bool = True) -> tuple[dict, dict]:
    """공식 _process_query로 질의별 결과를 얻고 공식 집계식을 그대로 적용한다."""
    import numpy as np
    from sklearn.neighbors import KDTree
    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))
    import eval.evaluate as E
    tree = KDTree(desc_db)
    kq = max(200, 30, max(int(round(len(desc_db) / 1000.0)), 1))
    distances, indices = tree.query(desc_q, k=kq)
    for probe in sorted({0, len(desc_q) // 2, len(desc_q) - 1}):
        dd, ii = tree.query(desc_q[probe:probe + 1], k=kq)
        assert np.array_equal(ii[0], indices[probe]) and np.array_equal(dd[0], distances[probe])

    class Cached:
        def query(self, x, k):
            assert k == kq
            return distances[cur:cur + 1], indices[cur:cur + 1]

    thresholds = np.linspace(0, 1, 1000)
    threshold_k = max(int(round(len(desc_db) / 1000.0)), 1)
    db_ts = np.array([float(d["timestamp"]) / 1e9 for d in db_meta])
    init_time = float(min(d["timestamp"] for d in db_meta)) / 1e9
    nq = len(desc_q)
    first_index = np.full(nq, -1)
    first_distance = np.full(nq, np.nan)
    first_pos = np.zeros(nq, np.int8)
    evaluated = np.zeros(nq, bool)
    recall_sum, opr_total, num_eval = np.zeros(30), 0, 0
    cells = np.zeros((4, 1000))
    for cur in range(nq):
        r = E._process_query(cur, desc_q[cur], q_meta[cur], Cached(), 30, threshold_k, desc_db, thresholds, m,
                             database_meta=db_meta, is_intra=True, init_time=init_time)
        if r is None:
            continue
        rec, one, tp, fp, tn, fn, c = r
        qt = float(q_meta[cur]["timestamp"]) / 1e9
        allowed = [(i, d) for i, d in zip(indices[cur], distances[cur]) if db_ts[i] < qt - 60]
        fi, fd = allowed[0]
        pos = [i for i in q_meta[cur].get(m, []) if db_ts[i] < qt - 60]
        assert int(fi in pos) == int(rec[0])
        assert np.array_equal(tp, (fd < thresholds) & (fi in pos))
        first_index[cur], first_distance[cur], first_pos[cur], evaluated[cur] = fi, fd, int(rec[0]), True
        recall_sum += rec
        opr_total += one
        cells += [tp, fp, tn, fn]
        num_eval += c
    curve = np.cumsum(recall_sum) / max(num_eval, 1)
    precisions = np.divide(cells[0], cells[0] + cells[1] + 1e-8)
    recalls = np.divide(cells[0], cells[0] + cells[3] + 1e-8)
    f1 = np.divide(2 * precisions * recalls, precisions + recalls + 1e-8)
    order = np.argsort(recalls)
    auc = float(np.trapezoid(precisions[order], recalls[order])) if hasattr(np, "trapezoid") else float(np.trapz(precisions[order], recalls[order]))
    out = {"n_frames": len(desc_db), "n_queries": nq, "n_evaluated": int(num_eval), "hits_at1": int(first_pos.sum()),
           "r1": float(curve[0]), "r5": float(curve[4]), "max_f1": float(f1.max()), "opr": opr_total / max(num_eval, 1),
           "auc": auc, "init_time_s": init_time,
           "query_with_positive_after90s": int(sum(1 for i, q in enumerate(q_meta)
               if float(q["timestamp"]) / 1e9 - init_time >= 90 and any(db_ts[j] < float(q["timestamp"]) / 1e9 - 60 for j in q.get(m, []))))}
    if official_pair:
        # 공식 파일은 그대로 두고, numpy 2에서 이름만 바뀐 trapz를 실행 시 연결한다.
        out["numpy_trapz_shim"] = not hasattr(np, "trapz")
        if out["numpy_trapz_shim"]:
            np.trapz = np.trapezoid
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            E._get_recall_for_pair(m, m, {m: desc_db}, {m: desc_q}, {m: dict(enumerate(q_meta))},
                                   {m: dict(enumerate(db_meta))}, n_jobs=1)
        line = buf.getvalue().strip().splitlines()[-1]
        g = re.search(r"Recall@1=([\d.]+)% OPR=([\d.]+)% MaxF1=([\d.]+) AUC=([\d.-]+) over (\d+)", line)
        out["official_pair_print"] = line
        out["official_pair_equal"] = bool(g and abs(float(g[1]) - 100 * out["r1"]) < 6e-5 and abs(float(g[3]) - out["max_f1"]) < 6e-5
                                          and int(g[5]) == num_eval and abs(float(g[2]) - 100 * out["opr"]) < 6e-5)
    if independent:
        out["independent_float64"] = independent_check(desc_db, desc_q, db_meta, q_meta, m, first_index, first_pos,
                                                        evaluated, thresholds, init_time)
    arrays = {"first_index": first_index, "first_distance": first_distance, "first_is_positive": first_pos,
              "evaluated": evaluated, "thresholds": thresholds, "tp_fp_tn_fn": cells}
    return out, arrays


def independent_check(desc_db, desc_q, db_meta, q_meta, m, first_index, first_pos, evaluated, thresholds, init_time) -> dict:
    """KDTree·공식 함수 없이 float64 전수 L2로 같은 규약을 다시 계산한다."""
    import numpy as np
    db = desc_db.astype(np.float64)
    db_ts = np.array([float(d["timestamp"]) / 1e9 for d in db_meta])
    kq = max(200, 30, max(int(round(len(db) / 1000.0)), 1))
    ev = np.zeros(len(desc_q), bool)
    fp_ = np.zeros(len(desc_q), np.int8)
    fd_ = np.full(len(desc_q), np.nan)
    same_first = 0
    for i in range(len(desc_q)):
        qt = float(q_meta[i]["timestamp"]) / 1e9
        pos = {j for j in q_meta[i].get(m, []) if db_ts[j] < qt - 60}
        if not pos or qt - init_time < 90:
            continue
        d = np.sqrt(((db - desc_q[i].astype(np.float64)) ** 2).sum(1))
        top = np.argsort(d, kind="stable")[:kq]
        top = top[db_ts[top] < qt - 60]
        if not len(top):
            continue
        ev[i], fp_[i], fd_[i] = True, int(top[0] in pos), d[top[0]]
        same_first += int(top[0] == first_index[i])
    acc = fd_[ev][:, None] < thresholds[None, :]
    pos = fp_[ev].astype(bool)[:, None]
    tp, fpos, fn = (acc & pos).sum(0), (acc & ~pos).sum(0), (~acc & pos).sum(0)
    pr, rc = tp / (tp + fpos + 1e-8), tp / (tp + fn + 1e-8)
    f1 = float((2 * pr * rc / (pr + rc + 1e-8)).max())
    return {"n_evaluated": int(ev.sum()), "evaluated_equal": bool(np.array_equal(ev, evaluated)),
            "same_first_index": same_first, "r1": float(fp_[ev].mean()) if ev.any() else 0.0, "max_f1": f1}


def stage_score(attempt: str) -> dict:
    _, ar = dirs(attempt)
    prep = read(ar / "prepare.json")
    rows = []
    for s in SEQUENCES:
        if not prep["plan"][s]["source"]:
            continue
        pay = load_payload(ar, s)
        db_meta, q_meta = list(pay["db"].values()), list(pay["query"].values())
        assert [d["query"] for d in db_meta] == [q["query"] for q in q_meta], "intra-session needs same order"
        desc = load_desc(ar / "desc" / s, len(db_meta))
        res, arrays = official_score(desc, desc, db_meta, q_meta, pay["m"], independent=True)
        res.update({"sequence": s, "condition": "C1_author_full", "source": pay["source"], "published": PUBLISHED[s]})
        save_score(ar, f"{s}_C1", res, arrays)
        rows.append(res)
    macro = {k: sum(r[k] for r in rows) / len(rows) for k in ("r1", "r5", "max_f1")} if len(rows) == 3 else None
    out = {"rows": rows, "macro": macro, "published_average": PROTOCOL["paper"]["published_average"]}
    write(ar / "score.json", out)
    return out


def save_score(ar: Path, name: str, res: dict, arrays: dict) -> None:
    import numpy as np
    (ar / "score").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ar / "score" / f"{name}.npz", **arrays)
    res["npz_sha256"] = sha(ar / "score" / f"{name}.npz")
    write(ar / "score" / f"{name}.json", res)


# ---------------------------------------------------------------- compare
def to_ns(ts: float) -> float:
    return float(ts) if float(ts) > 1e17 else float(ts) * 1e9


def regenerated(s: str) -> dict:
    import numpy as np
    z = np.load(NATIVE / "inputs" / s / "index.npz")
    with (NATIVE / "inputs" / s / "metadata.pickle").open("rb") as f:
        meta = pickle.load(f)
    return {"timestamps": z["timestamps"].astype(np.int64), "poses": z["poses"], "scan_ids": z["scan_ids"],
            "meta": [meta[i] for i in range(len(meta))]}


def match_frames(ta, tr, tol_ns: float = 50e6):
    import numpy as np
    order = np.argsort(tr)
    trs = tr[order]
    pairs = []
    for ia, t in enumerate(ta):
        j = np.searchsorted(trs, t)
        best = min((k for k in (j - 1, j) if 0 <= k < len(trs)), key=lambda k: abs(trs[k] - t))
        ir = int(order[best])
        # 상호 최근접 확인
        ja = int(np.argmin(np.abs(ta - tr[ir])))
        if ja == ia and abs(tr[ir] - t) <= tol_ns:
            pairs.append((ia, ir, float(tr[ir] - t)))
    return pairs


def model_input(img):
    import cv2
    import numpy as np
    img = np.concatenate([img[:, -28:], img, img[:, :28]], axis=1)
    return cv2.resize(img, (img.shape[1], 126), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def stage_compare(attempt: str) -> dict:
    import numpy as np
    _, ar = dirs(attempt)
    prep = read(ar / "prepare.json")
    out = {"sequences": {}}
    for s in SEQUENCES:
        if not prep["plan"][s]["source"]:
            continue
        pay = load_payload(ar, s)
        m = pay["m"]
        db_meta, q_meta = list(pay["db"].values()), list(pay["query"].values())
        reg = regenerated(s)
        ta = np.array([to_ns(d["timestamp"]) for d in db_meta])
        tr = reg["timestamps"].astype(np.float64)
        pairs = match_frames(ta, tr)
        ia = np.array([p[0] for p in pairs], int)
        ir = np.array([p[1] for p in pairs], int)
        dt = np.array([p[2] for p in pairs])
        rec = {"n_author": len(ta), "n_regenerated": len(tr), "n_matched": len(pairs),
               "author_only": int(len(ta) - len(pairs)), "regenerated_only": int(len(tr) - len(pairs)),
               "dt_ms": {"max_abs": float(np.abs(dt).max() / 1e6) if len(dt) else None,
                         "median_abs": float(np.median(np.abs(dt)) / 1e6) if len(dt) else None},
               "author_timestamp_unit": "ns" if float(db_meta[0]["timestamp"]) > 1e17 else "s_or_other"}
        steps_a = np.linalg.norm(np.diff(np.array([[d["northing"], d["easting"]] for d in db_meta]), axis=0), axis=1)
        steps_r = np.linalg.norm(np.diff(reg["poses"][:, :2, 3], axis=0), axis=1)
        rec["sampling_step_m"] = {"author_median": float(np.median(steps_a)), "author_min": float(steps_a.min()),
                                  "regenerated_median": float(np.median(steps_r)), "regenerated_min": float(steps_r.min())}
        a_only = sorted(set(range(len(ta))) - set(ia.tolist()))
        rec["author_only_time_s_from_start"] = [float((ta[i] - ta.min()) / 1e9) for i in a_only[:400]]
        r_only = sorted(set(range(len(tr))) - set(ir.tolist()))
        rec["regenerated_only_time_s_from_start"] = [float((tr[i] - tr.min()) / 1e9) for i in r_only[:400]]
        # 영상 비교
        img = {"k15": [], "k25": []}
        shapes = {}
        closer = {"k15": 0, "k25": 0, "tie": 0}
        empty_a, empty_r = [], []
        for a_i, r_i in zip(ia, ir):
            A = np.load(Path(pay["image_base"]) / db_meta[a_i]["query"]).astype(np.float32)
            shapes[str(A.shape)] = shapes.get(str(A.shape), 0) + 1
            ma = model_input(A)
            empty_a.append(float((A[..., 1] == 0).mean()))
            normal = {}
            for k in ("k15", "k25"):
                R = np.load(NATIVE / k / "data/validation" / s / f"{r_i:06d}.npy").astype(np.float32)
                mr = model_input(R)
                d = np.abs(ma - mr)
                row = {"mean": d.mean(axis=(0, 1)).tolist(), "max": d.max(axis=(0, 1)).tolist(),
                       "exact": (d <= 1e-6).mean(axis=(0, 1)).tolist()}
                if A.shape == R.shape:
                    row["raw_max"] = np.abs(A - R).max(axis=(0, 1)).tolist()
                img[k].append(row)
                normal[k] = row["mean"][2]
                if k == "k25":
                    empty_r.append(float((R[..., 1] == 0).mean()))
            closer["k15" if normal["k15"] < normal["k25"] else "k25" if normal["k25"] < normal["k15"] else "tie"] += 1
        rec["image"] = {"author_shapes": shapes, "normal_channel_closer": closer,
                        "author_empty_range_fraction_mean": float(np.mean(empty_a)) if empty_a else None,
                        "regenerated_empty_range_fraction_mean": float(np.mean(empty_r)) if empty_r else None}
        for k in ("k15", "k25"):
            if img[k]:
                rec["image"][k] = {"mean_abs_by_channel": np.mean([x["mean"] for x in img[k]], 0).tolist(),
                                   "max_abs_by_channel": np.max([x["max"] for x in img[k]], 0).tolist(),
                                   "exact_fraction_by_channel": np.mean([x["exact"] for x in img[k]], 0).tolist(),
                                   "frames_all_channels_exact": int(sum(min(x["exact"]) == 1.0 for x in img[k]))}
        # 정답 비교(평가 시 60초 필터 뒤)
        r_of_a = {int(a): n for n, a in enumerate(ia)}
        r_of_r = {int(r): n for n, r in enumerate(ir)}
        tsa = ta / 1e9
        tsr = tr / 1e9
        same = jac = xor = 0
        unmatched_pos_a = unmatched_pos_r = 0
        lab_a, lab_r = [], []
        for n, (a_i, r_i) in enumerate(zip(ia, ir)):
            pa = [j for j in q_meta[a_i].get(m, []) if tsa[j] < tsa[a_i] - 60]
            pr = [j for j in reg["meta"][r_i].get(0, []) if tsr[j] < tsr[r_i] - 60]
            A_ = {r_of_a[j] for j in pa if j in r_of_a}
            B_ = {r_of_r[j] for j in pr if j in r_of_r}
            unmatched_pos_a += sum(j not in r_of_a for j in pa)
            unmatched_pos_r += sum(j not in r_of_r for j in pr)
            lab_a.append(sorted(A_))
            lab_r.append(sorted(B_))
            same += A_ == B_
            xor += bool(A_) != bool(B_)
            jac += (len(A_ & B_) / len(A_ | B_)) if (A_ | B_) else 1.0
        rec["labels_on_matched"] = {"equal_sets": int(same), "mean_jaccard": jac / max(len(ia), 1),
                                    "positive_presence_differs": int(xor),
                                    "author_positives_outside_matched": int(unmatched_pos_a),
                                    "regenerated_positives_outside_matched": int(unmatched_pos_r)}
        # 사전 지정 분해
        dec = {}
        desc_a = load_desc(ar / "desc" / s, len(db_meta))
        c1 = read(ar / "score" / f"{s}_C1.json")
        dec["C1"] = {k: c1[k] for k in ("r1", "max_f1", "n_evaluated", "n_frames")}

        # 제한 집합의 양성은 원 목록을 공통 영상 위치로 옮기고, 60초 필터는 평가 함수가 적용한다.
        lab_a_raw = [[r_of_a[j] for j in q_meta[a_i].get(m, []) if j in r_of_a] for a_i in ia]
        meta_a_r = [{"query": db_meta[a_i]["query"], "timestamp": float(ta[a_i]), 0: lab_a_raw[n_]}
                    for n_, a_i in enumerate(ia)]
        res, arr = official_score(desc_a[ia], desc_a[ia], meta_a_r, meta_a_r, 0, official_pair=False)
        save_score(ar, f"{s}_C2", res, arr)
        dec["C2"] = {k: res[k] for k in ("r1", "max_f1", "n_evaluated", "n_frames")}
        reg_meta = reg["meta"]
        lab_r_raw = [[r_of_r[j] for j in reg_meta[r_i].get(0, []) if j in r_of_r] for r_i in ir]
        meta_r_r = [{"query": reg_meta[r_i]["query"], "timestamp": float(tr[r_i]), 0: lab_r_raw[n_]} for n_, r_i in enumerate(ir)]
        for k in ("k15", "k25"):
            droot = NATIVE / k / "author/desc_batch1" / s
            desc_r = load_desc(droot, len(tr), key="indices")
            res, arr = official_score(desc_r[ir], desc_r[ir], meta_a_r, meta_a_r, 0, official_pair=False)
            save_score(ar, f"{s}_C3_{k}", res, arr)
            dec[f"C3_{k}"] = {kk: res[kk] for kk in ("r1", "max_f1", "n_evaluated", "n_frames")}
            res, arr = official_score(desc_r[ir], desc_r[ir], meta_r_r, meta_r_r, 0, official_pair=False)
            save_score(ar, f"{s}_C4_{k}", res, arr)
            dec[f"C4_{k}"] = {kk: res[kk] for kk in ("r1", "max_f1", "n_evaluated", "n_frames")}
            prev = read(NATIVE / k / "author" / f"native_batch1_{s}.json")
            dec[f"C5_{k}"] = {"r1": prev["r1"], "max_f1": prev["official_max_f1"], "n_evaluated": prev["official_evaluated_queries"],
                              "n_frames": prev["n_frames"], "source": str((NATIVE / k / "author" / f"native_batch1_{s}.json").relative_to(REPO))}
        rec["decomposition"] = dec
        out["sequences"][s] = rec
    write(ar / "compare.json", out)
    return out


# ---------------------------------------------------------------- report
def stage_report(attempt: str) -> dict:
    auth, ar = dirs(attempt)
    inv = read(ar / "inventory.json") or {}
    prep = read(ar / "prepare.json") or {}
    sc = read(ar / "score.json") or {}
    cmp_ = read(ar / "compare.json") or {}
    proto = read(RUN / "protocol.json") or {}
    notes = []
    if inv and not inv.get("any_mulran_data"):
        notes.append("압축본 전체 목록에서 MulRan DCC01-03 영상이 없다. 원본 입력 평가는 수행하지 않았다")
    elif inv and not inv.get("mulran_inputs_present"):
        notes.append("MulRan 자료가 일부 시퀀스에만 있다")
    if prep and not prep.get("ready"):
        notes.append("영상 또는 정답이 없는 시퀀스가 있어 해당 시퀀스는 평가하지 않았다")
    rows = [{"sequence": r["sequence"], "r1": r["r1"], "r5": r["r5"], "max_f1": r["max_f1"], "opr": r["opr"], "auc": r["auc"],
             "n_evaluated": r["n_evaluated"], "n_frames": r["n_frames"], "published": r["published"], "source": r["source"],
             "official_pair_equal": r.get("official_pair_equal"), "independent_float64": r.get("independent_float64"),
             "delta_pp": {"r1": 100 * (r["r1"] - r["published"]["r1"]), "f1": 100 * (r["max_f1"] - r["published"]["f1"])}}
            for r in sc.get("rows", [])]
    summary = {}
    for s, rec in (cmp_.get("sequences") or {}).items():
        summary[s] = {"frames": [rec["n_author"], rec["n_regenerated"], rec["n_matched"]],
                      "normal_closer": rec["image"]["normal_channel_closer"],
                      "labels_equal_sets": rec["labels_on_matched"]["equal_sets"],
                      "decomposition_r1": {k: v["r1"] for k, v in rec["decomposition"].items()}}
    tree_top = {}
    for k, v in (inv.get("tree") or {}).items():
        top = "/".join(k.split("/")[:3])
        t = tree_top.setdefault(top, {"files": 0, "bytes": 0})
        t["files"] += v["files"]
        t["bytes"] += v["bytes"]
    hashes = {"protocol.json": sha(RUN / "protocol.json") if (RUN / "protocol.json").exists() else None,
              "extract_result.json": sha(auth / "extract_result.json") if (auth / "extract_result.json").exists() else None,
              "manifest.jsonl": sha(auth / "manifest.jsonl") if (auth / "manifest.jsonl").exists() else None,
              "model": PROTOCOL["model"]["sha256"], "evaluate.py": sha(OFFICIAL / "eval/evaluate.py"),
              "eval_script": sha(Path(__file__))}
    ex = read(auth / "extract_result.json") or {}
    value = {"attempt": attempt, "protocol_frozen_unix": proto.get("frozen_unix"),
             "archive": {k: ex.get(k) for k in ("compressed_bytes", "compressed_sha256", "gzip_members",
                                                 "gzip_crc32_isize_verified_by_zlib", "decompressed_bytes", "tar_members",
                                                 "counts", "n_blocked", "selected_files", "selected_bytes")},
             "inventory_summary": {"any_mulran_data": inv.get("any_mulran_data"), "mulran_inputs_present": inv.get("mulran_inputs_present"),
                                   "mulran_labels_present": inv.get("mulran_labels_present"),
                                   "validation_dirs": inv.get("validation_dirs"), "top_level": tree_top,
                                   "pickles": [{"name": p["name"], "sets": [{"n": st["n"], "sequences": st["sequences"]} for st in p.get("sets", [])],
                                                "load_error": p.get("load_error")} for p in inv.get("pickles", [])]},
             "plan": {s: {"source": v.get("source"), "n_frames": v.get("n_frames")} for s, v in (prep.get("plan") or {}).items()},
             "primary_rows": rows, "primary_macro": sc.get("macro"), "published_average": PROTOCOL["paper"]["published_average"],
             "comparison_summary": summary, "notes": notes, "hashes": hashes,
             "chain_status": read(ar / "chain_status.json")}
    write(ar / "server_report.json", value)
    return value


# ---------------------------------------------------------------- selftest
def stage_selftest_build(attempt: str) -> dict:
    """재생성 DCC01 k25 영상(하드링크)·정답으로 저자 압축본 형식의 합성 입력을 만든다.

    전체 체인이 기존 k25 DCC01 결과(R@1 0.946615, F1 0.973226)를 그대로 다시 내는지 검사하는 용도다.
    """
    assert attempt.startswith("selftest"), "selftest attempt only"
    auth, _ = dirs(attempt)
    assert not auth.exists(), auth
    s = "DCC01"
    reg = regenerated(s)
    root = auth / "files/data/validation" / s
    root.mkdir(parents=True)
    rows = []
    for i in range(len(reg["meta"])):
        src = NATIVE / "k25/data/validation" / s / f"{i:06d}.npy"
        os.link(src, root / f"{i:06d}.npy")
        name = f"data/validation/{s}/{i:06d}.npy"
        rows.append({"name": name, "type": "reg", "size": src.stat().st_size, "sha256": sha(src),
                     "saved": f"files/{name}", "class": "mulran_data",
                     "npy": {"descr": "<f4", "shape": [64, 1022, 3], "fortran_order": False}})
    lines = [f"{m['timestamp']} {m['northing']} {m['easting']} 0 0 0 0 1" for m in reg["meta"]]
    (root / "poses.txt").write_text("\n".join(lines) + "\n")
    rows.append({"name": f"data/validation/{s}/poses.txt", "type": "reg", "size": (root / "poses.txt").stat().st_size,
                 "sha256": sha(root / "poses.txt"), "saved": f"files/data/validation/{s}/poses.txt", "class": "metadata"})
    st = {i: {"query": m["query"], "northing": m["northing"], "easting": m["easting"], "timestamp": m["timestamp"], 0: m[0]}
          for i, m in enumerate(reg["meta"])}
    for kind in ("db", "query"):
        p = auth / f"files/data/{kind}_mulran_{s}.pickle"
        with p.open("wb") as f:
            pickle.dump([st], f, protocol=pickle.HIGHEST_PROTOCOL)
        rows.append({"name": f"data/{kind}_mulran_{s}.pickle", "type": "reg", "size": p.stat().st_size, "sha256": sha(p),
                     "saved": f"files/data/{kind}_mulran_{s}.pickle", "class": "metadata"})
    with (auth / "manifest.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = {"attempt": attempt, "passed": True, "synthetic_selftest": True, "n_blocked": 0,
           "expected": {"r1": 0.9466145833333334, "max_f1": 0.973226233277035}}
    write(auth / "extract_result.json", res)
    return res


# ---------------------------------------------------------------- chain (server host)
def stage_chain(attempt: str) -> None:
    _, ar = dirs(attempt)
    ar.mkdir(parents=True, exist_ok=True)
    st = {"attempt": attempt, "pid": os.getpid(), "started_unix": time.time(), "stages": [], "finished": False}
    base = ["docker", "exec", "-w", CONTAINER_REPO, "ring_baseline"]
    env = ["env", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=4", "NUMEXPR_NUM_THREADS=4",
           "CUDA_VISIBLE_DEVICES=0", "PYTHONDONTWRITEBYTECODE=1"]
    py = ["python3", "-u", "scripts/_review_imlpr_author_eval.py"]
    # 호스트와 컨테이너의 락 경로가 같은 파일이면 한 번만 잡는다(같은 파일을 두 번 잡으면 교착된다).
    host_inode = os.stat(LOCK).st_ino if os.path.exists(LOCK) else None
    r = subprocess.run(base + ["stat", "-L", "-c", "%i", LOCK], capture_output=True, text=True)
    container_inode = int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip().isdigit() else None
    same_lock = host_inode is not None and host_inode == container_inode
    st["gpu_lock"] = {"path": LOCK, "host_inode": host_inode, "container_inode": container_inode,
                      "mode": "host_only_same_file" if same_lock else "host_then_container"}

    def run(name: str, lock: bool = False) -> bool:
        cmd = base + env + py + [name, "--attempt", attempt]
        if lock:
            if same_lock:
                cmd = ["flock", "-x", LOCK] + cmd
            else:
                cmd = ["flock", "-x", LOCK] + base + ["flock", "-x", LOCK] + env + py + [name, "--attempt", attempt]
        row = {"stage": name, "command": " ".join(cmd), "start_unix": time.time()}
        st["stages"].append(row)
        st["current"] = name
        write(ar / "chain_status.json", st)
        with (ar / f"stage_{name}.log").open("w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        row.update({"rc": rc, "end_unix": time.time()})
        write(ar / "chain_status.json", st)
        return rc == 0

    ok = run("protocol_check") and run("inspect")
    present = False
    if ok:
        inv = read(ar / "inventory.json") or {}
        present = bool(inv.get("any_mulran_data") or inv.get("mulran_labels_present"))
        st["mulran_present"] = present
    if ok and present:
        ok = run("prepare")
        prep = read(ar / "prepare.json") or {}
        if ok and any(v.get("source") for v in prep.get("plan", {}).values()):
            ok = run("infer", lock=True) and run("score") and run("compare")
    report_ok = run("report")
    st.update({"finished": True, "ok": bool(ok and report_ok), "finished_unix": time.time(), "current": None})
    write(ar / "chain_status.json", st)
    run("report")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["protocol", "protocol_check", "inspect", "prepare", "infer", "score", "compare",
                                      "report", "chain", "selftest_build"])
    ap.add_argument("--attempt", default="")
    a = ap.parse_args()
    if a.stage in ("protocol", "protocol_check"):
        if a.stage == "protocol_check":
            assert (RUN / "protocol.json").exists(), "protocol not frozen"
        print(json.dumps(stage_protocol()["frozen_unix"]))
    elif a.stage == "chain":
        stage_chain(a.attempt)
    else:
        fn = {"inspect": stage_inspect, "prepare": stage_prepare, "infer": stage_infer, "score": stage_score,
              "compare": stage_compare, "report": stage_report, "selftest_build": stage_selftest_build}[a.stage]
        r = fn(a.attempt)
        print(json.dumps({"stage": a.stage, "ok": True}), flush=True)
