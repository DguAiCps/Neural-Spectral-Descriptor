#!/usr/bin/env python3
"""현재 시간 NSD의 문맥을 고정 승자의 최종 수락 점수에 연결한다.

ring_baseline에서 실행한다. 학습 8 적합 후 검증 9·3시드를 보고하며
시험 자료나 가중치 업데이트를 사용하지 않는다.
"""
from __future__ import annotations

import concurrent.futures
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_spectral_evidence as exp
import _pdv2_train_failure as previous
from _pdv2_context_accept_lib import (
    ARMS, FEATURES, discrimination, fit_score, labels, pair_features,
    predict_score, unit,
)
from _pdv2_matching_mine import TRAIN, cache_path, spec_path
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_pilot import VAL
from _pr_eval_lib import classify_queries
from _train_attention_neighbors import array_hash

OUT = REPO / "results/pdv2_context_accept_20261018"
PRIOR = REPO / "results/pdv2_train_failure_20261018"
VAL_SOURCE = REPO / "results/pdv2_sequence_evidence_20261018"
BASELINE = "arc_unique_b100"
METHODS = ("baseline",) + tuple(ARMS)
FIELDS = ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist")


def family(sequence: str) -> str:
    """KITTI-360을 KITTI 계열로 묶는다."""
    return "kitti" if sequence.startswith("kitti") else sequence.split("/")[0]


def prepare() -> dict:
    """새 결과를 계산하기 전에 입력·대조군·확장 기준을 동결한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = exp.read(target)
        for p, expected in spec["sources"].items():
            assert sha256(REPO / p) == expected, p
        return spec
    old = exp.protocol(True)
    frozen = exp.read(previous.FREEZE)
    sources = {REPO / p for p in frozen["runtime_sha256"]}
    sources.update(REPO / p for p in old["input_sha256"] if "capacity" not in p)
    sources.update((Path(__file__), REPO / "scripts/_pdv2_context_accept_lib.py",
                    REPO / "tests/retrieval/test_context_accept.py", previous.PREPARED,
                    previous.GRAPH, previous.FREEZE, Path(previous.__file__)))
    sources.update(PRIOR.glob("ledger_*.npz"))
    sources.update(PRIOR / p for p in ("retrieval_keys.npy", "training_metadata.npz", "keys_complete.json"))
    for seed in (1, 2, 3):
        sources.add(REPO / frozen["seeds"][str(seed)]["checkpoint"])
        sources.add(exp.old.OUT / f"s{seed}/standard_original/training_graph.json")
        for sensor, seq in VAL:
            sources.add(VAL_SOURCE / f"s{seed}/standard_{sensor}_{seq}.npz")
    for sensor, seq in TRAIN:
        sources.update((cache_path(sensor+"/"+seq), spec_path(sensor+"/"+seq)))
    spec = {"created_unix": time.time(), "seeds": frozen["seeds"],
            "selected": old["selected"]["standard"], "arms": ARMS,
            "train": [s+"/"+q for s, q in TRAIN], "val": [s+"/"+q for s, q in VAL],
            "scope": "학습GT8·검증KISS9,현재시간전용NSD3시드. 모델·후보·승자고정. 새시험평가없음.",
            "fit": "각시드학습만. 모호질의제외,승자5m이내=1,나머지=0. 2계열균등→계열내시퀀스균등→질의균등. 표준화도같은학습가중치. 양의가중합+편향,BCE+L2(.001),새신경망없음.",
            "features": "d센터,문맥128센터,전체416키센터,d시간평균,문맥시간평균. 시간평균은기존스펙트럼선택방향과동일대응·중심비중·반복가중치. gate곱/게이트포함문맥내적은진단만.",
            "selection": "각시드AP>=baseline+.001,R100>=baseline-.001,3시드평균각계열AP>=baseline-.0025. 통과조건만평균AP최고. 모두실패하면유지. 새조건추가없음.",
            "interpretation": "문맥가지128차원은자기프레임정보도포함. 유용성이있어도시간간선의인과적기여주장은용량대조필요. 학습GT와검증KISS입력차이유지.",
            "prior_difference": "20261015가중합공동학습과달리현재시간모델동결,최종승자동결,시간스펙트럼기준,문맥가지를분리. 검색이나인코더를재학습하지않음.",
            "test_used": False, "encoder_updates": 0,
            "sources": {str(p.relative_to(REPO)): sha256(p) for p in sorted(sources)}}
    write_json(target, spec)
    print("FROZEN", len(sources), flush=True)
    return spec


def train_root(seed: int) -> Path:
    """시드1의 검증된 원장과 새 시드별 원장을 구분한다."""
    return PRIOR if seed == 1 else OUT / f"train/s{seed}"


def generate_training(seed: int, spec: dict) -> None:
    """선택된 시드의 실제 학습 그래프를 동일 검색 규칙으로 재생한다."""
    if seed == 1:
        return
    root = train_root(seed)
    root.mkdir(parents=True, exist_ok=True)
    done = root / "keys_complete.json"
    if not done.exists():
        graph = torch.load(previous.GRAPH, map_location="cpu", weights_only=False)
        keep = graph.edge_type == 0
        graph.edge_index, graph.edge_attr, graph.edge_type = (
            graph.edge_index[:, keep], graph.edge_attr[keep], graph.edge_type[keep])
        if hasattr(graph, "edge_pose_label"):
            graph.edge_pose_label = graph.edge_pose_label[keep]
        hashes = {k: array_hash(getattr(graph, k).numpy()) for k in ("x", "edge_index", "edge_attr", "edge_type")}
        expected = exp.read(exp.old.OUT / f"s{seed}/standard_original/training_graph.json")["hashes"]
        assert hashes == expected
        epoch = spec["seeds"][str(seed)]["epoch"]
        _, model = exp.old.load_model(seed, "standard_original", epoch)
        with torch.no_grad():
            key = unit(model(graph.to("cuda")).float().cpu().numpy())
        assert key.shape == (72306, 416) and np.isfinite(key).all()
        prefix = []
        meta = exp.arrays(PRIOR / "training_metadata.npz")
        from torch_geometric.data import Data
        for ordinal in range(len(TRAIN)):
            ids = np.flatnonzero(meta["sequences"] == ordinal)
            for local in (len(ids)//3, 2*len(ids)//3):
                query, end = ids[local], ids[local]+16
                mask = (graph.edge_index[0] < end) & (graph.edge_index[1] < end)
                small = Data(x=graph.x[:end], edge_index=graph.edge_index[:, mask],
                             edge_attr=graph.edge_attr[mask], edge_type=graph.edge_type[mask])
                with torch.no_grad():
                    one = unit(model(small)[query:query+1].float().cpu().numpy())[0]
                error = float(np.max(np.abs(one-key[query])))
                np.testing.assert_allclose(one, key[query], atol=3e-6, rtol=3e-5)
                prefix.append({"sequence": ordinal, "query": int(local), "max_error": error})
        np.save(root / "retrieval_keys.npy", key)
        shutil.copyfile(PRIOR / "training_metadata.npz", root / "training_metadata.npz")
        write_json(done, {"graph_hashes": hashes, "prefix_checks": prefix,
                         "outputs": {p.name: sha256(p) for p in (root/"retrieval_keys.npy", root/"training_metadata.npz")}})
        del model, graph
        torch.cuda.empty_cache()
    for p, expected in exp.read(done)["outputs"].items():
        assert sha256(root / p) == expected
    previous.OUT = root
    replay_spec = {"train_roster": spec["train"], "selected": spec["seeds"][str(seed)]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda s: previous.replay(s, replay_spec), spec["train"]))


def feature_file(role: str, seed: int, sequence: str) -> Path:
    """고정 승자 특징 원장의 경로다."""
    return OUT / f"features/{role}/s{seed}/{sequence.replace('/', '_')}.npz"


def extract(role: str, seed: int, sequence: str, spec: dict) -> None:
    """누출 없는 기존 관측 범위에서 특징을 만들고 기준 점수를 대조한다."""
    target = feature_file(role, seed, sequence)
    if target.exists():
        assert sha256(target) == exp.read(target.with_suffix(".json"))["sha256"]
        return
    sensor, seq = sequence.split("/")
    if role == "train":
        root = train_root(seed)
        source = root / f"ledger_{sensor}_{seq}.npz"
        ledger = exp.arrays(source)
        meta = exp.arrays(root / "training_metadata.npz")
        ids = np.flatnonzero(meta["sequences"] == spec["train"].index(sequence))
        key_path = root / "retrieval_keys.npy"
        key = np.load(key_path, mmap_mode="r")[ids]
        cachefile, specfile = cache_path(sequence), spec_path(sequence)
        raw = ledger[spec["seeds"][str(seed)]["confidence"]["decision"]]
    else:
        source = VAL_SOURCE / f"s{seed}/standard_{sensor}_{seq}.npz"
        ledger = exp.arrays(source)
        key_path = REPO / spec["selected"][str(seed)]["folder"] / f"keys_val_{sensor}_{seq}.npz"
        key = exp.arrays(key_path)["embedding"]
        _, _, paths, _ = exp.old.base.inputs("val", sequence)
        cachefile, specfile = paths["cache"], paths["spec"]
        raw = ledger["raw"]
    cache = exp.arrays(cachefile)
    with np.load(specfile) as z:
        np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
        bank = exp.old.base.bank_for(exp.OP, z)
    xyz = cache["poses"][:, :3, 3]
    if role == "train":
        xyz = xyz.astype(np.float64)
    path = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    q, j = ledger["query_idx"], ledger["j_star"]
    assert np.all(j <= q-30) and np.all(q < len(key)-15)
    channel = spec["seeds"][str(seed)]["confidence"]["decision"]
    chunks = []
    for start in range(0, len(q), 256):
        sl = slice(start, start+256)
        chunks.append(pair_features(key, bank, path, q[sl], j[sl], channel, raw[sl]))
    features = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    np.testing.assert_array_equal(features["baseline"], ledger[BASELINE])
    checks = []
    for row in (len(q)//3, 2*len(q)//3):
        sl, end = slice(row, row+1), int(q[row]+16)
        small = pair_features(key[:end], exp.truncate_bank(bank, end), path[:end], q[sl], j[sl], channel, raw[sl])
        for name in features:
            np.testing.assert_allclose(small[name], features[name][sl], atol=1e-12, rtol=0)
        checks.append(int(q[row]))
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **{f: ledger[f] for f in FIELDS}, **features)
    write_json(target.with_suffix(".json"), {"sequence": sequence, "seed": seed, "role": role,
        "rows": len(q), "sha256": sha256(target), "prefix_queries": checks,
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in (source, key_path, cachefile, specfile)},
        "discrimination": discrimination(features, ledger, spec["seeds"][str(seed)]["new_tau"]["tau"])})
    print("FEATURES", role, seed, sequence, len(q), flush=True)


def fit_all(spec: dict) -> dict:
    """검증 특징 계산 전에 각 시드의 학습 전용 가중치를 저장한다."""
    target = OUT / "fitted.json"
    if target.exists():
        result = exp.read(target)
        for p, expected in result["training_sha256"].items():
            assert sha256(REPO / p) == expected
        return result
    result = {"created_unix": time.time(), "models": {}, "training_sha256": {}, "test_used": False}
    for seed in (1, 2, 3):
        entries, weights, ys = [], [], []
        for sequence in spec["train"]:
            path = feature_file("train", seed, sequence)
            a = exp.arrays(path)
            eligible, y = labels(a)
            entries.append({f: a[f][eligible] for f in FEATURES})
            ys.append(y[eligible])
            count = sum(family(s) == family(sequence) for s in spec["train"])
            weights.append(np.full(int(eligible.sum()), 1/(2*count*eligible.sum())))
            result["training_sha256"][str(path.relative_to(REPO))] = sha256(path)
        features = {f: np.concatenate([a[f] for a in entries]) for f in FEATURES}
        result["models"][str(seed)] = {}
        for name, fields in ARMS.items():
            x = np.column_stack([features[f] for f in fields]).astype(np.float64)
            fitted = fit_score(x, np.concatenate(ys), np.concatenate(weights))
            result["models"][str(seed)][name] = {"fields": fields, **fitted}
    write_json(target, result)
    print("FITTED_TRAIN_ONLY", flush=True)
    return result


def macro(per_sequence: dict) -> dict:
    """양성이 있는 시퀀스→계열 균등 평균을 계산한다."""
    families = {}
    for group in sorted({family(s) for s in per_sequence}):
        rows = [v for s, v in per_sequence.items() if family(s) == group and v["n_pos"]]
        if rows:
            families[group] = {m: float(np.mean([r[m] for r in rows])) for m in exp.METRICS}
    return {"families": families,
            "macro": {m: float(np.mean([v[m] for v in families.values()])) for m in exp.METRICS}}


def report(spec: dict, fits: dict) -> None:
    """모든 조건의 최종 지표와 계열별 손익을 빠짐없이 저장한다."""
    report = {"per_seed": {}, "mean_std": {}, "selected": None, "test_used": False,
              "fitted_sha256": sha256(OUT/"fitted.json"), "protocol_sha256": sha256(OUT/"protocol.json")}
    outputs = {}
    for seed in (1, 2, 3):
        report["per_seed"][str(seed)] = {}
        for role in ("train", "val"):
            metrics = {m: {} for m in METHODS}
            standalone = {f: {} for f in FEATURES if f != "baseline"}
            for sequence in spec[role]:
                p = feature_file(role, seed, sequence)
                a = exp.arrays(p)
                predictions = {"baseline": a["baseline"]}
                for method, model in fits["models"][str(seed)].items():
                    predictions[method] = predict_score(np.column_stack([a[f] for f in model["fields"]]), model)
                for method, score in predictions.items():
                    metrics[method][sequence] = exp.metrics(score, a)
                for f in standalone:
                    standalone[f][sequence] = exp.metrics(a[f], a)
                for m in exp.METRICS:
                    if metrics["baseline"][sequence][m] is not None:
                        np.testing.assert_allclose(metrics["baseline"][sequence][m], metrics["baseline_fit"][sequence][m], atol=1e-12)
                path = OUT / f"scores/{role}/s{seed}/{sequence.replace('/', '_')}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **{f: a[f] for f in FIELDS}, **predictions)
                outputs[str(path.relative_to(REPO))] = sha256(path)
            report["per_seed"][str(seed)][role] = {m: {**macro(v), "per_sequence": v} for m, v in metrics.items()}
            report["per_seed"][str(seed)][role+"_standalone"] = {m: macro(v) for m, v in standalone.items()}
    for role in ("train", "val"):
        report["mean_std"][role] = {}
        for method in METHODS:
            rows = [report["per_seed"][str(s)][role][method] for s in (1,2,3)]
            report["mean_std"][role][method] = {
                "macro": {m: {"mean": float(np.mean([r["macro"][m] for r in rows])), "std": float(np.std([r["macro"][m] for r in rows]))} for m in exp.METRICS},
                "families": {f: {m: float(np.mean([r["families"][f][m] for r in rows])) for m in exp.METRICS} for f in rows[0]["families"]}}
    eligibility = {}
    for method in ARMS:
        rows = []
        for seed in (1, 2, 3):
            values = report["per_seed"][str(seed)]["val"]
            now, base = values[method]["macro"], values["baseline"]["macro"]
            rows.append(now["AP"] >= base["AP"]+.001 and now["R@100P"] >= base["R@100P"]-.001)
        averaged = report["mean_std"]["val"]
        families_ok = all(averaged[method]["families"][f]["AP"] >= averaged["baseline"]["families"][f]["AP"]-.0025 for f in ("kitti", "helipr", "mulran"))
        eligibility[method] = {"seeds_pass": rows, "families_pass": families_ok, "pass": all(rows) and families_ok}
    candidates = [m for m, v in eligibility.items() if v["pass"]]
    if candidates:
        report["selected"] = max(candidates, key=lambda m: report["mean_std"]["val"][m]["macro"]["AP"]["mean"])
    report["eligibility"] = eligibility
    report["output_sha256"] = outputs
    write_json(OUT / "summary.json", report)
    for name, value in report["mean_std"]["val"].items():
        print("FINAL_VAL", name, {m: round(value["macro"][m]["mean"]*100, 4) for m in exp.METRICS}, flush=True)
    print("SELECTED", report["selected"], flush=True)


def main() -> None:
    """학습 특징·적합·검증·집계 순서로 실행한다."""
    started = time.time()
    torch.set_num_threads(1)
    spec = prepare()
    for seed in (1, 2, 3):
        generate_training(seed, spec)
        for sequence in spec["train"]:
            extract("train", seed, sequence, spec)
    fitted = fit_all(spec)
    fit_hash = sha256(OUT / "fitted.json")
    for seed in (1, 2, 3):
        for sequence in spec["val"]:
            extract("val", seed, sequence, spec)
    assert sha256(OUT / "fitted.json") == fit_hash
    report(spec, fitted)
    write_json(OUT / "status.json", {"status": "complete", "runtime_s": time.time()-started})


if __name__ == "__main__":
    main()
