#!/usr/bin/env python3
"""현재 288D·시간 전용 NSD의 표4 대조를 최종 판정까지 측정한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_spectral_keys as source
import _pdv2_spectral_whitening as white
from _causal_pr_dump import prefix_topk
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash
from _xs_eval_lib import align_scores, column_align_scores, minmax, variant_inputs
from retrieval.sequence_evidence import neighbor_plans, path_means, repeated_candidate_weights
from retrieval.spectral_evidence import correlation_profiles

OUT = REPO / "results/pdv2_table4_current_20261030"
MAIN = white.OUT / "fixed_test"
LOG = REPO / "logs/table4_current_20261030"
RETRIEVAL = ("key_only", "aux_only", "no_alignment")
LEARNED = ("graph_free", "standard")
ARMS = (*RETRIEVAL, *LEARNED)
SEEDS, EPOCHS = (1, 2, 3), (10, 20)
METRICS = source.METRICS
FIELDS = source.base.FIELDS
read = source.read


def root(seed: int, arm: str) -> Path:
    """시드와 대조군의 독립 결과 폴더를 반환한다."""
    return OUT / f"s{seed}" / arm


def model(arm: str, seed: int) -> torch.nn.Module:
    """표준 Q/K/V 또는 거의 같은 용량의 노드별 MLP를 만든다."""
    assert arm in LEARNED
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return source.base.model("point" if arm == "graph_free" else "standard")


def protocol(inputs: bool = False) -> dict:
    """동결 소스와 필요한 경우 학습 입력의 무결성을 검사한다."""
    spec = read(OUT / "protocol.json")
    for group in (("runtime", "inputs") if inputs else ("runtime",)):
        for path, digest in spec[group].items():
            assert sha256(REPO / path) == digest, path
    return spec


def prepare() -> dict:
    """검증과 시험 대조를 실행하기 전에 모든 정의를 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        return protocol(True)
    old = source.protocol(True)
    main = read(MAIN / "freeze.json")
    assert read(MAIN / "table2_summary.json")["checks_passed"]
    runtime = dict(old["runtime"])
    for p in (Path(__file__), REPO / "scripts/_pdv2_table4_current_run.py",
              REPO / "tests/retrieval/test_table4_current.py",
              REPO / "scripts/_pdv2_spectral_whitening.py",
              REPO / "src/gnn/graph_free.py"):
        runtime[str(p.relative_to(REPO))] = sha256(p)
    inputs = dict(old["inputs"])
    inputs.update({str(p.relative_to(REPO)): sha256(p) for p in (
        MAIN / "freeze.json", MAIN / "table2_summary.json",
        source.OUT / "test_freeze.json", source.OUT / "validation_summary.json",
        white.OUT / "statistics.npz", white.OUT / "blend/summary.json")})
    initial, parameters, reused = {}, {}, {}
    graph_hashes = old["graph_hashes"]["octave288"]
    parent = REPO / "results/pdv2_temporal_derivatives_20261020"
    assert read(parent / "protocol.json")["graph_hashes"] == graph_hashes
    for seed in SEEDS:
        initial[str(seed)] = {}
        main_initial = read(source.root(seed, "octave288") / "initial_model.json")
        for arm in LEARNED:
            net = model(arm, seed)
            values = {k: array_hash(v.numpy()) for k, v in net.state_dict().items()}
            initial[str(seed)][arm] = values
            parameters[arm] = sum(p.numel() for p in net.parameters())
            common = set(values) & set(main_initial)
            assert all(values[k] == main_initial[k] for k in common), (seed, arm)
            if arm == "standard":
                assert values == main_initial
            if seed == 1:
                previous = parent / "s1" / ("point" if arm == "graph_free" else "standard")
                marker = read(previous / "training_complete.json")
                assert marker["epochs"] == 20 and marker["updates"] == 900
                assert read(previous / "initial_model.json") == values
                assert marker["orders"] == read(source.root(seed, "octave288") / "training_complete.json")["orders"]
                for name, digest in marker["checkpoints"].items():
                    assert sha256(previous / "checkpoints" / name) == digest
                    inputs[str((previous / "checkpoints" / name).relative_to(REPO))] = digest
                for name in ("training_complete.json", "initial_model.json", "training_progress.json"):
                    inputs[str((previous / name).relative_to(REPO))] = sha256(previous / name)
                reused[arm] = str(previous.relative_to(REPO))
    spec = {
        "created_unix": time.time(), "seeds": SEEDS, "epochs": EPOCHS,
        "arms": ARMS, "train": old["train"], "val": old["val"], "test": main["test"],
        "cities": main["cities"], "confidence": old["confidence"],
        "main_models": main["models"], "fraction": 0.25,
        "graph_hashes": graph_hashes, "initial": initial,
        "parameters": parameters, "reused_seed1": reused,
        "training": "현재 주모델과 동일 GT 8·집계288(no_interdiff)·시간±5·2층·128문맥·gate초기.0625. "
                    "같은183249삼중쌍/순서·Adam5e-4/1e-5·InfoNCE·AMP·20에포크900갱신. 직접 GNNTrainer 사용. "
                    "시드1 두 대조만 동일 학습함수·그래프·초기값·모든 배치순서·예산을 검증한 체크포인트 재사용. 시드2/3 새 학습.",
        "graph_free": "간선 메시지 전달 대신 각 노드 독립 128→204→128 MLP 두 층. 나머지 잔차/정규화/출력/게이트 동일. "
                      "167705 대 주모델167561 매개변수(+0.086%); 추론 이웃 불사용. 최종 거리대응 시간 판정은 유지하므로 모든 시간 정보 제거가 아님.",
        "standard": "주모델의 차이 K/V-tanh·기하 상호작용 블록을 표준 절대 이웃 Q/K/V·가산 기하 편향으로 교체. "
                    "167561 매개변수·동일 시간 간선/입출력/깊이. 전체 attention 블록 비교이며 차이 연산 하나만의 분해로 주장하지 않음.",
        "retrieval": "key_only=200+0, aux_only=0+200, 나머지100+100. 중복 제거 뒤 보충 없음. 나머지 네 점수/시드별 융합/최종 판정 유지.",
        "no_alignment": "두 색인100+100과 융합/시간이웃/25%보정을 유지. 두 스펙트럼 채널의 회전최대값을 "
                        "후보순위·중심·모든 시간 이웃·보정 원본/변환 뱅크에서 오직 0-shift 값으로 바꿈. "
                        "허용 판정 c_spec/c_col 중 보정된 최종 검증 AP 최대, 1e-12 동률은 기존 시드 채널. 시험 전 동결.",
        "selection": "학습 대조는 주모델의 체크포인트 선택과 동일하게 10/20에포크의 보정 전 최종 arc_unique_b100 "
                     "검증 AP(시퀀스→3계열) 최대, 1e-12 동률 이른 에포크. 25%보정은 이후 공통 고정 적용. "
                     "각 대조의 보정 후 검증 지표도 모두 보존. 시험 결과로 선택 변경하지 않음.",
        "evaluation": "KISS 자세·q+15·j<=q-30·공통195726질의. 최종 승자와 수락 점수의 AP/maxF1/R100/R1. "
                      "KITTI07 원장 보존하되 양성 없는 평균 제외;45시퀀스→6도시→3시드 평균/표본표준편차.",
        "test_exposure": "기존 개발에서 본 시험 분할의 추가 분석이다. 새 미노출 확증/신규 사전등록이라고 주장하지 않는다.",
        "test_used_for_selection": False, "runtime": runtime, "inputs": inputs,
    }
    write_json(OUT / "protocol.json", spec)
    return spec


def train(seed: int, arm: str) -> None:
    """현재 주모델과 같은 학습 함수와 표본을 사용한다."""
    spec = protocol()
    folder = root(seed, arm)
    marker = folder / "training_complete.json"
    if marker.exists():
        for name, digest in read(marker)["checkpoints"].items():
            assert sha256(folder / "checkpoints" / name) == digest
        return
    if seed == 1:
        previous = REPO / spec["reused_seed1"][arm]
        (folder / "checkpoints").mkdir(parents=True, exist_ok=True)
        for name in read(previous / "training_complete.json")["checkpoints"]:
            shutil.copy2(previous / "checkpoints" / name, folder / "checkpoints" / name)
        for name in ("training_complete.json", "initial_model.json", "training_progress.json"):
            shutil.copy2(previous / name, folder / name)
        write_json(folder / "reuse_audit.json", {
            "source": str(previous.relative_to(REPO)), "same_graph": True,
            "same_initial": True, "same_900_updates": True, "same_all_batch_orders": True,
            "same_training_function": "_pdv2_temporal_derivatives.train",
        })
    else:
        base = source.base
        original_model = base.model
        # 전역 모델 함수를 바꾸기 전에 원래 생성 함수를 보존한다.
        def chosen(selected: str):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                return original_model("point" if selected == "graph_free" else "standard")
        base.PREPARED = source.OUT / "prepared_octave288.pt"
        base.GRAPH = source.OUT / "graph_octave288.pt"
        base.root, base.model, base.EPOCHS = root, chosen, EPOCHS
        base.protocol = lambda *args: spec
        source.ORIGINAL_TRAIN(seed, arm)
    assert read(folder / "initial_model.json") == spec["initial"][str(seed)][arm]
    expected = read(source.root(seed, "octave288") / "training_complete.json")
    assert read(marker)["orders"] == expected["orders"]


def temporal_score(bank: tuple, travel: np.ndarray, q: np.ndarray, j: np.ndarray,
                   channel: str, raw: np.ndarray, zero_shift: bool = False) -> np.ndarray:
    """현재 거리 대응 시간 판정에서 각 스펙트럼의 각도 선택만 바꾼다."""
    profile = correlation_profiles(bank, q, j, channel)
    center = profile[:, 0] if zero_shift else profile.max(1)
    qn, jn, valid = neighbor_plans(travel, q, j)["arc_unique"]
    n = len(travel)
    unique, inverse = np.unique((qn * n + jn)[valid], return_inverse=True)
    profile = correlation_profiles(bank, unique // n, unique % n, channel)
    peaks = profile[:, 0] if zero_shift else profile.max(1)
    values = np.zeros(qn.shape, dtype=peaks.dtype)
    values[valid] = peaks[inverse]
    means = path_means(center, values, valid, repeated_candidate_weights(jn, valid))
    beta = (2. / 3.) * (11. / 10.)
    return (1 - beta) * raw + beta * means.max(0)


def final_scores(bank: tuple, corrected: tuple, travel: np.ndarray, row: dict,
                 channel: str, zero_shift: bool) -> dict:
    """원본/25% 상관 보정의 최종 수락 점수를 반환한다."""
    q, j = row["query_idx"], row["j_star"]
    all_values = {}
    for name, current in (("original", bank), ("corrected", corrected)):
        values = []
        for start in range(0, len(q), 256):
            sl = slice(start, start + 256)
            if name == "original":
                raw = row[channel][sl]
            else:
                profile = correlation_profiles(current, q[sl], j[sl], channel)
                raw = profile[:, 0] if zero_shift else profile.max(1)
            values.append(temporal_score(current, travel, q[sl], j[sl], channel, raw, zero_shift))
        all_values[name] = np.concatenate(values)
    all_values["score"] = .75 * all_values["original"] + .25 * all_values["corrected"]
    assert all(np.isfinite(v).all() for v in all_values.values())
    return all_values


def retrieve(key: np.ndarray, bank: tuple, positions: np.ndarray, min_past: np.ndarray,
             weights: list, queries: np.ndarray, arms: tuple) -> dict:
    """후보 검색부터 최종 승자까지 대조별로 다시 계산한다."""
    hkey, hfft, hnorm, column = bank
    output, hashes = {}, {}
    for arm in arms:
        output[arm] = {k: np.empty(len(queries), dtype=dtype) for k, dtype in (
            ("j_star", np.int64), ("top1_geo_dist", np.float64), ("c_spec", np.float64),
            ("c_col", np.float64), ("score_abs", np.float64), ("score_margin", np.float64),
            ("n_candidates", np.int64), ("union_hit", bool))}
        output[arm].update(query_idx=queries.copy(), min_past_gt_dist=min_past[queries])
        hashes[arm] = hashlib.sha256()
    for index, query in enumerate(queries):
        q = int(query)
        ka100 = prefix_topk(key, q, q - 29, 100)
        aa100 = prefix_topk(hkey, q, q - 29, 100)
        # 동점이 있으면 200개 결과의 앞100과 원래100개 검색이 다를 수 있다.
        ka = prefix_topk(key, q, q - 29, 200) if "key_only" in arms else ka100
        aa = prefix_topk(hkey, q, q - 29, 200) if "aux_only" in arms else aa100
        union = np.unique(np.r_[ka100, aa100])
        for arm in arms:
            candidates = np.sort(ka) if arm == "key_only" else np.sort(aa) if arm == "aux_only" else union
            assert len(candidates) <= 200 and candidates.max() <= q - 30
            hashes[arm].update(np.asarray([q, len(candidates)], dtype="<i8").tobytes())
            hashes[arm].update(candidates.astype("<i8").tobytes())
            if arm == "no_alignment":
                qq = np.full(len(candidates), q, np.int64)
                raw_spec = correlation_profiles(bank, qq, candidates, "c_spec")[:, 0]
                raw_col = correlation_profiles(bank, qq, candidates, "c_col")[:, 0]
            else:
                raw_spec = align_scores(hfft, hnorm, q, candidates)
                raw_col = column_align_scores(column, q, candidates)
            score = (weights[0] * minmax(key[candidates] @ key[q])
                     + weights[1] * minmax(hkey[candidates] @ hkey[q])
                     + weights[2] * minmax(raw_spec) + weights[3] * minmax(raw_col))
            assert score.dtype == np.float32
            order = np.argsort(-score, kind="stable")
            best, j = int(order[0]), int(candidates[order[0]])
            distances = np.linalg.norm(positions[candidates] - positions[q], axis=1)
            values = dict(j_star=j, top1_geo_dist=distances[best], c_spec=raw_spec[best],
                          c_col=raw_col[best], score_abs=float(key[q] @ key[j]),
                          score_margin=float(score[best] - score[order[1]]) if len(order) > 1 else 0.,
                          n_candidates=len(candidates), union_hit=bool(np.any(distances <= 5)))
            for name, value in values.items():
                output[arm][name][index] = value
        if index and index % 4000 == 0:
            print("QUERIES", index, len(queries), flush=True)
    return {a: (output[a], hashes[a].hexdigest()) for a in arms}


def reference_paths(seed: int, role: str, sequence: str) -> tuple:
    """현재 주모델의 검증/시험 원천 경로를 반환한다."""
    epoch = read(MAIN / "freeze.json")["models"][str(seed)]["epoch"]
    folder = source.root(seed, "octave288") / ("test" if role == "test" else f"epoch{epoch}")
    path = folder / (sequence.replace("/", "_") + ".npz")
    return path, path.with_name("keys_" + path.name), path.with_name("coverage_" + path.name)


def worker(seed: int, role: str, sequence: str, arm: str, epoch: int = 0,
           smoke: bool = False) -> None:
    """한 시퀀스의 실제 키·후보·최종 판정과 접두 입력 검사를 저장한다."""
    spec = protocol()
    active = RETRIEVAL if arm == "retrieval" else (arm,)
    if role == "test":
        frozen = read(OUT / f"freeze_{arm}.json")
        assert frozen["protocol_sha256"] == sha256(OUT / "protocol.json")
        for name, digest in frozen["sources"].items():
            assert sha256(REPO / name) == digest, name
        if arm in LEARNED:
            assert epoch == frozen["models"][str(seed)]["epoch"]
    destinations = {
        a: root(seed, a) / (f"epoch{epoch}" if role == "val" and arm in LEARNED else role)
        / (sequence.replace("/", "_") + ".npz") for a in active}
    if smoke:
        destinations = {a: OUT / "smoke" / f"s{seed}_{a}_{sequence.replace('/', '_')}.npz" for a in (*active, "full_reference")}
    if all(p.with_suffix(".json").exists() for p in destinations.values()):
        for p in destinations.values():
            assert sha256(p) == read(p.with_suffix(".json"))["sha256"]
        return
    started = time.time()
    refpath, keypath, coveragepath = reference_paths(seed, role, sequence)
    info = read(refpath.with_suffix(".json"))
    for path, field in ((refpath, "sha256"), (keypath, "keys_sha256"), (coveragepath, "coverage_sha256")):
        assert sha256(path) == info[field]
    reference, coverage = source.base.exp.arrays(refpath), source.base.exp.arrays(coveragepath)
    _, _, paths, min_past = source.base.exp.old.base.inputs(role, sequence)
    cache = source.base.exp.arrays(paths["cache"])
    with np.load(paths["spec"]) as z:
        np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
        bank = source.base.exp.old.base.bank_for(source.base.exp.OP, z)
    corrected = variant_inputs(white.transform(white.coefficients(paths["spec"]),
                               source.base.exp.arrays(white.OUT / "statistics.npz")["full"]), 8)
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    net = None
    if arm == "retrieval":
        key_archive = source.base.exp.arrays(keypath)
        np.testing.assert_array_equal(key_archive["scan_ids"], cache["scan_ids"])
        key = key_archive["embedding"]
        checkpoint = REPO / spec["main_models"][str(seed)]["checkpoint"]
    else:
        checkpoint = root(seed, arm) / "checkpoints" / f"checkpoint_epoch_{epoch}.pth"
        assert sha256(checkpoint) == read(root(seed, arm) / "training_complete.json")["checkpoints"][checkpoint.name]
        net = model(arm, seed).cuda().eval()
        net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
        net.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(np.load(source.OUT / "remetric_octave288.npy")).float().cuda(), requires_grad=False)
        key = source.make_keys(net, cache["descriptors"].astype(np.float32), cache["poses"])
    queries = reference["query_idx"]
    if smoke:
        queries = queries[np.unique(np.linspace(0, len(queries) - 1, 31, dtype=int))]
    chosen = np.searchsorted(reference["query_idx"], queries)
    weights = [float(w) for w in spec["confidence"][str(seed)]["tuple"]]
    evaluated = (*active, "full_reference") if arm == "retrieval" and role == "val" else active
    values = retrieve(key, bank, cache["poses_gt"][:, :3, 3], min_past, weights, queries, evaluated)
    if "full_reference" in values:
        row, digest = values["full_reference"]
        for name in row:
            np.testing.assert_array_equal(row[name], coverage[name][chosen], err_msg=name)
        assert digest == values["no_alignment"][1]
        channel = spec["confidence"][str(seed)]["decision"]
        recreated = final_scores(bank, corrected, travel, row, channel, False)
        np.testing.assert_array_equal(recreated["original"], reference[source.METHOD][chosen])
        saved_white = source.base.exp.arrays(white.OUT / f"s{seed}" / refpath.name)
        expected = .75 * saved_white["identity"][chosen] + .25 * saved_white["full"][chosen]
        np.testing.assert_array_equal(recreated["score"], expected)
        if not smoke:
            proof = OUT / "reference_reproduction" / f"s{seed}_{sequence.replace('/', '_')}.json"
            proof.parent.mkdir(parents=True, exist_ok=True)
            write_json(proof, {"all_fields_equal": True, "final_score_equal": True,
                              "queries": len(queries), "candidate_sha256": digest,
                              "reference_sha256": sha256(refpath)})
    for current in (evaluated if smoke else active):
        row, digest = values[current]
        row["group"] = reference["group"][chosen]
        np.testing.assert_array_equal(row["query_idx"], queries)
        np.testing.assert_array_equal(row["min_past_gt_dist"], reference["min_past_gt_dist"][chosen])
        assert np.all(row["j_star"] <= queries - 30) and np.all(queries + 15 < len(key))
        channels = ("c_spec", "c_col") if current == "no_alignment" else (spec["confidence"][str(seed)]["decision"],)
        score_values, metrics, prefixes = {}, {}, []
        for channel in channels:
            final = final_scores(bank, corrected, travel, row, channel, current == "no_alignment")
            for name, value in final.items():
                field = name + "_" + channel if current == "no_alignment" else name
                score_values[field] = value
                primary = independent_metrics(row, value)
                other = source.base.exp.metrics(value, row)
                for metric in METRICS:
                    if primary[metric] is not None:
                        np.testing.assert_allclose(primary[metric], other[metric], atol=1e-12, rtol=0)
                metrics[field] = primary
        for at in (len(queries) // 3, 2 * len(queries) // 3):
            q, end = int(queries[at]), int(queries[at] + 16)
            truncated = key[:end]
            key_error = 0.
            if net is not None:
                truncated = source.make_keys(net, cache["descriptors"].astype(np.float32), cache["poses"], end)
                np.testing.assert_allclose(truncated[:q + 1], key[:q + 1], atol=4e-6, rtol=4e-5)
                key_error = float(abs(truncated[:q + 1] - key[:q + 1]).max())
            tiny_bank = source.base.exp.truncate_bank(bank, end)
            tiny_corrected = source.base.exp.truncate_bank(corrected, end)
            tiny, _ = retrieve(truncated, tiny_bank, cache["poses_gt"][:end, :3, 3], min_past[:end],
                               weights, queries[at:at + 1], (current,))[current]
            assert int(tiny["j_star"][0]) == int(row["j_star"][at])
            for channel in channels:
                tiny_scores = final_scores(tiny_bank, tiny_corrected, travel[:end], tiny, channel, current == "no_alignment")
                for name, value in tiny_scores.items():
                    field = name + "_" + channel if current == "no_alignment" else name
                    np.testing.assert_array_equal(value, score_values[field][at:at + 1])
            prefixes.append({"query": q, "key_max_error": key_error, "winner_equal": True, "score_error": 0.})
        path = destinations[current]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **row, **score_values)
        write_json(path.with_suffix(".json"), {
            "sha256": sha256(path), "protocol_sha256": sha256(OUT / "protocol.json"),
            "checkpoint_sha256": sha256(checkpoint), "candidate_sha256": digest,
            "metrics": metrics, "prefixes": prefixes, "seconds": time.time() - started,
            "inputs": {str(p.relative_to(REPO)): sha256(p) for p in (*paths.values(), refpath, keypath, coveragepath)},
            "role": role, "seed": seed, "arm": current, "sequence": sequence,
        })
    print("DONE", seed, arm, role, epoch, sequence, round(time.time() - started, 2), flush=True)


def macro(rows: dict, groups: dict) -> dict:
    """양성 시퀀스와 각 장소 그룹을 균등 평균한다."""
    values = {}
    for name, sequences in groups.items():
        selected = [rows[s] for s in sequences if rows[s]["n_pos"]]
        assert selected, name
        values[name] = {m: float(np.mean([r[m] for r in selected])) for m in METRICS}
    return {"groups": values, "macro": {m: float(np.mean([r[m] for r in values.values()])) for m in METRICS}}


def validation_summary(seed: int, arm: str, folder: str, field: str) -> dict:
    """검증 9의 전체 최종 원장을 재계산한다."""
    spec = protocol()
    rows, count = {}, 0
    for sequence in spec["val"]:
        p = root(seed, arm) / folder / (sequence.replace("/", "_") + ".npz")
        record = read(p.with_suffix(".json"))
        assert sha256(p) == record["sha256"]
        row = source.base.exp.arrays(p)
        rows[sequence] = independent_metrics(row, row[field])
        assert rows[sequence] == record["metrics"][field]
        count += len(row["query_idx"])
    assert count == 99236
    groups = {f: [s for s in rows if s.split("/")[0].startswith(f)] for f in ("kitti", "helipr", "mulran")}
    return {**macro(rows, groups), "sequences": rows, "query_count": count}


def freeze(arm: str) -> dict:
    """대조별 검증 선택과 체크포인트를 시험 실행 전에 고정한다."""
    spec = protocol()
    target = OUT / f"freeze_{arm}.json"
    if target.exists():
        value = read(target)
        for p, h in value["sources"].items():
            assert sha256(REPO / p) == h, p
        return value
    sources, choices, validations = {}, {}, {}
    active = RETRIEVAL if arm == "retrieval" else (arm,)
    for seed in SEEDS:
        validations[str(seed)] = {}
        if arm == "retrieval":
            for current in RETRIEVAL:
                fields = ("score_c_spec", "score_c_col") if current == "no_alignment" else ("score",)
                validations[str(seed)][current] = {field: validation_summary(seed, current, "val", field) for field in fields}
            existing = spec["confidence"][str(seed)]["decision"]
            ranked = sorted(("c_spec", "c_col"), key=lambda c: (
                -round(validations[str(seed)]["no_alignment"]["score_" + c]["macro"]["AP"], 12), c != existing))
            choices[str(seed)] = {"no_alignment_channel": ranked[0]}
            for sequence in spec["val"]:
                p = OUT / "reference_reproduction" / f"s{seed}_{sequence.replace('/', '_')}.json"
                assert read(p)["final_score_equal"] and read(p)["all_fields_equal"]
                sources[str(p.relative_to(REPO))] = sha256(p)
        else:
            for epoch in EPOCHS:
                validations[str(seed)][str(epoch)] = {field: validation_summary(seed, arm, f"epoch{epoch}", field) for field in ("original", "score")}
            best = min(EPOCHS, key=lambda ep: (-round(validations[str(seed)][str(ep)]["original"]["macro"]["AP"], 12), ep))
            p = root(seed, arm) / "checkpoints" / f"checkpoint_epoch_{best}.pth"
            choices[str(seed)] = {"epoch": best, "checkpoint": str(p.relative_to(REPO)), "sha256": sha256(p)}
            sources[str(p.relative_to(REPO))] = sha256(p)
        for current in active:
            folders = ("val",) if arm == "retrieval" else tuple(f"epoch{ep}" for ep in EPOCHS)
            for folder in folders:
                for sequence in spec["val"]:
                    p = root(seed, current) / folder / (sequence.replace("/", "_") + ".npz")
                    for item in (p, p.with_suffix(".json")):
                        sources[str(item.relative_to(REPO))] = sha256(item)
    value = {"created_unix": time.time(), "arm": arm, "models": choices,
             "validation": validations, "sources": sources,
             "protocol_sha256": sha256(OUT / "protocol.json"), "test_used_for_selection": False}
    write_json(target, value)
    return value


def finish(arm: str) -> None:
    """검산을 마친 3시드 행만 표 연결용 요약에 넣는다."""
    spec, frozen = protocol(), freeze(arm)
    active = RETRIEVAL if arm == "retrieval" else (arm,)
    result = {"created_unix": time.time(), "protocol_sha256": sha256(OUT / "protocol.json"),
              "freeze_sha256": sha256(OUT / f"freeze_{arm}.json"), "arms": {}, "sources": {}}
    for current in active:
        per_seed, per_sequence, count = {}, {}, {}
        for seed in SEEDS:
            rows, total = {}, 0
            field = "score_" + frozen["models"][str(seed)]["no_alignment_channel"] if current == "no_alignment" else "score"
            for sequence in spec["test"]:
                p = root(seed, current) / "test" / (sequence.replace("/", "_") + ".npz")
                record = read(p.with_suffix(".json"))
                assert sha256(p) == record["sha256"]
                assert record["protocol_sha256"] == sha256(OUT / "protocol.json")
                row = source.base.exp.arrays(p)
                reference = source.base.exp.arrays(reference_paths(seed, "test", sequence)[0])
                for name in ("query_idx", "min_past_gt_dist", "group"):
                    np.testing.assert_array_equal(row[name], reference[name])
                rows[sequence] = independent_metrics(row, row[field])
                assert rows[sequence] == record["metrics"][field]
                assert all(r["winner_equal"] and r["score_error"] == 0 for r in record["prefixes"])
                total += len(row["query_idx"])
                result["sources"][str(p.relative_to(REPO))] = sha256(p)
            assert total == 195726 and sum(bool(r["n_pos"]) for r in rows.values()) == 45
            count[str(seed)] = total
            per_sequence[str(seed)] = rows
            per_seed[str(seed)] = macro(rows, spec["cities"])
        stats = {m: {"mean": float(np.mean([r["macro"][m] for r in per_seed.values()])),
                     "std": float(np.std([r["macro"][m] for r in per_seed.values()], ddof=1))} for m in METRICS}
        result["arms"][current] = {"mean_std": stats, "per_seed": per_seed,
                                  "per_sequence": per_sequence, "query_counts": count,
                                  "final_metrics": True, "checks_passed": True}
    result["checks_passed"] = True
    write_json(OUT / f"summary_{arm}.json", result)
    print("VERIFIED", arm, {a: r["mean_std"] for a, r in result["arms"].items()}, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "eval", "freeze", "finish", "smoke"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--arm", choices=("retrieval", *LEARNED), default="retrieval")
    parser.add_argument("--role", choices=("val", "test"), default="val")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--sequence", default="kitti360/0004")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "train":
        train(args.seed, args.arm)
    elif args.action in ("eval", "smoke"):
        worker(args.seed, args.role, args.sequence, args.arm, args.epoch, args.action == "smoke")
    elif args.action == "freeze":
        freeze(args.arm)
    else:
        finish(args.arm)
