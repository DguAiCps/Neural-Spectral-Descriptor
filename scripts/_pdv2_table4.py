#!/usr/bin/env python3
"""표 4 빈 대조군의 최종 매칭·판정 원장과 세 시드 집계를 만든다."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_no_context_ablation import OPS, sha256, write_json
from _pdv2_lag15_campaign import OUT as BASE, CONFIG, RVEC, bank_for, inputs, STATS
from _pdv2_pilot import CITY, FAMILY, TRAIN, VAL, TEST
from _causal_pr_dump import prefix_topk
from _xs_eval_lib import align_scores, column_align_scores, minmax, normalize
from _pr_eval_lib import AGG_METRICS, classify_queries, operating_point, pr_sweep

OUT = REPO / "results/pdv2_table4_20261014"
SEEDS = (1, 2, 3)
VARIANTS = ("graph_free", "key_only", "aux_only", "no_alignment")
RETRIEVAL = VARIANTS[1:]
METRICS = (*AGG_METRICS, "R@1", "P_at_tau", "R_at_tau")


def read(path: Path) -> dict:
    """JSON 파일을 읽는다."""
    return json.loads(path.read_text())


def seed_dir(seed: int) -> Path:
    """시드별 새 출력 폴더를 만든다."""
    path = OUT / f"s{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ledger(seed: int, role: str, variant: str, op: str, sensor: str, seq: str) -> Path:
    """대조군 원장 경로를 반환한다."""
    return seed_dir(seed) / f"{role}_{variant}_{op}_{sensor}_{seq}.npz"


def protocol() -> dict:
    """시험 결과를 만들기 전에 비교 조건과 코드를 고정한다."""
    path = OUT / "protocol.json"
    if path.exists():
        result = read(path)
        for name, digest in result["sha256"].items():
            assert sha256(REPO / name) == digest, name
        return result
    files = [Path(__file__), REPO / "scripts/_pdv2_table4_run.py",
             REPO / "tests/test_table4.py", OUT / "training_protocol.json",
             REPO / "docs/DECOMP_PREREGISTRATION.md", CONFIG, RVEC,
             BASE / "freeze.json", BASE / "summary.json"]
    files.extend(REPO / p for p in ("scripts/_pdv2_lag15_campaign.py",
                 "scripts/_pdv2_no_context_ablation.py", "scripts/_pr_eval_lib.py",
                 "scripts/_xs_eval_lib.py", "scripts/_causal_pr_dump.py",
                 "scripts/_pdv2_pilot.py", "src/gnn/graph_free.py", "src/gnn/model.py"))
    files.extend(BASE / f"s{s}/confidence_selection.json" for s in SEEDS)
    result = {"date": "2026-10-14", "seeds": list(SEEDS), "variants": list(VARIANTS),
              "roster": {r: [s + "/" + q for s, q in rows]
                         for r, rows in (("train", TRAIN), ("val", VAL), ("test", TEST))},
              "queries": "30 <= q < N-15; candidates j <= q-30",
              "primary_metrics": ["AP", "max_F1", "R@100P"],
              "primary_scope": "전체 검색·매칭 뒤 선택된 1등 후보와 허용된 판정 점수. 중간 검색 지표는 표에 쓰지 않는다.",
              "aggregation": "45 positive-containing sequences -> six cities -> three seed means and sample SD",
              "selection": "graph_free/key_only/aux_only는 해당 전체 모델의 검증 선택값 고정. no_alignment는 w_a=w_c=0, w_f,w_m 유지; absolute/margin만 검증 선택.",
              "confidence_rule": "검증 시퀀스 AP -> 센서 계열 평균 -> 계열 균등 평균 최대. absolute가 최고값 0.005 이내면 absolute; 같은 기존 검증 F1 임계값 규칙.",
              "candidate_rule": "key_only 200+0, aux_only 0+200, 나머지 100+100. 중복 제거 후 보충 없음. no_alignment와 full_reference의 후보 해시 일치 확인.",
              "test_guard": "retrieval은 세 시드 검증 재현과 no_alignment 선택 동결 후, graph_free는 세 시드 학습 완료·체크포인트 동결 후 시험.",
              "test_exposure": "기존 주표 시험 결과를 이미 본 추가 실험. 확증·신규 사전 등록으로 주장하지 않는다.",
              "sha256": {str(p.relative_to(REPO)): sha256(p) for p in files}}
    write_json(path, result)
    return result


def guard(group: str) -> dict:
    """시험 전에 선택·모델 동결과 코드 해시를 확인한다."""
    protocol()
    value = read(OUT / f"freeze_{group}.json")
    assert value["protocol_sha256"] == sha256(OUT / "protocol.json")
    for name, digest in value["sha256"].items():
        assert sha256(REPO / name) == digest, name
    return value


def selections(seed: int, variant: str) -> dict:
    """전체 모델 고정 선택 또는 검증에서 고른 비정렬 판정을 읽는다."""
    if variant == "no_alignment" and (seed_dir(seed) / "no_alignment_selection.json").exists():
        return read(seed_dir(seed) / "no_alignment_selection.json")["ops"]
    return read(BASE / f"s{seed}/confidence_selection.json")["ops"]


def evaluate(emb: np.ndarray, bank: tuple, positions: np.ndarray,
             min_past: np.ndarray, weights: tuple, variants: tuple,
             queries: np.ndarray | None = None) -> dict:
    """각 대조군에서 후보 생성부터 최종 후보·수락 점수까지 계산한다."""
    hkey, hfft, hnorm, column = bank
    if queries is None:
        queries = np.arange(30, len(emb) - 15, dtype=np.int64)
    n = len(queries)
    output, digests = {}, {}
    for variant in variants:
        fields = ("top1_geo_dist", "score_abs", "score_margin")
        if variant != "no_alignment":
            fields += ("c_spec", "c_col")
        output[variant] = {name: np.empty(n, np.float64) for name in fields}
        output[variant].update(query_idx=queries.copy(), min_past_gt_dist=min_past[queries],
                               j_star=np.empty(n, np.int64), n_candidates=np.empty(n, np.int64),
                               union_hit=np.empty(n, bool))
        digests[variant] = hashlib.sha256()
    for qi, q in enumerate(queries):
        q = int(q)
        pools = {}
        if any(v in variants for v in ("full_reference", "graph_free", "no_alignment")):
            a = prefix_topk(emb, q, q - 29, 100)
            b = prefix_topk(hkey, q, q - 29, 100)
            pools["union"] = np.unique(np.r_[a, b])
        if "key_only" in variants:
            pools["key_only"] = np.sort(prefix_topk(emb, q, q - 29, 200))
        if "aux_only" in variants:
            pools["aux_only"] = np.sort(prefix_topk(hkey, q, q - 29, 200))
        for variant in variants:
            candidates = pools[variant if variant in ("key_only", "aux_only") else "union"]
            assert 0 < len(candidates) <= 200 and candidates.max() <= q - 30
            digests[variant].update(np.asarray([q, len(candidates)], dtype="<i8").tobytes())
            digests[variant].update(candidates.astype("<i8").tobytes())
            sn, sh = minmax(emb[candidates] @ emb[q]), minmax(hkey[candidates] @ hkey[q])
            score = weights[0] * sn + weights[1] * sh
            if variant != "no_alignment":
                ra = align_scores(hfft, hnorm, q, candidates)
                rc = column_align_scores(column, q, candidates)
                score = score + weights[2] * minmax(ra) + weights[3] * minmax(rc)
            assert score.dtype == np.float32
            order = np.argsort(-score, kind="stable")
            best = int(order[0])
            j = int(candidates[best])
            distances = np.linalg.norm(positions[candidates] - positions[q], axis=1)
            values = {"j_star": j, "top1_geo_dist": distances[best],
                      "n_candidates": len(candidates), "union_hit": bool(np.any(distances <= 5)),
                      "score_abs": float(emb[q] @ emb[j]),
                      "score_margin": float(score[best] - score[order[1]]) if len(order) > 1 else 0.}
            if variant != "no_alignment":
                values.update(c_spec=ra[best], c_col=rc[best])
            for name, value in values.items():
                output[variant][name][qi] = value
        if qi and qi % 4000 == 0:
            print("QUERIES", qi, n, flush=True)
    return {v: (a, digests[v].hexdigest()) for v, a in output.items()}


def graph_free_keys(seed: int, raw: np.ndarray) -> np.ndarray:
    """검증에서 선택된 노드별 투영에 동일 재계량을 적용한다."""
    import torch
    from torch_geometric.data import Data
    from gnn.graph_free import create_graph_free_model
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = create_graph_free_model(input_dim=288, hidden_dim=128, context_dim=128,
                                    n_layers=2, n_heads=4, dropout=0.1, norm_type="batch_norm",
                                    use_residual_gate=True, gate_hidden_dim=64, gate_initial_alpha=0.0625,
                                    key_remetrize_config={"enabled": True, "learnable": False,
                                                          "init_path": str(RVEC)}).to("cuda")
    ckpt = REPO / f"checkpoints/pdv2_graph_free_s{seed}/best_model.pth"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    missing, extra = model.load_state_dict(state, strict=False)
    assert missing == ["gnn.key_scale"] and not extra, (missing, extra)
    model.eval()
    graph = Data(x=torch.from_numpy(raw).to("cuda"),
                 edge_index=torch.empty((2, 0), dtype=torch.long, device="cuda"))
    with torch.no_grad():
        result = normalize(model(graph).float().cpu().numpy())
        cut = Data(x=graph.x[:257], edge_index=graph.edge_index)
        prefix = normalize(model(cut).float().cpu().numpy())
    np.testing.assert_allclose(prefix, result[:257], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(normalize(result[:, :288]), normalize(raw * np.load(RVEC)), atol=3e-7)
    assert result.shape == (len(raw), 416) and np.isfinite(result).all()
    return result


def worker(seed: int, role: str, sequence: str, group: str) -> None:
    """한 시퀀스의 두 운영점 원장을 생성하고 기준 결과를 재현한다."""
    protocol()
    if role == "test":
        guard(group)
    sensor, seq, paths, min_past = inputs(role, sequence)
    variants = RETRIEVAL if group == "retrieval" else ("graph_free",)
    all_paths = [ledger(seed, role, v, op, sensor, seq) for v in variants for op in OPS]
    if all(p.exists() for p in all_paths):
        for p in all_paths:
            with np.load(p) as z:
                assert json.loads(str(z["meta"]))["protocol_sha256"] == sha256(OUT / "protocol.json")
        return
    started = time.time()
    key_path = BASE / f"s{seed}/keys_{role}_{sensor}_{seq}.npz"
    with np.load(paths["cache"]) as cache, np.load(paths["spec"]) as spectrum, np.load(key_path) as keys:
        np.testing.assert_array_equal(cache["scan_ids"], spectrum["scan_ids"])
        np.testing.assert_array_equal(cache["scan_ids"], keys["scan_ids"])
        positions = cache["poses_gt"][:, :3, 3]
        emb = (keys["symmetric"] if group == "retrieval" else
               graph_free_keys(seed, cache["descriptors"].astype(np.float32)))
        code_hash = sha256(OUT / "protocol.json")
        hashes = {str(p.relative_to(REPO)): sha256(p) for p in (paths["cache"], paths["spec"], key_path)}
        if group == "graph_free":
            checkpoint = REPO / f"checkpoints/pdv2_graph_free_s{seed}/best_model.pth"
            hashes[str(checkpoint.relative_to(REPO))] = sha256(checkpoint)
        checks = {}
        for op in OPS:
            active = (*variants, "full_reference") if role == "val" and group == "retrieval" else variants
            weights = tuple(float(w) for w in selections(seed, "key_only")[op]["tuple"])
            evaluated = evaluate(emb, bank_for(op, spectrum), positions, min_past, weights, active)
            if "full_reference" in evaluated:
                reproduced, digest = evaluated["full_reference"]
                reference = BASE / f"s{seed}/val_selected_symmetric_{op}_{sensor}_{seq}.npz"
                with np.load(reference) as old:
                    for name, a in reproduced.items():
                        np.testing.assert_allclose(a, old[name], rtol=0, atol=1e-12, err_msg=name)
                assert digest == evaluated["no_alignment"][1]
                checks[op] = {"reference_sha256": sha256(reference), "query_count": len(reproduced["query_idx"]),
                              "candidate_sha256": digest, "all_fields_equal": True}
            for variant in variants:
                a, digest = evaluated[variant]
                assert np.all(a["j_star"] <= a["query_idx"] - 30)
                assert all(np.isfinite(v).all() for v in a.values())
                meta = {"seed": seed, "role": role, "variant": variant, "op": op, "sequence": sequence,
                        "protocol_sha256": code_hash, "input_sha256": hashes, "candidate_sha256": digest,
                        "tuple": list(weights[:2]) + [0., 0.] if variant == "no_alignment" else list(weights),
                        "runtime_s": time.time() - started}
                if role == "test":
                    meta["freeze_sha256"] = sha256(OUT / f"freeze_{group}.json")
                dest = ledger(seed, role, variant, op, sensor, seq)
                tmp = dest.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, **a, meta=json.dumps(meta))
                tmp.replace(dest)
                print("LEDGER_DONE", seed, role, variant, op, sequence, flush=True)
        if checks:
            write_json(seed_dir(seed) / f"reproduction_{sensor}_{seq}.json", checks)


def select_confidence(seed: int) -> None:
    """정렬을 제외한 두 판정 점수만 검증셋에서 비교한다."""
    protocol()
    from _pdv2_c3 import _family_macro_tau_star
    target = seed_dir(seed) / "no_alignment_selection.json"
    if target.exists():
        assert read(target)["protocol_sha256"] == sha256(OUT / "protocol.json")
        return
    chosen = {}
    for op in OPS:
        groups, sweeps = {}, {}
        for sensor, seq in VAL:
            name = sensor + "/" + seq
            with np.load(ledger(seed, "val", "no_alignment", op, sensor, seq)) as z:
                assert "c_spec" not in z and "c_col" not in z
                qc = classify_queries(z["min_past_gt_dist"])
                if not qc["n_pos"]:
                    continue
                groups.setdefault(FAMILY.get(sensor, sensor), []).append(name)
                sweeps[name] = {stat: pr_sweep(z[STATS[stat]], z["top1_geo_dist"], qc)
                                for stat in ("absolute", "margin")}
        ap = {stat: statistics.mean(statistics.mean(sweeps[n][stat]["AP"] for n in names)
                                   for names in groups.values()) for stat in ("absolute", "margin")}
        decision = "absolute" if ap["absolute"] >= max(ap.values()) - .005 else "margin"
        tau, f1, _ = _family_macro_tau_star({n: row[decision] for n, row in sweeps.items()}, groups)
        w = selections(seed, "key_only")[op]["tuple"]
        chosen[op] = {"tuple": [*w[:2], 0., 0.], "decision": decision, "tau_star": tau,
                      "family_macro_AP": ap, "val_family_macro_F1": f1}
    write_json(target, {"ops": chosen, "protocol_sha256": sha256(OUT / "protocol.json")})


def freeze(group: str) -> None:
    """세 시드의 검증 또는 학습 완료를 확인하고 시험 설정을 고정한다."""
    protocol()
    target = OUT / f"freeze_{group}.json"
    if target.exists():
        guard(group)
        return
    files = []
    for seed in SEEDS:
        files.append(BASE / f"s{seed}/confidence_selection.json")
        if group == "retrieval":
            files.append(seed_dir(seed) / "no_alignment_selection.json")
            for sensor, seq in VAL:
                reproduction = seed_dir(seed) / f"reproduction_{sensor}_{seq}.json"
                assert all(r["all_fields_equal"] for r in read(reproduction).values())
                files.append(reproduction)
                files.extend(ledger(seed, "val", v, op, sensor, seq) for v in RETRIEVAL for op in OPS)
        else:
            marker = OUT / f"training_s{seed}_complete.json"
            ckpt = REPO / f"checkpoints/pdv2_graph_free_s{seed}/best_model.pth"
            assert read(marker)["checkpoint_sha256"] == sha256(ckpt)
            files.extend((marker, ckpt))
    write_json(target, {"created_unix": time.time(), "protocol_sha256": sha256(OUT / "protocol.json"),
                        "sha256": {str(p.relative_to(REPO)): sha256(p) for p in files}})


def summarize(group: str) -> None:
    """최종 후보의 판정 성능을 같은 도시·시드 규칙으로 집계한다."""
    guard(group)
    variants = RETRIEVAL if group == "retrieval" else ("graph_free",)
    result = {"protocol_sha256": sha256(OUT / "protocol.json"), "final_system_metrics": True, "ops": {}}
    for op in OPS:
        result["ops"][op] = {}
        for variant in variants:
            seeds = {}
            for seed in SEEDS:
                selection = selections(seed, variant)[op]
                per, candidate_counts = {}, []
                for sensor, seq in TEST:
                    with np.load(ledger(seed, "test", variant, op, sensor, seq)) as z:
                        meta = json.loads(str(z["meta"]))
                        assert meta["freeze_sha256"] == sha256(OUT / f"freeze_{group}.json")
                        assert meta["protocol_sha256"] == result["protocol_sha256"]
                        for p, digest in meta["input_sha256"].items():
                            assert sha256(REPO / p) == digest, p
                        ref = BASE / f"s{seed}/test_selected_symmetric_{op}_{sensor}_{seq}.npz"
                        with np.load(ref) as old:
                            for k in ("query_idx", "min_past_gt_dist"):
                                np.testing.assert_array_equal(z[k], old[k])
                            if variant == "no_alignment":
                                np.testing.assert_array_equal(z["n_candidates"], old["n_candidates"])
                        assert np.all(z["j_star"] <= z["query_idx"] - 30)
                        qc = classify_queries(z["min_past_gt_dist"])
                        score = z[STATS[selection["decision"]]]
                        row = {m: None for m in METRICS}
                        if qc["n_pos"]:
                            sweep = pr_sweep(score, z["top1_geo_dist"], qc)
                            threshold = operating_point(score, z["top1_geo_dist"], qc, selection["tau_star"])
                            row.update({m: float(sweep[m]) for m in AGG_METRICS})
                            row.update({"R@1": float((z["top1_geo_dist"][qc["positive"]] <= 5).mean()),
                                        "P_at_tau": float(threshold["precision"]), "R_at_tau": float(threshold["recall"])})
                        row.update(n_queries=qc["n_total"], n_positive=qc["n_pos"], n_negative=qc["n_neg"],
                                   negative_false_accepts=int((qc["negative"] & (score >= selection["tau_star"])).sum()))
                        candidate_counts.extend(z["n_candidates"].tolist())
                        per[sensor + "/" + seq] = row
                assert sum(r["n_queries"] for r in per.values()) == 195726
                assert sum(r["n_positive"] for r in per.values()) == 61682
                assert [s for s, r in per.items() if not r["n_positive"]] == ["kitti/07"]
                cities = {city: {m: statistics.mean(per[s][m] for s in names if per[s]["n_positive"])
                                 for m in METRICS} for city, names in CITY.items()}
                seeds[str(seed)] = {"per_sequence": per, "city_groups": cities, "selection": selection,
                                    "macro": {m: statistics.mean(r[m] for r in cities.values()) for m in METRICS},
                                    "candidate_count_mean": float(np.mean(candidate_counts)),
                                    "candidate_count_p95": float(np.quantile(candidate_counts, .95))}
            means = {}
            for m in METRICS:
                values = [r["macro"][m] for r in seeds.values()]
                means[m] = {"mean": statistics.mean(values), "std": statistics.stdev(values), "values": values}
            result["ops"][op][variant] = {"per_seed": seeds, "mean_std": means}
    write_json(OUT / f"summary_{group}.json", result)
    if all((OUT / f"summary_{g}.json").exists() for g in ("retrieval", "graph_free")):
        merged = {"protocol_sha256": result["protocol_sha256"], "final_system_metrics": True,
                  "base_summary_sha256": sha256(BASE / "summary.json"), "ops": {op: {} for op in OPS}}
        for g in ("retrieval", "graph_free"):
            source = read(OUT / f"summary_{g}.json")
            assert source["protocol_sha256"] == result["protocol_sha256"]
            for op in OPS:
                merged["ops"][op].update(source["ops"][op])
        write_json(OUT / "summary.json", merged)


def main() -> None:
    """검증·동결·시험·집계를 분리한 명령을 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("protocol", "worker", "select", "freeze", "report"))
    parser.add_argument("--group", choices=("retrieval", "graph_free"), default="retrieval")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=1)
    parser.add_argument("--role", choices=("val", "test"), default="val")
    parser.add_argument("--seq")
    args = parser.parse_args()
    if args.action == "protocol": protocol()
    elif args.action == "worker": worker(args.seed, args.role, args.seq, args.group)
    elif args.action == "select": select_confidence(args.seed)
    elif args.action == "freeze": freeze(args.group)
    elif args.action == "report": summarize(args.group)


if __name__ == "__main__":
    main()
