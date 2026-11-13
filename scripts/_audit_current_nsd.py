#!/usr/bin/env python3
"""현재 원고의 NSD 실행 구조와 저장된 질의 결과를 검증한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python3 scripts/_audit_current_nsd.py --seed 1 --seq kitti/09 --role val

기존 결과와 모델은 읽기만 한다. 감사 결과는 별도 폴더에 저장한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]

from _causal_branch_fusion import (  # noqa: E402
    VARIANTS, make_model_causal, nsd_embedding_causal,
)
from _causal_pr_dump import dump_op  # noqa: E402
from _pdv2_pilot import TEST, VAL  # noqa: E402
from _xs_eval_lib import (  # noqa: E402
    align_scores, hspec_inputs, variant_inputs,
)


def local_path(value: str) -> Path:
    """결과 메타데이터의 컨테이너 경로를 현재 리포로 바꾼다."""
    return REPO / value.split("/Neural-Spectral-Codec/", 1)[1]


def sha256(path: Path) -> str:
    """파일 전체의 SHA-256을 계산한다."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_roster() -> dict:
    """주표 출처 해시와 330개 검증·시험 ledger의 실행 경로를 확인한다."""
    manifest = json.loads((REPO / "docs/paper/body_v2.2/tables_v2/source_manifest.json").read_text())
    source_checks = {item["path"]: sha256(REPO / item["path"]) == item["sha256"]
                     for item in manifest["sources"]}
    assert all(source_checks.values()), source_checks
    checked = 0
    for seed in (1, 2, 3):
        root = REPO / f"results/pdv2_c3_bevclip_s{seed}/odom"
        selected = json.loads((root / "causal_tuple_selection.json").read_text())["selections"]
        for role, roster in (("val", VAL), ("test", TEST)):
            for sensor, seq in roster:
                for op in ("pbev128_main", "hspec12"):
                    prefix = "valdump" if role == "val" else "dump"
                    path = root / f"{prefix}_pdv2_causal_{op}_s{seed}_{sensor}_{seq}.npz"
                    with np.load(path) as z:
                        meta = json.loads(str(z["meta"]))
                        assert meta["seed"] == seed and meta["pool_k"] == 100
                        assert meta["min_index_gap"] == 30
                        assert meta["tuple"] == selected[f"pdv2_causal/{op}"]["tuple"]
                        assert np.array_equal(z["query_idx"], np.arange(30, meta["n_frames"]))
                        assert np.all(z["j_star"] <= z["query_idx"] - 30)
                    for value in {**meta["artifacts"], **{k: v for k, v in meta["caches"].items() if k != "marker"}}.values():
                        assert local_path(value).is_file(), value
                    assert f"pdv2_causal_s{seed}/best_model.pth" in meta["artifacts"]["checkpoint"]
                    assert "preprocessed_v2_odom_bevclip/" in meta["caches"]["spec"]
                    if sensor == "nclt":
                        assert "/nclt_fov/odom/" in meta["caches"]["cache"]
                    else:
                        assert "/preprocessed_v2_odom/" in meta["caches"]["cache"]
                    checked += 1
    return {"checked_ledgers": checked, "paper_source_hashes": source_checks,
            "note": "메타데이터 경로·현재 파일 존재·주표 원천 해시 확인. 생성 당시 모델 해시는 ledger에 없다."}


def main() -> None:
    """실제 체크포인트와 입력으로 두 번 통과·최종 매칭을 재현한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--role", choices=("val", "test"), required=True)
    parser.add_argument("--audit-roster", action="store_true")
    parser.add_argument("--out", type=Path, default=REPO / "results/nsd_structure_audit_20261013")
    args = parser.parse_args()
    torch.set_num_threads(2)
    sensor, sequence = args.seq.split("/")
    assert (sensor, sequence) in (VAL if args.role == "val" else TEST)
    root = REPO / f"results/pdv2_c3_bevclip_s{args.seed}/odom"
    prefix = "valdump" if args.role == "val" else "dump"
    stem = f"{prefix}_pdv2_causal_pbev128_main_s{args.seed}_{sensor}_{sequence}.npz"
    with np.load(root / stem) as z:
        meta = json.loads(str(z["meta"]))
    paths = {k: local_path(v) for k, v in meta["artifacts"].items()}
    cache_paths = {k: local_path(v) for k, v in meta["caches"].items() if k != "marker"}
    VARIANTS["audit_current"] = {"ckpt_dir": f"pdv2_causal_s{args.seed}",
                                 "rvec": paths["rvec"], "cls": paths["classifier"],
                                 "config": paths["config"]}
    cfg, model, clf, blob, weights, checkpoint = make_model_causal("audit_current", args.seed, "cuda")
    assert checkpoint == paths["checkpoint"]
    core = model.gnn if hasattr(model, "gnn") else model
    checkpoint_blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_blob["model_state_dict"]
    missing = sorted(set(model.state_dict()) - set(state))
    unexpected = sorted(set(state) - set(model.state_dict()))
    assert missing == ["gnn.key_scale"] and not unexpected, (missing, unexpected)
    for key, value in state.items():
        assert torch.equal(value, model.state_dict()[key].cpu()), key
    np.testing.assert_array_equal(core.key_scale.detach().cpu().numpy(), weights)
    assert core.input_dim == 288 and core.context_dim == 128
    assert core.spectral_policy is None and core.use_residual_gate
    assert len(core.convs) == 2 and all(c.value_source == "diff" for c in core.convs)
    assert all(c.heads == 4 and c.head_dim == 32 for c in core.convs)
    inactive = {key: getattr(core, key, None) is None for key in
                ("phase_projector", "phase_edge_bias", "phase_alignment_bias", "edge_gate")}
    assert all(inactive.values()), inactive
    cache = np.load(cache_paths["cache"])
    layout = np.load(cache_paths["layouts"])
    spectrum = np.load(cache_paths["spec"])
    for z in (layout, spectrum):
        np.testing.assert_array_equal(cache["scan_ids"], z["scan_ids"])
    descriptors = cache["descriptors"].astype(np.float32)
    layouts = layout["nsd_layouts"].astype(np.float32)
    weighted = descriptors * weights
    weighted /= np.maximum(np.linalg.norm(weighted, axis=1, keepdims=True), 1e-8)
    traces = []

    def trace_pass(module, inputs, output):
        graph = inputs[0]
        np.testing.assert_array_equal(graph.x.cpu().numpy(), descriptors)
        edge_index = graph.edge_index.cpu().numpy()
        edge_type = graph.edge_type.cpu().numpy()
        assert np.all(edge_index[0] < edge_index[1])
        edges = {}
        for kind, name in ((0, "temporal"), (1, "similarity")):
            ei = edge_index[:, edge_type == kind]
            lag = ei[1] - ei[0]
            degree = np.bincount(ei[1], minlength=len(descriptors))
            edges[name] = {"n": int(len(lag)), "max_in_degree": int(degree.max()),
                           "lag_min": int(lag.min()) if len(lag) else None,
                           "lag_max": int(lag.max()) if len(lag) else None}
            assert degree.max() <= 10
            if kind == 0:
                assert np.all((lag >= 1) & (lag <= 10))
            elif len(lag):
                assert lag.min() >= (11 if not traces else 30)
        emitted = output.detach().cpu().numpy()
        np.testing.assert_allclose(emitted[:, :288], weighted, rtol=2e-6, atol=2e-7)
        alpha = core._last_alpha.cpu().numpy().ravel()
        np.testing.assert_allclose(np.linalg.norm(emitted[:, 288:], axis=1), alpha,
                                   rtol=2e-6, atol=2e-7)
        traces.append({"input_shape": list(graph.x.shape), "output_shape": list(output.shape),
                       "edges": edges, "gate_min_median_max": np.quantile(alpha, [0, .5, 1]).tolist(),
                       "raw_block_matches_normalized_weighted_key": True})

    handle = model.register_forward_hook(trace_pass)
    embedding = nsd_embedding_causal(cache, sensor, descriptors, layouts,
                                     cfg, model, clf, blob, weights, "cuda")
    handle.remove()
    assert len(traces) == 2
    assert embedding.shape == (len(descriptors), 416)
    np.testing.assert_allclose(np.linalg.norm(embedding, axis=1), 1, atol=2e-7)
    positions = cache["poses_gt"][:, :3, 3]
    minimum = np.full(len(embedding), np.nan)
    for query in range(30, len(embedding)):
        minimum[query] = np.linalg.norm(positions[:query - 29] - positions[query], axis=1).min()
    confidence = json.loads((root / "confidence_selection_align.json").read_text())["ops"]
    replay = {}
    for op in ("pbev128_main", "hspec12"):
        bank = (variant_inputs(spectrum["sk_re"].astype(np.float32)
                               + 1j * spectrum["sk_im"].astype(np.float32), 8)
                if op == "pbev128_main" else hspec_inputs(spectrum["raw_height"], 12))
        source = root / f"{prefix}_pdv2_causal_{op}_s{args.seed}_{sensor}_{sequence}.npz"
        old = np.load(source)
        op_meta = json.loads(str(old["meta"]))
        print(f"REPLAY {args.seq} seed={args.seed} op={op}", flush=True)
        new = dump_op(embedding, *bank, positions, minimum, op_meta["tuple"], 100)
        np.testing.assert_array_equal(new["query_idx"], old["query_idx"])
        np.testing.assert_allclose(new["min_past_gt_dist"], old["min_past_gt_dist"], atol=1e-9)
        changed = new["j_star"] != old["j_star"]
        old_align = np.load(source.with_name("align_" + source.name))
        align = np.array([align_scores(bank[1], bank[2], int(q), np.array([int(j)]))[0]
                          for q, j in zip(new["query_idx"], new["j_star"])])
        replay[op] = {"n_queries": len(changed), "changed_top1": int(changed.sum()),
                       "changed_correctness": int(np.count_nonzero((new["top1_geo_dist"] <= 5)
                                                                   != (old["top1_geo_dist"] <= 5))),
                       "max_key_cosine_diff": float(np.max(np.abs(new["score_abs"] - old["score_abs"]))),
                       "max_spectral_score_diff": float(np.max(np.abs(align - old_align["c_spec"]))),
                       "changed_candidate_counts": int(np.count_nonzero(new["n_candidates"] != old["n_candidates"])),
                       "fusion_weights": op_meta["tuple"], "decision_score": confidence[op]["decision"],
                       "decision_threshold": confidence[op]["tau_star"],
                       "auxiliary_magnitude_shape": list(bank[0].shape),
                       "source_ledger_sha256": sha256(source)}
    report = {"seed": args.seed, "sequence": args.seq, "role": args.role,
              "checkpoint_epoch_zero_based": checkpoint_blob["epoch"],
              "checkpoint_weights_exactly_loaded": True, "missing_keys": missing,
              "missing_key_explanation": "재계량 벡터는 학습 체크포인트 밖의 고정 artifact에서 추가한다.",
              "unexpected_keys": unexpected, "effective_config": cfg,
              "model": str(model), "classifier": str(clf),
              "trained_refiner_parameters": sum(p.numel() for n, p in model.named_parameters()
                                                 if p.requires_grad and not n.endswith("key_scale")),
              "classifier_parameters": sum(p.numel() for p in clf.parameters()),
              "inactive_modules": inactive, "passes": traces, "replay": replay,
              "artifacts": {k: {"path": str(p.relative_to(REPO)), "sha256": sha256(p)}
                            for k, p in paths.items()},
              "caches": {k: str(p.relative_to(REPO)) for k, p in cache_paths.items()},
              "source_hashes": {str(p.relative_to(REPO)): sha256(p) for p in
                                (Path(__file__), REPO / "scripts/_causal_branch_fusion.py",
                                 REPO / "scripts/_causal_edge_features.py", REPO / "src/gnn/model.py",
                                 REPO / "scripts/_causal_pr_dump.py", REPO / "scripts/_xs_eval_lib.py")}}
    if args.audit_roster:
        report["roster"] = audit_roster()
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"s{args.seed}_{sensor}_{sequence}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"report": str(out), "parameters": report["trained_refiner_parameters"],
                      "passes": traces, "replay": replay}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
