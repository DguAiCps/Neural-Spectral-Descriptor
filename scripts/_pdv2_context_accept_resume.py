#!/usr/bin/env python3
"""FP32 접두 검사 실패를 보존하고 FP64 가용성 검사를 추가한다."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_context_accept as run
from _pdv2_context_accept_lib import unit
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash


def main() -> None:
    """선택 기준은 유지하고 FP64로 접두 입력의 동일성을 확인한다."""
    torch.set_num_threads(1)
    spec = run.prepare()
    root = run.train_root(3)
    amendment = run.OUT / "numerical_amendment.json"
    if not amendment.exists():
        assert not (run.OUT / "fitted.json").exists()
        assert not (run.OUT / "features/val").exists()
        write_json(amendment, {
            "reason": "시드3FP32전체/접두키일부좌표최대7.2345e-6차이로atol3e-6검사실패. 결과선택전수치원인검사추가.",
            "rule": "원FP32추론유지. 동일전체입력재실행오차와FP64전체·접두를계산. 16개FP64접두최대오차<1e-10,FP32전체/접두<1e-4,FP32/FP64<1e-4필수. 학습·검증조건/가중치적합/선택규칙불변.",
            "script_sha256": sha256(Path(__file__)),
            "original_protocol_sha256": sha256(run.OUT/"protocol.json"),
            "failed_log_sha256": sha256(REPO/"logs/pdv2_context_accept_20261018.log"),
            "test_used": False, "validation_features_computed": False})
    assert sha256(Path(__file__)) == run.exp.read(amendment)["script_sha256"]
    if not (root / "keys_complete.json").exists():
        graph = torch.load(run.previous.GRAPH, map_location="cpu", weights_only=False)
        keep = graph.edge_type == 0
        graph.edge_index, graph.edge_attr, graph.edge_type = graph.edge_index[:, keep], graph.edge_attr[keep], graph.edge_type[keep]
        if hasattr(graph, "edge_pose_label"):
            graph.edge_pose_label = graph.edge_pose_label[keep]
        hashes = {k: array_hash(getattr(graph, k).numpy()) for k in ("x", "edge_index", "edge_attr", "edge_type")}
        assert hashes == run.exp.read(run.exp.old.OUT/"s3/standard_original/training_graph.json")["hashes"]
        _, model = run.exp.old.load_model(3, "standard_original", spec["seeds"]["3"]["epoch"])
        graph = graph.to("cuda")
        with torch.no_grad():
            key = unit(model(graph).float().cpu().numpy())
            repeat = unit(model(graph).float().cpu().numpy())
        repeat_error = float(np.max(np.abs(key-repeat)))
        assert repeat_error < 1e-4
        meta = run.exp.arrays(run.PRIOR/"training_metadata.npz")
        queries = []
        for ordinal in range(8):
            ids = np.flatnonzero(meta["sequences"] == ordinal)
            queries.extend(int(ids[q]) for q in (len(ids)//3, 2*len(ids)//3))
        float_prefix = []
        for q in queries:
            end = q+16
            mask = (graph.edge_index[0] < end) & (graph.edge_index[1] < end)
            small = Data(x=graph.x[:end], edge_index=graph.edge_index[:, mask], edge_attr=graph.edge_attr[mask], edge_type=graph.edge_type[mask])
            with torch.no_grad():
                value = unit(model(small)[q:q+1].float().cpu().numpy())[0]
            float_prefix.append(value)
        model = model.double()
        graph.x, graph.edge_attr = graph.x.double(), graph.edge_attr.double()
        with torch.no_grad():
            double = unit(model(graph).cpu().numpy())
        checks = []
        for row, q in enumerate(queries):
            end = q+16
            mask = (graph.edge_index[0] < end) & (graph.edge_index[1] < end)
            small = Data(x=graph.x[:end], edge_index=graph.edge_index[:, mask], edge_attr=graph.edge_attr[mask], edge_type=graph.edge_type[mask])
            with torch.no_grad():
                value = unit(model(small)[q:q+1].cpu().numpy())[0]
            error64 = float(np.max(np.abs(value-double[q])))
            error32 = float(np.max(np.abs(float_prefix[row]-key[q])))
            error_precision = float(max(np.max(np.abs(key[q]-double[q])), np.max(np.abs(float_prefix[row]-value))))
            assert error64 < 1e-10 and error32 < 1e-4 and error_precision < 1e-4
            checks.append({"global_query": q, "fp64_prefix_error": error64,
                           "fp32_prefix_error": error32, "fp32_fp64_error": error_precision})
        np.save(root/"retrieval_keys.npy", key)
        shutil.copyfile(run.PRIOR/"training_metadata.npz", root/"training_metadata.npz")
        write_json(root/"keys_complete.json", {"graph_hashes": hashes, "prefix_checks": checks,
            "fp32_repeat_max_error": repeat_error, "fp32_fp64_all_max_error": float(np.max(np.abs(key-double))),
            "amendment_sha256": sha256(amendment),
            "outputs": {p.name: sha256(p) for p in (root/"retrieval_keys.npy", root/"training_metadata.npz")}})
        print("NUMERICAL_CHECK_PASSED", repeat_error, max(c["fp64_prefix_error"] for c in checks), flush=True)
        del model, graph, small
        torch.cuda.empty_cache()
    run.main()


if __name__ == "__main__":
    main()
