#!/usr/bin/env python3
"""고정 손실의 기울기를 모델 파라미터로 전달하는 국소 방향 검사다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
from _pdv2_train_failure import OUT, GRAPH, arrays, read, distribution
from _pdv2_no_context_ablation import sha256, write_json
import _pdv2_temporal_focus as focus


def pair_cosine(raw: np.ndarray, q: np.ndarray, j: np.ndarray, rvec: np.ndarray) -> np.ndarray:
    """고정 재계량을 적용한 검색 키의 쌍별 코사인을 고정밀로 구한다."""
    output = raw.astype(np.float64)
    output[:, :288] *= rvec[None]
    output[:, :288] /= np.linalg.norm(output[:, :288], axis=1, keepdims=True)
    output /= np.linalg.norm(output, axis=1, keepdims=True)
    return (output[q]*output[j]).sum(1)


def main() -> None:
    """실제 업데이트 없이 기존 손실 하강 방향과 오답 유사도의 국소 반응을 잰다."""
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert read(OUT/"status.json")["status"] == "complete"
    spec = read(OUT/"protocol.json")
    sources = {Path(__file__), GRAPH, OUT/"training_output.npy", OUT/"training_output_gradient.npy",
               OUT/"training_metadata.npz", OUT/"exposure_arrays.npz", OUT/"protocol.json",
               REPO/spec["selected"]["checkpoint"], focus.base.RVEC}
    record = {"sources": {str(p.relative_to(REPO)): sha256(p) for p in sorted(sources)},
              "epsilon": [.01, .005], "scope": "고정 모델 FP32 eval에서 재평가한 10에포크 손실합의 실제 파라미터 기울기. 단위 기울기 방향 중앙 차분, 가중치 저장·최적화·임계값 변경 없음. 과거 SGD/Adam 학습 경로 재현 아님."}
    protocol = OUT/"parameter_protocol.json"
    if protocol.exists():
        assert read(protocol) == record
    else:
        write_json(protocol, record)
    graph = torch.load(GRAPH, map_location="cpu", weights_only=False)
    keep = graph.edge_type == 0
    edge_index, edge_attr, edge_type = graph.edge_index[:, keep], graph.edge_attr[keep], graph.edge_type[keep]
    full = Data(x=graph.x.cuda(), edge_index=edge_index.cuda(), edge_attr=edge_attr.cuda(), edge_type=edge_type.cuda())
    meta = arrays(OUT/"training_metadata.npz")
    original = np.load(OUT/"training_output.npy")
    upstream = np.load(OUT/"training_output_gradient.npy")
    _, model = focus.load_model(1, "standard_original", 10)
    model.gnn.key_scale = None
    model.gnn.use_key_remetrize = False
    model.zero_grad(set_to_none=True)
    maximum_error = 0.
    for ordinal, sequence in enumerate(spec["train_roster"]):
        ids = np.flatnonzero(meta["sequences"] == ordinal)
        mask = (edge_index[0] >= ids[0]) & (edge_index[0] <= ids[-1])
        assert bool(((edge_index[1, mask] >= ids[0]) & (edge_index[1, mask] <= ids[-1])).all())
        local = Data(x=graph.x[ids].cuda(), edge_index=(edge_index[:, mask]-ids[0]).cuda(),
                     edge_attr=edge_attr[mask].cuda(), edge_type=edge_type[mask].cuda())
        output = model(local)
        current = output.detach().cpu().numpy()
        maximum_error = max(maximum_error, float(np.max(np.abs(current-original[ids]))))
        np.testing.assert_allclose(current, original[ids], atol=3e-6, rtol=3e-5)
        output.backward(torch.from_numpy(upstream[ids]).cuda())
        del output, local
        print("PARAMETER_BACKWARD", sequence, flush=True)
    parameters = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
    assert parameters
    norm = float(torch.sqrt(sum(p.grad.double().square().sum() for _, p in parameters)))
    assert np.isfinite(norm) and norm > 0
    directions = {name: p.grad.detach().clone()/norm for name, p in parameters}
    initial = {name: p.detach().clone() for name, p in parameters}
    norms = {name: float(p.grad.norm()) for name, p in parameters}
    a = arrays(OUT/"exposure_arrays.npz")
    q = np.flatnonzero(a["cohort"] >= 0)
    j = a["winner"][q]
    rvec = np.load(focus.base.RVEC)
    with torch.no_grad():
        baseline = model(full).float().cpu().numpy()
        repeat = model(full).float().cpu().numpy()
    noise = np.abs(pair_cosine(baseline, q, j, rvec)-pair_cosine(repeat, q, j, rvec))
    responses, linear_checks = [], []
    for epsilon in record["epsilon"]:
        endpoints = []
        for sign in (-1, 1):
            with torch.no_grad():
                for name, p in parameters:
                    p.copy_(initial[name]+sign*epsilon*directions[name])
                endpoints.append(model(full).float().cpu().numpy())
        minus, plus = endpoints
        response = (pair_cosine(minus, q, j, rvec)-pair_cosine(plus, q, j, rvec))/(2*epsilon)
        responses.append(response)
        derivative = float(np.sum((plus.astype(np.float64)-minus)*upstream)/(2*epsilon))
        relative_error = abs(derivative-norm)/norm
        assert relative_error < .05, (epsilon, derivative, norm)
        linear_checks.append({"epsilon": epsilon, "finite_derivative": derivative,
                              "autograd_expected": norm, "relative_error": relative_error})
    with torch.no_grad():
        for name, p in parameters:
            p.copy_(initial[name])
        restored = model(full).float().cpu().numpy()
    np.testing.assert_allclose(restored, baseline, atol=3e-6, rtol=3e-5)
    # Two step sizes must agree beyond their observed discrepancy and repeated-forward noise.
    error = np.abs(responses[0]-responses[1]) + 4*noise/min(record["epsilon"]) + 1e-7
    stable_down = (responses[0] < -error) & (responses[1] < -error)
    stable_up = (responses[0] > error) & (responses[1] > error)
    results = {}
    groups = {"all": np.ones(len(q), bool), "kitti": meta["sequences"][q] < 4, "helipr": meta["sequences"][q] >= 4}
    for group, selected in groups.items():
        results[group] = {}
        for name, cohort in (("false_accept", 1), ("negative_reject", 0), ("positive_correct_accept", 2)):
            mask = selected & (a["cohort"][q] == cohort)
            results[group][name] = {"n": int(mask.sum()), "stable_decrease": int((mask & stable_down).sum()),
                "stable_increase": int((mask & stable_up).sum()), "uncertain": int((mask & ~stable_down & ~stable_up).sum()),
                "response": distribution(responses[1][mask])}
    path = OUT/"parameter_response.npz"
    np.savez_compressed(path, query=q, winner=j, response_eps01=responses[0], response_eps005=responses[1],
                        error_estimate=error, stable_decrease=stable_down, stable_increase=stable_up)
    write_json(OUT/"parameter_summary.json", {"status": "passed", "groups": results,
        "gradient_norm": norm, "parameter_gradient_norms": norms, "linear_checks": linear_checks,
        "sequence_forward_max_error": maximum_error, "repeat_pair_cosine_max_error": float(noise.max()),
        "source_protocol_sha256": sha256(protocol), "response_sha256": sha256(path),
        "scope": record["scope"], "weights_restored": True, "test_used": False})
    print("PARAMETER_RESULT", results, flush=True)


if __name__ == "__main__":
    main()
