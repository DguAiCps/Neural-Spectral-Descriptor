#!/usr/bin/env python3
"""고정 시간 모델에서 중심·이웃·차이 성분의 장소 구분력을 진단한다."""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.utils import softmax

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_nonlinear_difference as previous
from _center_information_lib import fixed_triplets, triplet_statistics
from _pdv2_no_context_ablation import sha256, write_json
from gnn.nonlinear_relation import NonlinearRelationConv

OUT = REPO / "results/pdv2_center_information_20261021"
ARMS = ("diff_tanh", "standard_tanh", "diff", "standard")
run = previous.base


def load_model(arm: str, epoch: int) -> torch.nn.Module:
    """고정 체크포인트와 검색 재계량을 복원한다."""
    net = previous.model(arm).cuda().eval()
    path = run.root(1, arm) / "checkpoints" / f"checkpoint_epoch_{epoch}.pth"
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
    net.gnn.key_scale = torch.nn.Parameter(
        torch.from_numpy(np.load(run.exp.old.base.RVEC)).float().cuda(), requires_grad=False)
    assert net.gnn.key_transform is None and net.gnn.residual
    return net


def capture(net: torch.nn.Module, graph) -> tuple:
    """실제 주의 가중치를 고정해 A, C, D와 잔차 후 특징을 추출한다."""
    features, errors, handles = {}, [], []

    def hook(index):
        def measure(conv, args, kwargs, output):
            h, edge = args[:2]
            src, tgt = edge
            if index:
                np.testing.assert_allclose(h.cpu().numpy(), features[f"l{index}_post"], atol=2e-5, rtol=2e-5)
            relation = h[src]-h[tgt] if conv.value_source == "diff" else h[src]
            q = conv.W_q(h[tgt]).view(-1, conv.heads, conv.head_dim)
            k = conv.W_k(relation).view(-1, conv.heads, conv.head_dim)
            if isinstance(conv, NonlinearRelationConv):
                k = k.tanh()
            scores = torch.einsum("ehd,ehd->eh", q, k)/conv.scale
            attr = args[2] if len(args) > 2 else kwargs.get("edge_attr")
            if attr is not None:
                scores = scores + conv.bias_mlp(attr)
            assert kwargs.get("edge_logit") is None and kwargs.get("edge_value_gate") is None
            attention = softmax(scores, tgt, num_nodes=len(h))
            center = conv.W_v(h)
            messages = (attention[..., None] * center[src].view(-1, conv.heads, conv.head_dim)).reshape(-1, conv.channels)
            average = torch.zeros_like(center).index_add_(0, tgt, messages)
            counts = torch.bincount(tgt, minlength=len(h))
            assert bool((counts > 0).all())
            difference = average-center
            expected = difference if conv.value_source == "diff" else average
            torch.testing.assert_close(expected, output, atol=3e-5, rtol=3e-5)
            errors.append(float((expected-output).abs().max()))
            # 차이 모델은 실제 출력을 저장하며 항등식의 반올림 차이도 검사한다.
            if conv.value_source == "diff":
                difference = output
            post = net.gnn.batch_norms[index](output)
            if index < len(net.gnn.convs)-1:
                post = torch.relu(post)
            post = post+h
            for name, tensor in {"input": h, "center": center, "average": average,
                                 "difference": difference, "post": post}.items():
                features[f"l{index+1}_{name}"] = tensor.cpu().numpy()
        return measure

    def output_hook(module, args, output):
        np.testing.assert_allclose(args[0].cpu().numpy(), features["l2_post"], atol=2e-5, rtol=2e-5)
        features["context"] = output.cpu().numpy()

    handles.extend(conv.register_forward_hook(hook(i), with_kwargs=True) for i, conv in enumerate(net.gnn.convs))
    handles.append(net.gnn.output_proj.register_forward_hook(output_hook))
    try:
        with torch.no_grad():
            key = run.normalize(net(graph).float().cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()
    features["raw"] = graph.x.cpu().numpy()*np.load(run.exp.old.base.RVEC)
    features["key"] = key
    assert all(np.isfinite(v).all() for v in features.values())
    return features, max(errors)


def measure(features: dict, triplets: np.ndarray, target: Path, metadata: dict) -> dict:
    """삼중쌍 통계와 독립 검산용 실제 벡터 표본을 저장한다."""
    packed = {"triplets": triplets, **metadata}
    for name, values in features.items():
        for field, result in triplet_statistics(values, triplets).items():
            packed[f"{name}_{field}"] = result
    sample = np.unique(np.linspace(0, len(triplets)-1, min(32, len(triplets))).astype(int))
    np.savez_compressed(target, **packed)
    np.savez_compressed(target.with_name("audit_"+target.name), row=sample,
                        **{name: values[triplets[sample]] for name, values in features.items()})
    return packed


def main() -> None:
    """자료·삼중쌍을 먼저 고정한 뒤 네 모델의 학습·검증 특징을 잰다."""
    OUT.mkdir(parents=True, exist_ok=True)
    spec = previous.protocol(True)
    old = previous.read(previous.OUT/"summary.json")
    epochs = {a: old["per_seed"]["1"][a]["selected"]["epoch"] for a in ARMS}
    sources = {**spec["runtime"], **spec["inputs"]}
    for p in (Path(__file__), REPO/"scripts/_center_information_lib.py",
              REPO/"tests/gnn/test_center_information.py", previous.OUT/"summary.json"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    ledgers = {a: {} for a in ARMS}
    for arm in ARMS:
        checkpoint = run.root(1, arm)/"checkpoints"/f"checkpoint_epoch_{epochs[arm]}.pth"
        h = previous.read(run.root(1, arm)/"training_complete.json")["checkpoints"][checkpoint.name]
        assert sha256(checkpoint) == h
        sources[str(checkpoint.relative_to(REPO))] = h
        for seq in spec["val"]:
            p = run.root(1, arm)/f"epoch{epochs[arm]}"/(seq.replace("/", "_")+".npz")
            info = previous.read(p.with_suffix(".json"))
            for path, expected in ((p, info["sha256"]), (p.with_name("keys_"+p.name), info["keys_sha256"])):
                assert sha256(path) == expected
                sources[str(path.relative_to(REPO))] = expected
            ledgers[arm][seq] = run.exp.arrays(p)
    design = {"created_unix": time.time(), "sources": sources, "epochs": epochs, "test_used": False,
        "primary": "선택된 diff_tanh의 마지막 층. 같은 주의 가중치에서 A=ΣaWv(hj), C=Wv(hi), D=A-C의 정답/오답 코사인 여유. 선형 diff는 반복 대조, 표준 모델 D는 같은 가중치의 가상 차이 성분이다.",
        "validation_pairs": "양성 질의 전체. p는 j<=q-30 중 3D GT 5m 이내 최근접(동률 작은번호), n은 같은 과거 중 GT>25m에서 고정 재계량 raw 코사인 최대(동률 작은번호). 동일 쌍을 네 모델에 사용. 모델 중간 특징·성능으로 쌍을 선택하지 않는다.",
        "error_pairs": "별도 손실 사례: 표준 계열 최종 정답, 차이 계열 최종 오답인 q에 p=표준의 최종승자, n=차이의 최종승자. 5~25m 오답도 포함해 표시. 사례로 조건화한 분석이며 일반 성능 추정이 아니다.",
        "training_pairs": "고정 original_epoch16 삼중쌍의 중복 행을 제거. 학습 자료의 성분 진단이며 온라인 과거 검색 대조는 아니다.",
        "features": "각 층 입력H/중심C/이웃평균A/차이D/실제 정규화·활성화·잔차 후post, 실제context/raw/key. raw가 최종키에 남으므로 중심 전체가 소실됐다고 해석하지 않는다.",
        "metrics": "margin=cos(q,p)-cos(q,n), margin>0 비율·각 코사인·영벡터. 각 시퀀스→KITTI/HeLiPR/MulRan→같은 계열 비중 평균과 통합개수 별도. p/n은 GT 진단라벨이지 배포 입력 아님.",
        "interpretation": "중간 특징 구분력 진단, AP·최종수락 개선·인과 증명 아님. 고정 쌍에서 A가 D보다 우수하고 실제 손실 사례에서도 복원이 보이면 같은 용량 중심 보존 학습을 다음 후보로 제안. 임의 기준으로 시험이나 추가 시드 자동 확장 없음.",
        "scope": "새 학습/추론 개입/후보 재선택/최종 점수 변경/시험/원고 교체 없음."}
    freeze = OUT/"protocol.json"
    if freeze.exists():
        assert previous.read(freeze)["sources"] == sources
    else:
        write_json(freeze, design)
    plans = {}
    for seq in spec["val"]:
        _, _, paths, _ = run.exp.old.base.inputs("val", seq)
        cache = run.exp.arrays(paths["cache"])
        ref = ledgers["standard_tanh"][seq]
        xyz = cache["poses_gt"][:, :3, 3].astype(float)
        triplets = fixed_triplets(cache["descriptors"]*np.load(run.exp.old.base.RVEC), xyz, ref["query_idx"])
        np.testing.assert_array_equal(triplets[:, 0], ref["query_idx"][ref["min_past_gt_dist"] <= 5])
        indexes = np.searchsorted(ref["query_idx"], triplets[:, 0])
        plan = {"triplets": triplets, "group": ref["group"][indexes]}
        for arm in ARMS:
            r = ledgers[arm][seq]
            np.testing.assert_array_equal(r["query_idx"], ref["query_idx"])
            plan[f"{arm}_correct"] = r["top1_geo_dist"][indexes] <= 5
        for arm, reference in (("diff", "standard"), ("diff_tanh", "standard_tanh")):
            r, rref = ledgers[arm][seq], ledgers[reference][seq]
            lost = (r["min_past_gt_dist"] <= 5) & (rref["top1_geo_dist"] <= 5) & (r["top1_geo_dist"] > 5)
            plan[f"cases_{arm}"] = np.stack([r["query_idx"][lost], rref["j_star"][lost], r["j_star"][lost]], axis=1)
            plan[f"cases_{arm}_distance"] = r["top1_geo_dist"][lost]
        path = OUT/("pairs_"+seq.replace("/", "_")+".npz")
        np.savez_compressed(path, **plan)
        plans[seq] = plan
        print("PAIRS", seq, len(triplets), flush=True)
    train_state = torch.load(run.PREPARED, map_location="cpu", weights_only=False)
    train_seq = np.asarray(train_state["sequences"])
    train_triplets = np.unique(run.triplets(16), axis=0)
    np.savez_compressed(OUT/"pairs_train.npz", triplets=train_triplets, sequence=train_seq[train_triplets[:, 0]])
    write_json(OUT/"pairs_frozen.json", {"created_unix": time.time(), "protocol_sha256": sha256(freeze),
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in sorted(OUT.glob("pairs_*.npz"))},
        "train_original": len(run.triplets(16)), "train_unique": len(train_triplets),
        "val_positive_queries": sum(len(p["triplets"]) for p in plans.values())})
    original = torch.load(run.GRAPH, map_location="cpu", weights_only=False)
    train_graph = run.temporal_subgraph(original, np.arange(original.num_nodes)).cuda()
    del original, train_state
    output_error = key_error = prefix_error = 0.
    prefix_checks = 0
    for arm in ARMS:
        net = load_model(arm, epochs[arm])
        features, error = capture(net, train_graph)
        output_error = max(output_error, error)
        measure(features, train_triplets, OUT/("train_"+arm+".npz"), {"sequence": train_seq[train_triplets[:, 0]]})
        del features
        print("TRAIN_FEATURES", arm, len(train_triplets), flush=True)
        for seq in spec["val"]:
            _, _, paths, _ = run.exp.old.base.inputs("val", seq)
            cache = run.exp.arrays(paths["cache"])
            graph = run.make_graph(cache["descriptors"].astype(np.float32), cache["poses"],
                                   np.zeros(len(cache["descriptors"]), np.int64)).cuda()
            features, error = capture(net, graph)
            output_error = max(output_error, error)
            path = run.root(1, arm)/f"epoch{epochs[arm]}"/("keys_"+seq.replace("/", "_")+".npz")
            expected = run.exp.arrays(path)["embedding"]
            np.testing.assert_allclose(features["key"], expected, atol=4e-6, rtol=4e-5)
            key_error = max(key_error, float(abs(features["key"]-expected).max()))
            plan = plans[seq]
            metadata = {k: v for k, v in plan.items() if k == "group" or k.endswith("_correct")}
            measure(features, plan["triplets"], OUT/("val_"+arm+"_"+seq.replace("/", "_")+".npz"), metadata)
            for case_arm in ("diff", "diff_tanh"):
                measure(features, plan[f"cases_{case_arm}"],
                        OUT/("cases_"+case_arm+"_"+arm+"_"+seq.replace("/", "_")+".npz"),
                        {"negative_distance": plan[f"cases_{case_arm}_distance"]})
            if arm == "diff_tanh":
                q = ledgers[arm][seq]["query_idx"]
                for t in (int(q[len(q)//3]), int(q[2*len(q)//3])):
                    prefix = run.make_graph(cache["descriptors"][:t+16].astype(np.float32),
                        cache["poses"][:t+16], np.zeros(t+16, np.int64)).cuda()
                    small, _ = capture(net, prefix)
                    for name in features:
                        np.testing.assert_allclose(small[name][t], features[name][t], atol=3e-5, rtol=3e-5)
                        prefix_error = max(prefix_error, float(abs(small[name][t]-features[name][t]).max()))
                    prefix_checks += 1
                    del prefix, small
            del graph, features
            print("VAL_FEATURES", arm, seq, flush=True)
        del net
    write_json(OUT/"measurement_complete.json", {"created_unix": time.time(), "protocol_sha256": sha256(freeze),
        "max_output_reconstruction_error": output_error, "max_saved_key_error": key_error,
        "prefix_checks": prefix_checks, "max_prefix_feature_error": prefix_error, "test_used": False,
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in sorted(OUT.glob("*.npz"))}})
    print("MEASUREMENT_COMPLETE", output_error, key_error, prefix_checks, prefix_error, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
