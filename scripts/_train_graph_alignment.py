#!/usr/bin/env python3
"""원래 학습기를 유지하며 매 업데이트의 유사 간선을 온라인 규칙으로 만든다."""
from __future__ import annotations

import copy
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_graph_alignment as exp
from _pdv2_no_context_ablation import write_json
from _train_attention_neighbors import array_hash


def select_online_edges(emb: np.ndarray, d_rm: np.ndarray, phase: np.ndarray,
                        poses: np.ndarray, clf, blob: dict, device: str) -> tuple:
    """운영 코드의 후보·분류기·선택·속성을 그대로 계산한다."""
    from _causal_edge_features import causal_edge_features
    from _causal_branch_fusion import select_edges_per_node
    src, dst, feats, _ = causal_edge_features(emb, d_rm, phase, poses, want_labels=False)
    if not len(src):
        return np.empty((2, 0), np.int64), np.empty((0, 5), np.float32)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        logits = clf(torch.from_numpy((feats - blob["mu"].astype(np.float32)) /
                                      blob["sd"].astype(np.float32)).float().to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    sel = select_edges_per_node(src, probs, keep_k=10)
    src, dst = src[sel], dst[sel]
    cos = feats[sel, 0]
    l2 = np.linalg.norm(emb[src] - emb[dst], axis=1)
    attr = np.stack([np.zeros_like(cos), np.zeros_like(cos), cos,
                     np.log1p(l2) / 5., probs[sel].astype(np.float32)], axis=1).astype(np.float32)
    assert np.all(dst <= src - 30)
    assert np.isfinite(attr).all()
    return np.stack([dst, src]), attr


class OnlineGraphRefresh:
    """불변 첫 그래프만 캐시하고 현재 모델로 매번 둘째 그래프를 다시 만든다."""

    def __init__(self, graph, model, destination: Path, cfg: dict, clf, blob: dict,
                 rvec: np.ndarray, records: list, workers: int = 3) -> None:
        self.graph, self.model, self.destination = graph, model, destination
        self.cfg, self.clf, self.blob, self.rvec = cfg, clf, blob, rvec
        self.records = records
        self.workers = workers
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.original_forward = model.forward
        self.calls = 0
        self.seconds = []
        mask = graph.edge_type == 0
        self.temporal_index = graph.edge_index[:, mask].clone()
        self.temporal_attr = graph.edge_attr[mask].clone()
        self.original_similarity = int((~mask).sum())
        self.preflight_done = False

    def refresh(self, epoch: int, updates: int) -> dict:
        """첫 통과는 미분 없이 추론 모드로 수행하고 간선을 즉시 갱신한다."""
        tick = time.monotonic()
        model = self.model
        audit_state = ({k: v.detach().clone() for k, v in model.state_dict().items()}
                       if self.calls == 0 else None)
        audit_rng = torch.get_rng_state().clone() if self.calls == 0 else None
        audit_cuda_rng = torch.cuda.get_rng_state_all() if self.calls == 0 and torch.cuda.is_available() else None
        states = [(m, m.training) for m in model.modules()]
        old_scale = model.gnn.key_scale
        assert old_scale is None, "손실의 키 재계량 설정을 바꾸면 안 된다"
        model.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(self.rvec).to(self.graph.x.device), requires_grad=False)
        model.eval()
        jobs = []
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                for row in self.records:
                    emb = self.original_forward(row["graph"]).float().cpu().numpy()
                    # 운영 경로도 f0를 여기서 전체 정규화하지 않는다.
                    jobs.append(self.pool.submit(select_online_edges, emb, row["d_rm"],
                                                 row["phase"], row["cache"]["poses"],
                                                 self.clf, self.blob, str(self.graph.x.device)))
        finally:
            model.gnn.key_scale = old_scale
            for module, state in states:
                module.training = state
        all_index, all_attr = [], []
        stats = []
        for row, future in zip(self.records, jobs):
            index, attr = future.result()
            all_index.append(index + row["offset"])
            all_attr.append(attr)
            stats.append({"sequence": row["sequence"], "nodes": len(row["cache"]["descriptors"]),
                          "similarity": index.shape[1], "probability_mean": float(attr[:, 4].mean()),
                          "probability_below_0_5": float((attr[:, 4] < .5).mean())})
        if audit_state is not None:
            assert all(torch.equal(v, model.state_dict()[k]) for k, v in audit_state.items())
            assert torch.equal(audit_rng, torch.get_rng_state())
            if audit_cuda_rng is not None:
                assert all(torch.equal(a, b) for a, b in zip(audit_cuda_rng, torch.cuda.get_rng_state_all()))
            write_json(self.destination / "probe_state_audit.json", {
                "weights_and_bn_buffers_unchanged": True, "rng_unchanged": True,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "gradient_disabled_for_edge_selection": True})
        index = torch.from_numpy(np.concatenate(all_index, axis=1)).to(self.graph.x.device)
        attr = torch.from_numpy(np.concatenate(all_attr)).to(self.graph.x.device)
        self.graph.edge_index = torch.cat([self.temporal_index, index], dim=1)
        self.graph.edge_attr = torch.cat([self.temporal_attr, attr])
        self.graph.edge_type = torch.cat([
            torch.zeros(self.temporal_index.shape[1], dtype=torch.long, device=index.device),
            torch.ones(index.shape[1], dtype=torch.long, device=index.device)])
        self.calls += 1
        duration = time.monotonic() - tick
        self.seconds.append(duration)
        result = {"refresh": self.calls, "epoch": epoch + 1, "optimizer_updates_before": updates,
                  "refresh_s": duration, "mean_refresh_s": float(np.mean(self.seconds)),
                  "remaining_refresh_s_estimate": (900 - self.calls) * float(np.mean(self.seconds)),
                  "temporal": self.temporal_index.shape[1], "similarity": index.shape[1],
                  "by_sequence": stats}
        with (self.destination / "refresh.jsonl").open("a") as f:
            import json
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        write_json(self.destination / "refresh_progress.json", result)
        if self.calls == 1 or self.calls % 15 == 0:
            print("GRAPH_REFRESH", result["refresh"], result["epoch"], round(duration, 2), flush=True)
        return result

    def close(self) -> None:
        """작업 스레드를 종료한다."""
        self.pool.shutdown(wait=True)


def make_refresh(graph, model, destination: Path) -> OnlineGraphRefresh:
    """기존 학습 노드/시간 간선과 정확히 대응하는 첫 그래프를 준비한다."""
    from _pdv2_pair_evaluate import ArrayCache
    from _pdv2_matching_mine import cache_path
    from _lag_branch_fusion import lag_pass1_graph
    from build_and_train_edge_classifier import range_phase_flat
    from _xs_eval_lib import normalize
    from _causal_branch_fusion import EdgeMLP, N_FEAT
    from _pdv2_lag15_campaign import CONFIG as inference_config
    from evaluate_kitti_checkpoint import _apply_encoder_preset
    import yaml
    import _pd_build_edge_classifier as data
    cfg = _apply_encoder_preset(yaml.safe_load(inference_config.read_text()), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = .0625
    rvec = np.load(exp.base.RVEC).astype(np.float32)
    blob = torch.load(exp.base.classifier_path(1), map_location="cpu", weights_only=False)
    # 추론용 GNN을 새로 만들지 않아 학습 생성자의 초기값 감사 훅을 건드리지 않는다.
    # 분류기 생성자의 난수 소비도 기존 표본 순서를 바꾸지 않는다.
    with torch.random.fork_rng():
        clf = EdgeMLP(N_FEAT)
    clf.load_state_dict(blob["state_dict"])
    clf.eval().to(graph.x.device)
    for p in clf.parameters():
        p.requires_grad_(False)
    data.PD_CACHE = REPO / "data/preprocessed_v2"
    rows, offset = [], 0
    for sensor, seq in exp.base.TRAIN:
        with np.load(cache_path(sensor + "/" + seq)) as z:
            cache = ArrayCache({k: z[k] for k in z.files})
        raw = cache["descriptors"].astype(np.float32)
        np.testing.assert_array_equal(graph.x[offset:offset + len(raw)].cpu().numpy(), raw)
        layout = data._layouts(sensor, seq, cache)
        first = lag_pass1_graph(cache, raw, cfg, str(graph.x.device), sensor)
        local_time = first.edge_type == 0
        global_time = ((graph.edge_type == 0) & (graph.edge_index[1] >= offset)
                       & (graph.edge_index[1] < offset + len(raw)))
        assert torch.equal(first.edge_index[:, local_time] + offset, graph.edge_index[:, global_time])
        assert torch.equal(first.edge_attr[local_time], graph.edge_attr[global_time])
        rows.append({"sequence": sensor + "/" + seq, "sensor": sensor, "offset": offset,
                     "cache": cache, "layout": layout, "graph": first,
                     "d_rm": normalize(raw * rvec[None]), "phase": range_phase_flat(layout)})
        offset += len(raw)
    assert offset == graph.num_nodes == 72306
    assert exp.read(destination / "initial_model.json") == exp.read(
        exp.arm_root("diff_full") / "initial_model.json")
    write_json(destination / "first_graphs.json", {"same_temporal_edges_and_attributes": True,
               "same_node_inputs": True, "sequences": [{"sequence": r["sequence"],
               "nodes": int(r["graph"].num_nodes), "similarity": int((r["graph"].edge_type == 1).sum()),
               "edge_index_sha256": array_hash(r["graph"].edge_index.cpu().numpy()),
               "edge_attr_sha256": array_hash(r["graph"].edge_attr.cpu().numpy())} for r in rows]})
    return OnlineGraphRefresh(graph, model, destination, cfg, clf, blob, rvec, rows,
                              workers=exp.CFG["feature_workers"])


def main() -> None:
    """원래 학습기와 감사 래퍼를 유지하고 그래프 갱신만 끼운다."""
    from gnn.trainer import GNNTrainer
    import _train_attention_neighbors as original
    exp.protocol(True)
    original.OUT = exp.OUT
    original.SAMPLES = exp.SAMPLES
    epoch_original = GNNTrainer.train_epoch
    refreshers = []

    def train_epoch(self, graph, *args, **kwargs):
        if not refreshers:
            dest = exp.arm_root(exp.NEW)
            refresher = make_refresh(graph, self.model, dest)
            refreshers.append(refresher)
            forward = self.model.forward

            def online_forward(input_graph):
                if self.model.training and torch.is_grad_enabled() and input_graph is graph:
                    refresher.refresh(self.epoch, self.matching_updates)
                return forward(input_graph)

            self.model.forward = online_forward
        before = refreshers[0].calls
        value = epoch_original(self, graph, *args, **kwargs)
        assert refreshers[0].calls - before == 45
        assert refreshers[0].calls == (self.epoch + 1) * 45
        return value

    GNNTrainer.train_epoch = train_epoch
    try:
        original.main()
    finally:
        for refresher in refreshers:
            refresher.close()


if __name__ == "__main__":
    main()
