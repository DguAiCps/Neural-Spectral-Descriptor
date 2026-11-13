#!/usr/bin/env python3
"""선택된 시간 모델과 거의 같은 파라미터 수의 투영을 학습·평가한다."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_temporal_focus as exp
import gnn.graph_free as graph_free
from _pdv2_no_context_ablation import sha256, write_json

PRIMARY = exp.OUT
OUT = REPO / "results/pdv2_temporal_capacity_20261018"
LOG = REPO / "logs/temporal_capacity_20261018"
BASE_FACTORY = graph_free.create_graph_free_model


class CapacityProjection(nn.Module):
    """128→204→128의 노드별 투영이며 이웃 입력을 사용하지 않는다."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(channels, 204), nn.ReLU(), nn.Linear(204, channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr=None, **kwargs) -> torch.Tensor:
        """각 노드를 독립적으로 변환한다."""
        return self.layers(x)


def create_capacity_model(**kwargs) -> nn.Module:
    """모든 추가 파라미터가 계산에 참여하는 투영 대조를 만든다."""
    model = BASE_FACTORY(**kwargs)
    model.gnn.convs = nn.ModuleList(CapacityProjection(model.gnn.hidden_dim)
                                   for _ in range(model.gnn.n_layers))
    return model


def checkpoint_dir(seed: int, arm: str) -> Path:
    """주실험과 분리된 모델 저장 경로다."""
    return REPO / f"checkpoints/pdv2_temporal_capacity_s{seed}_{arm}"


def protocol(data: bool = False) -> dict:
    """주실험과 대조군의 동결 입력을 검사한다."""
    spec = exp.read(OUT / "protocol.json")
    for field in ("runtime_sha256", "input_sha256") if data else ("runtime_sha256",):
        for p, h in spec[field].items():
            assert sha256(REPO / p) == h, p
    return spec


def configure() -> str:
    """현재 프로세스의 기존 실행기만 새 대조군 경로로 연결한다."""
    spec = exp.read(OUT / "protocol.json")
    arm = spec["arm"]
    exp.OUT, exp.ARMS, exp.REUSED = OUT, (arm,), {}
    # 학습 YAML과 표본은 주실험의 동결 파일을 그대로 읽는다.
    exp.training_config = lambda unused: PRIMARY / "diff.yaml"
    exp.checkpoint_dir, exp.protocol = checkpoint_dir, protocol
    exp.base.OUT, exp.base.ARMS = OUT, (arm,)
    exp.base.checkpoint_dir, exp.base.protocol = checkpoint_dir, protocol
    graph_free.create_graph_free_model = create_capacity_model
    return arm


def prepare() -> None:
    """집계·표본 후보 선택 뒤 용량 대조의 예산과 구조를 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    primary = exp.protocol(True)
    choice = exp.read(PRIMARY / "expansion.json")
    assert choice["winner"] is not None, "확장할 시간 모델이 없음"
    arm = "point_" + choice["winner"].split("_")[1]
    runtime = dict(primary["runtime_sha256"])
    for path in (Path(__file__), REPO / "tests/gnn/test_temporal_capacity.py"):
        runtime[str(path.relative_to(REPO))] = sha256(path)
    inputs = dict(primary["input_sha256"])
    for path in (PRIMARY / "protocol.json", PRIMARY / "expansion.json"):
        inputs[str(path.relative_to(REPO))] = sha256(path)
    write_json(OUT / "protocol.json", {"created_unix": time.time(), "arm": arm,
        "winner": choice["winner"], "config": primary["config"], "test_used": False,
        "scope": "시간 모델167561파라미터에 대해 노드별MLP167705개(+144,+0.086%)의 대조. 같은입력/출력/잔차/게이트/정규화,2개의128→204→128 변환. 숨은 폭204는파라미터수로정하며검증탐색없음. 계산깊이와함수형태가완전히같은대조는아님.",
        "training": "선택된 시간모델과 같은 원래/반대25%표본,같은시드/순열,20에포크900업데이트.10/20중최종NSD/H평균AP선택.기존융합/판정/q+15고정.주실험후보재선택없음.",
        "runtime_sha256": runtime, "input_sha256": inputs})


def call(args: list, label: str) -> None:
    """독립 프로세스에서 대조군 경로 설정을 적용한다."""
    with (LOG / f"{label}.log").open("a") as handle:
        p = subprocess.run([sys.executable, "-u", str(Path(__file__)), *args], cwd=REPO,
                           stdout=handle, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError((LOG / f"{label}.log").read_text()[-5000:])


def run() -> None:
    """세 시드의 큰 투영 대조를 고정 예산으로 학습·최종 평가한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    LOG.mkdir(parents=True, exist_ok=True)
    prepare()
    arm = configure()
    started = time.time()
    selected = {}
    for seed in (1, 2, 3):
        marker = exp.arm_dir(seed, arm) / "training_complete.json"
        dest = checkpoint_dir(seed, arm)
        if not marker.exists():
            assert not dest.exists() or not any(dest.iterdir())
            tick = time.time()
            write_json(OUT / "status.json", {"status": "running", "stage": "training", "seed": seed})
            call(["train", "--arm", arm, "--seed", str(seed), "--config", str(PRIMARY / "diff.yaml"),
                  "--checkpoint-dir", str(dest), "--data-root", "/workspace/data", "--encoder-preset", "no_interdiff",
                  "--use-gated-context", "--gate-initial-alpha", "0.0625"], f"train_s{seed}")
            hashes = {}
            for ep in exp.EPOCHS:
                path = dest / f"checkpoint_epoch_{ep}.pth"
                blob = torch.load(path, map_location="cpu", weights_only=False)
                assert blob["epoch"] == ep-1
                assert all(torch.isfinite(v).all() for v in blob["model_state_dict"].values())
                assert all(int(v["step"].item()) == ep*45 for v in blob["optimizer_state_dict"]["state"].values())
                hashes[path.name] = sha256(path)
            write_json(marker, {"checkpoints": hashes, "runtime_s": time.time()-tick})
        jobs = [(ep, s + "/" + q) for ep in exp.EPOCHS for s, q in exp.base.VAL]

        def evaluate(job: tuple) -> None:
            ep, seq = job
            for action in ("keys", "score"):
                call([action, "--seed", str(seed), "--arm", arm, "--epoch", str(ep), "--sequence", seq],
                     f"{action}_s{seed}_{ep}_{seq.replace('/', '_')}")

        with ThreadPoolExecutor(max_workers=2) as pool:
            for done, future in enumerate(as_completed([pool.submit(evaluate, j) for j in jobs]), 1):
                future.result()
                write_json(OUT / "status.json", {"status": "running", "stage": "validation", "seed": seed, "done": done, "total": len(jobs)})
                print("CAPACITY_SCORED", seed, done, len(jobs), flush=True)
        selected[str(seed)] = exp.select(seed, (arm,))
        print("CAPACITY_SELECTED", seed, flush=True)
    write_json(OUT / "summary.json", {"per_seed": selected, "reference_winner": protocol()["winner"], "test_used": False})
    write_json(OUT / "status.json", {"status": "evaluated_pending_audit", "runtime_s": time.time()-started, "test_used": False})


def verify() -> None:
    """최종 원장·입력 범위를 검사하고 선택된 시간 모델과 비교한다."""
    arm = configure()
    from _pdv2_temporal_focus_verify import main as verify_ledgers
    verify_ledgers()
    primary = exp.read(PRIMARY / "summary.json")
    capacity = exp.read(OUT / "summary.json")
    winner = protocol()["winner"]
    rows, sources = {}, {}
    for seed in (1, 2, 3):
        temporal = primary["per_seed"][str(seed)][winner]
        point = capacity["per_seed"][str(seed)][arm]
        a = exp.read(PRIMARY / f"s{seed}/{winner}/initial_model.json")["tensor_sha256"]
        b = exp.read(OUT / f"s{seed}/{arm}/initial_model.json")["tensor_sha256"]
        common = set(a) & set(b)
        assert len(common) == 23 and all(a[k] == b[k] for k in common)
        a = exp.read(PRIMARY / f"s{seed}/{winner}/batch_order.json")["epochs"]
        b = exp.read(OUT / f"s{seed}/{arm}/batch_order.json")["epochs"]
        assert [r["sha256"] for r in a] == [r["sha256"] for r in b]
        assert exp.read(PRIMARY / f"s{seed}/{winner}/training_graph.json") == exp.read(OUT / f"s{seed}/{arm}/training_graph.json")
        rows[str(seed)] = {op: {"temporal": temporal["ops"][op]["macro"], "capacity_point": point["ops"][op]["macro"],
            "delta_pp": {m: (temporal["ops"][op]["macro"][m]-point["ops"][op]["macro"][m])*100 for m in exp.base.PRIMARY}}
            for op in exp.base.OPS}
        for model in (temporal, point):
            p = REPO / model["checkpoint"]
            assert sha256(p) == model["checkpoint_sha256"]
            sources[str(p.relative_to(REPO))] = sha256(p)
    stats = {op: {m: {"mean_pp": float(np.mean([rows[str(s)][op]["delta_pp"][m] for s in (1, 2, 3)])),
                        "std_pp": float(np.std([rows[str(s)][op]["delta_pp"][m] for s in (1, 2, 3)], ddof=1)),
                        "values_pp": [rows[str(s)][op]["delta_pp"][m] for s in (1, 2, 3)]}
                  for m in exp.base.PRIMARY} for op in exp.base.OPS}
    sources.update({str(p.relative_to(REPO)): sha256(p) for p in (PRIMARY / "summary.json", OUT / "summary.json", OUT / "protocol.json")})
    write_json(OUT / "comparison.json", {"rows": rows, "effects": stats, "sources": sources,
        "shared_initial_tensors": 23, "same_triplets_and_order": True, "same_graph": True,
        "temporal_parameters": 167561, "point_parameters": 167705, "test_used": False})
    status = exp.read(OUT / "status.json")
    write_json(OUT / "status.json", {**status, "status": "complete"})
    print("CAPACITY_VERIFIED", stats, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "train", "keys", "score", "verify"))
    args, rest = parser.parse_known_args()
    if args.action == "run":
        try:
            run()
        except Exception as error:
            OUT.mkdir(parents=True, exist_ok=True)
            write_json(OUT / "status.json", {"status": "failed", "error": repr(error)})
            raise
    elif args.action == "verify":
        verify()
    elif args.action == "train":
        configure()
        import _train_temporal_focus
        sys.argv = ["scripts/_train_temporal_focus.py", *rest]
        _train_temporal_focus.main()
    else:
        configure()
        p = argparse.ArgumentParser()
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--arm", required=True)
        p.add_argument("--epoch", type=int, required=True)
        p.add_argument("--sequence", required=True)
        a = p.parse_args(rest)
        if args.action == "keys":
            exp.keys(a.seed, a.arm, a.epoch, a.sequence)
        else:
            exp.base.score(a.seed, a.arm, a.epoch, "val", a.sequence)
