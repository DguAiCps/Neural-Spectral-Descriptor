#!/usr/bin/env python3
"""같은 학습 표본에서 배치/명시적 오답의 손실 분리만 대조한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 scripts/_pdv2_nsd_stratified.py run
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_temporal_focus as exp
import _pdv2_temporal_capacity as cap
from _pdv2_no_context_ablation import sha256, write_json

PRIMARY = exp.OUT
OUT = REPO / "results/pdv2_nsd_stratified_20261018"
LOG = REPO / "logs/nsd_stratified_20261018"
CONFIG = REPO / "configs/pdv2_nsd_stratified.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
ARMS = tuple(CFG["arms"])
OP = "pbev128_main"


def protocol(data: bool = False) -> dict:
    """고정 실행 코드와 학습·검증 입력을 검사한다."""
    spec = exp.read(OUT / "protocol.json")
    for field in ("runtime_sha256", "input_sha256") if data else ("runtime_sha256",):
        for name, value in spec[field].items():
            assert sha256(REPO / name) == value, name
    return spec


def checkpoint_dir(seed: int, arm: str) -> Path:
    """기존 모델과 분리된 실험 경로다."""
    return REPO / f"checkpoints/pdv2_nsd_stratified_s{seed}_{arm}"


def configure() -> None:
    """현재 프로세스에만 새 경로·손실·용량 대조를 연결한다."""
    import gnn.graph_free as graph_free
    exp.OUT, exp.ARMS, exp.REUSED = OUT, ARMS, {}
    exp.training_config = lambda arm: PRIMARY / ("standard.yaml" if arm.startswith("standard") else "diff.yaml")
    exp.checkpoint_dir, exp.protocol = checkpoint_dir, protocol
    exp.base.OUT, exp.base.ARMS, exp.base.OPS = OUT, ARMS, (OP,)
    exp.base.checkpoint_dir, exp.base.protocol = checkpoint_dir, protocol
    graph_free.create_graph_free_model = cap.create_capacity_model


def prepare() -> None:
    """재학습 전 NSD 예산·확장 규칙을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    old = exp.protocol(True)
    cap.protocol(True)
    runtime = dict(cap.protocol()["runtime_sha256"])
    for path in (Path(__file__), CONFIG, REPO / "src/gnn/stratified_contrastive.py",
                 REPO / "tests/gnn/test_stratified_contrastive.py"):
        runtime[str(path.relative_to(REPO))] = sha256(path)
    inputs = dict(old["input_sha256"])
    for root in (PRIMARY, cap.OUT):
        for path in [root / "summary.json", *root.glob("s*/*/initial_model.json"),
                     *root.glob("s*/*/batch_order.json"), *root.glob("s*/*/training_graph.json")]:
            inputs[str(path.relative_to(REPO))] = sha256(path)
    config = dict(old["config"])
    config["arms"] = list(ARMS)
    write_json(OUT / "protocol.json", {"created_unix": time.time(), "config": config,
        "experiment_config": CFG, "test_used": False,
        "change": "기존InfoNCE 혼합분모를 배치오답CE와명시오답binaryCE의0.5/0.5평균으로교체. 두기존모델과같은전체초기상태·원래표본·배치순서·시간그래프·20에포크900업데이트. 투영은167705개용량대조. 추론변경없음.",
        "selection": "NSD최종검증AP로10/20중선택. 두모델각각선택하며NSD-H미사용. 시드1시간모델이기존시간및새투영보다각AP>=0.1%p이고R100손실<=0.1%p이면같은설정시드2/3확장. 미달이면예비종료. 시험평가는별도고정없이는실행안함.",
        "runtime_sha256": runtime, "input_sha256": inputs})


def train(rest: list) -> None:
    """학습 실행에서만 대조 손실을 교체한다."""
    configure()
    import gnn.trainer as trainer
    from gnn.stratified_contrastive import StratifiedContrastiveLoss
    trainer.InfoNCELoss = StratifiedContrastiveLoss
    original = trainer.GNNTrainer.__init__

    def checked_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        assert isinstance(self.criterion, StratifiedContrastiveLoss)

    trainer.GNNTrainer.__init__ = checked_init
    import _train_temporal_focus
    sys.argv = ["scripts/_train_temporal_focus.py", *rest]
    _train_temporal_focus.main()


def run() -> None:
    """동일 예산의 시드1 대조와 조건부 확장을 수행한다."""
    import subprocess
    prepare()
    configure()
    LOG.mkdir(parents=True, exist_ok=True)
    import _pdv2_temporal_focus_run as runner

    def call(args: list, label: str) -> None:
        action, rest = ("train", args[1:]) if args[0].endswith("_train_temporal_focus.py") else (args[1], args[2:])
        with (LOG / f"{label}.log").open("a") as log:
            result = subprocess.run([sys.executable, "-u", str(Path(__file__)), action, *rest], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError((LOG / f"{label}.log").read_text()[-5000:])

    runner.call = call
    start = time.time()
    selected = {"1": runner.run_seed(1, ARMS)}
    raw = exp.read(PRIMARY / "summary.json")["per_seed"]["1"]["standard_original"]
    baseline = max(raw["candidates"], key=lambda r: r["ops"][OP]["macro"]["AP"])["ops"][OP]["macro"]
    temporal = selected["1"]["standard_original"]["ops"][OP]["macro"]
    point = selected["1"]["point_original"]["ops"][OP]["macro"]
    comparisons = {name: {m: temporal[m]-other[m] for m in exp.base.PRIMARY} for name, other in (("old_standard", baseline), ("new_capacity_point", point))}
    expand = all(r["AP"] >= CFG["expansion_min_ap_gain"] and r["R@100P"] >= -CFG["expansion_r100_loss_tolerance"] for r in comparisons.values())
    write_json(OUT / "expansion.json", {"expand": expand, "comparisons": comparisons, "test_used": False})
    if expand:
        for seed in (2, 3):
            selected[str(seed)] = runner.run_seed(seed, ARMS)
    write_json(OUT / "summary.json", {"per_seed": selected, "expand": expand, "test_used": False})
    verify()
    write_json(OUT / "status.json", {"status": "complete", "runtime_s": time.time()-start, "test_used": False})
    print("COMPLETE", comparisons, "expanded", expand, flush=True)


def verify() -> None:
    """최종 원장과 원래 대조군의 초기화·표본·그래프 일치를 확인한다."""
    configure()
    from _pdv2_temporal_focus_verify import main as verify_ledgers
    verify_ledgers()
    summary = exp.read(OUT / "summary.json")
    records = []
    for seed in summary["per_seed"]:
        for arm in ARMS:
            original = PRIMARY if arm.startswith("standard") else cap.OUT
            new = OUT / f"s{seed}/{arm}"
            old = original / f"s{seed}/{arm}"
            for name in ("initial_model.json", "batch_order.json", "training_graph.json"):
                assert exp.read(new/name) == exp.read(old/name), (seed, arm, name)
            records.append({"seed": seed, "arm": arm, "identical_full_initial_state": True,
                            "identical_triplets_and_order": True, "identical_graph": True})
    write_json(OUT / "control_validation.json", {"status": "passed", "comparisons": records, "test_used": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "train", "keys", "score", "verify"))
    args, rest = parser.parse_known_args()
    if args.action in ("prepare", "run", "verify"):
        globals()[args.action]()
    elif args.action == "train":
        train(rest)
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
