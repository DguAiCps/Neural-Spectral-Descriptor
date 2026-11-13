#!/usr/bin/env python3
"""같은 KISS 쌍의 계열 손실 비중과 720→900회 학습 경과를 대조한다."""
import argparse
import logging
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_native_pairs as parent
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash
from gnn.family_weighting import equal_family_mean, infonce_rows

OUT = REPO / "results/pdv2_family_balance_20261020"
SAMPLES = parent.OUT / "kiss_large"
ARMS = ("natural", "balanced")
EVAL_ARMS = tuple(f"{a}_{n}" for a in ARMS for n in (720, 900))
align, read = parent.align, parent.read


def protocol() -> dict:
    """실행 전에 동결한 소스와 입력을 확인한다."""
    spec = read(OUT / "protocol.json")
    for path, expected in spec["sources"].items():
        assert sha256(REPO / path) == expected, path
    return spec


def training_data() -> dict:
    """학습 KISS의 원 특징·계열·정답 자세만 읽는다."""
    return align.previous.load(parent.INPUT / "o_inputs.npz")


def trips(epoch: int) -> np.ndarray:
    """기존 네 표본 갱신 묶음을 같은 순서로 읽는다."""
    refresh = 1 + 5 * ((epoch - 1) // 5)
    return np.load(SAMPLES / f"triplets_epoch{refresh}.npz")["triplets"]


def prepare() -> None:
    """자료·비중·진단 배치·900회 주조건을 검증 전에 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol()
        return
    previous = read(parent.OUT / "completion_validation.json")
    assert previous["status"] == "passed"
    for path, expected in previous["sources"].items():
        assert sha256(REPO / path) == expected, path
    source = parent.protocol()
    data = training_data()
    audits, sources = {}, dict(previous["sources"])
    for epoch in range(1, 21):
        ordered = trips(epoch).copy()
        assert ordered.shape == (182919, 3)
        permutation = np.arange(len(ordered))
        np.random.RandomState(100000 + epoch).shuffle(permutation)
        ordered = ordered[permutation]
        labels = (data["sequences"][ordered[:, 0]] >= 4).astype(np.int64)
        counts = [np.bincount(labels[i:i+1024], minlength=2).tolist()
                  for i in range(0, len(labels), 1024)]
        assert np.min(counts) > 0
        audits[str(epoch)] = {
            "permutation_sha256": array_hash(permutation),
            "triplets_sha256": array_hash(ordered), "batch_family_counts": counts,
        }
    probe = trips(16).copy()
    np.random.RandomState(100016).shuffle(probe)
    indices = np.unique(np.linspace(0, 178, 24).astype(int))
    arrays = {str(i): probe[i*1024:min((i+1)*1024, len(probe))] for i in indices}
    np.savez_compressed(OUT / "probe_batches.npz", **arrays)
    paths = [
        Path(__file__), REPO / "scripts/_pdv2_family_balance_diagnose.py",
        REPO / "scripts/_pdv2_family_balance_run.py",
        REPO / "src/gnn/family_weighting.py",
        REPO / "tests/gnn/test_family_weighting.py", OUT / "probe_batches.npz",
        parent.OUT / "summary.json", parent.OUT / "protocol.json",
    ]
    sources.update({str(p.relative_to(REPO)): sha256(p) for p in paths})
    spec = {
        "created_unix": time.time(), "seed": 1, "train": source["train"], "val": source["val"],
        "arms": list(ARMS), "eval_arms": list(EVAL_ARMS), "primary": "balanced_900",
        "epochs": 20, "updates_per_epoch": 45, "triplets_per_epoch": 182919,
        "sample_source": str(SAMPLES.relative_to(REPO)), "batch_audit": audits,
        "graph": source["plans"]["kiss_large"]["graph"],
        "training": "KISS 전체 선택별 쌍·그래프·초기값·시드·배치 순서·원 InfoNCE 후보·학습률·AMP 고정. 20에포크 900업데이트. 16에포크 720회와 20에포크 900회 저장. 새 표본 복원 추가 없음.",
        "weighting": "natural=기존 행 평균. balanced=배치 내 KITTI 평균 손실/2 + Bridge 평균 손실/2. 두 계열 모두 있는 179배치×20 검증. 후보 분모·hard negative·노드/그래프·정규화 구조 유지.",
        "diagnostics": "고정 epoch16 순열에서 균등 간격으로 24개 미니배치. 초기/기존900/새 두 모델의720,900. 전체 학습 그래프 train 모드 FP32·TF32비활성, 매 배치 상태 복원·같은 dropout seed. 계열 평균 손실의 실제 파라미터 기울기. 학습은AMP유지. clipping/Adam 이전 국소값이며 과거 업데이트 기여율이 아님.",
        "validation": "기존 KISS검증9·99236질의·q+15·j<=q-30·100+100·고정 융합. 실제 후보/승자 재검색. 기존 판정과 고정 OOF 문맥 가중합. 점수 재적합 없음.",
        "promotion": "balanced_900이 natural_900 및 기존 gt_large900 각각에 대해 평균AP +.001 이상, R100 손실<=.001, 계열AP 손실<=.0025이면 해당 판정의3시드확증후보. 720회는경과진단이며사후주조건선택안함. 시험/원고변경없음.",
        "limits": "한 시드·반복 검증. 배치 후보와 전체 그래프 정규화의 자료 편중은 남는다. 720→900은같은자료모집단·마이닝규칙·연속학습경로지만LR도예정스케줄에따라변한다. 국소 기울기 크기는 Adam 최종 이동량이나 전체일반화원인의 비율이 아님.",
        "sources": sources, "test_used": False,
    }
    write_json(OUT / "protocol.json", spec)
    print("PREPARED", len(sources), "sources", len(indices), "probe batches", flush=True)


class AuditedLoss(torch.nn.Module):
    """원 후보를 유지하고, 미리 확인한 앵커 계열로 손실만 집계한다."""
    def __init__(self, original: torch.nn.Module, balanced: bool) -> None:
        super().__init__()
        self.original = original
        self.balanced = balanced
        self.temperature = original.temperature
        self.labels = None
        self.offset = 0
        self.counts = []

    def forward(self, anchors, positives, negatives):
        size = len(anchors)
        labels = self.labels[self.offset:self.offset+size]
        assert len(labels) == size and self.offset % 1024 == 0
        self.offset += size
        self.counts.append(np.bincount(labels, minlength=2).tolist())
        if not self.balanced:
            return self.original(anchors, positives, negatives)
        rows = infonce_rows(anchors, positives, negatives, self.temperature)
        return equal_family_mean(rows, torch.as_tensor(labels, device=rows.device))


def train(arm: str) -> None:
    """원 배치 그대로 두 모델을 학습하고 연속 경로의 두 지점을 저장한다."""
    from gnn.trainer import GNNTrainer
    spec = protocol()
    root = OUT / arm
    if (root / "training_complete.json").exists():
        done = read(root / "training_complete.json")
        assert sha256(root / "checkpoints/final_model.pth") == done["checkpoint_sha256"]
        return
    assert not (root / "checkpoints").exists(), "미완료 모델을 덮어쓰지 않는다"
    root.mkdir(exist_ok=True)
    data = training_data()
    graph = torch.load(REPO / spec["graph"], map_location="cpu", weights_only=False)
    random.seed(1); np.random.seed(1); torch.manual_seed(1); torch.cuda.manual_seed_all(1)
    net = align.old.model()
    write_json(root / "initial_model.json", {k: array_hash(v.numpy()) for k, v in net.state_dict().items()})
    trainer = GNNTrainer(net, learning_rate=5e-4, weight_decay=1e-5, temperature=.1,
                         checkpoint_dir=str(root / "checkpoints"), use_multi_gpu=False,
                         patience=100, use_amp=True, loss_type="infonce")
    assert trainer.context_aux_lambda == trainer.phase_token_aux_lambda == 0
    trainer.scaler = torch.amp.GradScaler("cuda", init_scale=128., growth_interval=1000000)
    criterion = AuditedLoss(trainer.criterion, arm == "balanced")
    trainer.criterion = criterion
    updates, records, orders = [], [], []
    trainer.optimizer.register_step_post_hook(lambda *args: updates.append(1))

    class FixedMiner:
        def mine_triplets(self, **kwargs):
            np.testing.assert_array_equal(kwargs["descriptors"], data["raw"])
            return trips(trainer.epoch + 1).tolist()

    original_shuffle, original_epoch = np.random.shuffle, trainer.train_epoch

    def audited_shuffle(values):
        before = np.random.get_state()
        original_shuffle(values)
        if isinstance(values, np.ndarray) and values.shape == (182919, 3):
            after = np.random.get_state()
            permutation = np.arange(len(values))
            np.random.set_state(before); original_shuffle(permutation); np.random.set_state(after)
            epoch = trainer.epoch + 1
            np.testing.assert_array_equal(values, trips(epoch)[permutation])
            record = {"permutation_sha256": array_hash(permutation), "triplets_sha256": array_hash(values)}
            for k, v in record.items():
                assert v == spec["batch_audit"][str(epoch)][k]
            orders.append(record)
            criterion.labels = (data["sequences"][values[:, 0]] >= 4).astype(np.int64)
            criterion.offset, criterion.counts = 0, []

    def audited_epoch(*args, **kwargs):
        epoch = trainer.epoch + 1
        seed = 100000 + epoch
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        lr = trainer.optimizer.param_groups[0]["lr"]
        value = original_epoch(*args, **kwargs)
        assert len(updates) == 45*epoch and len(orders) == epoch
        assert criterion.offset == 182919
        assert criterion.counts == spec["batch_audit"][str(epoch)]["batch_family_counts"]
        assert np.isfinite(value) and all(torch.isfinite(v).all() for v in net.state_dict().values())
        if epoch == 16:
            trainer.save_checkpoint("update_720.pth")
        records.append({"epoch": epoch, "updates": len(updates), "loss": float(value),
                        "learning_rate": lr, "family_counts_verified": True, **orders[-1]})
        write_json(root / "training_progress.json", {"records": records})
        print("EPOCH", arm, epoch, round(float(value), 6), len(updates), flush=True)
        return value

    np.random.shuffle, trainer.train_epoch = audited_shuffle, audited_epoch
    started = time.time()
    try:
        trainer.train(graph, data["gt"], data["raw"], data["sequences"], val_datasets=None,
                      n_epochs=20, triplet_miner=FixedMiner(), mine_every_n_epochs=5, n_triplets_per_anchor=3)
    finally:
        np.random.shuffle = original_shuffle
    checkpoints = {}
    for updates_count, name in ((720, "update_720.pth"), (900, "final_model.pth")):
        dest = OUT / f"{arm}_{updates_count}" / "checkpoints/final_model.pth"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "checkpoints" / name, dest)
        checkpoints[str(updates_count)] = sha256(dest)
    replay = {}
    if arm == "natural":
        reference = torch.load(SAMPLES / "checkpoints/final_model.pth", map_location="cpu", weights_only=False)["model_state_dict"]
        current = torch.load(root / "checkpoints/final_model.pth", map_location="cpu", weights_only=False)["model_state_dict"]
        replay = {"all_tensors_exact": all(torch.equal(current[k], v) for k, v in reference.items()),
                  "maximum_parameter_or_buffer_difference": max(float((current[k].double()-v.double()).abs().max()) for k, v in reference.items())}
    write_json(root / "training_complete.json", {
        "epochs": 20, "updates": len(updates), "runtime_s": time.time()-started,
        "order_hashes": orders, "checkpoints": checkpoints, "reference_replay": replay,
        "initial_sha256": sha256(root / "initial_model.json"),
        "checkpoint_sha256": sha256(root / "checkpoints/final_model.pth"), "test_used": False})


def validation(arm: str) -> None:
    """동결한 최종 평가기로 네 체크포인트를 각각 검색·판정한다."""
    protocol()
    align.OUT = OUT
    align.validation(arm)


def report() -> None:
    """주조건과 학습 경과의 최종 지표를 구분해 집계한다."""
    spec = protocol()
    previous = read(parent.OUT / "summary.json")
    per = {"reference_gt": previous["per_sequence"]["gt_large"],
           "reference_full": previous["per_sequence"]["reference"]}
    for arm in EVAL_ARMS:
        per[arm] = {m: {} for m in ("baseline", "context_score")}
        for sequence in spec["val"]:
            p = OUT / arm / "val" / (sequence.replace("/", "_") + ".npz")
            info = read(p.with_suffix(".json"))
            assert sha256(p) == info["sha256"]
            for method in per[arm]:
                per[arm][method][sequence] = info["metrics"][method]
    aggregate = {a: {m: align.accept.macro(rows) for m, rows in methods.items()} for a, methods in per.items()}
    checks, effects = {}, {}
    for method in ("baseline", "context_score"):
        now = aggregate["balanced_900"][method]
        checks[method] = {}
        for reference in ("natural_900", "reference_gt"):
            base = aggregate[reference][method]
            checks[method][reference] = {
                "mean_AP": now["macro"]["AP"] >= base["macro"]["AP"]+.001,
                "mean_R100": now["macro"]["R@100P"] >= base["macro"]["R@100P"]-.001,
                "family_AP": all(now["families"][f]["AP"] >= base["families"][f]["AP"]-.0025 for f in base["families"])}
        effects[method] = {}
        for name, a, b in (
            ("weighting720", "natural_720", "balanced_720"), ("weighting900", "natural_900", "balanced_900"),
            ("natural_progress", "natural_720", "natural_900"), ("balanced_progress", "balanced_720", "balanced_900"),
        ):
            effects[method][name] = {m: 100*(aggregate[b][method]["macro"][m]-aggregate[a][method]["macro"][m])
                                     for m in align.accept.exp.METRICS}
    write_json(OUT / "summary.json", {
        "seed": 1, "aggregate": aggregate, "per_sequence": per, "effects_pp": effects,
        "expansion_checks": checks,
        "eligible_for_three_seed_confirmation": {m: all(all(x.values()) for x in c.values()) for m, c in checks.items()},
        "protocol_sha256": sha256(OUT / "protocol.json"), "test_used": False})
    for arm in EVAL_ARMS:
        print("FINAL", arm, {m: x["macro"] for m, x in aggregate[arm].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "val", "report"))
    parser.add_argument("--arm", choices=(*ARMS, *EVAL_ARMS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.WARNING)
    if args.action == "prepare": prepare()
    elif args.action == "train": train(args.arm)
    elif args.action == "val": validation(args.arm)
    else: report()
