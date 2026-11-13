#!/usr/bin/env python3
"""시간 전용 모델의 같은 장소·회색 구간 오답 제외를 최종 판정으로 비교한다."""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
import _pdv2_center_preservation as previous
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash
from gnn.negative_exclusion import exclusion_masks, excluded_infonce

base = previous.base
OUT = REPO/"results/pdv2_negative_exclusion_20261021"
CONFIG = REPO/"configs/pdv2_negative_exclusion.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
ARMS = tuple(CFG["arms"])
METHOD = CFG["final_score"]
read = previous.read


def model(arm: str) -> torch.nn.Module:
    """기존 구조·초기화 그대로 손실만 바꿀 모델을 만든다."""
    name = CFG["models"][arm]
    return previous.previous.model(name) if name == "standard_tanh" else previous.model(name)


def reference_folder(name: str) -> Path:
    """동결된 동일 예산 대조군의 실제 결과 위치다."""
    parent = previous.OUT if name.startswith("center_") else previous.previous.OUT
    return parent/"s1"/name


def protocol(inputs: bool = False) -> dict:
    """현재 실행과 선택한 입력의 내용 해시를 확인한다."""
    spec = read(OUT/"protocol.json")
    for field in (("runtime", "inputs") if inputs else ("runtime",)):
        for path, expected in spec[field].items():
            assert sha256(REPO/path) == expected, path
    return spec


def prepare() -> None:
    """학습·검증 전에 네 조건과 기존 대조, 평가 규칙을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/"protocol.json").exists():
        protocol(True)
        return
    old = previous.protocol(True)
    summary = read(previous.OUT/"summary.json")
    refs = {name: (summary["per_seed"]["1"] if name.startswith("center_") else summary["references"])[name]
            for name in CFG["references"]}
    sources = dict(old["inputs"])
    paths = [previous.OUT/"protocol.json", previous.OUT/"summary.json"]
    for name in CFG["references"]:
        root = reference_folder(name)
        paths.extend(root/p for p in ("initial_model.json", "training_complete.json", "training_progress.json"))
        for epoch in (10, 20):
            paths.append(root/f"checkpoints/checkpoint_epoch_{epoch}.pth")
            paths.extend((root/f"epoch{epoch}").glob("*.npz"))
            paths.extend((root/f"epoch{epoch}").glob("*.json"))
    sources.update({str(p.relative_to(REPO)): sha256(p) for p in paths})
    initial = read(reference_folder("center_zero")/"initial_model.json")
    counts = {}
    for arm in ARMS:
        torch.manual_seed(1)
        net = model(arm)
        assert {k: array_hash(v.numpy()) for k, v in net.state_dict().items()} == initial
        counts[arm] = sum(p.numel() for p in net.parameters())
    assert set(counts.values()) == {167561}
    runtime = dict(old["runtime"])
    for p in (CONFIG, Path(__file__), REPO/"scripts/_pdv2_negative_exclusion_run.py",
              REPO/"src/gnn/negative_exclusion.py", REPO/"tests/gnn/test_negative_exclusion.py"):
        runtime[str(p.relative_to(REPO))] = sha256(p)
    write_json(OUT/"protocol.json", {"created_unix": time.time(), "runtime": runtime, "inputs": sources,
        "config": CFG, "train": old["train"], "val": old["val"], "confidence": old["confidence"],
        "graph_hashes": old["graph_hashes"], "parameters": counts, "references": refs,
        "test_used": False, "initial_all_tensors_identical": True,
        "hypothesis": "시간 전용 현재모델의 학습 오답 분모에 같은장소/회색쌍이 섞인다. 이를 제외하면 검색과최종판정이 개선되는지 확인한다. 기존 전체그래프 추가학습 mask5의 작은효과를 인지한 후속탐색이다.",
        "loss": "같은 검증된좌표계에서GT 3D거리<=5 또는<=25m인 off-diagonal 배치후보와지정음성을분모에서제외. 지정양성은유지. 가까운쌍을새양성으로추가하지않음. 시간간격제외없음. KITTI0..3독립좌표,Bridge01..04공통100. 다른좌표계항그대로.",
        "training": "기존GT학습8·72306노드·722820시간간선·183249삼중쌍·초기43텐서·배치순서동일. 네조건새20epoch900업데이트. Adam5e-4/wd1e-5/InfoNCEtemperature.1/AMP/기존warmup-cosine. no_interdiff288·게이트초기.0625,직접GNNTrainer. 재계량/융합재적합없음.",
        "evaluation": "검증9·99236질의·q+15·j<=q-30·100+100·기존최종arc_unique_b100. 각모델후보/승자재검색,모든10/20epoch보존. GT는학습라벨/평가정답에만사용. 추론연산/비용변경없음.",
        "selection": "주조건diff_mask25,mask5는양성만제외하는대조. full/standard동일25제외로효과분리. 각조건10/20최종AP최대(1e-12동률이른epoch). 각무제외대조보다AP+.1%p,R100손실<=.1%p,계열AP손실<=.25%p인지보고. 모두보고하며이번은시드1예비까지;시험/원고자동승격없음.",
        "diagnosis": "검증전에기존세선택모델을학습그래프FP32 eval에서확인. 각1/6/11/16epoch순열균등24배치:중복/회색항개수,softmax오답logit기울기몫. 실제학습의파라미터기여율아님. 실제학습도모든batch마스크수를기록.",
        "novelty": "같은장소오답제외 자체를신규성으로주장하지않는다. 새추론모듈/파라미터없음."})
    print("PREPARED", counts, flush=True)


def diagnose() -> None:
    """기존 선택 모델의 학습 분모 충돌을 결과 선택 전에 측정한다."""
    spec = protocol(True)
    state = torch.load(base.PREPARED, map_location="cpu", weights_only=False)
    positions = torch.as_tensor(state["poses"][:, :3, 3], device="cuda", dtype=torch.float64)
    frames = torch.as_tensor(state["frames"], device="cuda")
    seq = np.asarray(state["sequences"])
    assert {int(s): np.unique(state["frames"][seq == s]).tolist() for s in np.unique(seq)} == {
        0: [0], 1: [1], 2: [2], 3: [3], 4: [100], 5: [100], 6: [100], 7: [100]}
    original = torch.load(base.GRAPH, map_location="cpu", weights_only=False)
    graph = base.temporal_subgraph(original, np.arange(original.num_nodes)).cuda()
    del original
    probes = []
    for epoch in (1, 6, 11, 16):
        trips = base.triplets(epoch).copy()
        np.random.RandomState(100000+epoch).shuffle(trips)
        for index in np.unique(np.linspace(0, 178, 24).astype(int)):
            probes.append((epoch, index, trips[index*1024:(index+1)*1024]))
    results = {}
    for arm in ("diff_mask25", "full_mask25", "standard_mask25"):
        name = CFG["models"][arm]
        epoch = spec["references"][name]["selected"]["epoch"]
        net = model(arm).cuda().eval()
        net.load_state_dict(torch.load(reference_folder(name)/f"checkpoints/checkpoint_epoch_{epoch}.pth",
                                      map_location="cpu", weights_only=False)["model_state_dict"])
        assert net.gnn.key_scale is None
        with torch.no_grad():
            embeddings = F.normalize(net(graph).float(), dim=1)
        rows = []
        with torch.no_grad():
            for e, index, batch in probes:
                trips = torch.as_tensor(batch, device="cuda")
                a, p, n = trips.T
                x, y, z = embeddings[a], embeddings[p], embeddings[n]
                logits = torch.cat(((x@y.T)/.1, (x*z).sum(1, keepdim=True)/.1), 1)
                prob = logits.softmax(1)
                negative_mass = 1.-prob[:, :len(a)].diagonal()
                record = {"epoch_order": e, "batch": int(index), "rows": len(batch)}
                for radius in (5., 25.):
                    cross, hard = exclusion_masks(positions, frames, trips, radius)
                    mask = torch.cat((cross, hard[:, None]), 1)
                    fraction = (prob*mask).sum(1)/negative_mass
                    record[str(int(radius))] = {"cross_pairs": int(cross.sum()),
                        "rows_with_cross": int(cross.any(1).sum()), "hard_pairs": int(hard.sum()),
                        "mean_negative_logit_gradient_fraction": float(fraction.mean()),
                        "mean_removed_probability": float((prob*mask).sum(1).mean())}
                # 작은 독립 CPU 3D 거리 표본으로 마스크 경계를 검산한다.
                for radius in (5., 25.):
                    cross, hard = exclusion_masks(positions, frames, trips[:8], radius)
                    ix = batch[:8]
                    pos, fr = state["poses"][:, :3, 3].astype(np.float64), state["frames"]
                    expected = (np.linalg.norm(pos[ix[:, 0], None]-pos[ix[:, 1]][None], axis=2) <= radius)
                    expected &= fr[ix[:, 0], None] == fr[ix[:, 1]][None]
                    np.fill_diagonal(expected, False)
                    np.testing.assert_array_equal(cross.cpu().numpy(), expected)
                    h = (np.linalg.norm(pos[ix[:, 0]]-pos[ix[:, 2]], axis=1) <= radius) & (fr[ix[:, 0]] == fr[ix[:, 2]])
                    np.testing.assert_array_equal(hard.cpu().numpy(), h)
                rows.append(record)
        count = sum(r["rows"] for r in rows)
        combined = {str(r): {key: sum(x[str(r)][key]*(x["rows"] if key.startswith("mean_") else 1) for x in rows)
                                /(count if key.startswith("mean_") else 1)
                    for key in rows[0][str(r)]} for r in (5, 25)}
        results[name] = {"rows": count, "batches": rows, "aggregate": combined}
        print("DIAGNOSE", name, count, combined, flush=True)
        del net, embeddings
    write_json(OUT/"training_diagnosis.json", {"results": results, "test_used": False,
        "scope": spec["diagnosis"], "mask_numpy_checks": len(probes)*3*2,
        "sources": {str((OUT/"protocol.json").relative_to(REPO)): sha256(OUT/"protocol.json")}})


class AuditedExclusion(torch.nn.Module):
    """실제 배치 순서와 동기화한 위치 라벨로 기존 손실 분모만 바꾼다."""
    def __init__(self, original: torch.nn.Module, state: dict, radius: float) -> None:
        super().__init__()
        self.temperature = original.temperature
        self.positions = torch.as_tensor(state["poses"][:, :3, 3], device="cuda", dtype=torch.float64)
        self.frames = torch.as_tensor(state["frames"], device="cuda")
        self.radius = radius
        self.ordered = None
        self.offset = 0
        self.records = []

    def forward(self, anchors, positives, negatives):
        size = len(anchors)
        batch = self.ordered[self.offset:self.offset+size]
        assert len(batch) == size and self.offset % 1024 == 0
        self.offset += size
        cross, hard = exclusion_masks(self.positions, self.frames, batch, self.radius)
        self.records.append({"rows": size, "cross_excluded": int(cross.sum()),
            "hard_excluded": int(hard.sum()), "rows_with_cross": int(cross.any(1).sum()),
            "zero_negative_rows": int(((~cross).sum(1)-1+(~hard).long() == 0).sum())})
        return excluded_infonce(anchors, positives, negatives, cross, hard, self.temperature)


def train(arm: str) -> None:
    """기존 학습 루프를 보존하고 criterion만 프로세스 안에서 주입한다."""
    import gnn.trainer as trainer_module
    original_class = trainer_module.GNNTrainer
    state = torch.load(base.PREPARED, map_location="cpu", weights_only=False)
    audits = []

    class ExclusionTrainer(original_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert self.context_aux_lambda == self.phase_token_aux_lambda == 0
            self.criterion = AuditedExclusion(self.criterion, state, CFG["radius"][arm])

        def train_epoch(self, *args, **kwargs):
            ordered = base.triplets(self.epoch+1).copy()
            rng = np.random.RandomState()
            rng.set_state(np.random.get_state())
            rng.shuffle(ordered)
            self.criterion.ordered = torch.as_tensor(ordered, device="cuda")
            self.criterion.offset, self.criterion.records = 0, []
            result = super().train_epoch(*args, **kwargs)
            assert self.criterion.offset == len(ordered) == 183249
            assert len(self.criterion.records) == 179
            audits.append({"epoch": self.epoch+1, "triplets": array_hash(ordered),
                "batches": self.criterion.records})
            write_json(base.root(1, arm)/"loss_batch_audit.json", {"epochs": audits, "radius": CFG["radius"][arm]})
            return result

    trainer_module.GNNTrainer = ExclusionTrainer
    try:
        base.train(1, arm)
    finally:
        trainer_module.GNNTrainer = original_class
    complete = read(base.root(1, arm)/"training_complete.json")
    for record, loss_audit in zip(complete["orders"], read(base.root(1, arm)/"loss_batch_audit.json")["epochs"]):
        assert record["triplets"] == loss_audit["triplets"]


def report() -> None:
    """최종 AP 선택과 동일 구조 손실 변경 효과를 함께 보고한다."""
    spec = protocol()
    models = {}
    for arm in ARMS:
        candidates = []
        for epoch in (10, 20):
            rows = {s: read(base.root(1, arm)/f"epoch{epoch}"/(s.replace("/", "_")+".json")) for s in spec["val"]}
            candidates.append({"epoch": epoch, **{m: base.macro({s: r["metrics"][m] for s, r in rows.items()})
                                                  for m in ("raw", METHOD)}})
        chosen = max(candidates, key=lambda x: (round(x[METHOD]["macro"]["AP"], 12), -x["epoch"]))
        models[arm] = {"selected": chosen, "candidates": candidates}
        print("FINAL", arm, chosen["epoch"], chosen[METHOD]["macro"], flush=True)
    effects, checks = {}, {}
    for arm, data in models.items():
        name = CFG["models"][arm]
        now, ref = data["selected"][METHOD], spec["references"][name]["selected"][METHOD]
        effects[arm] = {"reference": name,
            "macro": {m: 100*(now["macro"][m]-ref["macro"][m]) for m in base.exp.METRICS},
            "families": {f: {m: 100*(now["families"][f][m]-r[m]) for m in base.exp.METRICS}
                         for f, r in ref["families"].items()}}
        checks[arm] = {"AP": effects[arm]["macro"]["AP"] >= .1,
            "R100": effects[arm]["macro"]["R@100P"] >= -.1,
            "family_AP": all(r["AP"] >= -.25 for r in effects[arm]["families"].values())}
    write_json(OUT/"summary.json", {"per_seed": {"1": models}, "references": spec["references"],
        "effects_pp": effects, "pilot_checks": checks, "protocol_sha256": sha256(OUT/"protocol.json"),
        "test_used": False})


base.OUT, base.CONFIG, base.CFG, base.ARMS = OUT, CONFIG, CFG, ARMS
base.model, base.protocol = model, protocol

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "diagnose", "train", "val", "report"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--epoch", type=int, choices=(10, 20))
    parser.add_argument("--sequence")
    args = parser.parse_args()
    torch.set_num_threads(4 if args.action == "train" else 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.WARNING)
    if args.action == "prepare":
        prepare()
    elif args.action == "diagnose":
        diagnose()
    elif args.action == "train":
        train(args.arm)
    elif args.action == "val":
        previous.validation(1, args.arm, args.epoch, args.sequence)
    else:
        report()
