#!/usr/bin/env python3
"""BEVPlace++ 원래 학습을 별도로 재실행하고 매 에포크·RNG를 보존한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _review_selection_bev import read, write, sha

OUT = ROOT / "results/pdv2_review_selection_full_20261105"


class EpochBoundary(Exception):
    """체크포인트와 RNG가 저장된 에포크 경계의 정상 중단이다."""


def prepare() -> dict:
    """원래 설정과 120개 미래 후보를 학습 시작 전에 고정한다."""
    path = OUT / "bev_rerun_protocol.json"
    if path.exists():
        return read(path)
    sources, configs = {}, {}
    for seed in (1,2,3):
        suffix = "" if seed == 1 else f"_s{seed}"
        p = ROOT / f"results/pdv2_bevplace_official{suffix}/train_selection.json"
        original = read(p)
        configs[str(seed)] = original["_meta"]["flags"]
        sources[str(p.relative_to(ROOT))] = sha(p)
    for name in ("scripts/_bevplace_official_train.py", "scripts/_bevplace_official_bev.py",
                 "scripts/_review_selection_full_bev_train.py", "_external/BEVPlace2/main.py",
                 "_external/BEVPlace2/REIN.py", "_external/BEVPlace2/kitti_dataset.py"):
        sources[name] = sha(ROOT / name)
    spec = dict(created_unix=time.time(), identity="통제된 재실행; 과거 학습의 비트 일치 복원이라고 주장하지 않음",
        seeds=[1,2,3], epoch_labels=list(range(40)), original_flags=configs,
        train="원래 GT 학습8·72,306프레임·공식 코드와 어댑터·Adam1e-4·batch4·40epoch",
        initialization="시드별 새 빈 cache에서 공식 getClusters와 ImageNet 초기화; 원래 학습 로그와 같은 경로",
        preserved="원래 train_epoch, make_val_sets, validate, optimizer, RNG 저장·복원 코드는 변경하지 않음",
        boundary="원래 체크포인트/RNG/선택JSON 저장 후 별도 에포크 사본을 만들고 종료; 다음 작업에서 원래 resume 사용",
        legacy_validation="원래 GT/양방향 R@1도 실행·보존; 공통 AP/R@1 선택에는 별도 KISS검증9 원장만 사용",
        scoring="새 scorer 없음. 매epoch 가중치로 원래 BEVPlace++ -L2 최종 점수",
        candidates="각 재실행 시드의 모든40epoch(0..39); 시드별 AP 최대/R@1 최대, 동률은 앞epoch",
        test_policy="40개 검증 완료 후 두 선택을 동결한 뒤에만 시험46 평가",
        storage="최신 resume 상태 외 모든 epoch 가중치와 RNG를 보존",
        source_sha256=sources)
    write(path, spec)
    return spec


def train_epoch(seed: int, target_epoch: int) -> None:
    """동결된 학습의 한 에포크를 실행하고 재개 가능한 경계에서 끝낸다."""
    spec = prepare()
    for p,digest in spec["source_sha256"].items():
        assert sha(ROOT/p) == digest, p
    import _bevplace_official_train as original
    home = OUT / "bev_rerun" / f"s{seed}"
    weights = home / "epochs"
    weights.mkdir(parents=True, exist_ok=True)
    target = weights / f"epoch_{target_epoch:03d}.pth.tar"
    if target.exists() and target.with_suffix(".json").exists():
        assert sha(target) == read(target.with_suffix(".json"))["sha256"]
        return
    selection = home / "train_selection.json"
    resume = read(selection)["_meta"]["logdir"] if selection.exists() else ""
    if resume:
        previous = max(r["epoch"] for r in read(selection)["epochs"])
        assert previous + 1 == target_epoch, (previous, target_epoch)
    else:
        assert target_epoch == 0
    save_json = original._write_json

    def boundary(path: Path, record: dict) -> None:
        save_json(path, record)
        if Path(path) != selection or not record.get("epochs"):
            return
        epoch = int(record["epochs"][-1]["epoch"])
        run = Path(record["_meta"]["logdir"])
        checkpoint = run / "checkpoint.pth.tar"
        rng = run / "rng_state.pt"
        dest = weights / f"epoch_{epoch:03d}.pth.tar"
        shutil.copy2(checkpoint, dest)
        shutil.copy2(rng, weights / f"rng_{epoch:03d}.pt")
        write(dest.with_suffix(".json"), dict(created_unix=time.time(), seed=seed, epoch=epoch,
            checkpoint=str(dest.relative_to(ROOT)), sha256=sha(dest), legacy=record["epochs"][-1],
            rng_sha256=sha(weights/f"rng_{epoch:03d}.pt"), protocol_sha256=sha(OUT/"bev_rerun_protocol.json")))
        write(home / "status.json", dict(status="epoch_complete", seed=seed, completed_epoch=epoch,
            next_epoch=epoch+1, updated_unix=time.time()))
        if epoch >= target_epoch:
            raise EpochBoundary()

    original._write_json = boundary
    args = argparse.Namespace(smoke=0, resume=resume, selection_out=selection)
    argv = ["--seed",str(seed),"--nEpochs","40","--runsPath",str(home/"runs")+"/",
            "--cachePath",str(home/"cache")+"/"]
    write(home / "status.json", dict(status="training", seed=seed, epoch=target_epoch,
        updated_unix=time.time(), pid=__import__("os").getpid(), argv=argv))
    try:
        original.train(args, argv)
    except EpochBoundary:
        if getattr(original.M,"writer",None) is not None:
            original.M.writer.close()
        print("EPOCH_BOUNDARY_COMPLETE",seed,target_epoch,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","epoch"))
    parser.add_argument("--seed",type=int,choices=(1,2,3))
    parser.add_argument("--epoch",type=int,choices=range(40))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        train_epoch(args.seed,args.epoch)
