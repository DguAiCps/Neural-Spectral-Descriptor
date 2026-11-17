#!/usr/bin/env python3
"""검증 선택이 동결되는 즉시 경로·텐서SHA를 협업용 명세로 전달한다."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _review_selection_bev import read, write, sha

OUT = ROOT / "results/pdv2_review_selection_full_20261105"


def main() -> None:
    """시험 지표를 읽지 않고 검증으로 확정한 선택본만 기록한다."""
    lock = (OUT / "selected_model_watch.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    methods = ("imlpr", "helios", "disco")
    scopes = dict(imlpr="시드별 best/final 두 보존 텐서; 100에포크 전수 선택 아님",
                  helios="보존 10에포크 주기 및 best의 고유 8/9/9개 텐서; 전체80에포크 선택 아님",
                  disco="시드별 원래20에포크의 전체 보존 가중치")
    delivered = []
    protocol = read(OUT / "retained_protocol.json")
    while len(delivered) < len(methods):
        for method in methods:
            if method in delivered:
                continue
            selection = OUT / "retained" / f"selection_{method}.json"
            validation = OUT / "retained" / f"validation_{method}.json"
            if not selection.exists() or not validation.exists():
                continue
            frozen = read(selection)
            if sha(validation) != frozen["validation_sha256"]:
                continue
            assert not frozen["test_used_for_selection"]
            assert frozen["protocol_sha256"] == sha(OUT / "retained_protocol.json")
            values = read(validation)
            models = {}
            for seed, choices in frozen["selected"].items():
                models[seed] = dict(original=frozen["original_choices"][seed]["candidate"],
                    choices={objective: dict(row, validation_metrics=values[seed][row["id"]]["macro"])
                             for objective, row in choices.items()})
            count = sum(len(r["candidates"]) for r in protocol["methods"][method].values())
            target = OUT / f"selection_delivery_{method}.json"
            if not target.exists():
                write(target, dict(created_unix=time.time(), selection_path=str(selection.relative_to(ROOT)),
                    selection_sha256=sha(selection), candidate_count=count, validation_ledgers=9 * count,
                    common_validation_queries_per_candidate=99236, test_used_for_selection=False,
                    test_status="검증에서 동결; 시험 완료 여부는 별도 report를 참조", scope=scopes[method], models=models))
            assert read(target)["selection_sha256"] == sha(selection)
            subprocess.run(["rsync", "-aO", "--no-owner", "--no-group", "--no-perms", str(target),
                "joms-gpu:/home/qkrrlans00/Neural-Spectral-Codec/" + str(OUT.relative_to(ROOT)) + "/"], check=True)
            delivered.append(method)
        write(OUT / "selected_model_watch_state.json", dict(status="complete" if len(delivered) == len(methods) else "waiting_selection",
            delivered=delivered, pid=os.getpid(), updated_unix=time.time()))
        if len(delivered) < len(methods):
            time.sleep(15)


if __name__ == "__main__":
    from _review_selection_local_reservation import require_background_available
    require_background_available()
    main()
