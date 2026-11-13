#!/usr/bin/env python3
"""전체 선택 대조의 보존 가중치를 추론 텐서로 중복 제거하고 동결한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _review_selection_bev import sha, read, write, VAL

OUT = ROOT / "results/pdv2_review_selection_full_20261105"


def tensor_hash(state: dict) -> tuple:
    """추론 텐서의 이름·형태·자료형·바이트를 해시한다."""
    import torch
    h = hashlib.sha256()
    count, elements = 0, 0
    for name, tensor in sorted(state.items()):
        assert torch.is_tensor(tensor), name
        tensor = tensor.detach().cpu().contiguous()
        assert torch.isfinite(tensor).all(), name
        h.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
        h.update(tensor.numpy().tobytes())
        count += 1
        elements += tensor.numel()
    return h.hexdigest(), count, elements


def epoch_of(method: str, seed: str, path: str, record: dict) -> int:
    """실제로 완료한 학습 횟수를 에포크 기준으로 기록한다."""
    name = Path(path).name
    if method == "helios":
        return int(record["candidates"][name]["epoch"])
    if method == "imlpr":
        return int(record["best"]["at_start_of_epoch"])-1 if name.endswith("_best.pth") else 100
    if method == "bevplace":
        return int(record["best"]["epoch"])+1 if name.startswith("model_best") else 40
    match = re.search(r"(\d+)(?:\.pth\.tar|\.ckpt|\.pth)$", name)
    assert match, name
    return int(match[1]) + (1 if method == "disco" else 0)


def main() -> None:
    """평가 전에 후보 파일·텐서와 공통 선택 규약을 고정한다."""
    import torch
    torch.set_num_threads(2)
    target = OUT / "retained_protocol.json"
    assert not target.exists(), "기존 명세를 덮어쓰지 않는다."
    source = OUT / "retained_source_inventory.json"
    previous = read(source)
    methods = {}
    for method, entry in previous["methods"].items():
        if "seeds" not in entry:
            continue
        methods[method] = {}
        for seed, meta in entry["seeds"].items():
            record = read(ROOT / meta["selection_file"])
            groups = {}
            for row in meta["retained_candidates"]:
                path = ROOT / row["path"]
                blob = torch.load(path, map_location="cpu", weights_only=False)
                if method == "ringsharp":
                    state = {"model."+k:v for k,v in blob["model"].items()}
                    state.update({"trans_cnn."+k:v for k,v in blob["trans_cnn"].items()})
                elif method in ("helios", "imlpr"):
                    state = {k.removeprefix("module."): v for k,v in blob.items()}
                else:
                    state = blob["state_dict"]
                digest, count, elements = tensor_hash(state)
                candidate = dict(path=row["path"], sha256=sha(path), model_tensor_sha256=digest,
                    epoch=epoch_of(method,seed,row["path"],record), tensors=count, elements=elements)
                groups.setdefault(digest, []).append(candidate)
            unique = []
            for digest, aliases in groups.items():
                aliases.sort(key=lambda r:(r["epoch"], r["path"]))
                representative = dict(aliases[0], aliases=aliases)
                representative["id"] = f"e{representative['epoch']:03d}_{digest[:12]}"
                unique.append(representative)
            unique.sort(key=lambda r:(r["epoch"],r["id"]))
            extra = {}
            if method == "seqot":
                fe = record["stage1"]["chosen"]
                extra["fixed_stage1"] = fe
                extra["stage_scope"] = "보존된 stage2 모델의 선택 대조; stage1 목적 대조와 필요한 후속학습은 별도 단계"
            methods[method][seed] = dict(candidates=unique, original_candidate_files=len(meta["retained_candidates"]),
                selection_source=meta["selection_file"], selection_sha256=sha(ROOT/meta["selection_file"]), **extra)
            write(OUT / "catalog_progress.json", dict(method=method, seed=seed, unique=len(unique), updated_unix=time.time()))
            print(method,seed,len(meta["retained_candidates"]),"->",len(unique),flush=True)
    write(target, dict(created_unix=time.time(), methods=methods,
        val=[s+"/"+q for s,q in VAL], n_queries_per_candidate=99236,
        queries="KISS keyframes; 30<=q<N-15", candidates="j<=q-30",
        scoring="방법별 현재 주표의 원래 추론·검색·점수 유지; scorer 추가 없음",
        selection="동일 후보집합·동일 원장에서 시드별 최종 AP 최대와 재방문 R@1 최대를 각각 선택",
        aggregation="양성 시퀀스->KITTI/KITTI360,HeLiPR,MulRan 세 계열 균등 평균",
        tie_break="앞 completed epoch, 이후 id 사전순", test_policy="모든 검증 후보 평가 후 선택 동결. 시험으로 설정을 선택하지 않음",
        source_sha256={str(source.relative_to(ROOT)):sha(source),str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}))


if __name__ == "__main__":
    main()
