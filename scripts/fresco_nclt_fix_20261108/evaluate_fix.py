#!/usr/bin/env python3
"""NCLT9 수정 재평가의 공식 MATLAB 단계와 원장 (서버 호스트, 2026-11-08).

  matlab --seq S     기존 evaluate.cmd_matlab: 새 NAS/test/<name>/bev.u8 -> keys/fresco/match/cand
  ledger --seq S     기존 evaluate.build_ledger: -> 새 NAS results/ledgers/test/dump_fresco_<name>.npz
기존 evaluate.py를 patch(fixenv.PATCHES)해서 그대로 쓴다. 공식 MATLAB 어댑터는 기존 NAS build/matlab의
fresco_adapter.m(sha256 확인, 읽기만), 간격 30, K=20, 점수·비유한 규칙 모두 기존 그대로.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixenv  # noqa: E402

common, codec_fix, stream, evaluate = fixenv.setup()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["matlab", "ledger"])
    ap.add_argument("--seq", required=True, choices=fixenv.NCLT9)
    a = ap.parse_args()
    assert evaluate.GAP == 30 and evaluate.SCORE_RATIO == 1e5 and evaluate.NONFINITE_SCORE == -1e12
    sd = json.loads((evaluate.seq_dir("test", *a.seq.split("/")) / "stream_done.json").read_text())
    assert sd["family"] == fixenv.FAMILY_NCLT and sd["driver"] == "fresco_prep_ncltflu", sd
    if a.cmd == "matlab":
        print(json.dumps(evaluate.cmd_matlab("test", a.seq)))
    else:
        print(json.dumps(evaluate.build_ledger("test", a.seq)))


if __name__ == "__main__":
    main()
