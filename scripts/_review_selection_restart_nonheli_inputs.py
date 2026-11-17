#!/usr/bin/env python3
"""D: 이전 후 비누적 ImLPR 입력을 서버의 보존 바이트와 전수 대조한다."""
from pathlib import Path
import subprocess
import time

from _review_selection_bev import read,write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/pdv2_review_selection_full_20261105"


def main():
    status=OUT/"restart_nonheli_inputs.json"
    migration=OUT/"restart_imlpr_storage.json"
    while not migration.exists() or read(migration)["status"]!="complete":
        write(status,dict(status="waiting_storage_migration",updated_unix=time.time()));time.sleep(10)
    write(status,dict(status="server_checksum_sync",started_unix=time.time()))
    remote="joms-gpu:/home/qkrrlans00/Neural-Spectral-Codec/results/pdv2_review_selection_full_20261105/inputs/imlpr/data/test_odom/"
    local=OUT/"inputs/imlpr/data/test_odom"
    subprocess.run(["rsync","-acO","--no-owner","--no-group","--no-perms","--exclude=helipr_*",remote,str(local)+"/"],check=True)
    write(status,dict(status="complete",server_all_file_checksum_comparison=True,completed_unix=time.time()))


if __name__=="__main__":main()
