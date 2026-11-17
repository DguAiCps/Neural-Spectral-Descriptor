#!/usr/bin/env python3
"""완성된 ImLPR native 입력만 해시 검증 후 서버 큐에 공개한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/pdv2_review_native_20261105/imlpr"
REMOTE = "/home/qkrrlans00/Neural-Spectral-Codec"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def main() -> None:
    """시퀀스 완료를 기다리며 전달 실패·완료 상태를 보존한다."""
    status = ROOT / "relay_status.json"
    completed = []
    for sequence in ("DCC01", "DCC02", "DCC03"):
        done = ROOT / "inputs" / sequence / "done.json"
        while not done.exists():
            write(status, {"waiting": sequence, "completed": completed, "unix_time": time.time()})
            time.sleep(30)
        record = json.loads(done.read_text())
        manifest = ROOT / f"transfer_manifest_{sequence}.json"
        files = [ROOT / "inputs" / sequence / name for name in
                 ("done.json", "index.npz", "metadata.pickle", "official_input_checks.json")]
        for k in (15, 25):
            images = sorted((ROOT / f"k{k}/data/validation" / sequence).glob("[0-9]*.npy"))
            assert len(images) == record["n_frames"], (sequence, k, len(images))
            files.extend(images)
        value = {str(p.relative_to(REPO)): sha(p) for p in files}
        write(manifest, {"sequence": sequence, "sha256": value})
        listing = ROOT / f"transfer_files_{sequence}.txt"
        listing.write_text("\n".join(list(value) + [str(manifest.relative_to(REPO))]) + "\n")
        write(status, {"transferring": sequence, "files": len(value), "completed": completed})
        subprocess.run(["rsync", "-az", "--files-from=" + str(listing), str(REPO) + "/", "joms-gpu:" + REMOTE + "/"], check=True)
        code = f'''import hashlib,json,time
from pathlib import Path
r=Path({REMOTE!r})
m=r/{str(manifest.relative_to(REPO))!r}
d=json.loads(m.read_text())
for name,expected in d["sha256"].items():
 h=hashlib.sha256()
 with (r/name).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 assert h.hexdigest()==expected,name
dest=r/"results/pdv2_review_native_20261105/imlpr/inputs"/d["sequence"]/"transfer_complete.json"
temp=dest.with_suffix(".tmp")
temp.write_text(json.dumps({{"verified_files":len(d["sha256"]),"manifest_sha256":hashlib.sha256(m.read_bytes()).hexdigest(),"unix_time":time.time()}},indent=2))
temp.replace(dest)
print(d["sequence"],len(d["sha256"]))
'''
        subprocess.run(["ssh", "joms-gpu", "python3", "-"], input=code, text=True, check=True)
        completed.append(sequence)
        write(status, {"completed": completed, "unix_time": time.time()})


if __name__ == "__main__":
    main()
