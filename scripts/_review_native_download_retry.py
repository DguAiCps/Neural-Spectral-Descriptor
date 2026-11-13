#!/usr/bin/env python3
"""공개 HeLiOS archive의 일시적인 다운로드 한도를 간격을 두고 재시도한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
import time

import requests

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105/learned_sources"
SOURCE = Path("/mnt/d/NSD_datasets/_review_native_20261105/helios")


def main() -> None:
    inventory = json.loads((OUT / "helios_drive_inventory.json").read_text())
    chosen = [r for r in inventory if r["path"].startswith("data_validation/Roundabout") and r["path"].endswith("LiDAR.tar.gz")]
    previous = [int(p.stem.rsplit("round", 1)[1]) for p in OUT.glob("helios_download_retry_round*.json")]
    first_attempt = max(previous, default=-1) + 1
    for attempt in range(first_attempt, first_attempt + 13):
        rows = []
        for record in chosen:
            target = SOURCE / record["path"]
            complete = target.with_suffix(".gz.complete")
            if complete.exists():
                rows.append({"path": record["path"], "complete": True, "reused": True})
                continue
            url = "https://drive.usercontent.google.com/download"
            row = {"path": record["path"], "attempt": attempt, "unix_time": time.time()}
            try:
                with requests.get(url, params={"id": record["id"], "export": "download", "authuser": "0", "confirm": "t"}, stream=True, timeout=(30, 120)) as response:
                    row.update(http_status=response.status_code, content_type=response.headers.get("Content-Type"))
                    if response.status_code != 200 or "text/html" in response.headers.get("Content-Type", ""):
                        text = response.text
                        row["quota_exceeded"] = "quota" in text.lower() or "too many users" in text.lower()
                        (OUT / f"helios_download_attempt{attempt}_{target.parent.name}.html").write_text(text)
                    else:
                        temp = target.with_suffix(".gz.retry.tmp")
                        h = hashlib.sha256()
                        with temp.open("wb") as stream:
                            for block in response.iter_content(1024 * 1024):
                                stream.write(block)
                                h.update(block)
                        with tarfile.open(temp) as archive:
                            members = archive.getmembers()
                            assert any(m.isfile() for m in members)
                            assert all(m.name == "LiDAR" or m.name.startswith("LiDAR/") for m in members)
                        temp.replace(target)
                        complete.write_text(json.dumps({"sha256": h.hexdigest(), "bytes": target.stat().st_size, "unix_time": time.time()}))
                        row.update(complete=True, sha256=h.hexdigest(), bytes=target.stat().st_size)
            except Exception as exc:
                row["error"] = repr(exc)
            rows.append(row)
            print(json.dumps(row), flush=True)
        report = {"attempt": attempt, "rows": rows, "completed": sum(r.get("complete", False) for r in rows), "total": len(chosen), "unix_time": time.time()}
        (OUT / f"helios_download_retry_round{attempt}.json").write_text(json.dumps(report, indent=2))
        (OUT / "helios_download_retry_status.json").write_text(json.dumps(report, indent=2))
        if report["completed"] == report["total"]:
            break
        time.sleep(4 * 3600)


if __name__ == "__main__":
    main()
