#!/usr/bin/env python3
"""helios_scratch pilot finalizer (server host, no torch).

Writes COMPUTE_DONE.json only when every planned evaluation exists (2..20 per
candidate, 2..40 for an extended one), the calibration finished and
verify_ledgers.all_ok is true; otherwise COMPUTE_FAILED.json with reasons.
Always writes official_pending.json: the official one-scan recheck is left for
root scheduling and is not started by the supervisor.
Usage: python3 scripts/_helios_scratch_pilot_20261108_finalize.py <work dir>
"""
import json
import os
import sys
from datetime import datetime, timezone

CW = "/workspace/Neural-Spectral-Codec/_external/overnight_repro_20261108/helios_scratch"
OLD = "/workspace/Neural-Spectral-Codec/_external/helios_official_pdv2/runs/Oct12_12-22-10/HeLiOS_default_trainning_10.pth"


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main(w):
    dec = load(f"{w}/decision_20.json")
    ext = (dec or {}).get("extend")
    why, summ, pending = [], {}, []
    for c in ("lr5e-4", "lr1e-4"):
        s = load(f"{w}/{c}/val_summary.json")
        summ[c] = s
        want = list(range(2, (40 if c == ext else 20) + 1, 2))
        got = [r["epoch"] for r in (s or {}).get("rows", [])]
        if got != want:
            why.append(f"{c}: evaluated epochs {got} != {want}")
        if s and s.get("val_best"):
            pending.append({"checkpoint": s["val_best"]["checkpoint"], "out": f"{CW}/{c}/official_val_best.json"})
    pending.append({"checkpoint": OLD, "out": f"{CW}/old_seed1_e10_b1.json"})
    cal = load(f"{w}/old_seed1_e10_b32.json") or {}
    if not cal.get("finished"):
        why.append("calibration (old seed1 e10, B=32) not finished")
    ver = load(f"{w}/verify_ledgers.json") or {}
    if not ver.get("all_ok"):
        why.append("verify_ledgers all_ok is not true")
    if dec is None:
        why.append("decision_20.json missing")
    out = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "summaries": summ,
           "decision": dec, "verify_all_ok": ver.get("all_ok"),
           "calibration_old_seed1_e10_b32": cal.get("family_macro_r1"),
           "official_single_scan_recheck": "pending (root schedules; flags/OFFICIAL_APPROVED)",
           "note": "B=32 batched-encoder raw val values (coordination validation_encoding_decision)"}
    json.dump({"items": pending, "status": "pending (not started by supervisor)",
               "how": "touch flags/OFFICIAL_APPROVED; then per item, one GPU stage at a time: docker exec ... "
                      "python3 scripts/_helios_scratch_pilot_20261108.py eval-official --checkpoint <c> --out <o>"},
              open(f"{w}/official_pending.json", "w"), indent=1)
    if why:
        out["reasons"] = why
        json.dump(out, open(f"{w}/COMPUTE_FAILED.json", "w"), indent=1)
        print("COMPUTE_FAILED", why)
        return 1
    out["complete"] = True
    json.dump(out, open(f"{w}/COMPUTE_DONE.json", "w"), indent=1)
    print("COMPUTE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
