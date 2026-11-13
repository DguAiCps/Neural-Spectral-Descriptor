#!/usr/bin/env python3
"""NCLT 수정(spec fresco_nclt_G3prime_v1) 구현 정확성 검사 V2~V5 (2026-11-08). 기술자·검색·매칭·점수 계산 없음.

  local   (로컬 D, CPU 1) 고정 8프레임과 합성 장면의 공식 입력 바이트·행·열·지면·높이를 spec 식과 numpy 이식으로 검사,
          서버 C++ 대조용 입력 묶음을 OUT/inputs/에 쓴다.
  server  (서버 호스트, CPU 1) 입력 묶음을 새 드라이버 fresco_prep_ncltflu, 기존 G3 드라이버(기존 NAS, 읽기만),
          같은 명령으로 다시 빌드한 G3 드라이버에 넣어 BEV·stats를 새 NAS prefreeze/에 쓴다.
  compare (로컬 D) 서버 출력과 numpy 이식의 BEV 비트·stats를 비교한다.
numpy 이식 = OVN/fresco_nclt_diagnosis/server_mirror/d34_server_f32.py의 emulate(공식 C++과 고정 8프레임 BEV 비트 동일 확인됨).
수정 이식은 같은 본문에서 `rr = (-elev + 10.67) / 1.3335` 한 줄의 10.67만 30.67로 바꾼 것이다(spec only_changed_token).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OVN = Path("/mnt/d/NSD_local/results/pdv2_overnight_repro_20261108")
OUT = OVN / "fresco_fix_pipeline/prefreeze"
EMU = OVN / "fresco_nclt_diagnosis/server_mirror/d34_server_f32.py"
EMU_SHA = "4f313435a48c2201f8ee2aef133f4ecf3ad813672da4505839a8a0ce8b7b4b82"
D4B = OVN / "fresco_nclt_diagnosis/d4b_input"
D4REF = OVN / "fresco_nclt_diagnosis/server_mirror/d4_float32.json"
ANG = OVN / "fresco_nclt_geometry/src_primary/s3_laser_angles.csv"
AUD = OVN / "fresco_fix_audit"
DT = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("i", "u1"), ("l", "u1")])
T = np.array([0.002, -0.004, -0.957])
S = np.diag([1.0, -1.0, -1.0])


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cround(a):
    return np.sign(a) * np.floor(np.abs(a) + 0.5)


def laser_angles():
    return np.degrees([float(l.split(",")[1]) for l in ANG.read_text().splitlines() if l.strip()])


def load_emulators():
    src = EMU.read_text()
    assert sha(src.encode()) == EMU_SHA, "emulator changed"
    ns = {"__name__": "d34_exec", "__file__": str(EMU)}
    exec(compile(src, str(EMU), "exec"), ns)
    a, b = src.index("def emulate(ob):"), src.index("def d4():")
    body = src[a:b]
    hook = ("    label = np.where(Gf == 1, 0, label)",
            "    label = np.where(Gf == 1, 0, label)\n    ST.update(occ=occ, label=label, last=last, Z=Z, row=row, col=col)")
    old_line = "    rr = (-elev + 10.67) / 1.3335"
    assert body.count(hook[0]) == 1 and body.count(old_line) == 1 and body.count("10.67") == 1
    meas_line = '"nonground_z_p10_p50_p90": np.percentile(Z[occ & (label != 0)], [10, 50, 90]).tolist(),'
    assert body.count(meas_line) == 1
    body = body.replace(meas_line, '"nonground_z_p10_p50_p90": np.percentile(Z[occ & (label != 0)], [10, 50, 90]).tolist() '
                                   'if (occ & (label != 0)).any() else None,')   # 계량 전용 줄(합성 소표본 빈 배열)
    g3 = body.replace("def emulate(ob):", "def emu_g3(ob, ST):").replace(*hook)
    fix = g3.replace("def emu_g3(ob, ST):", "def emu_fix(ob, ST):").replace(
        old_line, "    rr = (-elev + 30.67) / 1.3335")
    exec(compile(g3, "emu_g3", "exec"), ns)
    exec(compile(fix, "emu_fix", "exec"), ns)
    return ns


def spec_bytes(u: np.ndarray) -> bytes:
    """spec.json official_file_bytes를 글자 그대로 옮긴 독립 구현(codec_fix와 따로 쓴다)."""
    xb = u[:, 0].astype(np.float32) * np.float32(0.005) - np.float32(100.0)
    yb = u[:, 1].astype(np.float32) * np.float32(0.005) - np.float32(100.0)
    zb = u[:, 2].astype(np.float32) * np.float32(0.005) - np.float32(100.0)
    xL = (xb - np.float32(0.002)).astype(np.float32)
    yL = (-(yb + np.float32(0.004))).astype(np.float32)
    zL = (-(zb + np.float32(0.957))).astype(np.float32)
    return (np.ascontiguousarray(-xL, "<f4").tobytes() + np.ascontiguousarray(yL, "<f4").tobytes()
            + np.ascontiguousarray(-zL, "<f4").tobytes() + np.zeros(len(u), "<f4").tobytes())


def raw_from_flu(P: np.ndarray, laser: np.ndarray) -> bytes:
    """합성 L_flu 점 -> body FRD (p_B = S p_L + t) -> NCLT u16 레코드."""
    B = P @ S + T
    u = np.clip(np.rint((B + 100.0) / 0.005), 0, 65535).astype("<u2")
    rec = np.zeros(len(u), DT)
    rec["x"], rec["y"], rec["z"], rec["l"] = u[:, 0], u[:, 1], u[:, 2], laser
    return rec.tobytes()


def synth_scene():
    """V3(c)(d): 라이다 1.0 m 아래 평면 + x = 6 m 수직 벽(방위 -40..40도, 높이 -1..6 m), 1056열 중심 방위.
    물리 앙각 = -빔각. 아래를 보는 빔(빔각 > 0)은 평면 또는 벽, 나머지 빔은 벽 구간에서만 벽에 맞는다."""
    ang = laser_angles()
    az = np.radians((np.arange(1056) + 0.0) * 360.0 / 1056)
    pts, lid, kind = [], [], []
    for l, a in enumerate(ang):
        el = np.radians(-a)
        for t in az:
            c, s = np.cos(t), np.sin(t)
            d_ground = 1.0 / np.tan(-el) if el < -1e-9 else np.inf
            d_wall = 6.0 / c if (abs(np.degrees(np.arctan2(s, c))) < 40.0) else np.inf
            if d_wall < d_ground:
                z = d_wall * np.tan(el)
                if -1.0 <= z <= 6.0:
                    pts.append((6.0, d_wall * s, z)); lid.append(l); kind.append(1)
            elif np.isfinite(d_ground):
                pts.append((d_ground * c, d_ground * s, -1.0)); lid.append(l); kind.append(0)
    return np.array(pts), np.array(lid, np.uint8), np.array(kind, np.int8)


def run_emu(fn, ob):
    ST = {}
    bev, st, meas, row, col = fn(ob, ST)
    return bev, st, row, col, ST


def cmd_local():
    import codec_fix
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inputs").mkdir(exist_ok=True)
    ns = load_emulators()
    ang = laser_angles()
    native = np.clip(cround((-ang + 10.67) / 1.3335), 0, 31).astype(int)
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "emulator_sha256": EMU_SHA,
           "codec_fix_sha256": sha((HERE / "codec_fix.py").read_bytes()), "checks": {}}
    C = res["checks"]
    # ---- V2 fixed 8: bytes, float64 reference, N, payload round trip
    ref = json.loads(D4REF.read_text())["D4"]["frames"]
    v2, fx = [], []
    for fr in ref:
        raw = (D4B / fr["file"]).read_bytes()
        assert sha(raw) == fr["sha256"]
        r = np.frombuffer(raw, DT)
        u = np.stack([r["x"], r["y"], r["z"]], 1).astype("<u2")
        payload, ob = codec_fix.encode(codec_fix.FAMILY, raw)
        ob_dec = codec_fix.decode(codec_fix.FAMILY, payload)
        a = np.frombuffer(ob, "<f4").reshape(4, -1)
        PL = np.stack([-a[0], a[1], -a[2]], 1).astype(np.float64)
        B64 = u.astype(np.float64) * 0.005 - 100.0
        P64 = (B64 - T) @ S
        old_payload, old_ob = codec_fix._old.encode("nclt_oxford_hdl32", raw)
        v2.append({"file": fr["file"], "n_raw": len(r), "n_official": a.shape[1],
                   "impl_eq_spec_sha256": sha(ob) == sha(spec_bytes(u)),
                   "decode_eq_encode_sha256": sha(ob_dec) == sha(ob),
                   "payload_u16_lossless": bool(np.array_equal(np.frombuffer(payload, "<u2").reshape(-1, 3), u)),
                   "payload_eq_old_g3_payload": payload == old_payload,
                   "old_g3_official_sha256_eq_reference": sha(old_ob) == sha(ns["official_bytes"](raw)[0]),
                   "max_abs_diff_vs_float64_m": float(np.abs(PL - P64).max()),
                   "intensity_block_zero": bool(np.all(a[3] == 0)),
                   "all_finite": bool(np.isfinite(a).all()),
                   "raw_20000_triplets": int(np.all(u == 20000, 1).sum())})
        # ---- V4 emulator metrics after / before (C++ 대조는 compare)
        _, st_g3, _, _, _ = run_emu(ns["emu_g3"], old_ob)
        bev, st, row, col, ST = run_emu(ns["emu_fix"], ob)
        las = r["l"].astype(int)
        exp = 31 - native[las]
        occ, label, last, Z = ST["occ"], ST["label"], ST["last"], ST["Z"]
        si = last[occ]; lab = label[occ]
        up = PL[si, 2]
        h = np.clip(np.trunc((Z[occ] + np.float32(2.0)).astype(np.float64) * 4.0), 0, 255).astype(int)
        ng = lab != 0
        kept = ng & (h > 0)
        rk = lambda v: np.argsort(np.argsort(v))
        np.save(OUT / f"emu_fix_{fr['file']}.npy", bev.astype(np.uint8))
        (OUT / "inputs" / fr["file"]).write_bytes(raw)
        fx.append({"file": fr["file"], "g3_emu_stats_eq_old_driver": st_g3.tolist() == fr["stats_driver"],
                   "stats_fix_emu": st.tolist(), "bev_fix_emu_sha256": sha(bev.astype(np.uint8).tobytes()),
                   "row_eq_laser_expected": float((row == exp).mean()),
                   "abs_drow_le1": float((np.abs(row - exp) <= 1).mean()),
                   "nonground_height0_share": float((h[ng] == 0).mean()),
                   "rank_corr_height_vs_phys_up": float(np.corrcoef(rk(h[kept]), rk(up[kept]))[0, 1]),
                   "row_range": [int(row.min()), int(row.max())]})
    C["V2_fixed8_bytes"] = {"frames": v2, "passed": all(f["impl_eq_spec_sha256"] and f["decode_eq_encode_sha256"]
                                                         and f["payload_u16_lossless"] and f["payload_eq_old_g3_payload"]
                                                         and f["old_g3_official_sha256_eq_reference"]
                                                         and f["n_raw"] == f["n_official"] and f["all_finite"]
                                                         and f["intensity_block_zero"]
                                                         and f["max_abs_diff_vs_float64_m"] <= 2e-5 for f in v2)}
    m = {k: float(np.mean([f[k] for f in fx])) for k in ("row_eq_laser_expected", "abs_drow_le1",
                                                          "nonground_height0_share", "rank_corr_height_vs_phys_up")}
    C["V4_fixed8_emulator"] = {"frames": fx, "mean": m, "targets": {
        "row_eq_laser_expected": "0.8055 +-0.001", "abs_drow_le1": ">= 0.999",
        "nonground_height0_share": "0.0084 +-0.001", "rank_corr_height_vs_phys_up": "0.997 +-0.002"},
        "passed": bool(abs(m["row_eq_laser_expected"] - 0.8055) <= 0.001 and m["abs_drow_le1"] >= 0.999
                       and abs(m["nonground_height0_share"] - 0.0084) <= 0.001
                       and abs(m["rank_corr_height_vs_phys_up"] - 0.997) <= 0.002
                       and all(f["g3_emu_stats_eq_old_driver"] for f in fx)),
        "note": "C++ 드라이버 비트 대조는 compare 단계(V4_cpp)"}
    # ---- V3 synthetic
    sref = json.loads((AUD / "synth_ref.json").read_text())["cases"]
    v3 = {}
    for name in ("synth_beams", "synth_axes"):
        raw = (AUD / f"{name}.bin").read_bytes()
        assert sha(raw) == sref[name]["raw_sha256"]
        _, ob = codec_fix.encode(codec_fix.FAMILY, raw)
        _, _, row, col, _ = run_emu(ns["emu_fix"], ob)
        las = np.frombuffer(raw, DT)["l"].astype(int)
        v3[name] = {"official_sha256_eq_audit": sha(ob) == sref[name]["official_bytes_sha256"],
                    "rows_eq_audit": row.tolist() == sref[name]["rows"],
                    "cols_eq_audit": col.tolist() == sref[name]["cols"]}
        if name == "synth_beams":
            v3[name]["rows_eq_31_minus_native"] = bool(np.all(row == 31 - native[las]))
        else:
            v3[name]["rows"], v3[name]["cols"] = row.tolist(), col.tolist()
            v3[name]["expected"] = {"rows": [23, 23], "cols": [0, 792]}
            v3[name]["matches_expected"] = row.tolist() == [23, 23] and col.tolist() == [0, 792]
        (OUT / "inputs" / f"{name}.bin").write_bytes(raw)
    # float32 수준(양자화 없이) 축 검사
    Bf = np.array([[10.002, -0.004, -0.957], [0.002, 9.996, -0.957]], np.float32)
    Pf = np.stack([Bf[:, 0] - np.float32(0.002), -(Bf[:, 1] + np.float32(0.004)), -(Bf[:, 2] + np.float32(0.957))], 1)
    v3["axes_float32_no_quantization"] = {"L": Pf.tolist(), "expected_L": [[10, 0, 0], [0, -10, 0]],
                                          "max_abs_err": float(np.abs(Pf - np.array([[10, 0, 0], [0, -10, 0]])).max())}
    P, lid, kind = synth_scene()
    raw = raw_from_flu(P, lid)
    (OUT / "inputs" / "synth_scene.bin").write_bytes(raw)
    _, ob = codec_fix.encode(codec_fix.FAMILY, raw)
    a = np.frombuffer(ob, "<f4").reshape(4, -1)
    PL = np.stack([-a[0], a[1], -a[2]], 1)
    bev, st, row, col, ST = run_emu(ns["emu_fix"], ob)
    occ, label, last = ST["occ"], ST["label"], ST["last"]
    surv = np.zeros(len(P), bool); surv[last[occ]] = True
    plab = np.full(len(P), 99); plab[last[occ]] = label[occ]
    gpt = surv & (kind == 0)
    wall_hi = surv & (kind == 1) & (PL[:, 2] > -0.7)
    wall_lo = surv & (kind == 1) & (PL[:, 2] <= -0.7)
    hb = np.trunc((PL[:, 2].astype(np.float32) + np.float32(2.0)).astype(np.float64) * 4.0).astype(int)
    v3["scene"] = {"n_points": int(len(P)), "n_ground_pts": int((kind == 0).sum()), "n_wall_pts": int((kind == 1).sum()),
                   "survivors_after_ordering": int(surv.sum()),
                   "rows_eq_31_minus_native": float((row == 31 - native[lid]).mean()),
                   "ground_rows": sorted(set(row[kind == 0].tolist())),
                   "ground_points_labelled_ground_share": float((plab[gpt] == 0).mean()),
                   "ground_z_L_max_abs_err": float(np.abs(PL[kind == 0, 2] + 1.0).max()),
                   "ground_height_bin_if_not_ground": sorted(set(hb[kind == 0].tolist())),
                   "wall_points_above_ground_0p3_nonground_share": float((plab[wall_hi] != 0).mean()),
                   "wall_base_points_le_0p3_ground_share": float((plab[wall_lo] == 0).mean()) if wall_lo.any() else None,
                   "stats": st.tolist()}
    # 높이 의미: u16 양자화 뒤 평면 z_L = -1.002(5 mm 격자 + 0.957 오프셋이라 -1.000은 표현 불가) -> trunc 경계 아래라 3.
    # float 수준 z_L = -1.0이면 int((z+2)*4) = 4. 벽 점은 높을수록 큰 값(물리 위 = 큰 높이값)인지, BEV 최댓값이 식과 같은지 본다.
    wall = surv & (kind == 1) & (plab != 0)
    zw = PL[wall, 2].astype(np.float32)
    bw = np.clip(np.trunc((zw + np.float32(2.0)).astype(np.float64) * 4.0), 0, 255).astype(int)
    sc = v3["scene"]
    sc.update({"ground_z_L_after_u16": sorted(set(np.round(PL[kind == 0, 2].astype(float), 4).tolist())),
               "ground_bin_float_level_minus1": int(np.trunc((np.float32(-1.0) + np.float32(2.0)) * 4.0)),
               "height_bin_monotone_in_z_for_wall": bool(np.all(np.diff(bw[np.argsort(zw, kind="stable")]) >= 0)),
               "wall_z_range": [float(zw.min()), float(zw.max())],
               "bev_max_eq_formula_of_max_wall_z": int(bev.max()) == int(bw.max()),
               "bev_max": int(bev.max())})
    C["V3_synthetic"] = dict(v3)
    gp = (v3["synth_beams"]["official_sha256_eq_audit"] and v3["synth_beams"]["rows_eq_audit"]
          and v3["synth_beams"]["cols_eq_audit"] and v3["synth_beams"]["rows_eq_31_minus_native"]
          and v3["synth_axes"]["official_sha256_eq_audit"] and v3["synth_axes"]["matches_expected"]
          and v3["axes_float32_no_quantization"]["max_abs_err"] < 1e-5
          and sc["rows_eq_31_minus_native"] == 1.0 and sc["ground_points_labelled_ground_share"] == 1.0
          and sc["wall_points_above_ground_0p3_nonground_share"] == 1.0
          and sc["ground_z_L_max_abs_err"] <= 0.006 and sc["ground_bin_float_level_minus1"] == 4
          and sc["height_bin_monotone_in_z_for_wall"] and sc["bev_max_eq_formula_of_max_wall_z"])
    C["V3_synthetic"]["passed"] = bool(gp)
    np.save(OUT / "emu_fix_synth_scene.bin.npy", bev.astype(np.uint8))
    for name in ("synth_beams", "synth_axes"):
        _, ob = codec_fix.encode(codec_fix.FAMILY, (AUD / f"{name}.bin").read_bytes())
        b, st2, *_ = run_emu(ns["emu_fix"], ob)
        np.save(OUT / f"emu_fix_{name}.bin.npy", b.astype(np.uint8))
        res.setdefault("emu_stats", {})[f"{name}.bin"] = st2.tolist()
    res.setdefault("emu_stats", {})["synth_scene.bin"] = st.tolist()
    for f in fx:
        res["emu_stats"][f["file"]] = f["stats_fix_emu"]
    res["inputs_sha256"] = {p.name: sha(p.read_bytes()) for p in sorted((OUT / "inputs").iterdir())}
    res["passed"] = all(c["passed"] for c in C.values())
    (OUT / "local_checks.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: c["passed"] for k, c in C.items()}), json.dumps(m), json.dumps(sc))
    return 0 if res["passed"] else 1


def cmd_server(inp: Path, out: Path):
    import fixenv
    common, codec_fix, stream, evaluate = fixenv.setup()
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(inp.glob("*.bin"))
    drivers = {"fix": fixenv.NEW_NAS / "build/fresco_prep_ncltflu",
               "g3_old": fixenv.OLD_NAS / "build/fresco_prep_oxford",
               "g3_rebuild": fixenv.NEW_NAS / "build/fresco_prep_oxford_g3_rebuild"}
    rec = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "drivers_sha256": {k: fixenv.sha256_file(p) for k, p in drivers.items()},
           "inputs_sha256": {p.name: fixenv.sha256_file(p) for p in files}, "runs": {}}
    for tag, drv in drivers.items():
        fam = codec_fix.FAMILY if tag == "fix" else "nclt_oxford_hdl32"
        blobs = [codec_fix.encode(fam, p.read_bytes())[1] for p in files]
        work = out / f"work_{tag}"
        work.mkdir(exist_ok=True)
        data = b"".join(len(b).to_bytes(4, "little") + b for b in blobs) + (0).to_bytes(4, "little")
        t0 = time.time()
        r = subprocess.run(["nice", "-n", "10", str(drv), str(work)], input=data, capture_output=True)
        res = {"rc": r.returncode, "stderr_tail": r.stderr.decode(errors="replace")[-600:],
               "sec": round(time.time() - t0, 2), "official_sha256": [sha(b) for b in blobs]}
        if r.returncode == 0:
            a = np.frombuffer(r.stdout, np.uint8).reshape(len(files), 224 * 224 + 16)
            (out / f"bev_{tag}.u8").write_bytes(a[:, :224 * 224].tobytes())
            res["stats"] = {p.name: a[k, 224 * 224:].copy().view("<i4").tolist() for k, p in enumerate(files)}
            res["bev_sha256"] = {p.name: sha(a[k, :224 * 224].tobytes()) for k, p in enumerate(files)}
        rec["runs"][tag] = res
    (out / "server_checks.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({t: r["rc"] for t, r in rec["runs"].items()}))


def cmd_compare(srv: Path):
    loc = json.loads((OUT / "local_checks.json").read_text())
    s = json.loads((srv / "server_checks.json").read_text())
    files = sorted(s["inputs_sha256"])
    assert s["inputs_sha256"] == loc["inputs_sha256"], "inputs differ"
    ref = {f["file"]: f for f in json.loads(D4REF.read_text())["D4"]["frames"]}
    fix = np.frombuffer((srv / "bev_fix.u8").read_bytes(), np.uint8).reshape(-1, 224, 224)
    g3o = np.frombuffer((srv / "bev_g3_old.u8").read_bytes(), np.uint8).reshape(-1, 224, 224)
    g3r = np.frombuffer((srv / "bev_g3_rebuild.u8").read_bytes(), np.uint8).reshape(-1, 224, 224)
    per = {}
    for k, f in enumerate(files):
        emu = np.load(OUT / (f"emu_fix_{f}.npy"))
        per[f] = {"fix_cpp_eq_emu_bev": bool(np.array_equal(fix[k], emu)),
                  "fix_cpp_eq_emu_stats": s["runs"]["fix"]["stats"][f] == loc["emu_stats"][f],
                  "g3_rebuild_eq_g3_old_bev": bool(np.array_equal(g3r[k], g3o[k])),
                  "g3_rebuild_eq_g3_old_stats": s["runs"]["g3_rebuild"]["stats"][f] == s["runs"]["g3_old"]["stats"][f],
                  "fix_differs_from_g3": bool(not np.array_equal(fix[k], g3o[k]))}
        if f in ref:
            per[f]["g3_old_stats_eq_d4_reference"] = s["runs"]["g3_old"]["stats"][f] == ref[f]["stats_driver"]
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "per_input": per,
           "rc": {t: r["rc"] for t, r in s["runs"].items()}, "drivers_sha256": s["drivers_sha256"],
           "server_checks_sha256": sha((srv / "server_checks.json").read_bytes())}
    keys = ("fix_cpp_eq_emu_bev", "fix_cpp_eq_emu_stats", "g3_rebuild_eq_g3_old_bev", "g3_rebuild_eq_g3_old_stats")
    out["passed"] = bool(all(v == 0 for v in out["rc"].values()) and all(all(p[k] for k in keys) for p in per.values())
                         and all(p.get("g3_old_stats_eq_d4_reference", True) for p in per.values())
                         and all(p["fix_differs_from_g3"] for f, p in per.items() if f in ref))
    (OUT / "compare_checks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"passed": out["passed"], "rc": out["rc"]}))
    return 0 if out["passed"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["local", "server", "compare"])
    ap.add_argument("--inp", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.cmd == "local":
        sys.exit(cmd_local())
    if a.cmd == "server":
        cmd_server(a.inp, a.out)
    else:
        sys.exit(cmd_compare(a.out))


if __name__ == "__main__":
    main()
