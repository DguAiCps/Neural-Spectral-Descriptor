#!/usr/bin/env python3
"""Cross-session full-key evaluation driver for the pd protocol.

Reads scripts/_pd_xsession_manifest.json and, per protocol:
  stage 1 (GPU) -- dump the frozen 416D keys of every db/query sequence with
    scripts/_pd_dump_keys.py, ONE sequence per subprocess (docker exec into
    the ring_baseline container by default; --no-docker runs python3 directly,
    e.g. when already inside the container). Existing key files are skipped.
  stage 2 (host, torch-free) -- scripts/evaluate_cross_session.py on the
    dumped npz files, results to
    results/pd2026/xsession_{protocol}_{variant}_s{seed}[_{field}].json.

Prerequisite per sequence: a pd2026 operating cache
data/preprocessed_pd2026/cache_*_{sensor}_[val_]{seq}.npz and an nsd-layout
file ((val|test)layouts_... or layouts_..._layout60). Missing prerequisites
are reported and the protocol is skipped -- nothing is built here.

Usage:
  python3 scripts/_pd_xsession_eval.py --variant pd --seed 1
  python3 scripts/_pd_xsession_eval.py --variant pd --seed 1 \
      --protocols nclt_0204_0317 --eval-only
  python3 scripts/_pd_xsession_eval.py --variant pd --seed 1 \
      --key-field pbev_mag         # branch magnitude channel instead of emb416
"""
from __future__ import annotations
import argparse
import glob
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PD_CACHE = REPO / "data/preprocessed_pd2026"
OUT = REPO / "results/pd2026"
MANIFEST = REPO / "scripts/_pd_xsession_manifest.json"
DOCKER = ["docker", "exec", "-e", "PYTHONDONTWRITEBYTECODE=1",
          "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline"]


def key_file(variant: str, seed: int, sensor: str, seq: str) -> Path:
    return OUT / f"keys_{variant}_s{seed}_{sensor}_{seq}.npz"


def cache_exists(sensor: str, seq: str) -> bool:
    return bool(glob.glob(str(PD_CACHE / f"cache_*_{sensor}_val_{seq}.npz"))
                or glob.glob(str(PD_CACHE / f"cache_*_{sensor}_test_{seq}.npz"))
                or glob.glob(str(PD_CACHE / f"cache_*_{sensor}_{seq}.npz")))


def run(cmd) -> None:
    print("RUN " + shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("pd", "pdnn"), required=True)
    ap.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--protocols", nargs="*", default=None,
                    help="manifest protocol names (default: all)")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--key-field", default="emb416",
                    help="npz field to evaluate (emb416, pbev_mag, hspec_mag)")
    ap.add_argument("--no-docker", action="store_true",
                    help="run the key dump with the local python3 instead of "
                         "docker exec ring_baseline")
    ap.add_argument("--dump-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    protocols = manifest["protocols"]
    names = args.protocols or list(protocols)
    unknown = [n for n in names if n not in protocols]
    if unknown:
        sys.exit(f"unknown protocols {unknown}; manifest has {list(protocols)}")

    failed = []
    for name in names:
        p = protocols[name]
        sensor = p["sensor"]
        seqs = list(p["db_seqs"]) + [p["query_seq"]]
        print(f"\n=== {name}: db={p['db_seqs']} query={p['query_seq']} "
              f"({args.variant} s{args.seed}) ===", flush=True)

        missing = [s for s in seqs if not cache_exists(sensor, s)]
        if missing:
            print(f"SKIP {name}: no pd2026 cache for "
                  f"{[f'{sensor}/{s}' for s in missing]} in {PD_CACHE}", flush=True)
            failed.append(name)
            continue

        if not args.eval_only:
            for seq in seqs:  # one sequence per subprocess (memory discipline)
                if key_file(args.variant, args.seed, sensor, seq).exists():
                    print(f"have {key_file(args.variant, args.seed, sensor, seq).name}",
                          flush=True)
                    continue
                dump = ["python3", "scripts/_pd_dump_keys.py",
                        "--variant", args.variant, "--seed", str(args.seed),
                        "--seqs", f"{sensor}/{seq}"]
                run(dump if args.no_docker else DOCKER + dump)
        if args.dump_only:
            continue

        keys = {s: key_file(args.variant, args.seed, sensor, s) for s in seqs}
        absent = [k.name for k in keys.values() if not k.exists()]
        if absent:
            print(f"SKIP {name} eval: missing key dumps {absent}", flush=True)
            failed.append(name)
            continue

        suffix = "" if args.key_field == "emb416" else f"_{args.key_field}"
        out = OUT / f"xsession_{name}_{args.variant}_s{args.seed}{suffix}.json"
        cmd = [sys.executable, str(REPO / "scripts/evaluate_cross_session.py"),
               "--db-cache", *[str(keys[s]) for s in p["db_seqs"]],
               "--query-cache", str(keys[p["query_seq"]]),
               "--key-field", args.key_field,
               "--threshold", "5", "--output", str(out)]
        if p.get("db_spacing_m") is not None:
            cmd += ["--db-spacing", str(p["db_spacing_m"])]
        if p.get("query_spacing_m") is not None:
            cmd += ["--query-spacing", str(p["query_spacing_m"])]
        run(cmd)

    if failed:
        print(f"\nINCOMPLETE: {failed}", flush=True)
        sys.exit(1)
    print("\nPD_XSESSION_DONE", flush=True)


if __name__ == "__main__":
    main()
