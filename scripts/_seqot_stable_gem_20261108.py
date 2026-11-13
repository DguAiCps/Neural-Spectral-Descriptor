#!/usr/bin/env python3
"""SeqOT paper recipe + numerical stabilization of the official GeM (2026-11-08).

Single root-approved recovery of the paper-recipe run
(results/pdv2_baseline_revision_20261107/seqot): its stage 2 (GeM, paper lr
5e-5) diverged in every seed at epoch 7 because the official float32 GeM
(modules/gem.py:17) underflows on fully clamped 20-frame columns once p >
6.32 and its backward overflows at p ~ 7.409
(OVN/seqot_nan_audit/report_ko.txt).  Only stage 2 is trained again, from p =
3 and a fresh Adam, for all 20 epochs, with the SAME function evaluated in the
log domain (scripts/_seqot_stable_gem_20261108_gem.py, a byte copy of
OVN/seqot_nan_audit/patch/gem_logdomain.py, sha256 50cc4504...):

    (mean_k clamp(x_k, eps)^p)^(1/p) = exp((logsumexp_k(p log clamp(x_k, eps)) - log n) / p)

eps 1e-6, the scalar parameter p, Adam lr 5e-5, StepLR(5, 0.9), triplet loss
margin 0.5, 1 + 6 + 6 windows, 20-frame centred windows, the plan of 17,136
anchors and its order, seed and the selection rule are unchanged.  The
official module file is not edited: a subclass overrides GeM.gem() and is
injected into the official-module table of scripts/_seqot_official_train.py,
so training, VAL scoring and the test descriptors all use it (state_dict key
'p' is unchanged; official checkpoints load as they are).

Read-only inputs (never rewritten): the chosen stage-1 checkpoints, the train
and VAL stage-1 sub-descriptor caches and the test range store of
_external/seqot_paper_recipe_20261107, the Eq. 13 labels, the old selections,
the pinned final-aggregation inputs (MANIFEST, 48 files).
Outputs: _external/seqot_stable_gem_20261108/{s1,s2,s3,records,flags,logs}.

Finite gates: every step (optimizer pre/post hooks: p.grad and p finite, loss
finite), every epoch (model, Adam state, the saved checkpoint re-read from
disk), every VAL / test descriptor.  A non-finite value stops the job at once
(FAILED record); a failed model is never scored as R@1 = 0.

Subcommands (inside ring_baseline; env SEED for the per-seed ones):
  freeze      write records/protocol.json (once; refuses to change it)
  selftest    operating-path tests of the injected GeM (CPU)
  valaudit    CPU VAL consistency: old GeM epochs 1..6 re-scored from the cached
              VAL sub-descriptors must reproduce the old (GPU) selection records
  train2      SEED: stage 2, 20 epochs, CPU, finite-gated
  select2     SEED: score the 20 epochs on VAL (CPU, cached sub-descriptors), choose
  valfreeze   all seeds selected -> records/val_freeze.json (hashes); test needs it
  test        SEED: 46 test sequences (GPU): stage-1 sub-descriptors from the test
              store, stabilized GeM, exact past top-1 ledgers
  aggregate   SEED: per-seed PR summary (_pdv2_c3, as the main table)
  final       3-seed common-query aggregation with the pinned aggregator, summary,
              verification
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path


def _find_repo() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "scripts" / "_seqot_official_train.py").exists():
            return p
    raise SystemExit("repository root not found")


REPO = _find_repo()
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
sys.dont_write_bytecode = True
os.environ.setdefault("SEQOT_EVAL_WINDOW", "centered")
assert os.environ["SEQOT_EVAL_WINDOW"] == "centered"

import numpy as np  # noqa: E402

import _review_seqot_paper_recipe_20261107 as R  # noqa: E402

S, P = R.S, R.P
TAG = "seqot_stable_gem_20261108"
NEW = REPO / "_external" / TAG
REC = NEW / "records"
FLAGS = NEW / "flags"
SEED = S.TRAIN_SEED
OLD_W = R.WORKP                                   # _external/seqot_paper_recipe_20261107
OLD_OUT = R.OUT                                   # results/pdv2_baseline_revision_20261107/seqot
OLD_HOME = OLD_W / f"s{SEED}" / "centered"
GEM_FILE = HERE / "_seqot_stable_gem_20261108_gem.py"
GEM_SHA256 = "50cc4504c6982a11d0a5c94cacec086408b7f30cb39e7f704206352ed9650563"
LR2, BOUND2, EPOCHS2, PLAN_STEPS = 5e-5, 21, 20, 17136
RSS_LIMIT = 3 * 2 ** 30
LABEL = "paper recipe + numerical stabilization of official GeM"


# ------------------------------------------------------------ helpers
def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path) -> str:
    return R.sha256(Path(path))


def write_json(path: Path, obj) -> None:
    R.write_json(Path(path), obj)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(REPO.resolve()))


def rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


class NonFinite(RuntimeError):
    pass


def fail(stage: str, why: str, **extra) -> None:
    """Record the failure of this job and stop it (the supervisor stops the chain)."""
    FLAGS.mkdir(parents=True, exist_ok=True)
    rec = {"utc": utc(), "stage": stage, "seed": SEED, "reason": why, **extra}
    write_json(FLAGS / f"FAILED_{stage}_s{SEED}.json", rec)
    print(f"FAILED {stage} s{SEED}: {why}", flush=True)
    sys.exit(2)


def plan_hash(plan) -> str:
    h = hashlib.sha256()
    for d, a, pos, neg in plan:
        h.update(json.dumps([int(d), int(a), [int(x) for x in pos], [int(x) for x in neg]]).encode())
    return h.hexdigest()


# ------------------------------------------------------------ the stabilized GeM
def load_gem_module():
    assert sha256(GEM_FILE) == GEM_SHA256, f"{GEM_FILE} is not the approved patch"
    spec = importlib.util.spec_from_file_location("_seqot_stable_gem_20261108_gem", GEM_FILE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def install() -> dict:
    """Paper-recipe patch of the official driver, then the stabilized GeM and the
    finite gates.  Returns the official-module table."""
    import torch
    R.patch()
    O = S.official()
    if "GeM_official" not in O:
        G = load_gem_module()
        O["GeM_official"] = O["GeM"]
        O["GeM"] = G.make_gem_logdomain(O["GeM_official"])
        O["loss_official"] = O["loss"]
        O["loss"] = types.SimpleNamespace(triplet_loss=_checked_triplet)
        S._lr_sched = _lr_sched_checked
        S._save_epoch = _save_epoch_checked
        S.centered_gem = _centered_gem_checked
    assert issubclass(O["GeM"], O["GeM_official"]) and O["GeM"] is not O["GeM_official"]
    assert Path(sys.modules[O["GeM_official"].__module__].__file__).resolve() == (S.OFFICIAL / "modules/gem.py").resolve()
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    return O


_ORIG_LR_SCHED = S._lr_sched
_ORIG_SAVE = S._save_epoch
_ORIG_CENTERED_GEM = S.centered_gem
_STEP = {"n": 0, "epoch_losses": [], "loss_nonfinite": 0}


def _checked_triplet(o1, o2, o3, margin, lazy=False, **kw):
    import torch
    loss = S._OFF["loss_official"].triplet_loss(o1, o2, o3, margin, lazy=lazy, **kw)
    if not bool(torch.isfinite(loss)):
        raise NonFinite(f"loss non-finite at step {_STEP['n']} of the epoch")
    return loss


def _lr_sched_checked(model, lr):
    """The official Adam + StepLR, plus read-only hooks: before every step all
    gradients must be finite, after it all parameters."""
    import torch
    opt, sch = _ORIG_LR_SCHED(model, lr)

    def pre(optimizer, args, kwargs):
        for g in optimizer.param_groups:
            for p in g["params"]:
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    raise NonFinite(f"gradient non-finite at step {_STEP['n']} (p = {p.detach().flatten()[:1].tolist()})")

    def post(optimizer, args, kwargs):
        for g in optimizer.param_groups:
            for p in g["params"]:
                if not bool(torch.isfinite(p).all()):
                    raise NonFinite(f"parameter non-finite after step {_STEP['n']}")
        _STEP["n"] += 1
        if _STEP["n"] % 2000 == 0 and rss_bytes() > RSS_LIMIT:
            raise MemoryError(f"RSS {rss_bytes() / 2 ** 30:.2f} GiB > 3 GiB")

    opt.register_step_pre_hook(pre)
    opt.register_step_post_hook(post)
    return opt, sch


def _tensors_finite(obj) -> bool:
    import torch
    if isinstance(obj, torch.Tensor):
        return bool(torch.isfinite(obj.float()).all())
    if isinstance(obj, dict):
        return all(_tensors_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_tensors_finite(v) for v in obj)
    if isinstance(obj, float):
        return math.isfinite(obj)
    return True


def _save_epoch_checked(run_dir, prefix, i, model, opt):
    """Epoch gate: model and Adam state finite -> official save -> the file
    re-read from disk is finite and equal to the model in memory."""
    import torch
    sd, osd = model.state_dict(), opt.state_dict()
    if not (_tensors_finite(sd) and _tensors_finite(osd["state"])):
        raise NonFinite(f"epoch {i}: model or optimizer state non-finite before save")
    _ORIG_SAVE(run_dir, prefix, i, model, opt)
    p = run_dir / f"{prefix}{i}.pth.tar"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    ok = (ck["epoch"] == i and _tensors_finite(ck["state_dict"]) and _tensors_finite(ck["optimizer"]["state"])
          and all(torch.equal(ck["state_dict"][k].cpu(), v.cpu()) for k, v in sd.items()))
    rec = {"epoch": i, "utc": utc(), "checkpoint": rel(p), "sha256": sha256(p), "finite": ok,
           "p": float(sd["p"].flatten()[0]), "steps": _STEP["n"],
           "adam_exp_avg_sq": float(osd["state"][0]["exp_avg_sq"].flatten()[0]) if osd["state"] else None,
           "rss_gib": round(rss_bytes() / 2 ** 30, 3)}
    with open(run_dir / "finite_gate.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"GATE epoch {i}: p {rec['p']:.6f} finite {ok} steps {_STEP['n']}", flush=True)
    if not ok:
        raise NonFinite(f"epoch {i}: saved checkpoint is not finite / not equal to the model")
    _STEP["n"] = 0


def _centered_gem_checked(gem, sub, need, n_raw, kf, dev):
    out = _ORIG_CENTERED_GEM(gem, sub, need, n_raw, kf, dev)
    if not np.all(np.isfinite(out)):
        raise NonFinite(f"descriptor non-finite ({int((~np.isfinite(out)).sum())} values)")
    return out


# ------------------------------------------------------------ fixed inputs
def old_selection(seed=SEED) -> dict:
    return read_json(OLD_OUT / f"s{seed}" / "train_selection.json")


def stage1_choice(seed=SEED):
    c = old_selection(seed)["stage1"]["chosen"]
    p = REPO / c["checkpoint"]
    assert sha256(p) == c["sha256"], f"{p} changed"
    return p, c


def train_subdesc_paths(seed=SEED):
    home = OLD_W / f"s{seed}" / "centered" / "subdesc" / "train"
    return {P.seq_name(s, q): home / f"{P.seq_name(s, q)}.npy" for s, q in S.TRAIN}, home / "meta.json"


def val_cache_paths(seed=SEED):
    home = OLD_W / f"s{seed}" / "centered" / "subdesc" / "val"
    return {P.seq_name(s, q): home / f"{P.seq_name(s, q)}.npz" for s, q in S.VAL}


def train_sources(seed=SEED):
    paths, meta = train_subdesc_paths(seed)
    _, c1 = stage1_choice(seed)
    assert read_json(meta)["stage1"]["sha256"] == c1["sha256"], "train sub-descriptors are not of the chosen stage 1"
    return [S.DescSource(np.load(paths[P.seq_name(s, q)])) for s, q in S.TRAIN]


def make_plan(sources):
    return S.anchor_plan(S.load_train_indices(),
                         lambda d, a: all(sources[d].exists(int(x)) for x in S.window(a, S.GEM_SEQLEN, S.TRAIN_WINDOW)),
                         *S.USE2)


def val_cache(seed=SEED):
    _, c1 = stage1_choice(seed)
    out = {}
    for (s, q) in S.VAL:
        f = val_cache_paths(seed)[P.seq_name(s, q)]
        with np.load(f) as z:
            assert str(z["stage1_sha256"]) == c1["sha256"], f"{f}: VAL cache not of the chosen stage 1"
            out[(s, q)] = (z["sub"], z["raw"])
    return out


def protocol() -> dict:
    p = REC / "protocol.json"
    assert p.exists(), "protocol not frozen"
    return read_json(p)


def check_protocol_inputs(seed=SEED) -> None:
    """Every fixed input of this seed must still have its frozen hash."""
    pr = protocol()
    for k, h in pr["source_sha256"].items():
        assert sha256(REPO / k) == h, f"source changed: {k}"
    fx = pr["fixed_inputs"][str(seed)]
    for k, h in fx["sha256"].items():
        assert sha256(REPO / k) == h, f"fixed input changed: {k}"


# ------------------------------------------------------------ freeze
SOURCES = ["scripts/_seqot_official_train.py", "scripts/_seqot_official_prep.py", "scripts/_seqot_official_dump.py",
           "scripts/_review_seqot_paper_recipe_20261107.py", "scripts/_review_seqot_paper_recipe_20261107_test.py",
           "scripts/_pdv2_c3.py", "scripts/_pr_eval_lib.py", "scripts/_pdv2_pilot.py",
           "_external/SeqOT/modules/gem.py", "_external/SeqOT/tools/loss.py",
           "_external/SeqOT/modules/seqTransformerCat.py", "_external/SeqOT/config/config.yml"]


def cmd_freeze() -> None:
    """Protocol of the recovery, written once before any training step."""
    out = REC / "protocol.json"
    install()
    fixed = {}
    for seed in (1, 2, 3):
        sel = old_selection(seed)
        c1 = sel["stage1"]["chosen"]
        assert sha256(REPO / c1["checkpoint"]) == c1["sha256"]
        tp, tmeta = train_subdesc_paths(seed)
        files = {rel(REPO / c1["checkpoint"]): c1["sha256"], rel(tmeta): sha256(tmeta),
                 rel(OLD_OUT / f"s{seed}/train_selection.json"): sha256(OLD_OUT / f"s{seed}/train_selection.json")}
        files.update({rel(p): sha256(p) for p in tp.values()})
        files.update({rel(p): sha256(p) for p in val_cache_paths(seed).values()})
        sources = [S.DescSource(np.load(p)) for p in tp.values()]
        plan = make_plan(sources)
        assert len(plan) == PLAN_STEPS, len(plan)
        fixed[str(seed)] = {"stage1_chosen": c1, "old_stage2_chosen": sel["stage2"]["chosen"],
                            "plan_steps": len(plan), "plan_sha256": plan_hash(plan),
                            "plan_per_sequence": S._plan_meta(plan, len(S.TRAIN)), "sha256": files}
    labels = {f"{s}/{q}": sha256(R.label_path(s, q)) for s, q in S.TRAIN}
    old_pr = read_json(OLD_OUT / "protocol.json")
    assert labels == old_pr["label_index_sha256"], "Eq. 13 labels differ from the paper-recipe protocol"
    man = OLD_W / "final_inputs" / "MANIFEST.json"
    m = read_json(man)
    assert m["n_files"] == 48 and len(m["sha256"]) == 48
    for k, h in m["sha256"].items():
        assert sha256(OLD_W / "final_inputs" / k) == h, k
    pr = {
        "frozen_utc": utc(),
        "label": LABEL,
        "worker": "seqot_stable_gem (claude-opus-5-5)",
        "authorization": "OVN/coordination.json seqot_stable_gem_authorization (approved 2026-11-07T20:28:03Z); "
                         "single approved recovery, not a tuning run",
        "frozen_before": "any stage-2 training step of this recovery (only read-only CPU timing probes ran before)",
        "scope": "stage 2 only, seeds 1/2/3, 20 epochs each from GeM p = 3 and a fresh Adam; stage 1, labels, "
                 "plan, lr, scheduler, margin, 6/6, windows, seed, selection rule, split train8/val9/test46 unchanged",
        "changed": {
            "only": "GeM forward evaluated in the log domain (same function, same eps, same parameter p)",
            "formula_official": "F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1/p)  (modules/gem.py:17)",
            "formula_used": "exp((logsumexp_k(p * log clamp(x_k, eps)) - log n) / p)",
            "identity": "(mean_k y_k^p)^(1/p) = exp(log(sum_k exp(p log y_k)) - log n) / p) for y_k = clamp(x_k, eps) > 0; "
                        "exact in real arithmetic; only floating-point rounding differs",
            "why": "official float32: whole-clamped column -> (1e-6)^p subnormal for p > 6.32, backward "
                   "y^(1/p - 1) overflows at p ~ 7.409 -> NaN p.grad (OVN/seqot_nan_audit, root-verified)",
            "patch_file": "scripts/_seqot_stable_gem_20261108_gem.py",
            "patch_sha256": GEM_SHA256,
            "patch_origin": "byte copy of OVN/seqot_nan_audit/patch/gem_logdomain.py (root server-CPU tests pass)",
            "injection": "subclass of the official GeM overriding gem(); placed in _seqot_official_train._OFF['GeM'] "
                         "so train_phase, _load_gem (VAL) and the test descriptors use it; official file untouched; "
                         "state_dict key 'p' unchanged",
            "used_in": "training, VAL selection and all test descriptors (no return to the official pow at test)",
        },
        "not_changed_explicitly": ["eps 1e-6", "p parameterisation (no clamp/bound)", "lr 5e-5", "StepLR(5, 0.9)",
                                   "Adam defaults", "triplet loss margin 0.5 lazy=False", "Npos = Nneg = 6",
                                   "Eq. 13 labels, negative rule 'le'", "plan order (17,136 anchors / epoch)",
                                   "20-frame centred windows, N + 1 rows", "seed_all(SEED) before model creation",
                                   "no gradient clipping, no NaN-step skipping, no AMP"],
        "not_a_tuning_argument": (
            "The objective, its parameters and the optimizer are identical; the log-domain form is the same "
            "function of (x, p) and has the same gradient in exact arithmetic.  It removes a float32 range "
            "failure, it introduces no degree of freedom and no alternative was tried or compared.  Before the "
            "failure the official float32 gradients agree with float64 (rel. err median 1.3e-6 / 3.4e-5 on 300 "
            "real batches) and the log form agrees with float64 within ~1e-6 (OVN/seqot_nan_audit)."),
        "training": {"device": "server ring_baseline CPU (CUDA_VISIBLE_DEVICES empty), 1 job, seeds serial, "
                               "torch threads 1, nice 10 / ionice idle, RSS guard 3 GiB",
                     "lr": LR2, "loop_bound": BOUND2, "epochs": "1..20", "plan_steps": PLAN_STEPS,
                     "init": "official GeM() p = 3 after S.seed_all(SEED); fresh Adam; no resume from the "
                             "diverged run (own finite-gated checkpoints only if this job restarts)",
                     "cpu_vs_gpu": "the diverged run trained on GPU; CPU replay of its epoch 6 reproduced p bit-"
                                   "exactly for seed 1 and within 2e-6 for seeds 2/3 (OVN/seqot_nan_audit). This "
                                   "run's epochs 1..6 are therefore not bitwise equal to the old ones (CPU + "
                                   "log-domain rounding); their agreement is recorded, not assumed.",
                     "finite_gates": "per step: loss, p.grad, p; per epoch: model, Adam state, saved file "
                                     "re-read; failure stops the job (no R@1 = 0 candidates)"},
        "selection": {
            "rule": "unchanged main-table rule: split-v2 VAL (9 seq) GT keyframes, R@1 at 5 m, |j-q| > 30 "
                    "exhaustive top-1, families equal / queries pooled (_pdv2_c3._family_pooled), official L2 "
                    "(family_r1_l2), official centred windows; all 20 epochs; max, ties -> earlier epoch "
                    "(_seqot_official_train._choose)",
            "device": "CPU from the existing cached VAL stage-1 sub-descriptors (read-only, stage1_sha256 "
                      "checked); allowed after valaudit shows the CPU computation reproduces the old GPU records",
            "valaudit_pass_rule": "for every seed, old stage-2 epochs 1..6 with the official GeM on CPU: per-"
                                  "sequence R@1 L2 query counts equal to the old records, or |family R@1 L2 diff| "
                                  "<= 5e-4 for every epoch with the same argmax; anything else -> FAILED, no "
                                  "CPU selection",
            "freeze": "records/val_freeze.json (sha256 of the 3 selections and the chosen checkpoints) is written "
                      "after all 3 seeds are chosen and before any test computation; test refuses to run without it",
            "test_not_used_for": "anything (no recipe, epoch, seed or numerics choice)"},
        "test": {"split": "split-v2 46 TEST sequences, KISS-ICP keyframes, queries t >= 30, past candidates j <= q-30 "
                          "exhaustive top-1, GT 5 m, official centred windows, reported ledger seqotl2 (score_l2)",
                 "inputs": "existing test range store _external/seqot_paper_recipe_20261107/test_inputs (read-only); "
                           "no raw re-extraction",
                 "stage1": "the old run kept no test sub-descriptors, so the chosen (unchanged) stage-1 networks "
                           "are run again on the stored range images (GPU, as the old test); the old GeM applied "
                           "to the same sub-descriptors must reproduce the old stored test descriptors (fidelity)",
                 "descriptors": "query and candidate descriptors regenerated with the stabilized GeM and the new "
                                "chosen p; exact ledgers recomputed; no old ledger reused",
                 "gpu_gate": "GPU0 only after one HeLiOS role's server COMPUTE_DONE.json (or COMPUTE_FAILED.json "
                             "with its supervisor gone), GPU0 free >= 25 GiB, under the existing flock "
                             "/tmp/nsd_native_reproduction_gpu0.lock; seeds serial",
                 "aggregation": "per seed 45 seq (KITTI07 excluded) -> 6 cities macro -> 3-seed mean, sample SD; "
                                "pinned _pdv2_main_baselines_official.aggregate (old final_inputs MANIFEST, 48 files) "
                                "on the 195,726 common queries / 61,682 positives"},
        "cost_note": "same network, input, windows and descriptor size; the GeM pooling is computed as log, "
                     "multiply, logsumexp, exp instead of pow, mean, pow. Runtime per query is not assumed "
                     "identical; it is to be re-measured later on the same hardware (not in this scope).",
        "budgets": {"server_root_free_min_gib": 35, "new_root_output_max_gib": 2, "nas": "not used"},
        "fixed_inputs": fixed,
        "label_index_sha256": labels,
        "final_inputs_manifest_sha256": sha256(man),
        "source_sha256": {k: sha256(REPO / k) for k in SOURCES},
        "own_sources_sha256": {f.name: sha256(f) for f in sorted(HERE.glob("_seqot_stable_gem_20261108*"))},
    }
    old_src = old_pr["source_sha256"]
    for k, h in pr["source_sha256"].items():
        if k in old_src:
            assert old_src[k] == h, f"{k} differs from the paper-recipe protocol"
    if out.exists():
        old = read_json(out)
        keep = {k: v for k, v in old.items() if k != "frozen_utc"}
        new = {k: v for k, v in pr.items() if k != "frozen_utc"}
        assert keep == new, "protocol.json exists with different content; refusing to change a frozen protocol"
        print("FREEZE_OK (unchanged)", flush=True)
        return
    write_json(out, pr)
    print("FREEZE_DONE", flush=True)


# ------------------------------------------------------------ selftest
def cmd_selftest() -> None:
    """Operating-path checks (CPU, synthetic + real cached inputs, nothing trained)."""
    import torch
    O = install()
    Off, New = O["GeM_official"], O["GeM"]
    rep = {"utc": utc(), "checks": {}}
    c = rep["checks"]
    g0, g1 = Off(), New()
    c["state_keys_equal"] = list(g0.state_dict()) == list(g1.state_dict()) == ["p"]
    c["init_p_3"] = float(g1.p) == 3.0 and float(g0.p) == 3.0
    c["load_official_paths_use_new_class"] = type(S._load_gem(None, torch.device("cpu"))) is New
    # old checkpoints load unchanged (strict)
    old = REPO / old_selection()["stage2"]["chosen"]["checkpoint"]
    ck = torch.load(old, map_location="cpu", weights_only=False)
    g1.load_state_dict(ck["state_dict"], strict=True)
    g0.load_state_dict(ck["state_dict"], strict=True)
    c["old_checkpoint_loads_strict"] = True
    # real cached VAL sub-descriptors, the operating window function, official vs log-domain vs float64
    cache = val_cache()
    (s, q), (sub, need) = next(iter(cache.items()))
    src = dict(S._val_sets())[(s, q)]
    kf = src.kf[:2000]
    d0 = _ORIG_CENTERED_GEM(g0.eval(), sub, need, src.n_raw, kf, torch.device("cpu"))
    d1 = S.centered_gem(g1.eval(), sub, need, src.n_raw, kf, torch.device("cpu"))
    c["shape_equal"] = d0.shape == d1.shape == (len(kf), 256)
    c["dtype_float32"] = d1.dtype == np.float32
    rep["real_val_maxabs_official_vs_log_at_old_p"] = float(np.max(np.abs(d0 - d1)))
    rep["old_p"] = float(g1.p)
    c["real_val_close"] = rep["real_val_maxabs_official_vs_log_at_old_p"] < 1e-5
    # synthetic, float64 reference, through p = 20, with a whole-clamped column; gradients
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(13, 20, 256, generator=gen) / 16
    x[:, :, 0] = -1.0
    worst = {}
    for pv in (1.0, 3.0, 5.0, 7.0, 7.41, 10.0, 15.0, 20.0):
        p = torch.tensor([pv], requires_grad=True)
        y = New.gem(None, x, p)
        y.sum().backward()
        p64 = torch.tensor([pv], dtype=torch.float64, requires_grad=True)
        y64 = torch.nn.functional.avg_pool2d(x.double().clamp(min=1e-6).pow(p64), (20, 1)).pow(1. / p64)
        y64.sum().backward()
        worst[str(pv)] = {"finite": bool(torch.isfinite(y).all() and torch.isfinite(p.grad).all()),
                          "fwd_maxabs_vs_f64": float((y.double() - y64).abs().max()),
                          "pgrad_rel_vs_f64": float(abs(p.grad.double() - p64.grad) / abs(p64.grad))}
    rep["synthetic"] = worst
    c["synthetic_finite_to_p20"] = all(v["finite"] for v in worst.values())
    c["synthetic_fwd_close"] = all(v["fwd_maxabs_vs_f64"] < 1e-6 for v in worst.values())
    c["synthetic_pgrad_close"] = all(v["pgrad_rel_vs_f64"] < 1e-3 for v in worst.values())
    # training shape path: one real step through train_phase's statements (no save)
    sources = train_sources()
    plan = make_plan(sources)
    d, a, pos, neg = plan[0]
    xb = torch.from_numpy(S.stage2_batch(sources[d], a, pos, neg, "centered"))
    m = New()
    out = m(xb).squeeze(1)
    c["train_batch_shape"] = tuple(xb.shape) == (13, 20, 256) and tuple(out.shape) == (13, 256)
    # gates fire on a NaN
    try:
        _checked_triplet(torch.full((1, 256), float("nan")), torch.zeros(6, 256), torch.zeros(6, 256), 0.5)
        c["loss_gate_fires"] = False
    except NonFinite:
        c["loss_gate_fires"] = True
    mm = New()
    opt, _ = S._lr_sched(mm, LR2)
    mm.p.grad = torch.tensor([float("nan")])
    try:
        opt.step()
        c["grad_gate_fires"] = False
    except NonFinite:
        c["grad_gate_fires"] = True
    c["grad_gate_left_p_unchanged"] = float(mm.p) == 3.0
    _STEP["n"] = 0
    rep["passed"] = all(c.values())
    write_json(REC / "selftest.json", rep)
    print(json.dumps(rep["checks"]), flush=True)
    print("SELFTEST_PASSED" if rep["passed"] else "SELFTEST_FAILED", flush=True)
    if not rep["passed"]:
        fail("selftest", "selftest failed", checks=c)


# ------------------------------------------------------------ CPU VAL consistency audit
def cmd_valaudit() -> None:
    import torch
    O = install()
    dev = torch.device("cpu")
    out = {"utc_start": utc(), "device": "cpu", "torch": torch.__version__, "seeds": {}}
    val_sets = S._val_sets()
    ok_all = True
    for seed in (1, 2, 3):
        sel = old_selection(seed)
        cache = val_cache(seed)
        run = OLD_W / f"s{seed}" / "centered" / "runs" / "stage2"
        per = {}
        for e in range(1, 7):
            p = run / f"amodel_gem{e}.pth.tar"
            rec = sel["stage2"]["epochs"][str(e)]
            assert sha256(p) == rec["sha256"]
            sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
            res = {}
            for name, cls in (("official", O["GeM_official"]), ("logdomain", O["GeM"])):
                g = cls()
                g.load_state_dict(sd)
                t0 = time.time()
                r = S.val_score(None, g.eval(), dev, val_sets, 64, cache)
                same_counts = all(round(r["per_seq"][k]["r1_l2"] * v["n_queries"]) ==
                                  round(rec["per_seq"][k]["r1_l2"] * v["n_queries"])
                                  and r["per_seq"][k]["n_queries"] == v["n_queries"]
                                  for k, v in rec["per_seq"].items())
                res[name] = {"family_r1_l2": r["family_r1_l2"], "family_r1_cos": r["family_r1_cos"],
                             "diff_vs_old_gpu": r["family_r1_l2"] - rec["family_r1_l2"],
                             "per_seq_counts_equal_old": bool(same_counts), "s": round(time.time() - t0, 1)}
            per[str(e)] = {"old_gpu_family_r1_l2": rec["family_r1_l2"], "p": float(sd["p"]), **res}
            print(f"valaudit s{seed} e{e}: old {rec['family_r1_l2']:.6f} cpu-official "
                  f"{res['official']['family_r1_l2']:.6f} cpu-log {res['logdomain']['family_r1_l2']:.6f}", flush=True)
        old_arg = min(per, key=lambda e: (-per[e]["old_gpu_family_r1_l2"], int(e)))
        cpu_arg = min(per, key=lambda e: (-per[e]["official"]["family_r1_l2"], int(e)))
        exact = all(v["official"]["per_seq_counts_equal_old"] for v in per.values())
        close = all(abs(v["official"]["diff_vs_old_gpu"]) <= 5e-4 for v in per.values())
        ok = exact or (close and old_arg == cpu_arg)
        ok_all &= ok
        out["seeds"][str(seed)] = {"epochs": per, "argmax_old_gpu": int(old_arg), "argmax_cpu_official": int(cpu_arg),
                                   "per_seq_counts_equal_all_epochs": exact,
                                   "max_abs_family_diff_official": max(abs(v["official"]["diff_vs_old_gpu"]) for v in per.values()),
                                   "max_abs_family_diff_logdomain": max(abs(v["logdomain"]["diff_vs_old_gpu"]) for v in per.values()),
                                   "passed": bool(ok)}
    out["rule"] = protocol()["selection"]["valaudit_pass_rule"]
    out["passed"] = bool(ok_all)
    out["utc_end"] = utc()
    write_json(REC / "val_audit.json", out)
    print("VALAUDIT_PASSED" if ok_all else "VALAUDIT_FAILED", flush=True)
    if not ok_all:
        fail("valaudit", "CPU VAL does not reproduce the old GPU records", detail=out)


# ------------------------------------------------------------ stage 2 training
def run_dir(seed=SEED) -> Path:
    return NEW / f"s{seed}" / "runs" / "stage2"


def cmd_train2() -> None:
    import torch
    install()
    check_protocol_inputs()
    pr = protocol()["fixed_inputs"][str(SEED)]
    p1, c1 = stage1_choice()
    S.seed_all()                                  # cmd_train2 order: seed, then data / model
    dev = torch.device("cpu")
    sources = train_sources()
    assert not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "CPU job only"
    plan = make_plan(sources)
    assert len(plan) == PLAN_STEPS and plan_hash(plan) == pr["plan_sha256"], "plan differs from the frozen protocol"
    rd = run_dir()
    rd.mkdir(parents=True, exist_ok=True)
    done = sorted(int(p.name[len("amodel_gem"):-len(".pth.tar")]) for p in rd.glob("amodel_gem*.pth.tar"))
    logs = [json.loads(x) for x in (rd / "log.jsonl").read_text().splitlines()] if (rd / "log.jsonl").exists() else []
    gates = [json.loads(x) for x in (rd / "finite_gate.jsonl").read_text().splitlines()] \
        if (rd / "finite_gate.jsonl").exists() else []
    if (rd / "finished").exists():
        print("SKIP train2: finished", flush=True)
        return
    if done:                                      # own finite-gated checkpoints only (restart of this job)
        ok_gates = {g["epoch"]: g for g in gates if g["finite"]}
        consistent = (done == list(range(1, len(done) + 1)) and [r["epoch"] for r in logs] == done
                      and all(e in ok_gates and sha256(rd / f"amodel_gem{e}.pth.tar") == ok_gates[e]["sha256"]
                              for e in done))
        if not consistent:
            fail("train2", "inconsistent partial run (checkpoints / log / gates); mechanical recovery needed",
                 checkpoints=done, log_epochs=[r["epoch"] for r in logs])
        with open(rd / "restarts.jsonl", "a") as f:
            f.write(json.dumps({"utc": utc(), "resume_from_epoch": done[-1]}) + "\n")
    meta = {"label": LABEL, "lr": LR2, "epochs": f"1..{BOUND2 - 1}", "anchors_per_epoch": len(plan),
            "plan_sha256": plan_hash(plan), "anchors_per_sequence": S._plan_meta(plan, len(S.TRAIN)),
            "use_pos_neg": S.USE2, "window": S.TRAIN_WINDOW, "negative_rule": S.NEG_RULE, **S._recipe_meta(2),
            "seed": SEED, "stage1": c1, "device": str(dev), "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(), "gem_class": f"{S._OFF['GeM'].__module__}.{S._OFF['GeM'].__name__}",
            "gem_patch_sha256": GEM_SHA256, "train_subdesc": {k: rel(v) for k, v in train_subdesc_paths()[0].items()},
            "init": "fresh: official GeM() p = 3, new Adam (no diverged-run state)"}
    if not (rd / "meta.json").exists():
        write_json(rd / "meta.json", meta)
    try:
        S.train_phase("gem", plan, sources, rd, LR2, BOUND2, dev, log=rd / "log.jsonl")
    except (NonFinite, MemoryError) as e:
        fail("train2", repr(e))
    logs = [json.loads(x) for x in (rd / "log.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in logs] == list(range(1, EPOCHS2 + 1)), "epoch log not 1..20 exactly once"
    assert all(r["steps"] == PLAN_STEPS and math.isfinite(r["loss"]) for r in logs)
    (rd / "finished").write_text(f"{BOUND2 - 1}\n")
    print(f"TRAIN2_DONE seed {SEED}", flush=True)


# ------------------------------------------------------------ VAL selection
def selection_path(seed=SEED) -> Path:
    return REC / f"s{seed}" / "train_selection.json"


def cmd_select2() -> None:
    import torch
    install()
    check_protocol_inputs()
    rd = run_dir()
    assert (rd / "finished").read_text().strip() == str(EPOCHS2), "stage 2 not finished"
    gates = {g["epoch"]: g for g in map(json.loads, (rd / "finite_gate.jsonl").read_text().splitlines()) if g["finite"]}
    dev = torch.device("cpu")
    val_sets = S._val_sets()
    cache = val_cache()
    va = read_json(REC / "val_audit.json")
    assert va["passed"], "CPU VAL audit did not pass"
    sp = selection_path()
    sel = read_json(sp) if sp.exists() else {
        "_meta": {**old_selection()["_meta"], "label": LABEL, "device": "cpu (cached VAL stage-1 sub-descriptors)",
                  "val_audit": rel(REC / "val_audit.json"), "gem": "log-domain (stabilized official GeM)"},
        "stage1": {"chosen": old_selection()["stage1"]["chosen"], "finished": True,
                   "source": rel(OLD_OUT / f"s{SEED}/train_selection.json"),
                   "note": "unchanged chosen stage 1 of the paper-recipe run (read-only)"},
        "stage2": {"epochs": {}}}
    if sel.get("finished"):
        print("SKIP select2: finished", flush=True)
        return
    for e in range(1, EPOCHS2 + 1):
        if str(e) in sel["stage2"]["epochs"]:
            continue
        p = rd / f"amodel_gem{e}.pth.tar"
        h = sha256(p)
        assert e in gates and gates[e]["sha256"] == h, f"epoch {e} has no passing finite gate"
        t0 = time.time()
        try:
            res = S.val_score(None, S._load_gem(p, dev), dev, val_sets, 64, cache)
        except NonFinite as ex:
            fail("select2", f"epoch {e}: {ex!r}")
        if not all(math.isfinite(res[k]) for k in ("family_r1_l2", "family_r1_cos")):
            fail("select2", f"epoch {e}: non-finite VAL score")
        res.update({"checkpoint": rel(p), "sha256": h, "p": gates[e]["p"], "val_s": round(time.time() - t0, 1)})
        sel["stage2"]["epochs"][str(e)] = res
        write_json(sp, sel)
        print(f"select2: epoch {e} p {gates[e]['p']:.4f} family R@1 l2 {res['family_r1_l2']:.4f} "
              f"cos {res['family_r1_cos']:.4f}", flush=True)
    st = sel["stage2"]
    assert sorted(map(int, st["epochs"])) == list(range(1, EPOCHS2 + 1))
    e = S._choose(st, "l2")
    st["chosen"] = {"epoch": e, "metric": "l2", **{k: st["epochs"][str(e)][k] for k in ("checkpoint", "sha256")}}
    st["finished"] = True
    sel["finished"] = True
    sel["chosen_utc"] = utc()
    write_json(sp, sel)
    print(f"SELECT2_DONE seed {SEED} epoch {e}", flush=True)


def cmd_valfreeze() -> None:
    out = REC / "val_freeze.json"
    if out.exists():
        print("VALFREEZE exists", flush=True)
        return
    fz = {"utc": utc(), "protocol_sha256": sha256(REC / "protocol.json"),
          "val_audit_sha256": sha256(REC / "val_audit.json"), "seeds": {}}
    for seed in (1, 2, 3):
        sp = selection_path(seed)
        sel = read_json(sp)
        assert sel.get("finished")
        c = sel["stage2"]["chosen"]
        assert sha256(REPO / c["checkpoint"]) == c["sha256"]
        fz["seeds"][str(seed)] = {"selection": rel(sp), "selection_sha256": sha256(sp),
                                  "stage1": sel["stage1"]["chosen"], "stage2": c,
                                  "stage2_p": sel["stage2"]["epochs"][str(c["epoch"])]["p"],
                                  "val_family_r1_l2": sel["stage2"]["epochs"][str(c["epoch"])]["family_r1_l2"]}
    fz["test_computed_before"] = False
    assert not any((NEW / f"s{s}" / "desc").exists() for s in (1, 2, 3)), "test outputs exist before freeze"
    write_json(out, fz)
    print("VALFREEZE_DONE", flush=True)


def frozen_choice(seed=SEED):
    fz = read_json(REC / "val_freeze.json")
    f = fz["seeds"][str(seed)]
    assert sha256(REPO / f["selection"]) == f["selection_sha256"], "selection changed after the freeze"
    assert fz["protocol_sha256"] == sha256(REC / "protocol.json")
    for st in ("stage1", "stage2"):
        assert sha256(REPO / f[st]["checkpoint"]) == f[st]["sha256"]
    return f


# ------------------------------------------------------------ test
def odom_dir(seed=SEED) -> Path:
    return REC / f"s{seed}" / "odom"


def desc_dir(seed=SEED) -> Path:
    return NEW / f"s{seed}" / "desc" / "test_odom"


def _patch_dump(D):
    D.SELECTION = selection_path()
    D.OUT = odom_dir()
    D.DESC = desc_dir()
    if getattr(D, "_stable_patched", False):
        return
    orig = D._meta

    def meta(*a, **k):
        m = orig(*a, **k)
        m["variant"] = f"SeqOT {LABEL} (arXiv 2209.07951 recipe retrained on split-v2 train)"
        m["training"] = ("paper recipe: Eq.13 labels, >0.3 positive otherwise negative, Npos=Nneg=6, phase 1 "
                         "Adam lr 5e-6 x 20 epochs (unchanged, paper-recipe run), phase 2 GeM lr 5e-5 x 20 epochs "
                         "retrained with the log-domain (same-function) GeM, StepLR(5,0.9); epochs chosen on v2 "
                         f"VAL (main-table rule), {rel(selection_path())}; protocol {rel(REC / 'protocol.json')}")
        m["gem_numerics"] = f"log-domain GeM, patch sha256 {GEM_SHA256} (training, VAL and test)"
        return m
    D._meta = meta
    D._stable_patched = True


def cmd_test() -> None:
    import torch
    import _seqot_official_dump as D
    import _review_seqot_paper_recipe_20261107_test as T
    O = install()
    check_protocol_inputs()
    _patch_dump(D)
    f = frozen_choice()
    dev = torch.device("cuda")
    assert torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "test runs on GPU0"
    fe = S._load_fe(REPO / f["stage1"]["checkpoint"], dev)
    gem = S._load_gem(REPO / f["stage2"]["checkpoint"], dev)
    assert type(gem) is O["GeM"]
    oc = old_selection()["stage2"]["chosen"]
    gem_old = O["GeM_official"]().to(dev)
    gem_old.load_state_dict(torch.load(REPO / oc["checkpoint"], map_location=dev, weights_only=False)["state_dict"])
    gem_old.eval()
    old_desc = OLD_HOME / "desc" / "test_odom"
    desc_dir().mkdir(parents=True, exist_ok=True)
    fid_path = REC / f"s{SEED}" / "test_fidelity.json"
    fid = read_json(fid_path) if fid_path.exists() else {}
    t_all = time.time()
    for sensor, seq in P.TEST:
        name = P.seq_name(sensor, seq)
        if name in fid and all((D.OUT / f"dump_{m}_{sensor}_{seq}.npz").exists() for m in D.METHODS) \
                and (desc_dir() / f"{name}.npz").exists():
            print(f"[skip] {sensor}/{seq}", flush=True)
            continue
        src = T.TestSource(sensor, seq)
        cp = P.cache_path("test_odom", sensor, seq)
        with np.load(cp) as z:
            assert np.array_equal(z["scan_ids"], src.kf) and np.array_equal(z["poses_gt"], src.kf_poses)
        t0 = time.time()
        need = S.centered_need(src, src.kf)
        sub = S.sub_descriptors(fe, src, need, dev, 64, "centered")
        if not np.all(np.isfinite(sub)):
            fail("test", f"{sensor}/{seq}: stage-1 sub-descriptors non-finite")
        try:
            final = S.centered_gem(gem, sub, need, src.n_raw, src.kf, dev)
        except NonFinite as ex:
            fail("test", f"{sensor}/{seq}: {ex!r}")
        t_enc = time.time() - t0
        final_old = _ORIG_CENTERED_GEM(gem_old, sub, need, src.n_raw, src.kf, dev)
        with np.load(old_desc / f"{name}.npz") as z:
            ref = z["final"]
            assert np.array_equal(z["kf_scan_ids"], src.kf)
        fid[name] = {"old_final_bitwise_equal": bool(np.array_equal(final_old, ref)),
                     "old_final_maxabs": float(np.max(np.abs(final_old - ref))),
                     "n_keyframes": int(len(src.kf)), "raw_scans_read": int(len(need))}
        np.savez(desc_dir() / f"{name}.npz", final=final, kf_scan_ids=src.kf,
                 stage1_sha256=f["stage1"]["sha256"], stage2_sha256=f["stage2"]["sha256"], gem_patch_sha256=GEM_SHA256)
        extra = {"artifacts": {"stage1": f["stage1"], "stage2": f["stage2"], "selection": rel(selection_path()),
                               "val_freeze": rel(REC / "val_freeze.json")},
                 "caches": {"operating_cache": rel(cp), "operating_cache_sha256": sha256(cp),
                            "positions": "poses_gt (scoring only)", "raw_scans_read": int(len(need)),
                            "n_raw": int(src.n_raw), "range_store": rel(T.TEST_STORE / name),
                            "range_sha256": src.done["range_sha256"]},
                 "timing": {"encode_s": round(t_enc, 1), "encode_ms_per_kf": round(1e3 * t_enc / max(len(final), 1), 2)},
                 "device": str(dev), "torch": torch.__version__}
        D.write_dump(sensor, seq, final, src.kf_poses[:, :3, 3], D.OUT, extra)
        write_json(fid_path, fid)
        print(f"  fidelity {name}: old bitwise {fid[name]['old_final_bitwise_equal']} "
              f"maxabs {fid[name]['old_final_maxabs']:.3g}", flush=True)
    assert len(fid) == len(P.TEST)
    write_json(REC / f"s{SEED}" / "test_done.json", {"utc": utc(), "seed": SEED, "total_s": round(time.time() - t_all, 1),
                                                       "n_sequences": len(fid)})
    print(f"TEST_DONE seed {SEED}", flush=True)


def cmd_aggregate() -> None:
    import _pdv2_c3 as c3
    import _seqot_official_dump as D
    R.patch()
    _patch_dump(D)
    c3.OUT_ROOT = REC / f"s{SEED}"
    c3.BASELINES = dict(D.METHODS)
    c3.cmd_aggregate_baselines("odom")
    p = odom_dir() / "pr_summary_baselines.json"
    s = read_json(p)
    s["_meta"]["seqot"] = (f"SeqOT {LABEL}; stage-2 epoch chosen on v2 VAL by L2 R@1 (main-table rule); "
                           f"REPORTED = seqotl2; {rel(selection_path())}")
    s["_meta"]["reported"] = D.REPORTED
    s["_meta"]["train_seed"] = SEED
    s["_meta"]["eval_window"] = "centered: official windows (main table; decision at keyframe q + 15)"
    p.write_text(json.dumps(s, indent=2))
    print(f"AGG_DONE seed {SEED}", flush=True)


# ------------------------------------------------------------ final
def _pinned_aggregator(track):
    man_p = OLD_W / "final_inputs" / "MANIFEST.json"
    assert sha256(man_p) == protocol()["final_inputs_manifest_sha256"], "pinned MANIFEST changed"
    man = read_json(man_p)
    assert man["n_files"] == 48
    for k, h in man["sha256"].items():
        assert sha256(OLD_W / "final_inputs" / k) == h, k
    spec = importlib.util.spec_from_file_location("_pdv2_main_baselines_official",
                                                  OLD_W / "final_inputs/scripts/_pdv2_main_baselines_official.py")
    M = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = M
    spec.loader.exec_module(M)
    M.REPO, M.COMMON = REPO, OLD_W / "final_inputs" / "pdv2_symmetric3_20261013"
    track(man_p)
    return M


def _training_record(seed) -> dict:
    rd = run_dir(seed)
    log = [json.loads(x) for x in (rd / "log.jsonl").read_text().splitlines()]
    gates = [json.loads(x) for x in (rd / "finite_gate.jsonl").read_text().splitlines()]
    sel = read_json(selection_path(seed))
    st = sel["stage2"]
    rec = {"epochs": len(log), "finished": (rd / "finished").read_text().strip(),
           "epoch_list": [r["epoch"] for r in log], "steps_per_epoch": sorted({r["steps"] for r in log}),
           "loss": [r["loss"] for r in log], "loss_finite": bool(all(math.isfinite(r["loss"]) for r in log)),
           "lr_next": [r["lr_next"] for r in log], "train_s": round(sum(r["seconds"] for r in log), 1),
           "p": [g["p"] for g in gates if g["finite"]], "finite_gates_passed": sum(g["finite"] for g in gates),
           "finite_gates_total": len(gates), "checkpoint_sha256_by_epoch": {g["epoch"]: g["sha256"] for g in gates},
           "restarts": (rd / "restarts.jsonl").read_text().splitlines() if (rd / "restarts.jsonl").exists() else [],
           "meta": read_json(rd / "meta.json"),
           "selected_epoch": st["chosen"]["epoch"], "checkpoint": st["chosen"]["checkpoint"],
           "checkpoint_sha256": st["chosen"]["sha256"],
           "val_family_r1_l2": {k: st["epochs"][k]["family_r1_l2"] for k in sorted(st["epochs"], key=int)},
           "val_scoring_s": round(sum(v.get("val_s", 0) or 0 for v in st["epochs"].values()), 1),
           "stage1": sel["stage1"]["chosen"], "selection_sha256": sha256(selection_path(seed))}
    expected_lr = [LR2 * 0.9 ** (e // 5) for e in range(1, EPOCHS2 + 1)]
    rec["lr_schedule_matches"] = bool(np.allclose(rec["lr_next"], expected_lr, rtol=1e-9, atol=0))
    for e in range(1, EPOCHS2 + 1):
        assert sha256(rd / f"amodel_gem{e}.pth.tar") == rec["checkpoint_sha256_by_epoch"][e]
    return rec


def cmd_final() -> None:
    from _pr_eval_lib import MIN_INDEX_GAP
    import _seqot_official_dump as D
    sources = {}

    def track(p):
        p = Path(p)
        sources[rel(p)] = sha256(p)
        return p

    M = _pinned_aggregator(track)
    for k in SOURCES:
        track(REPO / k)
    for f in sorted(HERE.glob("_seqot_stable_gem_20261108*")):
        track(f)
    track(REC / "protocol.json")
    fz = read_json(track(REC / "val_freeze.json"))
    per_seed, ledgers, checks = {}, {}, {}
    for seed in (1, 2, 3):
        folder = odom_dir(seed)
        e = M.aggregate(rel(folder), "dump_seqotl2", "score_l2", "pr_summary_baselines.json", "seqotl2", track)
        assert e is not None, f"seed {seed}: ledgers missing"
        assert e["n_queries"] == 195726 and e["n_positive"] == 61682, (e["n_queries"], e["n_positive"])
        per_seed[str(seed)] = {"source": e["source"], "macro": e["macro"], "city_groups": e["city_groups"],
                               "n_queries": e["n_queries"], "n_positive": e["n_positive"]}
        f = fz["seeds"][str(seed)]
        lst = {}
        c = {"ledgers": 0, "causal": True, "checkpoint_match": True, "finite": True, "gem_numerics_logdomain": True}
        first_ledger = None
        for sensor, seq in P.TEST:
            for m in D.METHODS:
                p = folder / f"dump_{m}_{sensor}_{seq}.npz"
                lst[p.name] = sha256(p)
                first_ledger = min(first_ledger or 1e30, os.path.getmtime(p))
                with np.load(p) as z:
                    meta = json.loads(str(z["meta"]))
                    c["ledgers"] += 1
                    c["causal"] &= bool(np.all(z["j_star"] <= z["query_idx"] - MIN_INDEX_GAP))
                    c["finite"] &= bool(np.all(np.isfinite(z[D.METHODS[m]])))
                    c["gem_numerics_logdomain"] &= GEM_SHA256 in meta.get("gem_numerics", "")
                    for st in ("stage1", "stage2"):
                        c["checkpoint_match"] &= meta["artifacts"][st]["sha256"] == f[st]["sha256"]
            with np.load(desc_dir(seed) / f"{P.seq_name(sensor, seq)}.npz") as z:
                c["finite"] &= bool(np.all(np.isfinite(z["final"])))
                c["checkpoint_match"] &= str(z["stage2_sha256"]) == f["stage2"]["sha256"]
        for st in ("stage1", "stage2"):
            c[f"{st}_checkpoint_sha256_now"] = sha256(REPO / f[st]["checkpoint"]) == f[st]["sha256"]
        c["selection_unchanged_since_freeze"] = sha256(selection_path(seed)) == f["selection_sha256"]
        c["val_freeze_before_first_ledger"] = os.path.getmtime(REC / "val_freeze.json") < first_ledger
        c["selection_before_first_ledger"] = os.path.getmtime(selection_path(seed)) < first_ledger
        fid = read_json(REC / f"s{seed}" / "test_fidelity.json")
        c["stage1_fidelity_sequences"] = len(fid)
        c["stage1_fidelity_bitwise_all"] = all(v["old_final_bitwise_equal"] for v in fid.values())
        c["stage1_fidelity_maxabs"] = max(v["old_final_maxabs"] for v in fid.values())
        tr = _training_record(seed)
        c["training_20_epochs_finite"] = (tr["epoch_list"] == list(range(1, 21)) and tr["loss_finite"]
                                          and tr["finite_gates_passed"] == 20 and tr["steps_per_epoch"] == [PLAN_STEPS])
        c["lr_schedule_matches"] = tr["lr_schedule_matches"]
        ledgers[str(seed)] = lst
        checks[str(seed)] = c
    M_ = M.METRICS
    vals = np.array([[per_seed[s]["macro"][k] for k in M_] for s in ("1", "2", "3")])
    stats = {k: {"mean": float(vals[:, i].mean()), "std": float(vals[:, i].std(ddof=1)),
                 "values": vals[:, i].tolist()} for i, k in enumerate(M_)}
    main_tab = read_json(track(OLD_W / "final_inputs/pdv2_baseline_seeds/summary.json"))["methods"]["seqot"]
    nan_run = read_json(track(OLD_OUT / "summary.json"))
    training = {str(s): _training_record(s) for s in (1, 2, 3)}
    summary = {"method": f"seqot ({LABEL})", "label": LABEL,
               "protocol": rel(REC / "protocol.json"), "protocol_sha256": sha256(REC / "protocol.json"),
               "val_freeze_sha256": sha256(REC / "val_freeze.json"),
               "aggregate": "per seed 45 seq (KITTI07 excluded) -> 6 cities macro; seed mean, sample SD (ddof=1); "
                            "pinned _pdv2_main_baselines_official.aggregate on the 195,726 common queries",
               "reported_ledger": "dump_seqotl2 (official L2)",
               "per_seed": per_seed, "mean_std": stats,
               "references_not_for_selection": {
                   "current_main_table_seqot_code_recipe": main_tab["mean_std"],
                   "paper_recipe_unstabilized_stage2_diverged_e7_preserved_diagnostic": nan_run["mean_std"]},
               "selected_epochs": {s: {"stage1": training[s]["stage1"]["epoch"], "stage2": training[s]["selected_epoch"]}
                                   for s in training},
               "training": training,
               "cost_note": protocol()["cost_note"]}
    write_json(REC / "summary.json", summary)
    va = read_json(REC / "val_audit.json")
    st_ = read_json(REC / "selftest.json")
    ver = {"passed": bool(st_["passed"] and va["passed"] and all(
               c["ledgers"] == 92 and c["causal"] and c["finite"] and c["checkpoint_match"]
               and c["gem_numerics_logdomain"] and c["stage1_checkpoint_sha256_now"]
               and c["stage2_checkpoint_sha256_now"] and c["selection_unchanged_since_freeze"]
               and c["val_freeze_before_first_ledger"] and c["selection_before_first_ledger"]
               and c["stage1_fidelity_sequences"] == 46 and c["stage1_fidelity_maxabs"] < 1e-4
               and c["training_20_epochs_finite"] and c["lr_schedule_matches"] for c in checks.values())),
           "per_seed_checks": checks, "selftest_passed": st_["passed"], "val_audit_passed": va["passed"],
           "common_queries": {s: per_seed[s]["n_queries"] for s in per_seed},
           "positives": {s: per_seed[s]["n_positive"] for s in per_seed},
           "aggregate_asserts": "M.aggregate asserts the method summary reproduction < 1e-12, common query set and "
                                "min_past_gt_dist equal to the reference, j* <= q-30; here also 195726 / 61682",
           "stage1_fidelity_rule": "old GeM on the regenerated test sub-descriptors vs the old stored test "
                                   "descriptors: max abs < 1e-4 (bitwise equality recorded)",
           "ledger_sha256": ledgers, "input_sha256": sources, "test_used_for_selection": False}
    write_json(REC / "verification.json", ver)
    print(json.dumps({"mean": {k: round(v["mean"], 4) for k, v in stats.items()},
                      "std": {k: round(v["std"], 4) for k, v in stats.items()}, "passed": ver["passed"]}), flush=True)
    print("FINAL_PASSED" if ver["passed"] else "FINAL_FAILED", flush=True)
    if not ver["passed"]:
        fail("final", "verification failed", checks=checks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["freeze", "selftest", "valaudit", "train2", "select2", "valfreeze",
                                    "test", "aggregate", "final"])
    a = ap.parse_args()
    {"freeze": cmd_freeze, "selftest": cmd_selftest, "valaudit": cmd_valaudit, "train2": cmd_train2,
     "select2": cmd_select2, "valfreeze": cmd_valfreeze, "test": cmd_test, "aggregate": cmd_aggregate,
     "final": cmd_final}[a.cmd]()


if __name__ == "__main__":
    main()
