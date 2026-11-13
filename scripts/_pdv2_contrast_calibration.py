#!/usr/bin/env python3
"""공통 성분 감산 뒤 학습 점수 척도의 불일치를 학습8에서 진단한다."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import _pdv2_diffattn_contrast_learn as e

OUT = e.OUT / "calibration_audit"


def fit_score(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict:
    """동일한 양의 로지스틱 적합을 두 변수 Newton법으로 검산한다."""
    from scipy.special import expit

    w = weights / weights.sum()
    mean = float(w @ x)
    scale = max(float(np.sqrt(w @ ((x - mean) ** 2))), 1e-6)
    z = (x - mean) / scale
    ridge = 0.001
    rate = np.clip(float(w @ y), 1e-6, 1 - 1e-6)
    theta = np.array([1.0, np.log(rate / (1 - rate))])

    def objective(t):
        value = z * t[0] + t[1]
        return float(w @ (np.logaddexp(0, value) - y * value) + 0.5 * ridge * t[0] ** 2)

    for iteration in range(100):
        prediction = expit(z * theta[0] + theta[1])
        residual = w * (prediction - y)
        grad = np.array([z @ residual + ridge * theta[0], residual.sum()])
        if np.max(abs(grad)) < 1e-9:
            break
        curvature = w * prediction * (1 - prediction)
        hessian = np.array(
            [
                [curvature @ (z * z) + ridge, curvature @ z],
                [curvature @ z, curvature.sum()],
            ]
        )
        direction = np.linalg.solve(hessian, grad)
        step = 1.0
        old = objective(theta)
        for _ in range(40):
            proposed = theta - step * direction
            if proposed[0] > 1e-6 and objective(proposed) <= old - 1e-4 * step * (
                grad @ direction
            ):
                break
            step *= 0.5
        else:
            assert np.max(abs(grad)) < 1e-7, grad
            break
        theta = proposed
    assert theta[0] > 0 and np.max(abs(grad)) < 1e-7
    return dict(
        mean=[mean],
        scale=[scale],
        coefficients=[float(theta[0])],
        intercept=float(theta[1]),
        loss=objective(theta),
        iterations=iteration,
        ridge=ridge,
        optimizer="damped_newton",
        gradient_inf=float(np.max(abs(grad))),
    )


def main() -> None:
    """같은 학습 승자의 점수·순위·손실과 새 공통 척도를 저장한다."""
    spec = e.prepare()
    OUT.mkdir(exist_ok=True)
    rows, scores, weights, balanced, labels = {}, {}, [], [], []
    for s in spec["train"]:
        name = s.replace("/", "_")
        path = OUT / (name + ".npz")
        data = e.e.run.ARR(e.e.prior.PARENT / "pools/train" / name / "pool.npz")
        row = {
            k: data[k]
            for k in (
                "query_idx",
                "j_star",
                "min_past_gt_dist",
                "top1_geo_dist",
                "group",
            )
        }
        if path.exists():
            values = e.e.run.ARR(path)
        else:
            graph = torch.load(
                e.OUT / "train_inputs" / (name + ".pt"),
                map_location="cuda",
                weights_only=False,
            )
            plan = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
            values = {}
            with torch.no_grad():
                for tag, eta in (("identity", 0.0), ("contrast", -0.5)):
                    z = e.e.prior.filtered(graph, eta)
                    parts = []
                    for start in range(0, len(row["query_idx"]), 128):
                        pp = {
                            k: torch.from_numpy(v[start : start + 128]).cuda()
                            for k, v in plan.items()
                        }
                        parts.append(e.e.prior.score(z, pp).cpu().numpy())
                    values[tag] = np.concatenate(parts).astype(np.float64)
            np.savez_compressed(path, **values)
            del graph, z
        keep = data["eligible"]
        y = data["correct"][keep].astype(np.float64)
        w = np.ones(len(y)) / len(y)
        classes = np.unique(y)
        b = np.zeros(len(y))
        for c in classes:
            b[y == c] = 1.0 / (len(classes) * int((y == c).sum()))
        weights.append(w)
        balanced.append(b)
        labels.append(y)
        for tag, value in values.items():
            scores.setdefault(tag, []).append(value[keep])
        rows[s] = {
            tag: e.e.run.base.independent_metrics(row, v) for tag, v in values.items()
        }
        print("TRAIN_SCORED", s, flush=True)
    weights, balanced, y = (
        np.concatenate(weights),
        np.concatenate(balanced),
        np.concatenate(labels),
    )
    weights /= weights.sum()
    balanced /= balanced.sum()
    scores = {k: np.concatenate(v) for k, v in scores.items()}
    fits = {}
    for tag, x in scores.items():
        for name, w in (("natural", weights), ("balanced", balanced)):
            fits[tag + "_" + name] = fit_score(x, y, w)

    def bce(x, w):
        return float(w @ (np.logaddexp(0, x) - y * x))

    losses = {}
    for tag, x in scores.items():
        losses[tag] = {"old_natural": bce(x, weights), "old_balanced": bce(x, balanced)}
        for name, fit in fits.items():
            losses[tag][name] = bce(e.e.run.predict_score(x[:, None], fit), balanced)
    results = {
        tag: e.e.run.base.macro({s: r[tag] for s, r in rows.items()}) for tag in scores
    }
    gradient = {"old": [], **{k: [] for k in fits}}
    for ordinal, s in enumerate(spec["train"]):
        name = s.replace("/", "_")
        data = e.e.run.ARR(e.e.prior.PARENT / "pools/train" / name / "pool.npz")
        graph = torch.load(
            e.OUT / "train_inputs" / (name + ".pt"),
            map_location="cuda",
            weights_only=False,
        )
        groups = [
            np.flatnonzero(data["eligible"] & (data["correct"] == c)) for c in (0, 1)
        ]
        groups = [v for v in groups if len(v)]
        rng = np.random.default_rng(46000 + ordinal)
        count = 64 // len(groups)
        chosen = rng.permutation(
            np.concatenate(
                [rng.choice(v, count, replace=len(v) < count) for v in groups]
            )
        )
        plan = {
            k: torch.from_numpy(data["plan_" + k][chosen]).cuda()
            for k in ("query", "candidate", "weight")
        }
        eta = torch.tensor(-0.5, device="cuda", requires_grad=True)
        z = e.e.prior.transport_difference(
            graph.spectrum,
            graph.transport_indices,
            graph.transport_mask,
            graph.transport_phase,
            eta.expand(2, len(graph.x), 1),
        )
        value = e.e.prior.score(z, plan)
        yb = torch.from_numpy(data["correct"][chosen]).cuda().float()
        for tag in gradient:
            if tag == "old":
                logit = value
            else:
                f = fits[tag]
                logit = (value - f["mean"][0]) * f["coefficients"][0] / f["scale"][
                    0
                ] + f["intercept"]
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            g = float(torch.autograd.grad(loss, eta, retain_graph=True)[0])
            gradient[tag].append(
                dict(sequence=s, loss=float(loss.detach()), d_loss_d_eta=g)
            )
        del graph, z, value, loss
    rec = dict(
        training_results=results,
        losses=losses,
        fits=fits,
        initial_gradient=gradient,
        correct_rate_natural=float(weights @ y),
        correct_rate_balanced=float(balanced @ y),
        validation_used=False,
        test_used=False,
        sources={
            str(p.relative_to(e.e.ROOT)): e.e.run.HASH(p)
            for p in sorted(OUT.glob("*.npz"))
        },
        source_sha256=e.e.run.HASH(Path(__file__)),
    )
    e.e.run.WRITE(OUT / "summary.json", rec)
    print(
        "AUDIT",
        results,
        losses,
        {k: np.mean([v["d_loss_d_eta"] for v in r]) for k, r in gradient.items()},
        flush=True,
    )


if __name__ == "__main__":
    e.e.configure()
    main()
