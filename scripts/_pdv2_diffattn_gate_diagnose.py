#!/usr/bin/env python3
"""게이트 학습 대조의 고정 학습 쌍과 매개변수 범위를 감사한다."""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import _pdv2_diffattn_gate_learning as experiment


def main() -> None:
    """검증 선택과 독립적으로 학습 입력의 변화·체크포인트를 확인한다."""
    run = experiment.run
    run.configure()
    spec = experiment.prepare()
    rows, scope = {}, {}
    for arm in (*experiment.FROZEN, *experiment.ARMS):
        experiment.activate(arm)
        epoch = 0 if arm in experiment.FROZEN else 8
        path = experiment.OUT / arm / f"epoch{epoch}.pth"
        net = run.model(path)
        initial = experiment.load_model(arm=arm)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        changed = []
        for name, value in net.state_dict().items():
            assert torch.isfinite(value).all(), name
            equal = torch.equal(value, initial.state_dict()[name])
            if not equal:
                changed.append(name)
            if "running_" in name or "num_batches_tracked" in name or name.endswith("key_scale"):
                assert equal, name
        if arm in experiment.FROZEN:
            assert not changed
        if arm == "recenter_convs":
            assert changed and all(n.startswith("gnn.convs.") for n in changed)
        if arm == "recenter_gate":
            assert changed and all(n.startswith("gnn.gate.") for n in changed)
        scope[arm] = dict(changed=changed, epoch=saved["epoch"], updates=saved["updates"],
                         checkpoint_sha256=run.HASH(path), fixed_buffers=True)
        fit = run.calibration()
        per = {}
        for seq in spec["train"]:
            folder = experiment.OUT / "pools" / seq.replace("/", "_")
            pool = run.ARR(folder / "pool.npz")
            graph = torch.load(folder / "graph.pt", map_location="cuda", weights_only=False)
            with torch.no_grad():
                key = F.normalize(net(graph), dim=1)
                alpha = net.gnn._last_alpha.cpu().numpy().ravel()
                plan = {k:torch.from_numpy(pool["plan_"+k]).cuda() for k in ("query", "candidate", "weight")}
                context = torch.cat([run.context_score(key,{k:v[i:i+256] for k,v in plan.items()})
                                     for i in range(0,len(plan["query"]),256)]).cpu().numpy()
            mask = pool["eligible"]
            positive = pool["positive"][mask]
            correct = pool["correct"][mask]
            logits = run.predict_score(pool["spectral"][mask,None],fit["spectral"])+(context-fit["context_mean"])/fit["context_scale"]
            categories = {}
            for label,keep in (("correct",correct),("wrong_place",positive&~correct),("nonrevisit",~positive)):
                y = correct[keep].astype(np.float64)
                z = logits[keep]
                categories[label] = dict(count=int(keep.sum()), mean_context=float(context[keep].mean()) if keep.any() else None,
                    mean_logit=float(z.mean()) if keep.any() else None,
                    bce=float((np.logaddexp(0,z)-y*z).mean()) if keep.any() else None)
            per[seq] = dict(alpha_mean=float(alpha.mean()),alpha_min=float(alpha.min()),alpha_max=float(alpha.max()),
                saturated_fraction=float(np.mean((alpha<.01)|(alpha>.99))),categories=categories)
            del graph,key,plan
        rows[arm] = per
        del net,initial
    path = experiment.OUT/"training_diagnostic.json"
    run.WRITE(path,dict(status="passed",fixed_original_winners=True,validation_used=False,test_used=False,
        rows=rows,checkpoint_scope=scope,sources={str(Path(__file__).relative_to(ROOT)):run.HASH(Path(__file__)),
        str((experiment.OUT/"protocol.json").relative_to(ROOT)):run.HASH(experiment.OUT/"protocol.json")}))
    print("AUDIT_PASSED",len(scope),flush=True)


if __name__ == "__main__":
    main()
