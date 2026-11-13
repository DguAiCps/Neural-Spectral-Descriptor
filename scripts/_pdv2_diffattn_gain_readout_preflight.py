#!/usr/bin/env python3
"""출력층 역할 변경의 공통 시작점과 실제 학습 경로를 검증한다."""
import numpy as np
import torch
import torch.nn.functional as F

import _pdv2_diffattn_gain_readout as e


def main() -> None:
    """모든 구조에서 초기 이득을 맞추고 한 갱신 후 두 층의 미분을 확인한다."""
    e.run.configure()
    e.pool("train", "kitti/06")
    data = e.run.ARR(e.OUT / "pools/train/kitti_06/pool.npz")
    graph = torch.load(
        e.prior.PARENT / "pools/kitti_06/graph.pt",
        map_location="cuda",
        weights_only=False,
    )
    z = e.prior.coefficients(e.run.spec_path("kitti/06"))
    chosen = np.flatnonzero(data["eligible"])[::17][:64]
    p = {
        k: torch.from_numpy(data["plan_" + k][chosen]).cuda()
        for k in ("query", "candidate", "weight")
    }
    target = torch.from_numpy(data["correct"][chosen]).cuda().float()
    reference = None
    records = {}
    for kind in e.KINDS:
        net = e.model(kind)
        initial = {n: v.detach().clone() for n, v in net.state_dict().items()}
        params = [v for v in net.parameters() if v.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-5)
        first = e.score(net(graph), z, p)
        if reference is None:
            reference = first.detach()
        torch.testing.assert_close(first, reference, atol=2e-6, rtol=0)
        err = float((first - reference).abs().max().detach())
        F.binary_cross_entropy_with_logits(first, target).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        second = e.score(net(graph), z, p)
        F.binary_cross_entropy_with_logits(second, target).backward()
        grads = {
            n: float(v.grad.norm())
            for n, v in net.named_parameters()
            if v.requires_grad
        }
        assert all(np.isfinite(v) for v in grads.values())
        if kind not in ("head_only", "global"):
            assert sum(v for k, v in grads.items() if ".convs." in k) > 0
        changed = [
            n for n, v in net.state_dict().items() if not torch.equal(v, initial[n])
        ]
        allowed = {n for n, v in net.named_parameters() if v.requires_grad}
        assert set(changed) <= allowed
        records[kind] = dict(
            initial_score_error=err,
            after_one_update_gradients=grads,
            trainable_parameters=sum(p.numel() for p in params),
            changed=changed,
        )
        print("PREFLIGHT", kind, err, sum(grads.values()), flush=True)
        del net, first, second, opt, initial
    e.run.WRITE(
        e.OUT / "preflight.json",
        dict(status="passed", structures=records, test_used=False),
    )


if __name__ == "__main__":
    main()
