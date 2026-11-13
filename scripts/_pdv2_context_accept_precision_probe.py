#!/usr/bin/env python3
"""접두 검사 오차를 모든 위치·정밀도에서 기록한다. 결과 선택은 하지 않는다."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
import _pdv2_context_accept as run
from _pdv2_context_accept_lib import unit
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """실패 조건을 완화하지 않고 원 오차의 크기와 위치를 저장한다."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    graph = torch.load(run.previous.GRAPH, map_location="cpu", weights_only=False)
    keep = graph.edge_type == 0
    graph.edge_index, graph.edge_attr, graph.edge_type = graph.edge_index[:,keep], graph.edge_attr[keep], graph.edge_type[keep]
    graph = graph.to("cuda")
    _, model = run.exp.old.load_model(3, "standard_original", 10)
    meta = run.exp.arrays(run.PRIOR/"training_metadata.npz")
    queries = []
    for ordinal in range(8):
        ids = np.flatnonzero(meta["sequences"] == ordinal)
        queries.extend(int(ids[q]) for q in (len(ids)//3, 2*len(ids)//3))
    records, values = {}, {}
    for precision in (32, 64):
        if precision == 64:
            graph.x, graph.edge_attr = graph.x.double(), graph.edge_attr.double()
            model = model.double()
        with torch.no_grad():
            full = unit(model(graph).cpu().numpy())
            again = unit(model(graph).cpu().numpy())
        records[str(precision)] = {"repeat_error": float(np.max(np.abs(full-again))), "prefix": []}
        values[str(precision)] = full
        np.save(run.OUT/f"precision_probe_fp{precision}.npy", full)
        for q in queries:
            mask = (graph.edge_index[0] < q+16) & (graph.edge_index[1] < q+16)
            small = Data(x=graph.x[:q+16], edge_index=graph.edge_index[:,mask], edge_attr=graph.edge_attr[mask], edge_type=graph.edge_type[mask])
            with torch.no_grad():
                one = unit(model(small)[q:q+1].cpu().numpy())[0]
            error = float(np.max(np.abs(one-full[q])))
            records[str(precision)]["prefix"].append({"global_query": q, "max_error": error})
            print("PRECISION", precision, q, error, flush=True)
    result = {"records": records, "fp32_fp64_all_error": float(np.max(np.abs(values["32"]-values["64"]))),
              "sources": {str(p.relative_to(REPO)): sha256(p) for p in (Path(__file__), run.previous.GRAPH,
                run.OUT/"precision_probe_fp32.npy", run.OUT/"precision_probe_fp64.npy")}}
    write_json(run.OUT/"precision_probe.json", result)
    print("CROSS", result["fp32_fp64_all_error"], flush=True)


if __name__ == "__main__":
    main()
