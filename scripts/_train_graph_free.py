#!/usr/bin/env python3
"""원래 학습·마이닝·검증 규칙으로 그래프 없는 투영 대조군을 학습한다."""
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]


def main() -> None:
    """이 프로세스의 모델 팩토리만 바꾸고 원래 학습 진입점을 실행한다."""
    import torch
    import gnn.model as model_module
    from gnn.graph_free import create_graph_free_model
    from _train_lag15_cached_mining import install

    assert "--encoder-preset" in sys.argv and "no_interdiff" in sys.argv
    assert "--use-gated-context" in sys.argv
    assert "--gate-initial-alpha" in sys.argv and "0.0625" in sys.argv
    torch.set_num_threads(1)
    install()
    model_module.create_spectral_gnn = create_graph_free_model
    sys.argv[0] = str(REPO / "train_multi_dataset.py")
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
