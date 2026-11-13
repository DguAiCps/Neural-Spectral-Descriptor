"""후보 대응 최적화의 독립 전수검사와 관측 범위 검사."""
import itertools

import numpy as np
import torch

from retrieval.candidate_paths import select_paths, transition_cost


def brute(indices, unary, path, query, radius=1):
    """작은 창의 모든 경로를 열거한다. 생산 DP를 사용하지 않는다."""
    k = indices.shape[1]
    best = np.full(k+1, -np.inf)
    for direction in (1, -1):
        for choices in itertools.product(range(k+1), repeat=2*radius+1):
            value = 0.
            for i, choice in enumerate(choices):
                t = query-radius+i
                if choice == k:
                    value += .5
                elif indices[t, choice] < 0:
                    value = -np.inf
                else:
                    value += unary[t, choice]
                if i:
                    previous = choices[i-1]
                    if previous == k and choice == k:
                        cost = 0.
                    elif previous == k or choice == k:
                        cost = .15
                    else:
                        step = path[t]-path[t-1]
                        delta = direction*(path[indices[t, choice]]-path[indices[t-1, previous]])
                        cost = np.inf if delta < -.8 or delta > 3*step+3.2 else .25*min(abs(delta-step)/max(step, .8), 3.)
                    value -= cost
            best[choices[radius]] = max(best[choices[radius]], value)
    return best


def test_dynamic_program_equals_exhaustive_paths():
    rng = np.random.default_rng(76)
    ids = rng.integers(0, 8, (45, 2))
    unary = rng.uniform(.1, .95, ids.shape)
    ids[34, 1] = -1
    path = np.cumsum(rng.uniform(.2, 1.8, 45))
    for q in (31, 35, 40):
        scores = brute(ids, unary, path, q)
        actual = select_paths(ids, unary, path, np.array([q]), radius=1, device="cpu")
        j = int(np.argmax(scores))
        assert actual["joint"][0] == (-1 if j == 2 else ids[q, j])
        assert actual["forced"][0] == ids[q, np.argmax(scores[:2])]
        np.testing.assert_allclose(actual["objective"][0], scores.max(), atol=1e-12)
        np.testing.assert_allclose(actual["margin"][0], scores[:2].max()-scores[-1], atol=1e-12)


def test_consistent_second_choice_and_reverse_path_recover():
    path = np.arange(80, dtype=float)
    ids = np.full((80, 2), -1, dtype=np.int64)
    unary = np.full(ids.shape, -np.inf)
    ids[45:56, 0] = [0, 20, 1, 21, 2, 22, 3, 23, 4, 24, 5]
    ids[45:56, 1] = np.arange(10, 21)
    unary[45:56] = [.85, .8]
    for reverse in (False, True):
        if reverse:
            ids[45:56, 1] = np.arange(20, 9, -1)
        result = select_paths(ids, unary, path, np.array([50]), device="cpu")
        assert result["joint"][0] == 15
        assert result["direction"][0] == (-1 if reverse else 1)


def test_no_match_missing_candidates_and_prefix_invariance():
    path = np.arange(85, dtype=float)
    ids = np.tile(np.arange(3), (85, 1))
    unary = np.full(ids.shape, .2)
    result = select_paths(ids, unary, path, np.array([40, 60]), device="cpu")
    assert (result["joint"] == -1).all()
    assert np.isfinite(result["objective"]).all()
    ids[:30] = -1
    for q in (30, 40, 60):
        full = select_paths(ids, unary, path, np.array([q]), device="cpu")
        end = q+16
        small = select_paths(ids[:end], unary[:end], path[:end], np.array([q]), device="cpu")
        for key in full:
            np.testing.assert_array_equal(full[key], small[key])


def test_gpu_cpu_agree_and_transition_null_is_finite():
    rng = np.random.default_rng(25)
    ids = rng.integers(0, 10, (70, 4))
    u = rng.uniform(.1, .9, ids.shape)
    path = np.cumsum(rng.uniform(.2, 2., 70))
    cpu = select_paths(ids, u, path, np.arange(35, 55), device="cpu")
    if torch.cuda.is_available():
        gpu = select_paths(ids, u, path, np.arange(35, 55), device="cuda")
        for key in cpu:
            np.testing.assert_allclose(cpu[key], gpu[key], atol=1e-12, rtol=0)
    p = torch.tensor([[0., 100., 0.]])
    c = transition_cost(p, p, torch.tensor([1.]), 1)
    assert torch.isfinite(c[:, -1]).all() and torch.isfinite(c[:, :, -1]).all()
