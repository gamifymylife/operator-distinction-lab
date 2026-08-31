import numpy as np

from operator_distinction.metrics import compare_trajectories
from operator_distinction.probes import generate_probe_pool
from operator_distinction.search import AdaptivePolicy, RandomPolicy, run_search


def _evaluator(probe):
    reference = np.zeros((2, 16))
    candidate = np.full((2, 16), np.linalg.norm(probe.features()))
    return compare_trajectories(reference, candidate), 0.001


def test_probe_pool_is_deterministic() -> None:
    left = generate_probe_pool(size=12, modes=4, seed=7)
    right = generate_probe_pool(size=12, modes=4, seed=7)
    assert left == right


def test_random_policy_is_reproducible() -> None:
    pool = generate_probe_pool(size=20, modes=4, seed=7)
    left = run_search(pool, RandomPolicy(11), 8, _evaluator)
    right = run_search(pool, RandomPolicy(11), 8, _evaluator)
    assert [item.sequence for item in left] == [item.sequence for item in right]


def test_adaptive_policy_never_repeats_probe() -> None:
    pool = generate_probe_pool(size=24, modes=4, seed=9)
    evaluations = run_search(pool, AdaptivePolicy(warmup=4), 16, _evaluator)
    selected = [item.sequence for item in evaluations]
    assert len(selected) == len(set(selected))
