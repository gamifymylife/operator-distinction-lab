import numpy as np

from operator_distinction.metrics import compare_trajectories


def test_identical_trajectories_have_zero_discrepancy() -> None:
    trajectory = np.arange(24, dtype=np.float64).reshape(3, 8)
    report = compare_trajectories(trajectory, trajectory.copy())
    assert report.selection_score == 0.0
    assert np.count_nonzero(report.residual) == 0


def test_vector_evidence_is_preserved() -> None:
    reference = np.zeros((4, 16))
    candidate = reference.copy()
    candidate[2, 5] = 1.0
    report = compare_trajectories(reference, candidate)
    assert report.residual.shape == (4, 16)
    assert report.l2_by_time.shape == (4,)
    assert report.spectral_energy_by_time.shape == (4, 3)
    assert report.selection_score > 0.0
