import numpy as np
import pytest

from operator_distinction.validation import (
    SealedTargets,
    fit_ridge,
    intrinsic_features,
    paired_discrepancy_features,
    ranking_rows,
)


def test_sealed_targets_require_prediction_commit_before_one_time_reveal() -> None:
    targets = np.asarray([0.1, 0.4, 0.2])
    sealed = SealedTargets.from_targets(targets)
    commits, revealed = sealed.commit_and_reveal({"model": np.asarray([0.2, 0.3, 0.1])})
    assert len(commits["model"]) == 64
    assert np.array_equal(revealed, targets)
    with pytest.raises(RuntimeError, match="only once"):
        sealed.commit_and_reveal({"model": targets})


def test_ridge_and_feature_contract_shapes() -> None:
    states = np.stack([np.sin(np.linspace(0, 2 * np.pi, 32, endpoint=False) + p) for p in range(4)])
    features = np.stack([intrinsic_features(state) for state in states])
    model = fit_ridge(features, np.arange(4, dtype=float), ridge=0.01)
    assert model.predict(features).shape == (4,)
    paired = paired_discrepancy_features(states, states + 0.1)
    assert paired.shape == (10,)


def test_ranking_reports_oracle_regret_and_threshold_cost() -> None:
    target = np.asarray([0.1, 0.5, 0.2, 0.4])
    rows = ranking_rows(target, np.asarray([1, 3, 2, 0]), [1, 2], 0.75)
    assert rows[0]["oracle_fraction"] == 1.0
    assert rows[0]["simple_regret"] == 0.0
    assert rows[0]["cost_to_threshold"] == 1
