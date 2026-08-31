import numpy as np

from operator_distinction.burgers import (
    BurgersConfig,
    BurgersReference,
    LowModeBurgersSurrogate,
)


def test_zero_state_is_fixed_point() -> None:
    config = BurgersConfig(grid_size=32, steps=4)
    state = np.zeros(config.grid_size)
    assert np.array_equal(BurgersReference(config).rollout(state), np.zeros((5, 32)))


def test_low_mode_surrogate_matches_reference_for_zero_state() -> None:
    config = BurgersConfig(grid_size=32, steps=4)
    state = np.zeros(config.grid_size)
    reference = BurgersReference(config).rollout(state)
    candidate = LowModeBurgersSurrogate(config, modes_kept=4).rollout(state)
    assert np.array_equal(reference, candidate)


def test_rollout_is_finite_and_has_expected_shape() -> None:
    config = BurgersConfig(grid_size=32, steps=6)
    x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    trajectory = BurgersReference(config).rollout(np.sin(x) + 0.2 * np.sin(5 * x))
    assert trajectory.shape == (7, 32)
    assert np.all(np.isfinite(trajectory))
