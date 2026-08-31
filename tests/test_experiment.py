import json

from operator_distinction.experiment import run_experiment


def test_small_experiment_writes_complete_artifacts(tmp_path) -> None:
    config = {
        "seed": 123,
        "pool_size": 16,
        "modes": 6,
        "grid_size": 32,
        "viscosity": 0.02,
        "dt": 0.0025,
        "steps": 4,
        "candidate_modes": 3,
        "target_rms": 0.5,
        "budgets": [4, 8],
        "random_seeds": [11, 23, 47],
        "neighborhood_size": 4,
        "neighborhood_sigma": 0.02,
    }
    output = tmp_path / "run"
    summary = run_experiment(config, output)
    assert summary["claim_level"] == "instrument_validation_only"
    assert (output / "summary.json").is_file()
    assert (output / "evaluations.jsonl").is_file()
    assert (output / "best_witness.npz").is_file()
    assert (output / "generalization.json").is_file()
    persisted = json.loads((output / "summary.json").read_text())
    assert set(persisted["policies"]) == {
        "fixed",
        "adaptive_v0_1",
        "random_11",
        "random_23",
        "random_47",
    }
