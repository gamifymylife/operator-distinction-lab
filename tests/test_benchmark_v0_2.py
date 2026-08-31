import json

from operator_distinction.benchmark_v0_2 import run_v0_2
from operator_distinction.manifest import write_sealed_manifest


def test_small_sealed_benchmark_writes_complete_artifacts(tmp_path) -> None:
    payload = {
        "protocol_version": "operator_distinction.v0.2",
        "experiment": "small_v0_2_test",
        "claim_level": "synthetic_predictive_validation_only",
        "operator": {
            "grid_size": 32,
            "viscosity": 0.02,
            "dt": 0.0025,
            "steps": 5,
            "families": [
                {"name": "spectral_cutoff", "parameter": "modes_kept", "values": [3]},
                {"name": "viscosity_scale", "parameter": "scale", "values": [0.7]},
                {"name": "advection_scale", "parameter": "scale", "values": [0.8]},
                {"name": "forcing_bias", "parameter": "amplitude", "values": [0.04]},
            ],
        },
        "interventions": {"modes": 6, "target_rms": 0.5},
        "splits": {
            "calibration_pool": {"size": 12, "seed": 101},
            "evaluation_pool": {"size": 8, "seed": 202},
            "protocols": ["interpolation", "leave_one_family_out"],
        },
        "evaluators": {
            "ridge": 0.01,
            "early_horizon_steps": 2,
            "feature_contract": {"candidate_implementation_access": False},
        },
        "evaluation": {
            "budgets": [2, 4, 8],
            "threshold_quantile": 0.75,
            "random_seeds": [11, 23, 47],
            "bootstrap_repeats": 64,
            "statistics_seed": 303,
        },
        "artifacts": {"schema_version": "test"},
    }
    manifest_path = tmp_path / "manifest.json"
    write_sealed_manifest(payload, manifest_path)
    output = tmp_path / "results"
    summary = run_v0_2(manifest_path, output)
    assert set(summary["protocols"]) == {"interpolation", "leave_one_family_out"}
    assert summary["claim_level"] == "synthetic_predictive_validation_only"
    expected = {
        "manifest.json",
        "summary.json",
        "split_audit.json",
        "provenance.json",
        "predictions.jsonl",
        "evaluations.jsonl",
        "case_metrics.csv",
        "report.html",
    }
    assert expected == {path.name for path in output.iterdir()}
    audit = json.loads((output / "split_audit.json").read_text())
    assert audit["pool_seeds_are_distinct"] is True
    assert len(summary["protocols"]["leave_one_family_out"]["fold_audits"]) == 4
