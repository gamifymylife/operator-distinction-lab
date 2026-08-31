from __future__ import annotations

import csv
import hashlib
import html
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .burgers import BurgersConfig, BurgersReference
from .manifest import load_sealed_manifest, sha256_json
from .metrics import compare_trajectories
from .operators_v0_2 import OperatorCase, build_candidate, operator_cases
from .probes import Probe, generate_probe_pool
from .validation import (
    SealedTargets,
    fit_ridge,
    intrinsic_features,
    paired_discrepancy_features,
    paired_effect,
    ranking_rows,
    regression_metrics,
    stable_seed,
)


@dataclass(frozen=True)
class CaseData:
    case: OperatorCase
    probe_ids: np.ndarray
    intrinsic: np.ndarray
    paired: np.ndarray
    targets: np.ndarray
    execution_seconds: np.ndarray


def _json_dump(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_case_data(
    probes: list[Probe],
    case: OperatorCase,
    config: BurgersConfig,
    target_rms: float,
    early_horizon: int,
    reference_trajectories: dict[int, np.ndarray],
    reference_seconds: dict[int, float],
) -> CaseData:
    x = np.linspace(0.0, 2.0 * np.pi, config.grid_size, endpoint=False)
    candidate = build_candidate(case, config)
    intrinsic_rows: list[np.ndarray] = []
    paired_rows: list[np.ndarray] = []
    targets: list[float] = []
    execution_seconds: list[float] = []
    probe_ids: list[int] = []
    for probe in probes:
        initial_state = probe.field(x, target_rms)
        reference = reference_trajectories[probe.probe_id]
        candidate_started = time.perf_counter()
        candidate_trajectory = candidate.rollout(initial_state)
        candidate_seconds = time.perf_counter() - candidate_started
        intrinsic_rows.append(intrinsic_features(initial_state))
        paired_rows.append(
            paired_discrepancy_features(
                reference[: early_horizon + 1], candidate_trajectory[: early_horizon + 1]
            )
        )
        # The prediction target excludes the early response used by the paired evaluator.
        late_report = compare_trajectories(
            reference[early_horizon + 1 :], candidate_trajectory[early_horizon + 1 :]
        )
        targets.append(late_report.selection_score)
        execution_seconds.append(reference_seconds[probe.probe_id] + candidate_seconds)
        probe_ids.append(probe.probe_id)
    return CaseData(
        case=case,
        probe_ids=np.asarray(probe_ids, dtype=np.int64),
        intrinsic=np.stack(intrinsic_rows),
        paired=np.stack(paired_rows),
        targets=np.asarray(targets, dtype=np.float64),
        execution_seconds=np.asarray(execution_seconds, dtype=np.float64),
    )


def _build_pool_data(
    probes: list[Probe],
    cases: list[OperatorCase],
    config: BurgersConfig,
    target_rms: float,
    early_horizon: int,
) -> list[CaseData]:
    x = np.linspace(0.0, 2.0 * np.pi, config.grid_size, endpoint=False)
    reference_operator = BurgersReference(config)
    reference_trajectories: dict[int, np.ndarray] = {}
    reference_seconds: dict[int, float] = {}
    for probe in probes:
        reference_started = time.perf_counter()
        reference_trajectories[probe.probe_id] = reference_operator.rollout(
            probe.field(x, target_rms)
        )
        reference_seconds[probe.probe_id] = time.perf_counter() - reference_started
    return [
        _build_case_data(
            probes,
            case,
            config,
            target_rms,
            early_horizon,
            reference_trajectories,
            reference_seconds,
        )
        for case in cases
    ]


def _stack_features(data: list[CaseData], paired: bool) -> np.ndarray:
    rows = [
        np.column_stack([case.intrinsic, case.paired]) if paired else case.intrinsic
        for case in data
    ]
    return np.vstack(rows)


def _stack_targets(data: list[CaseData]) -> np.ndarray:
    return np.concatenate([case.targets for case in data])


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    keys = sorted({(row["evaluator"], int(row["budget"])) for row in rows})
    for evaluator, budget in keys:
        selected = [
            row
            for row in rows
            if row["evaluator"] == evaluator and int(row["budget"]) == budget
        ]
        aggregate.setdefault(evaluator, []).append(
            {
                "budget": budget,
                "cases": len(selected),
                "mean_oracle_fraction": float(
                    np.mean([row["oracle_fraction"] for row in selected])
                ),
                "mean_simple_regret": float(
                    np.mean([row["simple_regret"] for row in selected])
                ),
                "threshold_discovery_probability": float(
                    np.mean([row["threshold_discovered"] for row in selected])
                ),
                "median_cost_to_threshold": float(
                    np.median([row["cost_to_threshold"] for row in selected])
                ),
                "mean_cumulative_seconds": float(
                    np.mean([row["cumulative_seconds"] for row in selected])
                ),
                "mean_mechanism_executions": float(
                    np.mean([row["mechanism_executions"] for row in selected])
                ),
                "mean_screening_mechanism_steps": float(
                    np.mean([row["screening_mechanism_steps"] for row in selected])
                ),
            }
        )
    return aggregate


def _comparison_statistics(
    rows: list[dict[str, Any]],
    budgets: list[int],
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for budget in budgets:
        paired_rows = {
            (row["fold"], row["case_id"]): row
            for row in rows
            if row["evaluator"] == "paired_discrepancy" and row["budget"] == budget
        }
        intrinsic_rows = {
            (row["fold"], row["case_id"]): row
            for row in rows
            if row["evaluator"] == "naive_intrinsic" and row["budget"] == budget
        }
        keys = sorted(set(paired_rows) & set(intrinsic_rows))
        candidate = np.asarray([paired_rows[key]["oracle_fraction"] for key in keys])
        baseline = np.asarray([intrinsic_rows[key]["oracle_fraction"] for key in keys])
        result = paired_effect(candidate, baseline, repeats, seed + budget)
        result.update(
            {
                "budget": budget,
                "estimand": "paired_minus_intrinsic_oracle_fraction",
            }
        )
        results.append(result)
    ordered = sorted(range(len(results)), key=lambda index: results[index]["sign_flip_p_value"])
    running = 0.0
    for rank, index in enumerate(ordered):
        multiplier = len(results) - rank
        adjusted = min(1.0, multiplier * results[index]["sign_flip_p_value"])
        running = max(running, adjusted)
        results[index]["holm_adjusted_p_value"] = running
    return results


def _evaluate_fold(
    protocol: str,
    fold: str,
    calibration: list[CaseData],
    evaluation: list[CaseData],
    manifest: dict[str, Any],
    prediction_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluator_config = manifest["evaluators"]
    ridge = float(evaluator_config["ridge"])
    train_targets = _stack_targets(calibration)
    intrinsic_model = fit_ridge(_stack_features(calibration, paired=False), train_targets, ridge)
    paired_model = fit_ridge(_stack_features(calibration, paired=True), train_targets, ridge)

    intrinsic_test = _stack_features(evaluation, paired=False)
    paired_test = _stack_features(evaluation, paired=True)
    predictions = {
        "naive_intrinsic": intrinsic_model.predict(intrinsic_test),
        "paired_discrepancy": paired_model.predict(paired_test),
    }
    hidden_targets = _stack_targets(evaluation)
    sealed = SealedTargets.from_targets(hidden_targets)
    prediction_commits, revealed_targets = sealed.commit_and_reveal(predictions)

    cursor = 0
    rows: list[dict[str, Any]] = []
    regression: dict[str, dict[str, float]] = {}
    budgets = [int(value) for value in manifest["evaluation"]["budgets"]]
    threshold_quantile = float(manifest["evaluation"]["threshold_quantile"])
    random_seeds = [int(seed) for seed in manifest["evaluation"]["random_seeds"]]
    early_horizon = int(manifest["evaluators"]["early_horizon_steps"])

    regression = {
        name: regression_metrics(revealed_targets, values) for name, values in predictions.items()
    }
    for case_index, case_data in enumerate(evaluation):
        count = len(case_data.targets)
        case_target = revealed_targets[cursor : cursor + count]
        case_predictions = {
            name: values[cursor : cursor + count] for name, values in predictions.items()
        }
        for local_index, probe_id in enumerate(case_data.probe_ids):
            prediction_records.append(
                {
                    "protocol": protocol,
                    "fold": fold,
                    "case_id": case_data.case.case_id,
                    "family": case_data.case.family,
                    "probe_id": int(probe_id),
                    "target_digest": sealed.target_digest,
                    "predictions": {
                        name: float(values[local_index])
                        for name, values in case_predictions.items()
                    },
                    "late_target": float(case_target[local_index]),
                }
            )
        orders: dict[str, np.ndarray] = {
            name: np.argsort(-values, kind="stable") for name, values in case_predictions.items()
        }
        orders["fixed"] = np.arange(count, dtype=np.int64)
        for random_seed in random_seeds:
            rng = np.random.default_rng(
                stable_seed(random_seed, f"{protocol}:{fold}:{case_data.case.case_id}")
            )
            orders[f"random_{random_seed}"] = rng.permutation(count)

        for evaluator, order in orders.items():
            screening_steps = 2 * early_horizon * count if evaluator == "paired_discrepancy" else 0
            started = time.perf_counter()
            evaluator_rows = ranking_rows(case_target, order, budgets, threshold_quantile)
            ranking_seconds = time.perf_counter() - started
            for row in evaluator_rows:
                budget = int(row["budget"])
                row.update(
                    {
                        "protocol": protocol,
                        "fold": fold,
                        "case_id": case_data.case.case_id,
                        "family": case_data.case.family,
                        "operator_parameter": case_data.case.parameter,
                        "operator_value": case_data.case.value,
                        "evaluator": evaluator,
                        "operator_comparisons": budget,
                        "mechanism_executions": 2 * budget,
                        "screening_mechanism_steps": screening_steps,
                        "cumulative_seconds": float(
                            np.sum(case_data.execution_seconds[order[:budget]])
                        ),
                        "ranking_seconds": ranking_seconds,
                    }
                )
                rows.append(row)
        cursor += count
    return rows, {
        "fold": fold,
        "training_families": sorted({data.case.family for data in calibration}),
        "evaluation_families": sorted({data.case.family for data in evaluation}),
        "calibration_examples": len(train_targets),
        "evaluation_examples": len(hidden_targets),
        "target_sha256": sealed.target_digest,
        "prediction_sha256": prediction_commits,
        "regression_metrics": regression,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_html(path: Path, summary: dict[str, Any]) -> None:
    table_rows: list[str] = []
    for protocol, protocol_summary in summary["protocols"].items():
        for evaluator, rows in protocol_summary["ranking"].items():
            final = rows[-1]
            table_rows.append(
                "<tr>"
                f"<td>{html.escape(protocol)}</td><td>{html.escape(evaluator)}</td>"
                f"<td>{final['budget']}</td>"
                f"<td>{final['mean_oracle_fraction']:.4f}</td>"
                f"<td>{final['threshold_discovery_probability']:.4f}</td>"
                "</tr>"
            )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Operator Distinction v0.2</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:8px;text-align:left}}
code{{background:#eee;padding:2px 4px}}</style></head><body>
<h1>Sealed operator-family validation v0.2</h1>
<p>Protocol seal: <code>{html.escape(summary['manifest_sha256'])}</code></p>
<p>Claim level: <strong>{html.escape(summary['claim_level'])}</strong></p>
<table><thead><tr><th>Protocol</th><th>Evaluator</th><th>Max budget</th>
<th>Mean oracle fraction</th><th>Threshold discovery</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>
<p>The paired evaluator uses an early interventional response to predict a disjoint late-horizon
target. Results are synthetic predictive evidence, not causal identification or physical validity.</p>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def run_v0_2(manifest_path: Path, output: Path) -> dict[str, Any]:
    manifest, manifest_digest = load_sealed_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    operator_config = manifest["operator"]
    config = BurgersConfig(
        grid_size=int(operator_config["grid_size"]),
        viscosity=float(operator_config["viscosity"]),
        dt=float(operator_config["dt"]),
        steps=int(operator_config["steps"]),
    )
    early_horizon = int(manifest["evaluators"]["early_horizon_steps"])
    if not 0 < early_horizon < config.steps:
        raise ValueError("early_horizon_steps must be inside the simulated time horizon")

    intervention_config = manifest["interventions"]
    split_config = manifest["splits"]
    calibration_pool = generate_probe_pool(
        size=int(split_config["calibration_pool"]["size"]),
        modes=int(intervention_config["modes"]),
        seed=int(split_config["calibration_pool"]["seed"]),
    )
    evaluation_pool = generate_probe_pool(
        size=int(split_config["evaluation_pool"]["size"]),
        modes=int(intervention_config["modes"]),
        seed=int(split_config["evaluation_pool"]["seed"]),
    )
    cases = operator_cases(operator_config["families"])
    target_rms = float(intervention_config["target_rms"])
    calibration = _build_pool_data(
        calibration_pool, cases, config, target_rms, early_horizon
    )
    evaluation = _build_pool_data(evaluation_pool, cases, config, target_rms, early_horizon)

    split_audit = {
        "protocols": split_config["protocols"],
        "calibration_pool": {
            **split_config["calibration_pool"],
            "probe_sha256": sha256_json([probe.to_dict() for probe in calibration_pool]),
        },
        "evaluation_pool": {
            **split_config["evaluation_pool"],
            "probe_sha256": sha256_json([probe.to_dict() for probe in evaluation_pool]),
        },
        "operator_cases": [case.to_dict() for case in cases],
        "operator_registry_sha256": sha256_json([case.to_dict() for case in cases]),
        "probe_id_namespaces_are_pool_local": True,
        "pool_seeds_are_distinct": (
            split_config["calibration_pool"]["seed"]
            != split_config["evaluation_pool"]["seed"]
        ),
        "evaluation_targets_withheld_until_prediction_commit": True,
    }

    predictions: list[dict[str, Any]] = []
    protocol_rows: dict[str, list[dict[str, Any]]] = {
        "interpolation": [],
        "leave_one_family_out": [],
    }
    fold_audits: dict[str, list[dict[str, Any]]] = {
        "interpolation": [],
        "leave_one_family_out": [],
    }
    rows, audit = _evaluate_fold(
        "interpolation",
        "known_families",
        calibration,
        evaluation,
        manifest,
        predictions,
    )
    protocol_rows["interpolation"].extend(rows)
    fold_audits["interpolation"].append(audit)

    family_names = sorted({case.family for case in cases})
    for held_out_family in family_names:
        train = [data for data in calibration if data.case.family != held_out_family]
        test = [data for data in evaluation if data.case.family == held_out_family]
        rows, audit = _evaluate_fold(
            "leave_one_family_out",
            held_out_family,
            train,
            test,
            manifest,
            predictions,
        )
        protocol_rows["leave_one_family_out"].extend(rows)
        fold_audits["leave_one_family_out"].append(audit)

    evaluation_config = manifest["evaluation"]
    budgets = [int(value) for value in evaluation_config["budgets"]]
    summary: dict[str, Any] = {
        "experiment": manifest["experiment"],
        "protocol_version": manifest["protocol_version"],
        "manifest_sha256": manifest_digest,
        "claim_level": manifest["claim_level"],
        "objective": (
            "Compare a paired discrepancy-aware evaluator with a naive intrinsic baseline "
            "under interpolation and held-out operator-family splits."
        ),
        "protocols": {},
        "elapsed_seconds": time.perf_counter() - started,
    }
    for protocol, rows in protocol_rows.items():
        summary["protocols"][protocol] = {
            "ranking": _aggregate_rows(rows),
            "paired_comparison": _comparison_statistics(
                rows,
                budgets,
                int(evaluation_config["bootstrap_repeats"]),
                int(evaluation_config["statistics_seed"]),
            ),
            "fold_audits": fold_audits[protocol],
        }

    source_root = Path(__file__).parent
    provenance = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": manifest_digest,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "source_sha256": {
            name: _source_hash(source_root / name)
            for name in [
                "benchmark_v0_2.py",
                "manifest.py",
                "operators_v0_2.py",
                "validation.py",
                "metrics.py",
                "probes.py",
                "burgers.py",
            ]
        },
        "burgers_config": asdict(config),
        "feature_contract": manifest["evaluators"]["feature_contract"],
        "outcome_reveal_contract": "predictions_hashed_before_evaluation_targets_revealed",
    }

    (output / "manifest.json").write_bytes(manifest_path.read_bytes())
    _json_dump(output / "summary.json", summary)
    _json_dump(output / "split_audit.json", split_audit)
    _json_dump(output / "provenance.json", provenance)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record in predictions:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    all_rows = protocol_rows["interpolation"] + protocol_rows["leave_one_family_out"]
    with (output / "evaluations.jsonl").open("w", encoding="utf-8") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    _write_csv(output / "case_metrics.csv", all_rows)
    _write_html(output / "report.html", summary)
    return summary
