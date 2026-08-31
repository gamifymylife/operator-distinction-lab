import json

import pytest

from operator_distinction.manifest import (
    ManifestError,
    seal_payload,
    sha256_json,
    verify_sealed_manifest,
)


def _payload() -> dict:
    return {
        "protocol_version": "operator_distinction.v0.2",
        "experiment": "test",
        "claim_level": "synthetic_predictive_validation_only",
        "operator": {
            "families": [
                {"name": "a"},
                {"name": "b"},
                {"name": "c"},
            ]
        },
        "interventions": {},
        "splits": {
            "protocols": ["interpolation", "leave_one_family_out"],
            "evaluation_pool": {"size": 4},
        },
        "evaluators": {},
        "evaluation": {"budgets": [1, 2, 4]},
        "artifacts": {},
    }


def test_canonical_hash_ignores_dictionary_insertion_order() -> None:
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_manifest_tampering_is_rejected() -> None:
    manifest = seal_payload(_payload())
    verify_sealed_manifest(manifest)
    tampered = json.loads(json.dumps(manifest))
    tampered["payload"]["evaluation"]["budgets"] = [1]
    with pytest.raises(ManifestError, match="seal mismatch"):
        verify_sealed_manifest(tampered)
