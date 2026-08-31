from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """Raised when a sealed protocol manifest is malformed or has been changed."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "payload": payload,
        "seal": {"algorithm": "sha256", "digest": sha256_json(payload)},
    }


def verify_sealed_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if set(manifest) != {"payload", "seal"}:
        raise ManifestError("sealed manifest must contain exactly payload and seal")
    payload = manifest["payload"]
    seal = manifest["seal"]
    if not isinstance(payload, dict) or not isinstance(seal, dict):
        raise ManifestError("payload and seal must be JSON objects")
    if seal.get("algorithm") != "sha256":
        raise ManifestError("only sha256 protocol seals are supported")
    observed = sha256_json(payload)
    expected = seal.get("digest")
    if expected != observed:
        raise ManifestError(f"protocol seal mismatch: expected {expected}, observed {observed}")
    _validate_v0_2_payload(payload)
    return payload


def load_sealed_manifest(path: Path) -> tuple[dict[str, Any], str]:
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    payload = verify_sealed_manifest(manifest)
    return payload, manifest["seal"]["digest"]


def write_sealed_manifest(payload: dict[str, Any], path: Path) -> str:
    manifest = seal_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest["seal"]["digest"]


def _validate_v0_2_payload(payload: dict[str, Any]) -> None:
    if payload.get("protocol_version") != "operator_distinction.v0.2":
        raise ManifestError("protocol_version must be operator_distinction.v0.2")
    required = {
        "experiment",
        "claim_level",
        "operator",
        "interventions",
        "splits",
        "evaluators",
        "evaluation",
        "artifacts",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ManifestError(f"manifest payload is missing: {', '.join(missing)}")
    budgets = payload["evaluation"].get("budgets", [])
    if not budgets or budgets != sorted(set(budgets)) or min(budgets) < 1:
        raise ManifestError("evaluation budgets must be positive, sorted, and unique")
    evaluation_size = int(payload["splits"]["evaluation_pool"]["size"])
    if max(budgets) > evaluation_size:
        raise ManifestError("maximum budget exceeds the evaluation pool size")
    families = payload["operator"].get("families", [])
    family_names = [entry.get("name") for entry in families]
    if len(families) < 3 or len(set(family_names)) != len(family_names):
        raise ManifestError("at least three uniquely named operator families are required")
    if payload["splits"].get("protocols") != ["interpolation", "leave_one_family_out"]:
        raise ManifestError("v0.2 requires interpolation and leave_one_family_out protocols")
