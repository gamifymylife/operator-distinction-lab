from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark_v0_2 import run_v0_2
from .experiment import load_config, run_experiment
from .manifest import load_sealed_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="operator-distinction")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a frozen boundary-search experiment")
    run.add_argument(
        "--config",
        type=Path,
        default=Path("configs/burgers_v0_1.json"),
        help="experiment configuration JSON",
    )
    run.add_argument("--output", type=Path, required=True, help="artifact directory")
    run_v2 = subparsers.add_parser(
        "run-v0-2", help="run the sealed held-out operator-family validation"
    )
    run_v2.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/burgers_v0_2.sealed.json"),
        help="sealed v0.2 protocol manifest",
    )
    run_v2.add_argument("--output", type=Path, required=True, help="artifact directory")
    verify = subparsers.add_parser("verify-manifest", help="verify a v0.2 protocol seal")
    verify.add_argument("manifest", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        summary = run_experiment(load_config(args.config), args.output)
        compact = {name: rows[-1] for name, rows in summary["policies"].items()}
        print(json.dumps(compact, indent=2, sort_keys=True))
    elif args.command == "run-v0-2":
        summary = run_v0_2(args.manifest, args.output)
        compact = {
            protocol: {
                evaluator: rows[-1]
                for evaluator, rows in values["ranking"].items()
            }
            for protocol, values in summary["protocols"].items()
        }
        print(json.dumps(compact, indent=2, sort_keys=True))
    elif args.command == "verify-manifest":
        _, digest = load_sealed_manifest(args.manifest)
        print(digest)


if __name__ == "__main__":
    main()
