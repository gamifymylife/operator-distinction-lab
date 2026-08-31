from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import load_config, run_experiment


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        summary = run_experiment(load_config(args.config), args.output)
        compact = {name: rows[-1] for name, rows in summary["policies"].items()}
        print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
