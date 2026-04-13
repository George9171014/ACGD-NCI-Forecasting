"""Evaluate saved prediction and target arrays."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from src.agents.evaluation_agent import EvaluationAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved NPZ containing predictions and targets.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--run-name", default="saved_predictions")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with np.load(args.npz) as data:
        predictions = data["predictions"]
        targets = data["targets"]
    report = EvaluationAgent().evaluate(
        targets,
        predictions,
        split="saved",
        report_dir=args.report_dir,
        run_name=args.run_name,
    )
    print(f"Evaluation report saved to: {report.get('path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

