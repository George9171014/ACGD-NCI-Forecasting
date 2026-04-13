"""Runnable smoke pipeline for the hierarchical ACGD workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.data.loaders import save_dataframe_csv
from src.data.synthetic import generate_synthetic_from_config
from src.pipelines.run_tier1_train import main as tier1_main
from src.pipelines.run_tier2_train import main as tier2_main
from src.utils.config import load_config, resolve_config_path
from src.utils.io import utc_timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local ACGD smoke pipeline.")
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--generate-sample", action="store_true")
    parser.add_argument("--skip-tier1", action="store_true", help="Only train Tier 2.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_csv = args.input_csv
    if args.generate_sample and input_csv is None:
        config = load_config("configs/stage2_dummy.yaml")
        sample_dir = resolve_config_path(config, "paths.data_sample_dir", create=True)
        input_csv = sample_dir / f"full_pipeline_synthetic_raw_{utc_timestamp()}.csv"
        save_dataframe_csv(generate_synthetic_from_config(config), input_csv)

    shared_args: list[str] = []
    if input_csv is not None:
        shared_args.extend(["--input-csv", str(input_csv)])

    if not args.skip_tier1:
        tier1_main(["--config", "configs/tier1_pv.yaml", *shared_args])
        tier1_main(["--config", "configs/tier1_wt.yaml", *shared_args])
    tier2_main(["--config", "configs/tier2_nci.yaml", *shared_args])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
