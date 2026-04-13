"""General command-line entry point for configuration and runtime checks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.utils.config import load_config, resolve_config_path, summarize_config, write_yaml
from src.utils.logging_utils import make_run_name, setup_logging
from src.utils.seed import get_torch_device, set_global_seed


def build_parser() -> argparse.ArgumentParser:
    """Create the project CLI parser."""

    parser = argparse.ArgumentParser(description="ACGD NCI forecasting project entry point.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the fully resolved configuration.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override the configured logging level.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override the configured run name.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load configuration, initialize utilities, and report runtime context."""

    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    experiment = config.setdefault("experiment", {})
    run_name = args.run_name or experiment.get("run_name") or make_run_name("acgd")
    experiment["run_name"] = run_name

    log_level = args.log_level or config.get("logging", {}).get("level", "INFO")
    logs_dir = resolve_config_path(config, "paths.logs_dir", create=True)
    logger = setup_logging(
        log_dir=logs_dir,
        run_name=run_name,
        level=log_level,
        file_enabled=bool(config.get("logging", {}).get("file_enabled", True)),
    )

    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    device = get_torch_device(str(config.get("device", "auto")), require_torch=False)

    resolved_config_path = logs_dir / f"{run_name}_resolved_config.yaml"
    write_yaml(config, resolved_config_path)

    logger.info("Loaded configuration: %s", Path(args.config).resolve())
    logger.info("Run name: %s", run_name)
    logger.info("Seed: %s", seed)
    logger.info("Device: %s", device)
    logger.info("Resolved config written to: %s", resolved_config_path)
    logger.info("Configuration check complete. Use src.pipelines.* modules for data, training, and inference workflows.")

    if args.print_config:
        print(summarize_config(config))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
