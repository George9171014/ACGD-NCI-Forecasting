"""Stage 2 feature-building pipeline for raw or synthetic ACGD data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from src.agents.preprocessing_agent import PreprocessingAgent
from src.data.loaders import coerce_timestamp_and_sort, load_csv_timeseries, save_dataframe_csv, summarize_time_range
from src.data.schemas import TimeSeriesSchema
from src.data.synthetic import generate_synthetic_from_config
from src.data.windows import (
    WindowingConfig,
    build_windows_from_dataframe,
    save_windowed_splits,
    split_windowed_arrays,
)
from src.utils.config import get_config_value, load_config, resolve_config_path, write_yaml
from src.utils.io import timestamped_path, utc_timestamp, write_json
from src.utils.logging_utils import make_run_name, setup_logging
from src.utils.seed import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    """Create the Stage 2 pipeline CLI parser."""

    parser = argparse.ArgumentParser(description="Build preprocessed features and sliding windows.")
    parser.add_argument("--config", type=Path, default=Path("configs/stage2_dummy.yaml"))
    parser.add_argument("--input-csv", type=Path, default=None, help="Optional raw input CSV. Overrides sample generation.")
    parser.add_argument("--generate-sample", action="store_true", help="Generate a synthetic sample dataset before preprocessing.")
    parser.add_argument("--run-name", default=None, help="Optional run name override.")
    return parser


def run_build_features(
    config: Mapping[str, Any],
    *,
    input_csv: Optional[str | Path] = None,
    generate_sample: Optional[bool] = None,
    run_name: Optional[str] = None,
) -> dict[str, Any]:
    """Run Stage 2 data loading, preprocessing, and window generation."""

    experiment = config.get("experiment", {})
    configured_run_name = run_name or experiment.get("run_name") or make_run_name("build_features")
    output_prefix = str(get_config_value(config, "build_features.output_prefix", configured_run_name))
    timestamp = utc_timestamp()
    artifact_stem = f"{output_prefix}_{timestamp}"

    logs_dir = resolve_config_path(config, "paths.logs_dir", create=True)
    reports_dir = resolve_config_path(config, "paths.reports_dir", create=True)
    processed_dir = resolve_config_path(config, "paths.data_processed_dir", create=True)
    sample_dir = resolve_config_path(config, "paths.data_sample_dir", create=True)

    logger = setup_logging(
        log_dir=logs_dir,
        run_name=artifact_stem,
        level=str(get_config_value(config, "logging.level", "INFO")),
        file_enabled=bool(get_config_value(config, "logging.file_enabled", True)),
    )

    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    schema = TimeSeriesSchema.from_config(config)
    should_generate_sample = bool(
        generate_sample
        if generate_sample is not None
        else get_config_value(config, "build_features.generate_sample", False)
    )

    raw_path: Optional[Path] = None
    if input_csv is not None:
        logger.info("Loading raw input CSV: %s", input_csv)
        frame = coerce_timestamp_and_sort(load_csv_timeseries(input_csv), schema)
        raw_path = Path(input_csv).expanduser().resolve()
    elif should_generate_sample:
        logger.info("Generating synthetic hourly ACGD dataset.")
        frame = generate_synthetic_from_config(config)
        raw_path = timestamped_path(sample_dir, f"{output_prefix}_synthetic_raw", ".csv")
        save_dataframe_csv(frame, raw_path)
        logger.info("Synthetic raw data saved to: %s", raw_path)
    else:
        configured_input = get_config_value(config, "data.input_csv")
        if configured_input is None:
            raise ValueError("No input CSV provided. Pass --input-csv or enable build_features.generate_sample.")
        raw_path = _resolve_project_relative_path(config, configured_input)
        frame = coerce_timestamp_and_sort(load_csv_timeseries(raw_path), schema)

    frame = add_auxiliary_forecast_features_if_needed(config, frame)
    schema = TimeSeriesSchema.from_config(config)
    validation_report = schema.validate(frame).to_dict()
    if not validation_report["passed"]:
        raise ValueError(f"Input data failed schema validation: {validation_report['missing_required_columns']}")

    agent = PreprocessingAgent.from_config(config, schema=schema)
    agent_result = agent.fit_transform(frame, schema=schema, report_dir=reports_dir, run_name=artifact_stem)
    processed_frame = agent_result.processed_frame

    processed_path: Optional[Path] = None
    if bool(get_config_value(config, "build_features.save_processed_csv", True)):
        processed_path = processed_dir / f"{artifact_stem}_processed.csv"
        save_dataframe_csv(processed_frame, processed_path)
        logger.info("Processed data saved to: %s", processed_path)

    windows_path: Optional[Path] = None
    windows_summary: Optional[dict[str, Any]] = None
    if bool(get_config_value(config, "build_features.save_windows", True)):
        windowing = WindowingConfig.from_config(config)
        windowed = build_windows_from_dataframe(
            processed_frame,
            feature_columns=list(schema.feature_columns),
            target_columns=schema.target_col,
            timestamp_col=schema.timestamp_col,
            lookback=windowing.lookback,
            horizon=windowing.horizon,
            stride=windowing.stride,
        )
        splits = split_windowed_arrays(
            windowed,
            train_fraction=float(get_config_value(config, "split.train", 0.7)),
            val_fraction=float(get_config_value(config, "split.val", 0.15)),
            test_fraction=float(get_config_value(config, "split.test", 0.15)),
        )
        windows_path = processed_dir / f"{artifact_stem}_windows.npz"
        save_windowed_splits(splits, windows_path)
        windows_summary = {
            "all": windowed.to_dict(),
            "splits": {name: split.to_dict() for name, split in splits.items()},
            "npz_keys": _npz_key_summary(windows_path),
        }
        logger.info("Window arrays saved to: %s", windows_path)

    resolved_config_path = reports_dir / f"{artifact_stem}_resolved_config.yaml"
    write_yaml(config, resolved_config_path)

    manifest = {
        "run_name": artifact_stem,
        "created_at_utc": timestamp,
        "schema_validation": validation_report,
        "raw_data": {
            "path": str(raw_path) if raw_path is not None else None,
            "time_range": summarize_time_range(frame, schema.timestamp_col),
        },
        "processed_data": {
            "path": str(processed_path) if processed_path is not None else None,
            "time_range": summarize_time_range(processed_frame, schema.timestamp_col),
        },
        "preprocessing": agent_result.output_paths,
        "windows": {
            "path": str(windows_path) if windows_path is not None else None,
            "summary": windows_summary,
        },
        "resolved_config": str(resolved_config_path),
    }
    manifest_path = timestamped_path(reports_dir, f"{output_prefix}_build_features_manifest", ".json")
    write_json(manifest, manifest_path)
    logger.info("Build-features manifest saved to: %s", manifest_path)

    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Stage 2 feature building."""

    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    manifest = run_build_features(
        config,
        input_csv=args.input_csv,
        generate_sample=True if args.generate_sample else None,
        run_name=args.run_name,
    )
    print(f"Stage 2 build complete. Manifest: {manifest['manifest_path']}")
    return 0


def _resolve_project_relative_path(config: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_root = Path(get_config_value(config, "runtime.project_root", Path.cwd()))
    return (project_root / path).resolve()


def add_auxiliary_forecast_features_if_needed(config: Mapping[str, Any], frame: Any) -> Any:
    """Add simple offline proxy renewable forecasts when Tier 2 columns are absent.

    This is intended for dummy-data and smoke-test runs. In a full hierarchical
    experiment, these columns should be replaced by Tier 1 ANN outputs.
    """

    if not bool(get_config_value(config, "build_features.auto_auxiliary_forecasts", True)):
        return frame

    result = frame.copy()
    pairs = {
        "pv_generation_forecast": "pv_generation",
        "wt_generation_forecast": "wt_generation",
    }
    for forecast_col, source_col in pairs.items():
        if forecast_col in result.columns or source_col not in result.columns:
            continue
        lagged = result[source_col].shift(24)
        result[forecast_col] = lagged.fillna(method="bfill").fillna(method="ffill")
    return result


def _npz_key_summary(path: str | Path) -> list[str]:
    with np.load(path) as data:
        return list(data.keys())


if __name__ == "__main__":
    raise SystemExit(main())
