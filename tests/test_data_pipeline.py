"""Tests for schema validation, synthetic data, preprocessing, and windowing."""

from __future__ import annotations

from src.agents.preprocessing_agent import PreprocessingAgent
from src.data.schemas import TimeSeriesSchema
from src.data.synthetic import generate_synthetic_hourly_acgd_data
from src.data.windows import build_windows_from_dataframe, split_windowed_arrays
from src.utils.config import load_config


def test_dummy_data_preprocessing_and_windowing() -> None:
    config = load_config("configs/default.yaml")
    schema = TimeSeriesSchema.from_config(config)
    frame = generate_synthetic_hourly_acgd_data(periods=240, missing_fraction=0.01, outlier_fraction=0.005)

    report = schema.validate(frame)
    assert report.passed

    result = PreprocessingAgent.from_config(config, schema=schema).fit_transform(frame, schema=schema)
    assert result.processed_frame[schema.numeric_model_columns()].isna().sum().sum() == 0

    windowed = build_windows_from_dataframe(
        result.processed_frame,
        feature_columns=list(schema.feature_columns),
        target_columns=schema.target_col,
        timestamp_col=schema.timestamp_col,
        lookback=72,
        horizon=24,
    )
    assert windowed.X.shape[1:] == (72, len(schema.feature_columns))
    assert windowed.y.shape[1:] == (24,)

    splits = split_windowed_arrays(windowed, train_fraction=0.7, val_fraction=0.15, test_fraction=0.15)
    assert set(splits) == {"train", "val", "test"}
    assert splits["train"].X.shape[0] > 0

