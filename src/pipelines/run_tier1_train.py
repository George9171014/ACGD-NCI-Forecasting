"""CLI for training a Tier 1 PV or WT ANN forecaster."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.pipelines.build_features import run_build_features
from src.utils.config import load_config
from src.utils.io import utc_timestamp
from src.utils.seed import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Tier 1 renewable ANN forecaster.")
    parser.add_argument("--config", type=Path, default=Path("configs/tier1_pv.yaml"))
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--windows-npz", type=Path, default=None)
    parser.add_argument("--generate-sample", action="store_true")
    parser.add_argument("--run-name", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from src.training.trainer_tier1 import train_tier1_from_windows
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError("PyTorch is required for training. Install dependencies with: pip install -r requirements.txt") from exc
        raise

    config = load_config(args.config)
    set_global_seed(int(config.get("seed", 42)))

    configured_name = args.run_name or config.get("experiment", {}).get("run_name", "tier1_ann")
    run_name = f"{configured_name}_{utc_timestamp()}"

    if args.windows_npz is None:
        manifest = run_build_features(
            config,
            input_csv=args.input_csv,
            generate_sample=True if args.generate_sample else None,
            run_name=run_name,
        )
        windows_path = manifest["windows"]["path"]
    else:
        windows_path = args.windows_npz

    result = train_tier1_from_windows(config=config, windows_path=windows_path, run_name=run_name)
    print(f"Tier 1 training complete. best_epoch={result.best_epoch}, best_val_loss={result.best_val_loss:.6f}")
    print(f"Best checkpoint: {result.best_checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
