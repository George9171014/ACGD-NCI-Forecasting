"""Inference CLI for a saved Tier 1 or Tier 2 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from src.utils.config import load_config, resolve_config_path
from src.utils.io import timestamped_path
from src.utils.seed import get_torch_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference from a saved checkpoint and window NPZ file.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--windows-npz", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--run-name", default="inference")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import torch

        from src.models.tier1_ann import Tier1ANNForecaster
        from src.models.tier2_acgd import Tier2ACGDForecaster
        from src.training.checkpointing import load_checkpoint
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError("PyTorch is required for inference. Install dependencies with: pip install -r requirements.txt") from exc
        raise

    config = load_config(args.config)
    device = get_torch_device(str(config.get("device", "auto")))

    with np.load(args.windows_npz) as data:
        features = data[f"X_{args.split}"]

    model_name = str(config.get("model", {}).get("name", ""))
    if model_name == "tier1_ann":
        model = Tier1ANNForecaster.from_config(config, input_dim=int(features.shape[1] * features.shape[2]))
    elif model_name == "tier2_cnn_gru_dsam":
        model = Tier2ACGDForecaster.from_config(config, input_dim=int(features.shape[2]))
    else:
        raise ValueError(f"Unsupported model for inference: {model_name}")

    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.to(device)
    model.eval()

    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(tensor)
        predictions = output.predictions if hasattr(output, "predictions") else output

    reports_dir = resolve_config_path(config, "paths.reports_dir", create=True)
    output_path = timestamped_path(reports_dir, f"{args.run_name}_{args.split}_predictions", ".npz")
    np.savez_compressed(output_path, predictions=predictions.detach().cpu().numpy())
    print(f"Inference predictions saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
