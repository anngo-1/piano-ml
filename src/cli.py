from __future__ import annotations

import argparse

from .config import load_config, seed_everything
from .data import prepare_maestro
from .export import export_onnx
from .generate import generate
from .train import train

DEFAULT_CONFIG = "configs/config.json"
DEFAULT_CHECKPOINT = "models/remi-modern-2048-ft/best_model.pt"


def main() -> None:
    parser = argparse.ArgumentParser(prog="pianogen")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config JSON.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Download and tokenize MAESTRO.")
    prep.add_argument("--overwrite", action="store_true")

    sub.add_parser("train", help="Train from the selected config.")

    sample = sub.add_parser("sample", aliases=["generate"], help="Generate a MIDI sample.")
    sample.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    sample.add_argument("--output", default="outputs/sample.mid")

    export = sub.add_parser("export-onnx", help="Export a checkpoint to ONNX.")
    export.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    export.add_argument("--output", default="models/model.onnx")

    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config.seed)

    if args.cmd == "prepare":
        prepare_maestro(config, overwrite=args.overwrite)
    elif args.cmd == "train":
        train(config)
    elif args.cmd in {"sample", "generate"}:
        generate(config, args.checkpoint, args.output)
    elif args.cmd == "export-onnx":
        export_onnx(config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
