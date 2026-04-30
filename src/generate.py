from __future__ import annotations

from pathlib import Path

import torch

from .config import TrainConfig, config_from_dict, load_config
from .remi import decode_midi_remi
from .sample import generate_tokens
from .train import build_model

_LEGACY_CONFIGS = {
    "remi-17m": Path("configs/config.json"),
    "remi-38m": Path("configs/38m.json"),
}


def load_config_from_checkpoint(checkpoint: str | Path) -> TrainConfig | None:
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or "config" not in payload:
        return load_legacy_config_for_checkpoint(checkpoint)

    raw_config = payload["config"]
    if isinstance(raw_config, dict):
        return config_from_dict(raw_config)
    if isinstance(raw_config, (str, Path)) and Path(raw_config).exists():
        return load_config(raw_config)
    return load_legacy_config_for_checkpoint(checkpoint)


def load_legacy_config_for_checkpoint(checkpoint: Path) -> TrainConfig | None:
    config_path = _LEGACY_CONFIGS.get(checkpoint.parent.name)
    if config_path is not None and config_path.exists():
        return load_config(config_path)
    return None


def resolve_config(config_path: str | Path | None, checkpoint: str | Path) -> TrainConfig:
    if config_path is not None:
        return load_config(config_path)
    config = load_config_from_checkpoint(checkpoint)
    if config is None:
        raise ValueError("Checkpoint does not include a loadable config. Pass --config for older checkpoints.")
    return config


def load_checkpoint(config: TrainConfig, checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    model = build_model(config, device)
    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    return model


def generate(config: TrainConfig, checkpoint: str | Path, output: str | Path | None = None) -> Path:
    config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(config, checkpoint, device)
    output_path = Path(output) if output is not None else config.output_dir / "sample.mid"
    tokens = generate_tokens(
        model,
        prompt=config.sampling.prompt,
        length=config.generation_len,
        seq_len=config.seq_len,
        temperature=config.sampling.temperature,
        top_k=config.sampling.top_k,
        top_p=config.sampling.top_p,
        device=device,
        pad_token=config.token_pad,
        repetition_penalty=config.sampling.repetition_penalty,
        constrained=config.sampling.constrained,
    )
    midi = decode_midi_remi(tokens, output_path)
    print(f"wrote {output_path} duration={midi.get_end_time():.2f}s tokens={len(tokens)}")
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default="models/remi-17m/best_model.pt")
    parser.add_argument("--output", default="outputs/sample.mid")
    args = parser.parse_args()

    from .config import seed_everything

    config = resolve_config(args.config, args.checkpoint)
    seed_everything(config.seed)
    generate(config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
