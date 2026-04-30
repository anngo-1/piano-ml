from __future__ import annotations

from pathlib import Path

import torch

from .config import TrainConfig
from .midi import decode_midi
from .remi import decode_midi_remi
from .remi_bpe import decode_midi_remi_bpe
from .sample import generate_tokens
from .train import build_model


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
        min_pitch=config.sampling.min_pitch,
        max_pitch=config.sampling.max_pitch,
        max_active_notes=config.sampling.max_active_notes,
        tokenizer=config.tokenizer,
    )
    if config.tokenizer == "remi":
        midi = decode_midi_remi(tokens, output_path)
    elif config.tokenizer == "remi_bpe":
        midi = decode_midi_remi_bpe(tokens, output_path, config.bpe_path)
    else:
        midi = decode_midi(tokens, output_path)
    print(f"wrote {output_path} duration={midi.get_end_time():.2f}s tokens={len(tokens)}")
    return output_path
