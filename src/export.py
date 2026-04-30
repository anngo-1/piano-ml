from __future__ import annotations

from pathlib import Path

import torch

from .config import TrainConfig
from .generate import load_checkpoint


def export_onnx(config: TrainConfig, checkpoint: str | Path, output: str | Path) -> Path:
    device = torch.device("cpu")
    model = load_checkpoint(config, checkpoint, device)
    model.eval()
    dummy = torch.ones(1, min(config.seq_len, 128), dtype=torch.long)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["tokens"],
        output_names=["logits"],
        dynamic_axes={"tokens": {1: "sequence"}, "logits": {1: "sequence"}},
        opset_version=17,
    )
    print(f"wrote {output}")
    return output
