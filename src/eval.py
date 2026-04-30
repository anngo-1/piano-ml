from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import load_config
from .generate import load_checkpoint


def _load_chunks(data_dir: Path, seq_len: int, pad_token: int) -> list[tuple[list[int], list[int]]]:
    chunks: list[tuple[list[int], list[int]]] = []
    for path in sorted(data_dir.glob("*.pickle")):
        with path.open("rb") as f:
            tokens = list(map(int, pickle.load(f)))
        if len(tokens) < 2:
            continue
        for start in range(0, len(tokens) - 1, seq_len):
            window = tokens[start : start + seq_len + 1]
            if len(window) < 2:
                continue
            src = window[:-1]
            tgt = window[1:]
            if len(src) < seq_len:
                src += [pad_token] * (seq_len - len(src))
                tgt += [pad_token] * (seq_len - len(tgt))
            chunks.append((src, tgt))
    return chunks


def evaluate(config_path: str | Path, checkpoint: str | Path, batch_size: int) -> tuple[float, float, int]:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(config, checkpoint, device).eval()
    chunks = _load_chunks(config.processed_dir / "validation", config.seq_len, config.token_pad)
    if not chunks:
        raise RuntimeError(f"no validation files found in {config.processed_dir / 'validation'}")

    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            src = torch.tensor(np.asarray([item[0] for item in batch]), device=device, dtype=torch.long)
            tgt = torch.tensor(np.asarray([item[1] for item in batch]), device=device, dtype=torch.long)
            logits = model(src)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=config.token_pad,
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_tokens += int((tgt != config.token_pad).sum().item())

    nll = total_loss / max(1, total_tokens)
    return nll, math.exp(nll), total_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--checkpoint", default="models/remi-17m/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    nll, ppl, tokens = evaluate(args.config, args.checkpoint, args.batch_size)
    print(f"tokens={tokens} nll={nll:.6f} ppl={ppl:.4f}")


if __name__ == "__main__":
    main()
