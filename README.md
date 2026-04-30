# pianogen

Train and sample a decoder-only Transformer that generates piano music as REMI tokens.

## Model Architecture

The model is a causal decoder-only Transformer over REMI event tokens. It predicts the next token, then decodes generated token streams back into MIDI.

Architecture details:

- 2048-token context window
- rotary position embeddings
- grouped-query self-attention
- PyTorch scaled dot-product attention
- RMSNorm
- SwiGLU feed-forward blocks
- tied token embedding / output projection
- KV-cached autoregressive generation

Training/inference details:

- MAESTRO v3.0.0 MIDI data
- REMI tokenization
- Muon optimizer
- cosine learning-rate schedule
- optional ONNX Runtime serving path
- optional int8 CPU inference path

## Install Dependencies

```bash
uv sync
```

For better WAV rendering, install FluidSynth: <https://www.fluidsynth.org/wiki/Download/#distributions>

## Training / Eval

Download MAESTRO v3.0.0 into `data/` and tokenize the train/validation splits into REMI sequences:

```bash
uv run python -m src.data --config configs/config.json
```

Train the 17.4M parameter config:

```bash
uv run python -m src.train --config configs/config.json
```

Train the 38.2M parameter config:

```bash
uv run python -m src.train --config configs/38m.json
```

Evaluate a checkpoint on the validation split:

```bash
uv run python -m src.eval \
  --config configs/config.json \
  --checkpoint models/remi-17m/best_model.pt
```

Training writes checkpoints under the `models_dir` in each config. The default config writes to `models/remi-17m/`; the 38.2M config writes to `models/remi-38m/`.

## Model Configs

| config | params | layers | width | heads | kv heads | mlp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `configs/config.json` | 17.4M | 8 | 384 | 6 | 2 | 1536 |
| `configs/38m.json` | 38.2M | 10 | 512 | 8 | 2 | 2048 |

## Inference

Generate a MIDI file from a checkpoint:

```bash
uv run python -m src.generate \
  --config configs/config.json \
  --checkpoint models/remi-17m/best_model.pt \
  --output outputs/sample.mid
```

Render the MIDI to WAV for listening:

```bash
uv run python -m src.render outputs/sample.mid --output outputs/sample.wav
```

`src.render` uses FluidSynth when it is installed. Otherwise it falls back to `pretty_midi` synthesis, which is lower quality.

PyTorch generation uses a KV cache. The app code can also use ONNX Runtime step models when those files are present. The int8 CPU path is intended for faster CPU serving and can sound slightly worse than FP32.

Optional audio UI, for listening to generated samples:

```bash
uv sync --extra app
uv run python app.py
```
