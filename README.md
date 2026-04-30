# pianogen

Messing around with a Transformer that generates piano music.

It trains on MAESTRO MIDI, predicts REMI tokens, turns them back into MIDI, and can render WAV audio in a small Gradio app.

## Model

Current model file:

```text
models/remi-17m/best_model.pt
```

Default config:

```text
configs/config.json
```

Architecture:

| config | params | layers | width | heads | kv heads | mlp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `configs/config.json` | 17.4M | 8 | 384 | 6 | 2 | 1536 |
| `configs/38m.json` | 38.2M | 10 | 512 | 8 | 2 | 2048 |

The model is a causal decoder-only Transformer with REMI tokens, RoPE, grouped-query attention, RMSNorm, SwiGLU, tied embeddings, and a 2048-token context.

## Inference

The app has two CPU modes:

- `Fast`: quantized ONNX, quicker but can sound a little worse
- `Quality`: FP32 ONNX

The local sampler uses PyTorch.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --system-site-packages .uv-venv
uv pip install --python .uv-venv/bin/python -e ".[app]"
```

## Run

Generate MIDI:

```bash
uv run python -m src.generate --output outputs/sample.mid
```

Run the app:

```bash
uv run python app.py
```

Open `http://localhost:7860`.

## Train

```bash
uv run python -m src.data --config configs/config.json
uv run python -m src.train --config configs/config.json
```

Try the 38.2M config:

```bash
uv run python -m src.train --config configs/38m.json
```

## Layout

```text
app.py                  local app launcher
configs/                configs
huggingface/space/      files for the Space
src/                    code
```
