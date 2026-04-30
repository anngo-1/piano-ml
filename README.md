# pianogen

Unconditional piano generation with a compact decoder-only Transformer trained on MAESTRO MIDI.

The model predicts the next REMI token, decodes tokens back to MIDI, then renders browser-previewable audio for the dashboard. License: MIT.

## Model

The default checkpoint is the 17.4M parameter REMI2048 model:

```text
checkpoint: models/remi-modern-2048-ft/best_model.pt
config:     configs/config.json
tokenizer:  REMI, 267 tokens
context:    2048 tokens
objective:  next-token prediction
```

Architecture:

| config | params | layers | width | heads | kv heads | mlp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `configs/config.json` | 17.4M | 8 | 384 | 6 | 2 | 1536 |
| `configs/38m.json` | 38.2M | 10 | 512 | 8 | 2 | 2048 |

Implementation details:

- decoder-only causal Transformer
- REMI tokenization with a 2048-token context
- grouped-query attention
- rotary position embeddings
- RMSNorm
- SwiGLU feed-forward blocks
- tied input/output embeddings
- PyTorch SDPA for training

Validation PPL is only comparable within the same tokenizer and eval protocol. The 38.2M config is a scale preset, not a promoted checkpoint.

## Results

The released checkpoint is the 17.4M parameter REMI2048 model. The 38.2M parameter config is included for scale experiments, but it is not currently the default release.

## Inference

Generation is unconditional. The sampler starts from the configured beginning token, autoregressively samples REMI tokens, decodes them to MIDI, then renders audio when using the dashboard.

Runtime options:

- `Fast`: quantized cached ONNX token-step model; faster CPU inference, may sound slightly lower quality
- `Quality`: FP32 cached ONNX token-step model
- PyTorch: used by the CLI sampler and training code

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --system-site-packages .uv-venv
uv pip install --python .uv-venv/bin/python -e ".[app]"
```

CLI generation:

```bash
.uv-venv/bin/pianogen sample --output outputs/sample.mid
```

Dashboard:

```bash
.uv-venv/bin/python app.py
```

Open `http://localhost:7860`.

The dashboard exposes audio preview and WAV download. MIDI is generated internally for rendering.

## Train

Prepare MAESTRO and train the 17.4M parameter model:

```bash
.uv-venv/bin/pianogen --config configs/config.json prepare
.uv-venv/bin/pianogen --config configs/config.json train
```

Train the 38.2M parameter preset:

```bash
.uv-venv/bin/pianogen --config configs/38m.json train
```

## Hugging Face Space

`huggingface/space/` contains the inference-only Space files. Do not upload MAESTRO data, run logs, training outputs, or experiment artifacts to the Space.

The released checkpoint is about 67 MB. Model weights should live on Hugging Face, not in the GitHub repo.

## Repository Layout

```text
app.py                  local dashboard launcher
configs/                model/training configs
huggingface/space/      Hugging Face Space files
scripts/                command-line utilities
src/                    package source
```

Generated artifacts are ignored by default: `data/`, `models/`, `outputs/`, `runs/`, and rendered media.
