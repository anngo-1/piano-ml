# pianogen

License: MIT

Unconditional piano audio generation with a compact REMI-token Transformer.

The repo is organized like a compact research project: code in `src/`, current configs in `configs/`, utilities in `scripts/`, and a separate `huggingface/space/` folder for the demo files to upload to Hugging Face.

## Model

```text
checkpoint: models/remi-modern-2048-ft/best_model.pt
config:     configs/config.json
tokenizer:  REMI, 267 tokens
params:     17.4M
context:    2048 tokens
val loss:   2.2208
val PPL:    9.21
```

`configs/config.json` is the default 17.4M parameter model. The 38.2M parameter preset uses the same REMI tokenizer, 2048-token context, GQA, RoPE, SwiGLU, RMSNorm, tied embeddings, cosine LR, and Muon optimizer.

| config | params | shape |
| --- | ---: | --- |
| `configs/config.json` | 17.4M | 8 layers, 384 width, 6 heads |
| `configs/38m.json` | 38.2M | 10 layers, 512 width, 8 heads |

Perplexity is only comparable within the same tokenizer. Do not compare REMI PPL against the old raw event tokenizer or the failed BPE experiment.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --system-site-packages .uv-venv
uv pip install --python .uv-venv/bin/python -e .
```

For the browser demo:

```bash
uv pip install --python .uv-venv/bin/python -e ".[app]"
```

## Generate

```bash
.uv-venv/bin/pianogen sample --output outputs/sample.mid
```

## Dashboard

```bash
.uv-venv/bin/python app.py
```

Open:

```text
http://localhost:7860
```

On a remote VM:

```bash
ssh -L 7860:127.0.0.1:7860 root@YOUR_HOST -p YOUR_PORT -i ~/.ssh/id_ed25519
```

The dashboard exposes audio only: browser preview plus downloadable WAV. MIDI is generated internally for rendering.
On CPU, the dashboard exposes `Fast` and `Quality` modes. `Fast` uses the quantized cached ONNX token-step model at `models/remi-modern-2048-ft/step-int8.onnx` and may sound slightly lower quality. `Quality` uses the FP32 cached ONNX model at `step.onnx`.

## Train

Training is optional for users. The default command trains the current 17.4M parameter REMI2048 model:

```bash
.uv-venv/bin/pianogen --config configs/config.json prepare
.uv-venv/bin/pianogen --config configs/config.json train
```

To train the 38.2M parameter model:

```bash
.uv-venv/bin/pianogen --config configs/38m.json train
```

## Inspect Generated MIDI

```bash
.uv-venv/bin/python scripts/midi_stats.py outputs/*.mid
```

## Hugging Face Space

Use only the files in:

```text
huggingface/space/
```

That folder is inference-only. Do not upload MAESTRO data, run logs, training outputs, or experiment artifacts to the Space.

The checkpoint is about 67 MB. Approximate download time:

- 100 Mbps: 6 seconds
- 25 Mbps: 22 seconds
- 10 Mbps: 55 seconds

## Layout

```text
app.py                  local dashboard launcher
configs/                current configs only
huggingface/space/      files to copy into a Hugging Face Space
scripts/                focused command-line utilities
src/                    package source
```

Generated artifacts are ignored by default: `data/`, `models/`, `outputs/`, `runs/`, and rendered media. Use Git LFS or Hugging Face for model checkpoints.

## Notes

The current recommended branch is plain REMI with the 2048-context fine-tune. The 38.2M parameter config is a scale preset, not an automatically better checkpoint.

## Publish Checklist

Before pushing publicly:

- License is MIT (`LICENSE`).
- Add the checkpoint with Git LFS if publishing weights in GitHub or Hugging Face.
- Keep generated artifacts out of git: `data/`, `models/`, `outputs/`, `runs/`, rendered media.
- Hugging Face requirements point at `https://github.com/anngo-1/piano-ml`.
- Copy only `huggingface/space/` into the Hugging Face Space.
- Include `MODEL_CARD.md` with the model release.
