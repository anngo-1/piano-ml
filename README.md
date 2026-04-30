# piano-ml

This repository trains a decoder-only Transformer from scratch on the MAESTRO piano dataset for next-token prediction over REMI event tokens. It includes the data tokenization, training, evaluation, sampling, MIDI/audio rendering, and ONNX Runtime serving code needed to turn generated token sequences back into piano music.

This is a research project I built to learn more about training next-token prediction models from scratch. It started as a class project in 2025, and more recently I revisited the code with the goal of improving performance, working on inference, and serving it online.

Live Hugging Face demo: <https://huggingface.co/spaces/anngo-1/piano-ml>

## Model Architecture

The model is a causal decoder-only Transformer over REMI event tokens. It predicts the next token, then decodes generated token streams back into MIDI.

Architecture details:

- 2048-token training/evaluation sequence length in the provided configs
- rotary position embeddings
- grouped-query self-attention
- scaled dot-product attention
- RMSNorm
- SwiGLU feed-forward blocks
- tied token embedding / output projection
- KV-cached generation

Training/inference details:

- MAESTRO v3.0.0 MIDI data
- REMI tokenization
- Muon optimizer
- cosine learning-rate schedule
- ONNX export utility for checkpointed models
- ONNX Runtime step-model serving in the app

## References

These are papers and projects that inspired this implementation. This repository is not a faithful reproduction of any one paper; it combines a decoder-only Transformer music model with REMI-style tokenization and modern Transformer components.

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) introduced the Transformer architecture.
- [Music Transformer](https://arxiv.org/abs/1809.04281) motivated Transformer-based symbolic music generation with long-range structure.
- [Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset](https://arxiv.org/abs/1810.12247) introduced the MAESTRO piano dataset used by this project.
- [Pop Music Transformer](https://arxiv.org/abs/2002.00212) introduced REMI-style beat-based event tokenization for expressive piano generation.
- [RoFormer](https://arxiv.org/abs/2104.09864) introduced rotary position embeddings.
- [GQA](https://aclanthology.org/2023.emnlp-main.298/) introduced grouped-query attention.
- [Root Mean Square Layer Normalization](https://papers.neurips.cc/paper/9403-root-mean-square-layer-normalization) introduced RMSNorm.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) introduced SwiGLU-style feed-forward variants.
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) introduced AdamW.
- [Muon](https://github.com/KellerJordan/Muon) is the optimizer implementation this project follows.

## Training / Eval

Install the base dependencies for data preparation, training, evaluation, and checkpoint-based inference:

```bash
uv sync
```

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
uv run python -m src.eval --checkpoint models/remi-17m/best_model.pt
```

Training writes checkpoints under the `models_dir` in each config. The default config writes to `models/remi-17m/`; the 38.2M config writes to `models/remi-38m/`.
New checkpoints include their training config, so evaluation, generation, and ONNX export can infer the model architecture from the checkpoint.

## Model Configs

| config | params | layers | width | heads | kv heads | mlp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `configs/config.json` | 17.4M | 8 | 384 | 6 | 2 | 1536 |
| `configs/38m.json` | 38.2M | 10 | 512 | 8 | 2 | 2048 |

## Inference

Install FluidSynth for best audio quality: <https://www.fluidsynth.org/wiki/Download/>

Generate a MIDI file from a checkpoint:

```bash
uv run python -m src.generate \
  --checkpoint models/remi-17m/best_model.pt \
  --output outputs/sample.mid
```

Render the MIDI to WAV for listening:

```bash
uv run python -m src.render outputs/sample.mid --output outputs/sample.wav
```

`src.render` renders through FluidSynth with the bundled `soundfonts/GeneralUser-GS.sf2` SoundFont and forces acoustic grand piano. If FluidSynth is unavailable, it falls back to `pretty_midi` synthesis.

Generation uses a KV cache. The app uses cached ONNX Runtime step models for faster local serving.

## App

Install FluidSynth for best audio quality: <https://www.fluidsynth.org/wiki/Download/>

To run the app locally with the published model artifacts:

```bash
uv run --with huggingface_hub python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="anngo-1/piano-ml",
    repo_type="space",
    local_dir=".",
    allow_patterns=[
        "models/remi-17m/best_model.pt",
        "models/remi-17m/step-int8.onnx",
        "models/remi-17m/step.onnx",
        "models/remi-17m/step.onnx.data",
    ],
)
PY
```

```bash
uv sync --extra app
uv run python app.py
```

The app ships with `soundfonts/GeneralUser-GS.sf2` and uses it as the default SoundFont for FluidSynth rendering. Generated MIDI is forced to acoustic grand piano before rendering. Set `PIANO_ML_FLUIDSYNTH_GAIN` to adjust render volume; the default is `0.9`. The bundled SoundFont license is included at `soundfonts/GeneralUser-GS-LICENSE.txt`.
