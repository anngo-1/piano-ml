# Model Card: pianogen REMI2048

## Summary

`pianogen` is an unconditional symbolic piano generator trained on MAESTRO MIDI. It generates REMI-style tokens and decodes them to piano MIDI; the dashboard renders audio from the generated MIDI and exposes a WAV preview/download.

## Model

```text
checkpoint: models/remi-modern-2048-ft/best_model.pt
config:     configs/config.json
params:     17.4M
tokenizer:  REMI, 267 tokens
context:    2048 tokens
```

Architecture:

- decoder-only Transformer
- RoPE positional encoding
- RMSNorm
- SwiGLU MLP
- grouped-query attention
- PyTorch scaled-dot-product attention

## Training Data

- Dataset: MAESTRO v3.0.0 MIDI
- Domain: solo piano performance MIDI
- Objective: next-token prediction over REMI tokens

## Metrics

```text
validation loss: 2.2208
validation PPL:  9.21
```

PPL is only comparable against models trained with the same tokenizer and validation setup.

## Intended Use

- unconditional piano sketch generation
- MIDI/audio demo applications
- research/educational exploration of symbolic music modeling

## Limitations

- Generates piano only.
- Not conditioned on style, composer, tempo, or prompt.
- Long-range form is imperfect.
- Audio quality depends on the renderer/soundfont, not only the model.
- The model may reproduce stylistic patterns from MAESTRO-like classical piano performances.

## Failed Branches

REMI+BPE512 was tested and sounded poor. The current recommended branch is plain REMI with the 2048-context fine-tune.
