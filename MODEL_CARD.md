# Model Card: pianogen REMI2048

## Summary

`pianogen` is an unconditional symbolic piano generator trained on MAESTRO MIDI. It predicts REMI tokens, decodes them to piano MIDI, and renders WAV audio for the dashboard.

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
- 2048-token causal context
- grouped-query attention, 6 query heads and 2 KV heads
- rotary position embeddings
- RMSNorm
- SwiGLU MLP
- tied input/output embeddings
- PyTorch scaled-dot-product attention
- cached ONNX token-step inference for CPU dashboard serving

## Training Data

- Dataset: MAESTRO v3.0.0 MIDI
- Domain: solo piano performance MIDI
- Objective: next-token prediction over REMI tokens

## Metrics

Validation PPL depends on the eval protocol. Use it only for same-tokenizer, same-scorer comparisons.

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
