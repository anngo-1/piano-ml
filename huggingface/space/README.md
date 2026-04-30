---
title: pianogen
emoji: 🎹
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
---

# pianogen

Unconditional piano audio generation with a REMI-token Transformer.

The app runs model inference server-side and returns an audio preview plus a downloadable WAV. MIDI is generated internally only.

Expected checkpoint path:

```text
models/remi-modern-2048-ft/best_model.pt
```

Expected config path:

```text
configs/config.json
```
