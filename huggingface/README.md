# Hugging Face Packaging

This directory is a staging area for publishing the browser demo to Hugging Face Spaces.

The Hugging Face Space should be inference-only. Do not copy training data, run logs, generated experiment outputs, or evaluation scripts into the Space.

Recommended setup:

1. Publish the main training/code repo to GitHub.
2. Create a Hugging Face Gradio Space.
3. Copy the contents of `huggingface/space/` into the Space repo.
4. Add the checkpoint with Git LFS at:

```text
models/remi-modern-2048-ft/best_model.pt
```

The Space UI serves audio only: browser preview plus downloadable WAV. MIDI is generated internally for rendering and is not exposed in the UI.

The checkpoint is about 67 MB. Approximate download time:

- 100 Mbps: 6 seconds
- 25 Mbps: 22 seconds
- 10 Mbps: 55 seconds

Dependency/startup time on Spaces is separate and can take a few minutes on cold start.
