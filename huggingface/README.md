# Hugging Face Packaging

This directory stages the browser demo for Hugging Face Spaces. The Space should stay inference-only: no MAESTRO data, run logs, generated outputs, or evaluation scripts.

Recommended setup:

1. Publish the main training/code repo to GitHub.
2. Create a Hugging Face Gradio Space.
3. Copy the contents of `huggingface/space/` into the Space repo.
4. Add the checkpoint with Git LFS at:

```text
models/remi-modern-2048-ft/best_model.pt
```

The Space UI serves audio only: browser preview plus downloadable WAV. `Fast` uses quantized cached ONNX and may sound slightly lower quality; `Quality` uses the FP32 cached ONNX model.

The checkpoint is about 67 MB. Approximate download time:

- 100 Mbps: 6 seconds
- 25 Mbps: 22 seconds
- 10 Mbps: 55 seconds

Dependency/startup time on Spaces is separate and can take a few minutes on cold start.
