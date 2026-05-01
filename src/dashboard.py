from __future__ import annotations

import inspect
import os
import secrets
import tempfile
import time
import warnings
import wave
from pathlib import Path

import gradio as gr
import numpy as np

from .audio import render_with_fluidsynth
from .config import load_config, seed_everything
from .onnx_runtime import OnnxCachedGenerator
from .remi import decode_midi_remi
from .render import force_acoustic_grand_piano

CONFIG_PATH = Path(os.getenv("PIANO_ML_CONFIG", "configs/config.json"))
CHECKPOINT_PATH = Path(os.getenv("PIANO_ML_CHECKPOINT", "models/remi-17m/best_model.pt"))
ONNX_STEP_PATH = Path(os.getenv("PIANO_ML_ONNX_STEP", "models/remi-17m/step-int8.onnx"))
ONNX_FP32_STEP_PATH = Path("models/remi-17m/step.onnx")
SAMPLE_RATE = 44100
AUDIO_TARGET_PEAK = 0.20

_MODELS: dict[str, object] = {}
_DEVICE = None
_CONFIG = None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}

PRESETS = {
    "Focused": {"temperature": 0.86, "top_k": 28, "top_p": 0.84, "repetition_penalty": 1.09, "seed": -1},
    "Balanced": {"temperature": 1.02, "top_k": 58, "top_p": 0.92, "repetition_penalty": 1.09, "seed": -1},
    "Creative": {"temperature": 1.18, "top_k": 110, "top_p": 0.96, "repetition_penalty": 1.09, "seed": -1},
}

AUDIO_RESET_HEAD = """
<style>
#piano-ml-audio.piano-ml-audio-loading::after,
.piano-ml-audio-loading::after {
  content: none !important;
  display: none !important;
}
</style>
<script>
(() => {
  const playerSelector = "#piano-ml-audio-preview audio";

  function audioSource(audio) {
    return audio.currentSrc || audio.src || "";
  }

  function resetPosition(audio, shouldPause) {
    if (!audio) {
      return;
    }
    if (shouldPause) {
      try {
        audio.pause();
      } catch (error) {
        // Some mobile browsers can throw while media metadata is changing.
      }
    }
    try {
      audio.currentTime = 0;
    } catch (error) {
      // Reset again when metadata is available.
    }
  }

  function markNewSource(audio) {
    const source = audioSource(audio);
    if (!source || source === audio.dataset.pianoMlLastSource) {
      return;
    }
    audio.dataset.pianoMlLastSource = source;
    audio.dataset.pianoMlResetOnPlay = "1";
    resetPosition(audio, false);
  }

  function bindAudio(audio) {
    if (!audio || audio.dataset.pianoMlResetBound === "1") {
      return;
    }
    audio.dataset.pianoMlResetBound = "1";
    audio.dataset.pianoMlLastSource = "";
    audio.dataset.pianoMlResetOnPlay = "0";
    audio.addEventListener("loadstart", () => markNewSource(audio));
    audio.addEventListener("loadedmetadata", () => markNewSource(audio));
    audio.addEventListener("play", () => {
      if (audio.dataset.pianoMlResetOnPlay !== "1") {
        return;
      }
      audio.dataset.pianoMlResetOnPlay = "0";
      resetPosition(audio, false);
    });
    markNewSource(audio);
  }

  function bindGeneratedAudio() {
    document.querySelectorAll(".piano-ml-audio-loading").forEach((preview) => {
      preview.classList.remove("piano-ml-audio-loading");
    });
    document.querySelectorAll(playerSelector).forEach(bindAudio);
  }

  window.pianoMlResetGeneratedAudio = () => {
    document.querySelectorAll(playerSelector).forEach((audio) => {
      resetPosition(audio, true);
      audio.dataset.pianoMlLastSource = audioSource(audio);
      audio.dataset.pianoMlResetOnPlay = "1";
    });
  };

  function start() {
    bindGeneratedAudio();
    if (document.body) {
      new MutationObserver(bindGeneratedAudio).observe(document.body, {
        childList: true,
        subtree: true,
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
</script>
"""

AUDIO_RESET_ON_GENERATE_JS = """
() => {
  if (window.pianoMlResetGeneratedAudio) {
    window.pianoMlResetGeneratedAudio();
  }
  return [null, null, ""];
}
"""


def audio_preview(path: str | Path | None = None) -> gr.Audio:
    return gr.Audio(
        value=None if path is None else str(path),
        label="Audio preview",
        type="filepath",
        interactive=False,
        autoplay=path is not None,
        elem_id="piano-ml-audio-preview",
        key=f"piano-ml-audio-preview-{secrets.token_hex(8)}",
        playback_position=0,
        preserved_by_key=[],
    )


def _onnx_path_for_mode(runtime_mode: str) -> tuple[Path, str]:
    if runtime_mode == "Quality":
        return ONNX_FP32_STEP_PATH, "onnxruntime-fp32"
    return ONNX_STEP_PATH, "onnxruntime-int8"


def load_model(runtime_mode: str) -> tuple[object, object, object, str]:
    global _CONFIG, _DEVICE
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    if _CONFIG is None:
        _CONFIG = load_config(CONFIG_PATH)
    config = _CONFIG
    requested_onnx_path, runtime = _onnx_path_for_mode(runtime_mode)
    onnx_path = requested_onnx_path if requested_onnx_path.exists() else ONNX_FP32_STEP_PATH
    if onnx_path.exists() and _env_flag("PIANO_ML_ONNX", True):
        cache_key = f"onnx:{onnx_path}"
        if cache_key not in _MODELS:
            _MODELS[cache_key] = OnnxCachedGenerator(config, onnx_path)
        model = _MODELS[cache_key]
        device = "cpu"
        _DEVICE = device
        if onnx_path != requested_onnx_path:
            runtime = "onnxruntime-fp32"
        return config, model, device, runtime

    import torch

    from .generate import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. Set PIANO_ML_CHECKPOINT or place the model file there."
        )
    cache_key = f"torch:{device.type}"
    if cache_key not in _MODELS:
        model = load_checkpoint(config, CHECKPOINT_PATH, device)
        if device.type == "cpu" and _env_flag("PIANO_ML_QUANTIZE", True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            setattr(model, "_piano_ml_quantized", True)
        model.eval()
        _MODELS[cache_key] = model
    model = _MODELS[cache_key]
    model.eval()
    _DEVICE = device
    runtime = f"{device.type}+int8" if getattr(model, "_piano_ml_quantized", False) else device.type
    return config, model, device, runtime


def write_wav(audio: np.ndarray, path: str | Path, sample_rate: int = SAMPLE_RATE) -> str:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * AUDIO_TARGET_PEAK
    pcm = np.asarray(audio * 32767.0, dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return str(path)



def render_audio(midi, midi_path: Path, wav_path: Path) -> tuple[str, str]:
    midi = force_acoustic_grand_piano(midi)
    midi.write(str(midi_path))
    if render_with_fluidsynth(
        midi_path,
        wav_path,
        SAMPLE_RATE,
        target_peak=AUDIO_TARGET_PEAK,
    ):
        return str(wav_path), "FluidSynth"
    audio = midi.synthesize(fs=SAMPLE_RATE)
    return write_wav(audio, wav_path), "pretty_midi"

def apply_preset(preset: str):
    values = PRESETS[preset]
    return (
        values["temperature"],
        values["top_k"],
        values["top_p"],
        values["repetition_penalty"],
        values["seed"],
    )


def resolve_seed(seed) -> int:
    try:
        seed_int = int(seed)
    except (TypeError, ValueError):
        seed_int = -1
    if seed_int < 0:
        return secrets.randbelow(2**31 - 1)
    return seed_int


def generate(
    runtime_mode: str,
    preset: str,
    length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    seed: int,
):
    started = time.perf_counter()
    seed_int = resolve_seed(seed)
    _, tokens, runtime = generate_tokens_for_dashboard(
        runtime_mode,
        int(length),
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        seed_int,
    )
    tmpdir = Path(tempfile.mkdtemp(prefix="piano_ml_"))
    midi_path = tmpdir / f"piano_ml_{preset.lower()}_{seed_int}.mid"
    wav_path = tmpdir / f"piano_ml_{preset.lower()}_{seed_int}.wav"
    midi = decode_midi_remi(tokens, midi_path)
    _, renderer = render_audio(midi, midi_path, wav_path)
    note_count = sum(len(inst.notes) for inst in midi.instruments)
    elapsed = time.perf_counter() - started
    summary = (
        f"Generated {len(tokens)} tokens, {note_count} notes, "
        f"audio duration {midi.get_end_time():.2f}s on {runtime} in {elapsed:.1f}s. "
        f"Audio renderer: {renderer}. Seed: {seed_int}."
    )
    return str(wav_path), str(wav_path), summary


def clear_outputs():
    return audio_preview(), None, ""


def generate_tokens_for_dashboard(
    runtime_mode: str,
    length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    seed_int: int,
    progress_callback=None,
):
    config, model, device, runtime = load_model(runtime_mode)
    tokens = sample_tokens(
        config,
        model,
        device,
        length,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        seed_int,
        progress_callback=progress_callback,
    )
    return config, tokens, runtime


def sample_tokens(
    config,
    model,
    device,
    length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    seed_int: int,
    progress_callback=None,
):
    if isinstance(model, OnnxCachedGenerator):
        import random

        random.seed(seed_int)
        np.random.seed(seed_int)
        return model.generate(
            length=int(length),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            prompt=[0],
            progress_callback=progress_callback,
        )

    seed_everything(seed_int)
    from .sample import generate_tokens

    return generate_tokens(
        model,
        prompt=[0],
        length=int(length),
        seq_len=config.seq_len,
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        device=device,
        pad_token=config.token_pad,
        repetition_penalty=float(repetition_penalty),
        constrained=True,
        progress_callback=progress_callback,
    )


def generate_stream(
    runtime_mode: str,
    preset: str,
    length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    progress=gr.Progress(),
):
    started = time.perf_counter()
    seed_int = resolve_seed(seed)
    target_length = int(length)
    yield gr.skip(), gr.skip(), f"Loading {runtime_mode.lower()} model..."

    load_started = time.perf_counter()
    config, model, device, runtime = load_model(runtime_mode)
    load_elapsed = time.perf_counter() - load_started

    yield gr.skip(), gr.skip(), f"Generating {target_length} tokens on {runtime}..."

    def report_progress(current: int, total: int) -> None:
        progress(current / max(total, 1), desc=f"Generating tokens: {current} / {total}")

    token_started = time.perf_counter()
    tokens = sample_tokens(
        config,
        model,
        device,
        target_length,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        seed_int,
        progress_callback=report_progress,
    )
    token_elapsed = time.perf_counter() - token_started
    tokens_per_second = len(tokens) / max(token_elapsed, 1e-6)

    yield gr.skip(), gr.skip(), (
        f"Generated {len(tokens)} / {target_length} tokens "
        f"({tokens_per_second:.1f} tok/s). Rendering audio..."
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="piano_ml_"))
    midi_path = tmpdir / f"piano_ml_{preset.lower()}_{seed_int}.mid"
    wav_path = tmpdir / f"piano_ml_{preset.lower()}_{seed_int}.wav"
    midi = decode_midi_remi(tokens, midi_path)
    render_started = time.perf_counter()
    audio_path, renderer = render_audio(midi, midi_path, wav_path)
    render_elapsed = time.perf_counter() - render_started
    note_count = sum(len(inst.notes) for inst in midi.instruments)
    elapsed = time.perf_counter() - started
    final_progress = (
        f"Done: {len(tokens)} / {target_length} tokens, {note_count} notes, "
        f"{tokens_per_second:.1f} tok/s, {runtime}, load {load_elapsed:.1f}s, "
        f"render {render_elapsed:.1f}s, "
        f"total {elapsed:.1f}s, {renderer}, seed {seed_int}."
    )
    yield audio_preview(audio_path), str(wav_path), final_progress


with gr.Blocks(title="piano-ml") as demo:
    gr.Markdown(
        "# piano-ml\n"
        "Sample short piano pieces from a 17M-parameter decoder-only model trained on REMI-style event tokens. "
        "Audio is rendered as acoustic grand piano with the bundled GeneralUser GS SoundFont. "
        "Code: [GitHub](https://github.com/anngo-1/piano-ml)."
    )
    with gr.Row():
        runtime_mode = gr.Radio(["Fast", "Quality"], value="Fast", label="Runtime")
        preset = gr.Dropdown(list(PRESETS), value="Balanced", label="Preset")
        length = gr.Slider(256, 2048, value=1024, step=128, label="Length (tokens)")
        seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
    gr.Markdown(
        "Fast uses the quantized 17M ONNX model. Quality uses the FP32 17M ONNX model. "
        "Length is measured in REMI tokens, not notes."
    )
    with gr.Row():
        temperature = gr.Slider(0.7, 1.35, value=1.02, step=0.01, label="Temperature")
        top_k = gr.Slider(0, 160, value=58, step=1, label="Top-k")
        top_p = gr.Slider(0.75, 1.0, value=0.92, step=0.01, label="Top-p")
        repetition_penalty = gr.Slider(1.0, 1.2, value=1.09, step=0.01, label="Repetition penalty")
    preset.change(apply_preset, inputs=preset, outputs=[temperature, top_k, top_p, repetition_penalty, seed])
    button = gr.Button("Generate", variant="primary")
    progress = gr.Textbox(label="Progress", interactive=False)
    audio = audio_preview()
    wav_file = gr.File(label="Download WAV")
    button.click(
        clear_outputs,
        outputs=[audio, wav_file, progress],
        queue=False,
        js=AUDIO_RESET_ON_GENERATE_JS,
    ).then(
        generate_stream,
        inputs=[runtime_mode, preset, length, temperature, top_k, top_p, repetition_penalty, seed],
        outputs=[audio, wav_file, progress],
        stream_every=0.25,
    )

def launch() -> None:
    kwargs = {
        "server_name": os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        "server_port": int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    }
    launch_params = inspect.signature(demo.launch).parameters
    if "head" in launch_params:
        kwargs["head"] = AUDIO_RESET_HEAD
    if "ssr_mode" in launch_params:
        kwargs["ssr_mode"] = False
    demo.launch(**kwargs)


if __name__ == "__main__":
    launch()
