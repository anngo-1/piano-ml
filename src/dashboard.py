from __future__ import annotations

import inspect
import os
import shutil
import subprocess
import tempfile
import time
import warnings
import wave
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from .config import load_config, seed_everything
from .generate import load_checkpoint
from .remi import decode_midi_remi
from .sample import generate_tokens

CONFIG_PATH = Path(os.getenv("PIANOGEN_CONFIG", "configs/config.json"))
CHECKPOINT_PATH = Path(os.getenv("PIANOGEN_CHECKPOINT", "models/remi-modern-2048-ft/best_model.pt"))
SAMPLE_RATE = 44100
SOUNDFONT_PATH = Path(os.getenv("PIANOGEN_SOUNDFONT", "/usr/share/sounds/sf2/FluidR3_GM.sf2"))

_MODEL: torch.nn.Module | None = None
_DEVICE: torch.device | None = None
_CONFIG = None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}

PRESETS = {
    "Focused": {"temperature": 0.86, "top_k": 28, "top_p": 0.84, "repetition_penalty": 1.06, "seed": 31},
    "Balanced": {"temperature": 1.02, "top_k": 58, "top_p": 0.92, "repetition_penalty": 1.07, "seed": 47},
    "Creative": {"temperature": 1.18, "top_k": 110, "top_p": 0.96, "repetition_penalty": 1.10, "seed": 83},
}


def load_model() -> tuple[object, torch.nn.Module, torch.device]:
    global _CONFIG, _MODEL, _DEVICE
    if _MODEL is not None and _CONFIG is not None and _DEVICE is not None:
        return _CONFIG, _MODEL, _DEVICE
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. Set PIANOGEN_CHECKPOINT or place the model file there."
        )
    config = load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(config, CHECKPOINT_PATH, device)
    if device.type == "cpu" and _env_flag("PIANOGEN_QUANTIZE", True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        setattr(model, "_pianogen_quantized", True)
    model.eval()
    _CONFIG, _MODEL, _DEVICE = config, model, device
    return config, model, device


def write_wav(audio: np.ndarray, path: str | Path, sample_rate: int = SAMPLE_RATE) -> str:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.95
    pcm = np.asarray(audio * 32767.0, dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return str(path)



def render_audio(midi, midi_path: Path, wav_path: Path) -> str:
    fluidsynth = shutil.which("fluidsynth")
    if fluidsynth and SOUNDFONT_PATH.exists():
        subprocess.run(
            [
                fluidsynth,
                "-ni",
                str(SOUNDFONT_PATH),
                str(midi_path),
                "-F",
                str(wav_path),
                "-r",
                str(SAMPLE_RATE),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return str(wav_path)
    audio = midi.synthesize(fs=SAMPLE_RATE)
    return write_wav(audio, wav_path)

def apply_preset(preset: str):
    values = PRESETS[preset]
    return (
        values["temperature"],
        values["top_k"],
        values["top_p"],
        values["repetition_penalty"],
        values["seed"],
    )


def generate(
    preset: str,
    length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    seed: int,
):
    started = time.perf_counter()
    config, model, device = load_model()
    seed_everything(int(seed))
    tokens = generate_tokens(
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
        tokenizer="remi",
    )
    tmpdir = Path(tempfile.mkdtemp(prefix="pianogen_"))
    midi_path = tmpdir / f"pianogen_{preset.lower()}_{seed}.mid"
    wav_path = tmpdir / f"pianogen_{preset.lower()}_{seed}.wav"
    midi = decode_midi_remi(tokens, midi_path)
    render_audio(midi, midi_path, wav_path)
    note_count = sum(len(inst.notes) for inst in midi.instruments)
    renderer = "FluidSynth" if shutil.which("fluidsynth") and SOUNDFONT_PATH.exists() else "pretty_midi"
    elapsed = time.perf_counter() - started
    runtime = f"{device.type}+int8" if getattr(model, "_pianogen_quantized", False) else device.type
    summary = (
        f"Generated {len(tokens)} tokens, {note_count} notes, "
        f"audio duration {midi.get_end_time():.2f}s on {runtime} in {elapsed:.1f}s. "
        f"Audio renderer: {renderer}."
    )
    return str(wav_path), str(wav_path), summary


with gr.Blocks(title="pianogen") as demo:
    gr.Markdown(
        "# pianogen\n"
        "Unconditional piano audio generation with a REMI-token Transformer. "
        "The app renders audio on the server and returns a preview plus a downloadable WAV."
    )
    with gr.Row():
        preset = gr.Dropdown(list(PRESETS), value="Balanced", label="Preset")
        length = gr.Slider(256, 3072, value=1024, step=128, label="Generation tokens")
        seed = gr.Number(value=47, precision=0, label="Seed")
    with gr.Row():
        temperature = gr.Slider(0.7, 1.35, value=1.02, step=0.01, label="Temperature")
        top_k = gr.Slider(0, 160, value=58, step=1, label="Top-k")
        top_p = gr.Slider(0.75, 1.0, value=0.92, step=0.01, label="Top-p")
        repetition_penalty = gr.Slider(1.0, 1.2, value=1.07, step=0.01, label="Repetition penalty")
    preset.change(apply_preset, inputs=preset, outputs=[temperature, top_k, top_p, repetition_penalty, seed])
    button = gr.Button("Generate", variant="primary")
    audio = gr.Audio(label="Audio preview", type="filepath")
    wav_file = gr.File(label="Download WAV")
    summary = gr.Textbox(label="Summary")
    button.click(
        generate,
        inputs=[preset, length, temperature, top_k, top_p, repetition_penalty, seed],
        outputs=[audio, wav_file, summary],
    )

def launch() -> None:
    kwargs = {
        "server_name": os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        "server_port": int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    }
    if "ssr_mode" in inspect.signature(demo.launch).parameters:
        kwargs["ssr_mode"] = False
    demo.launch(**kwargs)


if __name__ == "__main__":
    launch()
