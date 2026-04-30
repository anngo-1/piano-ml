from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pretty_midi

SAMPLE_RATE = 44100
SOUNDFONT_PATH = Path(os.getenv("PIANOGEN_SOUNDFONT", "/usr/share/sounds/sf2/FluidR3_GM.sf2"))


def _write_wav(audio: np.ndarray, path: str | Path, sample_rate: int = SAMPLE_RATE) -> Path:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.95
    pcm = np.asarray(audio * 32767.0, dtype=np.int16)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return output


def render_midi(midi_path: str | Path, output_path: str | Path) -> tuple[Path, str]:
    midi_path = Path(midi_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fluidsynth = shutil.which("fluidsynth")
    if fluidsynth and SOUNDFONT_PATH.exists():
        subprocess.run(
            [
                fluidsynth,
                "-ni",
                str(SOUNDFONT_PATH),
                str(midi_path),
                "-F",
                str(output_path),
                "-r",
                str(SAMPLE_RATE),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return output_path, "FluidSynth"

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    audio = midi.synthesize(fs=SAMPLE_RATE)
    return _write_wav(audio, output_path), "pretty_midi"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("midi", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output if args.output is not None else args.midi.with_suffix(".wav")
    path, renderer = render_midi(args.midi, output)
    print(f"wrote {path} renderer={renderer}")


if __name__ == "__main__":
    main()
