from __future__ import annotations

import argparse
import tempfile
import wave
from pathlib import Path

import numpy as np
import pretty_midi

from .audio import render_with_fluidsynth
from .remi import ACOUSTIC_GRAND_PIANO_PROGRAM

SAMPLE_RATE = 44100


def force_acoustic_grand_piano(midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        notes.extend(instrument.notes)

    _, tempi = midi.get_tempo_changes()
    initial_tempo = float(tempi[0]) if len(tempi) else 120.0
    piano_midi = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
    piano = pretty_midi.Instrument(program=ACOUSTIC_GRAND_PIANO_PROGRAM, is_drum=False)
    piano.notes = sorted(notes, key=lambda note: (note.start, note.pitch, note.end))
    piano_midi.instruments.append(piano)
    return piano_midi


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
    midi = force_acoustic_grand_piano(pretty_midi.PrettyMIDI(str(midi_path)))
    with tempfile.TemporaryDirectory(prefix="piano_ml_render_") as tmpdir:
        piano_midi_path = Path(tmpdir) / "piano.mid"
        midi.write(str(piano_midi_path))
        if render_with_fluidsynth(piano_midi_path, output_path, SAMPLE_RATE):
            return output_path, "FluidSynth"

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
