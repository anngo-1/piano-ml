from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pretty_midi

from .constants import (
    RANGE_NOTE_OFF,
    RANGE_NOTE_ON,
    RANGE_TIME_SHIFT,
    RANGE_VELOCITY,
    TIME_QUANTIZATION_SECONDS,
    TOKEN_END,
)


def encode_midi(path: str | Path, transpose: int | None = None) -> np.ndarray:
    midi = pretty_midi.PrettyMIDI(str(path))
    if transpose is None:
        transpose = 0

    events: list[dict[str, float | int | str]] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in sorted(instrument.notes, key=lambda n: (n.start, n.pitch, n.end)):
            pitch = int(np.clip(note.pitch + transpose, 0, 127))
            if note.velocity <= 0 or note.end <= note.start:
                continue
            events.append({"type": "note_on", "time": note.start, "pitch": pitch, "velocity": note.velocity})
            events.append({"type": "note_off", "time": note.end, "pitch": pitch})

    if not events:
        return np.array([], dtype=np.uint16)

    events.sort(key=lambda e: (float(e["time"]), 0 if e["type"] == "note_off" else 1, int(e["pitch"])))
    tokens: list[int] = []
    last_time = 0.0
    for event in events:
        delta = float(event["time"]) - last_time
        if delta > TIME_QUANTIZATION_SECONDS / 2:
            steps = max(1, int(round(delta / TIME_QUANTIZATION_SECONDS)))
            while steps > 0:
                shift = min(steps, RANGE_TIME_SHIFT)
                tokens.append(RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VELOCITY + shift - 1)
                steps -= shift

        if event["type"] == "note_on":
            tokens.append(int(event["pitch"]))
            velocity_bucket = min(int(int(event["velocity"]) * RANGE_VELOCITY / 128.0), RANGE_VELOCITY - 1)
            tokens.append(RANGE_NOTE_ON + RANGE_NOTE_OFF + velocity_bucket)
        else:
            tokens.append(RANGE_NOTE_ON + int(event["pitch"]))
        last_time = float(event["time"])

    return np.array(tokens, dtype=np.uint16)


def encode_midi_with_random_transpose(path: str | Path, semitones: int = 5) -> np.ndarray:
    return encode_midi(path, transpose=random.randint(-semitones, semitones))


def decode_midi(tokens: list[int] | np.ndarray, path: str | Path) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    active_notes: dict[int, tuple[float, int]] = {}
    current_time = 0.0
    i = 0
    token_list = [int(t) for t in tokens]

    while i < len(token_list):
        token = token_list[i]
        i += 1
        if token >= TOKEN_END:
            continue
        if RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VELOCITY <= token < TOKEN_END:
            current_time += (token - (RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VELOCITY) + 1) * TIME_QUANTIZATION_SECONDS
        elif token < RANGE_NOTE_ON:
            pitch = token
            velocity = 80
            if i < len(token_list) and RANGE_NOTE_ON + RANGE_NOTE_OFF <= token_list[i] < RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VELOCITY:
                velocity_idx = token_list[i] - (RANGE_NOTE_ON + RANGE_NOTE_OFF)
                velocity = max(1, min(127, int(round((velocity_idx / float(RANGE_VELOCITY - 1)) * 126.0)) + 1))
                i += 1
            active_notes[pitch] = (current_time, velocity)
        elif RANGE_NOTE_ON <= token < RANGE_NOTE_ON + RANGE_NOTE_OFF:
            pitch = token - RANGE_NOTE_ON
            if pitch in active_notes:
                start, velocity = active_notes.pop(pitch)
                if current_time > start:
                    instrument.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=current_time))

    for pitch, (start, velocity) in active_notes.items():
        end = max(current_time, start + 0.1)
        instrument.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))

    midi.instruments.append(instrument)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return midi
