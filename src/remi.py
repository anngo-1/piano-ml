from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi

REMI_BAR = 0
REMI_POS_START = 1
REMI_POSITIONS_PER_BAR = 48
REMI_PITCH_MIN = 21
REMI_PITCH_MAX = 108
REMI_PITCH_COUNT = REMI_PITCH_MAX - REMI_PITCH_MIN + 1
REMI_PITCH_START = REMI_POS_START + REMI_POSITIONS_PER_BAR
REMI_DUR_START = REMI_PITCH_START + REMI_PITCH_COUNT
REMI_DURATION_BINS = 96
REMI_VEL_START = REMI_DUR_START + REMI_DURATION_BINS
REMI_VELOCITY_BINS = 32
REMI_TOKEN_END = REMI_VEL_START + REMI_VELOCITY_BINS
REMI_TOKEN_PAD = REMI_TOKEN_END + 1
REMI_VOCAB_SIZE = REMI_TOKEN_PAD + 1


def remi_vocab_size() -> int:
    return REMI_VOCAB_SIZE


def remi_token_pad() -> int:
    return REMI_TOKEN_PAD


def remi_token_end() -> int:
    return REMI_TOKEN_END


def remi_token_label(token: int) -> str:
    token = int(token)
    if token == REMI_BAR:
        return "BAR"
    if REMI_POS_START <= token < REMI_PITCH_START:
        return f"POS_{token - REMI_POS_START}"
    if REMI_PITCH_START <= token < REMI_DUR_START:
        return f"PITCH_{REMI_PITCH_MIN + token - REMI_PITCH_START}"
    if REMI_DUR_START <= token < REMI_VEL_START:
        return f"DUR_{token - REMI_DUR_START + 1}"
    if REMI_VEL_START <= token < REMI_TOKEN_END:
        velocity_idx = token - REMI_VEL_START
        velocity = max(1, min(127, int(round((velocity_idx / float(REMI_VELOCITY_BINS - 1)) * 126.0)) + 1))
        return f"VEL_{velocity}"
    if token == REMI_TOKEN_END:
        return "END"
    if token == REMI_TOKEN_PAD:
        return "PAD"
    return f"TOKEN_{token}"


def _estimate_bpm(midi: pretty_midi.PrettyMIDI) -> float:
    _, tempi = midi.get_tempo_changes()
    if len(tempi):
        bpm = float(np.median(tempi))
    else:
        bpm = float(midi.estimate_tempo())
    if not np.isfinite(bpm) or bpm <= 20 or bpm >= 260:
        return 120.0
    return bpm


def encode_midi_remi(path: str | Path, transpose: int = 0) -> np.ndarray:
    midi = pretty_midi.PrettyMIDI(str(path))
    bpm = _estimate_bpm(midi)
    seconds_per_beat = 60.0 / bpm
    seconds_per_bar = seconds_per_beat * 4.0
    seconds_per_pos = seconds_per_bar / REMI_POSITIONS_PER_BAR

    notes: list[tuple[int, int, int, int]] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            if note.velocity <= 0 or note.end <= note.start:
                continue
            pitch = int(note.pitch + transpose)
            if pitch < REMI_PITCH_MIN or pitch > REMI_PITCH_MAX:
                continue
            start_step = max(0, int(round(note.start / seconds_per_pos)))
            duration = max(1, int(round((note.end - note.start) / seconds_per_pos)))
            duration = min(duration, REMI_DURATION_BINS)
            velocity = min(int(note.velocity * REMI_VELOCITY_BINS / 128.0), REMI_VELOCITY_BINS - 1)
            notes.append((start_step, pitch, duration, velocity))

    if len(notes) < 16:
        return np.array([], dtype=np.uint16)

    notes.sort(key=lambda item: (item[0], item[1], item[2]))
    tokens: list[int] = []
    current_bar = -1
    current_pos = -1
    for start_step, pitch, duration, velocity in notes:
        bar = start_step // REMI_POSITIONS_PER_BAR
        pos = start_step % REMI_POSITIONS_PER_BAR
        while current_bar < bar:
            tokens.append(REMI_BAR)
            current_bar += 1
            current_pos = -1
        if pos != current_pos:
            tokens.append(REMI_POS_START + pos)
            current_pos = pos
        tokens.extend(
            [
                REMI_PITCH_START + (pitch - REMI_PITCH_MIN),
                REMI_DUR_START + duration - 1,
                REMI_VEL_START + velocity,
            ]
        )

    tokens.append(REMI_TOKEN_END)
    return np.array(tokens, dtype=np.uint16)


def decode_midi_remi(tokens: list[int] | np.ndarray, path: str | Path, bpm: float = 120.0) -> pretty_midi.PrettyMIDI:
    seconds_per_bar = (60.0 / bpm) * 4.0
    seconds_per_pos = seconds_per_bar / REMI_POSITIONS_PER_BAR
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=0)
    token_list = [int(t) for t in tokens]
    bar = -1
    pos = 0
    i = 0

    while i < len(token_list):
        token = token_list[i]
        i += 1
        if token == REMI_TOKEN_END or token == REMI_TOKEN_PAD:
            continue
        if token == REMI_BAR:
            bar += 1
            pos = 0
            continue
        if REMI_POS_START <= token < REMI_PITCH_START:
            pos = token - REMI_POS_START
            continue
        if REMI_PITCH_START <= token < REMI_DUR_START:
            if i + 1 >= len(token_list):
                break
            dur_token = token_list[i]
            vel_token = token_list[i + 1]
            i += 2
            if not (REMI_DUR_START <= dur_token < REMI_VEL_START):
                continue
            if not (REMI_VEL_START <= vel_token < REMI_TOKEN_END):
                continue
            pitch = REMI_PITCH_MIN + token - REMI_PITCH_START
            duration = dur_token - REMI_DUR_START + 1
            velocity_idx = vel_token - REMI_VEL_START
            velocity = max(1, min(127, int(round((velocity_idx / float(REMI_VELOCITY_BINS - 1)) * 126.0)) + 1))
            current_bar = max(bar, 0)
            start = (current_bar * REMI_POSITIONS_PER_BAR + pos) * seconds_per_pos
            end = start + duration * seconds_per_pos
            if end > start:
                instrument.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))

    instrument.notes.sort(key=lambda n: (n.start, n.pitch, n.end))
    midi.instruments.append(instrument)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return midi
