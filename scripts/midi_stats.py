from __future__ import annotations

import argparse
from pathlib import Path

import pretty_midi


def summarize(path: Path) -> str:
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = [note for instrument in midi.instruments for note in instrument.notes]
    duration = midi.get_end_time()
    notes_per_second = len(notes) / max(duration, 1e-6)
    return f"{path}\tduration={duration:.2f}s\tnotes={len(notes)}\tnotes_per_second={notes_per_second:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize generated MIDI files.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        print(summarize(path))


if __name__ == "__main__":
    main()
