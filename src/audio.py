from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_SOUNDFONT_PATH = PROJECT_ROOT / "soundfonts" / "GeneralUser-GS.sf2"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def find_soundfont() -> Path | None:
    if BUNDLED_SOUNDFONT_PATH.exists():
        return BUNDLED_SOUNDFONT_PATH

    configured = os.getenv("PIANO_ML_SOUNDFONT")
    if configured:
        path = Path(configured).expanduser()
        return path if path.exists() else None

    candidates = [
        PROJECT_ROOT / "soundfonts" / "GeneralUser-GS.sf3",
        Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
        Path("/usr/share/soundfonts/FluidR3_GM.sf2"),
        Path("/Library/Audio/Sounds/Banks/FluidR3_GM.sf2"),
        Path("~/Library/Audio/Sounds/Banks/FluidR3_GM.sf2").expanduser(),
    ]
    for path in candidates:
        if path.exists():
            return path

    for root in (Path("/opt/homebrew"), Path("/usr/local")):
        for pattern in ("**/FluidR3_GM.sf2", "**/GeneralUser-GS.sf2", "**/GeneralUser-GS.sf3"):
            match = next(root.glob(pattern), None)
            if match is not None:
                return match
    return None


def normalize_wav(path: str | Path, target_peak: float = 0.98) -> None:
    path = Path(path)
    data = path.read_bytes()
    if len(data) <= 44:
        return

    import wave

    with wave.open(str(path), "rb") as source:
        params = source.getparams()
        frames = source.readframes(params.nframes)
    if params.sampwidth != 2 or not frames:
        return

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0:
        return

    scale = min((32767.0 * target_peak) / peak, 8.0)
    normalized = np.clip(audio * scale, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setparams(params)
        output.writeframes(normalized.tobytes())


def render_with_fluidsynth(
    midi_path: str | Path,
    wav_path: str | Path,
    sample_rate: int,
    target_peak: float = 0.98,
) -> bool:
    fluidsynth = shutil.which("fluidsynth")
    if not fluidsynth:
        return False
    soundfont_path = find_soundfont()
    if soundfont_path is None:
        return False

    output = Path(wav_path)
    output.unlink(missing_ok=True)
    timeout = _env_int("PIANO_ML_FLUIDSYNTH_TIMEOUT", 300)
    try:
        subprocess.run(
            [
                fluidsynth,
                "-ni",
                "-q",
                "-T",
                "wav",
                "-F",
                str(output),
                "-r",
                str(sample_rate),
                str(soundfont_path),
                str(midi_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        output.unlink(missing_ok=True)
        return False
    if not output.exists() or output.stat().st_size <= 0:
        return False
    normalize_wav(output, target_peak)
    return True
