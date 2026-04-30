from __future__ import annotations

import pickle
import random
import shutil
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import TrainConfig
from .remi import encode_midi_remi


def download_maestro(config: TrainConfig) -> Path:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = config.data_dir / "maestro-v3.0.0"
    if dataset_dir.exists():
        return dataset_dir

    zip_path = config.data_dir / "maestro-v3.0.0-midi.zip"
    with requests.get(config.maestro_url, stream=True, timeout=300) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with zip_path.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading MAESTRO") as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(config.data_dir)
    zip_path.unlink(missing_ok=True)
    return dataset_dir


def load_metadata(dataset_dir: str | Path) -> pd.DataFrame:
    dataset_dir = Path(dataset_dir)
    csv_path = dataset_dir / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"MAESTRO metadata not found: {csv_path}")
    return pd.read_csv(csv_path)


def _process_one(args: tuple[str, str, bool]) -> bool:
    midi_path, output_path, augment = args
    try:
        transpose = random.randint(-5, 5) if augment else 0
        tokens = encode_midi_remi(midi_path, transpose=transpose)
        if tokens.size <= 1:
            return False
        with Path(output_path).open("wb") as f:
            pickle.dump(tokens, f)
        return True
    except Exception:
        return False


def preprocess_split(
    metadata: pd.DataFrame,
    maestro_dir: str | Path,
    split: str,
    output_dir: str | Path,
    workers: int = 8,
    overwrite: bool = False,
) -> int:
    maestro_dir = Path(maestro_dir)
    split_dir = Path(output_dir) / split
    if overwrite and split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    split_df = metadata[metadata["split"] == split]
    existing = list(split_dir.glob("*.pickle"))
    if existing and len(existing) >= max(1, int(0.9 * len(split_df))):
        return len(existing)

    jobs: list[tuple[str, str, bool]] = []
    for _, row in split_df.iterrows():
        midi_path = maestro_dir / row["midi_filename"]
        if not midi_path.exists():
            continue
        output_name = Path(row["midi_filename"]).with_suffix(".pickle").name
        jobs.append((str(midi_path), str(split_dir / output_name), split == "train"))

    if not jobs:
        return 0

    count = 0
    with Pool(processes=min(workers, len(jobs))) as pool:
        for ok in tqdm(pool.imap_unordered(_process_one, jobs), total=len(jobs), desc=f"Preprocessing {split}"):
            count += int(ok)
    return count


class MusicTokenDataset(Dataset):
    def __init__(self, data_dir: str | Path, seq_len: int, pad_token: int, files_limit: int | None = None):
        self.seq_len = seq_len
        self.pad_token = pad_token
        paths = sorted(Path(data_dir).glob("*.pickle"))
        if files_limit is not None and files_limit > 0:
            paths = random.sample(paths, min(files_limit, len(paths)))

        self.data: list[np.ndarray] = []
        for path in paths:
            try:
                with path.open("rb") as f:
                    tokens = pickle.load(f)
                tokens = np.asarray(tokens, dtype=np.int64)
                if len(tokens) > 1:
                    self.data.append(tokens)
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.data) * 5

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.data:
            src = torch.full((self.seq_len,), self.pad_token, dtype=torch.long)
            return src, src.clone()

        tokens = self.data[idx % len(self.data)]
        if len(tokens) <= self.seq_len + 1:
            seq = np.full(self.seq_len + 1, self.pad_token, dtype=np.int64)
            seq[: len(tokens)] = tokens
        else:
            start = random.randint(0, len(tokens) - self.seq_len - 1)
            seq = tokens[start : start + self.seq_len + 1]

        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)


def prepare_maestro(config: TrainConfig, overwrite: bool = False) -> None:
    dataset_dir = download_maestro(config)
    metadata = load_metadata(dataset_dir)
    preprocess_split(metadata, dataset_dir, "train", config.processed_dir, config.num_workers, overwrite)
    preprocess_split(metadata, dataset_dir, "validation", config.processed_dir, config.num_workers, overwrite)


def main() -> None:
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare_maestro(load_config(args.config), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
