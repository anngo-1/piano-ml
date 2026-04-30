from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from .remi import REMI_TOKEN_PAD, REMI_VOCAB_SIZE, decode_midi_remi


def _merge_pair(sequence: list[int], pair: tuple[int, int], new_token: int) -> list[int]:
    merged: list[int] = []
    i = 0
    while i < len(sequence):
        if i + 1 < len(sequence) and sequence[i] == pair[0] and sequence[i + 1] == pair[1]:
            merged.append(new_token)
            i += 2
        else:
            merged.append(sequence[i])
            i += 1
    return merged


def _count_pairs(sequences: Iterable[list[int]]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for sequence in sequences:
        counts.update(zip(sequence, sequence[1:]))
    return counts


def learn_bpe(sequences: list[list[int]], target_vocab_size: int, min_pair_count: int = 8) -> dict:
    if target_vocab_size <= REMI_VOCAB_SIZE:
        raise ValueError("target_vocab_size must be larger than REMI_VOCAB_SIZE")

    merges: list[tuple[int, int]] = []
    next_token = REMI_VOCAB_SIZE
    while next_token < target_vocab_size:
        counts = _count_pairs(sequences)
        if not counts:
            break
        pair, count = counts.most_common(1)[0]
        if count < min_pair_count:
            break
        sequences = [_merge_pair(sequence, pair, next_token) for sequence in sequences]
        merges.append(pair)
        next_token += 1
        if len(merges) % 100 == 0:
            print(f"learned_merges={len(merges)} vocab={next_token} last_pair_count={count}", flush=True)

    return {
        "base_vocab_size": REMI_VOCAB_SIZE,
        "pad_token": REMI_TOKEN_PAD,
        "vocab_size": next_token,
        "merges": [[int(a), int(b)] for a, b in merges],
    }


def encode_bpe_tokens(tokens: list[int] | np.ndarray, merges: list[list[int]]) -> np.ndarray:
    sequence = [int(t) for t in tokens]
    for offset, pair_list in enumerate(merges):
        pair = (int(pair_list[0]), int(pair_list[1]))
        sequence = _merge_pair(sequence, pair, REMI_VOCAB_SIZE + offset)
    return np.asarray(sequence, dtype=np.uint16 if REMI_VOCAB_SIZE + len(merges) <= 65535 else np.uint32)


def load_bpe(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def save_bpe(payload: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def token_expansions(payload: dict) -> dict[int, list[int]]:
    expansions: dict[int, list[int]] = {idx: [idx] for idx in range(int(payload["base_vocab_size"]))}
    for offset, pair in enumerate(payload["merges"]):
        token = REMI_VOCAB_SIZE + offset
        left, right = int(pair[0]), int(pair[1])
        expansions[token] = expansions[left] + expansions[right]
    return expansions


def decode_bpe_tokens(tokens: list[int] | np.ndarray, payload: dict) -> list[int]:
    expansions = token_expansions(payload)
    remi_tokens: list[int] = []
    for token in tokens:
        remi_tokens.extend(expansions.get(int(token), []))
    return remi_tokens


def decode_midi_remi_bpe(tokens: list[int] | np.ndarray, path: str | Path, bpe_path: str | Path):
    payload = load_bpe(bpe_path)
    return decode_midi_remi(decode_bpe_tokens(tokens, payload), path)


def load_sequences(data_dir: str | Path) -> list[list[int]]:
    sequences: list[list[int]] = []
    for path in sorted(Path(data_dir).glob("*.pickle")):
        with path.open("rb") as f:
            tokens = pickle.load(f)
        sequence = [int(t) for t in np.asarray(tokens).tolist()]
        if len(sequence) > 1:
            sequences.append(sequence)
    return sequences
