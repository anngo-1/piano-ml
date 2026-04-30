from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .config import TrainConfig
from .remi import REMI_BAR, REMI_DUR_START, REMI_PITCH_START, REMI_POS_START, REMI_TOKEN_END, REMI_VEL_START

_REMI_ALLOWED = {
    "structure": np.asarray([REMI_BAR, *range(REMI_POS_START, REMI_DUR_START)], dtype=np.int64),
    "duration": np.arange(REMI_DUR_START, REMI_VEL_START, dtype=np.int64),
    "velocity": np.arange(REMI_VEL_START, REMI_TOKEN_END, dtype=np.int64),
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _observe_remi(expect: str, token: int) -> str:
    if token == REMI_TOKEN_END:
        return "structure"
    if expect == "duration" and REMI_DUR_START <= token < REMI_VEL_START:
        return "velocity"
    if expect == "velocity" and REMI_VEL_START <= token < REMI_TOKEN_END:
        return "structure"
    if REMI_PITCH_START <= token < REMI_DUR_START:
        return "duration"
    if token == REMI_BAR or REMI_POS_START <= token < REMI_PITCH_START:
        return "structure"
    return expect


def _sample_next(
    logits: np.ndarray,
    allowed: np.ndarray,
    pad_token: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    recent_tokens: list[int],
) -> int | None:
    logits = logits.astype(np.float32, copy=True)
    logits[pad_token] = -np.inf
    masked = np.full_like(logits, -np.inf)
    masked[allowed] = logits[allowed]
    logits = masked

    if repetition_penalty > 1.0 and recent_tokens:
        for token in set(recent_tokens):
            if 0 <= token < logits.size and np.isfinite(logits[token]):
                logits[token] = logits[token] / repetition_penalty if logits[token] > 0 else logits[token] * repetition_penalty

    logits = logits / max(float(temperature), 1e-5)
    finite = np.flatnonzero(np.isfinite(logits))
    if finite.size == 0:
        return None

    if top_k > 0 and finite.size > top_k:
        keep = finite[np.argpartition(logits[finite], -top_k)[-top_k:]]
        top_mask = np.full_like(logits, -np.inf)
        top_mask[keep] = logits[keep]
        logits = top_mask
        finite = keep

    probs = np.exp(logits[finite] - np.max(logits[finite]))
    probs = probs / probs.sum()

    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        keep_count = max(1, int(np.searchsorted(cumulative, top_p, side="right") + 1))
        keep_order = order[:keep_count]
        finite = finite[keep_order]
        probs = probs[keep_order]
        probs = probs / probs.sum()

    return int(np.random.choice(finite, p=probs))


class OnnxCachedGenerator:
    def __init__(self, config: TrainConfig, model_path: str | Path):
        import onnxruntime as ort

        self.config = config
        self.model_path = Path(model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = _env_int("PIANOGEN_ONNX_THREADS", 4)
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.head_dim = config.model.d_model // config.model.num_heads
        self.num_kv_heads = config.model.num_kv_heads or config.model.num_heads
        self.num_cache_tensors = config.model.num_layers * 2

    def generate(
        self,
        length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        prompt: list[int] | None = None,
    ) -> list[int]:
        tokens = list(prompt or [0])
        expect = "structure"
        for token in tokens:
            expect = _observe_remi(expect, int(token))

        caches = [
            np.empty((1, self.num_kv_heads, 0, self.head_dim), dtype=np.float32)
            for _ in range(self.num_cache_tensors)
        ]
        while len(tokens) < length:
            feeds = {
                self.input_names[0]: np.asarray([[tokens[-1]]], dtype=np.int64),
                self.input_names[1]: np.asarray(len(tokens) - 1, dtype=np.int64),
            }
            feeds.update({name: cache for name, cache in zip(self.input_names[2:], caches, strict=True)})
            outputs = self.session.run(None, feeds)
            logits = outputs[0][0, -1]
            caches = list(outputs[1:])
            next_token = _sample_next(
                logits,
                _REMI_ALLOWED[expect],
                self.config.token_pad,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                tokens[-128:],
            )
            if next_token is None:
                break
            tokens.append(next_token)
            expect = _observe_remi(expect, next_token)
        return tokens
