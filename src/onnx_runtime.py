from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import TrainConfig
from .sample import RemiGrammarState, filter_logits


class OnnxCachedGenerator:
    def __init__(self, config: TrainConfig, model_path: str | Path):
        import onnxruntime as ort

        self.config = config
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
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
        grammar = RemiGrammarState()
        for token in tokens:
            grammar.observe(int(token))

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
            logits = torch.from_numpy(outputs[0][0, -1])
            caches = list(outputs[1:])
            logits = filter_logits(
                logits,
                temperature,
                top_k,
                top_p,
                self.config.token_pad,
                repetition_penalty=repetition_penalty,
                recent_tokens=tokens[-128:],
                allowed_tokens=grammar.allowed(torch.device("cpu")),
            )
            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or probs.sum() <= 0:
                break
            next_token = int(torch.multinomial(probs, 1).item())
            tokens.append(next_token)
            grammar.observe(next_token)
        return tokens
