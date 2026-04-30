from __future__ import annotations

import torch
import torch.nn.functional as F

from .constants import RANGE_NOTE_OFF, RANGE_NOTE_ON, RANGE_TIME_SHIFT, RANGE_VELOCITY, TOKEN_END, TOKEN_PAD
from .remi import REMI_BAR, REMI_DUR_START, REMI_PITCH_START, REMI_POS_START, REMI_TOKEN_END, REMI_VEL_START


def filter_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    pad_token: int = TOKEN_PAD,
    repetition_penalty: float = 1.0,
    recent_tokens: list[int] | None = None,
    allowed_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = logits.float().clone()
    logits[pad_token] = -float("inf")

    if allowed_tokens is not None:
        mask = torch.full_like(logits, -float("inf"))
        mask[allowed_tokens] = logits[allowed_tokens]
        logits = mask

    if repetition_penalty > 1.0 and recent_tokens:
        for token in set(recent_tokens):
            if 0 <= token < logits.numel() and torch.isfinite(logits[token]):
                logits[token] = logits[token] / repetition_penalty if logits[token] > 0 else logits[token] * repetition_penalty

    logits = logits / max(temperature, 1e-5)

    if top_k > 0:
        finite_count = int(torch.isfinite(logits).sum().item())
        k = min(top_k, finite_count)
        if k > 0:
            values, _ = torch.topk(logits, k)
            logits[logits < values[-1]] = -float("inf")

    if top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits[sorted_idx[remove]] = -float("inf")

    return logits


class TokenGrammarState:
    def __init__(self, min_pitch: int = 21, max_pitch: int = 108, max_active_notes: int = 16):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.max_active_notes = max_active_notes
        self.active: set[int] = set()
        self.expect_velocity = False

    def observe(self, token: int) -> None:
        if self.expect_velocity:
            self.expect_velocity = False
            return
        if self.min_pitch <= token <= self.max_pitch:
            self.active.add(token)
            self.expect_velocity = True
        elif RANGE_NOTE_ON <= token < RANGE_NOTE_ON + RANGE_NOTE_OFF:
            self.active.discard(token - RANGE_NOTE_ON)

    def allowed(self, device: torch.device) -> torch.Tensor:
        if self.expect_velocity:
            return torch.arange(
                RANGE_NOTE_ON + RANGE_NOTE_OFF,
                RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VELOCITY,
                device=device,
                dtype=torch.long,
            )

        allowed: list[int] = []
        allowed.extend(range(RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VELOCITY, TOKEN_END))
        allowed.extend(RANGE_NOTE_ON + p for p in sorted(self.active))
        if len(self.active) < self.max_active_notes:
            allowed.extend(range(self.min_pitch, self.max_pitch + 1))
        return torch.tensor(sorted(set(allowed)), device=device, dtype=torch.long)


class RemiGrammarState:
    def __init__(self):
        self.expect = "structure"

    def observe(self, token: int) -> None:
        if token == REMI_TOKEN_END:
            self.expect = "structure"
        elif self.expect == "duration" and REMI_DUR_START <= token < REMI_VEL_START:
            self.expect = "velocity"
        elif self.expect == "velocity" and REMI_VEL_START <= token < REMI_TOKEN_END:
            self.expect = "structure"
        elif REMI_PITCH_START <= token < REMI_DUR_START:
            self.expect = "duration"
        elif token == REMI_BAR or REMI_POS_START <= token < REMI_PITCH_START:
            self.expect = "structure"

    def allowed(self, device: torch.device) -> torch.Tensor:
        if self.expect == "duration":
            return torch.arange(REMI_DUR_START, REMI_VEL_START, device=device, dtype=torch.long)
        if self.expect == "velocity":
            return torch.arange(REMI_VEL_START, REMI_TOKEN_END, device=device, dtype=torch.long)
        allowed = [REMI_BAR]
        allowed.extend(range(REMI_POS_START, REMI_DUR_START))
        return torch.tensor(allowed, device=device, dtype=torch.long)


@torch.no_grad()
def generate_tokens(
    model: torch.nn.Module,
    prompt: list[int],
    length: int,
    seq_len: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
    pad_token: int = TOKEN_PAD,
    repetition_penalty: float = 1.0,
    constrained: bool = True,
    min_pitch: int = 21,
    max_pitch: int = 108,
    max_active_notes: int = 16,
    tokenizer: str = "event",
) -> list[int]:
    model.eval()
    tokens = list(prompt)
    grammar = RemiGrammarState() if tokenizer == "remi" else TokenGrammarState(min_pitch=min_pitch, max_pitch=max_pitch, max_active_notes=max_active_notes)
    for token in tokens:
        grammar.observe(int(token))

    if hasattr(model, "forward_cached"):
        return generate_tokens_cached(
            model,
            tokens,
            length,
            temperature,
            top_k,
            top_p,
            device,
            pad_token=pad_token,
            repetition_penalty=repetition_penalty,
            constrained=constrained,
            grammar=grammar,
        )

    while len(tokens) < length:
        context = tokens[-seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        allowed = grammar.allowed(device) if constrained else None
        logits = filter_logits(
            logits,
            temperature,
            top_k,
            top_p,
            pad_token,
            repetition_penalty=repetition_penalty,
            recent_tokens=tokens[-128:],
            allowed_tokens=allowed,
        )
        probs = F.softmax(logits, dim=-1)
        if torch.isnan(probs).any() or probs.sum() <= 0:
            break
        next_token = int(torch.multinomial(probs, 1).item())
        tokens.append(next_token)
        grammar.observe(next_token)
    return tokens


@torch.no_grad()
def generate_tokens_cached(
    model: torch.nn.Module,
    tokens: list[int],
    length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
    pad_token: int,
    repetition_penalty: float,
    constrained: bool,
    grammar: RemiGrammarState | TokenGrammarState,
) -> list[int]:
    caches = None
    position = 0
    input_token = int(tokens[-1])

    while len(tokens) < length:
        x = torch.tensor([[input_token]], dtype=torch.long, device=device)
        logits_batch, caches = model.forward_cached(x, caches, start_pos=position, max_cache_len=length)
        logits = logits_batch[0, -1]
        allowed = grammar.allowed(device) if constrained else None
        logits = filter_logits(
            logits,
            temperature,
            top_k,
            top_p,
            pad_token,
            repetition_penalty=repetition_penalty,
            recent_tokens=tokens[-128:],
            allowed_tokens=allowed,
        )
        probs = F.softmax(logits, dim=-1)
        if torch.isnan(probs).any() or probs.sum() <= 0:
            break
        next_token = int(torch.multinomial(probs, 1).item())
        tokens.append(next_token)
        grammar.observe(next_token)
        input_token = next_token
        position += 1
    return tokens
