from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

KVCache = tuple[torch.Tensor, torch.Tensor, int]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


def _grouped_query_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    groups: int,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, num_heads, seq_len, head_dim = q.shape
    num_kv_heads = k.size(1)
    q = q.view(batch, num_kv_heads, groups, seq_len, head_dim)
    scores = torch.einsum("bhgld,bhsd->bhgls", q, k) * (head_dim ** -0.5)
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask.view(1, 1, 1, seq_len, k.size(2)), -float("inf"))
    weights = torch.softmax(scores, dim=-1)
    y = torch.einsum("bhgls,bhsd->bhgld", weights, v)
    return y.reshape(batch, num_heads, seq_len, head_dim)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        cos = emb.cos().to(dtype=dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float,
        rope_theta: float,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, theta=rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(seq_len, x.device, q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(y)

    def forward_cached(
        self,
        x: torch.Tensor,
        cache: KVCache | None,
        start_pos: int,
        max_cache_len: int | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(seq_len, x.device, q.dtype, start_pos=start_pos)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        cache_len = start_pos + seq_len
        if cache is None:
            capacity = max(max_cache_len or cache_len, cache_len)
            cache_k = torch.empty(
                batch,
                self.num_kv_heads,
                capacity,
                self.head_dim,
                device=x.device,
                dtype=k.dtype,
            )
            cache_v = torch.empty_like(cache_k)
        else:
            cache_k, cache_v, _ = cache
            if cache_len > cache_k.size(2):
                raise ValueError(f"KV cache capacity {cache_k.size(2)} is smaller than required length {cache_len}")
        cache_k[:, :, start_pos:cache_len, :] = k
        cache_v[:, :, start_pos:cache_len, :] = v
        next_cache = (cache_k, cache_v, cache_len)

        attn_k = cache_k[:, :, :cache_len, :]
        attn_v = cache_v[:, :, :cache_len, :]
        attn_mask = None
        if seq_len > 1:
            query_positions = torch.arange(start_pos, cache_len, device=x.device)
            key_positions = torch.arange(cache_len, device=x.device)
            attn_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            y = _grouped_query_attention(q, attn_k, attn_v, repeats, attn_mask)
        else:
            y = F.scaled_dot_product_attention(
                q,
                attn_k,
                attn_v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(y), next_cache


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_dim: int,
        dropout: float,
        rope_theta: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, num_kv_heads, dropout, rope_theta)
        self.ffn = SwiGLU(d_model, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x

    def forward_cached(
        self,
        x: torch.Tensor,
        cache: KVCache | None,
        start_pos: int,
        max_cache_len: int | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        attn_out, next_cache = self.attn.forward_cached(self.attn_norm(x), cache, start_pos, max_cache_len)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, next_cache


class MusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float,
        max_seq_len: int,
        padding_idx: int,
        num_kv_heads: int | None = None,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        del max_seq_len
        self.d_model = d_model
        self.padding_idx = padding_idx
        num_kv_heads = num_kv_heads or num_heads
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, num_kv_heads, ffn_dim, dropout, rope_theta)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = RMSNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)
        self.fc_out.weight = self.embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed_dropout(self.embedding(tokens))
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(self.ln_f(x))

    def forward_cached(
        self,
        tokens: torch.Tensor,
        caches: list[KVCache | None] | None = None,
        start_pos: int = 0,
        max_cache_len: int | None = None,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        if caches is None:
            caches = [None] * len(self.layers)
        x = self.embed_dropout(self.embedding(tokens))
        next_caches = []
        for layer, cache in zip(self.layers, caches, strict=True):
            x, next_cache = layer.forward_cached(x, cache, start_pos, max_cache_len)
            next_caches.append(next_cache)
        return self.fc_out(self.ln_f(x)), next_caches


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
