import torch
import torch.nn as nn
from torchtyping import TensorType

def build_rope_cache(
    seq_len: int, head_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin tables for RoPE.
    Returns (cos, sin) each of shape (seq_len, head_dim).
    Called once per forward pass; cheap because seq_len is small.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    theta = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)  # (T, head_dim/2)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # (T, head_dim)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # (T, head_dim)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension to implement RoPE."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to Q and K.
    q, k: (B, num_heads, T, head_dim)
    cos, sin: (T, head_dim) — broadcast over B and heads
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        context_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
        assert model_dim % num_heads == 0, (
            f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads
        self.repeats = num_heads // num_kv_heads

        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(context_length, context_length, dtype=torch.bool)),
        )

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        q = (
            self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        )  # (B, H,   T, hd)
        k = (
            self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        )  # (B, Hkv, T, hd)
        v = (
            self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        )  # (B, Hkv, T, hd)

        cos, sin = build_rope_cache(T, self.head_dim, x.device)
        q, k = apply_rope(q, k, cos, sin)

        if self.repeats > 1:
            k = k.repeat_interleave(self.repeats, dim=1)  # (B, H, T, hd)
            v = v.repeat_interleave(self.repeats, dim=1)

        scores = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)  # (B, H, T, T)
        scores = scores.masked_fill(~self.tril[:T, :T], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj_dropout(self.output_proj(out))
