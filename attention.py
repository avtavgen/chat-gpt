import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):
    def __init__(
        self, model_dim: int, head_size: int, context_length: int, dropout: float = 0.1
    ):
        super().__init__()
        self.key_gen = nn.Linear(model_dim, head_size, bias=False)
        self.query_gen = nn.Linear(model_dim, head_size, bias=False)
        self.value_gen = nn.Linear(model_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(context_length, context_length, dtype=torch.bool)),
        )

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        B, T, C = embedded.shape

        k = self.key_gen(embedded)  # (B, T, head_size)
        q = self.query_gen(embedded)  # (B, T, head_size)
        v = self.value_gen(embedded)  # (B, T, head_size)

        head_size = k.shape[-1]
        scores = q @ k.transpose(1, 2) / (head_size**0.5)  # (B, T, T)

        scores = scores.masked_fill(~self.tril[:T, :T], float("-inf"))
        scores = nn.functional.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        return scores @ v


class MultiHeadedSelfAttention(nn.Module):
    def __init__(
        self, model_dim: int, num_heads: int, context_length: int, dropout: float = 0.1
    ):
        super().__init__()
        head_size = model_dim // num_heads
        self.att_heads = nn.ModuleList(
            [
                SingleHeadAttention(model_dim, head_size, context_length, dropout)
                for _ in range(num_heads)
            ]
        )
        self.output_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        concatenated = torch.cat([h(embedded) for h in self.att_heads], dim=-1)
        return self.dropout(self.output_proj(concatenated))
