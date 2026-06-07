import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):
    def __init__(self, model_dim: int, head_size: int):
        super().__init__()
        self.key_gen = nn.Linear(model_dim, head_size, bias=False)
        self.query_gen = nn.Linear(model_dim, head_size, bias=False)
        self.value_gen = nn.Linear(model_dim, head_size, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        k = self.key_gen(embedded)
        q = self.query_gen(embedded)
        v = self.value_gen(embedded)

        scores = q @ torch.transpose(k, 1, 2)
        context_length, attention_dim = k.shape[1], k.shape[2]
        scores = scores / (attention_dim**0.5)

        lower_triangular = torch.tril(torch.ones(context_length, context_length))
        mask = lower_triangular == 0
        mask = mask.to(scores.device)
        scores = scores.masked_fill(mask, float("-inf"))
        scores = nn.functional.softmax(scores, dim=2)

        return scores @ v


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.att_heads = nn.ModuleList()
        for i in range(num_heads):
            self.att_heads.append(
                SingleHeadAttention(model_dim, model_dim // num_heads)
            )
        self.output_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        head_outputs = []
        for head in self.att_heads:
            head_outputs.append(head(embedded))
        concatenated = torch.cat(head_outputs, dim=2)
        return self.output_proj(concatenated)
