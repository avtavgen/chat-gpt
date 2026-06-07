from attention import MultiHeadedSelfAttention
from neural_network import VanillaNeuralNetwork
from torchtyping import TensorType
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(
        self, model_dim: int, num_heads: int, context_length: int, dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadedSelfAttention(
            model_dim, num_heads, context_length, dropout
        )
        self.linear_network = VanillaNeuralNetwork(model_dim, dropout)
        self.first_norm = nn.LayerNorm(model_dim)
        self.second_norm = nn.LayerNorm(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        embedded = embedded + self.attention(self.first_norm(embedded))
        embedded = embedded + self.linear_network(self.second_norm(embedded))
        return embedded
