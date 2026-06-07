import torch.nn as nn
from torchtyping import TensorType

from attention import MultiHeadedSelfAttention
from neural_network import VanillaNeuralNetwork


class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.attention = MultiHeadedSelfAttention(model_dim, num_heads)
        self.linear_network = VanillaNeuralNetwork(model_dim)
        self.first_norm = nn.LayerNorm(model_dim)
        self.second_norm = nn.LayerNorm(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        embedded = embedded + self.attention(
            self.first_norm(embedded)
        )
        embedded = embedded + self.linear_network(
            self.second_norm(embedded)
        )
        return embedded
