import torch.nn as nn
from torchtyping import TensorType

class VanillaNeuralNetwork(nn.Module):
    def __init__(self, model_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        return self.net(x)
