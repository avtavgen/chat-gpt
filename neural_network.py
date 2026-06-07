import torch.nn as nn
from torchtyping import TensorType

class VanillaNeuralNetwork(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.up_projection = nn.Linear(model_dim, model_dim * 4)
        self.relu = nn.ReLU()
        self.down_projection = nn.Linear(model_dim * 4, model_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        return self.dropout(self.down_projection(self.relu(self.up_projection(x))))
