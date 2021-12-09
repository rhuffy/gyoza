import torch.nn as nn

from .embedding_model import EmbeddingModel
from ..ConcreteDropout.condrop import ConcreteDropout


class LinearEmbedding(nn.Module, EmbeddingModel):
    def __init__(self, in_features: int, out_features: int, weight_regularizer: float = 1e-6, dropout_regularizer: float = 1e-3):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.cd = ConcreteDropout(weight_regulariser=weight_regularizer, dropout_regulariser=dropout_regularizer)

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.cd(x, nn.Sequential(self.linear, self.relu))
