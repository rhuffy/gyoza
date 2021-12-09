import torch.nn as nn
import torch

from .embedding_model import EmbeddingModel
from ..ConcreteDropout.condrop import ConcreteDropout


class LinearEmbedding(nn.Module, EmbeddingModel):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        weight_regularizer: float = 1e-6,
        dropout_regularizer: float = 1e-3,
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.cd1 = ConcreteDropout(
            weight_regulariser=weight_regularizer, dropout_regulariser=dropout_regularizer
        )

        self.relu1 = nn.LeakyReLU()

        self.linear2 = nn.Linear(hidden_features, 1)
        self.cd2 = ConcreteDropout(
            weight_regulariser=weight_regularizer, dropout_regulariser=dropout_regularizer
        )

        self.relu2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.cd1(x, nn.Sequential(self.linear1, self.relu1))
        x = self.cd2(x, nn.Sequential(self.linear2, self.relu2))
        return x
