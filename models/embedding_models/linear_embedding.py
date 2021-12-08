import torch.nn as nn

from .embedding_model import EmbeddingModel


class LinearEmbedding(nn.Module, EmbeddingModel):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
