import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from document_featurizer import DocumentFeaturizer
from lstmn import BiLSTMN


class LSTMNDocumentFeaturizer(nn.Module, DocumentFeaturizer):
    def __init__(self, num_embeddings: int, embedding_dim: int, hidden_dim: int, out_size: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.model = BiLSTMN(embedding_dim, hidden_dim)

        # attention over seq_model output
        self.query_vector = nn.Parameter(torch.randn(1, 64))
        self.attn_w = nn.Parameter(torch.randn(64, hidden_dim * 2))

        self.linear1 = nn.Linear(hidden_dim * 2, 32)
        self.linear2 = nn.Linear(32, out_size)

    # documents.shape = [length, 1]
    def forward(self, documents: np.ndarray):
        # embeds.shape = [length, 1, embedding_dim]
        embeds = self.embedding(documents)
        embed = self.model(embeds)

        attn = torch.mm(torch.mm(self.query_vector, self.attn_w), embed.t())
        # output: (1, frame)
        attn = F.softmax(attn, dim=1).t()

        ctxt = torch.sum(attn * embed, dim=0)

        embed = F.relu(self.linear1(ctxt))
        return torch.sigmoid(self.linear2(embed))
