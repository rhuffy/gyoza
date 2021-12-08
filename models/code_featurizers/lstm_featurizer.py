import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .document_featurizer import DocumentFeaturizer


class LSTMDocumentFeaturizer(nn.Module, DocumentFeaturizer):
    def __init__(self, num_embeddings: int, embedding_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)

    # documents.shape = [length, batch_size], lengths.shape = [lengths]
    def forward(self, documents: np.ndarray):
        # embeds.shape = [length, batch_size, embedding_dim]
        embeds = self.embedding(documents)
        lstm_out, _ = self.lstm(embeds)
        # padded_seq.shape = [length, batch_size, hidden_dim]
        last = lstm_out[-1]
        return self.linear(last).squeeze()
