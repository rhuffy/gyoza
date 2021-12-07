import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .document_featurizer import DocumentFeaturizer


class LSTMDocumentFeaturizer(nn.Module, DocumentFeaturizer):
    def __init__(self, num_embeddings: int, embedding_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)

    # documents.shape = [length, batch_size], lengths.shape = [lengths]
    def forward(self, documents: np.ndarray, lengths: np.ndarray, is_sorted=False):
        # embeds.shape = [length, batch_size, embedding_dim]
        embeds = self.embedding(documents)
        packed_seq = pack_padded_sequence(embeds, lengths, enforce_sorted=is_sorted)
        lstm_out, _ = self.lstm(packed_seq)
        # padded_seq.shape = [length, batch_size, hidden_dim]
        padded_seq, lengths = pad_packed_sequence(lstm_out)
        # simple model: use only last lstm output
        last = padded_seq[-1]
        return self.linear(last).squeeze()
