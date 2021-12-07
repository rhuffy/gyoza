import torch
import torch.nn as nn
import torch.nn.functional as F
from .document_featurizer import DocumentFeaturizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EncoderLSTMStack(nn.Module, DocumentFeaturizer):
    def __init__(self, num_embeddings: int, embedding_dim: int, hidden_size: int, out_size: int, nlayers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.rnn_f = LSTMStack(nlayers, embedding_dim, hidden_size)
        self.rnn_b = LSTMStack(nlayers, embedding_dim, hidden_size)

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        # attention over seq_model output
        self.query_vector = nn.Parameter(torch.randn(1, 64))
        self.attn_w = nn.Parameter(torch.randn(64, hidden_size * 2))

        self.linear1 = nn.Linear(hidden_size * 2, 32)
        self.linear2 = nn.Linear(32, out_size)

    def init_state(self):
        state = []
        for i in range(self.nlayers):
            h_0 = torch.zeros(1, self.hidden_size, device=device)
            c_0 = torch.zeros(1, self.hidden_size, device=device)
            r_0 = torch.zeros(1, self.hidden_size, device=device)
            state.append(((h_0, c_0), r_0, None, None))
        return state

    def forward(self, embeds):
        state_f = self.init_state()
        state_b = self.init_state()
        mem_f = torch.zeros(embeds.size(0), self.hidden_size, device=device)
        mem_b = torch.zeros(embeds.size(0), self.hidden_size, device=device)

        for i in range(embeds.size(0)):
            embeds_input = embeds[i].view(1, -1)
            o_t, state_f = self.rnn_f(embeds_input, state_f)
            mem_f[i] = o_t

        for i in reversed(range(embeds.size(0))):
            embeds_input = embeds[i].view(1, -1)
            o_t, state_b = self.rnn_b(embeds_input, state_b)
            mem_b[i] = o_t

        embed = torch.cat((mem_f, mem_b), dim=1)

        attn = torch.mm(torch.mm(self.query_vector, self.attn_w), embed.t())
        # output: (1, frame)
        attn = F.softmax(attn, dim=1).t()

        ctxt = torch.sum(attn * embed, dim=0)

        embed = F.relu(self.linear1(ctxt))
        return torch.sigmoid(self.linear2(embed))


class LSTMStackCell(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super().__init__()
        self.rnn = nn.LSTMCell(embedding_dim + hidden_size, hidden_size)
        self.d_layer = nn.Linear(hidden_size, 1)
        self.u_layer = nn.Linear(hidden_size, 1)
        self.v_layer = nn.Linear(hidden_size, hidden_size)
        self.o_layer = nn.Linear(hidden_size, hidden_size)
        self.rnn_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, inpt, state):
        prev_hidden, prev_r, prev_V, prev_s = state
        rnn_input = torch.cat((inpt, prev_r), dim=1)
        rnn_output, new_hidden = self.rnn(rnn_input, prev_hidden)
        rnn_output = self.rnn_out(rnn_output)
        d_t = self.d_layer(rnn_output)
        u_t = self.u_layer(rnn_output)
        v_t = self.v_layer(rnn_output)
        o_t = self.o_layer(rnn_output)

        if prev_V is None:
            new_V = v_t.detach()
        else:
            new_V = torch.cat((prev_V, v_t.detach()), dim=0)

        if prev_s is None:
            new_s = d_t.detach()
        else:
            shid_prev_s = torch.flip(torch.cumsum(
                torch.flip(prev_s, [0]), dim=0), [0]) - prev_s
            new_s = torch.clamp(
                prev_s - torch.clamp(u_t.item() - shid_prev_s, min=0), min=0)
            new_s = torch.cat((new_s, d_t.detach()), dim=0)

        shid_new_s = torch.flip(torch.cumsum(
            torch.flip(new_s, [0]), dim=0), [0]) - new_s
        r_scalars = torch.min(new_s, torch.clamp(1 - shid_new_s, min=0))
        new_r = torch.sum(r_scalars.view(-1, 1) * new_V, dim=0).view(1, -1)
        return o_t, ((rnn_output, new_hidden), new_r, new_V, new_s)


class LSTMStack(nn.Module):
    def __init__(self, nlayers, embedding_dim, hidden_size):
        super().__init__()
        self.nlayers = nlayers
        layers = [LSTMStackCell(embedding_dim, hidden_size)]
        layers.extend([LSTMStackCell(hidden_size, hidden_size)
                      for _ in range(nlayers-1)])
        self.rnn_layers = nn.ModuleList(layers)

    def forward(self, inpt, state):
        new_state = []
        for i in range(self.nlayers):
            inpt, ns = self.rnn_layers[i](inpt, state[i])
            new_state.append(ns)
        return inpt, new_state
