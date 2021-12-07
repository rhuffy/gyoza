import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTMN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1, tape_depth=30):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tape_depth = tape_depth

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout)

        self.attn_wh = nn.Parameter(torch.randn(num_layers, hidden_size, hidden_size))
        self.attn_wx = nn.Parameter(torch.randn(num_layers, hidden_size, input_size))
        self.attn_wht = nn.Parameter(torch.randn(num_layers, hidden_size, hidden_size))
        self.attn_v = nn.Parameter(torch.randn(num_layers, hidden_size))

    def forward(self, xs):
        batch_size = xs.size(1)

        self.hidden_state_cum = torch.zeros(xs.size(0), self.hidden_size, device=device)

        self.hidden_state_tape = torch.zeros(
            self.tape_depth, self.num_layers, batch_size, self.hidden_size, device=device
        )
        self.cell_state_tape = torch.zeros(
            self.tape_depth, self.num_layers, batch_size, self.hidden_size, device=device
        )

        self.prev_hidden_tape = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )
        self.prev_cell_tape = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )

        assert len(xs) >= self.tape_depth

        for i in range(self.tape_depth):
            _, (hidden_state, cell_state) = self.lstm(
                xs[i].unsqueeze(0), (self.prev_hidden_tape, self.prev_cell_tape)
            )

            self.hidden_state_tape[i] = hidden_state.detach()
            self.hidden_state_cum[i] = hidden_state[-1].squeeze()
            self.cell_state_tape[i] = cell_state.detach()

        for i, x in enumerate(xs[self.tape_depth :]):
            hidden_attn = torch.einsum('jll,ijkl->ijkl', self.attn_wh, self.hidden_state_tape)
            x_attn = torch.einsum('ijk,lk->ilj', self.attn_wx, x)
            hidden_t_attn = torch.einsum('ijj,ikj->ikj', self.attn_wht, self.prev_hidden_tape)

            a = torch.tanh(hidden_attn + x_attn + hidden_t_attn)
            a = torch.einsum('ijkl,jl->ijk', a, self.attn_v)
            alpha = F.softmax(a, dim=0)

            self.prev_hidden_tape = torch.einsum('ijk,ijkl->jkl', alpha, self.hidden_state_tape)
            self.prev_cell_tape = torch.einsum('ijk,ijkl->jkl', alpha, self.cell_state_tape)

            _, (hidden_state, cell_state) = self.lstm(
                x.unsqueeze(0), (self.prev_hidden_tape, self.prev_cell_tape)
            )

            self.hidden_state_tape[0 : self.tape_depth - 1] = self.hidden_state_tape[
                1 : self.tape_depth
            ].clone()
            self.hidden_state_tape[-1] = hidden_state
            self.hidden_state_cum[i] = hidden_state[-1].squeeze()

            self.cell_state_tape[0 : self.tape_depth - 1] = self.cell_state_tape[
                1 : self.tape_depth
            ].clone()
            self.cell_state_tape[-1] = cell_state

        return self.hidden_state_cum


class BiLSTMN(nn.Module):
    def __init__(self, input_size, hidden_size, **kwargs):
        super(BiLSTMN, self).__init__()

        self.f = LSTMN(input_size, hidden_size, **kwargs)
        self.b = LSTMN(input_size, hidden_size, **kwargs)

    def forward(self, xs):
        f = self.f(xs)
        b = self.b(torch.flip(xs, [0]))

        return torch.cat((f, b), dim=1)
