import torch
import torch.nn as nn
from torch.autograd import Variable


class SRU_Layer(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(SRU_Layer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.weight = nn.Parameter(torch.Tensor(input_dim, self.hidden_dim * 3))
        dist_max = (2.0 / self.input_dim) ** 0.5
        self.weight.data.uniform_(-dist_max, dist_max)

        self.bias = Variable(torch.zeros(hidden_dim * 2))

    def calculate_sru_layer(self, U, x):
        seq_len = x.size(0)
        batch = x.size(-2)

        U = U.view(seq_len, batch, self.hidden_dim, U.size(-1) // self.hidden_dim)

        forget_bias, reset_bias = self.bias.view(2, self.hidden_dim)
        f_t = torch.sigmoid((U[..., 1] + forget_bias))
        r_t = torch.sigmoid((U[..., 2] + reset_bias))
        h_t = torch.zeros(seq_len, batch, self.hidden_dim)

        x_ = x.view(seq_len, batch, self.hidden_dim)

        c_prev = torch.zeros(batch, self.hidden_dim)
        for t in range(seq_len):
            # Formula (4) from the paper: f[t] * c[t-1] + (1 − f[t]) * U[t, 2]
            c_t = f_t[t, ...] * c_prev + ((1-f_t[t, ...]) * U[..., 0][t, :, :])
            c_prev = c_t

            # Formula (5):  r[t] * c[t] + (1 − r[t]) * x[t]
            h_t[t, ...] = r_t[t, ...] * c_t.tanh() + ((1 - r_t[t, ...]) * x_[t, ...])

        return h_t, c_t

    def forward(self, x):
        U = torch.matmul(x.contiguous().view(-1, self.input_dim), self.weight)

        xt, ct = self.calculate_sru_layer(U, x)

        ht = self.dropout(xt)
        ct = self.dropout(ct)

        return ht, ct


class SRU(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(SRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_size = hidden_dim

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = SRU_Layer(input_dim=input_dim, hidden_dim=self.hidden_dim)
            self.layers.append(layer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            output, _ = layer(x)

        return output
