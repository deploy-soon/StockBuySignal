import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, num_features=10, num_layers=1, num_hidden=32):
        super(Model, self).__init__()
        self.num_features = num_features
        #self.seq_len = lag
        self.num_hidden = num_hidden # number of hidden states
        self.num_layers = num_layers

        self.feature_map = nn.Linear(num_features, num_features)

        self.lstm = nn.LSTM(input_size=self.num_features,
                            hidden_size=self.num_hidden,
                            num_layers=self.num_layers,
                            batch_first=True)

        #std = np.sqrt(1 / self.num_hidden)
        #self.attn_W = nn.Parameter(torch.Tensor(self.num_hidden, self.num_hidden),
        #                           requires_grad=True)
        #nn.init.uniform_(self.attn_W, -std, std)
        #self.attn_b = nn.Parameter(torch.Tensor(self.num_hidden, 1),
        #                           requires_grad=True)
        self.attn = nn.Linear(self.num_hidden, self.num_hidden)
        self.attn_u = nn.Parameter(torch.Tensor(self.num_hidden, 1),
                                   requires_grad=True)
        nn.init.normal_(self.attn_u)
        #nn.init.uniform_(self.attn_u, -std, std)


        self.fc = torch.nn.Linear(self.num_hidden * 2, 1)

    def init_hidden(self, batch_size, device="cuda"):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.num_layers, batch_size, self.num_hidden, device=device)
        cell_state = torch.zeros(self.num_layers, batch_size, self.num_hidden, device=device)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        self.init_hidden(batch_size)

        x = self.feature_map(x)
        x = torch.tanh(x)
        lstm_out, self.hidden = self.lstm(x, self.hidden)

        # (batch_size,seq_len,num_directions * hidden_size)
        #attn_weight = torch.matmul(lstm_out, self.attn_W) + self.attn_b
        attn_weight = torch.tanh(self.attn(lstm_out))
        attn_score = torch.matmul(attn_weight, self.attn_u).squeeze()
        weights = F.softmax(attn_score, dim=1)
        weighted = torch.mul(lstm_out, weights.unsqueeze(-1).expand_as(lstm_out))
        weighted_sum = torch.sum(weighted, dim=1)
        x = torch.cat((lstm_out[:, -1, :], weighted_sum), 1)
        x = self.fc(x).squeeze()
        x = F.sigmoid(x)
        return x

