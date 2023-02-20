import torch
import torch.nn as nn
import math
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Attention_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dim_attn):
        super(Attention_LSTM, self).__init__()
        self.input_size = input_size
        self.dim_attn = dim_attn
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)

        self.linear_q = nn.Linear(input_size, dim_attn, bias=False).to(device)
        self.linear_k = nn.Linear(input_size, dim_attn, bias=False).to(device)
        self.linear_v = nn.Linear(input_size, input_size, bias=False).to(device)
        self.scared = 1 / math.sqrt(dim_attn)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        query = self.linear_q(x)
        key = self.linear_k(x)
        value = self.linear_v(x)
        score = torch.bmm(query, key.transpose(1, 2)) * self.scared
        score = torch.softmax(score, dim=-1)
        attn = torch.bmm(score, value)

        out, _ = self.lstm(attn, (h0, c0))
        out = self.fc(out)
        # 返回最后一个时间步的输出
        return out[:, -1, :]
