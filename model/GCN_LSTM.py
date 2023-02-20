import torch
import torch.nn as nn
from model.LSTM import LSTM
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 计算GCN公式中的Lsym
def calculate_laplacian_with_self_loop(matrix):
    A_bar = matrix+torch.eye(matrix.size(0))  # 147x147
    D_bar = A_bar.sum(1)  # D出度矩阵 147x1
    D_inv_sqrt = torch.pow(D_bar, -0.5).flatten()  # 1x147
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    D_mat_inv_sqrt = torch.diag(D_inv_sqrt)  # 147x147
    normalized_laplacian = (
        A_bar.matmul(D_mat_inv_sqrt).transpose(0, 1).matmul(D_mat_inv_sqrt)
    )  # 147x147
    return normalized_laplacian


class GCN(nn.Module):
    def __init__(self, adj, input_size, look_back):
        super(GCN, self).__init__()
        self.register_buffer(
            'laplacian', calculate_laplacian_with_self_loop(
                torch.FloatTensor(adj)).to(device)
        )  # 引入Lsym，作为一个参数，名为laplacian
        # self.register_buffer('name', tensor)作用是定义一组参数
        # 该组参数在模型训练的时候不会更新，
        # 但是保存模型的时候，该组参数又作为模型的一部分被保存

        self.node_num = adj.shape[0]
        self.input_size = input_size
        self.look_back = look_back
        self.weight = nn.Parameter(
            torch.FloatTensor(self.look_back, self.look_back)
        ).to(device)
        self.reset_parmeters()

    def reset_parmeters(self):
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('tanh'))

    def forward(self, x):
        ax = self.laplacian @ x  # (147,147)*(64,147,8)=(64,147,8)
        output = ax @ self.weight  # (64,147,8)
        return output


class GCNLSTM(nn.Module):
    def __init__(self, adj, input_size, hidden_size, output_size, num_layers, look_back):
        super(GCNLSTM, self).__init__()
        self.GCN = GCN(adj, input_size, look_back)
        self.LSTM = LSTM(input_size, hidden_size, output_size, num_layers)

    def forward(self, x):
        gcn_out = self.GCN(x).transpose(1, 2)
        out = self.LSTM(gcn_out)
        return out
