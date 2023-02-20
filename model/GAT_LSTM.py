import torch
import torch.nn as nn
import torch.nn.functional as F
from model.LSTM import LSTM
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # W是与输入h相乘的参数
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features))).to(device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 因为公式中a要与将两个Wh进行concat的结果相乘，所以这里需要是2*out_features
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1))).to(device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h(64, 147, 8)
        Wh = torch.matmul(h, self.W)  # (64,147,64)
        before_relu = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(before_relu, self.a).squeeze(3)).to(device)  # (64, 147, 147)

        zero_vec = -9e15 * torch.ones_like(e).to(device)
        attention = torch.where(adj > 0, e, zero_vec)  # 只对邻居节点进行注意力加权
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # (64, 147, 64)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[1]  # 样本数
        Wh_repeat = Wh.repeat_interleave(N, dim=1)  # (batch, N*N, out_features)(64, 147*147, 64)
        Wh_repeat1 = Wh.repeat_interleave(N, dim=1)
        # 完成Whi和Whj的concat
        combined_mat = torch.cat([Wh_repeat, Wh_repeat1], dim=2)  # (64, 147*147, 64*2)
        return combined_mat.view(-1, N, N, 2*self.out_features)   # (64, 147, 147, 64*2)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nheads, dropout, alpha):
        super(GAT, self).__init__()
        self.dropout = dropout
        # 多头注意力
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True)
                           for _ in range(nheads)]

        # 给多头注意力机制的每一头加上名称
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'. format(i), attention)

        self.out_att = GraphAttentionLayer(nhid*nheads, nhid*nheads, dropout, alpha, concat=False)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)  # (64, 147, 8)
        # 将多头合并,将两个(64, 147, 64)合并成为(64, 147, 128)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # output = F.log_softmax(x, dim=1)
        # 得到加了attention之后的x
        return x


class GATLSTM(nn.Module):
    def __init__(self, adj):
        super(GATLSTM, self).__init__()
        self.adj = adj
        self.LSTM = LSTM(input_size=147, hidden_size=256, output_size=147, num_layers=2)
        self.GAT = GAT(nfeat=8, nhid=64, nheads=2, dropout=0.001, alpha=0.1)

    def forward(self, x):
        gat_output = self.GAT(x, self.adj)  # (64, 147, 128)
        output = self.LSTM(gat_output.transpose(1, 2))
        return output
