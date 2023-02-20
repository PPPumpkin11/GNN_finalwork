import numpy as np
import torch
import torch.nn as nn

from data_pro.data_process import load_hsr_data, visualization, DataStandard, data2loader
from model.GAT_LSTM import GATLSTM
from train_and_val import train, show_loss

# 超参数设置===============================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
epoch_num = 100
lr = 0.001
# 以前8个时刻的情况来预测第9个时刻的情况，时间步长为8
look_back = 8

# 数据读取=================================================================
# 读取流入和流出数据以及邻接矩阵
dataset_in, dataset_out, adj = load_hsr_data()
# 特征维度数（147个站点）
data_dim = dataset_in.shape[1]

# 处理邻接矩阵
adj = adj.astype('float32')
adj = torch.tensor(adj).to(device)


x_train_in, y_train_in, x_val_in, y_val_in, data_in = DataStandard(dataset_in, look_back)
x_train_out, y_train_out, x_val_out, y_val_out, data_out = DataStandard(dataset_out)

# GCN输入数据的第二和第三个维度与LSTM是相反的，于是加入维度交换部分
x_train_in = x_train_in.transpose(0, 2, 1)
x_train_out = x_train_out.transpose(0, 2, 1)
x_val_in = x_val_in.transpose(0, 2, 1)
x_val_out = x_val_out.transpose(0, 2, 1)

y_train = np.hstack((y_train_in, y_train_out))
y_val = np.hstack((y_val_in, y_val_out))

train_in_loader = data2loader(x_train_in, y_train_in, batch_size)
val_in_loader = data2loader(x_val_in, y_val_in, batch_size)
train_out_loader = data2loader(x_train_out, y_train_out, batch_size)
val_out_loader = data2loader(x_val_out, y_val_out, batch_size)


# 定义模型、优化器、损失函数===================================================
model = GATLSTM(adj)
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=5e-4)
loss_function = nn.MSELoss()


# 训练以及可视化==============================================================
predict_train_in, predict_validation_in,  train_loss_list, val_loss_list = \
    train(train_in_loader, val_in_loader, epoch_num, optimizer, loss_function, model)
visualization(data_in, predict_train_in, predict_validation_in, look_back=8, k=0)
# predict_train_out, predict_validation_out,  train_loss_list, val_loss_list = \
#     train(train_out_loader, val_out_loader, epoch_num, optimizer, loss_function, model)
# visualization(data_out, predict_train_out, predict_validation_out, look_back=8, k=0)
show_loss(train_loss_list)
show_loss(val_loss_list)
