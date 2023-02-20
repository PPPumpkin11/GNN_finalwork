import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.utils.data as data


def load_hsr_data():
    hsr_adj = pd.read_csv(r'../data/HSR_adj.csv', header=None)
    adj = np.mat(hsr_adj)
    hsr_tf_in = pd.read_csv(r'../data/HSR_inflow.csv')
    hsr_tf_in = hsr_tf_in.fillna(0)
    hsr_tf_out = pd.read_csv(r'../data/HSR_outflow.csv')
    hsr_tf_out = hsr_tf_out.fillna(0)
    return hsr_tf_in, hsr_tf_out, adj


def create_dataset(dataset, look_back=8):
    # 创建数据集
    # 这个函数用于将每look_back个时刻的数据构成features
    # 将第look_back+1个构成labels
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        x = dataset[i:i+look_back, :]
        dataX.append(x)
        y = dataset[i+look_back, :]
        dataY.append(y)
    return np.array(dataX), np.array(dataY)


def visualization(dataset1, predict_train1, predict_validation1, look_back=8, k=0):
    # 因为predict_train1是以batch的形式输入本函数的
    # 所以将batch拆开成一个个的预测值用于可视化
    predict_train1_new = []
    for i in range(len(predict_train1)):
        for j in range(predict_train1[i].shape[0]):
            predict_train1_batch = predict_train1[i]
            predict_train1_new.append(predict_train1_batch[j, :])
    predict_train1_new = np.array(predict_train1_new)
    predict_validation1_new = []
    for i in range(len(predict_validation1)):
        for j in range(predict_validation1[i].shape[0]):
            predict_validation1_batch = predict_validation1[i]
            predict_validation1_new.append(predict_validation1_batch[j, :])
    # 构建通过训练数据集进行预测的图表数据
    predict_train_plot = np.empty_like(dataset1)
    predict_train_plot[:, :] = np.nan
    predict_train_plot[look_back:len(predict_train1_new) + look_back, :] = predict_train1_new

    # 构建通过评估数据集进行预测的图表数据
    predict_validation_plot = np.empty_like(dataset1)
    predict_validation_plot[:, :] = np.nan
    predict_validation_plot[len(predict_train1_new) + look_back * 2 + 1: len(dataset1) - 1, :] = predict_validation1_new

    # 图表显示第一个地铁站的时序曲线
    plt.plot(dataset1[:, k], color='black')
    plt.plot(predict_train_plot[:, k], color='green')
    plt.plot(predict_validation_plot[:, k], color='red')
    plt.legend(["True value", "train", "test"], loc='upper left')
    plt.show()


def DataStandard(dataset1, look_back=8):
    # 标准化数据
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset1)

    # 7：1的比例划分训练集和验证集
    train_size = int(len(dataset) * 0.875)
    train, validation = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # 创建dataset，使数据产生相关性
    x_train, y_train = create_dataset(train, look_back)
    x_validation, y_validation = create_dataset(validation, look_back)

    # x_train的shape为(950,8,147) 950为以look_back划分出来的组数
    # x_val的shape为(128, 8, 147)

    # 将数据转换成[样本，特征维度, 时间步长]的形式
    # x_train变为(950,147,8)
    # x_train = np.reshape(x_train, (x_train.shape[0], data_dim, x_train.shape[1]))
    # x_validation = np.reshape(x_validation, (x_validation.shape[0], data_dim, x_validation.shape[1]))

    return x_train, y_train, x_validation, y_validation, dataset


def data2loader(X_data, Y_data, batch_size):
    X_data_tensor = torch.from_numpy(X_data.astype(np.float32))
    Y_data_tensor = torch.from_numpy(Y_data.astype(np.float32))
    tensor_dataset = data.TensorDataset(X_data_tensor, Y_data_tensor)
    loader = data.DataLoader(tensor_dataset, batch_size)
    return loader
