from data_process import load_hsr_data
import numpy as np
import pandas as pd
from pandas import Series
from matplotlib import pyplot as plt
import seaborn as sns
import random

# 数据读取===============================================
# 读取流入和流出数据以及邻接矩阵
dataset_in, dataset_out, adj = load_hsr_data()
# 特征维度数（147个站点）
data_dim = dataset_in.shape[1]

dataset_in_array = np.array(dataset_in)
dataset_out_array = np.array(dataset_out)

adj = adj.astype('float32')

# 画出站点的出度分布=======================================
out_degree = []
for i in range(data_dim):
    no_zero = np.nonzero(adj[i])
    out_degree.append(len(no_zero[1]))

station = [i for i in range(147)]
plt.figure()
plt.bar(station, out_degree, color="green")
plt.xlabel('station')
plt.ylabel('out degree')
plt.title('Out degree of every station')
plt.show()

# 每一个出度选择一个站点，展示该站点各个时刻的客流量============
# 统计出度为0，1，2，3，4，5的分别是哪些站点
idx = {}
idx[0] = []
idx[1] = []
idx[2] = []
idx[3] = []
idx[4] = []
idx[5] = []
for i in range(data_dim):
    if out_degree[i] == 0:
        idx[0].append(i)
    if out_degree[i] == 1:
        idx[1].append(i)
    if out_degree[i] == 2:
        idx[2].append(i)
    if out_degree[i] == 3:
        idx[3].append(i)
    if out_degree[i] == 4:
        idx[4].append(i)
    if out_degree[i] == 5:
        idx[5].append(i)

# 从0-5每一种出度中选择一个站点，将该站点的输入客流量存放到show_station中
show_station = {}
for i in range(6):
    if len(idx[i]) == 1:
        col_num = idx[i][0]  # 取出dataset中对应的列的索引
        show_station[i] = dataset_in_array[:, col_num]
    if len(idx[i]) != 1:
        idx_choice = random.choice(idx[i])
        show_station[i] = dataset_in_array[:, idx_choice]

plt.figure()
t = [i for i in range(len(show_station[0]))]
plt.plot(t, show_station[0])
plt.plot(t, show_station[1])
plt.plot(t, show_station[2])
plt.plot(t, show_station[3])
plt.plot(t, show_station[4])
plt.plot(t, show_station[5])
plt.legend(['out degree=0', 'out degree = 1', 'out degree = 2',
            'out degree = 3', 'out degree = 4', 'out degree=5'])
plt.show()

# 分析时序相关性并可视化，以及计算站点之间的相关性==============
# 用热力图可视化相关性
dataset_in_frame = pd.DataFrame()
for i in range(data_dim):
    value = dataset_in_array[:, i]
    dataset_in_frame[i] = value

fig = plt.figure()
ax = fig.add_subplot(111)
b = dataset_in_frame.corr()
sns.heatmap(dataset_in_frame.corr(), ax=ax, cmap='Blues', square=True)
plt.show()

# 计算两两之间的相关性，存放到矩阵corr中
corr = np.zeros((data_dim, data_dim))
for i in range(data_dim):
    for j in range(data_dim):
        s1 = Series(dataset_in.iloc[:, i])
        s2 = Series(dataset_in.iloc[:, j])
        corr[i, j] = s1.corr(s2)
print(corr)
