├─data
      ├─ HSR_adj.csv  147个站点彼此之间的连接关系
      ├─ HSR_inflow.csv  历史时刻各个地铁站点的客流流入情况
      └─ HSR_outflow.csv  历史时刻各个地铁站点的客流流出情况
├─data_pro
      ├─ data_analysis.py  数据分析，包括站点的出度、不同出度站点客流量以及站点之间相关性的分析
      └─ data_process.py  数据处理，包括数据读取，建立数据关系、数据标准化、按batch读取数据以及结果可视化
├─main
       ├─ main_Attention_LSTM.py  Self-attention+LSTM预测客流量主函数
       ├─ main_GAT_LSTM.py  GAT+LSTM预测客流量主函数
       ├─ main_GCN_LSTM.py  GCN+LSTM预测客流量主函数
       └─ main_LSTM.py  时序预测模型LSTM预测客流量主函数
├─model
       ├─ Attention_LSTM.py  Self-attention+LSTM网络模型的类
       ├─ GAT_LSTM.py   GAT、GAT+LSTM网络模型的类
       ├─ GCN_LSTM.py   GCN、GCN+LSTM网络模型的类
       └─ LSTM.py   时序预测LSTM模型的类
└─ train_and_val.py  训练和验证函数
