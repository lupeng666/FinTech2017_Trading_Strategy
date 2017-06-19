# -*- coding: utf-8 -*-
"""
@Time    : 2017/6/19 16:19
@Author  : Elvis
"""
"""
 corr_feat.py
  
"""
import pandas as pd
import matplotlib.pyplot as plt

price = pd.read_csv("../LSTM_data/price.csv")
refer = pd.read_csv("../LSTM_data/refer.csv")

feat = pd.merge(price, refer, on="Time")
feat = feat.drop('Time', 1)
feat.head()
feat.hist()
from pandas.tools.plotting import scatter_matrix

scatter_matrix(feat, figsize=(10, 10))
plt.show()
plt.savefig("../LSTM_output/feat_plot.png")

feat_corr = feat.corr()  # 计算变量之间的相关系数矩阵
import seaborn as sn
import numpy as np
plt.figure(figsize=(16, 12))
sn.heatmap(feat_corr, annot=True)
plt.xticks(rotation=30)
plt.yticks(rotation=0)
plt.savefig("../LSTM_output/feat_corr.pdf")

feat_ext = feat_corr.columns[abs(feat_corr["price"]) > 0.65]
# ['price', 'TIPS-5Y', 'TIPS-10Y', 'TIPS-20Y', 'TIPS-LONG',
#        'UST BILL 10-Y RETURN', 'LIBOR-OVERNIGHT', 'SPDR:t', 'USD/CNY']
len(feat_ext) # 9
feat_use = feat[feat_ext]
feat_use = feat_use.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
feat_use.describe()


