# -*- coding: utf-8 -*-
"""
@Time    : 2017/6/12 0:58
@Author  : Elvis
"""
"""
 plot.py
  
"""
import numpy as np
import matplotlib.pyplot as plt

k1 = 0.2
k2 = 0.5
X = np.arange(.0, 1.0, 0.001)
Y = []
for x in X:
    if x <= k1:
        Y.append(.0)
    elif k1 < x < k2:
        Y.append(0.9 * (x - k1) / (k2 - k1) + 0.1)
    else:
        Y.append(1)


fig = plt.figure(figsize=(8, 4))  # Create a `figure' instance
ax = fig.add_subplot(111)  # Create a `axes' instance in the figure
ax.plot(X, Y)  # Create a Line2D instance in the axes
plt.ylim(-0.2, 1.2)
# plt.xlabel("预测涨幅")
# plt.ylabel("持仓量")
fig.show()
fig.savefig("test.pdf")
