# LSTM模型代码说明 #

## 代码运行说明 ##
剔除周末数据，将数据放在LSTM_data文件夹下 (price.csv为价格数据，refer.csv为附件2中各种参考指标数据) ，切换工作路径至LSTM_src下，运行python run.py即可。


## 代码模块说明 ##
- run.py控制程序流程。
- lstm.py
	- normalize 函数用于数据标准化。
	- load_feat 函数构造特征，划分训练集测试集。
	- build_model 函数搭建网络结构。
	- predict_and_postprocess 函数进行预测和对结果进行后处理。
 
- plot.py
	- plot_results 函数涨幅预测结果散点图。
	- plot_accumulate_precision 函数计算累计涨跌预测准确率，并可视化。

## 网络结构 ##


## 模型结果 ##
1. 涨幅预测情况  
![](http://i.imgur.com/LV04ZTg.png)
2. 涨跌预测累计准确率   
![](http://i.imgur.com/rqJdcEZ.png)