import pandas as pd
import numpy as np
import models
from trainer import Trainer  # 导入 Trainer 类


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置 TensorFlow 日志级别为仅显示错误信息


data = pd.read_csv("./tmdb-box-office-prediction/data.csv")
x = data.drop(['id', 'revenue'], axis=1)
y = np.log1p(data['revenue'])


# 其余模型无需在定义时显式指明参数
model = models.XGBoostModel()
# 直方图梯度提升模型需要本行
# 参数可选：absolute_error, poisson, quantile, squared_error, gamma
# model = models.HistGradientBoostingModel(loss="absolute_error")
# 注意力模型需要本行
# model = models.AttentionModel(x.shape[1])
# 神经网络需要本行
# model.build_model(input_dim=x_train.shape[1])

trainer = Trainer(model)


# # 使用简单的数据划分方式
trainer.train(x, y, eval_metric='rmsle', test_size=0.2)

# 或者使用 k 折交叉验证
trainer.train(x, y, eval_metric='rmsle', k_folds=5)
