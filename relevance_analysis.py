import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 读取数据集，假设数据集已经包括了 x 和 y
# 这里假设你已经读取了数据集到 x 和 y 变量中，x 是特征，y 是标签
data = pd.read_csv("./tmdb-box-office-prediction/data.csv")
x = data.drop(['id', 'revenue'], axis=1)
y = np.log1p(data['revenue'])


if isinstance(x, pd.DataFrame):
    selected_features = np.random.choice(x.columns, size=10, replace=False)
    x_selected = x[selected_features]

    # 计算特征之间的相关性矩阵
    corr_matrix = x_selected.corr()

    # 绘制热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Randomly Selected Features')
    plt.show()
else:
    print("x should be a DataFrame containing features.")
