import pandas as pd

# 读取原始数据集
data = pd.read_csv('model_evaluation_results.csv')

# 重排数据集
reshaped_data = data.pivot(index='Model', columns='Eval Metric', values='Evaluation')

# 获取执行时间信息
execution_times = data.pivot(index='Model', columns='Eval Metric', values='Execution Time (s)')

# 将执行时间添加到每个指标的右侧
for column in reshaped_data.columns:
    reshaped_data[column, 'Execution Time (s)'] = execution_times[column]

# 保存重排后的数据到CSV文件
reshaped_data.to_csv('reshaped_data.csv')
