import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import Lasso


def feature_selection_pipeline(x, y, feature_selection_method='mutual_info', threshold=0.3):
    if feature_selection_method == 'mutual_info':
        # 计算特征与标签之间的互信息
        mutual_info = mutual_info_regression(x, y)

        # 创建一个包含特征名和对应互信息的 DataFrame
        mi_scores = pd.Series(mutual_info, index=x.columns)
        mi_scores = mi_scores.sort_values(ascending=False)

        # 选择互信息较大的特征（例如，选择前10个特征）
        selected_features = mi_scores.head(10).index.tolist()

        # 更新特征集合为选择的特征集
        x_selected = x[selected_features]
    elif feature_selection_method == 'rfe':
        # 使用递归特征消除
        estimator = Lasso()
        rfe = RFE(estimator, n_features_to_select=10)
        rfe.fit(x, y)

        # 选择被选为重要特征的列
        selected_features = x.columns[rfe.support_]

        # 更新特征集合为选择的特征集
        x_selected = x[selected_features]
    elif feature_selection_method == 'lasso':
        # 使用LASSO回归进行特征选择
        lasso = Lasso(alpha=0.1)  # 可以根据需要调整 alpha 参数
        lasso.fit(x, y)

        # 获取系数不为零的特征
        selected_features = x.columns[lasso.coef_ != 0]

        # 更新特征集合为选择的特征集
        x_selected = x[selected_features]
    else:
        # 其他特征选择方法的处理
        # TODO: 添加其他特征选择方法的处理

        # 这里为了示例直接使用了mutual_info的处理方式
        mutual_info = mutual_info_regression(x, y)
        mi_scores = pd.Series(mutual_info, index=x.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        selected_features = mi_scores.head(10).index.tolist()
        x_selected = x[selected_features]

    # 设定阈值限制特征之间的互信息
    mi_between_features = x_selected.corr().abs()
    np.fill_diagonal(mi_between_features.values, 0)

    # 根据阈值筛选特征
    selected_features_filtered = [x_selected.columns[0]]
    for feature in x_selected.columns[1:]:
        correlated = mi_between_features[feature][selected_features_filtered].max()
        if correlated < threshold:
            selected_features_filtered.append(feature)

    # 更新特征集合为选择的特征集
    x_selected_filtered = x_selected[selected_features_filtered]

    return x_selected_filtered


if __name__ == "__main__":
    data = pd.read_csv("./tmdb-box-office-prediction/data.csv")
    x = data.drop(['id', 'revenue'], axis=1)
    y = np.log1p(data['revenue'])

    # 使用 RFE 方法进行特征选择
    selected_features_rfe = feature_selection_pipeline(x, y, feature_selection_method='rfe')

    # 使用 LASSO 方法进行特征选择
    selected_features_lasso = feature_selection_pipeline(x, y, feature_selection_method='lasso')

    # 存储新选择的特征数据到CSV文件
    selected_features_rfe_filename = "./tmdb-box-office-prediction/selected_features_rfe.csv"
    selected_features_rfe.to_csv(selected_features_rfe_filename, index=False)

    selected_features_lasso_filename = "./tmdb-box-office-prediction/selected_features_lasso.csv"
    selected_features_lasso.to_csv(selected_features_lasso_filename, index=False)
