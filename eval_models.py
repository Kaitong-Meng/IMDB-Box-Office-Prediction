import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import models


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def evaluate_models(models, x_data, y_data, eval_metric='mse'):
    results = []

    for model_name, model_instance in models.items():
        start_time = time.time()

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

        print(f"Training {model_name} model...")

        # Handling specific model instantiation or setup
        if model_name == 'HistGradientBoostingModel':
            model = model_instance(loss="absolute_error")
        elif model_name == 'AttentionModel':
            model = model_instance(x_data.shape[1])
        elif model_name == 'NeuralNetworkModel':
            model = model_instance()  # Instantiate the model
            model.build_model(input_dim=x_train.shape[1])  # Customize the NeuralNetworkModel
        else:
            model = model_instance()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        if eval_metric == 'mse':
            eval_result = np.mean((y_pred - y_test) ** 2)
        elif eval_metric == 'rmse':
            eval_result = np.sqrt(np.mean((y_pred - y_test) ** 2))
        elif eval_metric == 'mae':
            eval_result = np.mean(np.abs(y_pred - y_test))
        elif eval_metric == 'rmsle':
            eval_result = rmsle(y_test, y_pred)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"{model_name} trained. Evaluation metric: {eval_metric}, Evaluation: {eval_result}, "
              f"Execution Time: {execution_time:.4f} seconds")

        results.append({
            'Model': model_name,
            'Eval Metric': eval_metric,
            'Evaluation': eval_result,
            'Execution Time (s)': execution_time
        })

    return results


if __name__ == "__main__":
    models = {
        'LinearRegressionModel': models.LinearRegressionModel,
        'DecisionTreeModel': models.DecisionTreeModel,
        'RandomForestModel': models.RandomForestModel,
        'GradientBoostingModel': models.GradientBoostingModel,
        'AdaBoostModel': models.AdaBoostModel,
        'LassoModel': models.LassoModel,
        'RidgeModel': models.RidgeModel,
        'NeuralNetworkModel': models.NeuralNetworkModel,
        'SVMRegressor': models.SVMRegressor,
        'KNNModel': models.KNNModel,
        'StackingModel': models.StackingModel,
        'AttentionModel': models.AttentionModel,
        'XGBoostModel': models.XGBoostModel,
        'LightGBMModel': models.LightGBMModel,
        'CatBoostModel': models.CatBoostModel,
        'HistGradientBoostingModel': models.HistGradientBoostingModel
    }

    eval_metrics = ['mse', 'rmse', 'mae', 'rmsle']

    data = pd.read_csv("./tmdb-box-office-prediction/data.csv")
    x = data.drop(['id', 'revenue'], axis=1)
    y = np.log1p(data['revenue'])

    all_results = []
    for metric in eval_metrics:
        results = evaluate_models(models, x, y, eval_metric=metric)
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv('model_evaluation_results.csv', index=False)
