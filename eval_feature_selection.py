import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import models
from eval_models import evaluate_models

# Define the model names and instances as you previously provided
models = {
    'LassoModel': models.LassoModel,
    'CatBoostModel': models.CatBoostModel,
}

# Define evaluation metrics
eval_metrics = ['mse', 'rmse', 'mae', 'rmsle']


# Function to evaluate models on different datasets
def evaluate_on_datasets(model_dict, datasets):
    all_results = []

    for dataset_path in datasets:
        x = pd.read_csv(dataset_path)
        y = np.log1p(pd.read_csv("./tmdb-box-office-prediction/data.csv")["revenue"])

        dataset_results = {'Dataset': dataset_path}
        for metric in eval_metrics:
            results = evaluate_models(model_dict, x, y, eval_metric=metric)
            for result in results:
                key = f"{result['Model']}_{result['Eval Metric']}"
                dataset_results[key] = result['Evaluation']

        all_results.append(dataset_results)

    return all_results


# Rest of the code remains the same until this point

# Example list of dataset addresses (replace this with your dataset addresses)
dataset_addresses = [
    "./tmdb-box-office-prediction/selected_features.csv",
    "./tmdb-box-office-prediction/selected_features_lasso.csv",
    "./tmdb-box-office-prediction/selected_features_rfe.csv",
]

# Get evaluation results on different datasets
results_on_datasets = evaluate_on_datasets(models, dataset_addresses)

# Create DataFrame and save the results
results_df = pd.DataFrame(results_on_datasets)
results_df.to_csv('model_evaluation_on_datasets.csv', index=False)
