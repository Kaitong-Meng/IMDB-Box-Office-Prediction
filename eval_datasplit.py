import numpy as np
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from trainer import Trainer  # Import your Trainer class
import models

# Dummy functions used for demonstration purposes
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

# Define dummy models (replace these with your actual models)
models = {
    'LassoModel': models.LassoModel,
    'CatBoostModel': models.CatBoostModel,
}

# Generate dummy data (replace this with your actual dataset)
x = pd.read_csv("./tmdb-box-office-prediction/selected_features.csv")
y = np.log1p(pd.read_csv("./tmdb-box-office-prediction/data.csv")["revenue"])

# Function to train and evaluate using Trainer class with direct split
def train_evaluate_direct_split(model_dict, x_data, y_data):
    eval_metrics = ['mse', 'rmse', 'mae', 'rmsle']
    all_results = []

    for model_name, model_instance in model_dict.items():
        trainer = Trainer(model_instance())

        for metric in eval_metrics:
            trainer.train(x_data, y_data, eval_metric=metric, test_size=0.2)

            result = {
                'Model': model_name,
                'Eval Metric': metric,
                'Evaluation': trainer.eval_result,
                'Execution Time (s)': trainer.elapsed_time,
                'Is 10-fold CV': 0  # Indicating direct split
            }
            all_results.append(result)

    return all_results

# Function to train and evaluate using Trainer class with 10-fold cross-validation
def train_evaluate_cross_val(model_dict, x_data, y_data):
    eval_metrics = ['mse', 'rmse', 'mae', 'rmsle']
    all_results = []

    for model_name, model_instance in model_dict.items():
        trainer = Trainer(model_instance())

        for metric in eval_metrics:
            trainer.train(x_data, y_data, eval_metric=metric, k_folds=10)

            result = {
                'Model': model_name,
                'Eval Metric': metric,
                'Evaluation': trainer.avg_eval_result,
                'Execution Time (s)': trainer.avg_time,
                'Is 10-fold CV': 1  # Indicating 10-fold CV
            }
            all_results.append(result)

    return all_results

# Evaluate models with direct split
direct_split_results = train_evaluate_direct_split(models, x, y)

# Evaluate models with 10-fold cross-validation
cross_val_results = train_evaluate_cross_val(models, x, y)

# Combine results
all_results = direct_split_results + cross_val_results

# Create a DataFrame to save the results
results_df = pd.DataFrame(all_results)

# Pivot the DataFrame to have a structure similar to the previous one
pivot_results = results_df.pivot_table(index=['Model', 'Is 10-fold CV'], columns=['Eval Metric'], values='Evaluation').reset_index()

# Save the results to a CSV file
pivot_results.to_csv('model_evaluation_on_datasplit.csv', index=False)
