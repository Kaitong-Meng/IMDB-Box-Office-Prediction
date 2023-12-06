# from sklearn.model_selection import KFold, train_test_split
# import numpy as np
# import timeit
#
#
# class Trainer:
#     def __init__(self, model):
#         self.model = model
#
#     def rmsle(self, y_true, y_pred):
#         return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
#
#     def train(self, x, y, eval_metric='mse', test_size=None, k_folds=None):
#         if k_folds is not None:
#             kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
#
#             # 用于存储每个折叠的评估结果
#             eval_results = []
#
#             def fold_evaluation(train_index, test_index):
#                 x_train_fold, x_val_fold = x.iloc[train_index], x.iloc[test_index]
#                 y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[test_index]
#
#                 start_time = timeit.default_timer()
#
#                 self.model.fit(x_train_fold, y_train_fold)
#                 y_pred = self.model.predict(x_val_fold)
#
#                 # 计算评估指标
#                 if eval_metric == 'mse':
#                     eval_result = np.mean((y_pred - y_val_fold) ** 2)
#                 elif eval_metric == 'rmse':
#                     eval_result = np.sqrt(np.mean((y_pred - y_val_fold) ** 2))
#                 elif eval_metric == 'mae':
#                     eval_result = np.mean(np.abs(y_pred - y_val_fold))
#                 elif eval_metric == 'rmsle':
#                     eval_result = self.rmsle(y_val_fold, y_pred)
#                 else:
#                     raise ValueError("Invalid evaluation metric. Choose among 'mse', 'rmse', 'mae'.")
#
#                 elapsed_time = timeit.default_timer() - start_time
#                 eval_results.append((eval_result, elapsed_time))
#
#             # 进行 k 折交叉验证
#             for train_index, test_index in kf.split(x):
#                 fold_evaluation(train_index, test_index)
#
#             # 计算平均评估结果和平均时间
#             avg_eval_result = np.mean([res[0] for res in eval_results])
#             avg_time = np.mean([res[1] for res in eval_results])
#             print(f"Average {eval_metric.capitalize()}: ", avg_eval_result)
#             print(f"Average Time: {avg_time:.6f} seconds")
#
#         elif test_size is not None:
#             x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
#
#             start_time = timeit.default_timer()
#
#             self.model.fit(x_train, y_train)
#             y_pred = self.model.predict(x_test)
#
#             # 计算评估指标
#             if eval_metric == 'mse':
#                 eval_result = np.mean((y_pred - y_test) ** 2)
#             elif eval_metric == 'rmse':
#                 eval_result = np.sqrt(np.mean((y_pred - y_test) ** 2))
#             elif eval_metric == 'mae':
#                 eval_result = np.mean(np.abs(y_pred - y_test))
#             elif eval_metric == 'rmsle':
#                 eval_result = self.rmsle(y_test, y_pred)
#             else:
#                 raise ValueError("Invalid evaluation metric. Choose among 'mse', 'rmse', 'mae'.")
#
#             elapsed_time = timeit.default_timer() - start_time
#
#             print(f"{eval_metric.capitalize()}: ", eval_result)
#             print(f"Time: {elapsed_time:.6f} seconds")
#
#         else:
#             raise ValueError("Please specify either 'test_size' for simple train-test split or 'k_folds' for k-fold cross-validation.")


from sklearn.model_selection import KFold, train_test_split
import numpy as np
import timeit


class Trainer:
    def __init__(self, model):
        self.model = model
        self.eval_result = None  # To store the evaluation result
        self.elapsed_time = None  # To store the elapsed time
        self.avg_eval_result = None  # To store the average evaluation result (for cross-validation)
        self.avg_time = None  # To store the average time (for cross-validation)

    def rmsle(self, y_true, y_pred):
        return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

    def train(self, x, y, eval_metric='mse', test_size=None, k_folds=None):
        if k_folds is not None:
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

            # Used to store the evaluation results for each fold
            eval_results = []

            def fold_evaluation(train_index, test_index):
                x_train_fold, x_val_fold = x.iloc[train_index], x.iloc[test_index]
                y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[test_index]

                start_time = timeit.default_timer()

                self.model.fit(x_train_fold, y_train_fold)
                y_pred = self.model.predict(x_val_fold)

                # Calculate the evaluation metric
                if eval_metric == 'mse':
                    eval_result = np.mean((y_pred - y_val_fold) ** 2)
                elif eval_metric == 'rmse':
                    eval_result = np.sqrt(np.mean((y_pred - y_val_fold) ** 2))
                elif eval_metric == 'mae':
                    eval_result = np.mean(np.abs(y_pred - y_val_fold))
                elif eval_metric == 'rmsle':
                    eval_result = self.rmsle(y_val_fold, y_pred)
                else:
                    raise ValueError("Invalid evaluation metric. Choose among 'mse', 'rmse', 'mae'.")

                elapsed_time = timeit.default_timer() - start_time
                eval_results.append((eval_result, elapsed_time))

            # Perform k-fold cross-validation
            for train_index, test_index in kf.split(x):
                fold_evaluation(train_index, test_index)

            # Calculate the average evaluation result and average time
            self.avg_eval_result = np.mean([res[0] for res in eval_results])
            self.avg_time = np.mean([res[1] for res in eval_results])
            print(f"Average {eval_metric.capitalize()}: ", self.avg_eval_result)
            print(f"Average Time: {self.avg_time:.6f} seconds")

        elif test_size is not None:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

            start_time = timeit.default_timer()

            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)

            # Calculate the evaluation metric
            if eval_metric == 'mse':
                eval_result = np.mean((y_pred - y_test) ** 2)
            elif eval_metric == 'rmse':
                eval_result = np.sqrt(np.mean((y_pred - y_test) ** 2))
            elif eval_metric == 'mae':
                eval_result = np.mean(np.abs(y_pred - y_test))
            elif eval_metric == 'rmsle':
                eval_result = self.rmsle(y_test, y_pred)
            else:
                raise ValueError("Invalid evaluation metric. Choose among 'mse', 'rmse', 'mae'.")

            elapsed_time = timeit.default_timer() - start_time

            self.eval_result = eval_result
            self.elapsed_time = elapsed_time

            print(f"{eval_metric.capitalize()}: ", self.eval_result)
            print(f"Time: {self.elapsed_time:.6f} seconds")

        else:
            raise ValueError("Please specify either 'test_size' for simple train-test split or 'k_folds' for k-fold cross-validation.")
