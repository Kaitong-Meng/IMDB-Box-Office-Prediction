from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from keras.models import Model, Sequential
from keras.layers import Dense, Input, dot, Activation
from keras import backend as K
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)


class DecisionTreeModel:
    def __init__(self):
        self.model = DecisionTreeRegressor()

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)


class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor()

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)


class GradientBoostingModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)


class AdaBoostModel:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.model = AdaBoostRegressor(
            base_estimator=DecisionTreeRegressor(max_depth=4),  # 可以更改基本的弱分类器
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)


class LassoModel:
    def __init__(self):
        self.model = Lasso()

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)


class RidgeModel:
    def __init__(self, alpha=0.5):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, x_train, y_train):
        # 数据标准化
        x_train_scaled = self.scaler.fit_transform(x_train)

        # 调整正则化参数并训练模型
        self.model.fit(x_train_scaled, y_train)

    def predict(self, x_test):
        # 数据标准化
        x_test_scaled = self.scaler.transform(x_test)

        # 预测模型
        return self.model.predict(x_test_scaled)


class NeuralNetworkModel:
    def __init__(self):
        self.model = Sequential()

    def build_model(self, input_dim):
        # 添加神经网络层
        self.model.add(Dense(units=256, activation='tanh', input_dim=input_dim))
        self.model.add(Dense(units=128, activation='tanh', input_dim=256))
        self.model.add(Dense(units=64, activation='tanh', input_dim=128))
        self.model.add(Dense(units=32, activation='tanh', input_dim=64))
        self.model.add(Dense(units=16, activation='tanh', input_dim=32))
        self.model.add(Dense(units=8, activation='tanh', input_dim=16))
        self.model.add(Dense(units=1, input_dim=8))  # 输出层，单个输出节点

        # 编译模型
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def fit(self, x_train, y_train, epochs=10, batch_size=32):
        # 训练模型
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, x_test):
        # 预测模型
        return self.model.predict(x_test).flatten()


class SVMRegressor:
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        self.scaler = StandardScaler()
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    def fit(self, x_train, y_train):
        # 数据标准化
        x_train_scaled = self.scaler.fit_transform(x_train)

        # 训练 SVR 模型
        self.model.fit(x_train_scaled, y_train)

    def predict(self, x_test):
        # 数据标准化
        x_test_scaled = self.scaler.transform(x_test)

        # 预测
        return self.model.predict(x_test_scaled)


class KNNModel:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)


class StackingModel:
    def __init__(self):
        # 定义基本模型
        base_models = [
            ('decision_tree', DecisionTreeRegressor()),
            ('random_forest', RandomForestRegressor())
            # 可以根据需要添加更多的基本模型
        ]

        # 定义元模型
        meta_model = LinearRegression()

        # 创建 StackingRegressor 模型
        self.model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)


class AttentionModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self.build_model()

    def build_model(self):
        inputs = Input(shape=(self.input_dim,))

        attention_probs1 = Dense(self.input_dim, activation='softmax')(inputs)
        attention_mul1 = dot([inputs, attention_probs1], axes=1)
        attention_mul1 = Dense(self.input_dim)(attention_mul1)
        attention_mul1 = Activation('relu')(attention_mul1)

        attention_probs2 = Dense(self.input_dim, activation='softmax')(attention_mul1)
        attention_mul2 = dot([attention_mul1, attention_probs2], axes=1)
        attention_mul2 = Dense(self.input_dim)(attention_mul2)
        attention_mul2 = Activation('relu')(attention_mul2)

        output = Dense(1)(attention_mul2)

        model = Model(inputs=inputs, outputs=output)
        return model

    def fit(self, x_train, y_train, epochs=10, batch_size=32):
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, x_test):
        return self.model.predict(x_test).reshape(-1)

    def get_attention_weights(self, x):
        layer_outputs = [layer.output for layer in self.model.layers[1:5]]  # 提取前四个层的输出
        get_attention = K.function([self.model.layers[0].input], layer_outputs)
        return get_attention([x])


class XGBoostModel:
    def __init__(self):
        self.model = XGBRegressor()

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)


class LightGBMModel:
    def __init__(self):
        self.model = LGBMRegressor()

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)


class CatBoostModel:
    def __init__(self):
        self.model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function='RMSE',
            eval_metric='RMSE')

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train, verbose=100)

    def predict(self, x_test):
        return self.model.predict(x_test)


class HistGradientBoostingModel:
    def __init__(self, loss='squared_error'):
        self.model = HistGradientBoostingRegressor(loss=loss, learning_rate=0.1, max_iter=100)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
