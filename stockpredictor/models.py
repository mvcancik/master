import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


DEFAULT_TARGET = "Close"


def train_lightgbm(df: pd.DataFrame, target: str = DEFAULT_TARGET):
    df = df.dropna()
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
    }
    model = lgb.train(params, train_data, valid_sets=[test_data])
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return model, rmse


def train_lstm(df: pd.DataFrame, target: str = DEFAULT_TARGET, epochs: int = 10):
    df = df.dropna()
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    X_train = np.expand_dims(X_train.values, axis=1)
    X_test = np.expand_dims(X_test.values, axis=1)

    model = Sequential()
    model.add(LSTM(50, input_shape=(1, X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    preds = model.predict(X_test).flatten()
    rmse = mean_squared_error(y_test, preds, squared=False)
    return model, rmse
