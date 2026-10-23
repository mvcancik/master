"""Fit only the explicitly supplied training rows; evaluation lives elsewhere."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class LSTMRegressor:
    """Real multi-session inputs, with feature/target scaling fitted on train only."""

    def __init__(self, epochs: int = 10, seed: int = 17):
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        import tensorflow as tf

        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 3 or X.shape[1] < 2:
            raise ValueError("LSTM requires (samples, lookback>=2, features)")
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.seed)
        self.input_shape_ = X.shape[1:]
        self.scaler_ = StandardScaler().fit(X.reshape(-1, X.shape[-1]))
        self.y_mean_, self.y_scale_ = float(y.mean()), float(y.std()) or 1.0
        self.model_ = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.input_shape_),
            tf.keras.layers.LSTM(16), tf.keras.layers.Dense(1),
        ])
        self.model_.compile(optimizer="adam", loss="mse")
        self.model_.fit(self._transform(X), (y - self.y_mean_) / self.y_scale_,
                        epochs=self.epochs, batch_size=32, shuffle=False, verbose=0)
        return self

    def _transform(self, X):
        X = np.asarray(X)
        if X.ndim != 3 or X.shape[1:] != self.input_shape_:
            raise ValueError("Prediction sequences must match the training shape")
        return self.scaler_.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    def predict(self, X):
        # Direct inference avoids creating another tf.data thread pool per fold.
        pred = self.model_(self._transform(X), training=False).numpy().reshape(-1)
        return pred * self.y_scale_ + self.y_mean_


def make_model(model_type: str = "lightgbm", *, epochs: int = 10):
    """Fixed parameters: tune separately on past-only inner walk-forward splits."""
    if model_type == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if model_type == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression", n_estimators=100, learning_rate=0.03,
            num_leaves=7, min_child_samples=20, reg_lambda=1.0,
            random_state=17, n_jobs=1, verbosity=-1,
            deterministic=True, force_col_wise=True,
        )
    if model_type == "lstm":
        return LSTMRegressor(epochs=epochs)
    raise ValueError(f"Unknown model: {model_type}")
