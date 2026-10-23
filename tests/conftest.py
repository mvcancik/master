import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices():
    rng = np.random.default_rng(231)
    n = 380
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))
    opening = close * np.exp(rng.normal(0, 0.004, n))
    return pd.DataFrame({
        "Open": opening, "Close": close,
        "High": np.maximum(opening, close) * 1.01,
        "Low": np.minimum(opening, close) * 0.99,
        "Volume": rng.integers(100_000, 500_000, n).astype(float),
    }, index=pd.bdate_range("2020-01-01", periods=n))


class MeanRecorder:
    def __init__(self):
        self.fit_calls = []
        self.predict_calls = []

    def fit(self, X, y):
        self.fit_calls.append((X.copy(), y.copy()))
        self.mean = float(y.mean())
        return self

    def predict(self, X):
        self.predict_calls.append(X.copy())
        return np.repeat(self.mean, len(X))
