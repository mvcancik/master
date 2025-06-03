import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

from .data import prepare_dataset
from .features import add_technical_features, add_calendar_features
from .models import train_lightgbm, train_lstm


def rolling_window_backtest(ticker: str, model_type: str = "lightgbm", window: int = 252, horizon: int = 5) -> Tuple[pd.DataFrame, float]:
    df = prepare_dataset(ticker)
    df = add_technical_features(df)
    df = add_calendar_features(df)
    predictions = []
    actuals = []

    for start in range(0, len(df) - window - horizon, horizon):
        train_df = df.iloc[start:start + window]
        test_df = df.iloc[start + window:start + window + horizon]

        if model_type == "lightgbm":
            model, _ = train_lightgbm(train_df)
            preds = model.predict(test_df.drop(columns=["Close"]))
        else:
            model, _ = train_lstm(train_df, epochs=5)
            X_test = test_df.drop(columns=["Close"]).values
            X_test = X_test[None, :, :]
            preds = model.predict(X_test).flatten()

        predictions.extend(preds)
        actuals.extend(test_df["Close"].values)

    result_df = pd.DataFrame({"pred": predictions, "actual": actuals})
    rmse = ((result_df["pred"] - result_df["actual"]) ** 2).mean() ** 0.5

    plt.figure(figsize=(10, 4))
    plt.plot(result_df["actual"].values, label="Actual")
    plt.plot(result_df["pred"].values, label="Predicted")
    plt.legend()
    plt.title(f"Rolling backtest for {ticker}")
    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()

    return result_df, rmse
