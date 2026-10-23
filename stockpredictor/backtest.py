"""Walk-forward prediction and a separate, single-position execution simulator."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .data import prepare_dataset, validate_prices
from .features import (FEATURE_COLUMNS, TARGET, features_after_close,
                       make_samples, sequence_inputs)
from .models import make_model


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame
    metrics: dict


def _validate_config(window, min_train, refit_every, model_type, lookback, epochs):
    for name, value in (("window", window), ("min_train", min_train),
                        ("refit_every", refit_every), ("lookback", lookback),
                        ("epochs", epochs)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if min_train > window or min_train < 2:
        raise ValueError("Require 2 <= min_train <= window")
    if model_type not in ("lightgbm", "ridge", "lstm"):
        raise ValueError(f"Unknown model: {model_type}")
    if model_type == "lstm" and lookback < 2:
        raise ValueError("LSTM lookback must be at least 2")


def _inputs(samples, model_type, lookback):
    X = sequence_inputs(samples, lookback if model_type == "lstm" else 1)
    valid = np.isfinite(X).reshape(len(X), -1).all(axis=1)
    return X, valid


def _training_indices(samples, valid, cutoff, window):
    # A chronological split alone is insufficient for a multi-session label.
    # Include only outcomes fully realized by the previous completed close.
    eligible = (valid & samples[TARGET].notna().to_numpy()
                & (samples.label_end <= cutoff).to_numpy())
    return np.flatnonzero(eligible)[-window:]


def walk_forward(
    prices: pd.DataFrame, *, horizon: int = 1, window: int = 252,
    min_train: int = 126, refit_every: int = 1, model_type: str = "lightgbm",
    lookback: int = 20, epochs: int = 10, cost_bps: float = 10.0,
    threshold: float = 0.0, estimator_factory: Callable | None = None,
) -> BacktestResult:
    """Predict each entry session using only prior completed information.

    refit_every controls retraining frequency, not the target horizon. Even
    between refits, each day's prediction uses that day's newly available
    lagged inputs. It is NOT a batch forecast issued at the first test date.
    Factory estimators must be fresh, unfitted, and use only supplied X/y.
    """
    _validate_config(window, min_train, refit_every, model_type, lookback, epochs)
    _validate_costs(cost_bps, threshold)
    bars = validate_prices(prices)
    samples = make_samples(bars, horizon)
    X, valid = _inputs(samples, model_type, lookback)
    factory = estimator_factory or (lambda: make_model(model_type, epochs=epochs))
    rows = []
    model = None
    for pos in np.flatnonzero(valid & samples[TARGET].notna().to_numpy()):
        row = samples.iloc[pos]
        cutoff = row.feature_asof
        if model is None or len(rows) % refit_every == 0:
            train = _training_indices(samples, valid, cutoff, window)
            if len(train) < min_train:
                continue
            y = samples[TARGET].iloc[train].to_numpy()
            model = factory()
            model.fit(X[train].copy(), y.copy())
            mean_prediction = float(y.mean())
            audit = {
                "fit_asof": cutoff, "train_rows": len(train),
                "train_start": samples.index[train[0]],
                "train_end": samples.index[train[-1]],
                "train_max_label_end": samples.label_end.iloc[train].max(),
            }
        pred = np.asarray(model.predict(X[pos:pos + 1])).reshape(-1)
        if pred.size != 1 or not np.isfinite(pred[0]):
            raise ValueError("Estimator must return one finite prediction per sample")
        rows.append({
            "entry_date": samples.index[pos], "feature_asof": cutoff,
            "exit_date": row.label_end, "prediction": float(pred[0]),
            "actual_return": row[TARGET], "mean_prediction": mean_prediction,
            "entry_price": row.entry_price, "exit_price": row.exit_price, **audit,
        })
    if not rows:
        raise ValueError("Insufficient history after feature warmup and label purging")
    predictions = pd.DataFrame(rows).set_index("entry_date")
    trades, equity, trade_metrics = simulate_trades(
        bars, predictions, cost_bps=cost_bps, threshold=threshold)
    actual = predictions.actual_return.to_numpy()
    pred = predictions.prediction.to_numpy()
    metrics = {
        "n_predictions": len(predictions),
        "return_rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "zero_return_rmse": float(np.sqrt(np.mean(actual ** 2))),
        "past_mean_rmse": float(np.sqrt(np.mean(
            (predictions.mean_prediction.to_numpy() - actual) ** 2))),
        "direction_accuracy": float(np.mean(np.sign(pred) == np.sign(actual))),
        **trade_metrics,
    }
    return BacktestResult(predictions, trades, equity, metrics)


def _validate_costs(cost_bps, threshold):
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("cost_bps must be finite and in [0, 10000), per side")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite, nonnegative net return")


def simulate_trades(prices, predictions, *, cost_bps=10.0, threshold=0.0):
    """Long/cash, one position per symbol, all capital, no overlapping reuse.

    A fixed per-side execution cost c buys at Open*(1+c), sells at Close*(1-c).
    Daily equity is marked to close, including intratrade drawdowns. Fractional
    units are a research convention; this is not a whole-share portfolio engine.
    """
    _validate_costs(cost_bps, threshold)
    c = cost_bps / 10000
    start, end = predictions.index[0], predictions.exit_date.max()
    cash, shares, exit_date = 1.0, 0.0, None
    trades, curve = [], []
    current_trade = None
    for date, bar in prices.loc[start:end].iterrows():
        if shares == 0 and date in predictions.index:
            signal = predictions.loc[date]
            expected_net = (1 + signal.prediction) * (1 - c) / (1 + c) - 1
            if expected_net > threshold:
                entry_equity = cash
                shares = cash / (bar.Open * (1 + c))
                cash = 0.0
                exit_date = signal.exit_date
                current_trade = {
                    "entry_date": date, "exit_date": exit_date,
                    "prediction": signal.prediction,
                    "entry_price": bar.Open, "entry_equity": entry_equity,
                }
        if shares > 0 and date == exit_date:
            cash = shares * bar.Close * (1 - c)
            trades.append({
                **current_trade, "exit_price": bar.Close,
                "net_return": cash / current_trade["entry_equity"] - 1,
                "exit_equity": cash,
            })
            shares, exit_date = 0.0, None
        curve.append({"date": date, "equity": cash + shares * bar.Close})
    if shares:
        raise ValueError("Simulation ended with an unclosed position")
    equity = pd.DataFrame(curve).set_index("date")
    peaks = np.maximum.accumulate(np.r_[1.0, equity.equity.to_numpy()])[1:]
    equity["drawdown"] = equity.equity / peaks - 1
    ledger = pd.DataFrame(trades, columns=[
        "entry_date", "exit_date", "prediction", "entry_price", "entry_equity",
        "exit_price", "net_return", "exit_equity",
    ])
    metrics = {
        "n_trades": len(ledger), "strategy_return": float(equity.equity.iloc[-1] - 1),
        "max_drawdown": float(equity.drawdown.min()),
        "mean_net_trade_return": float(ledger.net_return.mean()) if len(ledger) else None,
        "net_win_rate": float((ledger.net_return > 0).mean()) if len(ledger) else None,
        "buy_hold_return": float(
            prices.loc[end, "Close"] * (1 - c) / (prices.loc[start, "Open"] * (1 + c)) - 1),
        "cash_return": 0.0, "cost_bps_per_side": float(cost_bps),
        "net_entry_threshold": float(threshold),
    }
    return ledger, equity, metrics


def forecast_next_open(
    prices: pd.DataFrame, *, horizon: int = 1, window: int = 252,
    min_train: int = 126, model_type: str = "lightgbm", lookback: int = 20,
    epochs: int = 10, estimator_factory: Callable | None = None,
) -> dict:
    """Forecast the next actual session from completed history; no future row needed."""
    _validate_config(window, min_train, 1, model_type, lookback, epochs)
    bars = validate_prices(prices)
    samples = make_samples(bars, horizon)
    X, valid = _inputs(samples, model_type, lookback)
    cutoff = bars.index[-1]
    train = _training_indices(samples, valid, cutoff, window)
    live = sequence_inputs(features_after_close(bars),
                           lookback if model_type == "lstm" else 1)[-1:]
    if len(train) < min_train or not np.isfinite(live).all():
        raise ValueError("Insufficient completed history for a next-open forecast")
    model = (estimator_factory or (lambda: make_model(model_type, epochs=epochs)))()
    model.fit(X[train].copy(), samples[TARGET].iloc[train].to_numpy().copy())
    prediction = np.asarray(model.predict(live)).reshape(-1)
    if prediction.size != 1 or not np.isfinite(prediction[0]):
        raise ValueError("Estimator must return one finite prediction")
    return {"asof_close": cutoff.date().isoformat(), "horizon_sessions": horizon,
            "predicted_gross_return": float(prediction[0]), "train_rows": len(train)}


def rolling_window_backtest(ticker: str, *, end: str, **kwargs) -> BacktestResult:
    """Download an explicit historical snapshot, then evaluate it."""
    return walk_forward(prepare_dataset(ticker, end=end), **kwargs)
