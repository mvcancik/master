"""Causal features, indexed by the session at whose open a trade can enter."""

import numpy as np
import pandas as pd

from .data import validate_prices

FEATURE_COLUMNS = (
    "return_1", "return_3", "ma_50_gap", "ma_200_gap", "rsi_14",
    "ema_10_gap", "ema_20_gap", "range_pct", "volume_ratio", "volatility_20",
)
TARGET = "forward_return"


def check_horizon(horizon: int) -> None:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in (1, 2, 3):
        raise ValueError("horizon must be 1, 2, or 3 trading sessions")


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Simple rolling RSI (not Wilder smoothing), retaining the PR definition."""
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    down = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rsi = 100 - 100 / (1 + up / down)
    return rsi.mask((up == 0) & (down == 0), 50.0)


def features_after_close(prices: pd.DataFrame) -> pd.DataFrame:
    """Only trailing observations, available once that session is complete.

    These features may be precomputed on a full immutable history: no centered
    rolling windows, backfills, global scaling, or future values are used.
    """
    bars = validate_prices(prices)
    close = bars.Close
    returns = close.pct_change(fill_method=None)
    features = pd.DataFrame(index=bars.index)
    features["return_1"] = returns
    features["return_3"] = close.pct_change(3, fill_method=None)
    for window in (50, 200):
        features[f"ma_{window}_gap"] = close / close.rolling(window).mean() - 1
    features["rsi_14"] = compute_rsi(close) / 100
    for span in (10, 20):
        features[f"ema_{span}_gap"] = close / close.ewm(
            span=span, adjust=False, min_periods=span).mean() - 1
    features["range_pct"] = (bars.High - bars.Low) / close
    features["volume_ratio"] = bars.Volume / bars.Volume.rolling(20).mean()
    features["volatility_20"] = returns.rolling(20).std()
    return features.loc[:, list(FEATURE_COLUMNS)].replace([np.inf, -np.inf], np.nan)


def make_samples(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """At entry session d: X uses <= close[d-1]; y = close[d+h-1]/open[d]-1.

    h=1 exits on the entry day's close; h=3 exits on the third session's close.
    Unknown trailing labels remain NaN. Dates/prices/targets are metadata and
    must never be passed to a model as predictors.
    """
    check_horizon(horizon)
    bars = validate_prices(prices)
    samples = features_after_close(bars).shift(1)
    dates = pd.Series(bars.index, index=bars.index)
    samples["feature_asof"] = dates.shift(1)
    samples["label_end"] = dates.shift(-(horizon - 1))
    samples["entry_price"] = bars.Open
    samples["exit_price"] = bars.Close.shift(-(horizon - 1))
    samples[TARGET] = samples.exit_price / samples.entry_price - 1
    return samples


def sequence_inputs(features: pd.DataFrame, lookback: int = 1) -> np.ndarray:
    """One sequence ending at each row; retain warmup rows as NaN.

    No row compaction: an unavailable feature cannot shorten a sequence or
    change the meaning of a holding period.
    """
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise ValueError("lookback must be a positive integer")
    values = features.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    if lookback == 1:
        return values
    result = np.full((len(values), lookback, values.shape[1]), np.nan)
    for end in range(lookback - 1, len(values)):
        result[end] = values[end - lookback + 1:end + 1]
    return result
