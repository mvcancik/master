import pandas as pd
import numpy as np


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to the dataframe."""
    df = df.copy()
    df["ma_50"] = df["Close"].rolling(window=50).mean()
    df["ma_200"] = df["Close"].rolling(window=200).mean()
    df["rsi"] = compute_rsi(df["Close"], window=14)
    return df


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window, min_periods=window).mean()
    ma_down = down.rolling(window=window, min_periods=window).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    return df
