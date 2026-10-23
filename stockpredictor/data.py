"""Single-symbol, completed daily bars. No synthetic sentiment or macro inputs."""

import numpy as np
import pandas as pd

OHLCV = ("Open", "High", "Low", "Close", "Volume")


def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Reject ambiguous/missing bars instead of silently changing holding periods."""
    if df.empty or isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected nonempty, single-symbol OHLCV columns")
    if not df.columns.is_unique or not set(OHLCV).issubset(df.columns):
        raise ValueError("Expected unique Open, High, Low, Close, Volume columns")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex of exchange session dates")
    if (df.index.hasnans or not df.index.is_unique
            or not df.index.is_monotonic_increasing):
        raise ValueError("Session dates must be unique and increasing")
    if df.index.tz is not None or not df.index.equals(df.index.normalize()):
        raise ValueError("Use timezone-naive, midnight exchange session dates")
    bars = df.loc[:, list(OHLCV)].astype(float).copy()
    if not np.isfinite(bars.to_numpy()).all():
        raise ValueError("Missing/nonfinite OHLCV; repair the source, do not backfill")
    if (bars[list(OHLCV[:4])] <= 0).any().any() or (bars.Volume < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    if ((bars.High < bars[["Open", "Close", "Low"]].max(axis=1)).any()
            or (bars.Low > bars[["Open", "Close", "High"]].min(axis=1)).any()):
        raise ValueError("Inconsistent OHLC prices")
    return bars


def normalize_download(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Support flat columns and either yfinance MultiIndex orientation."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels != 2:
            raise ValueError("Unsupported download columns")
        field_levels = [i for i in range(2)
                        if set(OHLCV).issubset(df.columns.get_level_values(i))]
        if len(field_levels) != 1:
            raise ValueError("Cannot identify OHLCV level")
        symbol_level = 1 - field_levels[0]
        symbols = df.columns.get_level_values(symbol_level).unique()
        if len(symbols) != 1 or str(symbols[0]).upper() != ticker.upper():
            raise ValueError("Expected exactly the requested ticker")
        df.columns = df.columns.droplevel(symbol_level)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return validate_prices(df)


def download_price_data(ticker: str, *, end: str, years: int = 5) -> pd.DataFrame:
    """Download bars strictly before end (an explicit YYYY-MM-DD cutoff).

    The caller must choose end so every returned session has completed. These
    adjusted research bars are not a point-in-time institutional price archive.
    """
    if years < 1:
        raise ValueError("years must be positive")
    import yfinance as yf

    cutoff = pd.Timestamp(end)
    if pd.isna(cutoff) or cutoff.tz is not None or cutoff != cutoff.normalize():
        raise ValueError("end must be a timezone-naive session date")
    start = cutoff - pd.DateOffset(years=years)
    df = yf.download(ticker, start=start.date().isoformat(),
                     end=cutoff.date().isoformat(), auto_adjust=True,
                     progress=False, multi_level_index=False)
    if df is None or df.empty:
        raise ValueError(f"No data found for {ticker}")
    df = normalize_download(df, ticker)
    if (df.index >= cutoff).any():
        raise ValueError("Provider returned bars at/after the exclusive cutoff")
    return df


def prepare_dataset(ticker: str, *, end: str, years: int = 5) -> pd.DataFrame:
    """Price-only research dataset. News and macro integrations are absent."""
    return download_price_data(ticker, end=end, years=years)
