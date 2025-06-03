import pandas as pd
import yfinance as yf

def download_price_data(ticker: str, years: int = 5) -> pd.DataFrame:
    """Download historical OHLCV data for the given ticker."""
    period = f"{years}y"
    df = yf.download(ticker, period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    return df


def fetch_macro_data() -> pd.DataFrame:
    """Placeholder for macroeconomic data retrieval."""
    return pd.DataFrame()


def fetch_news_sentiment(ticker: str) -> pd.DataFrame:
    """Placeholder for news and sentiment data retrieval."""
    return pd.DataFrame()


def prepare_dataset(ticker: str) -> pd.DataFrame:
    """Combine price, macro, and sentiment data."""
    price_df = download_price_data(ticker)
    macro_df = fetch_macro_data()
    sentiment_df = fetch_news_sentiment(ticker)

    df = price_df.copy()
    if not macro_df.empty:
        df = df.join(macro_df, how="left")
    if not sentiment_df.empty:
        df = df.join(sentiment_df, how="left")
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df
