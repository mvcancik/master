import streamlit as st
import pandas as pd

from .data import prepare_dataset
from .features import add_technical_features, add_calendar_features
from .models import train_lightgbm


def main():
    st.title("Stock Predictor Dashboard")
    ticker = st.text_input("Ticker", "AAPL")
    if st.button("Train and Predict"):
        df = prepare_dataset(ticker)
        df = add_technical_features(df)
        df = add_calendar_features(df)
        model, rmse = train_lightgbm(df)
        st.write(f"Model trained. RMSE={rmse:.4f}")
        st.line_chart(df["Close"])


if __name__ == "__main__":
    main()
