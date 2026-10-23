"""Optional dashboard: delegates to the same validated evaluation path as the CLI."""

import streamlit as st

from stockpredictor.backtest import rolling_window_backtest


def main():
    st.title("Short-horizon stock research")
    st.caption("Price-only, next-open entries. No news, intraday stops or live execution.")
    ticker = st.text_input("Ticker", "AAPL")
    end = st.text_input("Exclusive data cutoff (YYYY-MM-DD)", "")
    horizon = st.selectbox("Holding sessions (entry open to final close)", [1, 2, 3])
    cost = st.number_input("Assumed cost per side (basis points)", min_value=0.0, value=10.0)
    if st.button("Run walk-forward evaluation"):
        try:
            result = rolling_window_backtest(ticker, end=end, horizon=horizon, cost_bps=cost)
        except (ValueError, ImportError) as exc:
            st.error(str(exc))
            return
        st.json(result.metrics)
        st.line_chart(result.equity["equity"])
        st.dataframe(result.predictions)


if __name__ == "__main__":
    main()
