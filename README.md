# Professional Stock Price Predictor

This project provides a modular framework for building stock price prediction models using machine learning and deep learning. It supports downloading up to five years of historical data, engineering common technical and calendar features, training models, running rolling backtests, and visualising results.

## Features

- Price and volume history via `yfinance`
- Placeholder hooks for macroeconomic and news sentiment data
- Technical and calendar feature engineering
- LightGBM and LSTM models
- Rolling-window backtesting with visualisations
- CLI interface and optional Streamlit dashboard

## Installation

```bash
pip install -r requirements.txt
```

## Examples

Train a LightGBM model:

```bash
python -m stockpredictor.cli train AAPL --model lightgbm
```

Run a rolling backtest using an LSTM:

```bash
python -m stockpredictor.cli backtest AAPL --model lstm
```

Launch the dashboard:

```bash
streamlit run -m stockpredictor.dashboard
```

## Disclaimer

This repository is intended for educational purposes only and should not be used for real investment decisions.
