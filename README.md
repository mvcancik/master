# Short-horizon stock research

A **price-only research framework** for 1–3 trading-session returns. It is not
a proven stock screener or an automated trading system. The original PRs #1
and #2 were identical and predicted the contemporaneous close; this revision
replaces that task with explicit, executable forward-return timing.

See [the code review and validation report](docs/REVIEW.md) for confirmed
defects, remaining limitations, and the historical sanity check.

## Timing contract

Every historical row is indexed by its **entry session d**:

1. At the close of session d−1 (or before the next open), form the signal.
   Every price/volume feature uses completed bars through d−1. Entry-day open,
   high, low, close and volume are never predictors.
2. Enter at session d's open, with an assumed execution cost.
3. Exit at close[d+h−1], where h is 1, 2 or 3 **trading sessions**.
4. The training target is close[d+h−1] / open[d] − 1, in decimal return units.
   For h=1 this is an intraday open-to-close trade; h=3 includes two overnights.
5. At each model fit, include only training labels whose exit session is at
   or before the last completed close. Exclude every still-unrealized label.

This schedule supports a night-before price screen. It does **not** simulate
open-time gap reactions, hourly decisions, after-hours entries, stop-losses or
profit-taking orders. A predicted return is not a probability or confidence
estimate.

Trailing features may safely be precomputed over a fixed history because each
row is causal. The regression tests verify prefix invariance and future-price
mutation invariance. Scaling is fitted exclusively to each training window.
Only FEATURE_COLUMNS can reach the model; targets, dates and fill prices cannot.

## Install and test

Use Python 3.12. From this repository's root folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Core tests need no market-data network access. Real LSTM tests are skipped unless
the optional TensorFlow dependency is installed:

```bash
python -m pip install -r requirements-lstm.txt
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
```

Ridge is the simple model comparator; LightGBM is the default; LSTM is an
experimental optional comparator, with real 20-session sequences and train-only
feature/target scaling. All hyperparameters are fixed. There is no automatic
tuning or test-driven model selection.

## Run a backtest

Use an explicit **exclusive** cutoff. For example, 2025-06-01 includes completed
sessions through 2025-05-30. The caller is responsible for choosing a cutoff that
does not include a partially completed session.

```bash
python -m stockpredictor.cli backtest AAPL \
  --end 2025-06-01 --model lightgbm --horizon 3 \
  --window 252 --min-train 126 --refit-every 5 \
  --cost-bps 10 --threshold 0 --output results/aapl-h3
```

The downloader requests five years of adjusted daily bars. For reproducible
research, retain a frozen CSV (first column: exchange session date; other
columns: Open, High, Low, Close, Volume) and run:

```bash
python -m stockpredictor.cli backtest AAPL \
  --csv /absolute/path/to/aapl.csv --end 2025-06-01 \
  --model ridge --horizon 3 --refit-every 5 --output results/aapl-ridge-h3
```

No backfilling, imputation or silent removal of missing bars is performed.
Duplicate dates, unsorted dates, nonfinite OHLCV, nonpositive prices and
inconsistent high/low values fail clearly. Entirely missing exchange sessions
cannot be detected without an external exchange calendar; verify source coverage.

window limits the number of eligible training samples; min-train allows the
initial window to grow to that maximum. Set both to 252 for a strictly fixed
252-sample window. Each prediction is issued sequentially with that session's
lagged inputs; refit-every only controls how often model parameters change.
Earlier test outcomes may enter later training windows once they have actually
been realized, as they would in production.

Outputs:

- predictions.csv: dated out-of-sample predictions, realized forward returns,
  past-mean baseline, feature cutoff, model-fit cutoff, training range and
  maximum training label-end date.
- trades.csv: the executed nonoverlapping long trades and net results.
- equity.csv: equity marked at every daily close and daily drawdown.
- metrics.json: return RMSE versus zero-return/past-mean forecasts, direction
  accuracy, trades, net win rate, average net trade, compounded strategy return,
  daily-close maximum drawdown, matched-period buy-and-hold and cash baselines.

Return figures are decimals: 0.01 means 1%. Strategy return is total, not
annualized. No Sharpe ratio or confidence interval is claimed from overlapping
forward labels.

## Execution assumptions

The example cost is **10 basis points per side**, an illustrative assumption
that must be calibrated to actual fills. Buy at Open × (1+c), sell at Close ×
(1−c); enter only when the corresponding predicted net return exceeds threshold.
The simulator allows one long position per symbol, using all modeled capital,
and waits until the following session after an exit before another entry.
Capital is not reused for overlapping daily signals. While in a trade it still
records research predictions, but does not execute them.

Equity and buy-and-hold use the same adjusted-price units. Fractional units,
no taxes, zero cash yield and no liquidity limits are simplifications.
Daily-close drawdown misses intraday losses. Adjusted prices are unsuitable for
exact historical whole-share order sizing. This is **not** a combined portfolio
engine and does not implement an 85% allocation cap, 2–3 simultaneous positions,
whole-share constraints, or stop/limit execution.

## Fit a next-open forecast

```bash
python -m stockpredictor.cli train AAPL --end 2025-06-01 --model lightgbm --horizon 3
```

This prints a forecast for the next actual trading session's open-to-final-close
return using the latest completed bar, plus the as-of date and training count.
It does not fabricate future bars, persist a production model, or label an
in-sample fit statistic as predictive accuracy. Check the as-of date for
staleness before interpreting a forecast.

## Optional dashboard

```bash
python -m pip install -r requirements-dashboard.txt
python -m streamlit run stockpredictor/dashboard.py
```

Enter a historical exclusive cutoff. The dashboard delegates to the same
walk-forward implementation; it has no separate training/evaluation logic.

## What remains before a trading decision

- Lock an untouched final evaluation period and an investable historical stock
  universe. Tune only inside earlier purged walk-forward folds, then evaluate
  once. The historical smoke check in docs/REVIEW.md is now explored data.
- Test ranking quality and top-k net results across stocks and market regimes,
  with realistic shared capital, correlation and position limits.
- Obtain point-in-time corporate actions, delisted securities and archived
  datasets; downloading adjusted history today does not prove what was known
  historically.
- Add timestamped news/catalysts only through backward as-of availability joins,
  using first-seen/received timestamps and revisions. News and macro data are
  **not implemented**; no current-news summaries are backfilled into history.
- Validate take-profit/stop-loss behavior on intraday quotes/trades with spreads,
  gaps, slippage, order timing and conservative same-bar handling.
- Run prospective paper screening and reconcile actual execution costs before
  considering real orders.

## API changes

The old train_lightgbm/train_lstm dataframe API and (predictions, price_rmse)
backtest tuple are intentionally replaced by make_model(X/y fit),
walk_forward → BacktestResult, and forecast_next_open. CLI/dashboard callers
are updated together. horizon now means the actual target holding period;
it no longer means the size of a batch of same-day close predictions.
