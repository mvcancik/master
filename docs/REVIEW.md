# Review of PRs #1 and #2 and validation of the replacement

Reviewed on October 22, 2026 (America/Los_Angeles). Conclusions are tied to the
immutable commits below. No original PR was merged, closed or edited.

## Comparison

| Item | PR #1 | PR #2 |
| --- | --- | --- |
| State at review | Open, unmerged | Open, unmerged |
| Created, UTC | 2025-06-03 20:52:47 | 2025-06-03 20:53:20 |
| Head | 337d00b73c42bd023bef61a1eb347ba90d2756c4 | 71c986fa9984e0087c75cd12a0b305a68390b2d2 |
| Base | ff8f78dc271ed7d7551cb072e125baa7238eadc3 | Same |
| Changed files / additions / deletions | 9 / 283 / 0 | 9 / 283 / 0 |
| Complete Git tree | 044b2df7cf8e3cdb4be268b71c8e93736849af5f | Same |

The PRs are byte-for-byte identical at the tree level; their commits differ.
A direct git diff between heads is empty. GitHub's three-dot compare uses the
merge base and lists the nine additions even though the two head trees match.
There is no second implementation to combine with the first.

Sources: [PR #1](https://github.com/mvcancik/master/pull/1),
[PR #2](https://github.com/mvcancik/master/pull/2).

## Confirmed findings

| Severity | Verified finding and original code | Consequence / correction |
| --- | --- | --- |
| Critical for forecasting | [models.py lines 10–17](https://github.com/mvcancik/master/blob/337d00b73c42bd023bef61a1eb347ba90d2756c4/stockpredictor/models.py#L10-L17) predicts the same row's Close. [features.py lines 7–9](https://github.com/mvcancik/master/blob/337d00b73c42bd023bef61a1eb347ba90d2756c4/stockpredictor/features.py#L7-L9) includes that Close in the averages/RSI. X also retains same-row High, Low and Volume. | Inputs require the completed day whose close is being predicted. Replace with forward open-to-close returns and lag all market features one session. |
| Critical for LSTM evaluation | [models.py lines 38–43](https://github.com/mvcancik/master/blob/337d00b73c42bd023bef61a1eb347ba90d2756c4/stockpredictor/models.py#L38-L43) trains shape (N,1,F). [backtest.py lines 25–31](https://github.com/mvcancik/master/blob/337d00b73c42bd023bef61a1eb347ba90d2756c4/stockpredictor/backtest.py#L25-L31) passes (1,horizon,F) and extends h actuals. | Whether a specific Keras version rejects the timestep shape or accepts it, Dense(1) produces one output for one input sequence, not h aligned predictions. Construct one genuine lookback sequence per target; scale only training data. |
| High methodology gap | [backtest.py lines 17–34](https://github.com/mvcancik/master/blob/337d00b73c42bd023bef61a1eb347ba90d2756c4/stockpredictor/backtest.py#L17-L34) uses horizon as test-batch length, not a future target. There is no execution timing, cost, strategy rule or capital accounting. | Price RMSE cannot establish screening/trading performance. Separate holding horizon from refit cadence; add net returns, baselines and nonoverlapping execution. |
| High if future labels are added without further changes | The original row split has no label-end availability check. | Merely adding shift(-h) would allow training labels to extend beyond the fit cutoff. This is a defect in a naive proposed fix, not an additional future-label leak in the original unshifted target. Replacement purges every unrealized label by date. |
| Moderate | [backtest.py line 17](https://github.com/mvcancik/master/blob/337d00b73c42bd023bef61a1eb347ba90d2756c4/stockpredictor/backtest.py#L17) uses an exclusive stop at len-window-horizon and lacks empty-result handling. | Can omit an exactly fitting final block and gives no clear insufficient-history failure. Replacement evaluates every complete eligible target and raises a clear error when none exists. |
| Moderate | [models.py lines 14–17 and 33–36](https://github.com/mvcancik/master/blob/337d00b73c42bd023bef61a1eb347ba90d2756c4/stockpredictor/models.py#L14-L17) drops warmup rows then withholds 20% inside every outer training window. | First 252-row window has only 53 complete rows after MA200 warmup, about 42 actually fitted. No final refit uses the held-out fifth. Replacement fits all eligible rows and scores only outer walk-forward predictions. |
| Version-dependent, reproduced here | [models.py lines 28 and 47](https://github.com/mvcancik/master/blob/337d00b73c42bd023bef61a1eb347ba90d2756c4/stockpredictor/models.py#L28) uses mean_squared_error(..., squared=False). | With installed scikit-learn 1.8.0: TypeError, unexpected keyword squared. Replacement calculates square root of mean squared error directly. |
| Provider-format-dependent, reproduced here | [data.py line 34](https://github.com/mvcancik/master/blob/337d00b73c42bd023bef61a1eb347ba90d2756c4/stockpredictor/data.py#L34) calls string replace on each column. | On yfinance MultiIndex tuples: AttributeError. Request flat output and validate/normalize either orientation for one ticker. |
| Missing implementation | [data.py lines 13–20](https://github.com/mvcancik/master/blob/337d00b73c42bd023bef61a1eb347ba90d2756c4/stockpredictor/data.py#L13-L20) returns empty macro and news dataframes. | No sentiment or catalyst integration exists. Original README actually called these placeholder hooks; it did not claim a completed feed. Replacement removes misleading combine-data behavior and explicitly labels all outputs price-only. |

Additional code facts: the original dashboard's “Train and Predict” action only
trains and charts historical Close; there is no live future forecast. TensorFlow
imports at module import time also make it mandatory for LightGBM and CLI help.
Both paths now use lazy optional dependencies and shared evaluation routines.

### Correction to the initial leakage diagnosis

Computing **trailing** moving averages and RSI over an entire fixed history is
not by itself look-ahead leakage. Those operations do not read subsequent rows.
The issue is their availability relative to the target and trade time.

Reproduction using the original feature code: increase Close at row 240 by
100; that row's MA50 changes from 315.5 to 317.5, while row 239 is unchanged.
This proves contemporaneous-target exposure and also shows the trailing
calculation does not contaminate earlier rows.

A later day's completed close can legitimately inform a still-later day's
sequential forecast, even when both days were initially in a test block. It
cannot inform a batch forecast claimed to have been issued before either close.
The new code documents sequential issuance explicitly.

No evidence was found of a global fitted scaler, random train/test shuffling,
or test-driven hyperparameter optimization in the original PRs. These should
not be reported as verified original defects. The LSTM has no scaler at all;
that is a conditioning concern, not global-scaler leakage.

## Reusable components and changes

- Keep the data/features/models/backtest/CLI/dashboard module organization.
- Reuse the yfinance acquisition concept, trailing SMA/RSI mathematics, and
  LightGBM/LSTM model families.
- Use price-relative rather than absolute-level features, plus short-term
  returns, EMA gaps, range, relative volume and realized volatility. Retain
  MA50/MA200 as slow context; they do not imply an edge at short horizons.
- Replace same-day target construction, nested holdout scoring, and backtest
  alignment. Fit only eligible rows, with a feature allowlist.
- Repair LSTM time context and train-only feature/target scaling; add Ridge as
  a simple comparator. No test set is used for fitting, scaling or early stopping.
- Replace the fake sentiment/macro integration with an explicit price-only scope.
- Preserve entry, feature-as-of, fit-as-of and label-end dates for audit.
- Separate prediction errors from a single-symbol trading simulation, with
  per-side costs, no capital reuse across overlapping signals, daily marks and
  matched-period baselines.

This intentionally changes the old internal APIs; all repository callers are
updated. The new branch builds on PR #1's content, and can supersede either
identical original PR after review.

## Validation

Local result: **64 tests passed**, including the optional real TensorFlow tests;
CLI help, Python compilation and diff whitespace checks passed. These tests
verify mechanics and temporal isolation, not profitability.

The automated suite tests:

- Exact forward targets and dates for 1, 2 and 3 sessions, including unavailable
  tail labels and the last eligible prediction.
- All market inputs lagged before entry; prefix computation equals full-history
  computation for the shared prefix.
- Future/current-entry price mutations leave earlier predictions unchanged
  across every horizon for real Ridge and LightGBM.
- Every recorded training X/y equals an independently selected eligible slice;
  every maximum training label-end is at/before the model-fit cutoff.
- Appending data does not rewrite prior predictions, including between refits.
- Real LSTM fit/inference, shape alignment, training-only scaling and end-to-end
  walk-forward/live forecast paths.
- Latest completed-history forecasts agree with the same historical next-open
  prediction, including real Ridge/LightGBM.
- Hand-calculated transaction costs, overnight gaps, no overlapping capital
  reuse, daily intratrade drawdowns and no-trade behavior.
- Bad data/configuration fail clearly; single-ticker column normalization,
  explicit data cutoffs, CLI CSV workflow and output files.

Observed local environment: Python 3.12.14; NumPy 2.3.5; pandas 2.2.3;
scikit-learn 1.8.0; LightGBM 4.7.0; TensorFlow CPU 2.21.0; pytest 9.1.1;
yfinance 1.7.0. Direct runtime versions are pinned; this is not a complete
transitive lockfile. The optional dashboard is not a trading execution interface.

Commands:

```bash
TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
python -m stockpredictor.cli --help
git diff --check
```

## Historical sanity check: AAPL

Downloaded 1,257 adjusted daily bars for 2020-06-01 through 2025-05-30 with an
exclusive end of 2025-06-01. The snapshot SHA-256 and full unrounded results are
in [aapl-smoke-summary.json](aapl-smoke-summary.json). The downloaded raw market
snapshot is not bundled; vendor revisions can change later downloads.

Settings were chosen before reading these results: maximum 252 eligible
training samples, minimum 126, refit every five prediction sessions, a zero
minimum predicted net-return threshold, 10 basis points cost **per side**,
one nonoverlapping long position, fixed default model settings. No hyperparameter
or threshold search was performed. All six runs are reported.

| Model | Holding sessions | Forecasts | Trades | Net total return | Max daily-close drawdown | Matched-period buy/hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge | 1 | 931 | 399 | −46.73% | −48.43% | +37.67% |
| Ridge | 2 | 929 | 283 | −34.57% | −38.08% | +37.78% |
| Ridge | 3 | 927 | 212 | −50.82% | −60.44% | +37.43% |
| LightGBM | 1 | 931 | 400 | −10.11% | −30.10% | +37.67% |
| LightGBM | 2 | 929 | 276 | −6.92% | −30.08% | +37.78% |
| LightGBM | 3 | 927 | 208 | −1.26% | −34.55% | +37.43% |

Out-of-sample entry dates begin September 15–17, 2021, depending on the horizon;
all evaluation periods end May 30, 2025. Total returns are not annualized.
Every model/horizon has worse return RMSE than the zero-return forecast and
the past-mean forecast. Direction accuracy ranges from 48.4% to 51.2%.

These are conditional single-symbol simulation results, not a causal claim
about all possible feature sets or strategies. They do not validate market-wide
ranking, news-driven selection, execution quality or live profitability.
The smoke-check period is now explored data, not an untouched final holdout.
Do not select the least-bad horizon and treat that selection as independent
evidence. LSTM received functionality tests, not a historical profitability study.

## Remaining risks and prerequisites for trusting strategy results

1. **Final evaluation discipline.** Predeclare the universe, trading rules,
   horizons, threshold and costs. Any later feature/model selection must use
   purged inner time splits; reserve a later untouched holdout and then
   prospective paper trading. Overlapping forward returns are dependent:
   use an appropriate block bootstrap for uncertainty, not an IID confidence
   interval or naive annualized Sharpe from h-day samples.
2. **Data integrity.** Today's adjusted yfinance history is not guaranteed
   point-in-time data. Audit split/dividend adjustments, revisions, missing
   exchange sessions, stale/delisted symbols and historical universe membership.
   Survivorship bias is a potential future universe-selection problem here,
   not a demonstrated bias in a universe implementation that does not exist.
3. **Strategy fit.** This patch supports night-before, next-open price screening.
   It does not include catalysts, opening-gap reactions or intraday monitoring.
   A 1-session target exits the entry day; define separate labels if the intended
   1-day hold must include an overnight.
4. **Catalyst availability.** Before adding news, store publication, first-seen,
   vendor-received and revision timestamps. Join backward by actual availability
   at the signal cutoff, with ingestion latency. Use original macro releases,
   not revised histories. A date-only left join or current headline summary
   cannot establish historical availability.
5. **Portfolio and execution.** Add top-k ranking evaluation, shared capital and
   correlated exposure limits across tickers. Model whole-share sizing, cash
   limits, opening/closing auction access, spreads, gaps, partial fills and
   realistic per-symbol slippage. The current simulator uses fractional units
   and all modeled capital for one position.
6. **Profit targets/stops.** The user's short-term strategy includes tight
   exits; fixed close exits are only a baseline. Daily OHLC cannot determine
   whether a target or stop was hit first within a session. Use timestamped
   intraday data and conservative fill assumptions before trusting those tests.
7. **Deployment.** Add stale-data checks, corporate-action handling, restart-safe
   position state, audit logging and prospective monitoring before connecting
   this research pipeline to an order system.

The corrected implementation is a testable research foundation. The available
evidence does **not** justify trusting either original PR's backtest or deploying
the corrected baseline as a reliable profitable screener.

## Methodology references

- [scikit-learn: data leakage and train-only preprocessing](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
- [scikit-learn: time-series splits and gaps](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [yfinance download API: adjustment and multi-level-column options](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [scikit-learn mean_squared_error API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)

A fixed row gap can be helpful, but the patch uses actual label-end eligibility
because that directly expresses when the target becomes observable.
