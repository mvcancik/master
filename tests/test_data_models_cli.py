import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stockpredictor.backtest import forecast_next_open, walk_forward
from stockpredictor.cli import main
from stockpredictor.data import download_price_data, normalize_download, validate_prices
from stockpredictor.models import LSTMRegressor, make_model


@pytest.mark.parametrize("reverse", [False, True])
def test_single_symbol_multiindex(prices, reverse):
    raw = prices.copy()
    raw.columns = pd.MultiIndex.from_tuples(
        [("AAPL", col) if reverse else (col, "AAPL") for col in raw.columns])
    pd.testing.assert_frame_equal(normalize_download(raw, "AAPL"), validate_prices(prices))


def test_reject_multiple_tickers(prices):
    raw = pd.concat({"AAPL": prices, "MSFT": prices}, axis=1)
    with pytest.raises(ValueError, match="exactly"):
        normalize_download(raw, "AAPL")


@pytest.mark.parametrize("problem", ["nan", "duplicate", "unsorted", "zero", "bad_high"])
def test_bad_bars_fail_before_splitting(prices, problem):
    changed = prices.copy()
    if problem == "nan":
        changed.iloc[230, 0] = np.nan
    elif problem == "duplicate":
        changed.index = changed.index.where(np.arange(len(changed)) != 230, changed.index[229])
    elif problem == "unsorted":
        changed = changed.iloc[::-1]
    elif problem == "zero":
        changed.iloc[230, 0] = 0
    else:
        changed.loc[changed.index[230], "High"] = 1
    with pytest.raises(ValueError):
        validate_prices(changed)


def test_download_requests_explicit_completed_cutoff(prices, monkeypatch):
    observed = {}
    def download(ticker, **kwargs):
        observed.update(kwargs)
        return prices.iloc[:10]
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=download))
    result = download_price_data("AAPL", end="2020-02-01")
    assert observed["end"] == "2020-02-01"
    assert observed["auto_adjust"] is True
    assert observed["multi_level_index"] is False
    assert len(result) == 10


def test_provider_cutoff_violation_rejected(prices, monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance",
                        SimpleNamespace(download=lambda *a, **kw: prices))
    with pytest.raises(ValueError, match="cutoff"):
        download_price_data("AAPL", end="2020-01-05")


def test_ridge_scaler_never_sees_test_values():
    X = np.arange(80.0).reshape(40, 2)
    model = make_model("ridge").fit(X, X[:, 0] * 0.01)
    before = model.named_steps["standardscaler"].mean_.copy()
    model.predict(np.full((1, 2), 1e8))
    np.testing.assert_array_equal(before, X.mean(axis=0))
    np.testing.assert_array_equal(before, model.named_steps["standardscaler"].mean_)


def test_csv_cli_end_to_end(prices, tmp_path, capsys):
    source = tmp_path / "bars.csv"
    prices.to_csv(source)
    output = tmp_path / "output"
    main(["backtest", "SYNTHETIC", "--csv", str(source), "--end", "2022-01-01",
          "--model", "ridge", "--horizon", "3", "--min-train", "30",
          "--output", str(output)])
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["n_predictions"] > 0
    for filename in ("predictions.csv", "trades.csv", "equity.csv", "metrics.json"):
        assert (output / filename).exists()
    main(["train", "SYNTHETIC", "--csv", str(source), "--end", "2022-01-01",
          "--model", "ridge", "--horizon", "3", "--min-train", "30"])
    forecast = json.loads(capsys.readouterr().out)
    assert forecast["asof_close"] == prices.index[-1].date().isoformat()
    assert "RMSE" not in forecast


@pytest.mark.tensorflow
def test_real_lstm_scaling_and_shape():
    pytest.importorskip("tensorflow")
    rng = np.random.default_rng(12)
    X, y = rng.normal(size=(12, 5, 3)), rng.normal(size=12) * .01
    model = LSTMRegressor(epochs=1).fit(X, y)
    expected_mean = X.reshape(-1, 3).mean(axis=0)
    np.testing.assert_allclose(model.scaler_.mean_, expected_mean)
    result = model.predict(X[:2] + 10)
    assert result.shape == (2,)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(model.scaler_.mean_, expected_mean)
    with pytest.raises(ValueError, match="shape"):
        model.predict(X[:1, :1])


@pytest.mark.tensorflow
def test_real_lstm_walk_forward_and_live(prices):
    pytest.importorskip("tensorflow")
    kwargs = dict(model_type="lstm", horizon=3, min_train=30, window=40,
                  lookback=5, epochs=1)
    result = walk_forward(prices.iloc[:245], refit_every=100, **kwargs)
    assert len(result.predictions) == 7
    assert (result.predictions.train_max_label_end <= result.predictions.fit_asof).all()
    forecast = forecast_next_open(prices.iloc[:245], **kwargs)
    assert np.isfinite(forecast["predicted_gross_return"])
