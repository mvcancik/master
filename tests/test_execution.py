import numpy as np
import pandas as pd
import pytest

from stockpredictor.backtest import simulate_trades, walk_forward


def fixture(horizon=3, prediction=1.0):
    dates = pd.bdate_range("2024-01-01", periods=7)
    prices = pd.DataFrame({"Open": 100.0, "Close": 110.0}, index=dates)
    entries = dates[:len(dates) - horizon + 1]
    signals = pd.DataFrame({
        "prediction": prediction,
        "exit_date": dates[horizon - 1:],
    }, index=entries)
    return prices, signals


def test_costs_and_nonoverlapping_positions():
    prices, signals = fixture()
    ledger, equity, metrics = simulate_trades(prices, signals, cost_bps=10)
    expected = 110 * .999 / (100 * 1.001) - 1
    assert list(ledger.entry_date) == list(prices.index[[0, 3]])
    assert list(ledger.exit_date) == list(prices.index[[2, 5]])
    np.testing.assert_allclose(ledger.net_return, expected)
    assert metrics["strategy_return"] == pytest.approx((1 + expected)**2 - 1)
    assert equity.equity.iloc[-1] == pytest.approx(np.prod(1 + ledger.net_return))


def test_close_to_open_gap_is_not_free_profit():
    prices, signals = fixture(horizon=1)
    prices["Open"] = 200
    prices["Close"] = 202
    ledger, _, _ = simulate_trades(prices, signals, cost_bps=0)
    np.testing.assert_allclose(ledger.net_return, 0.01)


def test_drawdown_includes_unrealized_loss():
    prices, signals = fixture(horizon=3)
    signals = signals.iloc[:1]
    prices.loc[prices.index[0], "Close"] = 100
    prices.loc[prices.index[1], "Close"] = 50
    _, equity, metrics = simulate_trades(prices, signals, cost_bps=0)
    assert equity.drawdown.iloc[1] == -0.5
    assert metrics["max_drawdown"] == -0.5
    assert metrics["strategy_return"] == pytest.approx(.1)


def test_cost_threshold_and_cash_behavior():
    prices, signals = fixture(prediction=.001)
    ledger, equity, metrics = simulate_trades(prices, signals, cost_bps=10)
    assert ledger.empty
    assert (equity.equity == 1).all()
    assert metrics["strategy_return"] == metrics["max_drawdown"] == 0
    assert metrics["net_win_rate"] is None


def test_costs_reduce_net_returns_with_fixed_trades():
    prices, signals = fixture()
    _, _, free = simulate_trades(prices, signals, cost_bps=0)
    _, _, paid = simulate_trades(prices, signals, cost_bps=50)
    assert paid["strategy_return"] < free["strategy_return"]
    assert paid["buy_hold_return"] < free["buy_hold_return"]
